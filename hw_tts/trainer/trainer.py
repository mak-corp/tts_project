import random
from pathlib import Path
from random import shuffle
from itertools import chain
from collections import defaultdict

import numpy as np
import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.mel_2_wav import WaveGlowInfer
from hw_tts.datasets.test_data import get_test_data, get_v1_test_data


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.use_waveglow = device != torch.device("cpu")
        self.waveglow = WaveGlowInfer(config["waveglow_path"], device) if self.use_waveglow else None
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"]["log_step"]

        self.train_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "mel_loss", "duration_loss", "pitch_loss", "energy_loss", *[m.name for m in self.metrics], writer=self.writer
        )

        self.is_v1_model = config["arch"]["type"] == "FastSpeechBaselineModel"

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch), start=1
        ):
            if 'error' in batch:
                continue

            if batch_idx > self.len_epoch:
                break
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 or batch_idx == 1:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx - 1)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(batch, is_train=True)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            
        log = last_train_metrics

        self._test_epoch(epoch, "test")

        # for part, dataloader in self.evaluation_dataloaders.items():
        #     val_log = self._evaluation_epoch(epoch, part, dataloader)
        #     log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["text", "src_pos", "mel_target", "mel_pos", "duration", "pitch", "energy"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)

        if is_train:
            self.optimizer.zero_grad()

        out = self.model(**batch)
        if isinstance(out, dict):
            batch.update(out)
        else:
            mel_output, duration_predictor_output = out
            batch["mel_output"] = mel_output
            batch["duration_predictor_output"] = duration_predictor_output

        losses = self.criterion(**batch)
        batch.update(losses)

        if is_train:
            losses["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name, loss_value in losses.items():
            metrics.update(loss_name, loss_value.item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader, start=1),
                    desc=part,
                    total=len(dataloader),
            ):
                # batch = self.process_batch(
                #     batch,
                #     is_train=False,
                #     metrics=self.evaluation_metrics,
                # )
                batch = self.move_batch_to_device(batch, self.device)
                mel_output = self.model(**batch)
                batch["mel_output"] = mel_output
                break
            
            print("Do eval logging...")
            self.writer.set_step(epoch * self.len_epoch, part)
            # self._log_scalars(self.evaluation_metrics)
            self._log_predictions(batch, is_train=False)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _test_epoch(self, epoch, part):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.is_v1_model:
            test_data = get_v1_test_data()
        else:
            test_data = get_test_data()

        self.model.eval()
        self.evaluation_metrics.reset()

        self.writer.set_step(epoch * self.len_epoch, part)
        
        with torch.no_grad():
            test_results = defaultdict(list)
            for text_id, batches in tqdm(test_data.items()):
                for batch in tqdm(batches):
                    batch = Trainer.move_batch_to_device(batch, self.device)
                    mel_output = self.model(**batch)
                    batch["mel_output"] = mel_output
                    test_results[text_id].append(batch)
        self._log_test_predictions(test_results)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_test_predictions(self, test_results):
        for text_id, batches in test_results.items():
            self.writer.add_text("text={}/text".format(text_id), batches[0]["raw_text"])
            for batch in batches:
                prefix = "text={}/{}".format(text_id, batch["name"])
                self._log_spectrogram(f"{prefix}/mel_output", batch["mel_output"], idx=0)
                if self.use_waveglow:
                    audio, sr = self.waveglow(batch["mel_output"][0])
                    self.writer.add_audio(f"{prefix}/audio", audio, sr)

    def _log_predictions(self, batch, is_train):
        if self.writer is None:
            return

        idx = np.random.choice(len(batch["text"]))
        self.writer.add_text("train/text", batch["raw_text"][idx])

        if self.use_waveglow:
            audio, sr = self.waveglow(batch["mel_output"][idx])
            self.writer.add_audio("train/audio", audio, sr)
            
            audio, sr = self.waveglow(batch["mel_target"][idx])
            self.writer.add_audio("train/audio_target", audio, sr)

        self._log_spectrogram("train/mel_output", batch["mel_output"], idx)
        self._log_spectrogram("train/mel_target", batch["mel_target"], idx)

    def _log_spectrogram(self, name, spectrogram_batch, idx=None):
        idx = idx if idx is not None else np.random.choice(len(spectrogram_batch))
        spectrogram = spectrogram_batch[idx].detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image(name, ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
