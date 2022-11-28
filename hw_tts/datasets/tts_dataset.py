from hw_tts.contrib.text import text_to_sequence
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import os


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line.strip())

        return txt


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners, pitch_path=None, energy_path=None, limit=None):
    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    size = len(text) if limit is None else min(len(text), limit)
    for i in tqdm(range(size)):
        mel_gt_target = np.load(os.path.join(mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1)))
        duration = np.load(os.path.join(alignment_path, str(i)+".npy"))
        character = torch.tensor(text_to_sequence(text[i], text_cleaners)).long()

        batch = {
            "raw_text": text[i],
            "text": character,
            "duration": torch.from_numpy(duration).int(),
            "mel_target": torch.from_numpy(mel_gt_target).float()
        }

        if pitch_path is not None and energy_path is not None:
            pitch = np.load(os.path.join(pitch_path, "ljspeech-pitch-%05d.npy" % (i+1)))
            energy = np.load(os.path.join(energy_path, "ljspeech-energy-%05d.npy" % (i+1)))
            batch.update({
                "pitch": torch.from_numpy(pitch).float(),
                "energy": torch.from_numpy(energy).float(),
            })
        
        buffer.append(batch)

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


class TTSDataset(BufferDataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path, text_cleaners, pitch=None, energy=None, limit=None):
        buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners, pitch, energy, limit)
        super(TTSDataset, self).__init__(buffer)
