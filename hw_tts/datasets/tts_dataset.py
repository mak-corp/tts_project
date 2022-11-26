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


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners, limit=None):
    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    size = len(text) if limit is None else min(len(text), limit)
    for i in tqdm(range(size)):
        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            alignment_path, str(i)+".npy"))

        character = torch.tensor(text_to_sequence(text[i], text_cleaners))
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({"raw_text": text[i], "text": character, "duration": duration,
                       "mel_target": mel_gt_target})

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
    def __init__(self, data_path, mel_ground_truth, alignment_path, text_cleaners, limit=None):
        buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners, limit)
        super(TTSDataset, self).__init__(buffer)
