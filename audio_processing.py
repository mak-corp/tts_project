import argparse
import librosa
import numpy as np
import os
from os.path import join
import pathlib
import pyworld
import time
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from hw_tts.contrib.audio.hparams_audio import hop_length
from hw_tts.contrib.audio.tools import get_mel


SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()


def get_pitch(wav, sr):
    pitch, t = pyworld.dio(
        wav.astype(np.float64),
        sr,
        frame_period=hop_length / sr * 1000,
    )
    pitch = pyworld.stonemask(wav.astype(np.float64), pitch, t, sr)
    return pitch


def get_mel_and_energy(wav_path):
    mel, energy = get_mel(wav_path)
    mel = mel.numpy().astype(np.float32)
    energy = energy.squeeze().numpy().astype(np.float32)
    return mel, energy


def process_utterance(wav_path):
    wav, sr = librosa.load(wav_path)
    pitch = get_pitch(wav, sr)
    mel, energy = get_mel_and_energy(wav_path)

    return mel.T, pitch, energy


def normalize(scaler, value_dir):
    for filename in os.listdir(value_dir):
        filename = os.path.join(value_dir, filename)
        values = (np.load(filename) - scaler.mean_[0]) / scaler.scale_[0]
        np.save(filename, values, allow_pickle=False)


def process_ljspeech(ljspeech_dir, output_dir, limit=None):
    print()
    print("============================== Processing started ==============================")
    print()

    os.makedirs(output_dir, exist_ok=False)

    mel_dir = join(output_dir, "mels")
    pitch_dir = join(output_dir, "pitch")
    energy_dir = join(output_dir, "energy")

    os.makedirs(mel_dir, exist_ok=False)
    os.makedirs(pitch_dir, exist_ok=False)
    os.makedirs(energy_dir, exist_ok=False)

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    texts = []
    with open(join(ljspeech_dir, "metadata.csv"), "r", encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            if limit is not None and idx >= limit:
                break
            wav_name, _, text = line.strip().split('|')
            texts.append(text)
            mel, pitch, energy = process_utterance(join(ljspeech_dir, "wavs", wav_name + ".wav"))

            pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
            energy_scaler.partial_fit(energy.reshape((-1, 1)))

            np.save(join(mel_dir, "ljspeech-mel-%05d.npy" % idx), mel, allow_pickle=False)
            np.save(join(pitch_dir, "ljspeech-pitch-%05d.npy" % idx), pitch, allow_pickle=False)
            np.save(join(energy_dir, "ljspeech-energy-%05d.npy" % idx), energy, allow_pickle=False)

    with open(join(output_dir, "train.txt"), "w", encoding='utf-8') as f:
        f.write('\n'.join(texts))
        f.write('\n')

    normalize(pitch_scaler, pitch_dir)
    normalize(energy_scaler, energy_dir)

    print()
    print("============================== Processing finished ==============================")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ljspeech-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False)
    args = parser.parse_args()

    process_ljspeech(args.ljspeech_dir, args.output_dir, args.limit)
