import numpy as np
import torch


class WaveGlow(object):
    def __init__(self, device):
        super(WaveGlow, self).__init__()
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32', map_location=device)
        waveglow = waveglow.remove_weightnorm(waveglow)
        self.waveglow = waveglow.to(device).eval()

    def __call__(self, mel):
        with torch.no_grad():
            audio = self.waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050
        return audio_numpy, rate
