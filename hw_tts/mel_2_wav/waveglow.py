import numpy as np
import torch


class WaveGlowInfer(object):
    MAX_WAV_VALUE = 32768.0

    def __init__(self, waveglow_path, device):
        super(WaveGlowInfer, self).__init__()
        # waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')
        # waveglow = waveglow.remove_weightnorm(waveglow)
        # self.waveglow = waveglow.to(device).eval()

        waveglow = torch.load(waveglow_path, map_location=device)['model']
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow.cuda().eval()
        for m in waveglow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

        self.waveglow = waveglow

    def __call__(self, mel, sigma=1.0):
        mel = mel.unsqueeze(0).transpose(-1, -2)
        with torch.no_grad():
            audio = self.waveglow.infer(mel, sigma=sigma)
            # audio = audio * self.MAX_WAV_VALUE
        audio = audio.squeeze()

        rate = 22050
        return audio, rate
