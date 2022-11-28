import numpy as np
import torch
from torch import nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_output, log_duration_prediction, energy_prediction, pitch_prediction,
                mel_target, duration, energy, pitch, **kwargs):

        mel_loss = self.l1_loss(mel_output, mel_target)
        log_duration = torch.clamp(torch.log(duration).float(), min=1e-8)
        # print("log_duration_prediction:", log_duration_prediction, "log_duration:", log_duration)
        duration_loss = self.mse_loss(log_duration_prediction, log_duration)
        pitch_loss = self.mse_loss(pitch_prediction, pitch)
        energy_loss = self.mse_loss(energy_prediction, energy)

        # print("mel_loss:", mel_loss, "duration_loss:", duration_loss, "pitch_loss:", pitch_loss, "energy_loss:", energy_loss)

        loss = 2 * mel_loss + duration_loss + pitch_loss + energy_loss
        return {
            "loss": loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
        }
