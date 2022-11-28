import numpy as np
import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_output, duration_predictor_output, mel_target, duration, **kwargs):
        mel_loss = self.mse_loss(mel_output, mel_target)

        duration_loss = self.l1_loss(duration_predictor_output,
                                               duration.float())

        loss = mel_loss + duration_loss
        return {
            "loss": loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
        }
