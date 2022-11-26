from hw_tts.mel_2_wav.waveglow import WaveGlowInfer

from .glow import * # Ensures that all the modules have been loaded in their new locations *first*.
from . import glow  # imports WrapperPackage/packageA
import sys
sys.modules['glow'] = glow  # creates a packageA entry in sys.modules


__all__ = [
    "WaveGlowInfer",
]
