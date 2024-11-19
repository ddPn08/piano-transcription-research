from typing import Literal

from pydantic import BaseModel

from .mel_spectrogram import MelSpectrogramConfig
from .midi import MidiConfig


class InputConfig(BaseModel):
    mel_spectrogram: MelSpectrogramConfig = MelSpectrogramConfig()


class OutputConfig(BaseModel):
    midi: MidiConfig = MidiConfig()


class HftTransformerConfig(BaseModel):
    type: Literal["hft_transformer"] = "hft_transformer"
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    num_frame: int = 128
    cnn_channel: int = 4
    cnn_kernel: int = 5
    hid_dim: int = 256
    margin_b: int = 32
    margin_f: int = 32
    num_layers: int = 3
    num_heads: int = 4
    pf_dim: int = 512
    dropout: float = 0.1
    num_velocity: int = 127
