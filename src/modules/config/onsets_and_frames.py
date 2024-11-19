from typing import Literal

from pydantic import BaseModel

from .mel_spectrogram import MelSpectrogramConfig
from .midi import MidiConfig


class InputConfig(BaseModel):
    mel_spectrogram: MelSpectrogramConfig = MelSpectrogramConfig()


class OutputConfig(BaseModel):
    midi: MidiConfig = MidiConfig()


class OnsetsAndFramesConfig(BaseModel):
    type: Literal["onsets_and_frames"] = "onsets_and_frames"
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    complexity: int = 48
