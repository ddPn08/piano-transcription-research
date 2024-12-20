from typing import Literal, Union

import torch
import torchaudio
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from pydantic import BaseModel

from .hft_transformer import HftTransformerConfig
from .logger import (
    TensorboardLoggerConfig,
    WandbLoggerConfig,
)
from .onsets_and_frames import OnsetsAndFramesConfig


class MelSpectrogramConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    f_min: float = 30
    f_max: float | None = 8000
    pad: int | None = None
    n_mels: int = 229
    window_fn: str | None = None
    power: float = 1.0
    normalized: bool | None = None
    wkwargs: dict | None = None
    center: bool | None = None
    pad_mode: str | None = None
    onesided: bool | None = None
    norm: str = "slaney"
    mel_scale: str | None = "htk"

    def mel_transform(self):
        window_fn = torch.hann_window
        if self.window_fn == "hamming":
            window_fn = torch.hamming_window
        elif self.window_fn == "blackman":
            window_fn = torch.blackman_window

        kwargs = {
            **self.model_dump(exclude={"window_fn"}),
            "window_fn": window_fn,
        }

        for k, v in list(kwargs.items()):
            if v is None:
                kwargs.pop(k)

        return torchaudio.transforms.MelSpectrogram(**kwargs)


class DatasetConfig(BaseModel):
    dataset_dir: str = "maetsro-v3.0.0"
    segment_frames: int = 128


class TrainingConfig(BaseModel):
    mode: Literal["note", "pedal"] = "note"
    accelerator: str = "gpu"
    devices: str = "0"
    precision: _PRECISION_INPUT = "32"
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    max_epochs: int = 100
    optimizer: Literal["adam"] = "adam"
    output_dir: str = "output"

    save_every_n_steps: int | None = None
    save_every_n_epochs: int | None = None

    logger: Union[TensorboardLoggerConfig, WandbLoggerConfig] = (
        TensorboardLoggerConfig()
    )

    resume_from_checkpoint: str | None = None


class Config(BaseModel):
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    model: Union[HftTransformerConfig, OnsetsAndFramesConfig] = HftTransformerConfig()
