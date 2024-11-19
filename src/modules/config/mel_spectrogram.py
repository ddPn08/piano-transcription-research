import torch
import torchaudio
from pydantic import BaseModel


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
