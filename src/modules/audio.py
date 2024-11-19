import torch
import torchaudio

from modules.config.mel_spectrogram import MelSpectrogramConfig


def load_audio(path: str, sample_rate: int = 16000):
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(0)
    return wav


def create_mel_transform(config: MelSpectrogramConfig):
    window_fn = torch.hann_window
    if config.window_fn == "hamming":
        window_fn = torch.hamming_window
    elif config.window_fn == "blackman":
        window_fn = torch.blackman_window

    kwargs = {
        **config.model_dump(exclude={"window_fn"}),
        "window_fn": window_fn,
    }

    for k, v in list(kwargs.items()):
        if v is None:
            kwargs.pop(k)

    return torchaudio.transforms.MelSpectrogram(**kwargs)
