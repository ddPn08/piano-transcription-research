import torchaudio


def load_audio(path: str, sample_rate: int = 16000):
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(0)
    return wav
