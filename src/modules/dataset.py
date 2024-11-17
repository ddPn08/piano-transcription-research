import json
import os
from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
import torch.utils.data as data
from pydantic import BaseModel

from .config import Config


class Metadata(BaseModel):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float


class Segment(BaseModel):
    audio: str
    label: str
    onset_sample: int
    offset_sample: int


@dataclass
class MidiLabel:
    onset: torch.Tensor
    offset: torch.Tensor
    frame: torch.Tensor
    velocity: torch.Tensor
    pedal_onset: torch.Tensor
    pedal_offset: torch.Tensor
    pedal_frame: torch.Tensor


class TranscriptionDataset(data.Dataset):
    def __init__(
        self,
        config: Config,
        files: List[Tuple[str, str, int]],
    ):
        self.config = config
        self.files = files
        self.segments: List[Segment] = []
        self.segment_samples = (
            config.dataset.segment_seconds * config.mel_spectrogram.sample_rate
        )
        self.mode = config.training.mode

        for audio, label, duration in self.files:
            num_samples = int(duration * config.mel_spectrogram.sample_rate)
            for i in range(0, num_samples, self.segment_samples):
                self.segments.append(
                    Segment(
                        audio=audio,
                        label=label,
                        onset_sample=i,
                        offset_sample=min(i + self.segment_samples, num_samples),
                    )
                )

    def __len__(self):
        return len(self.segments)

    def get_item_note(self, idx: int):
        segment = self.segments[idx]

        audio: torch.Tensor = torch.load(segment.audio, weights_only=True)  # (T,)
        label = torch.load(segment.label, weights_only=True)
        label = MidiLabel(**label)

        start_frame = segment.onset_sample // self.config.mel_spectrogram.hop_length
        end_frame = segment.offset_sample // self.config.mel_spectrogram.hop_length

        audio = audio[segment.onset_sample : segment.offset_sample]

        onset = label.onset[start_frame:end_frame]
        offset = label.offset[start_frame:end_frame]
        frame = label.frame[start_frame:end_frame]
        velocity = label.velocity[start_frame:end_frame]

        segment_frame = self.segment_samples // self.config.mel_spectrogram.hop_length

        padding = self.segment_samples - len(audio)
        padding_frame = segment_frame - onset.shape[0]
        pad_audio = torch.zeros(padding, dtype=audio.dtype, device=audio.device)
        pad_label = torch.zeros(
            (padding_frame, onset.shape[1]),
            dtype=onset.dtype,
            device=onset.device,
        )

        audio = torch.cat([audio, pad_audio])
        onset = torch.cat([onset, pad_label])
        offset = torch.cat([offset, pad_label])
        frame = torch.cat([frame, pad_label])
        velocity = torch.cat([velocity, pad_label])

        return audio, onset, offset, frame, velocity

    def get_item_pedal(self, idx: int):
        segment = self.segments[idx]

        audio = torch.load(segment.audio, weights_only=True)
        label = torch.load(segment.label, weights_only=True)
        label = MidiLabel(**label)

        start_frame = segment.onset_sample // self.config.mel_spectrogram.hop_length
        end_frame = segment.offset_sample // self.config.mel_spectrogram.hop_length

        audio = audio[segment.onset_sample : segment.offset_sample]
        onset = label.pedal_onset[start_frame:end_frame]
        offset = label.pedal_offset[start_frame:end_frame]
        frame = label.pedal_frame[start_frame:end_frame]

        segment_frame = self.segment_samples // self.config.mel_spectrogram.hop_length

        padding = self.segment_samples - len(audio)
        padding_frame = segment_frame - onset.shape[0]

        pad_audio = torch.zeros(padding, dtype=audio.dtype, device=audio.device)
        pad_label = torch.zeros(padding_frame, dtype=onset.dtype, device=onset.device)

        audio = torch.cat([audio, pad_audio])
        onset = torch.cat([onset, pad_label])
        offset = torch.cat([offset, pad_label])
        frame = torch.cat([frame, pad_label])

        return audio, onset, offset, frame

    def __getitem__(self, idx: int):
        if self.mode == "note":
            return self.get_item_note(idx)
        else:
            return self.get_item_pedal(idx)

    def note_collate_fn(self, batch: torch.Tensor):
        audio, onset, offset, frame, velocity = zip(*batch)
        audio = torch.stack(audio)
        onset = torch.stack(onset)
        offset = torch.stack(offset)
        frame = torch.stack(frame)
        velocity = torch.stack(velocity)

        return audio, onset, offset, frame, velocity

    def pedal_collate_fn(self, batch: torch.Tensor):
        audio, onset, offset, frame = zip(*batch)
        audio = torch.stack(audio)
        onset = torch.stack(onset)
        offset = torch.stack(offset)
        frame = torch.stack(frame)

        return audio, onset, offset, frame

    def collate_fn(self, batch: torch.Tensor):
        if self.mode == "note":
            return self.note_collate_fn(batch)
        else:
            return self.pedal_collate_fn(batch)


class MaestroDataset(TranscriptionDataset):
    def __init__(
        self,
        config: Config,
        split: Literal["train", "validation", "test"] = "train",
    ):
        with open(os.path.join(config.dataset.dataset_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            metadata = [Metadata(**m) for m in metadata]

        metadata = [m for m in metadata if m.split == split]

        files = []
        for m in metadata:
            audio = os.path.join(
                config.dataset.dataset_dir,
                "wav",
                m.split,
                m.audio_filename.replace("/", "-").replace("wav", "pt"),
            )
            label = os.path.join(
                config.dataset.dataset_dir,
                "label",
                m.split,
                m.midi_filename.replace("/", "-").replace("midi", "pt"),
            )
            files.append((audio, label, m.duration))

        super().__init__(config, files)
