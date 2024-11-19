from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.utils.data as data

from ..config import Config


@dataclass
class Segment:
    audio: str
    label: str
    onset_frame: int
    offset_frame: int


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
        self.mode = config.training.mode

        for audio, label, duration in self.files:
            num_frames = int(
                duration
                * config.model.input.mel_spectrogram.sample_rate
                / config.model.input.mel_spectrogram.hop_length
            )
            for i in range(0, num_frames, config.dataset.segment_frames):
                self.segments.append(
                    Segment(
                        audio=audio,
                        label=label,
                        onset_frame=i,
                        offset_frame=min(i + config.dataset.segment_frames, num_frames),
                    )
                )

    def __len__(self):
        return len(self.segments)

    def load_audio(self, audio_path: str, onset_frame: int, offset_frame: int):
        audio: torch.Tensor = torch.load(audio_path, weights_only=True)

        if self.config.model.type == "hft_transformer":
            audio_start_frame = onset_frame - self.config.model.margin_b
            audio_end_frame = offset_frame + self.config.model.margin_f
            audio_start_sample = (
                audio_start_frame * self.config.model.input.mel_spectrogram.hop_length
            )
            audio_end_sample = (
                audio_end_frame * self.config.model.input.mel_spectrogram.hop_length
            )
            audio = audio[audio_start_sample:audio_end_sample]

            segment_samples = (
                self.config.dataset.segment_frames
                + self.config.model.margin_b
                + self.config.model.margin_f
            ) * self.config.model.input.mel_spectrogram.hop_length
            pad = torch.zeros(
                segment_samples - len(audio), dtype=audio.dtype, device=audio.device
            )
            audio = torch.cat([audio, pad])
        else:
            onset_sample = (
                onset_frame * self.config.model.input.mel_spectrogram.hop_length
            )
            offset_sample = (
                offset_frame * self.config.model.input.mel_spectrogram.hop_length
            )
            segment_samples = (
                self.config.dataset.segment_frames
                * self.config.model.input.mel_spectrogram.hop_length
            )
            audio = audio[onset_sample:offset_sample]
            pad = torch.zeros(
                segment_samples - len(audio),
                dtype=audio.dtype,
                device=audio.device,
            )
            audio = torch.cat([audio, pad])

        return audio

    def get_item_note(self, idx: int):
        segment = self.segments[idx]

        audio = self.load_audio(
            segment.audio, segment.onset_frame, segment.offset_frame
        )
        label = torch.load(segment.label, weights_only=True)
        label = MidiLabel(**label)

        onset = label.onset[segment.onset_frame : segment.offset_frame]
        offset = label.offset[segment.onset_frame : segment.offset_frame]
        frame = label.frame[segment.onset_frame : segment.offset_frame]
        velocity = label.velocity[segment.onset_frame : segment.offset_frame]

        padding_frame = self.config.dataset.segment_frames - onset.shape[0]
        pad_label = torch.zeros(
            (padding_frame, onset.shape[1]),
            dtype=onset.dtype,
            device=onset.device,
        )
        pad_velocity = torch.zeros(
            (padding_frame, velocity.shape[1]),
            dtype=velocity.dtype,
            device=velocity.device,
        )

        onset = torch.cat([onset, pad_label])
        offset = torch.cat([offset, pad_label])
        frame = torch.cat([frame, pad_label])
        velocity = torch.cat([velocity, pad_velocity])

        return audio, onset, offset, frame, velocity

    def get_item_pedal(self, idx: int):
        segment = self.segments[idx]

        audio = self.load_audio(
            segment.audio, segment.onset_frame, segment.offset_frame
        )
        label = torch.load(segment.label, weights_only=True)
        label = MidiLabel(**label)

        onset = label.pedal_onset[segment.onset_frame : segment.offset_frame]
        offset = label.pedal_offset[segment.onset_frame : segment.offset_frame]
        frame = label.pedal_frame[segment.onset_frame : segment.offset_frame]

        padding_frame = self.config.dataset.segment_frames - onset.shape[0]
        pad_label = torch.zeros(
            padding_frame,
            dtype=onset.dtype,
            device=onset.device,
        )

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
