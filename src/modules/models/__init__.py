"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

from typing import Literal

import torch
from torch import nn


class PianoTranscriptionBaseModel(nn.Module):
    mode: Literal["note", "pedal"]

    @classmethod
    def preprocess_label(
        cls,
        onset: torch.Tensor,
        offset: torch.Tensor,
        frame: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ):
        if cls.mode == "note":
            return onset, offset, frame, velocity
        else:
            return onset, offset, frame
