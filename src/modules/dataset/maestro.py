import json
import os
from typing import List, Literal

from pydantic import BaseModel

from ..config import Config
from . import TranscriptionDataset


class Metadata(BaseModel):
    canonical_composer: str
    canonical_title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float


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


def load_metadata(dataset_path: str):
    with open(os.path.join(dataset_path, "maestro-v3.0.0.json"), "r") as f:
        raw_metadata = json.load(f)

    metadata: List[Metadata] = []
    keys = list(raw_metadata.keys())

    for idx in range(len(raw_metadata[keys[0]])):
        data = {}
        for key in keys:
            data[key] = raw_metadata[key][str(idx)]
        metadata.append(Metadata.model_validate(data))

    return metadata
