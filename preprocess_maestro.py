import multiprocessing as mp
import os
from dataclasses import asdict
from typing import List

import fire
import torch
import tqdm
import yaml
from pydantic import RootModel

from modules.audio import load_audio
from modules.config import Config
from modules.dataset import Metadata
from modules.dataset.maestro import load_metadata
from modules.label import create_label_with_sharpness
from modules.midi import parse_midi


def process_metadata(
    worker_id: int,
    config: Config,
    metadata: List[Metadata],
    dataset_path: str,
    audio_dir: str,
    label_dir: str,
    force_preprocess: bool,
):
    for data in tqdm.tqdm(metadata, desc=f"Worker {worker_id}", position=worker_id):
        audio_path = os.path.join(
            audio_dir,
            data.split,
            data.audio_filename.replace("/", "-").replace("wav", "pt"),
        )
        label_path = os.path.join(
            label_dir,
            data.split,
            data.midi_filename.replace("/", "-").replace("midi", "pt"),
        )

        if (
            not force_preprocess
            and os.path.exists(label_path)
            and os.path.exists(audio_path)
        ):
            continue

        audio = load_audio(os.path.join(dataset_path, data.audio_filename))

        notes, pedals = parse_midi(os.path.join(dataset_path, data.midi_filename))
        midi_label = create_label_with_sharpness(
            notes,
            pedals,
            config.midi.max_midi,
            config.midi.min_midi,
            config.mel_spectrogram.hop_length,
            config.mel_spectrogram.sample_rate,
        )

        torch.save(audio, audio_path)
        torch.save(asdict(midi_label), label_path)


def main(
    config_path: str = "config.yaml",
    dataset_path: str = "maestro-v3.0.0",
    dest_path: str = "maestro-v3.0.0-preprocessed",
    num_workers: int = 4,
    force_preprocess: bool = False,
):
    with open(config_path, "r") as f:
        config = Config.model_validate(yaml.safe_load(f))

    metadata = load_metadata(dataset_path)

    label_dir = os.path.join(dest_path, "label")
    audio_dir = os.path.join(dest_path, "wav")
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(label_dir, split), exist_ok=True)
        os.makedirs(os.path.join(audio_dir, split), exist_ok=True)

    valid_metadata = [m for m in metadata if m.split in ["validation"]][:10]
    train_metadata = [m for m in metadata if m.split in ["train"]][:100]

    metadata = valid_metadata + train_metadata

    processes = []

    for idx in range(num_workers):
        p = mp.Process(
            target=process_metadata,
            args=(
                idx,
                config,
                metadata[idx::num_workers],
                dataset_path,
                audio_dir,
                label_dir,
                force_preprocess,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    metadata_path = os.path.join(dest_path, "metadata.json")
    with open(metadata_path, "w") as f:
        f.write(RootModel(metadata).model_dump_json())


if __name__ == "__main__":
    fire.Fire(main)
