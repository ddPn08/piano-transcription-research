import json
import multiprocessing as mp
import os
from typing import List

import fire
import torch
import tqdm
from pydantic import RootModel

from src.audio import load_audio
from src.dataset import Metadata
from src.midi import label_events, parse_midi


def process_metadata(
    worker_id: int,
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
        note_label, velocity, pedal = label_events(notes, pedals, len(audio))

        torch.save(audio, audio_path)
        torch.save(
            {"note": note_label, "velocity": velocity, "pedal": pedal}, label_path
        )


def main(
    dataset_path: str = "maestro-v3.0.0",
    dest_path: str = "maestro-v3.0.0-preprocessed",
    num_workers: int = 4,
    force_preprocess: bool = False,
):
    with open(os.path.join(dataset_path, "maestro-v3.0.0.json"), "r") as f:
        raw_metadata = json.load(f)

    metadata: List[Metadata] = []
    keys = list(raw_metadata.keys())

    for idx in range(len(raw_metadata[keys[0]])):
        data = {}
        for key in keys:
            data[key] = raw_metadata[key][str(idx)]
        metadata.append(Metadata.model_validate(data))

    label_dir = os.path.join(dest_path, "label")
    audio_dir = os.path.join(dest_path, "wav")
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(label_dir, split), exist_ok=True)
        os.makedirs(os.path.join(audio_dir, split), exist_ok=True)

    processes = []

    for idx in range(num_workers):
        p = mp.Process(
            target=process_metadata,
            args=(
                idx,
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
