import json
import os
from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd
import streamlit as st
from tqdm import tqdm

from modules.dataset import Metadata
from modules.midi import parse_midi


def load_metadata(dataset_dir):
    with open(os.path.join(dataset_dir, "maestro-v3.0.0.json"), "r") as f:
        raw_metadata = json.load(f)

    metadata_list: List[Metadata] = []
    keys = list(raw_metadata.keys())

    for idx in range(len(raw_metadata[keys[0]])):
        data = {}
        for key in keys:
            data[key] = raw_metadata[key][str(idx)]
        metadata_list.append(Metadata.model_validate(data))
    return metadata_list


def process_metadata(args: Tuple[Metadata, str]):
    metadata, dataset_dir = args
    midi_path = os.path.join(dataset_dir, metadata.midi_filename)
    if os.path.exists(midi_path):
        notes, pedals = parse_midi(midi_path)
        notes_count = len(notes)
        pedals_count = len(pedals)
    else:
        notes_count = 0
        pedals_count = 0
    return {
        **metadata.model_dump(),
        "notes_count": notes_count,
        "pedals_count": pedals_count,
    }


def main():
    st.title("メタデータとノート、ペダルの表示")

    dataset_dir = st.text_input("データセットのディレクトリを入力してください", ".")

    if dataset_dir and os.path.exists(os.path.join(dataset_dir, "maestro-v3.0.0.json")):
        with st.spinner("データを読み込んでいます..."):
            cached = os.path.exists("metadata.json")
            if cached:
                with open("metadata.json", "r") as f:
                    data_list = json.load(f)
            else:
                metadata_list = load_metadata(dataset_dir)
                with Pool() as pool:
                    data_list = list(
                        tqdm(
                            pool.imap(
                                process_metadata,
                                [(metadata, dataset_dir) for metadata in metadata_list],
                            ),
                            total=len(metadata_list),
                        )
                    )
            df = pd.DataFrame(data_list)
            st.dataframe(df)

            # cache
            with open("metadata.json", "w") as f:
                json.dump(data_list, f)
    else:
        st.error("データセットディレクトリが見つかりません")


if __name__ == "__main__":
    main()
