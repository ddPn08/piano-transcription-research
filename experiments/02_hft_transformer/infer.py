from typing import Optional

import fire
import numpy as np
import torch
import tqdm
import yaml

from modules.audio import create_mel_transform, load_audio
from modules.config import Config
from modules.config.hft_transformer import HftTransformerConfig
from modules.label import create_notes, create_pedals, extract_notes, extract_pedals
from modules.midi import create_midi
from modules.models.hft_transformer import (
    HftTransformer,
    HftTransformerParams,
    HftTransformerPedal,
)


def fix_state_dict(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(k.startswith("model.") for k in state_dict):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    return state_dict


def main(
    wav_path: str,
    output_path: str,
    model_path: str,
    config_path: str,
    pedal_model_path: Optional[str] = None,
    device: str = "cpu",
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    pedal_onset_threshold: float = 0.5,
    pedal_offset_threshold: float = 0.5,
    pedal_frame_threshold: float = 0.5,
):
    device = torch.device(device)

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        if "model" in data:
            config = Config(**data).model
        else:
            config = HftTransformerConfig(**data)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = fix_state_dict(state_dict)
    params = HftTransformerParams(
        n_frame=config.num_frame,
        n_mels=config.input.mel_spectrogram.n_mels,
        cnn_channel=4,
        cnn_kernel=5,
        hid_dim=256,
        n_margin=config.margin_b,
        n_layers=3,
        n_heads=4,
        pf_dim=512,
        dropout=0.1,
        n_velocity=127,
        n_note=config.output.midi.max_midi - config.output.midi.min_midi + 1,
    )

    model = HftTransformer(params)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pedal_model: Optional[HftTransformerPedal] = None
    if pedal_model_path is not None:
        state_dict = torch.load(
            pedal_model_path, map_location=device, weights_only=True
        )
        state_dict = fix_state_dict(state_dict)
        pedal_model = HftTransformerPedal(params)
        pedal_model.load_state_dict(state_dict)
        pedal_model.to(device)
        pedal_model.eval()

    audio = load_audio(
        wav_path, sample_rate=config.input.mel_spectrogram.sample_rate
    ).to(device)
    mel_transform = create_mel_transform(config.input.mel_spectrogram).to(device)

    mel_spec = mel_transform(audio)
    mel_spec = (torch.log(mel_spec + 1e-8)).T

    num_frames = mel_spec.shape[0]

    a_tmp_b = torch.full(
        [config.margin_b, config.input.mel_spectrogram.n_mels],
        1e-8,
        dtype=torch.float32,
        device=device,
    )
    len_s = (
        int(np.ceil(mel_spec.shape[0] / config.num_frame) * config.num_frame)
        - mel_spec.shape[0]
    )
    a_tmp_f = torch.full(
        [
            len_s + config.margin_f,
            config.input.mel_spectrogram.n_mels,
        ],
        1e-8,
        dtype=torch.float32,
        device=device,
    )
    mel_spec = torch.cat([a_tmp_b, mel_spec, a_tmp_f], axis=0)

    onset_pred_A_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    offset_pred_A_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    frame_pred_A_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    velocity_pred_A_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    onset_pred_B_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    offset_pred_B_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    frame_pred_B_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )
    velocity_pred_B_all = torch.zeros(
        (
            num_frames + len_s,
            config.output.midi.max_midi - config.output.midi.min_midi + 1,
        )
    )

    pedal_onset_pred_A_all = torch.zeros(num_frames + len_s)
    pedal_offset_pred_A_all = torch.zeros(num_frames + len_s)
    pedal_frame_pred_A_all = torch.zeros(num_frames + len_s)
    pedal_onset_pred_B_all = torch.zeros(num_frames + len_s)
    pedal_offset_pred_B_all = torch.zeros(num_frames + len_s)
    pedal_frame_pred_B_all = torch.zeros(num_frames + len_s)

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, num_frames, config.num_frame)):
            x = mel_spec[
                i : i + config.margin_b + config.num_frame + config.margin_f
            ].T.unsqueeze(0)

            (
                onset_pred_A,
                offset_pred_A,
                frame_pred_A,
                velocity_pred_A,
                _,
                onset_pred_B,
                offset_pred_B,
                frame_pred_B,
                velocity_pred_B,
            ) = model(x)

            onset_pred_A = onset_pred_A.sigmoid().squeeze(0).detach().cpu()
            offset_pred_A = offset_pred_A.sigmoid().squeeze(0).detach().cpu()
            frame_pred_A = frame_pred_A.sigmoid().squeeze(0).detach().cpu()
            velocity_pred_A = velocity_pred_A.squeeze(0).argmax(2).detach().cpu()

            onset_pred_B = onset_pred_B.sigmoid().squeeze(0).detach().cpu()
            offset_pred_B = offset_pred_B.sigmoid().squeeze(0).detach().cpu()
            frame_pred_B = frame_pred_B.sigmoid().squeeze(0).detach().cpu()
            velocity_pred_B = velocity_pred_B.squeeze(0).argmax(2).detach().cpu()

            onset_pred_A_all[i : i + config.num_frame] = onset_pred_A
            offset_pred_A_all[i : i + config.num_frame] = offset_pred_A
            frame_pred_A_all[i : i + config.num_frame] = frame_pred_A
            velocity_pred_A_all[i : i + config.num_frame] = velocity_pred_A

            onset_pred_B_all[i : i + config.num_frame] = onset_pred_B
            offset_pred_B_all[i : i + config.num_frame] = offset_pred_B
            frame_pred_B_all[i : i + config.num_frame] = frame_pred_B
            velocity_pred_B_all[i : i + config.num_frame] = velocity_pred_B

            if pedal_model is not None:
                (
                    onset_pred_A,
                    offset_pred_A,
                    frame_pred_A,
                    _,
                    onset_pred_B,
                    offset_pred_B,
                    frame_pred_B,
                ) = pedal_model(x)

                onset_pred_A = onset_pred_A.sigmoid().squeeze(0).detach().cpu()
                offset_pred_A = offset_pred_A.sigmoid().squeeze(0).detach().cpu()
                frame_pred_A = frame_pred_A.sigmoid().squeeze(0).detach().cpu()

                onset_pred_B = onset_pred_B.sigmoid().squeeze(0).detach().cpu()
                offset_pred_B = offset_pred_B.sigmoid().squeeze(0).detach().cpu()
                frame_pred_B = frame_pred_B.sigmoid().squeeze(0).detach().cpu()

                pedal_onset_pred_A_all[i : i + config.num_frame] = onset_pred_A
                pedal_offset_pred_A_all[i : i + config.num_frame] = offset_pred_A
                pedal_frame_pred_A_all[i : i + config.num_frame] = frame_pred_A

                pedal_onset_pred_B_all[i : i + config.num_frame] = onset_pred_B
                pedal_offset_pred_B_all[i : i + config.num_frame] = offset_pred_B
                pedal_frame_pred_B_all[i : i + config.num_frame] = frame_pred_B

    pitches_A, intervals_A, velocities_A = extract_notes(
        onset_pred_A_all,
        offset_pred_A_all,
        frame_pred_A_all,
        velocity_pred_A_all,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
        min_midi=config.output.midi.min_midi,
        max_midi=config.output.midi.max_midi,
    )
    pitches_B, intervals_B, velocities_B = extract_notes(
        onset_pred_B_all,
        offset_pred_B_all,
        frame_pred_B_all,
        velocity_pred_B_all,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
        min_midi=config.output.midi.min_midi,
        max_midi=config.output.midi.max_midi,
    )
    intervals_pedal_A = extract_pedals(
        pedal_onset_pred_A_all,
        pedal_offset_pred_A_all,
        pedal_frame_pred_A_all,
        onset_threshold=pedal_onset_threshold,
        offset_threshold=pedal_offset_threshold,
        frame_threshold=pedal_frame_threshold,
    )
    intervals_pedal_B = extract_pedals(
        pedal_onset_pred_B_all,
        pedal_offset_pred_B_all,
        pedal_frame_pred_B_all,
        onset_threshold=pedal_onset_threshold,
        offset_threshold=pedal_offset_threshold,
        frame_threshold=pedal_frame_threshold,
    )

    pitches, intervals, velocities = (
        np.array(pitches_A + pitches_B),
        np.array(intervals_A + intervals_B),
        np.array(velocities_A + velocities_B),
    )
    intervals_pedal = intervals_pedal_A + intervals_pedal_B

    scaling = (
        config.input.mel_spectrogram.hop_length
        / config.input.mel_spectrogram.sample_rate
    )

    intervals = (intervals * scaling).reshape(-1, 2)
    i_pedal_intervals_pedalest = (intervals_pedal * scaling).reshape(-1, 2)

    velocities = velocities / 127.0

    notes = create_notes(pitches, intervals, velocities, config.output.midi.min_midi)
    pedals = create_pedals(i_pedal_intervals_pedalest)

    midi = create_midi(
        notes,
        pedals,
    )
    midi.write(output_path)


if __name__ == "__main__":
    fire.Fire(main)
