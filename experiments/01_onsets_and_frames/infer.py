from typing import Optional

import fire
import torch
import tqdm
import yaml

from modules.audio import load_audio
from modules.config import Config
from modules.label import extract_notes, extract_pedals
from modules.midi import create_midi
from modules.models.onsets_and_frames import OnsetsAndFrames, OnsetsAndFramesPedal


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
        config = Config(**yaml.safe_load(f))

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = fix_state_dict(state_dict)
    model = OnsetsAndFrames(
        config.mel_spectrogram.n_mels,
        config.midi.max_pitch - config.midi.min_pitch + 1,
        config.model.model_complexity,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    pedal_model: Optional[OnsetsAndFramesPedal] = None
    if pedal_model_path is not None:
        state_dict = torch.load(
            pedal_model_path, map_location=device, weights_only=True
        )
        state_dict = fix_state_dict(state_dict)
        pedal_model = OnsetsAndFramesPedal(
            config.mel_spectrogram.n_mels, config.model.model_complexity
        )
        pedal_model.load_state_dict(state_dict)
        pedal_model.to(device)
        pedal_model.eval()

    audio = load_audio(wav_path, sample_rate=config.mel_spectrogram.sample_rate)
    mel_transform = config.mel_spectrogram.mel_transform().to(device)

    num_frames = (len(audio) - 1) // config.mel_spectrogram.hop_length + 1

    onset_pred_all = torch.zeros(
        (num_frames, config.midi.max_pitch - config.midi.min_pitch + 1)
    )
    offset_pred_all = torch.zeros(
        (num_frames, config.midi.max_pitch - config.midi.min_pitch + 1)
    )
    frame_pred_all = torch.zeros(
        (num_frames, config.midi.max_pitch - config.midi.min_pitch + 1)
    )
    velocity_pred_all = torch.zeros(
        (num_frames, config.midi.max_pitch - config.midi.min_pitch + 1)
    )

    pedal_onset_pred_all = torch.zeros(num_frames)
    pedal_offset_pred_all = torch.zeros(num_frames)
    pedal_frame_pred_all = torch.zeros(num_frames)

    segment_samples = (
        config.dataset.segment_seconds * config.mel_spectrogram.sample_rate
    )

    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(audio), segment_samples)):
            frame = i // config.mel_spectrogram.hop_length
            x = audio[i : i + segment_samples].to(device)
            mel = mel_transform(x.reshape(-1, x.shape[-1])[:, :-1])
            mel = torch.log(torch.clamp(mel, min=1e-5))
            mel = mel.transpose(-1, -2)
            onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

            onset_pred = (
                onset_pred.sigmoid()
                .reshape((onset_pred.shape[1], onset_pred.shape[2]))
                .detach()
                .cpu()
            )
            offset_pred = (
                offset_pred.sigmoid()
                .reshape((offset_pred.shape[1], offset_pred.shape[2]))
                .detach()
                .cpu()
            )
            frame_pred = (
                frame_pred.sigmoid()
                .reshape((frame_pred.shape[1], frame_pred.shape[2]))
                .detach()
                .cpu()
            )
            velocity_pred = (
                velocity_pred.sigmoid()
                .reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
                .detach()
                .cpu()
            )

            onset_pred_all[frame : frame + onset_pred.shape[0]] = onset_pred
            offset_pred_all[frame : frame + offset_pred.shape[0]] = offset_pred
            frame_pred_all[frame : frame + frame_pred.shape[0]] = frame_pred
            velocity_pred_all[frame : frame + velocity_pred.shape[0]] = velocity_pred

            if pedal_model is not None:
                onset_pred, offset_pred, _, frame_pred = pedal_model(mel)

                onset_pred = (
                    onset_pred.sigmoid().reshape(onset_pred.shape[1]).detach().cpu()
                )
                offset_pred = (
                    offset_pred.sigmoid().reshape(offset_pred.shape[1]).detach().cpu()
                )
                frame_pred = (
                    frame_pred.sigmoid().reshape(frame_pred.shape[1]).detach().cpu()
                )

                pedal_onset_pred_all[frame : frame + onset_pred.shape[0]] = onset_pred
                pedal_offset_pred_all[frame : frame + offset_pred.shape[0]] = (
                    offset_pred
                )
                pedal_frame_pred_all[frame : frame + frame_pred.shape[0]] = frame_pred

    p_est, i_est, v_est = extract_notes(
        onset_pred_all,
        offset_pred_all,
        frame_pred_all,
        velocity_pred_all,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
        min_midi=config.midi.min_pitch,
        max_midi=config.midi.max_pitch,
    )
    i_pedal_est = extract_pedals(
        pedal_onset_pred_all,
        pedal_offset_pred_all,
        pedal_frame_pred_all,
        onset_threshold=pedal_onset_threshold,
        offset_threshold=pedal_offset_threshold,
        frame_threshold=pedal_frame_threshold,
    )

    scaling = config.mel_spectrogram.hop_length / config.mel_spectrogram.sample_rate

    i_est = (i_est * scaling).reshape(-1, 2)
    i_pedal_est = (i_pedal_est * scaling).reshape(-1, 2)

    midi = create_midi(
        p_est,
        i_est,
        v_est,
        i_pedal_est,
        config.midi.min_pitch,
    )
    midi.write(output_path)


if __name__ == "__main__":
    fire.Fire(main)
