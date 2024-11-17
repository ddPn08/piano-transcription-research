from typing import Tuple

import numpy as np
import torch


def extract_notes(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    velocity: torch.Tensor,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(
                max(0, np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)
            )

    return np.array(pitches), np.array(intervals), np.array(velocities)


def extract_pedals(
    onsets: torch.Tensor,
    frames: torch.Tensor,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    onset_diff = torch.cat([onsets[:1], onsets[1:] - onsets[:-1]], dim=0) == 1

    intervals = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()

        onset = frame
        offset = frame

        while onsets[offset].item() or frames[offset].item():
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            intervals.append([onset, offset])

    return np.array(intervals)


def notes_to_frames(pitches: np.ndarray, intervals: np.ndarray, shape: Tuple):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs


def pedals_to_frames(intervals: torch.Tensor, shape: Tuple):
    """
    Takes lists specifying pedal sequences and return

    Parameters
    ----------
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    pedal: np.ndarray containing the pedal values
    """
    freq_value = 20
    roll = np.zeros(tuple(shape))
    for onset, offset in intervals:
        roll[onset:offset, freq_value] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
