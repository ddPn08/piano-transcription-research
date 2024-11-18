import sys

import numpy as np
import torch
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as evaluate_notes_with_velocity,
)
from mir_eval.util import midi_to_hz
from scipy.stats import hmean

from modules.decoding import (
    notes_to_frames,
    pedals_to_frames,
)

eps = sys.float_info.epsilon


def evaluate_note(
    pitch_ref: np.ndarray,
    intervals_ref: np.ndarray,
    velocity_ref: np.ndarray,
    pitch_est: np.ndarray,
    intervals_est: np.ndarray,
    velocity_est: np.ndarray,
    frame_shape: torch.Size,
    hop_length: int,
    sample_rate: int,
    min_midi: int,
):
    metrics = {}

    t_ref, f_ref = notes_to_frames(pitch_ref, intervals_ref, frame_shape)
    t_est, f_est = notes_to_frames(pitch_est, intervals_est, frame_shape)

    scaling = hop_length / sample_rate

    intervals_ref = (intervals_ref * scaling).reshape(-1, 2)
    pitch_ref = np.array([midi_to_hz(min_midi + midi) for midi in pitch_ref])
    intervals_est = (intervals_est * scaling).reshape(-1, 2)
    pitch_est = np.array([midi_to_hz(min_midi + midi) for midi in pitch_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [
        np.array([midi_to_hz(min_midi + midi) for midi in freqs]) for freqs in f_ref
    ]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [
        np.array([midi_to_hz(min_midi + midi) for midi in freqs]) for freqs in f_est
    ]

    p, r, f, o = evaluate_notes(
        intervals_ref, pitch_ref, intervals_est, pitch_est, offset_ratio=None
    )
    metrics["metric/note/precision"] = p
    metrics["metric/note/recall"] = r
    metrics["metric/note/f1"] = f
    metrics["metric/note/overlap"] = o

    p, r, f, o = evaluate_notes(intervals_ref, pitch_ref, intervals_est, pitch_est)
    metrics["metric/note-with-offsets/precision"] = p
    metrics["metric/note-with-offsets/recall"] = r
    metrics["metric/note-with-offsets/f1"] = f
    metrics["metric/note-with-offsets/overlap"] = o

    p, r, f, o = evaluate_notes_with_velocity(
        intervals_ref,
        pitch_ref,
        velocity_ref,
        intervals_est,
        pitch_est,
        velocity_est,
        offset_ratio=None,
        velocity_tolerance=0.1,
    )
    metrics["metric/note-with-velocity/precision"] = p
    metrics["metric/note-with-velocity/recall"] = r
    metrics["metric/note-with-velocity/f1"] = f
    metrics["metric/note-with-velocity/overlap"] = o

    p, r, f, o = evaluate_notes_with_velocity(
        intervals_ref,
        pitch_ref,
        velocity_ref,
        intervals_est,
        pitch_est,
        velocity_est,
        velocity_tolerance=0.1,
    )
    metrics["metric/note-with-offsets-and-velocity/precision"] = p
    metrics["metric/note-with-offsets-and-velocity/recall"] = r
    metrics["metric/note-with-offsets-and-velocity/f1"] = f
    metrics["metric/note-with-offsets-and-velocity/overlap"] = o

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics["metric/frame/f1"] = (
        hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps]) - eps
    )

    for key, loss in frame_metrics.items():
        metrics["metric/frame/" + key.lower().replace(" ", "_")] = loss

    return metrics


def evaluate_pedal(
    intervals_ref: np.ndarray,
    intervals_est: np.ndarray,
    frame_shape: torch.Size,
    hop_length: int,
    sample_rate: int,
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    metrics = {}

    t_ref, f_ref = pedals_to_frames(intervals_ref, frame_shape)
    t_est, f_est = pedals_to_frames(intervals_est, frame_shape)

    scaling = hop_length / sample_rate

    intervals_ref = (intervals_ref * scaling).reshape(-1, 2)
    intervals_est = (intervals_est * scaling).reshape(-1, 2)

    t_ref = t_ref.astype(np.float64) * scaling
    t_est = t_est.astype(np.float64) * scaling

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics["metric/frame/f1"] = (
        hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps]) - eps
    )

    for key, loss in frame_metrics.items():
        metrics["metric/frame/" + key.lower().replace(" ", "_")] = loss

    return metrics
