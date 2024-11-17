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
from modules.label import (
    extract_notes,
    extract_pedals,
)

eps = sys.float_info.epsilon


def evaluate_note(
    onset_true: torch.Tensor,
    offset_true: torch.Tensor,
    frame_true: torch.Tensor,
    velocity_true: torch.Tensor,
    onset_pred: torch.Tensor,
    offset_pred: torch.Tensor,
    frame_pred: torch.Tensor,
    velocity_pred: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    min_midi: int,
    max_midi: int,
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    metrics = {}

    p_ref, i_ref, v_ref = extract_notes(
        onset_true,
        offset_true,
        frame_true,
        velocity_true,
        min_midi,
        max_midi,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
    )
    p_est, i_est, v_est = extract_notes(
        onset_pred,
        offset_pred,
        frame_pred,
        velocity_pred,
        min_midi,
        max_midi,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
    )

    t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_true.shape)
    t_est, f_est = notes_to_frames(p_est, i_est, frame_pred.shape)

    scaling = hop_length / sample_rate

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(min_midi + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(min_midi + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [
        np.array([midi_to_hz(min_midi + midi) for midi in freqs]) for freqs in f_ref
    ]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [
        np.array([midi_to_hz(min_midi + midi) for midi in freqs]) for freqs in f_est
    ]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics["metric/note/precision"] = p
    metrics["metric/note/recall"] = r
    metrics["metric/note/f1"] = f
    metrics["metric/note/overlap"] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics["metric/note-with-offsets/precision"] = p
    metrics["metric/note-with-offsets/recall"] = r
    metrics["metric/note-with-offsets/f1"] = f
    metrics["metric/note-with-offsets/overlap"] = o

    p, r, f, o = evaluate_notes_with_velocity(
        i_ref,
        p_ref,
        v_ref,
        i_est,
        p_est,
        v_est,
        offset_ratio=None,
        velocity_tolerance=0.1,
    )
    metrics["metric/note-with-velocity/precision"] = p
    metrics["metric/note-with-velocity/recall"] = r
    metrics["metric/note-with-velocity/f1"] = f
    metrics["metric/note-with-velocity/overlap"] = o

    p, r, f, o = evaluate_notes_with_velocity(
        i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1
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
    onset_true: torch.Tensor,
    offset_true: torch.Tensor,
    frame_true: torch.Tensor,
    onset_pred: torch.Tensor,
    offset_pred: torch.Tensor,
    frame_pred: torch.Tensor,
    hop_length: int,
    sample_rate: int,
    onset_threshold: float = 0.5,
    offset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
):
    metrics = {}

    i_ref = extract_pedals(
        onset_true,
        offset_true,
        frame_true,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
    )
    i_est = extract_pedals(
        onset_pred,
        offset_pred,
        frame_pred,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        frame_threshold=frame_threshold,
    )

    t_ref, f_ref = pedals_to_frames(i_ref, frame_true.shape)
    t_est, f_est = pedals_to_frames(i_est, frame_pred.shape)

    scaling = hop_length / sample_rate

    i_ref = (i_ref * scaling).reshape(-1, 2)
    i_est = (i_est * scaling).reshape(-1, 2)

    t_ref = t_ref.astype(np.float64) * scaling
    t_est = t_est.astype(np.float64) * scaling

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics["metric/frame/f1"] = (
        hmean([frame_metrics["Precision"] + eps, frame_metrics["Recall"] + eps]) - eps
    )

    for key, loss in frame_metrics.items():
        metrics["metric/frame/" + key.lower().replace(" ", "_")] = loss

    return metrics
