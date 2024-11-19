from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .dataset import MidiLabel
from .midi import Note, Pedal


@dataclass
class Detection:
    loc: int
    time: float


def label_event(
    label_array: torch.Tensor,
    event_frames: List[int],
    event_ms: List[float],
    sharpness: int,
    hop_ms: float,
    num_frames: int,
    pitch: int | None = None,
    velocity: int | None = None,
    velocity_array: torch.Tensor | None = None,
):
    for event_frame, event_time_ms in zip(event_frames, event_ms):
        frame_range = range(
            max(0, event_frame - sharpness),
            min(num_frames, event_frame + sharpness + 1),
        )
        for frame_idx in frame_range:
            time_diff_ms = abs((frame_idx * hop_ms) - event_time_ms)
            value = max(0.0, 1.0 - (time_diff_ms / (sharpness * hop_ms)))
            if pitch is not None:
                label_array[frame_idx, pitch] = max(
                    label_array[frame_idx, pitch], value
                )
                if velocity_array is not None and label_array[frame_idx, pitch] >= 0.5:
                    velocity_array[frame_idx, pitch] = velocity
            else:
                label_array[frame_idx] = max(label_array[frame_idx], value)


def create_label_with_sharpness(
    notes: List[Note],
    pedals: List[Pedal],
    max_midi: int,
    min_midi: int,
    hop_length: int,
    sample_rate: int,
    offset_duration_tolerance_flag: bool = False,
):
    hop_ms = 1000 * hop_length / sample_rate
    onset_tolerance = int(50.0 / hop_ms + 0.5)
    offset_tolerance = int(50.0 / hop_ms + 0.5)
    num_frames_per_sec = sample_rate / hop_length
    max_offset = max(
        [note.end for note in notes]
        + ([pedal.end for pedal in pedals] if pedals else [])
    )
    num_frames = int(max_offset * num_frames_per_sec + 0.5) + 1
    num_notes = max_midi - min_midi + 1

    frame_label = torch.zeros((num_frames, num_notes), dtype=torch.int8)
    onset_label = torch.zeros((num_frames, num_notes))
    offset_label = torch.zeros((num_frames, num_notes))
    velocity_label = torch.zeros((num_frames, num_notes), dtype=torch.int8)
    pedal_frame_label = torch.zeros(num_frames, dtype=torch.int8)
    pedal_onset_label = torch.zeros(num_frames)
    pedal_offset_label = torch.zeros(num_frames)

    for note in notes:
        pitch = note.pitch - min_midi
        onset_frame = int(note.start * num_frames_per_sec + 0.5)
        onset_ms = note.start * 1000.0
        offset_frame = int(note.end * num_frames_per_sec + 0.5)
        offset_ms = note.end * 1000.0
        onset_sharpness = onset_tolerance
        offset_sharpness = offset_tolerance

        if offset_duration_tolerance_flag:
            duration_ms = offset_ms - onset_ms
            offset_duration_tolerance = int(duration_ms * 0.2 / hop_ms + 0.5)
            offset_sharpness = max(offset_tolerance, offset_duration_tolerance)

        velocity = note.velocity

        label_event(
            label_array=onset_label,
            event_frames=[onset_frame],
            event_ms=[onset_ms],
            sharpness=onset_sharpness,
            hop_ms=hop_ms,
            num_frames=num_frames,
            pitch=pitch,
            velocity=velocity,
            velocity_array=velocity_label,
        )

        label_offset = not any(
            other_note
            for other_note in notes
            if other_note is not note
            and other_note.pitch == note.pitch
            and other_note.end == note.start
        )
        if label_offset:
            label_event(
                label_array=offset_label,
                event_frames=[offset_frame],
                event_ms=[offset_ms],
                sharpness=offset_sharpness,
                hop_ms=hop_ms,
                num_frames=num_frames,
                pitch=pitch,
            )

        frame_label[onset_frame : offset_frame + 1, pitch] = 1

    for pedal in pedals:
        onpedal_frame = int(pedal.start * num_frames_per_sec + 0.5)
        onpedal_ms = pedal.start * 1000.0
        offpedal_frame = int(pedal.end * num_frames_per_sec + 0.5)
        offpedal_ms = pedal.end * 1000.0
        onpedal_sharpness = onset_tolerance
        offpedal_sharpness = offset_tolerance

        if offset_duration_tolerance_flag:
            duration_ms = offpedal_ms - onpedal_ms
            offpedal_duration_tolerance = int(duration_ms * 0.2 / hop_ms + 0.5)
            offpedal_sharpness = max(offset_tolerance, offpedal_duration_tolerance)

        label_event(
            label_array=pedal_onset_label,
            event_frames=[onpedal_frame],
            event_ms=[onpedal_ms],
            sharpness=onpedal_sharpness,
            hop_ms=hop_ms,
            num_frames=num_frames,
        )

        label_offset_pedal = not any(
            other_pedal
            for other_pedal in pedals
            if other_pedal is not pedal and other_pedal.end == pedal.start
        )
        if label_offset_pedal:
            label_event(
                label_array=pedal_offset_label,
                event_frames=[offpedal_frame],
                event_ms=[offpedal_ms],
                sharpness=offpedal_sharpness,
                hop_ms=hop_ms,
                num_frames=num_frames,
            )

        # is_on = False
        # for frame_idx in range(
        #     onpedal_frame - onset_sharpness, offpedal_frame + offset_sharpness + 1
        # ):
        #     if frame_idx < 0 or frame_idx >= num_frames:
        #         continue
        #     onset_value = pedal_onset_label[frame_idx]
        #     offset_value = pedal_offset_label[frame_idx]
        #     if not is_on and onset_value > 0.8:
        #         is_on = True
        #         pedal_frame_label[frame_idx] = 1
        #         continue

        #     if is_on and offset_value > 0.8:
        #         is_on = False
        #         break

        #     if is_on:
        #         pedal_frame_label[frame_idx] = 1
        pedal_frame_label[onpedal_frame : offpedal_frame + 1] = 1

    return MidiLabel(
        onset=onset_label,
        offset=offset_label,
        frame=frame_label > 0,
        velocity=velocity_label,
        pedal_onset=pedal_onset_label,
        pedal_offset=pedal_offset_label,
        pedal_frame=pedal_frame_label > 0,
    )


def create_label(
    notes: List[Note],
    pedals: List[Pedal],
    audio_length: int,
    max_midi: int,
    min_midi: int,
    hop_length: int,
    sample_rate: int,
    onset_length: int,
    offset_length: int,
):
    hops_in_onset = onset_length // hop_length
    hops_in_offset = offset_length // hop_length

    num_keys = max_midi - min_midi + 1
    num_frames = (audio_length - 1) // hop_length + 1

    note_label = torch.zeros((num_frames, num_keys), dtype=torch.uint8)
    velocity = torch.zeros((num_frames, num_keys), dtype=torch.uint8)

    pedal_label = torch.zeros(num_frames, dtype=torch.uint8)

    for note in notes:
        left = int(round(note.start * sample_rate / hop_length))
        onset_right = min(num_frames, left + hops_in_onset)
        frame_right = int(round(note.end * sample_rate / hop_length))
        frame_right = min(num_frames, frame_right)
        offset_right = min(num_frames, frame_right + hops_in_offset)

        f = int(note.pitch) - min_midi
        note_label[left:onset_right, f] = 3
        note_label[onset_right:frame_right, f] = 2
        note_label[frame_right:offset_right, f] = 1
        velocity[left:frame_right, f] = note.velocity

    for pedal in pedals:
        left = int(round(pedal.start * sample_rate / hop_length))
        onset_right = min(num_frames, left + hops_in_onset)
        frame_right = int(round(pedal.end * sample_rate / hop_length))
        frame_right = min(num_frames, frame_right)
        offset_right = min(num_frames, frame_right + hops_in_offset)

        pedal_label[left:onset_right] = 3
        pedal_label[onset_right:frame_right] = 2
        pedal_label[frame_right:offset_right] = 1

    return note_label, velocity, pedal_label


def detect_event(
    data: np.ndarray,
    thredhold: float,
):
    result: List[Detection] = []
    for i in range(len(data)):
        if data[i] >= thredhold:
            left_flag = True
            for ii in range(i - 1, -1, -1):
                if data[i] > data[ii]:
                    left_flag = True
                    break
                elif data[i] < data[ii]:
                    left_flag = False
                    break
            right_flag = True
            for ii in range(i + 1, len(data)):
                if data[i] > data[ii]:
                    right_flag = True
                    break
                elif data[i] < data[ii]:
                    right_flag = False
                    break

            if (left_flag is True) and (right_flag is True):
                if (i == 0) or (i == len(data) - 1):
                    time = i
                else:
                    if data[i - 1] == data[i + 1]:
                        time = i
                    elif data[i - 1] > data[i + 1]:
                        time = i - (
                            0.5 * (data[i - 1] - data[i + 1]) / (data[i] - data[i + 1])
                        )
                    else:
                        time = i + (
                            0.5 * (data[i + 1] - data[i - 1]) / (data[i] - data[i - 1])
                        )
                result.append(Detection(loc=i, time=time))

    return result


def process_label(
    pitch: int,
    onset_detections: List[Detection],
    offset_detections: List[Detection],
    frame: np.ndarray,
    threshold_frame: float,
    velocity: np.ndarray = None,
    mode_offset="shorter",
):
    num_frames = frame.shape[0]

    for onset_idx in range(len(onset_detections)):
        onset_loc = onset_detections[onset_idx].loc
        onset_time = onset_detections[onset_idx].time

        if onset_idx + 1 < len(onset_detections):
            next_onset_loc = onset_detections[onset_idx + 1].loc
            next_onset_time = onset_detections[onset_idx + 1].time
        else:
            next_onset_loc = num_frames
            next_onset_time = next_onset_loc

        offset_loc = onset_loc + 1
        found_offset = False

        for offset_detection in offset_detections:
            if onset_loc < offset_detection.loc:
                offset_loc = offset_detection.loc
                offset_time = offset_detection.time
                found_offset = True
                break

        if offset_loc > next_onset_loc:
            offset_loc = next_onset_loc
            offset_time = next_onset_time

        if not found_offset:
            offset_time = next_onset_time

        frame_offset_loc = onset_loc + 1
        found_frame_offset = False

        for frame_idx in range(onset_loc + 1, next_onset_loc):
            if frame[frame_idx][pitch] < threshold_frame:
                frame_offset_loc = frame_idx
                frame_offset_time = frame_offset_loc
                found_frame_offset = True
                break

        if not found_frame_offset:
            frame_offset_loc = next_onset_loc
            frame_offset_time = next_onset_time

        pitch_value = int(pitch)
        velocity_value = velocity[onset_loc][pitch] if velocity is not None else 0

        if not found_offset and not found_frame_offset:
            offset_value = next_onset_time
        elif found_offset and not found_frame_offset:
            offset_value = offset_time
        elif not found_offset and found_frame_offset:
            offset_value = frame_offset_time
        else:
            if mode_offset == "offset":
                offset_value = offset_time
            elif mode_offset == "longer":
                offset_value = (
                    offset_time if offset_loc >= frame_offset_loc else frame_offset_time
                )
            else:
                offset_value = (
                    offset_time if offset_loc <= frame_offset_loc else frame_offset_time
                )

        yield (onset_time, offset_value, pitch_value, velocity_value)


def extract_notes(
    onset: np.ndarray,
    offset: np.ndarray,
    frame: np.ndarray,
    velocity: np.ndarray,
    min_midi: int,
    max_midi: int,
    onset_threshold=0.5,
    offset_threshold=0.5,
    frame_threshold=0.5,
    mode_velocity="ignore_zero",
    mode_offset="shorter",
):
    num_notes = max_midi - min_midi + 1

    intervals: List[Tuple[int, int]] = []
    pitches: List[int] = []
    velocities: List[float] = []

    for pitch in range(num_notes):
        a_onset_detect = detect_event(onset[:, pitch], onset_threshold)
        a_offset_detect = detect_event(offset[:, pitch], offset_threshold)

        for time_onset, offset_value, pitch_value, velocity_value in process_label(
            pitch,
            a_onset_detect,
            a_offset_detect,
            frame,
            frame_threshold,
            velocity,
            mode_offset,
        ):
            if time_onset >= offset_value:
                continue

            if mode_velocity != "ignore_zero":
                intervals.append((time_onset, offset_value))
                pitches.append(pitch_value)
                velocities.append(velocity_value)
            else:
                if velocity_value > 0:
                    intervals.append((time_onset, offset_value))
                    pitches.append(pitch_value)
                    velocities.append(velocity_value)

            if (
                (len(intervals) > 1)
                and (pitches[len(pitches) - 1] == pitches[len(pitches) - 2])
                and (
                    intervals[len(intervals) - 1][0] < intervals[len(intervals) - 2][1]
                )
            ):
                new_onset = intervals[len(intervals) - 2][0]
                new_offset = intervals[len(intervals) - 1][1]
                intervals[len(intervals) - 2] = (new_onset, new_offset)
                intervals.pop()
                pitches.pop()
                velocities.pop()

    return pitches, intervals, velocities


def extract_pedals(
    onset: np.ndarray,
    offset: np.ndarray,
    frame: np.ndarray,
    onset_threshold=0.5,
    offset_threshold=0.5,
    frame_threshold=0.5,
    mode_offset="shorter",
):
    intervals: List[Tuple[float, float]] = []

    frame = frame[:, None]

    onset_detections = detect_event(onset, onset_threshold)
    offset_detections = detect_event(offset, offset_threshold)

    for time_onset, offset_value, _, _ in process_label(
        0,
        onset_detections,
        offset_detections,
        frame,
        frame_threshold,
        None,
        mode_offset,
    ):
        intervals.append((time_onset, offset_value))

    return np.array(intervals)


def create_notes(
    pitches: np.ndarray,
    intervals: np.ndarray,
    velocities: np.ndarray,
    min_midi: int,
):
    notes: List[Note] = []
    for p, i, v in zip(pitches, intervals, velocities):
        onset = i[0].item()
        offset = i[1].item()
        pitch = p.item() + min_midi
        velocity = min(127, max(0, int(v.item() * 127)))

        notes.append(Note(pitch, onset, offset, velocity))

    return notes


def create_pedals(
    intervals: np.ndarray,
):
    pedals: List[Pedal] = []
    for onset, offset in intervals:
        onset = onset.item()
        offset = offset.item()
        pedals.append(Pedal(onset, offset))
    return pedals
