from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pretty_midi as pm


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: int


@dataclass
class Pedal:
    start: float
    end: float


def parse_midi(midi_path: str):
    midi = pm.PrettyMIDI(midi_path)

    notes: List[Note] = []
    pedals: List[Pedal] = []

    for note in midi.instruments[0].notes:
        notes.append(Note(note.pitch, note.start, note.end, note.velocity))

    pedal: Optional[Pedal] = None

    for cc in midi.instruments[0].control_changes:
        if cc.number == 64:
            if cc.value > 64:
                if pedal is None:
                    pedal = Pedal(cc.time, None)
            elif pedal is not None:
                pedal.end = cc.time
                pedals.append(pedal)
                pedal = None
                continue
            elif len(pedals) > 0:
                pedals[-1].end = cc.time

    return notes, pedals


def create_midi(
    pitches: np.ndarray,
    intervals: np.ndarray,
    velocities: np.ndarray,
    pedal_intervals: np.ndarray,
    min_midi: int,
):
    notes: List[Note] = []

    for idx, (onset, offset) in enumerate(intervals):
        onset = onset.item()
        offset = offset.item()
        pitch = pitches[idx].item() + min_midi
        velocity = min(127, max(0, int(velocities[idx].item() * 127)))

        notes.append(Note(pitch, onset, offset, velocity))

    pedal_events: List[Pedal] = []
    for onset, offset in pedal_intervals:
        onset = onset.item()
        offset = offset.item()
        pedal_events.append(Pedal(onset, offset))

    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(0)

    for note in notes:
        instrument.notes.append(
            pm.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            )
        )

    for pedal in pedal_events:
        cc = pm.ControlChange(number=64, value=127, time=pedal.start)
        instrument.control_changes.append(cc)
        cc = pm.ControlChange(number=64, value=0, time=pedal.end)
        instrument.control_changes.append(cc)

    midi.instruments.append(instrument)

    return midi
