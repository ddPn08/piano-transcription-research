from dataclasses import dataclass
from typing import List, Optional

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

    is_off = True
    for cc in midi.instruments[0].control_changes:
        if cc.number == 64:
            if cc.value == 0:
                is_off = True
            if cc.value >= 64:
                is_off = False
                if pedal is None:
                    pedal = Pedal(cc.time, None)
            elif pedal is not None:
                pedal.end = cc.time
                pedals.append(pedal)
                pedal = None
            elif len(pedals) > 0 and not is_off:
                pedals[-1].end = cc.time

    return notes, pedals


def create_midi(
    notes: List[Note],
    pedals: List[Pedal],
):
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

    for pedal in pedals:
        cc = pm.ControlChange(number=64, value=127, time=pedal.start)
        instrument.control_changes.append(cc)
        cc = pm.ControlChange(number=64, value=0, time=pedal.end)
        instrument.control_changes.append(cc)

    midi.instruments.append(instrument)

    return midi
