import fire

from modules.config import Config
from modules.label import (
    create_label_with_sharpness,
    create_notes,
    create_pedals,
    extract_notes,
    extract_pedals,
)
from modules.midi import create_midi, parse_midi


def main(midi_path: str):
    config = Config()

    notes, pedals = parse_midi(midi_path)

    label = create_label_with_sharpness(
        notes,
        pedals,
        config.model.output.midi.max_midi,
        config.model.output.midi.min_midi,
        config.model.input.mel_spectrogram.hop_length,
        config.model.input.mel_spectrogram.sample_rate,
    )

    pitch_ref, intervals_ref, velocities_ref = extract_notes(
        label.onset,
        label.offset,
        label.frame,
        label.velocity / 127.0,
        config.model.output.midi.min_midi,
        config.model.output.midi.max_midi,
    )
    pedals_intervals_ref = extract_pedals(
        label.pedal_onset,
        label.pedal_offset,
        label.pedal_frame,
    )

    scaling = (
        config.model.input.mel_spectrogram.hop_length
        / config.model.input.mel_spectrogram.sample_rate
    )

    intervals_ref = (intervals_ref * scaling).reshape(-1, 2)
    pedals_intervals_ref = (pedals_intervals_ref * scaling).reshape(-1, 2)

    notes_ref = create_notes(
        pitch_ref, intervals_ref, velocities_ref, config.model.output.midi.min_midi
    )
    pedals_ref = create_pedals(pedals_intervals_ref)

    midi = create_midi(
        notes_ref,
        pedals_ref,
    )
    midi.write("output.mid")

    midi_2 = create_midi(
        notes,
        pedals,
    )
    midi_2.write("output_2.mid")


if __name__ == "__main__":
    fire.Fire(main)
