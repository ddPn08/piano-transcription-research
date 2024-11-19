from pydantic import BaseModel


class MidiConfig(BaseModel):
    min_midi: int = 21
    max_midi: int = 108
