from typing import Literal, Sequence

from pydantic import BaseModel


class TensorboardLoggerConfig(BaseModel):
    type: Literal["tensorboard"] = "tensorboard"
    name: str = "default"
    project: str = "piano-transcription-research"


class WandbLoggerConfig(BaseModel):
    type: Literal["wandb"] = "wandb"
    name: str | None = None
    project: str | None = None
    id: str | None = None
    tags: Sequence | None = None
    notes: str | None = None
