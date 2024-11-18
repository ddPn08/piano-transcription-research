import torch
import torch.nn.functional as F
from pydantic import BaseModel

from modules.models import PianoTranscriptionBaseModel

from .decoder import Decoder, DecoderPedal
from .encoder import Encoder


class HftTransformerParams(BaseModel):
    n_frame: int
    n_mels: int
    cnn_channel: int
    cnn_kernel: int
    hid_dim: int
    n_margin: int
    n_layers: int
    n_heads: int
    pf_dim: int
    dropout: float
    n_velocity: int
    n_note: int


class HftTransformer(PianoTranscriptionBaseModel):
    mode = "note"

    @staticmethod
    def preprocess_label(
        onset: torch.Tensor,
        offset: torch.Tensor,
        frame: torch.Tensor,
        velocity: torch.Tensor,
    ):
        onset = onset.float()
        offset = offset.float()
        frame = frame.float()
        velocity = velocity.long()
        return onset, offset, frame, velocity

    def __init__(self, params: HftTransformerParams):
        super().__init__()
        self.encoder = Encoder(
            n_frame=params.n_frame,
            n_bin=params.n_mels,
            cnn_channel=params.cnn_channel,
            cnn_kernel=params.cnn_kernel,
            hid_dim=params.hid_dim,
            n_margin=params.n_margin,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            pf_dim=params.pf_dim,
            dropout=params.dropout,
        )
        self.decoder = Decoder(
            n_frame=params.n_frame,
            n_bin=params.n_mels,
            n_note=params.n_note,
            n_velocity=params.n_velocity,
            hid_dim=params.hid_dim,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            pf_dim=params.pf_dim,
            dropout=params.dropout,
        )

    def forward(self, spec: torch.Tensor):
        enc_vector = self.encoder(spec)
        return self.decoder(enc_vector)


class HftTransformerPedal(PianoTranscriptionBaseModel):
    mode = "pedal"

    @staticmethod
    def preprocess_label(
        onset: torch.Tensor,
        offset: torch.Tensor,
        frame: torch.Tensor,
    ):
        onset = onset.float()
        offset = offset.float()
        frame = frame.float()
        return onset, offset, frame

    def __init__(self, params: HftTransformerParams):
        super().__init__()
        self.encoder = Encoder(
            n_frame=params.n_frame,
            n_bin=params.n_mels,
            cnn_channel=params.cnn_channel,
            cnn_kernel=params.cnn_kernel,
            hid_dim=params.hid_dim,
            n_margin=params.n_margin,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            pf_dim=params.pf_dim,
            dropout=params.dropout,
        )
        self.decoder = DecoderPedal(
            n_frame=params.n_frame,
            n_bin=params.n_mels,
            hid_dim=params.hid_dim,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            pf_dim=params.pf_dim,
            dropout=params.dropout,
        )

    def forward(self, spec: torch.Tensor):
        enc_vector = self.encoder(spec)
        return self.decoder(enc_vector)
