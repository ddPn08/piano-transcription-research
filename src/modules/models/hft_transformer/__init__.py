import torch
import torch.nn.functional as F
from pydantic import BaseModel

from modules.models import PianoTranscriptionBaseModel

from .decoder import Decoder, DecoderPedal
from .encoder import Encoder


class HftTransformerParams(BaseModel):
    n_frame: int
    n_bin: int
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
    def preprocess_mel_spectrogram(mel_spec: torch.Tensor):
        return torch.log(torch.clamp(mel_spec, 1e-8))

    @staticmethod
    def calculate_loss(
        onset_pred: torch.Tensor,
        onset_label: torch.Tensor,
        offset_pred: torch.Tensor,
        offset_label: torch.Tensor,
        frame_pred: torch.Tensor,
        frame_label: torch.Tensor,
        velocity_pred: torch.Tensor | None = None,
        velocity_label: torch.Tensor | None = None,
    ):
        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)
        velocity_loss = F.cross_entropy(velocity_pred, velocity_label)

        return onset_loss, offset_loss, frame_loss, velocity_loss

    def __init__(self, params: HftTransformerParams):
        super().__init__()
        self.encoder = Encoder(
            n_frame=params.n_frame,
            n_bin=params.n_bin,
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
            n_bin=params.n_bin,
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
    def preprocess_mel_spectrogram(mel_spec: torch.Tensor):
        return torch.log(torch.clamp(mel_spec, 1e-8))

    @staticmethod
    def calculate_loss(
        onset_pred: torch.Tensor,
        onset_label: torch.Tensor,
        offset_pred: torch.Tensor,
        offset_label: torch.Tensor,
        frame_pred: torch.Tensor,
        frame_label: torch.Tensor,
    ):
        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)

        return onset_loss, offset_loss, frame_loss

    def __init__(self, params: HftTransformerParams):
        super().__init__()
        self.encoder = Encoder(
            n_frame=params.n_frame,
            n_bin=params.n_bin,
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
            n_bin=params.n_bin,
            hid_dim=params.hid_dim,
            n_layers=params.n_layers,
            n_heads=params.n_heads,
            pf_dim=params.pf_dim,
            dropout=params.dropout,
        )

    def forward(self, spec: torch.Tensor):
        enc_vector = self.encoder(spec)
        return self.decoder(enc_vector)
