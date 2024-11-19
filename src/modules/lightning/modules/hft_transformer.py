from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from modules.label import extract_notes, extract_pedals
from modules.models.hft_transformer import HftTransformer, HftTransformerPedal

from ...config import Config
from ...evaluate import evaluate_note, evaluate_pedal


def weighted_mse_loss(
    velocity_pred: torch.Tensor, velocity_label: torch.Tensor, onset_label: torch.Tensor
):
    denominator = onset_label.sum()
    if denominator.item() == 0:
        return denominator
    else:
        return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class TranscriberModule(LightningModule):
    def __init__(
        self,
        config: Config,
        model: HftTransformer | HftTransformerPedal,
        optimizer_class: Any,
    ):
        super().__init__()
        self.config = config
        self.mode = model.mode
        self.model = model
        self.optimizer_class = optimizer_class
        self.lr = config.training.learning_rate

        self.mel_transform = config.mel_spectrogram.mel_transform().to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def mel_spectrogram(self, audio: torch.Tensor):
        mel_spec = self.mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-8).T

        num_feature_frames = (
            self.config.hft_transformer.margin_b
            + self.config.dataset.segment_frames
            + self.config.hft_transformer.margin_f
        )

        if mel_spec.shape[0] < num_feature_frames:
            pad = torch.zeros(
                num_feature_frames - mel_spec.shape[0],
                mel_spec.shape[1],
                dtype=mel_spec.dtype,
            )
            mel_spec = torch.cat([mel_spec, pad], dim=0)

        return mel_spec.T

    def calculate_loss(
        self,
        onset_pred: torch.Tensor,
        onset_label: torch.Tensor,
        offset_pred: torch.Tensor,
        offset_label: torch.Tensor,
        frame_pred: torch.Tensor,
        frame_label: torch.Tensor,
        velocity_pred: torch.Tensor | None = None,
        velocity_label: torch.Tensor | None = None,
    ):
        onset_pred = onset_pred.contiguous().view(-1)
        offset_pred = offset_pred.contiguous().view(-1)
        frame_pred = frame_pred.contiguous().view(-1)

        onset_label = onset_label.contiguous().view(-1)
        offset_label = offset_label.contiguous().view(-1)
        frame_label = frame_label.contiguous().view(-1)
        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)

        if velocity_pred is not None and velocity_label is not None:
            velocity_pred_dim = velocity_pred.shape[-1]
            velocity_pred = velocity_pred.contiguous().view(-1, velocity_pred_dim)

            velocity_label = velocity_label.contiguous().view(-1)

            velocity_loss = F.cross_entropy(velocity_pred, velocity_label)
            return onset_loss, offset_loss, frame_loss, velocity_loss
        else:
            return onset_loss, offset_loss, frame_loss

    def note_step(self, batch: torch.Tensor):
        audio, onset_label, offset_label, frame_label, velocity_label = batch
        onset_label, offset_label, frame_label, velocity_label = (
            self.model.preprocess_label(
                onset_label, offset_label, frame_label, velocity_label
            )
        )

        mel_spec = self.mel_spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1])

        (
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            velocity_pred_A,
            _,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
            velocity_pred_B,
        ) = self.model(mel_spec)

        return (
            onset_label,
            offset_label,
            frame_label,
            velocity_label,
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            velocity_pred_A,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
            velocity_pred_B,
        )

    def pedal_step(self, batch: torch.Tensor):
        audio, onset_label, offset_label, frame_label = batch
        (
            onset_label,
            offset_label,
            frame_label,
        ) = self.model.preprocess_label(onset_label, offset_label, frame_label)

        mel_spec = self.mel_spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1])

        (
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            _,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
        ) = self.model(mel_spec)

        return (
            onset_label,
            offset_label,
            frame_label,
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
        )

    def note_training_step(self, batch: torch.Tensor, _: int):
        (
            onset_label,
            offset_label,
            frame_label,
            velocity_label,
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            velocity_pred_A,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
            velocity_pred_B,
        ) = self.note_step(batch)

        onset_loss_A, offset_loss_A, frame_loss_A, velocity_loss_A = (
            self.calculate_loss(
                onset_pred_A,
                onset_label,
                offset_pred_A,
                offset_label,
                frame_pred_A,
                frame_label,
                velocity_pred_A,
                velocity_label,
            )
        )
        onset_loss_B, offset_loss_B, frame_loss_B, velocity_loss_B = (
            self.calculate_loss(
                onset_pred_B,
                onset_label,
                offset_pred_B,
                offset_label,
                frame_pred_B,
                frame_label,
                velocity_pred_B,
                velocity_label,
            )
        )

        loss_A = onset_loss_A + offset_loss_A + frame_loss_A + velocity_loss_A
        loss_B = onset_loss_B + offset_loss_B + frame_loss_B + velocity_loss_B

        loss = loss_A + loss_B

        self.log("loss/onset_A", onset_loss_A)
        self.log("loss/offset_A", offset_loss_A)
        self.log("loss/frame_A", frame_loss_A)
        self.log("loss/velocity_A", velocity_loss_A)
        self.log("loss/total_A", loss_A)

        self.log("loss/onset_B", onset_loss_B)
        self.log("loss/offset_B", offset_loss_B)
        self.log("loss/frame_B", frame_loss_B)
        self.log("loss/velocity_B", velocity_loss_B)
        self.log("loss/total_B", loss_B)

        self.log("loss/total", loss, prog_bar=True)

        return loss

    def pedal_training_step(self, batch: torch.Tensor, _: int):
        (
            onset_label,
            offset_label,
            frame_label,
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
        ) = self.pedal_step(batch)

        onset_loss_A, offset_loss_A, frame_loss_A = self.calculate_loss(
            onset_pred_A,
            onset_label,
            offset_pred_A,
            offset_label,
            frame_pred_A,
            frame_label,
        )

        onset_loss_B, offset_loss_B, frame_loss_B = self.calculate_loss(
            onset_pred_B,
            onset_label,
            offset_pred_B,
            offset_label,
            frame_pred_B,
            frame_label,
        )

        loss_A = onset_loss_A + offset_loss_A + frame_loss_A
        loss_B = onset_loss_B + offset_loss_B + frame_loss_B

        loss = loss_A + loss_B

        self.log("loss/onset_A", onset_loss_A)
        self.log("loss/offset_A", offset_loss_A)
        self.log("loss/frame_A", frame_loss_A)
        self.log("loss/total_A", loss_A)

        self.log("loss/onset_B", onset_loss_B)
        self.log("loss/offset_B", offset_loss_B)
        self.log("loss/frame_B", frame_loss_B)
        self.log("loss/total_B", loss_B)

        self.log("loss/total", loss, prog_bar=True)

        return loss

    def training_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            return self.note_training_step(batch, _)
        else:
            return self.pedal_training_step(batch, _)

    def note_validation_step(self, batch: torch.Tensor, _: int):
        (
            onset_label,
            offset_label,
            frame_label,
            velocity_label,
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            velocity_pred_A,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
            velocity_pred_B,
        ) = self.note_step(batch)

        for i in range(onset_label.shape[0]):
            pitches_ref, intervals_ref, velocities_ref = extract_notes(
                onset_label[i].cpu().numpy(),
                offset_label[i].cpu().numpy(),
                frame_label[i].cpu().numpy(),
                velocity_label[i].cpu().numpy(),
                self.config.midi.min_midi,
                self.config.midi.max_midi,
            )
            pitches_est_A, intervals_est_A, velocities_est_A = extract_notes(
                onset_pred_A[i].sigmoid().detach().cpu().numpy(),
                offset_pred_A[i].sigmoid().detach().cpu().numpy(),
                frame_pred_A[i].sigmoid().detach().cpu().numpy(),
                velocity_pred_A[i].sigmoid().argmax(2).detach().cpu().numpy(),
                self.config.midi.min_midi,
                self.config.midi.max_midi,
            )
            pitches_est_B, intervals_est_B, velocities_est_B = extract_notes(
                onset_pred_B[i].sigmoid().detach().cpu().numpy(),
                offset_pred_B[i].sigmoid().detach().cpu().numpy(),
                frame_pred_B[i].sigmoid().detach().cpu().numpy(),
                velocity_pred_B[i].sigmoid().argmax(2).detach().cpu().numpy(),
                self.config.midi.min_midi,
                self.config.midi.max_midi,
            )

            pitches_est, intervals_est, velocities_est = (
                pitches_est_A.extend(pitches_est_B),
                intervals_est_A.extend(intervals_est_B),
                velocities_est_A.extend(velocities_est_B),
            )

            metrics = evaluate_note(
                np.array(pitches_ref),
                np.array(intervals_ref),
                np.array(velocities_ref),
                np.array(pitches_est),
                np.array(intervals_est),
                np.array(velocities_est),
                frame_label.shape,
                self.config.mel_spectrogram.hop_length,
                self.config.mel_spectrogram.sample_rate,
                self.config.midi.min_midi,
            )
            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        onset_loss_A, offset_loss_A, frame_loss_A, velocity_loss_A = (
            self.calculate_loss(
                onset_pred_A,
                onset_label,
                offset_pred_A,
                offset_label,
                frame_pred_A,
                frame_label,
                velocity_pred_A,
                velocity_label,
            )
        )

        onset_loss_B, offset_loss_B, frame_loss_B, velocity_loss_B = (
            self.calculate_loss(
                onset_pred_B,
                onset_label,
                offset_pred_B,
                offset_label,
                frame_pred_B,
                frame_label,
                velocity_pred_B,
                velocity_label,
            )
        )
        loss_A = onset_loss_A + offset_loss_A + frame_loss_A + velocity_loss_A
        loss_B = onset_loss_B + offset_loss_B + frame_loss_B + velocity_loss_B

        loss = loss_A + loss_B

        self.log("val/loss/onset_A", onset_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset_A", offset_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame_A", frame_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/velocity_A", velocity_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/total_A", loss_A, on_epoch=True, sync_dist=True)

        self.log("val/loss/onset_B", onset_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset_B", offset_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame_B", frame_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/velocity_B", velocity_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/total_B", loss_B, on_epoch=True, sync_dist=True)

        self.log("val/loss/total", loss, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def pedal_validation_step(self, batch: torch.Tensor, _: int):
        (
            onset_label,
            offset_label,
            frame_label,
            onset_pred_A,
            offset_pred_A,
            frame_pred_A,
            onset_pred_B,
            offset_pred_B,
            frame_pred_B,
        ) = self.pedal_step(batch)

        onset_loss_A, offset_loss_A, frame_loss_A = self.calculate_loss(
            onset_pred_A,
            onset_label,
            offset_pred_A,
            offset_label,
            frame_pred_A,
            frame_label,
        )

        onset_loss_B, offset_loss_B, frame_loss_B = self.calculate_loss(
            onset_pred_B,
            onset_label,
            offset_pred_B,
            offset_label,
            frame_pred_B,
            frame_label,
        )

        for i in range(onset_label.shape[0]):
            intervals_ref = extract_pedals(
                onset_label[i].cpu().numpy(),
                offset_label[i].cpu().numpy(),
                frame_label[i].cpu().numpy(),
            )
            intervals_est_A = extract_pedals(
                onset_pred_A[i].sigmoid().detach().cpu().numpy(),
                offset_pred_A[i].sigmoid().detach().cpu().numpy(),
                frame_pred_A[i].sigmoid().detach().cpu().numpy(),
            )
            intervals_est_B = extract_pedals(
                onset_pred_B[i].sigmoid().detach().cpu().numpy(),
                offset_pred_B[i].sigmoid().detach().cpu().numpy(),
                frame_pred_B[i].sigmoid().detach().cpu().numpy(),
            )

            intervals_est = intervals_est_A.extend(intervals_est_B)

            metrics = evaluate_pedal(
                np.array(intervals_ref),
                np.array(intervals_est),
                frame_label.shape,
                self.config.mel_spectrogram.hop_length,
                self.config.mel_spectrogram.sample_rate,
            )
            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        loss_A = onset_loss_A + offset_loss_A + frame_loss_A
        loss_B = onset_loss_B + offset_loss_B + frame_loss_B

        loss = loss_A + loss_B

        self.log("val/loss/onset_A", onset_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset_A", offset_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame_A", frame_loss_A, on_epoch=True, sync_dist=True)
        self.log("val/loss/total_A", loss_A, on_epoch=True, sync_dist=True)

        self.log("val/loss/onset_B", onset_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset_B", offset_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame_B", frame_loss_B, on_epoch=True, sync_dist=True)
        self.log("val/loss/total_B", loss_B, on_epoch=True, sync_dist=True)

        self.log("val/loss/total", loss, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            return self.note_validation_step(batch, _)
        else:
            return self.pedal_validation_step(batch, _)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)
