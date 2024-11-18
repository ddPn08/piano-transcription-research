from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from modules.label import extract_notes, extract_pedals
from modules.models.onsets_and_frames import OnsetsAndFrames, OnsetsAndFramesPedal

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
        model: OnsetsAndFrames | OnsetsAndFramesPedal,
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
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        mel_spec = mel_spec.transpose(-1, -2)
        return mel_spec

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
        onset_loss = F.binary_cross_entropy_with_logits(onset_pred, onset_label)
        offset_loss = F.binary_cross_entropy_with_logits(offset_pred, offset_label)
        frame_loss = F.binary_cross_entropy_with_logits(frame_pred, frame_label)

        if velocity_pred is not None and velocity_label is not None:
            velocity_loss = weighted_mse_loss(
                velocity_pred, velocity_label, onset_label
            )
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

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.model(mel_spec)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)
        velocity_pred = velocity_pred.reshape(velocity_label.shape)

        return (
            onset_label,
            offset_label,
            frame_label,
            velocity_label,
            onset_pred,
            offset_pred,
            frame_pred,
            velocity_pred,
        )

    def pedal_step(self, batch: torch.Tensor):
        audio, onset_label, offset_label, frame_label = batch
        (
            onset_label,
            offset_label,
            frame_label,
        ) = self.model.preprocess_label(onset_label, offset_label, frame_label)

        mel_spec = self.mel_spectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1])

        onset_pred, offset_pred, _, frame_pred = self.model(mel_spec)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)

        return (
            onset_label,
            offset_label,
            frame_label,
            onset_pred,
            offset_pred,
            frame_pred,
        )

    def note_training_step(self, batch: torch.Tensor, _: int):
        (
            onset_label,
            offset_label,
            frame_label,
            velocity_label,
            onset_pred,
            offset_pred,
            frame_pred,
            velocity_pred,
        ) = self.note_step(batch)

        onset_loss, offset_loss, frame_loss, velocity_loss = self.calculate_loss(
            onset_pred,
            onset_label,
            offset_pred,
            offset_label,
            frame_pred,
            frame_label,
            velocity_pred,
            velocity_label,
        )

        loss = onset_loss + offset_loss + frame_loss + velocity_loss

        self.log("loss/onset", onset_loss)
        self.log("loss/offset", offset_loss)
        self.log("loss/frame", frame_loss)
        self.log("loss/velocity", velocity_loss)
        self.log("loss/total", loss, prog_bar=True)
        self.log(
            "loss/epoch",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def pedal_training_step(self, batch: torch.Tensor, _: int):
        onset_label, offset_label, frame_label, onset_pred, offset_pred, frame_pred = (
            self.pedal_step(batch)
        )

        onset_loss, offset_loss, frame_loss = self.calculate_loss(
            onset_pred,
            onset_label,
            offset_pred,
            offset_label,
            frame_pred,
            frame_label,
        )

        loss = onset_loss + offset_loss + frame_loss

        self.log("loss/onset", onset_loss)
        self.log("loss/offset", offset_loss)
        self.log("loss/frame", frame_loss)
        self.log("loss/total", loss, prog_bar=True)
        self.log(
            "loss/epoch",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

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
            onset_pred,
            offset_pred,
            frame_pred,
            velocity_pred,
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
            pitches_est, intervals_est, velocities_est = extract_notes(
                onset_pred[i].sigmoid().detach().cpu().numpy(),
                offset_pred[i].sigmoid().detach().cpu().numpy(),
                frame_pred[i].sigmoid().detach().cpu().numpy(),
                velocity_pred[i].sigmoid().detach().cpu().numpy(),
                self.config.midi.min_midi,
                self.config.midi.max_midi,
            )
            metrics = evaluate_note(
                pitches_ref,
                intervals_ref,
                velocities_ref,
                pitches_est,
                intervals_est,
                velocities_est,
                frame_label.shape,
                self.config.mel_spectrogram.hop_length,
                self.config.mel_spectrogram.sample_rate,
                self.config.midi.min_midi,
            )

            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        onset_loss, offset_loss, frame_loss, velocity_loss = self.calculate_loss(
            onset_pred,
            onset_label,
            offset_pred,
            offset_label,
            frame_pred,
            frame_label,
            velocity_pred,
            velocity_label,
        )

        loss = onset_loss + offset_loss + frame_loss + velocity_loss

        self.log("val/loss/onset", onset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset", offset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame", frame_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/velocity", velocity_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/total", loss, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def pedal_validation_step(self, batch: torch.Tensor, _: int):
        onset_label, offset_label, frame_label, onset_pred, offset_pred, frame_pred = (
            self.pedal_step(batch)
        )

        onset_loss, offset_loss, frame_loss = self.calculate_loss(
            onset_pred,
            onset_label,
            offset_pred,
            offset_label,
            frame_pred,
            frame_label,
        )

        for i in range(onset_label.shape[0]):
            intervals_ref = extract_pedals(
                onset_label[i].cpu().numpy(),
                offset_label[i].cpu().numpy(),
                frame_label[i].cpu().numpy(),
            )
            intervals_est = extract_pedals(
                onset_pred[i].sigmoid().detach().cpu().numpy(),
                offset_pred[i].sigmoid().detach().cpu().numpy(),
                frame_pred[i].sigmoid().detach().cpu().numpy(),
            )
            metrics = evaluate_pedal(
                intervals_ref,
                intervals_est,
                frame_label.shape,
                self.config.mel_spectrogram.hop_length,
                self.config.mel_spectrogram.sample_rate,
            )

            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        loss = onset_loss + offset_loss + frame_loss

        self.log("val/loss/onset", onset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/offset", offset_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/frame", frame_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss/total", loss, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            return self.note_validation_step(batch, _)
        else:
            return self.pedal_validation_step(batch, _)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)
