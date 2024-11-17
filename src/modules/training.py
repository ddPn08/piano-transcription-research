from typing import Any

import torch
from lightning.pytorch import LightningModule

from modules.models import PianoTranscriptionBaseModel

from .config import Config
from .evaluate import evaluate_note, evaluate_pedal


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
        model: PianoTranscriptionBaseModel,
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

    def note_training_step(self, batch: torch.Tensor, _: int):
        audio, onset_label, offset_label, frame_label, velocity_label = batch
        onset_label, offset_label, frame_label, velocity_label = (
            self.model.preprocess_label(
                onset_label, offset_label, frame_label, velocity_label
            )
        )

        mel_spec = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel_spec = self.model.preprocess_mel_spectrogram(mel_spec)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.model(mel_spec)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)
        velocity_pred = velocity_pred.reshape(velocity_label.shape)

        onset_loss, offset_loss, frame_loss, velocity_loss = self.model.calculate_loss(
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
        audio, onset_label, offset_label, frame_label = batch
        (
            onset_label,
            offset_label,
            frame_label,
        ) = self.model.preprocess_label(onset_label, offset_label, frame_label)

        mel_spec = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel_spec = self.model.preprocess_mel_spectrogram(mel_spec)

        onset_pred, offset_pred, _, frame_pred = self.model(mel_spec)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)

        onset_loss, offset_loss, frame_loss = self.model.calculate_loss(
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
        audio, onset_label, offset_label, frame_label, velocity_label = batch
        onset_label, offset_label, frame_label, velocity_label = (
            self.model.preprocess_label(
                onset_label, offset_label, frame_label, velocity_label
            )
        )

        mel = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self.model(mel)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)
        velocity_pred = velocity_pred.reshape(velocity_label.shape)

        for i in range(onset_label.shape[0]):
            metrics = evaluate_note(
                onset_label[i],
                offset_label[i],
                frame_label[i],
                velocity_label[i],
                onset_pred[i],
                offset_pred[i],
                frame_pred[i],
                velocity_pred[i],
                self.config.mel_spectrogram.hop_length,
                self.config.mel_spectrogram.sample_rate,
                self.config.midi.min_pitch,
                self.config.midi.max_pitch,
            )

            for key, value in metrics.items():
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        onset_loss, offset_loss, frame_loss, velocity_loss = self.model.calculate_loss(
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
        audio, onset_label, offset_label, frame_label = batch
        (
            onset_label,
            offset_label,
            frame_label,
        ) = self.model.preprocess_label(onset_label, offset_label, frame_label)

        mel_spec = self.mel_transform(audio.reshape(-1, audio.shape[-1])[:, :-1])
        mel_spec = self.model.preprocess_mel_spectrogram(mel_spec)

        onset_pred, offset_pred, frame_pred = self.model(mel_spec)

        onset_pred = onset_pred.reshape(onset_label.shape)
        offset_pred = offset_pred.reshape(offset_label.shape)
        frame_pred = frame_pred.reshape(frame_label.shape)

        onset_loss, offset_loss, frame_loss = self.model.calculate_loss(
            onset_pred,
            onset_label,
            offset_pred,
            offset_label,
            frame_pred,
            frame_label,
        )

        for i in range(onset_label.shape[0]):
            metrics = evaluate_pedal(
                onset_label[i],
                offset_label[i],
                frame_label[i],
                onset_pred[i],
                offset_pred[i],
                frame_pred[i],
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

    def validation_step(self, batch: torch.Tensor, _: int):
        if self.mode == "note":
            return self.note_validation_step(batch, _)
        else:
            return self.pedal_validation_step(batch, _)

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), lr=self.lr)
