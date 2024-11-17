import torch
import torch.nn.functional as F
from torch import nn

from . import PianoTranscriptionBaseModel


def weighted_mse_loss(
    velocity_pred: torch.Tensor, velocity_label: torch.Tensor, onset_label: torch.Tensor
):
    denominator = onset_label.sum()
    if denominator.item() == 0:
        return denominator
    else:
        return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features: int, recurrent_features: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_features, recurrent_features, batch_first=True, bidirectional=True
        )

    def forward(self, x: torch.Tensor):
        if self.training:
            return self.rnn(x)[0]
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(
                batch_size,
                sequence_length,
                num_directions * hidden_size,
                device=x.device,
            )

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


def sequence_model(input_size, output_size):
    return BiLSTM(input_size, output_size // 2)


class ConvStack(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(PianoTranscriptionBaseModel):
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
        velocity = velocity.float() / 127.0
        return onset, offset, frame, velocity

    @staticmethod
    def preprocess_mel_spectrogram(mel_spec: torch.Tensor):
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        mel_spec = mel_spec.transpose(-1, -2)
        return mel_spec

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
        velocity_loss = weighted_mse_loss(velocity_pred, velocity_label, onset_label)
        return onset_loss, offset_loss, frame_loss, velocity_loss

    def __init__(
        self, input_features: int, output_features: int, model_complexity: int = 48
    ):
        super().__init__()

        model_size = model_complexity * 16

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
        )

    def forward(self, x: torch.Tensor):
        onset_pred = self.onset_stack(x)
        offset_pred = self.offset_stack(x)
        activation_pred = self.frame_stack(x)
        combined_pred = torch.cat(
            [onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1
        )
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(x)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred


class OnsetsAndFramesPedal(PianoTranscriptionBaseModel):
    mode = "pedal"

    @staticmethod
    def preprocess_mel_spectrogram(mel_spec: torch.Tensor):
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        mel_spec = mel_spec.transpose(-1, -2)
        return mel_spec

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

    def __init__(self, input_features: int, model_complexity: int = 48):
        super().__init__()

        output_features = 1
        model_size = model_complexity * 16

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
        )

    def forward(self, x: torch.Tensor):
        onset_pred = self.onset_stack(x)
        offset_pred = self.offset_stack(x)
        activation_pred = self.frame_stack(x)
        combined_pred = torch.cat(
            [onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1
        )
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, activation_pred, frame_pred
