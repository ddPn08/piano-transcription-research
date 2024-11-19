import os

import torch
import torch.utils.data as data
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from modules.config import Config
from modules.dataset.maestro import MaestroDataset
from modules.lightning.modules.hft_transformer import HftTransformerTrainingModule
from modules.lightning.modules.onsets_and_frames import OnsetsAndFramesTrainingModule
from modules.models.hft_transformer import (
    HftTransformer,
    HftTransformerParams,
    HftTransformerPedal,
)
from modules.models.onsets_and_frames import (
    OnsetsAndFrames,
    OnsetsAndFramesParams,
    OnsetsAndFramesPedal,
)

torch.set_float32_matmul_precision("medium")


def init_logger(config: Config):
    if config.training.logger.type == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            name=config.training.logger.name,
            project=config.training.logger.project,
            id=config.training.logger.id,
            tags=config.training.logger.tags,
            notes=config.training.logger.notes,
        )
    elif config.training.logger.type == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger("lightning_logs", name=config.training.logger.name)
    else:
        return None


def init_model(config: Config):
    if config.model.type == "onsets_and_frames":
        if config.training.mode == "note":
            return OnsetsAndFrames(
                OnsetsAndFramesParams(
                    input_features=config.model.input.mel_spectrogram.n_mels,
                    output_features=config.model.output.midi.max_midi
                    - config.model.output.midi.min_midi
                    + 1,
                    model_complexity=config.model.complexity,
                )
            )
        elif config.training.mode == "pedal":
            return OnsetsAndFramesPedal(
                OnsetsAndFramesParams(
                    input_features=config.model.input.mel_spectrogram.n_mels,
                    output_features=1,
                    model_complexity=config.model.complexity,
                )
            )
        else:
            raise ValueError("Invalid mode")
    elif config.model.type == "hft_transformer":
        params = HftTransformerParams(
            n_frame=config.model.num_frame,
            n_mels=config.model.input.mel_spectrogram.n_mels,
            cnn_channel=config.model.cnn_channel,
            cnn_kernel=config.model.cnn_kernel,
            hid_dim=config.model.hid_dim,
            n_margin=config.model.margin_b,
            n_layers=config.model.num_layers,
            n_heads=config.model.num_heads,
            pf_dim=config.model.pf_dim,
            dropout=config.model.dropout,
            n_velocity=config.model.num_velocity,
            n_note=config.model.output.midi.max_midi
            - config.model.output.midi.min_midi
            + 1,
        )
        model_class = (
            HftTransformer if config.training.mode == "note" else HftTransformerPedal
        )
        return model_class(params)

    else:
        raise ValueError("Invalid model type")


def train(
    config: Config,
):
    dataset = MaestroDataset(config, "train")
    val_dataset = MaestroDataset(config, "validation")

    dataloader = data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=dataset.collate_fn,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=dataset.collate_fn,
    )

    model = init_model(config)

    if config.training.optimizer == "adam":
        optimizer_class = torch.optim.Adam
    else:
        raise ValueError("Invalid optimizer")

    if config.model.type == "onsets_and_frames":
        module = OnsetsAndFramesTrainingModule(config, model, optimizer_class)
    elif config.model.type == "hft_transformer":
        module = HftTransformerTrainingModule(config, model, optimizer_class)
    else:
        raise ValueError("Invalid model type")

    os.makedirs(config.training.output_dir, exist_ok=True)
    config_path = os.path.join(config.training.output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config.model.model_dump_json())

    checkpoint_dir = os.path.join(config.training.output_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=config.training.save_every_n_epochs,
            every_n_train_steps=config.training.save_every_n_steps,
            dirpath=checkpoint_dir,
            save_top_k=10,
            mode="max",
            monitor="step",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = init_logger(config)

    trainer = Trainer(
        logger=logger,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        max_epochs=config.training.max_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=config.training.precision,
    )
    trainer.fit(
        module,
        dataloader,
        val_dataloader,
        ckpt_path=config.training.resume_from_checkpoint,
    )

    state_dict = module.model.state_dict()
    torch.save(state_dict, os.path.join(config.training.output_dir, "model.pt"))
