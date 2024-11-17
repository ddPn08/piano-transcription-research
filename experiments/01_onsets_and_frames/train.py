import os

import fire
import torch
import torch.utils.data as data
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from modules.config import Config
from modules.dataset import MaestroDataset
from modules.models.onsets_and_frames import OnsetsAndFrames, OnsetsAndFramesPedal
from modules.training import TranscriberModule

torch.set_float32_matmul_precision("medium")


def main(
    config: str = "config.yml",
):
    with open(config) as f:
        config: Config = Config(**yaml.safe_load(f))

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

    model = (
        OnsetsAndFrames(
            config.mel_spectrogram.n_mels,
            config.midi.max_pitch - config.midi.min_pitch + 1,
            config.model.model_complexity,
        )
        if config.training.mode == "note"
        else OnsetsAndFramesPedal(config.mel_spectrogram, config.model.model_complexity)
    )

    if config.training.optimizer == "adam":
        optimizer_class = torch.optim.Adam
    else:
        raise ValueError("Invalid optimizer")

    module = TranscriberModule(config, model, optimizer_class)

    checkpoint_dir = os.path.join(config.training.output_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            every_n_epochs=1,
            dirpath=checkpoint_dir,
            save_top_k=10,
            mode="max",
            monitor="epoch",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if config.training.logger == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            name=config.training.logger_name,
            project=config.training.logger_project,
        )
    elif logger == "tensorboard":
        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger("lightning_logs", name=config.training.logger_name)
    else:
        logger = None

    trainer = Trainer(
        logger=logger,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        max_epochs=config.training.max_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=config.training.precision,
    )
    trainer.fit(module, dataloader, val_dataloader, ckpt_path=config.training.resume_from_checkpoint)

    state_dict = module.model.state_dict()
    torch.save(state_dict, os.path.join(config.training.output_dir, "model.pt"))


if __name__ == "__main__":
    fire.Fire(main)
