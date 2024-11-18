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
from modules.dataset.maestro import MaestroDataset
from modules.lightning.modules.hft_transformer import TranscriberModule
from modules.models.hft_transformer import (
    HftTransformer,
    HftTransformerParams,
    HftTransformerPedal,
)

torch.set_float32_matmul_precision("medium")


def main(
    config: str = "config.yaml",
):
    with open(config) as f:
        config: Config = Config(**yaml.safe_load(f))

    if config.hft_transformer is None:
        raise ValueError("config.hft_transformer is None")

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

    params = HftTransformerParams(
        n_frame=config.dataset.segment_frames,
        n_mels=config.mel_spectrogram.n_mels,
        cnn_channel=4,
        cnn_kernel=5,
        hid_dim=256,
        n_margin=config.hft_transformer.margin_b,
        n_layers=3,
        n_heads=4,
        pf_dim=512,
        dropout=0.1,
        n_velocity=127,
        n_note=config.midi.max_midi - config.midi.min_midi + 1,
    )

    model = (
        HftTransformer(params)
        if config.training.mode == "note"
        else HftTransformerPedal(params)
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
    elif config.training.logger == "tensorboard":
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
    trainer.fit(
        module,
        dataloader,
        val_dataloader,
        ckpt_path=config.training.resume_from_checkpoint,
    )

    state_dict = module.model.state_dict()
    torch.save(state_dict, os.path.join(config.training.output_dir, "model.pt"))


if __name__ == "__main__":
    fire.Fire(main)
