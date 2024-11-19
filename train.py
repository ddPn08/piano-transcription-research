import fire
import torch
import yaml

from modules.config import Config
from modules.train import train

torch.set_float32_matmul_precision("medium")


def main(
    config: str = "config.yaml",
):
    with open(config) as f:
        config: Config = Config(**yaml.safe_load(f))

    train(config)


if __name__ == "__main__":
    fire.Fire(main)
