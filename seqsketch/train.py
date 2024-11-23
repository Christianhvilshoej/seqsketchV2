import argparse
from pathlib import Path
import time
import pytorch_lightning as pl


from seqsketch.utils.config import Config
import argparse
import os


def main(args):
    os.chdir(Path(__file__).parents[1])
    # load the config file
    configurator = Config(
        config_file=Path("seqsketch").joinpath("configs", args.config), cli_args=args
    )
    config = configurator.get_config()

    # create folder structure
    configurator.create_model_dir()

    print(configurator)

    dataloader = configurator.get_dataloader()
    model = configurator.get_model()
    wandb_logger = configurator.get_logger()
    callbacks = configurator.get_callbacks(model=model, dataloader=dataloader)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        precision=config.trainer.precision,
        accelerator=config.trainer.accelerator,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    )

    trainer.fit(model, dataloader)


def entrypoint():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config", type=Path, required=False, help="Path to the config file"
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
