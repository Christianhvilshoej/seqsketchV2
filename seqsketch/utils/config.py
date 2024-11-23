# class for loading a yaml config file and accessing the config values
import yaml
import os
from dotmap import DotMap
from pathlib import Path
from seqsketch.utils.modules import get_class_from_string
import importlib
from pytorch_lightning.loggers import WandbLogger
import time
from pytorch_lightning.callbacks import ModelCheckpoint


class Config:
    def __init__(self, config_file=None, cli_args=None):
        if config_file:
            self.config = self.load_config(config_file)
        else:
            self.config = self.load_config(
                Path(__file__).parent.joinpath("default.yaml")
            )

    def create_inference_dir(self):
        inference_start = str(time.strftime("%Y%m%d_%H%M%S"))
        self.config.inference_start = inference_start
        inf_dir = Path(self.config.model_dir).joinpath(
            "inference", f"{self.config.inference.tag}_{inference_start}"
        )
        inf_dir.mkdir(parents=True, exist_ok=True)
        # dump the config file in the inference directory
        with open(inf_dir.joinpath("config.yaml"), "w") as f:
            yaml.dump(self.config.toDict(), f)
        # create sequential samples directory
        seq_dir = inf_dir.joinpath("sequential_samples")
        seq_dir.mkdir(parents=True, exist_ok=True)

    def get_config(self):
        return self.config

    def load_config(self, config_file):
        # load the config file and return a DotMap object
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return DotMap(config)

    def update(self, config_file):
        # merge the config file with the default config
        new_config = self.load_config(config_file)
        self.config.update(new_config)
        return self.config

    def add_cli_args(self, cli_args):
        # update the config with the cli args
        self.config.config_file = cli_args.config

    def create_model_dir(self):
        root = Path(__file__).parents[2]
        run_start = str(time.strftime("%Y%m%d_%H%M%S"))
        self.config.run_start = run_start
        self.model_dir = root.joinpath(
            "models", self.config.model_name, f"{self.config.version_name}_{run_start}"
        )
        # save relative path to root
        self.config.model_dir = str(self.model_dir.relative_to(root))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # save the config file in the model directory
        with open(self.model_dir.joinpath("config.yaml"), "w") as f:
            yaml.dump(self.config.toDict(), f)

    def get_logger(self):
        # Set WANDB_DIR environment variable
        logger = WandbLogger(
            name=f"{self.config.model_name}-{self.config.version_name}-{self.config.run_start}",
            project=self.config.project_name,
            log_model=False,
            save_dir=self.model_dir,
            offline=self.config.logger.offline,
        )
        return logger

    def get_callbacks(self, **kwargs):
        callbacks = []
        callback_list = self.config.trainer.callbacks
        for callback in callback_list:
            if callback == "ModelCheckpoint":
                callback = ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=self.model_dir.joinpath("checkpoints"),
                    filename="Checkpoint_epoch-{epoch:03d}_val_loss-{val_loss:.3f}",
                    save_top_k=1,  # saves 3 best models based on monitored value
                    save_last=True,  # additionally overwrites a file last.ckpt after each epoch
                    every_n_epochs=1,
                )
            elif callback == "ImageLogger":
                cls = get_class_from_string(f"seqsketch.callbacks.{callback}")
                callback = cls(kwargs["model"], kwargs["dataloader"])
            elif callback == "ImageLogger2":
                cls = get_class_from_string(f"seqsketch.callbacks.{callback}")
                callback = cls(kwargs["model"], kwargs["dataloader"])
            else:
                raise NotImplementedError(f"Callback {callback} not implemented")
            callbacks.append(callback)
        return callbacks

    def get_dataloader(self):
        cls = get_class_from_string(f"seqsketch.data.{self.config.data.module}")
        data_module = cls(self.config.data.params)
        data_module.prepare_data()
        data_module.setup()
        return data_module

    def get_model(self, load_pretrained_weights=False):
        cls = get_class_from_string(f"seqsketch.models.{self.config.model.module}")
        model = cls(self.config.model.params)
        if load_pretrained_weights:
            model.load_pretrained_weights(
                Path(self.config.model_dir).joinpath(
                    "checkpoints", self.config.inference.checkpoint
                )
            )
        return model

    def __str__(self) -> str:
        def print_dotmap(dotmap, indent=0):
            for key, value in dotmap.items():
                if isinstance(value, DotMap):
                    print("  " * indent + f"{key}:")
                    print_dotmap(
                        value, indent + 1
                    )  # Recursive call with increased indentation
                else:
                    print("  " * indent + f"{key}: {value}")

        print("<<<<<<<<<<<<<< Start of Config >>>>>>>>>>>>>>")
        print_dotmap(self.config)
        print("<<<<<<<<<<<<<< End of Config >>>>>>>>>>>>>>")
        return ""
