import argparse
from pathlib import Path
import time
import pytorch_lightning as pl


from seqsketch.utils.config import Config
import argparse
import os


def main(args):
    # set current working directory as Path(__file__).parents[1]
    os.chdir(Path(__file__).parents[1])
    # load the config file
    configurator = Config(
        cli_args=args,
        config_file=Path("models").joinpath(args.model_folder, "config.yaml"),
    )
    configurator.update(
        Path(configurator.config.model_dir).joinpath(args.inference_config)
    )
    config = configurator.get_config()

    print(configurator)

    model = configurator.get_model(load_pretrained_weights=True)
    model = model.to(config.inference.device)
    model.eval()

    # create inference folder
    configurator.create_inference_dir()

    # generate sequential samples
    for i in range(config.inference.num_samples):
        sketch_sequence = model(bs=1, iterations=config.inference.num_iterations)
        # make subfolder for each sample
        sample_dir = config.model_dir.joinpath(f"sample_{i}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        for j, sketch in enumerate(sketch_sequence):
            sketch.save(sample_dir.joinpath(f"sketch_{j}.png"))


def entrypoint():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--inference_config", type=str, default="inference_config.yaml")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    entrypoint()
