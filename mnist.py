import os

import click

from src.models.test import test
from src.models.train import train


@click.command()
@click.option(
    "--mode", required=True, help="Set `train` or `inference` mode for this script"
)
@click.option(
    "--model",
    show_default=True,
    default="models/mnist.pth",
    type=click.Path(),
    required=True,
    help="If mode='train', set a path to a file where a model checkpoint is saved. If mode='inference', set a path to your saved model checkpoint",
)
@click.option(
    "--dataset",
    show_default=True,
    default="data/processed/train.csv",
    type=click.Path(exists=True),
    help="Required if mode='train'. Path to your csv file with training data. Each row includes a path to an image and a label",
)
@click.option(
    "--input",
    show_default=True,
    default="data/processed/test_no_labels.csv",
    type=click.Path(exists=True),
    help="Required if mode='inference'. Path to your csv file with training data. Each row includes a path to a test image",
)
@click.option(
    "--output",
    show_default=True,
    default="results.csv",
    type=click.Path(),
    help="Required if mode='inference'. Path to your csv file with prediction results",
)
def main(mode: str, dataset: str, model: str, input: str, output: str):
    if mode == "train":
        train(dataset, model)

    elif mode == "inference":
        if not os.path.exists(model):
            print(f"Invalid value for '--model': Path '{model}' does not exist.")
            return
        test(
            model_path=model,
            annotation_path=input,
            output_path=output,
        )
    else:
        print(
            f"Incorrect value for '--mode': '{mode}'. Available mode names: 'train', 'inference'"
        )


if __name__ == "__main__":
    main()
