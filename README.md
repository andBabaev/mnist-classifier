# MNIST classifier

## Description

This project includes the code for training of CNN for classification of MNIST
dataset. The classifier is a custom CNN developed using Pytorch.

## Environment

- Ubuntu 22.04
- Python 3.10

To create development virtual environment:

```bash
virtualenv .venv
source .venv/bin/activate

poetry install 
# or
# pip install -r requrements.txt
```

## Steps to reproduce

1. Extract `testing` and `training` folders from `mnist_png.tar.gz` file to `data/raw`. You can donwload this archive [here](https://github.com/myleott/mnist_png). You need to get the following directory tree

    ```text
    data
    |__ raw
        |__ training
        |   |__ 0
        |       |__ train_img.png
        |       |__...
        |__ testing
            |__ 0
                |__ test_img.png
                |__ ...
    ```

2. Generate annotation files with `src/data/make_dataset.py` script. It creates 3 annotation files: for train data, testing data, testing data without labels. This file supports the following options:

    - `--train_path` PATH - Path to folder with training data [default: data/raw/training; required]
    - `--test_path` PATH - Path to folder with test data  [default: data/raw/testing; required]
    - `--output_train` PATH - Path to save csv file with annotation for train data  [default: data/processed/train.csv; required]
    - `--output_test` PATH - Path to save csv file with annotation for test data  [default: data/processed/test.csv; required]
    - `--output_test_no_labels` PATH - Path to save csv file with annotation (without labels, only file paths) for test data [default: data/processed/ test_no_labels.csv; required]

    You can run with default options

    ```bash
    python src/data/make_dataset.py
    ```

3. Run training with `mnist.py` script. Also this script was used for test data. mnist.py support several options:

    - `--mode` TEXT - Set `train` or `inference` mode for this script  [required]
    - `--model` PATH - If mode='train', set a path to a file where a model checkpoint is saved. If mode='inference', set a path to your saved model checkpoint  [default: models/mnist.pth; required]
    - `--dataset` PATH - Required if mode='train'. Path to your csv file with training data. Each row includes a path to an image and a label  [default: data/processed/train.csv]
    - `--input` PATH - Required if mode='inference'. Path to your csv file with training data. Each row includes a path to a test image [default: data/processed/test_no_labels.csv]
    - `--output` PATH - Required if mode='inference'. Path to your csv file with prediction results  [default: results.csv]

    You must specify `--mode` option. It can have `train` or `inference` value. For other options
    you can use default parameters. Run

    ```bash
    python mnist.py --mode train
    python mnist.py --mode inference
    ```

    First command will do training of model and save the best checkpoint. Hyperparameters of training process can be changed at `src/config.py` file. Second command will load this checkpoint and test the model on test set of MNIST. This command

4. To compute test metrics you can use `evaluate.py` script. Options:

    - `--ground-truth`, `-gt` PATH - Path to csv file with annotations of test data. Each row includes path to image and ground-truth label  [default: data/processed/test.csv; required]
    - `--predictions` PATH-Path to csv file with predictions for test data. Each row includes path to image and predicted label  [default: results.csv]

    You can use default options and run:

    ```bash
    python evaluate.py
    ```

    This script will print accuracy value and confusion matrix in a terminal
