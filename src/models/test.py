import pandas as pd
import torch
from tqdm import tqdm

from src.config import *
from src.data.dataset import MNISTDataset
from src.models.classifier import MNISTclassifier
from src.utils import get_transform, seed_everything


def test(model_path: str, annotation_path: str, output_path: str):

    seed_everything(RANDOM_STATE)

    # load and split data
    df = pd.read_csv(annotation_path, names=["img_path"], header=None)
    # df = df.sample(frac=1).iloc[:200, :]
    n_classes = N_CLASSES

    print(f"Test data: {df.shape[0]} images")

    # create test dataloader
    transform = get_transform()
    test_dataset = MNISTDataset(
        img_paths=df["img_path"],
        labels=None,
        preprocess_fn=transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2
    )

    # create model from checkpoint
    model = MNISTclassifier(n_classes=n_classes)
    model.load_state_dict(torch.load(model_path))

    pred_ids = []
    with tqdm(test_loader, unit="batch") as tepoch:
        for inputs, _ in tepoch:

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            pred_ids.extend(preds.tolist())

    df["label"] = pred_ids
    df.to_csv(output_path, index=False, header=False)

    print(f"Predictions saved at {output_path}")
