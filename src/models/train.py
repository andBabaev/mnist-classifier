import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import *
from src.data.dataset import MNISTDataset
from src.models.classifier import MNISTclassifier
from src.utils import get_transform, seed_everything


def train(annotation_path: str, output_path: str):

    seed_everything(RANDOM_STATE)

    # load and split data
    df = pd.read_csv(annotation_path, names=["img_path", "label"], header=None)
    # uncomment for debugging on small part of dataset
    # df = df.sample(frac=1).iloc[:200, :]

    train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(
        df["img_path"].values,
        df["label"].values,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=df["label"].values,
        random_state=RANDOM_STATE,
    )

    print(f"Training data: {train_imgs.shape[0]} images")
    print(f"Validation data: {valid_imgs.shape[0]} images")

    # create train dataloader
    transform = get_transform()
    n_classes = df["label"].max() + 1
    train_dataset = MNISTDataset(
        img_paths=train_imgs,
        labels=train_labels,
        n_classes=n_classes,
        preprocess_fn=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
    )

    # valid dataloader
    valid_dataset = MNISTDataset(
        img_paths=valid_imgs,
        labels=valid_labels,
        n_classes=n_classes,
        preprocess_fn=transform,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2
    )
    print("Dataset loaded\n")

    model = MNISTclassifier(n_classes=n_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_valid_acc = 0.0

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        # training stage
        training_loss = 0.0
        log_step = 50
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                tepoch.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
                training_loss += loss.item()
                if i % log_step == 0:
                    tepoch.set_postfix(loss=training_loss / log_step)
                    training_loss = 0.0

            # validation stage
            valid_loss = 0.0
            correct = 0
            total = 0
            model.eval()

            for inputs, labels in valid_loader:

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                valid_loss += loss.item()

                _, preds = torch.max(outputs.data, 1)
                _, label_ids = torch.max(labels.data, 1)
                correct += (preds == label_ids).sum().item()

            valid_loss = valid_loss / len(valid_loader)
            accuracy = 100 * correct / len(valid_dataset)
            print(f"val_loss: {valid_loss:.4f} val_accuracy: {accuracy:.4f} %")

            # save the best model
            if accuracy > best_valid_acc:
                best_valid_acc = accuracy
                torch.save(model.state_dict(), output_path)

    print(
        f"\nThe best model saved at: {output_path}\nThe best valid accuracy: {best_valid_acc:.4f} %"
    )
