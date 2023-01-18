import click
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


@click.command()
@click.option(
    " --ground-truth",
    "-gt",
    "ground_truth",
    show_default=True,
    default="data/processed/test.csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to csv file with annotations of test data. Each row includes path to image and ground-truth label",
)
@click.option(
    "--predictions",
    show_default=True,
    default="results.csv",
    type=click.Path(exists=True),
    help="Path to csv file with predictions for test data. Each row includes path to image and predicted label",
)
def main(ground_truth: str, predictions: str):
    target_df = pd.read_csv(ground_truth, header=None, names=["img_path", "label"])
    pred_df = pd.read_csv(predictions, header=None, names=["img_path", "pred"])

    merged_df = pd.merge(target_df, pred_df, how="left", on="img_path")
    accuracy = accuracy_score(merged_df["label"].values, merged_df["pred"].values)
    print(f"Accuracy: {accuracy}")

    # print confusion matrix as pandas dataframe
    cm = confusion_matrix(
        merged_df["label"].values, merged_df["pred"].values, labels=list(range(10))
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true:{id}" for id in range(10)],
        columns=[f"pred:{id}" for id in range(10)],
    )
    print("Confusion matrix:\n", cm_df)


if __name__ == "__main__":
    main()
