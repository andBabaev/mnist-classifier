import os

import click


@click.command()
@click.option(
    "--train_path",
    show_default=True,
    default="data/raw/training",
    type=click.Path(exists=True),
    required=True,
    help="Path to folder with training data",
)
@click.option(
    "--test_path",
    show_default=True,
    required=True,
    default="data/raw/testing",
    type=click.Path(exists=True),
    help="Path to folder with test data",
)
@click.option(
    "--output_train",
    show_default=True,
    required=True,
    default="data/processed/train.csv",
    type=click.Path(),
    help="Path to save csv file with annotation for train data",
)
@click.option(
    "--output_test",
    show_default=True,
    required=True,
    default="data/processed/test.csv",
    type=click.Path(),
    help="Path to save csv file with annotation for test data",
)
@click.option(
    "--output_test_no_labels",
    show_default=True,
    required=True,
    default="data/processed/test_no_labels.csv",
    type=click.Path(),
    help="Path to save csv file with annotation (without labels, only file paths) for test data",
)
def main(
    train_path: str,
    test_path: str,
    output_train: str,
    output_test: str,
    output_test_no_labels: str,
):
    def dset_to_csv(input_path: str, output_path: str, add_label: bool = True) -> None:
        with open(output_path, "w") as f:
            for r, _, fnames in os.walk(input_path):
                for fname in fnames:
                    img_path = os.path.join(r, fname)
                    if add_label:
                        label = r.rsplit("/", 1)[-1]
                        f.write(f"{img_path},{label}\n")
                    else:
                        f.write(f"{img_path}\n")

    dset_to_csv(train_path, output_train)
    dset_to_csv(test_path, output_test)
    dset_to_csv(test_path, output_test_no_labels, add_label=False)


if __name__ == "__main__":
    main()
