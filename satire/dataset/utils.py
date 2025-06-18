# pyright: basic
import os
import sys

import pandas as pd
from datasets import Dataset


def read_dataset(path: str) -> Dataset:
    df = load_dataset(path)
    return Dataset.from_pandas(df)


def load_dataset(dataset_path: str | None = None) -> pd.DataFrame:
    """Load the dataset from the specified path or use default."""
    if dataset_path is None:
        dataset_path = "./data/csv/selerosa.csv"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Current directory:", os.getcwd())
        print(
            "Available files:",
            os.listdir(
                os.path.dirname(dataset_path) if os.path.dirname(dataset_path) else "."
            ),
        )
        sys.exit(1)

    return pd.read_csv(dataset_path)


def dump_sentences(dataset_path: str, output_path: str) -> None:
    df = pd.read_csv(dataset_path)
    sentences = df["sentence"].tolist()
    with open(output_path, "w") as f:
        for sentence in sentences:
            cleaned_sentence = sentence.replace("\n", " ")
            f.write(cleaned_sentence + "\n")
