# pyright: basic

import argparse
import os
from dataclasses import asdict, dataclass
from typing import Literal

import datasets
import numpy as np
import torch
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import wandb
from satire.experiments.baselines.llm import evaluate_predictions


@dataclass
class BERTArguments:
    """Arguments for BERT-based irony detection evaluation."""

    model: str = "dumitrescustefan/bert-base-romanian-cased-v1"
    data_file: str = "data/csv/selerosa.csv"
    batch_size: int = 32
    seed: int = 42
    output_dir: str = "."
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


def parse_args() -> BERTArguments:
    defaults = BERTArguments()
    parser = argparse.ArgumentParser(
        description="Evaluate BERT model for irony detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=defaults.model,
        help="Path to the fine-tuned BERT model directory",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=defaults.data_file,
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults.output_dir,
        help="Directory for output files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        help="Device to use (auto, cpu, cuda, mps)",
    )
    parsed_args = parser.parse_args()
    return BERTArguments(**vars(parsed_args))


def preprocess_function(tokenizer):
    """Preprocessing function for tokenizing text data."""

    def process(example):
        result = tokenizer(example["sentence"], truncation=True, padding=False)
        result["labels"] = example["label"]
        return result

    return process


def log_outputs(
    gt: list[int | None],
    preds: list[int | None],
    test_dataset: datasets.Dataset,
    outputs_logits: list[list[float]],
) -> None:
    """Log predictions and outputs to wandb table."""
    table = wandb.Table(
        columns=[
            "idx",
            "sentence",
            "ground_truth",
            "prediction",
            "confidence",
            "logits",
        ]
    )
    for idx, data in enumerate(test_dataset):
        data = dict(data)
        confidence = max(outputs_logits[idx]) if outputs_logits[idx] else 0.0
        table.add_data(
            idx,
            data["sentence"],
            gt[idx],
            preds[idx],
            confidence,
            str(outputs_logits[idx]),
        )
    wandb.log({"predictions_table": table})


def get_device(device_arg: str) -> torch.device:
    """Determine the device to use for inference."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("MPS not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def predict_batch(
    batch,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> tuple[list[int], list[list[float]]]:
    """Make predictions on a batch of data."""
    model.eval()  # type: ignore

    batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

    with torch.no_grad():
        outputs = model(**batch)  # type: ignore
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        pred_list = predictions.cpu().numpy().tolist()
        logits_list = logits.cpu().numpy().tolist()

    return pred_list, logits_list


def main(args: BERTArguments) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wandb_group = os.environ.get("WANDB_GROUP")
    wandb_tags = os.environ.get("WANDB_TAGS")
    tags_list = wandb_tags.split(",") if wandb_tags else None

    wandb.init(config=asdict(args), group=wandb_group, tags=tags_list)

    print(f"Loading model from {args.model}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = get_device(args.device)
    model.to(device)
    print(f"Using device: {device}")

    dataset = datasets.load_dataset("csv", data_files=args.data_file)
    assert isinstance(dataset, DatasetDict)
    dataset = dataset["train"]
    test_dataset = dataset.filter(lambda example: example["split"] == "test")

    print(f"Test dataset size: {len(test_dataset)}")

    tokenized_test = test_dataset.map(preprocess_function(tokenizer), batched=True)
    tokenized_test = tokenized_test.remove_columns(
        [
            col
            for col in tokenized_test.column_names
            if col not in ["input_ids", "attention_mask", "labels"]
        ]
    )
    tokenized_test.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader = DataLoader(
        tokenized_test,  # type: ignore
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )

    all_predictions = []
    all_logits = []
    all_labels = []

    print("Making predictions...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        labels = batch["labels"].cpu().numpy().tolist()
        all_labels.extend(labels)

        batch_preds, batch_logits = predict_batch(batch, model, device)
        all_predictions.extend(batch_preds)
        all_logits.extend(batch_logits)

    gt: list[int | None] = all_labels
    preds: list[int | None] = all_predictions

    log_outputs(gt, preds, test_dataset, all_logits)
    evaluate_predictions(gt, preds, args=args)  # type: ignore

    wandb.finish()


if __name__ == "__main__":
    main(parse_args())
