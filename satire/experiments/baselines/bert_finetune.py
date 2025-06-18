# pyright: basic

import argparse
import os
import wandb

import datasets
import numpy as np
from datasets.dataset_dict import DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT model for classification"
    )

    # Data arguments
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/csv/selerosa.csv",
        help="Path to the data file",
    )

    # Model arguments
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="dumitrescustefan/bert-base-romanian-cased-v1",
        help="Path or name of the pre-trained model checkpoint",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of classes for classification",
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        help="Freeze all BERT layers except the classifier",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory to save logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging",
    )

    # Output arguments
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./bert-baseline",
        help="Directory to save the final model",
    )

    return parser.parse_args()


def preprocess_function(tokenizer):
    def process(example):
        result = tokenizer(example["sentence"], truncation=True)
        result["labels"] = example["label_0"]
        return result

    return process


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dataset = datasets.load_dataset("csv", data_files=args.data_file)
    assert isinstance(dataset, DatasetDict)
    dataset = dataset["train"]

    train_dataset = dataset.filter(lambda example: example["split"] == "train")
    valid_dataset = dataset.filter(lambda example: example["split"] == "validation")
    test_dataset = dataset.filter(lambda example: example["split"] == "test")

    all_splits = datasets.DatasetDict(
        {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenized_datasets = all_splits.map(preprocess_function(tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=args.num_labels
    )

    if args.freeze_bert:
        print("Freezing BERT encoder layers, only classifier will be trained.")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=args.seed,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=os.environ.get("WANDB_NAME", None),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    wandb.log(results)

    model.save_pretrained(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)


if __name__ == "__main__":
    main(parse_args())
