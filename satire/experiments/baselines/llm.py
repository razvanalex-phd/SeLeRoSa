# pyright: basic
import argparse
import concurrent.futures
import functools
import json
import os
import random
from dataclasses import asdict, dataclass

import backoff
import datasets
from datasets.dataset_dict import DatasetDict
from openai import OpenAI, RateLimitError
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

import wandb
from satire.experiments.baselines.prompts import (
    CLASS_0,
    CLASS_1,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_SIMPLE,
    USER_PROMPT_CHAT,
    USER_PROMPT_COMPLETION,
    build_prompt,
)


@dataclass
class PredSample:
    idx: int
    prediction: int
    label: int
    output: str
    reasoning: str | None
    prompt: str | list[dict[str, str]]


@dataclass
class Prediction:
    answer: int
    output: str
    prompt: str | list[dict[str, str]]
    reasoning: str | None = None


@dataclass
class LLMArguments:
    """Arguments for LLM-based irony detection."""

    from_ft_model: bool = False
    use_completion: bool = False
    merge_system_into_user: bool = False
    model: str = "llama3.1:latest"
    data_file: str = "data/csv/iter1.csv"
    seed: int = 42
    output_dir: str = "."
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.0
    top_k: int = -1  # OpenAI default: disabled (-1)
    top_p: float = 1.0  # OpenAI default: 1.0
    min_p: float = 0.0  # OpenAI default: 0.0
    num_workers: int = 8


def parse_args() -> LLMArguments:
    defaults = LLMArguments()
    parser = argparse.ArgumentParser(description="Run LLM-based irony detection")
    parser.add_argument(
        "--from-ft-model",
        action="store_true",
        default=defaults.from_ft_model,
        help="Use SYSTEM_PROMPT_SIMPLE instead of SYSTEM_PROMPT_CHAT for fine-tuned models",
    )
    parser.add_argument(
        "--use-completion",
        action="store_true",
        default=defaults.use_completion,
        help="Use completion (not chat) API for non-chat models like GPT-2",
    )
    parser.add_argument(
        "--merge-system-into-user",
        action="store_true",
        default=defaults.merge_system_into_user,
        help="Merge system prompt into user prompt (for models that do not support system prompts)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=defaults.model,
        help="Model name to use",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=defaults.data_file,
        help="Path to dataset CSV file",
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
        "--base-url",
        type=str,
        default=defaults.base_url,
        help="Base URL for OpenAI API",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help="Temperature for LLM generation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=defaults.top_k,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=defaults.top_p,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=defaults.min_p,
        help="Min-p sampling parameter",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.num_workers,
        help="Number of worker threads for parallel requests",
    )
    parsed_args = parser.parse_args()
    return LLMArguments(**vars(parsed_args))


def parser_output_completion(output: str, class0: str, class1: str) -> int:
    answer = None
    for label in (class0, class1):
        if label in output:
            answer = label
            break
    if answer is None:
        answer = output.split()[0] if output else "-1"
    answer = "".join(filter(str.isalpha, answer))

    if CLASS_0 in answer:
        return 0
    if CLASS_1 in answer:
        return 1
    return -1


def parse_output_chat(output: str, class0: str, class1: str) -> int:
    try:
        index = output.upper().find("RĂSPUNS:")
        answer = output[index:].split(":")[1].split("\n")[0].strip()
    except:
        print("Falling back to simple searching the class label.")
        answer = output.strip()
    answer = answer.lower()
    return parser_output_completion(answer, class0, class1)


def is_openai_reasoning_model(model: str) -> bool:
    """Check if the model is an OpenAI reasoning model that requires Responses API."""
    reasoning_models = {
        "o3",
        "o3-mini",
        "o3-pro",
        "o4-mini",
        "o1",
        "o1-mini",
        "o1-preview",
        "o1-pro",
    }
    # Check if model name contains any reasoning model identifier
    model_lower = model.lower()
    return any(reasoning_model in model_lower for reasoning_model in reasoning_models)


def is_rate_limit_error(e):
    if hasattr(e, "status_code") and e.status_code == 429:
        return True
    if isinstance(e, RateLimitError):
        return True
    return False


def process_completion(
    _idx: int,
    data: dict,
    class0: str,
    class1: str,
    *,
    client: OpenAI,
    args: LLMArguments,
) -> Prediction:
    # This is for base models that do not support chat mode
    prompt = build_prompt(
        data,
        system=SYSTEM_PROMPT_SIMPLE,
        user=USER_PROMPT_COMPLETION,
        class0=class0,
        class1=class1,
        use_chat_template=False,
    )
    assert isinstance(prompt, str)
    # Build extra parameters for sampling
    extra_params = {}
    if args.top_k != -1:  # Only include if not using OpenAI default
        extra_params["top_k"] = args.top_k
    if args.min_p != 0.0:  # Only include if not using OpenAI default
        extra_params["min_p"] = args.min_p

    completion_args = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 5,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if extra_params:
        completion_args["extra_body"] = extra_params

    response = client.completions.create(**completion_args)
    output = response.choices[0].text.strip()
    answer = parser_output_completion(output, class0, class1)
    # NOTE: Reasoning is not available for completion API
    return Prediction(prompt=prompt, answer=answer, output=output, reasoning=None)


def process_responses_api(
    _idx: int,
    data: dict,
    class0: str,
    class1: str,
    *,
    client: OpenAI,
    args: LLMArguments,
) -> Prediction:
    """Process using OpenAI Responses API for reasoning models."""
    if args.from_ft_model:
        system = SYSTEM_PROMPT_SIMPLE.format(class0=class0, class1=class1)
    else:
        system = SYSTEM_PROMPT_CHAT.format(class0=class0, class1=class1)

    messages = build_prompt(
        data,
        system="",
        user=USER_PROMPT_CHAT,
        class0=class0,
        class1=class1,
        use_chat_template=True,
        merge_system_into_user=True,
    )
    messages = messages["messages"]
    assert isinstance(messages, list) and len(messages) == 1

    response = client.responses.create(
        model=args.model,
        instructions=system,
        input=messages[0]["content"],
        reasoning={
            "effort": "medium",
            # "summary": "auto", # TODO: Requires organisation verification
        },
        max_output_tokens=30000,
    )

    output = response.output_text
    if output is None:
        raise Exception("No output")

    reasoning = response.reasoning.summary if response.reasoning else None

    answer = parse_output_chat(output, class0, class1)
    return Prediction(
        prompt=messages,
        answer=answer,
        output=output,
        reasoning=reasoning,
    )


def process_chat(
    _idx: int,
    data: dict,
    class0: str,
    class1: str,
    *,
    client: OpenAI,
    args: LLMArguments,
) -> Prediction:
    if args.from_ft_model:
        system = SYSTEM_PROMPT_SIMPLE.format(class0=class0, class1=class1)
    else:
        system = SYSTEM_PROMPT_CHAT.format(class0=class0, class1=class1)

    messages = build_prompt(
        data,
        system=system,
        user=USER_PROMPT_CHAT,
        class0=class0,
        class1=class1,
        use_chat_template=True,
        merge_system_into_user=args.merge_system_into_user,
    )
    messages = messages["messages"]
    assert isinstance(messages, list)
    # Build extra parameters for sampling
    extra_params = {}
    if args.top_k != -1:  # Only include if not using OpenAI default
        extra_params["top_k"] = args.top_k
    if args.min_p != 0.0:  # Only include if not using OpenAI default
        extra_params["min_p"] = args.min_p

    chat_args = {
        "model": args.model,
        "messages": messages,  # type: ignore
        "max_tokens": 50,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if extra_params:
        chat_args["extra_body"] = extra_params

    if is_openai_reasoning_model(args.model):
        # NOTE: this should not be reachable. Probably, we should expand this
        # for other reasoning models, if to be used.
        chat_args["reasoning"] = {"effort": "medium"}

    response = client.chat.completions.create(**chat_args)

    output = response.choices[0].message.content
    if output is None:
        raise Exception("No output")

    # TODO: Not implemented yet for reasoning. Probably, should parse what is
    # inside <thinking> tags, and the output should be what is outside those
    # tags. Private LLMs from OpenAI do not provide access to reasoning tokens.
    reasoning = None

    answer = parse_output_chat(output, class0, class1)
    return Prediction(
        prompt=messages,
        answer=answer,
        output=output,
        reasoning=reasoning,
    )


def randomize_classes(
    class0: str,
    class1: str,
) -> tuple[str, str]:
    """Randomly shuffle classes to avoid bias."""
    classes = [class0, class1]
    random.shuffle(classes)
    return classes[0], classes[1]


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    giveup=lambda e: not is_rate_limit_error(e),
    jitter=None,
)
def process(
    idx_data: tuple[int, dict],
    *,
    client: OpenAI,
    args: LLMArguments,
) -> PredSample:
    idx, data = idx_data
    data = dict(data)
    class0, class1 = randomize_classes(CLASS_0, CLASS_1)

    process_args = {
        "_idx": idx,
        "data": data,
        "class0": class0,
        "class1": class1,
        "client": client,
        "args": args,
    }

    try:
        if args.use_completion:
            ret = process_completion(**process_args)
        elif is_openai_reasoning_model(args.model):
            ret = process_responses_api(**process_args)
        else:
            ret = process_chat(**process_args)

        return PredSample(
            idx=idx,
            prediction=ret.answer,
            label=int(data["label"]),
            prompt=ret.prompt,
            output=ret.output,
            reasoning=ret.reasoning,
        )
    except KeyboardInterrupt:
        exit(-1)
    except Exception:
        raise


def process_with_fallback(
    idx_data: tuple[int, dict],
    *,
    client: OpenAI,
    args: LLMArguments,
) -> PredSample:
    try:
        return process(idx_data, client=client, args=args)
    except Exception as e:
        print(f"Error processing idx {idx_data[0]}: {e}")
        idx, data = idx_data
        return PredSample(
            idx=idx,
            prediction=-1,
            label=int(data["label"]),
            prompt="",
            output="",
            reasoning=None,
        )


def evaluate_predictions(
    gt: list[int | None],
    preds: list[int | None],
    *,
    args: LLMArguments,
) -> None:
    """Evaluate model predictions and log results to wandb."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Save predictions and ground truth
    preds_path = os.path.join(args.output_dir, "preds.txt")
    gt_path = os.path.join(args.output_dir, "gt.txt")

    with open(preds_path, "wt") as f:
        f.write("\n".join(map(str, preds)))
    with open(gt_path, "wt") as f:
        f.write("\n".join(map(str, gt)))

    # Calculate metrics
    acc = sum([p == g for p, g in zip(preds, gt)]) / len(gt)
    precision = precision_score(gt, preds, average="weighted", zero_division=0)  # type: ignore
    recall = recall_score(gt, preds, average="weighted", zero_division=0)  # type: ignore
    f1 = f1_score(gt, preds, average="weighted", zero_division=0)  # type: ignore

    # Log metrics to wandb
    wandb.log(
        {
            "eval/accuracy": acc,
            "eval/precision": precision,
            "eval/recall": recall,
            "eval/f1": f1,
        }
    )

    # Print results
    print(f"Model: {args.model}")
    print(f"Data file: {args.data_file}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


def log_outputs(
    gt: list[int | None],
    preds: list[int | None],
    test_dataset: datasets.Dataset,
    prompt: list[str | None],
    llm_outputs: list[str | None],
    llm_reasoning: list[str | None],
) -> None:
    table = wandb.Table(
        columns=[
            "idx",
            "sentence",
            "prompt",
            "ground_truth",
            "prediction",
            "llm_output",
            "llm_reasoning",
        ]
    )
    for idx, data in enumerate(test_dataset):
        data = dict(data)
        table.add_data(
            idx,
            data["sentence"],
            prompt[idx],
            gt[idx],
            preds[idx],
            llm_outputs[idx],
            llm_reasoning[idx],
        )
    wandb.log({"predictions_table": table})


def main(args: LLMArguments) -> None:
    random.seed(args.seed)

    # Get wandb group and tags from environment variables
    wandb_group = os.environ.get("WANDB_GROUP")
    wandb_tags = os.environ.get("WANDB_TAGS")
    tags_list = wandb_tags.split(",") if wandb_tags else None

    wandb.init(config=asdict(args), group=wandb_group, tags=tags_list)

    client = OpenAI(base_url=args.base_url)

    dataset = datasets.load_dataset("csv", data_files=args.data_file)
    assert isinstance(dataset, DatasetDict)
    dataset = dataset["train"]
    test_dataset = dataset.filter(lambda example: example["split"] == "test")

    preds: list[int | None] = [None] * len(test_dataset)
    gt: list[int | None] = [None] * len(test_dataset)
    llm_outputs: list[str | None] = [None] * len(test_dataset)
    llm_reasoning: list[str | None] = [None] * len(test_dataset)
    prompt: list[str | None] = [None] * len(test_dataset)

    job_with_args = functools.partial(
        process_with_fallback,
        client=client,
        args=args,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        results = list(
            tqdm(
                executor.map(job_with_args, enumerate(test_dataset)),
                total=len(test_dataset),
            )
        )

    for result in results:
        idx = result.idx
        preds[idx] = result.prediction
        gt[idx] = result.label
        llm_outputs[idx] = result.output
        llm_reasoning[idx] = result.reasoning
        prompt[idx] = json.dumps(result.prompt, ensure_ascii=False)

    log_outputs(gt, preds, test_dataset, prompt, llm_outputs, llm_reasoning)
    evaluate_predictions(gt, preds, args=args)

    wandb.finish()


if __name__ == "__main__":
    main(parse_args())
