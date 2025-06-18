import argparse
import gc
import os
import random
import shutil
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    PromptEncoderConfig,
    PromptTuningConfig,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTConfig, SFTTrainer, setup_chat_format

from satire.experiments.baselines.prompts import (
    ASSISTANT_PROMPT,
    CLASS_0,
    CLASS_1,
    SYSTEM_PROMPT_SIMPLE,
    USER_PROMPT_CHAT,
    build_training_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_select_size", type=int, default=-1)
    parser.add_argument("--seed", type=int)

    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        default=False,
        help="Use chat template for prompt construction (set to False for models like GPT2)",
    )

    parser.add_argument(
        "--to_tokens",
        action="store_true",
        default=False,
        help="Tokenize prompts in build_prompts (set to True for tokenized input, False for plain text). Default: False. GPT2 fine-tuning should use True.",
    )

    parser.add_argument("--classes", type=lambda arg: arg.split(","))
    parser.add_argument("--label_field", type=str)
    parser.add_argument("--merge_system_into_user", action="store_true", default=False)

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--new_model", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")

    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bnb_8bit_quant_type", type=str, default="int8")
    parser.add_argument("--bnb_8bit_use_double_quant", action="store_true")
    parser.add_argument("--bnb_8bit_compute_dtype", type=str, default="bfloat16")

    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument(
        "--r", type=int, default=16, help="Dimension of low-rank matrices"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="Scaling factor for weight matrices",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--lora_fan_in_fan_out",
        action="store_true",
        default=False,
        help="""Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.""",
    )
    parser.add_argument("--bias", type=str, default="none", help="Bias setting")
    parser.add_argument(
        "--modules_to_save",
        nargs="+",
        default=["embed_tokens", "lm_head"],
        help="Modules to save",
    )
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM", help="Task type")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
        help="Target modules",
    )

    parser.add_argument("--prompt_tuning", action="store_true", default=False)
    parser.add_argument("--p_tuning", action="store_true", default=False)
    parser.add_argument("--p_tuning_num_virtual_tokens", type=int, default=20)
    parser.add_argument("--p_tuning_reparameterization_type", type=str, default="MLP")
    parser.add_argument("--p_tuning_hidden_size", type=int, default=256)
    parser.add_argument("--p_tuning_num_layers", type=int, default=2)
    parser.add_argument("--p_tuning_dropout", type=float, default=0.0)

    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=10,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy",
    )
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument(
        "--optim", type=str, default="paged_adamw_8bit", help="Optimizer"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="LR scheduler type",
    )
    parser.add_argument("--warmup_steps", type=float, default=0.1, help="Warmup steps")

    parser.add_argument(
        "--dataset_text_field",
        type=str,
        required=True,
        help="Name of the text field in the dataset",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--save-to-huggingface",
        action="store_true",
        default=False,
    )

    parser.add_argument("--no_merge_peft", action="store_true", default=False)
    parser.add_argument(
        "--merge_device",
        type=str,
        default="cpu",
        help="Device on which to merge lora params",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="The device map on which to load the model. Set to None when using"
        " accelerate.  Otherwise, 'auto' shoud suffice",
    )

    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        default=False,
        help="Enable checkpointing",
    )
    parser.add_argument(
        "--checkpoint_save_steps",
        type=int,
        default=100,
        help="Frequency of saving checkpoints (in steps)",
    )
    parser.add_argument(
        "--restore_from_checkpoint",
        type=str,
        default=None,
        help="Path to restore training from a checkpoint",
    )

    args = parser.parse_args()

    if args.prompt_tuning and len(args.system_path) > 0:
        print(
            "[WARN] system_path should be empty when prompt tuning is "
            "enabled, since the embeddings from the prompt tuning are "
            "meant to replace the system prompt"
        )

    return args


def setup_quantization(args: argparse.Namespace) -> BitsAndBytesConfig | None:
    quantization_config = None

    if args.load_in_4bit:
        print("Loading model in 4-bit")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        )
    elif args.load_in_8bit:
        print("Loading model in 8-bit")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            bnb_8bit_quant_type=args.bnb_8bit_quant_type,
            bnb_8bit_use_double_quant=args.bnb_8bit_use_double_quant,
            bnb_8bit_compute_dtype=args.bnb_8bit_compute_dtype,
        )
    return quantization_config


def setup_peft(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizer | None = None,
    model: PreTrainedModel | None = None,
) -> PeftConfig | None:
    peft_config = None

    if args.lora:
        print("Using LoRA")
        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.bias,
            modules_to_save=args.modules_to_save,
            task_type=args.task_type,
            target_modules=args.target_modules,
            fan_in_fan_out=args.lora_fan_in_fan_out,
        )
    elif args.prompt_tuning:
        print("Using prompt tuning")
        assert tokenizer is not None
        assert model is not None

        prompt = SYSTEM_PROMPT_SIMPLE.format(
            class0=args.classes[0] or CLASS_0,
            class1=args.classes[1] or CLASS_1,
        )
        input_ids = tokenizer.encode(prompt)

        peft_config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type=args.task_type,
            num_virtual_tokens=len(input_ids),
            token_dim=model.config.hidden_size,  # type: ignore
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=prompt,
            tokenizer_name_or_path=args.base_model,
            tokenizer_kwargs={
                "add_eos_token": True,
                "use_fast": True,
            },
        )
        print(
            f"Initializing prompt tuning with {len(input_ids)} virtual tokens "
            f"of size {model.config.hidden_size}"  # type: ignore
        )
    elif args.p_tuning:
        print("Using P-Tuning")
        assert model is not None

        peft_config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=args.task_type,
            num_virtual_tokens=args.p_tuning_num_virtual_tokens,
            token_dim=model.config.hidden_size,  # type: ignore
            encoder_reparameterization_type=args.p_tuning_reparameterization_type,
            encoder_hidden_size=args.p_tuning_hidden_size,
            encoder_dropout=args.p_tuning_dropout,
            encoder_num_layers=args.p_tuning_num_layers,
        )

    return peft_config


def load_dataset_from_args(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    dataset = load_dataset("csv", data_files=args.dataset)
    assert isinstance(dataset, DatasetDict)
    dataset = dataset["train"]

    dataset_train = dataset.filter(lambda example: example["split"] == "train")
    dataset_val = dataset.filter(lambda example: example["split"] == "validation")

    if args.train_select_size > 0:
        print(f"Selecting {args.train_select_size} samples from the training set")
        dataset_train = dataset_train.select(
            np.random.choice(
                len(dataset_train),
                size=(args.train_select_size,),
                replace=False,
            )
        )

    if (args.classes is not None) and (args.label_field is not None):
        dataset_train = dataset_train.map(
            lambda x: {
                "labels": args.classes[int(x[args.label_field])],
                **x,
            }
        )
        dataset_val = dataset_val.map(
            lambda x: {
                "labels": args.classes[int(x[args.label_field])],
                **x,
            }
        )

    return dataset_train, dataset_val


def build_prompts(
    dataset: Dataset,
    merge_system_into_user: bool,
    use_chat_template: bool = True,
) -> Dataset:
    def to_dataset_format(example: dict[str, Any]) -> dict[str, Any]:
        return build_training_prompt(
            example,
            system=SYSTEM_PROMPT_SIMPLE,
            user=USER_PROMPT_CHAT,
            assistant=ASSISTANT_PROMPT,
            use_chat_template=use_chat_template,
            merge_system_into_user=merge_system_into_user,
        )

    dataset = dataset.map(to_dataset_format)

    # Check if labels are correct
    for data in dataset:
        expected = ASSISTANT_PROMPT.format(answer=str(data["labels"]))  # type: ignore
        if use_chat_template:
            got = data["messages"][-1]["content"]  # type: ignore
        else:
            got = data["completion"]  # type: ignore
        assert expected == got, f"Expected: {expected}, got: {got}."

    if use_chat_template:
        dataset = dataset.select_columns(["messages"])
    else:
        dataset = dataset.select_columns(["prompt", "completion"])

    return dataset


def requires_eager_attention(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    eager_attention_patterns = [
        "gemma-2",
        "gemma-3",
        "rogemma2",
    ]
    return any(pattern in model_name_lower for pattern in eager_attention_patterns)


def fine_tune(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load dataset
    dataset_train, dataset_val = load_dataset_from_args(args)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, add_eos_token=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_train = build_prompts(
        dataset_train,
        args.merge_system_into_user,
        use_chat_template=args.use_chat_template,
    )
    dataset_val = build_prompts(
        dataset_val,
        args.merge_system_into_user,
        use_chat_template=args.use_chat_template,
    )

    print(dataset_train[0])
    print()
    print(dataset_val[0])
    print()

    # Quantization
    quantization_config = setup_quantization(args)

    # Load base model
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": args.device_map,
    }
    if requires_eager_attention(args.base_model):
        print("Using eager attention for models that require it")
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    model.generation_config.do_sample = True
    model.config.use_cache = False

    if args.to_tokens:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Cast the layernorm in fp32, make output embedding layer require
    # grads, add the upcasting of the lmhead to fp32
    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    # PEFT configuration
    peft_config = setup_peft(args, tokenizer=tokenizer, model=model)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        run_name=os.environ.get("WANDB_NAME"),
        completion_only_loss=False,
        label_names=["labels"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=args.checkpoint_save_steps if args.enable_checkpointing else 500,
        save_total_limit=1,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train model
    try:
        if args.restore_from_checkpoint:
            trainer.train(resume_from_checkpoint=args.restore_from_checkpoint)
        else:
            trainer.train()
    finally:
        trainer.save_model(os.path.join(args.save_path, args.new_model))


def merge_peft_model(args: argparse.Namespace) -> None:
    model_save_path = os.path.join(args.save_path, args.new_model)

    # Load tokenizer from the fine-tuned model directory (with added tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.save_path, args.new_model),
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

    # Load base model and resize embeddings to match tokenizer
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "return_dict": True,
        "torch_dtype": torch.float16,
        "device_map": args.merge_device,
    }
    if requires_eager_attention(args.base_model):
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    model.generation_config.do_sample = True

    # Apply the same chat format setup as during fine-tuning if needed
    if args.to_tokens:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Save the  model to a temporary directory
    tmp_model_dir = os.path.join(args.save_path, "tmp_resized_base")
    model.save_pretrained(tmp_model_dir)

    model_kwargs_tmp = {
        "low_cpu_mem_usage": True,
        "return_dict": True,
        "torch_dtype": torch.float16,
        "device_map": args.merge_device,
    }
    if requires_eager_attention(args.base_model):
        model_kwargs_tmp["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        tmp_model_dir,
        **model_kwargs_tmp,
    )
    try:
        model = PeftModel.from_pretrained(
            model,
            model_save_path,
            device_map=args.merge_device,
        )
        model = model.merge_and_unload()  # type: ignore
    except:
        print("Could not merge peft model. Will save without.")

    # Save trained model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # --- PATCH: Write minimal preprocessing_config.json for Gemma 3 models (except 1B) ---
    def is_gemma3_model(model_name):
        name = model_name.lower()
        return name.startswith("google/gemma-3-")

    def gemma3_model_size(model_name):
        # Returns size as string, e.g. "1b", "4b", "12b", "27b"
        name = model_name.lower()
        if name.startswith("google/gemma-3-"):
            return name.split("-")[-2]  # e.g. "1b", "4b", etc.
        return None

    if is_gemma3_model(args.base_model):
        size = gemma3_model_size(args.base_model)
        if size and size != "1b":
            # Download preprocessor_config.json using huggingface_hub (requires login)
            config_path = os.path.join(model_save_path, "preprocessor_config.json")
            if not os.path.exists(config_path):
                try:
                    from huggingface_hub import hf_hub_download

                    repo_id = f"google/gemma-3-{size}-it"
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id, filename="preprocessor_config.json"
                    )
                    shutil.copy(downloaded_path, config_path)
                    print(
                        f"[INFO] Downloaded preprocessor_config.json from Hugging Face Hub to {config_path}"
                    )
                except Exception as e:
                    print(
                        f"[WARN] Could not download preprocessor_config.json from Hugging Face Hub: {e}\nWriting minimal file instead."
                    )
                    with open(config_path, "w") as f:
                        f.write("{}\n")
                    print(
                        f"[INFO] Wrote minimal preprocessor_config.json to {config_path}"
                    )

    print(f"Saved to {model_save_path}")

    shutil.rmtree(tmp_model_dir, ignore_errors=True)

    if args.save_to_huggingface:
        try:
            model.push_to_hub(repo_id=args.new_model, private=True)
            tokenizer.push_to_hub(repo_id=args.new_model, private=True)
        except KeyboardInterrupt:
            pass


def main(args: argparse.Namespace) -> None:
    try:
        fine_tune(args)
    except KeyboardInterrupt:
        pass

    # Empty VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        while gc.collect():
            pass
        torch.cuda.empty_cache()

    if not args.no_merge_peft:
        for _ in range(5):
            try:
                merge_peft_model(args)
                break
            except Exception as e:
                if "out of memory" in str(e):
                    print("Trying merge device cpu")
                    args.merge_device = "cpu"
                    continue
                raise


if __name__ == "__main__":
    main(parse_args())
