#!/bin/bash

export WANDB_PROJECT=SeLeRoSa
export BASE_MODEL="readerbench/RoGPT2-large"
export NEW_MODEL="ro_gpt2_large_ft"
export MAX_TOKENS=1024
export TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export DEVICE_MAP="auto"
export USE_LORA="false"
export LORA_R="0"
export LORA_ALPHA="0"
export LORA_DROPOUT="0.0"
export LORA_TARGET_MODULES=""
export LOAD_IN_4BIT="false"
export LOAD_IN_8BIT="false"
export ENABLE_LORA=0
export QUANTIZATION_TYPE="none"
export LORA_FAN_IN_FAN_OUT="true"
export TO_TOKENS=1
export USE_CHAT_TEMPLATE=1
export NO_MERGE_PEFT=1

source "$(dirname "$0")/../generic/generic_llm_finetune.sh"
