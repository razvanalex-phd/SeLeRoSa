#!/bin/bash

export BASE_MODEL="OpenLLM-Ro/RoGemma2-9b-Instruct-DPO"
export NEW_MODEL="rogemma2_instruct_dpo_ft"
export MAX_TOKENS=1024
export TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export MERGE_SYSTEM_INTO_USER=1
export USE_CHAT_TEMPLATE=1

source "$(dirname "$0")/../generic/generic_llm_finetune.sh"
