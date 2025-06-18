#!/bin/bash

export BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NEW_MODEL="llama3_1_ft"
export MAX_TOKENS=1024
export TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export DEVICE_MAP="auto"
export USE_CHAT_TEMPLATE=1

source "$(dirname "$0")/../generic/generic_llm_finetune.sh"
