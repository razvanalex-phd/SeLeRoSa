#!/bin/bash

export BASE_MODEL="google/gemma-3-4b-it"
export NEW_MODEL="gemma3_4b_it_ft"
export MAX_TOKENS=1024
export TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=32
export GRADIENT_ACCUMULATION_STEPS=1
export USE_CHAT_TEMPLATE=1

source "$(dirname "$0")/../generic/generic_llm_finetune.sh"
