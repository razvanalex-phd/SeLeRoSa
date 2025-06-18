#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa
export MODEL_NAME="readerbench/RoGPT2-large"
export PORT=11434
export HOST=127.0.0.1
export BASE_URL="http://${HOST}:${PORT}/v1"
export DISABLE_BITSANDBYTES=1
export USE_COMPLETION=1

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
