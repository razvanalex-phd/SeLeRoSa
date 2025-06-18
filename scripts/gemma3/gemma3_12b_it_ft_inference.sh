#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa
export MODEL_NAME="results/gemma3_12b_it_ft"
export PORT=11434
export HOST=127.0.0.1
export BASE_URL="http://${HOST}:${PORT}/v1"
export DTYPE=bfloat16
export FROM_FT_MODEL=1
export TEMPERATURE=1.0
export TOP_K=64
export TOP_P=0.95
export MIN_P=0.0

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
