#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa
export MODEL_NAME="results/rollama3_1_instruct_dpo_ft"
export PORT=11434
export HOST=127.0.0.1
export BASE_URL="http://${HOST}:${PORT}/v1"
export FROM_FT_MODEL=1

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
