#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa
export MODEL_NAME=OpenLLM-Ro/RoMistral-7b-Instruct-DPO
export PORT=11434
export HOST=127.0.0.1
export BASE_URL="http://${HOST}:${PORT}/v1"

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
