#!/bin/bash

export NUM_RUNS=4
export WANDB_PROJECT=SeLeRoSa
export MODEL_NAME=o4-mini-2025-04-16
export BASE_URL="https://api.openai.com/v1"
export LLM_BACKEND="openai"

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
