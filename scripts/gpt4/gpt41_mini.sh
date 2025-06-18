#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa
export MODEL_NAME=gpt-4.1-mini-2025-04-14
export BASE_URL="https://api.openai.com/v1"
export LLM_BACKEND="openai"

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
