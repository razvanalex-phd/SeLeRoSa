#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa

# NOTE: The huggingface model is painfully slow, probably it is an issue in
# scheduling on vllm. Use use ollama which is faster. There should be no
# performance difference between GGUF and the original safetensors.
#
# When using Ollama, you should convert the model first to GGUF format. Check
# the [README.md](./scripts/README.md) for instructions.
#
# Otherwise, use the original safetensors model:
#
export MODEL_NAME="results/gemma3_1b_it_merged"
# 
# And remove the `export LLM_BACKEND=ollama` line.
#
# export LLM_BACKEND=ollama
# export MODEL_NAME="gemma3_1b_ft"

export PORT=11434
export HOST=127.0.0.1
export BASE_URL="http://${HOST}:${PORT}/v1"
export DTYPE=half
export FROM_FT_MODEL=1
export DISABLE_BITSANDBYTES=1
export TEMPERATURE=1.0
export TOP_K=64
export TOP_P=0.95
export MIN_P=0.0

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
