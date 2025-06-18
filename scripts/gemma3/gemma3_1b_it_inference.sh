#!/bin/bash

export NUM_RUNS=5
export WANDB_PROJECT=SeLeRoSa

# NOTE: The huggingface model is painfully slow, probably it is an issue in
# scheduling on vllm. Use use ollama which is faster. There should be no
# performance difference between GGUF and the original safetensors.
#
# When using Ollama, you should download the model first:
#
#   ollama pull hf.co/unsloth/gemma-3-1b-it-GGUF:BF16
#
# On MacOS, test using the Q8_K_XL quantization
#
# export MODEL_NAME=google/gemma-3-1b-it
export MODEL_NAME=hf.co/unsloth/gemma-3-1b-it-GGUF:BF16
export LLM_BACKEND=ollama

export PORT=11434
export HOST=127.0.0.1
export BASE_URL="http://${HOST}:${PORT}/v1"
export DTYPE=bfloat16
export DISABLE_BITSANDBYTES=1
export TEMPERATURE=1.0
export TOP_K=64
export TOP_P=0.95
export MIN_P=0.0

source "$(dirname "$0")/../generic/generic_llm_inference.sh"
