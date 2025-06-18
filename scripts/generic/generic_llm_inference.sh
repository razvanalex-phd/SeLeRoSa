#!/bin/bash

# Usage: source this script after exporting all required variables:
#   NUM_RUNS, WANDB_PROJECT, MODEL_NAME, PORT, HOST, BASE_URL
#
# This script supports multiple LLM backends controlled by LLM_BACKEND variable:
# - LLM_BACKEND=vllm (default if not set): Use vLLM server
# - LLM_BACKEND=ollama: Use Ollama server
# - LLM_BACKEND=openai: Use external OpenAI-compatible API
#
# Example in a per-LLM script:
#   export NUM_RUNS=1
#   export WANDB_PROJECT=SeLeRoSa
#   export MODEL_NAME=llama3.1:latest
#   export PORT=11434
#   export HOST=127.0.0.1
#   export BASE_URL="http://${HOST}:${PORT}/v1"
#   export LLM_BACKEND=ollama  # Optional, defaults to vllm
#   source ./generic_llm_inference.sh


wait_for_server() {
    local SERVER_TYPE="$1"
    local SERVER_PID="$2"

    sleep 5
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] $SERVER_TYPE server failed to start (process not running). Exiting."
        exit 1
    fi

    MAX_WAIT=600
    WAITED=0
    INTERVAL=5
    PROGRESS_INTERVAL=30
    while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
        sleep $INTERVAL
        WAITED=$((WAITED+INTERVAL))
        if [ $((WAITED % PROGRESS_INTERVAL)) -eq 0 ]; then
            echo "[INFO] Waiting for $SERVER_TYPE server to open port $PORT... ($WAITED seconds elapsed)"
        fi
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "[ERROR] $SERVER_TYPE server did not open port $PORT after $MAX_WAIT seconds. Exiting."
            kill "$SERVER_PID" 2>/dev/null
            exit 1
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ERROR] $SERVER_TYPE server process exited unexpectedly while waiting for port. Exiting."
            exit 1
        fi
    done
}

LLM_BACKEND=${LLM_BACKEND:-vllm}

if [ "$LLM_BACKEND" = "ollama" ]; then
    if command -v ollama >/dev/null 2>&1; then
        echo "Using Ollama backend. Starting Ollama server and serving model..."
        export OPENAI_API_KEY="x"

        ollama serve > ollama_server.log 2>&1 &
        OLLAMA_SERVE_PID=$!

        ollama pull "${MODEL_NAME}" > ollama_pull.log 2>&1

        wait_for_server "Ollama" "$OLLAMA_SERVE_PID"

        echo "Starting to serve model: ${MODEL_NAME}"
        ollama run "${MODEL_NAME}" --keepalive -1 > ollama_model.log 2>&1 &
        OLLAMA_MODEL_PID=$!

        echo "Waiting for model to load..."
        sleep 10

        LOCAL_LLM=1
        SERVER_TYPE="Ollama"
        SERVER_PID="$OLLAMA_SERVE_PID"
    else
        echo "[ERROR] Ollama backend requested but ollama command not found. Exiting."
        exit 1
    fi
elif [ "$LLM_BACKEND" = "vllm" ]; then
    if python -c "import vllm" 2>/dev/null; then
        echo "Using vLLM backend. Starting vLLM OpenAI server in background..."
        VLLM_CMD=(python -m vllm.entrypoints.openai.api_server \
            --model "${MODEL_NAME}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --max-model-len 1024 \
            --uvicorn-log-level error \
            --disable-uvicorn-access-log)
        if [ -n "${DTYPE}" ]; then
            VLLM_CMD+=(--dtype "${DTYPE}")
        fi
        if [ "${DISABLE_BITSANDBYTES}" != "1" ]; then
            VLLM_CMD+=(--quantization bitsandbytes)
        fi
        "${VLLM_CMD[@]}" > vllm_server.log 2>&1 &
        VLLM_PID=$!

        wait_for_server "vLLM" "$VLLM_PID"

        LOCAL_LLM=1
        SERVER_TYPE="vLLM"
        SERVER_PID="$VLLM_PID"
    else
        echo "[ERROR] vLLM backend requested but vLLM not installed. Exiting."
        exit 1
    fi
elif [ "$LLM_BACKEND" = "openai" ]; then
    echo "Using OpenAI-compatible external API backend."
    LOCAL_LLM=0
else
    echo "[ERROR] Unknown LLM_BACKEND: $LLM_BACKEND. Supported values: vllm, ollama, openai"
    exit 1
fi

LLM_NAME="${MODEL_NAME%%:*}"

export WANDB_GROUP="${LLM_NAME}"
export WANDB_TAGS="$LLM_NAME,inference,llm"

for i in $(seq 1 "$NUM_RUNS"); do
    OUTPUT_DIR="results/inference/${LLM_NAME}/${i}"
    mkdir -p "$OUTPUT_DIR"

    export WANDB_NAME="${MODEL_NAME}_inference_$i"
    CMD=(python satire/experiments/baselines/llm.py \
        --data_file data/csv/selerosa.csv \
        --model "${MODEL_NAME}" \
        --seed "$i" \
        --output-dir "${OUTPUT_DIR}")

    if [ -n "${TEMPERATURE}" ]; then
        CMD+=(--temperature "${TEMPERATURE}")
    else
        CMD+=(--temperature 0.8)
    fi
    if [ -n "${TOP_K}" ]; then
        CMD+=(--top-k "${TOP_K}")
    fi
    if [ -n "${TOP_P}" ]; then
        CMD+=(--top-p "${TOP_P}")
    fi
    if [ -n "${MIN_P}" ]; then
        CMD+=(--min-p "${MIN_P}")
    fi
    if [ "${FROM_FT_MODEL}" = "1" ]; then
        CMD+=(--from-ft-model)
    fi
    if [ -n "$BASE_URL" ]; then
        CMD+=(--base-url "${BASE_URL}")
    fi
    if [ "${USE_COMPLETION}" = "1" ]; then
        CMD+=(--use-completion)
    fi
    if [ "${MERGE_SYSTEM_INTO_USER}" = "1" ]; then
        CMD+=(--merge-system-into-user)
    fi
    "${CMD[@]}"
done

if [ "$LOCAL_LLM" = "1" ]; then
    echo "Killing $SERVER_TYPE server (PID $SERVER_PID)"
    kill "$SERVER_PID"

    if [ "$SERVER_TYPE" = "Ollama" ] && [ -n "$OLLAMA_MODEL_PID" ]; then
        echo "Killing Ollama model process (PID $OLLAMA_MODEL_PID)"
        kill "$OLLAMA_MODEL_PID" 2>/dev/null
    fi
fi
