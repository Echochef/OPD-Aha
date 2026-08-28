#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local checkpoint or Hugging Face repo ID}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

# Required for stable Qwen3.5 GDN inference with vLLM 0.18.0 on H100.
export VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE="${VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image": 1}' \
  --gdn-prefill-backend triton \
  --trust-remote-code \
  --no-enable-log-requests \
  "$@"
