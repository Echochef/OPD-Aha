#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALE="${1:-4b}"
shift || true

case "${SCALE,,}" in
  4b)
    DEFAULT_MODEL="Qwen/Qwen3.5-4B"
    DEFAULT_EXPERIMENT="vcu_mean_color_frozen_beta4_step70"
    SELECTED_CHECKPOINT=40
    ;;
  9b)
    DEFAULT_MODEL="Qwen/Qwen3.5-9B"
    DEFAULT_EXPERIMENT="vcu_mean_color_frozen_beta4_qwen35_9b_step70"
    SELECTED_CHECKPOINT=30
    ;;
  *)
    echo "Usage: $0 {4b|9b} [Hydra override ...]" >&2
    exit 2
    ;;
esac

export MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
export EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME_OVERRIDE:-$DEFAULT_EXPERIMENT}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${PROJECT_ROOT}/data/train.parquet}"

# Exact contract used by both released beta=4 runs.
export TEACHER_MODEL_SOURCE="${TEACHER_MODEL_SOURCE:-legacy}"
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-frozen}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.0}"
export COUNTERFACTUAL_NULL_MODE="${COUNTERFACTUAL_NULL_MODE:-mean_color}"
export COUNTERFACTUAL_EXTRAPOLATION_BETA="${COUNTERFACTUAL_EXTRAPOLATION_BETA:-4.0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-96}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export ALPHA="${ALPHA:-0.5}"
export LR="${LR:-2e-6}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export DATA_SEED="${DATA_SEED:-42}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
export TRAINER_NNODES="${TRAINER_NNODES:-2}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-2}"
export TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-70}"

echo "Training scale: ${SCALE,,}"
echo "Base model: ${MODEL_PATH}"
echo "Run length: ${TRAINER_TOTAL_TRAINING_STEPS} steps; save every ${TRAINER_SAVE_FREQ} steps"
echo "Published selection after evaluation: global_step_${SELECTED_CHECKPOINT}"

exec "${PROJECT_ROOT}/scripts/run_visual_counterfactual_unit.sh" "$@"
