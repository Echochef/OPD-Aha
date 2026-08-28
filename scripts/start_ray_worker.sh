#!/usr/bin/env bash

set -euo pipefail

RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:?Set RAY_HEAD_ADDRESS to HEAD_IP:PORT}"
NUM_GPUS="${NUM_GPUS:-4}"

exec ray start \
  --address "$RAY_HEAD_ADDRESS" \
  --num-gpus "$NUM_GPUS" \
  --block
