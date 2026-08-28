#!/usr/bin/env bash

set -euo pipefail

HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
NUM_GPUS="${NUM_GPUS:-4}"

exec ray start --head \
  --node-ip-address "$HEAD_IP" \
  --port "$RAY_PORT" \
  --num-gpus "$NUM_GPUS" \
  --block
