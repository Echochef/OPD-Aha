#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCHMARK="${BENCHMARK:-vstar,hrbench-4k,hrbench-8k,mme-realworld-cn,mme-realworld,zoombench}"
export SEED="${SEED:-42}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export ENABLE_THINKING="${ENABLE_THINKING:-False}"

for protocol in pure-llm rule-first native-exact; do
  echo "Scoring protocol: $protocol"
  JUDGE_PROTOCOL="$protocol" \
  JUDGE_OUT_DIR="${JUDGE_OUT_DIR_BASE:-judge}/$protocol" \
    "${SCRIPT_DIR}/run_eval.sh"
done
