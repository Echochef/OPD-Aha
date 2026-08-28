#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts eval -type f -name '*.sh' -print | sort)

python3 -m compileall -q scripts eval tests verl
if python3 -c 'import torch, pytest' 2>/dev/null; then
  python3 -m pytest -q tests/test_counterfactual_beta.py
elif python3 -c 'import torch' 2>/dev/null; then
  python3 -c 'import runpy; ns = runpy.run_path("tests/test_counterfactual_beta.py"); ns["test_beta_two_target_matches_two_residual_units"]()'
else
  echo "Torch is unavailable; skipping the tensor beta test in this lightweight environment."
fi
python3 -m json.tool provenance/models.json >/dev/null
python3 -m json.tool results/verified_metrics.json >/dev/null

if rg -n --hidden \
  --glob '!.git/**' --glob '!scripts/verify_release.sh' \
  '(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})' .; then
  echo "Potential access token found in release tree" >&2
  exit 1
fi

test "$(git rev-parse --show-toplevel)" = "$PROJECT_ROOT"
test -f README.md
test -f .gitignore
test -f AGENTS.md

echo "Release verification passed."
