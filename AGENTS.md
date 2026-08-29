# Repository instructions

## Purpose

This repository is the canonical implementation of OPD-Aha. It contains training, inference, and
evaluation code for fine-grained visual perception and multimodal mathematical reasoning.

## Setup and tests

- Use Python 3.12 and install dependencies from `requirements.txt`.
- Run `python -m pytest -q tests/test_counterfactual_beta.py` after changing the training objective.
- Run `python eval/math/qwen3vl_wemath_strict.py selftest` after changing the WeMath parser.

## Repository constraints

- Never commit model weights, datasets, checkpoints, outputs, caches, logs, credentials, or tokens.
- Keep generated evaluation outputs under ignored paths.
- Preserve the default OPD-Aha training recipe unless a change is explicitly introduced as a new
  experiment.
- Keep training, inference, and evaluation commands portable; do not commit machine-specific paths.
