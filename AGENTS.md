# Repository instructions

## Purpose

This repository is the canonical editable reproducibility release for the Vision-OPD beta=4 Qwen3.5-4B checkpoint 40 and Qwen3.5-9B checkpoint 30 models. Historical OSU source and run directories are read-only evidence.

## Setup and verification

- Use Python 3.12 and `requirements.txt` for training and vLLM evaluation.
- Run `bash scripts/verify_release.sh` after code or metadata changes.
- Full training uses a Ray cluster with two nodes and four H100 GPUs per node unless a scale change is explicitly documented.

## Repository constraints

- Never commit model weights, datasets, checkpoints, outputs, caches, logs, credentials, or tokens.
- Keep generated evaluation outputs under ignored paths.
- Preserve the beta=4 matched contract unless a change is explicitly a new experiment.
- Do not rewrite `source_manifest.sha256`; it authenticates the pre-release OSU snapshot.
- Back model-card claims with `results/verified_metrics.json` and `provenance/models.json`.
