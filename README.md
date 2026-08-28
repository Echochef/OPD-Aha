# Vision-OPD beta=4 reproducibility release

This repository contains the training, inference, and evaluation code for two selected Vision-OPD beta=4 checkpoints.

| Model | Selected checkpoint | Hugging Face |
|---|---:|---|
| Qwen3.5-4B | 40 | [Vision-OPD-Beta4-4B-ckpt40](https://huggingface.co/Echo23333456/Vision-OPD-Beta4-4B-ckpt40) |
| Qwen3.5-9B | 30 | [Vision-OPD-Beta4-9B-ckpt30](https://huggingface.co/Echo23333456/Vision-OPD-Beta4-9B-ckpt30) |

Both model repositories and the GitHub repository are private at release time.

The release is derived from the exact OSU training snapshot. All 484 entries in [`source_manifest.sha256`](source_manifest.sha256) passed SHA-256 verification before portable release edits were applied. Release changes are tracked by Git.

Included here:

- beta=4 on-policy self-distillation training for 4B and 9B;
- FSDP checkpoint merge and vLLM serving code;
- fine-grained perception preparation, inference, and three scoring paths;
- MathVerse, MathVista, WeMath, MathVision, and DynaMath evaluation;
- verified model hashes, training contracts, protocols, and completed scores.

See [`provenance/models.json`](provenance/models.json) and [`results/verified_metrics.json`](results/verified_metrics.json). The 9B checkpoint has verified perception results; no beta=4 9B math score is claimed. The same math pipeline can evaluate either released model.

## Environment

The exact recorded environment used Python 3.12, PyTorch 2.10.0, Transformers 5.5.0, vLLM 0.18.0, Ray 2.53.0, and FlashInfer 0.6.6.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The original training used two nodes with four H100 GPUs per node. Within a scheduler allocation, start a Ray head on the first node and a worker on the second:

```bash
NUM_GPUS=4 bash scripts/start_ray_head.sh
RAY_HEAD_ADDRESS=<head-ip>:6379 NUM_GPUS=4 bash scripts/start_ray_worker.sh
```

Run training on the head after all eight GPUs appear in `ray status`.

## Training data

Prepare Vision-OPD-6K from `yuanqianhao/Vision-OPD-6K`:

```bash
python scripts/prepare_data.py --data-dir ./data
```

The historical OSU `train.parquet` SHA-256 is `2bb335bca8a3989a4abda0ed93884e830e4420917e2c08fdd3aa1182ac41ddcd`. Its rows contain absolute image paths, so a parquet regenerated elsewhere will have a different byte hash even with the same samples.

## Reproduce training

Both releases use beta 4.0, mean-color counterfactual input, a frozen bbox-image teacher, student top-100 plus tail support, alpha 0.5, seed 42, global batch 96, eight rollouts, 70 steps, and checkpoints every ten steps.

```bash
# Qwen3.5-4B; select global_step_40 after evaluation
bash scripts/train_beta4.sh 4b

# Qwen3.5-9B; select global_step_30 after evaluation
bash scripts/train_beta4.sh 9b
```

Paths and cluster dimensions are configurable:

```bash
TASK_TRAIN_FILE=/data/vision-opd/train.parquet \
TRAINER_NNODES=2 TRAINER_N_GPUS_PER_NODE=4 \
bash scripts/train_beta4.sh 4b
```

Merge a saved FSDP actor checkpoint:

```bash
BASE_DIR=checkpoints/vcu_mean_color_frozen_beta4_step70/global_step_40 \
bash scripts/merge_checkpoint.sh
```

## Serve a checkpoint

```bash
MODEL_PATH=Echo23333456/Vision-OPD-Beta4-4B-ckpt40 \
SERVED_MODEL_NAME=vision-opd-beta4-4b \
bash scripts/serve_model.sh
```

The wrapper includes the completed H100 evaluation settings: Triton GDN prefill and `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`.

## Fine-grained perception evaluation

Start the released model server and a GPT-OSS-120B OpenAI-compatible judge server, then run:

```bash
API_BASE=http://127.0.0.1:8000/v1 \
OPENAI_MODEL_ID=vision-opd-beta4-4b \
JUDGE_API_BASE=http://127.0.0.1:8001/v1 \
JUDGE_MODEL=openai/gpt-oss-120b \
bash eval/run_perception_repro.sh
```

The default suite is V*Bench, HRBench-4K, HRBench-8K, MME-RealWorld-CN, MME-RealWorld, and ZoomBench with seed 42, maximum response length 8192, and thinking disabled. Pure-LLM, rule-first, and native-exact judge outputs are separated. Set `BENCHMARK` to run a subset.

## Math reasoning evaluation

See [`eval/math/README.md`](eval/math/README.md) for the exact 4096-token protocol and commands for MathVerse MINI, MathVista MINI, WeMath testmini, MathVision testmini, and DynaMath test.

## Verification

```bash
bash scripts/verify_release.sh
```

This checks shell and Python syntax, beta method tests, release metadata, and accidental secret inclusion. Full GPU training and benchmark inference remain separate long-running checks.

## License

Apache-2.0. Dataset and base-model licenses remain governed by their upstream repositories.
