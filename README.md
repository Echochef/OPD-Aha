<p align="center">
  <img src="assets/opd-aha-header.svg" alt="OPD-Aha — Fine-grained visual perception" width="960">
</p>

<h3 align="center">Learning to See Fine Details for Multimodal LLMs<br>via On-Policy Self-Distillation</h3>

<p align="center">
  Official implementation of <b>OPD-Aha</b> for fine-grained visual perception<br>
  and multimodal mathematical reasoning.
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-0F766E?style=flat-square&amp;logo=terminal&amp;logoColor=white" alt="Quick Start"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-334155?style=flat-square" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp; / &nbsp;
  <a href="#training">Training</a> &nbsp; / &nbsp;
  <a href="#inference">Inference</a> &nbsp; / &nbsp;
  <a href="#evaluation">Evaluation</a>
</p>

---

## News

- **`2026-08-29`** Training, inference, and evaluation code released.

## Overview

OPD-Aha trains a multimodal language model with a frozen visual teacher and a counterfactual visual
input. The training objective emphasizes visual evidence that changes the teacher distribution while
preserving the standard on-policy learning workflow.

The repository includes:

- multi-node training built on `verl`;
- FSDP checkpoint merging and vLLM serving;
- fine-grained perception evaluation on V*Bench, HR-Bench, MME-RealWorld, and ZoomBench;
- mathematical reasoning evaluation on MathVista, MathVerse, WeMath, MathVision, and DynaMath.

## Repository Layout

| Path | Description |
| --- | --- |
| `verl/` | Distributed training and rollout implementation |
| `scripts/` | Data preparation, training, checkpoint merging, and serving entrypoints |
| `eval/` | Fine-grained perception inference and scoring |
| `eval/math/` | Mathematical reasoning inference and scoring |

## Quick Start

### 1. Environment

```bash
conda create -n opd-aha python=3.12 -y
conda activate opd-aha

pip install -r requirements.txt
pip install -e .
```

### 2. Training data

Prepare [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K):

```bash
python scripts/prepare_data.py --data-dir ./data
```

## Training

OPD-Aha uses Ray for multi-node training. Start the head and worker processes inside an existing
scheduler allocation:

```bash
# Head node
NUM_GPUS=4 bash scripts/start_ray_head.sh

# Worker node
RAY_HEAD_ADDRESS=<head-ip>:6379 NUM_GPUS=4 bash scripts/start_ray_worker.sh
```

Launch the 4B or 9B recipe from the head node:

```bash
bash scripts/train_opd_aha.sh 4b
bash scripts/train_opd_aha.sh 9b
```

Paths and cluster dimensions can be supplied through environment variables:

```bash
TASK_TRAIN_FILE=/path/to/train.parquet \
TRAINER_NNODES=2 \
TRAINER_N_GPUS_PER_NODE=4 \
  bash scripts/train_opd_aha.sh 4b
```

Merge an FSDP actor checkpoint after training:

```bash
BASE_DIR=/path/to/global_step_xx bash scripts/merge_checkpoint.sh
```

## Inference

Serve a Hugging Face or locally merged checkpoint with vLLM:

```bash
MODEL_PATH=/path/to/model \
SERVED_MODEL_NAME=opd-aha \
  bash scripts/serve_model.sh
```

## Evaluation

### Fine-grained perception

Start the model server and an OpenAI-compatible judge server, then run:

```bash
API_BASE=http://127.0.0.1:8000/v1 \
OPENAI_MODEL_ID=opd-aha \
JUDGE_API_BASE=http://127.0.0.1:8001/v1 \
JUDGE_MODEL=openai/gpt-oss-120b \
  bash eval/run_perception_eval.sh
```

Use `BENCHMARK` to select a subset of V*Bench, HR-Bench-4K, HR-Bench-8K,
MME-RealWorld-CN, MME-RealWorld, and ZoomBench.

### Mathematical reasoning

The mathematical reasoning suite provides data preparation, sharded inference, and scoring for
MathVerse, MathVista, WeMath, MathVision, and DynaMath. See
[`eval/math/README.md`](eval/math/README.md) for commands.

## Acknowledgements

OPD-Aha builds on excellent open-source projects and datasets, including
[`verl`](https://github.com/volcengine/verl),
[`Qwen`](https://github.com/QwenLM/Qwen3-VL),
[`vLLM`](https://github.com/vllm-project/vllm), and
[`Vision-OPD`](https://github.com/VisionOPD/Vision-OPD).

## License

This project is released under the Apache-2.0 License. Datasets and base models remain subject to
their respective licenses.
