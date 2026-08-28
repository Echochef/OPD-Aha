---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
pipeline_tag: image-text-to-text
tags:
- vision-language
- vision-opd
- on-policy-distillation
---

# Vision-OPD beta=4 Qwen3.5-4B checkpoint 40

This repository contains the merged checkpoint selected at global step 40 from the Vision-OPD beta=4 Qwen3.5-4B run.

- Mean-color counterfactual input; beta 4.0.
- Frozen bbox-image teacher; update rate 0.0.
- Student top-100 plus tail support; alpha 0.5.
- 70 steps; save every ten; seed 42; global batch 96; rollout count 8.
- Two nodes with four H100 GPUs per node.

`model.safetensors` is 10,350,019,328 bytes with SHA-256 `0932e38fb2bc3d992f14e91ed4bf28d82aac4521869c166bc25182eb75d37278`.

## Verified results

Pure GPT-OSS scoring: V* 93.19, HRBench-4K 86.88, HRBench-8K 83.25, MME-RealWorld-CN 71.62, MME-RealWorld 73.17, ZoomBench 62.49.

Primary 4096-token math protocol: MathVista 80.00, MathVerse 75.66, WeMath row 84.66 / strict 69.43 / loose 81.62, MathVision 42.43, DynaMath average 70.64 / worst 41.72.

Training, serving, evaluation, exact hashes, and full metrics are in [Echochef/vision-opd-beta4-repro](https://github.com/Echochef/vision-opd-beta4-repro).

For H100 serving with vLLM 0.18.0, use Triton GDN prefill and set `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`.
