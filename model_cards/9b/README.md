---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
pipeline_tag: image-text-to-text
tags:
- vision-language
- vision-opd
- on-policy-distillation
---

# Vision-OPD beta=4 Qwen3.5-9B checkpoint 30

This repository contains the merged checkpoint selected at global step 30 from the matched Vision-OPD beta=4 Qwen3.5-9B run. Relative to the released 4B contract, the intended method change is model scale only.

- Mean-color counterfactual input; beta 4.0.
- Frozen bbox-image teacher; update rate 0.0.
- Student top-100 plus tail support; alpha 0.5.
- 70 steps; save every ten; seed 42; global batch 96; rollout count 8.
- Two nodes with four H100 GPUs per node.

`model.safetensors` is 18,819,722,392 bytes with SHA-256 `7bed5430a32bdfb186c50275f5ddfa7925bc4b871e78f0a3406734e56e31dcc5`.

## Verified fine-grained perception results

Pure GPT-OSS: V* 94.76, HRBench-4K 88.25, HRBench-8K 85.25, MME-RealWorld-CN 72.55, MME-RealWorld 74.40, ZoomBench 60.95.

Rule-first plus GPT-OSS: V* 94.76, HRBench-4K 89.50, HRBench-8K 86.50, MME-RealWorld-CN 75.38, MME-RealWorld 77.90, ZoomBench 60.95.

No beta=4 9B math score is claimed. The released code can run the same MathVerse, MathVista, WeMath, MathVision, and DynaMath protocol on this checkpoint.

Training, serving, evaluation, exact hashes, and full metrics are in [Echochef/vision-opd-beta4-repro](https://github.com/Echochef/vision-opd-beta4-repro).

For H100 serving with vLLM 0.18.0, use Triton GDN prefill and set `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`.
