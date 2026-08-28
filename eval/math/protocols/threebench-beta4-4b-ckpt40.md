# VCU beta=4 unanswer ckpt40 math evaluation

Target model:

- Training family: `visual_counterfactual_unit`, the non-answer-weighted beta sweep implementation.
- Null input: mean color.
- Counterfactual extrapolation beta: `4.0`.
- Checkpoint: `global_step_40`.
- Merged model SHA-256: `0932e38fb2bc3d992f14e91ed4bf28d82aac4521869c166bc25182eb75d37278`.

Evaluation protocol:

- Benchmarks: MathVerse MINI (3,940 rows), WeMath testmini (1,740 rows), MathVista MINI/testmini (1,000 rows).
- Prompt and manifest files are byte-for-byte copies of the completed Qwen3.5-4B base evaluation.
- BF16 inference; temperature 0.7; top-p 0.8; top-k 20; presence penalty 1.5; repetition penalty 1.0; max tokens 4,096; pixels 3,584 to 401,408; seed 42; `enable_thinking=False`.
- MathVerse uses 8 shards, WeMath 4 shards, and MathVista 4 shards as 16 independent one-hour A100-80G array tasks; debug QoS allows up to two tasks to run concurrently.
- Scoring uses the official VLMEvalKit prompts with GPT-OSS-120B as judge.
- The one-row smoke ran on Cardinal H100 with Triton GDN prefill and `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`; all formal benchmark inference remains on Ascend A100-80G GPUs.

Completion gates:

- Smoke: exactly one row, no duplicate/missing/error rows, `finish_reason=stop`, zero truncation.
- Formal inference: exact expected row counts, zero duplicate/missing/error rows, and all shard completion markers.
- Final scoring: all three `accuracy.json` files plus `evaluation.complete`.
