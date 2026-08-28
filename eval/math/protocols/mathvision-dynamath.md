# Qwen3.5-4B MathVision and DynaMath evaluation

- Models: Qwen3.5-4B base, beta=4 checkpoint 40, and standard Vision-OPD checkpoint 40.
- Generation: BF16, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5,
  repetition penalty 1.0, seed 42, max tokens 4096, and `enable_thinking=False`.
- Images: one image per sample, min pixels 3584 and max pixels 401408.
- MathVision: `MathLLMs/MathVision`, `testmini`, 304 rows, lmms-eval
  `mathvision_testmini` prompt and `mathvision_standard_eval` rule scoring.
- DynaMath: `kcz358/DynaMath` revision
  `c72be604052511aa6f2e94164139f8a2e801bd33`, `test`, 5010 rows, 501 groups
  with 10 variants each, lmms-eval `dynamath_reasoning` prompt, and official
  `average`/`worst` rule scoring with `USE_LLM_JUDGE=False`.
- This run intentionally overrides the benchmark configs' larger generation caps with the
  user-requested 4096-token cap.
