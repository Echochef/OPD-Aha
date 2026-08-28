# Math reasoning evaluation

This directory reproduces the completed Qwen3.5-4B beta=4 checkpoint-40 math evaluation and can run the same protocol on the 9B checkpoint.

## Shared protocol

- BF16; temperature 0.7; top-p 0.8; top-k 20.
- Presence penalty 1.5; repetition penalty 1.0; seed 42.
- Maximum 4096 generated tokens; thinking disabled.
- One image per sample, 3,584 to 401,408 pixels.

Serve a model with `scripts/serve_model.sh`. The commands below assume:

```bash
MODEL_API=http://127.0.0.1:8000/v1
MODEL_ID=vision-opd-beta4-4b
JUDGE_API=http://127.0.0.1:8001/v1
WORK=/path/to/evaluation-work
```

## MathVerse MINI

Pass `--per-version 0` to prepare all 3,940 rows. The completed run used eight shards.

```bash
python eval/math/qwen3vl_mathverse_termination.py prepare \
  --mathverse-root /datasets/MathVerse \
  --output-root "$WORK/mathverse" --per-version 0 --seed 42

for shard in 0 1 2 3 4 5 6 7; do
  python eval/math/qwen3vl_mathverse_termination.py infer \
    --output-root "$WORK/mathverse" --api-base "$MODEL_API" --model-id "$MODEL_ID" \
    --shard-id "$shard" --num-shards 8 --seed 42
done

python eval/math/qwen3vl_mathverse_termination.py summarize --output-root "$WORK/mathverse"
python eval/math/judge_mathverse_gptoss.py \
  --inference-root "$WORK/mathverse" \
  --mathverse-parquet /datasets/MathVerse/testmini.parquet \
  --output-root "$WORK/mathverse/judged" --api-base "$JUDGE_API" \
  --judge-model openai/gpt-oss-120b
```

## MathVista MINI

Use the 1,000-row `AI4Math/MathVista` testmini parquet and four shards.

```bash
python eval/math/qwen35_mathvista.py prepare \
  --mathvista-parquet /datasets/MathVista/testmini.parquet \
  --output-root "$WORK/mathvista"

for shard in 0 1 2 3; do
  python eval/math/qwen35_mathvista.py infer \
    --output-root "$WORK/mathvista" --api-base "$MODEL_API" --model-id "$MODEL_ID" \
    --shard-id "$shard" --num-shards 4 --seed 42
done

python eval/math/qwen35_mathvista.py summarize --output-root "$WORK/mathvista"
python eval/math/judge_mathvista_gptoss.py \
  --inference-root "$WORK/mathvista" --output-root "$WORK/mathvista/judged" \
  --api-base "$JUDGE_API" --judge-model openai/gpt-oss-120b
```

## WeMath testmini

Use `We-Math/We-Math` revision `527c44d4d94c4e3c7c98157460146a8b18c8420a`, 1,740 rows, and four shards.

```bash
python eval/math/qwen3vl_wemath_strict.py prepare \
  --wemath-parquet /datasets/WeMath/testmini.parquet \
  --output-root "$WORK/wemath"

for shard in 0 1 2 3; do
  python eval/math/qwen3vl_wemath_strict.py infer \
    --output-root "$WORK/wemath" --api-base "$MODEL_API" --model-id "$MODEL_ID" \
    --shard-id "$shard" --num-shards 4 --seed 42
done

python eval/math/qwen3vl_wemath_strict.py summarize --output-root "$WORK/wemath"
python eval/math/judge_wemath_gptoss.py \
  --inference-root "$WORK/wemath" --output-root "$WORK/wemath/judged" \
  --api-base "$JUDGE_API" --judge-model openai/gpt-oss-120b
```

The three benchmarks above use official VLMEvalKit prompts with GPT-OSS-120B as the substituted judge. Completion requires exact row counts, unique sample IDs, zero inference errors, all shards, and final accuracy files.

## MathVision and DynaMath

MathVision uses `MathLLMs/MathVision` testmini (304 rows). DynaMath uses `kcz358/DynaMath` revision `c72be604052511aa6f2e94164139f8a2e801bd33`, test split (5,010 rows; 501 groups with ten variants).

```bash
python eval/math/mathvision_dynamath/prepare_benchmarks.py \
  --output-root "$WORK/mvdm" --models beta4_4b \
  --mathvision-parquet /datasets/MathVision/testmini.parquet \
  --dynamath-parquet /datasets/DynaMath/test.parquet

python eval/math/mathvision_dynamath/infer_benchmarks.py \
  --output-root "$WORK/mvdm/beta4_4b/mathvision" \
  --api-base "$MODEL_API" --model-id "$MODEL_ID" --max-tokens 4096

for shard in 0 1 2 3 4 5 6 7; do
  python eval/math/mathvision_dynamath/infer_benchmarks.py \
    --output-root "$WORK/mvdm/beta4_4b/dynamath" \
    --api-base "$MODEL_API" --model-id "$MODEL_ID" --max-tokens 4096 \
    --shard-id "$shard" --num-shards 8
done

LMMS_EVAL_ROOT=/path/to/lmms-eval \
python eval/math/mathvision_dynamath/score_official.py \
  --result-root "$WORK/mvdm" --models beta4_4b --allowed-max-tokens 4096
```

MathVision uses `mathvision_standard_eval`. DynaMath uses official average and worst-group rule scoring with `USE_LLM_JUDGE=False`. Exact source protocols are preserved in [`protocols/`](protocols/).
