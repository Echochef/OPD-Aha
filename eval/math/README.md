# Mathematical reasoning evaluation

This directory contains data preparation, sharded inference, and scoring code for MathVerse,
MathVista, WeMath, MathVision, and DynaMath.

## Configuration

Serve the model with `scripts/serve_model.sh` and configure the endpoints:

```bash
export MODEL_API=http://127.0.0.1:8000/v1
export MODEL_ID=opd-aha
export JUDGE_API=http://127.0.0.1:8001/v1
export WORK=/path/to/evaluation-work
```

The default generation configuration uses BF16, temperature 0.7, top-p 0.8, top-k 20, seed 42,
4,096 generated tokens, and disabled thinking.

## MathVerse MINI

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

## MathVision and DynaMath

```bash
python eval/math/mathvision_dynamath/prepare_benchmarks.py \
  --output-root "$WORK/mvdm" --models opd_aha \
  --mathvision-parquet /datasets/MathVision/testmini.parquet \
  --dynamath-parquet /datasets/DynaMath/test.parquet

python eval/math/mathvision_dynamath/infer_benchmarks.py \
  --output-root "$WORK/mvdm/opd_aha/mathvision" \
  --api-base "$MODEL_API" --model-id "$MODEL_ID" --max-tokens 4096

for shard in 0 1 2 3 4 5 6 7; do
  python eval/math/mathvision_dynamath/infer_benchmarks.py \
    --output-root "$WORK/mvdm/opd_aha/dynamath" \
    --api-base "$MODEL_API" --model-id "$MODEL_ID" --max-tokens 4096 \
    --shard-id "$shard" --num-shards 8
done

LMMS_EVAL_ROOT=/path/to/lmms-eval \
python eval/math/mathvision_dynamath/score_official.py \
  --result-root "$WORK/mvdm" --models opd_aha --allowed-max-tokens 4096
```

MathVision uses the official `mathvision_standard_eval` scorer. DynaMath uses the official average
and worst-group rule scoring with `USE_LLM_JUDGE=False`.
