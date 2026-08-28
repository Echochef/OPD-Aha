#!/usr/bin/env python3
"""Score complete MathVerse outputs with the official two-stage prompts and GPT-OSS."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def extract_examples() -> list[str]:
    return [
        """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""",
        """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""",
        """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""",
        """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""",
        """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38°)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""",
        """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""",
    ]


def score_examples() -> list[str]:
    return [
        """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""",
        """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2√{{3}}\nC:√{{3}}\nD:2√{{2}}
[Standard Answer]: C
[Model_answer] : B:2√{{3}}
Judgement: 0
""",
        """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""",
        """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2√{{3}}\nC:√{{3}}\nD:2√{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""",
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def compact_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def extraction_prompt(prediction: str) -> str:
    task = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
"""
    prompt = task
    for example in extract_examples():
        prompt += example + "\n\n"
    return f"{prompt}7.\nModel response: '{prediction}'\nExtracted Answer: "


def extract_mcq_option(prediction: str) -> str | None:
    patterns = [
        re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", flags=re.IGNORECASE),
        re.compile(
            r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**\(?([A-D])\)?",
            flags=re.IGNORECASE,
        ),
        re.compile(r"\b([A-D])\b", flags=re.IGNORECASE),
    ]
    for pattern in patterns:
        matches = pattern.findall(prediction or "")
        if matches:
            return matches[-1].upper()
    return None


def clean_freeform_extract(raw: str) -> str:
    value = raw.strip()
    demo_tail = "f(x) = -x^2 - 2x + 1"
    if demo_tail in value:
        value = value.rsplit(demo_tail, 1)[1].strip()
    value = re.sub(r"^Extracted\s+Answer\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip().strip("`").strip()


def extract_last_boxed_answer(prediction: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", prediction or ""))
    for match in reversed(starts):
        depth = 1
        begin = match.end()
        for index in range(begin, len(prediction)):
            if prediction[index] == "{":
                depth += 1
            elif prediction[index] == "}":
                depth -= 1
                if depth == 0:
                    value = prediction[begin:index].strip()
                    return value or None
    return None


def score_prompt(question: str, answer: str, extracted: str) -> str:
    task = """
Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
"""
    prompt = task
    for example in score_examples():
        prompt += example + "\n\n"
    test = f"""
    [Question]: {question}
    [Standard Answer]: {answer}
    [Model_answer] : {extracted}
    Judgement:"""
    return f"{prompt}{test}"


def parse_score(text: str) -> bool | None:
    value = text.strip()
    if value in {"0", "1"}:
        return value == "1"
    labelled = re.findall(r"Judge(?:ment)?\s*:\s*\**\s*([01])", value, flags=re.IGNORECASE)
    if labelled and len(set(labelled)) == 1:
        return labelled[0] == "1"
    if re.fullmatch(r"[\s`*01]+", value):
        digits = re.findall(r"[01]", value)
        if digits and len(set(digits)) == 1:
            return digits[0] == "1"
    return None


class JudgeClient:
    def __init__(self, api_base: str, model: str, max_tokens: int):
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.local = threading.local()

    def client(self):
        from openai import OpenAI

        value = getattr(self.local, "client", None)
        if value is None:
            value = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=self.api_base, timeout=1800)
            self.local.client = value
        return value

    def call(
        self,
        prompt: str,
        validator: Callable[[str], bool],
        retries: int = 5,
        choices: list[str] | None = None,
    ) -> tuple[str, int]:
        last = ""
        for attempt in range(1, retries + 1):
            try:
                extra_body: dict[str, Any] = {"reasoning_effort": "low"}
                if choices is not None:
                    extra_body["structured_outputs"] = {"choice": choices}
                response = self.client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=(attempt - 1) * 0.5,
                    max_tokens=self.max_tokens,
                    extra_body=extra_body,
                )
                last = (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last = f"API_ERROR: {type(exc).__name__}: {exc}"
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            if validator(last):
                return last, attempt
        return last, retries


def parallel_stage(
    rows: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    path: Path,
    parallel_workers: int,
    label: str,
) -> list[dict[str, Any]]:
    existing = {row["sample_uid"]: row for row in read_jsonl(path) if row.get("valid")}
    todo = [row for row in rows if row["sample_uid"] not in existing]
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(worker, row): row["sample_uid"] for row in todo}
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(path, result)
            if result.get("valid"):
                existing[result["sample_uid"]] = result
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == len(todo):
                print(f"{label}={completed}/{len(todo)} uid={result['sample_uid']}", flush=True)
    ordered = [existing[row["sample_uid"]] for row in rows if row["sample_uid"] in existing]
    compact_jsonl(path, ordered)
    if len(ordered) != len(rows):
        raise RuntimeError(f"{label}: only {len(ordered)}/{len(rows)} valid results")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--mathverse-parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    inference_root = Path(args.inference_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = read_jsonl(inference_root / "manifest.jsonl")
    import pandas as pd

    source_df = pd.read_parquet(args.mathverse_parquet)
    if len(source_df) != 3940:
        raise RuntimeError(f"unexpected MathVerse parquet rows: {len(source_df)}")
    source_by_key: dict[tuple[str, str], Any] = {}
    for _, dataset_row in source_df.iterrows():
        key = (str(dataset_row["sample_index"]), str(dataset_row["problem_version"]))
        if key in source_by_key:
            raise RuntimeError(f"duplicate MathVerse source key: {key}")
        source_by_key[key] = dataset_row
    raw_paths = sorted((inference_root / "raw").glob("*.jsonl"))
    predictions = [row for path in raw_paths for row in read_jsonl(path)]
    pred_by_uid = {row["sample_uid"]: row for row in predictions}
    if len(manifest) != 3940 or len(pred_by_uid) != 3940:
        raise RuntimeError(f"incomplete inference: manifest={len(manifest)} predictions={len(pred_by_uid)}")
    errors = [row for row in pred_by_uid.values() if row.get("finish_reason") == "error"]
    if errors:
        raise RuntimeError(f"inference contains {len(errors)} error rows")

    rows = []
    for source in manifest:
        source_key = (str(source["sample_index"]), str(source["problem_version"]))
        if source_key not in source_by_key:
            raise RuntimeError(f"missing MathVerse source key: {source_key}")
        dataset_source = source_by_key[source_key]
        prediction = pred_by_uid[source["sample_uid"]]
        rows.append(
            {
                **source,
                "prediction": prediction["model_answer"],
                "question_for_eval": str(dataset_source["question_for_eval"]),
                "metadata": dataset_source["metadata"],
                "finish_reason": prediction["finish_reason"],
                "completion_tokens": prediction.get("completion_tokens"),
            }
        )

    judge = JudgeClient(args.api_base, args.judge_model, args.judge_max_tokens)

    def extract_one(row: dict[str, Any]) -> dict[str, Any]:
        if row["question_type"] == "multi-choice":
            extracted = extract_mcq_option(row["prediction"])
            return {
                "sample_uid": row["sample_uid"],
                "raw_extract": extracted or "",
                "extract": extracted or "",
                "attempts": 0,
                "source": "mcq_final_option",
                "valid": extracted is not None,
            }
        raw, attempts = judge.call(
            extraction_prompt(row["prediction"]),
            lambda value: bool(clean_freeform_extract(value))
            or bool(extract_last_boxed_answer(row["prediction"])),
        )
        extracted = clean_freeform_extract(raw) or extract_last_boxed_answer(row["prediction"]) or ""
        return {
            "sample_uid": row["sample_uid"],
            "raw_extract": raw,
            "extract": extracted,
            "attempts": attempts,
            "source": "gpt_oss_freeform_extract",
            "valid": bool(extracted),
        }

    extracts = parallel_stage(
        rows,
        extract_one,
        output_root / "extract.jsonl",
        args.parallel_workers,
        "extract",
    )
    extract_by_uid = {row["sample_uid"]: row for row in extracts}

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        extracted = extract_by_uid[row["sample_uid"]]["extract"]
        if extracted.strip() == row["gold"].strip():
            return {
                "sample_uid": row["sample_uid"],
                "raw_score": "1",
                "score": True,
                "source": "exact_prefetch",
                "attempts": 0,
                "valid": True,
            }
        raw, attempts = judge.call(
            score_prompt(row["question_for_eval"], row["gold"], extracted),
            lambda value: parse_score(value) is not None,
            choices=["0", "1"],
        )
        parsed = parse_score(raw)
        return {
            "sample_uid": row["sample_uid"],
            "raw_score": raw,
            "score": parsed,
            "source": "gpt_oss",
            "attempts": attempts,
            "valid": parsed is not None,
        }

    scores = parallel_stage(
        rows,
        score_one,
        output_root / "score_rows.jsonl",
        args.parallel_workers,
        "score",
    )
    score_by_uid = {row["sample_uid"]: row for row in scores}

    detailed = []
    for row in rows:
        detailed.append(
            {
                **{key: row[key] for key in (
                    "sample_uid", "sample_index", "problem_index", "problem_version", "question_type",
                    "gold", "finish_reason", "completion_tokens"
                )},
                "extract": extract_by_uid[row["sample_uid"]]["extract"],
                "correct": bool(score_by_uid[row["sample_uid"]]["score"]),
                "judge_source": score_by_uid[row["sample_uid"]]["source"],
            }
        )
    compact_jsonl(output_root / "details.jsonl", detailed)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row["correct"]) for row in items)
        return {
            "rows": len(items),
            "correct": correct,
            "accuracy": correct / len(items) if items else 0.0,
            "finish_reasons": dict(Counter(row["finish_reason"] for row in items)),
        }

    report = {
        "complete": len(detailed) == 3940,
        "judge_model": args.judge_model,
        "protocol": "VLMEvalKit MathVerse scoring prompts with GPT-OSS-120B substituted for the default GPT-4o-mini judge; explicit final-option parsing is used for MCQ extraction and GPT-OSS demo echo is removed from free-form extraction",
        "overall": metrics(detailed),
        "by_problem_version": {
            version: metrics([row for row in detailed if row["problem_version"] == version])
            for version in sorted({row["problem_version"] for row in detailed})
        },
        "by_question_type": {
            kind: metrics([row for row in detailed if row["question_type"] == kind])
            for kind in sorted({row["question_type"] for row in detailed})
        },
        "judge_sources": dict(Counter(row["judge_source"] for row in detailed)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "accuracy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
