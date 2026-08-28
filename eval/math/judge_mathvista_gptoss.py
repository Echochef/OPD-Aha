#!/usr/bin/env python3
"""Evaluate MathVista_MINI with the official VLMEvalKit extraction prompt and GPT-OSS."""

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
from typing import Any


EXAMPLES = (
    """Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.

Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14
""",
    """Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.

Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6
""",
    """Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.

Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45
""",
    """Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.

Question: Between which two years does the line graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]
""",
    """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.

Question: What fraction of the shape is blue?

Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
""",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compact_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def extraction_prompt(question: str, prediction: str) -> str:
    prompt = (
        "Please read the following example.\n"
        "Then extract the answer from the model response and type it at the end of the prompt.\n\n"
    )
    for example in EXAMPLES:
        prompt += example + "\n"
    prompt += question + "\n"
    prompt += "Model respone: " + prediction
    prompt += "Extracted answer:"
    return prompt


def mcq_option(text: str, choices: dict[str, str]) -> str | None:
    answer = str(text)
    normalized = answer
    for char in ".()[],:;!*#{}":
        normalized = normalized.replace(char, " ")
    splits = normalized.split()
    present = [choice for choice in choices if choice in splits]
    if len(present) == 1 and splits.index(present[0]) > len(splits) - 5:
        return present[0]
    lowered = answer.lower()
    matching = [key for key, value in choices.items() if str(value).lower() in lowered]
    if len(matching) == 1:
        return matching[0]
    return None


def normalize_extract(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"^Extracted\s+answer\s*:\s*", "", value, flags=re.IGNORECASE)
    return value.strip().strip("`").strip()


def parsed_value(row: dict[str, Any], text: str) -> Any:
    value = normalize_extract(text)
    try:
        if row["question_type"] == "multi_choice":
            return mcq_option(value, row["choices"])
        if row["answer_type"] == "integer":
            return int(value)
        if row["answer_type"] == "float":
            return float(value)
        return value
    except (TypeError, ValueError):
        return None


def is_correct(row: dict[str, Any], value: Any) -> bool:
    if row["question_type"] == "multi_choice":
        return value == row["answer_option"]
    if row["answer_type"] == "integer":
        try:
            return value == int(row["gold"])
        except (TypeError, ValueError):
            return False
    if row["answer_type"] == "float":
        try:
            return value == float(row["gold"])
        except (TypeError, ValueError):
            return False
    return str(value) == str(row["gold"])


class Judge:
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

    def extract(self, prompt: str) -> tuple[str, int]:
        last = ""
        for attempt in range(1, 6):
            try:
                response = self.client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=(attempt - 1) * 0.5,
                    max_tokens=self.max_tokens,
                    extra_body={"reasoning_effort": "low"},
                )
                last = (response.choices[0].message.content or "").strip()
                if normalize_extract(last):
                    return last, attempt
            except Exception as exc:
                last = f"API_ERROR: {type(exc).__name__}: {exc}"
            time.sleep(min(2 ** (attempt - 1), 8))
        return last, 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    args = parser.parse_args()

    inference_root = Path(args.inference_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = read_jsonl(inference_root / "manifest.jsonl")
    predictions = [
        row
        for path in sorted((inference_root / "raw").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    pred_by_uid = {row["sample_uid"]: row for row in predictions}
    if len(manifest) != 1000 or len(pred_by_uid) != 1000:
        raise RuntimeError(f"incomplete MathVista inference: {len(manifest)=} {len(pred_by_uid)=}")
    if any(row.get("finish_reason") == "error" for row in predictions):
        raise RuntimeError("MathVista inference contains error rows")

    existing_path = output_root / "extract.jsonl"
    existing = {row["sample_uid"]: row for row in read_jsonl(existing_path) if row.get("valid")}
    rows = [{**row, "prediction": pred_by_uid[row["sample_uid"]]["model_answer"]} for row in manifest]
    judge = Judge(args.api_base, args.judge_model, args.judge_max_tokens)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        prefetched = parsed_value(row, row["prediction"])
        if prefetched is not None:
            return {
                "sample_uid": row["sample_uid"],
                "raw_extract": str(prefetched),
                "extract": prefetched,
                "correct": is_correct(row, prefetched),
                "source": "official_prefetch",
                "attempts": 0,
                "valid": True,
            }
        raw, attempts = judge.extract(extraction_prompt(row["question"], row["prediction"]))
        extracted = parsed_value(row, raw)
        return {
            "sample_uid": row["sample_uid"],
            "raw_extract": raw,
            "extract": extracted,
            "correct": is_correct(row, extracted),
            "source": "gpt_oss_official_extraction_prompt",
            "attempts": attempts,
            "valid": extracted is not None,
        }

    todo = [row for row in rows if row["sample_uid"] not in existing]
    output_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, existing_path.open(
        "a", encoding="utf-8"
    ) as handle:
        futures = {executor.submit(run_one, row): row["sample_uid"] for row in todo}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if result["valid"]:
                existing[result["sample_uid"]] = result
            if index == 1 or index % 50 == 0 or index == len(todo):
                print(f"mathvista_judge={index}/{len(todo)}", flush=True)
    ordered = [existing[row["sample_uid"]] for row in rows if row["sample_uid"] in existing]
    compact_jsonl(existing_path, ordered)
    if len(ordered) != len(rows):
        raise RuntimeError(f"only {len(ordered)}/{len(rows)} valid MathVista extracts")

    merged = [{**row, **existing[row["sample_uid"]]} for row in rows]
    compact_jsonl(output_root / "details.jsonl", merged)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        hits = sum(bool(row["correct"]) for row in items)
        return {"rows": len(items), "correct": hits, "accuracy": hits / len(items) if items else 0.0}

    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        for skill in row["skills"]:
            by_skill[skill].append(row)
    report = {
        "complete": len(merged) == 1000,
        "dataset": "MathVista_MINI/testmini",
        "judge_model": args.judge_model,
        "protocol": "VLMEvalKit MathVista official answer-extraction prompt with GPT-OSS-120B substituted for the default API judge; official deterministic prefetch is retained",
        "overall": metrics(merged),
        "by_task": {key: metrics([row for row in merged if row["task"] == key]) for key in sorted({row["task"] for row in merged})},
        "by_skill": {key: metrics(value) for key, value in sorted(by_skill.items())},
        "judge_sources": dict(Counter(row["source"] for row in merged)),
    }
    (output_root / "accuracy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
