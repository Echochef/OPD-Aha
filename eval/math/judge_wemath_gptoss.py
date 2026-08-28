#!/usr/bin/env python3
"""Evaluate WeMath with the official VLMEvalKit option-matching prompt and GPT-OSS."""

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

import qwen3vl_wemath_strict as wemath


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


def build_prompt(question: str, choices: dict[str, str], prediction: str) -> str:
    options = " ".join(f"{letter}. {value}" for letter, value in choices.items())
    template = (
        "You are an AI assistant who will help me to match an answer with several options of a "
        "single-choice question. You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. If the meaning of all "
        "options are significantly different from the answer, output Z. Your should output a single "
        "uppercase character in A, B, C, D, E, F, G (if they are valid options), and Z. \n"
        "Example 1: \nQuestion: <start>\nWhat is the main object in image?\n"
        "Options: A. teddy bear B. rabbit C. cat D. dog\n<end>\n"
        "Answer: <start>\na cute teddy bear\n<end>\nYour output: A\n"
        "Example 2: \nQuestion: <start>\nWhat is the main object in image?\n"
        "Options: A. teddy bear B. rabbit C. cat D. dog\n<end>\n"
        "Answer: <start>\nSpider\n<end>\nYour output: Z\n"
        "Example 3: \nQuestion: <start>\n{}\nOptions: {}\n<end>\n"
        "Answer: <start>\n{}\n<end>\nYour output: "
    )
    clean_question = question.replace(
        "Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n"
        "<Thought process>: <<your thought process>> <Answer>: <<your option>>",
        "",
    )
    return template.format(clean_question, options, prediction)


def parse_option(text: str, choices: dict[str, str]) -> str | None:
    value = str(text).strip().upper()
    if value in {*choices, "Z"}:
        return value
    matches = re.findall(r"\b([A-HZ])\b", value)
    valid = [match for match in matches if match in choices or match == "Z"]
    if valid and len(set(valid)) == 1:
        return valid[0]
    return None


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

    def call(self, prompt: str, choices: dict[str, str]) -> tuple[str, int]:
        allowed = list(choices) + ["Z"]
        last = ""
        for attempt in range(1, 4):
            try:
                response = self.client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=(attempt - 1) * 0.5,
                    max_tokens=self.max_tokens,
                    extra_body={
                        "reasoning_effort": "low",
                        "structured_outputs": {"choice": allowed},
                    },
                )
                last = (response.choices[0].message.content or "").strip()
                if parse_option(last, choices) is not None:
                    return last, attempt
            except Exception as exc:
                last = f"API_ERROR: {type(exc).__name__}: {exc}"
            time.sleep(min(2 ** (attempt - 1), 4))
        return last, 3


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
    if len(manifest) != wemath.EXPECTED_ROWS or len(pred_by_uid) != wemath.EXPECTED_ROWS:
        raise RuntimeError(f"incomplete WeMath inference: {len(manifest)=} {len(pred_by_uid)=}")
    if any(row.get("finish_reason") == "error" for row in predictions):
        raise RuntimeError("WeMath inference contains error rows")

    rows = []
    for row in manifest:
        prediction = pred_by_uid[row["sample_uid"]]
        rows.append({**row, "model_answer": prediction["model_answer"]})
    existing_path = output_root / "judge_rows.jsonl"
    existing = {row["sample_uid"]: row for row in read_jsonl(existing_path) if row.get("valid")}
    judge = Judge(args.api_base, args.judge_model, args.judge_max_tokens)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        prefetched = wemath.vlmevalkit_exact_answer(row["model_answer"], row["choices"])
        if prefetched is not None:
            return {
                "sample_uid": row["sample_uid"],
                "prediction": prefetched,
                "raw_judge": prefetched,
                "source": "official_prefetch",
                "attempts": 0,
                "valid": True,
            }
        raw, attempts = judge.call(build_prompt(row["question"], row["choices"], row["model_answer"]), row["choices"])
        parsed = parse_option(raw, row["choices"])
        return {
            "sample_uid": row["sample_uid"],
            "prediction": parsed,
            "raw_judge": raw,
            "source": "gpt_oss_official_wemath_prompt",
            "attempts": attempts,
            "valid": parsed is not None,
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
                print(f"wemath_judge={index}/{len(todo)}", flush=True)
    ordered = [existing[row["sample_uid"]] for row in rows if row["sample_uid"] in existing]
    compact_jsonl(existing_path, ordered)
    if len(ordered) != len(rows):
        raise RuntimeError(f"only {len(ordered)}/{len(rows)} valid WeMath judge rows")

    official, details = wemath.score_rows(
        rows,
        lambda row: existing[row["sample_uid"]]["prediction"],
    )
    judged_details = []
    details_by_uid = {row["sample_uid"]: row for row in details}
    for row in rows:
        judged_details.append(
            {
                **details_by_uid[row["sample_uid"]],
                "judge_source": existing[row["sample_uid"]]["source"],
                "model_answer": row["model_answer"],
            }
        )
    compact_jsonl(output_root / "details.jsonl", judged_details)
    report = {
        "complete": len(judged_details) == wemath.EXPECTED_ROWS,
        "dataset": "WeMath/testmini",
        "judge_model": args.judge_model,
        "protocol": "VLMEvalKit WeMath official option-prefetch and option-matching prompt with GPT-OSS-120B substituted for the default API judge",
        "metrics": official,
        "judge_sources": dict(Counter(row["judge_source"] for row in judged_details)),
    }
    (output_root / "accuracy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
