#!/usr/bin/env python3
"""Prepare, infer, and audit MathVista_MINI with the ViCuR generation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import qwen3vl_mathverse_termination as shared


EXPECTED_ROWS = 1000


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    raise ValueError("MathVista row has no embedded decoded_image bytes")


def command_prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    source = Path(args.mathvista_parquet).resolve()
    output = Path(args.output_root).resolve()
    table = pq.read_table(source)
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} MathVista_MINI rows, found {table.num_rows}")
    rows = table.to_pylist()
    if args.max_rows:
        rows = rows[: args.max_rows]

    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        blob = image_bytes(row["decoded_image"])
        digest = hashlib.sha256(blob).hexdigest()
        suffix = Path(str(row.get("image") or "image.png")).suffix or ".png"
        image_path = image_dir / f"{digest}{suffix}"
        if not image_path.is_file():
            image_path.write_bytes(blob)

        choices_list = [str(value) for value in (row.get("choices") or [])]
        choices = {chr(ord("A") + index): value for index, value in enumerate(choices_list)}
        answer = str(row["answer"])
        answer_option = None
        if choices:
            matches = [letter for letter, value in choices.items() if value == answer]
            if not matches:
                raise ValueError(f"cannot resolve MathVista option for pid={row['pid']}: {answer!r}")
            # The official MathVista builder uses list.index(), so duplicate
            # choice text resolves to the first matching option (pid 781).
            answer_option = matches[0]

        metadata = row.get("metadata") or {}
        manifest.append(
            {
                "sample_uid": f"mathvista:{row['pid']}",
                "ordinal": ordinal,
                "pid": str(row["pid"]),
                "gold": answer,
                "answer_option": answer_option,
                "question_type": str(row["question_type"]),
                "answer_type": str(row["answer_type"]),
                "precision": row.get("precision"),
                "unit": row.get("unit"),
                "choices": choices,
                "task": str(metadata.get("task", "unknown")),
                "skills": [str(value) for value in (metadata.get("skills") or [])],
                # The official dataset query includes its answer-format hint and options.
                "question": str(row["query"]),
                "image_path": str(image_path),
            }
        )

    if len({row["sample_uid"] for row in manifest}) != len(manifest):
        raise ValueError("duplicate MathVista sample_uid")
    shared.write_jsonl(output / "manifest.jsonl", manifest)
    summary = {
        "dataset": "MathVista_MINI/testmini",
        "huggingface_dataset": "AI4Math/MathVista",
        "huggingface_file": source.name,
        "rows": len(manifest),
        "full_split_rows": EXPECTED_ROWS,
        "system_prompt": shared.SYSTEM_PROMPT,
        "prompt_field": "query",
        "generation": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_tokens": 4096,
            "min_pixels": 3584,
            "max_pixels": 401408,
            "enable_thinking": False,
        },
        "question_types": dict(Counter(row["question_type"] for row in manifest)),
        "answer_types": dict(Counter(row["answer_type"] for row in manifest)),
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_summarize(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    manifest = shared.read_jsonl(output / "manifest.jsonl")
    raw_paths = (
        [output / "raw.jsonl"]
        if (output / "raw.jsonl").is_file()
        else sorted((output / "raw").glob("*.jsonl"))
    )
    records = [row for path in raw_paths for row in shared.read_jsonl(path)]
    by_uid: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in records:
        uid = str(row["sample_uid"])
        if uid in by_uid:
            duplicates.append(uid)
        by_uid[uid] = row
    missing = [row["sample_uid"] for row in manifest if row["sample_uid"] not in by_uid]
    ordered = [by_uid[row["sample_uid"]] for row in manifest if row["sample_uid"] in by_uid]
    errors = [row for row in ordered if row.get("finish_reason") == "error"]
    finish = Counter(row.get("finish_reason", "missing") for row in ordered)
    tokens = [row["completion_tokens"] for row in ordered if isinstance(row.get("completion_tokens"), int)]
    report = {
        "complete": len(ordered) == len(manifest) and not duplicates and not errors,
        "expected_rows": len(manifest),
        "completed_rows": len(ordered),
        "duplicate_rows": len(duplicates),
        "missing_rows": len(missing),
        "error_rows": len(errors),
        "finish_reasons": dict(finish),
        "truncation_rate": finish.get("length", 0) / len(ordered) if ordered else None,
        "completion_tokens": {
            "mean": statistics.mean(tokens) if tokens else None,
            "median": statistics.median(tokens) if tokens else None,
            "min": min(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
    }
    (output / "inference_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--mathvista-parquet", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--max-rows", type=int, default=0)
    prepare.set_defaults(func=command_prepare)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--output-root", required=True)
    infer.add_argument("--api-base", required=True)
    infer.add_argument("--model-id", default="Qwen3.5-4B")
    infer.add_argument("--parallel-workers", type=int, default=8)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--shard-id", type=int, default=0)
    infer.add_argument("--num-shards", type=int, default=1)
    infer.add_argument("--max-retries", type=int, default=3)
    infer.set_defaults(func=shared.command_infer)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", required=True)
    summarize.set_defaults(func=command_summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
