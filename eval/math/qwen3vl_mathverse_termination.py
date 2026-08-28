#!/usr/bin/env python3
"""Measure Qwen3-VL termination on a stratified MathVerse_MINI smoke."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "You are a helpful assistant."
VERSIONS = (
    "Text Dominant",
    "Text Lite",
    "Vision Intensive",
    "Vision Dominant",
    "Vision Only",
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    raise ValueError("MathVerse row has no embedded image bytes")


def select_rows(rows: list[dict[str, Any]], per_version: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for version in VERSIONS:
        version_rows = [row for row in rows if row["problem_version"] == version]
        if per_version <= 0 or per_version >= len(version_rows):
            selected.extend(sorted(version_rows, key=lambda row: int(row["problem_index"])))
            continue
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in version_rows:
            by_type[str(row["question_type"])].append(row)
        types = sorted(by_type)
        quotas = {kind: per_version // len(types) for kind in types}
        for kind in types[: per_version % len(types)]:
            quotas[kind] += 1
        chosen: list[dict[str, Any]] = []
        for kind in types:
            candidates = sorted(by_type[kind], key=lambda row: int(row["problem_index"]))
            chosen.extend(rng.sample(candidates, quotas[kind]))
        selected.extend(sorted(chosen, key=lambda row: int(row["problem_index"])))
    return selected


def command_prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    source = Path(args.mathverse_root).resolve() / "testmini.parquet"
    output = Path(args.output_root).resolve()
    rows = pq.read_table(source).to_pylist()
    chosen = select_rows(rows, args.per_version, args.seed)
    manifest: list[dict[str, Any]] = []
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for ordinal, row in enumerate(chosen):
        blob = image_bytes(row["image"])
        digest = hashlib.sha256(blob).hexdigest()
        image_path = image_dir / f"{digest}.png"
        if not image_path.is_file():
            image_path.write_bytes(blob)
        manifest.append(
            {
                "sample_uid": f"mathverse:{row['sample_index']}",
                "ordinal": ordinal,
                "sample_index": str(row["sample_index"]),
                "problem_index": str(row["problem_index"]),
                "problem_version": str(row["problem_version"]),
                "question_type": str(row["question_type"]),
                "gold": str(row["answer"]),
                "question": str(row["question"]),
                "image_path": str(image_path),
            }
        )
    if len({row["sample_uid"] for row in manifest}) != len(manifest):
        raise ValueError("duplicate MathVerse sample_uid")
    write_jsonl(output / "manifest.jsonl", manifest)
    summary = {
        "dataset": "MathVerse_MINI",
        "source": str(source),
        "rows": len(manifest),
        "per_version": args.per_version,
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "question_field": "question",
        "enable_thinking": False,
        "versions": dict(Counter(row["problem_version"] for row in manifest)),
        "question_types": dict(Counter(row["question_type"] for row in manifest)),
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def data_uri(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def command_infer(args: argparse.Namespace) -> None:
    from openai import OpenAI

    output = Path(args.output_root).resolve()
    manifest = [
        row
        for index, row in enumerate(read_jsonl(output / "manifest.jsonl"))
        if index % args.num_shards == args.shard_id
    ]
    raw_path = (
        output / "raw.jsonl"
        if args.num_shards == 1
        else output / "raw" / f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"
    )
    previous_rows = read_jsonl(raw_path)
    existing = {
        row["sample_uid"]: row
        for row in previous_rows
        if row.get("finish_reason") not in {None, "error"}
        and not str(row.get("model_answer", "")).startswith("[ERROR]")
    }
    todo = [row for row in manifest if row["sample_uid"] not in existing]
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=args.api_base, timeout=3600)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_uri(row["image_path"])},
                                },
                                {"type": "text", "text": row["question"]},
                            ],
                        },
                    ],
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    max_tokens=4096,
                    seed=args.seed,
                    extra_body={
                        "top_k": 20,
                        "repetition_penalty": 1.0,
                        "mm_processor_kwargs": {"min_pixels": 3584, "max_pixels": 401408},
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                choice = response.choices[0]
                usage = response.usage
                return {
                    **{key: value for key, value in row.items() if key not in {"question", "image_path"}},
                    "model_answer": (choice.message.content or "").strip(),
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "elapsed_seconds": time.time() - started,
                    "attempts": attempt,
                    "model_id": args.model_id,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1.0,
                    "min_pixels": 3584,
                    "max_pixels": 401408,
                    "enable_thinking": False,
                }
            except Exception as exc:
                last_error = exc
                if attempt < args.max_retries:
                    time.sleep(1.0)
        assert last_error is not None
        return {
            **{key: value for key, value in row.items() if key not in {"question", "image_path"}},
            "model_answer": f"[ERROR] {type(last_error).__name__}: {last_error}",
            "finish_reason": "error",
            "elapsed_seconds": time.time() - started,
            "attempts": args.max_retries,
            "model_id": args.model_id,
        }

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, raw_path.open(
        "a", encoding="utf-8"
    ) as handle:
        futures = {executor.submit(run_one, row): row["sample_uid"] for row in todo}
        completed = 0
        for future in as_completed(futures):
            record = future.result()
            existing[record["sample_uid"]] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            if completed == 1 or completed % 5 == 0 or completed == len(todo):
                print(
                    f"shard={args.shard_id}/{args.num_shards} completed={completed}/{len(todo)} "
                    f"uid={record['sample_uid']}",
                    flush=True,
                )
    write_jsonl(raw_path, sorted(existing.values(), key=lambda row: int(row["ordinal"])))


def command_summarize(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    manifest = read_jsonl(output / "manifest.jsonl")
    raw_paths = [output / "raw.jsonl"] if (output / "raw.jsonl").is_file() else sorted((output / "raw").glob("*.jsonl"))
    records = [row for path in raw_paths for row in read_jsonl(path)]
    by_uid = {row["sample_uid"]: row for row in records}

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        finish = Counter(row.get("finish_reason", "missing") for row in rows)
        tokens = [row["completion_tokens"] for row in rows if isinstance(row.get("completion_tokens"), int)]
        return {
            "rows": len(rows),
            "finish_reasons": dict(finish),
            "truncation_rate": finish.get("length", 0) / len(rows) if rows else 0.0,
            "completion_tokens": {
                "mean": statistics.mean(tokens) if tokens else None,
                "median": statistics.median(tokens) if tokens else None,
                "min": min(tokens) if tokens else None,
                "max": max(tokens) if tokens else None,
            },
        }

    ordered = [by_uid[row["sample_uid"]] for row in manifest if row["sample_uid"] in by_uid]
    report = {
        "complete": len(ordered) == len(manifest),
        "expected_rows": len(manifest),
        "completed_rows": len(ordered),
        "overall": block(ordered),
        "by_problem_version": {
            version: block([row for row in ordered if row["problem_version"] == version])
            for version in VERSIONS
        },
        "by_question_type": {
            kind: block([row for row in ordered if row["question_type"] == kind])
            for kind in sorted({row["question_type"] for row in ordered})
        },
    }
    (output / "termination_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--mathverse-root", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--per-version", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=command_prepare)
    infer = subparsers.add_parser("infer")
    infer.add_argument("--output-root", required=True)
    infer.add_argument("--api-base", required=True)
    infer.add_argument("--model-id", default="Qwen3-VL-8B-Instruct")
    infer.add_argument("--parallel-workers", type=int, default=8)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--shard-id", type=int, default=0)
    infer.add_argument("--num-shards", type=int, default=1)
    infer.add_argument("--max-retries", type=int, default=3)
    infer.set_defaults(func=command_infer)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", required=True)
    summarize.set_defaults(func=command_summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
