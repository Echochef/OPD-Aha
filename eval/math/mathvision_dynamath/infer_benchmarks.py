#!/usr/bin/env python3
"""Run protocol-matched Qwen3.5 inference for MathVision and DynaMath."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def data_uri(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    from openai import OpenAI

    output = Path(args.output_root).resolve()
    all_manifest = read_jsonl(output / "manifest.jsonl")
    manifest = [
        row for index, row in enumerate(all_manifest) if index % args.num_shards == args.shard_id
    ]
    raw_path = (
        output / "raw.jsonl"
        if args.num_shards == 1
        else output / "raw" / f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"
    )
    previous = read_jsonl(raw_path)
    existing = {
        row["sample_uid"]: row
        for row in previous
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
                        {"role": "system", "content": row["system_prompt"]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_uri(row["image_path"])}},
                                {"type": "text", "text": row["question"]},
                            ],
                        },
                    ],
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1.5,
                    max_tokens=args.max_tokens,
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
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"question", "image_path", "system_prompt"}
                    },
                    "model_answer": (choice.message.content or "").strip(),
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "elapsed_seconds": time.time() - started,
                    "attempts": attempt,
                    "model_id": args.model_id,
                    "max_tokens": args.max_tokens,
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
            **{
                key: value
                for key, value in row.items()
                if key not in {"question", "image_path", "system_prompt"}
            },
            "model_answer": f"[ERROR] {type(last_error).__name__}: {last_error}",
            "finish_reason": "error",
            "elapsed_seconds": time.time() - started,
            "attempts": args.max_retries,
            "model_id": args.model_id,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
        }

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor, raw_path.open(
        "a", encoding="utf-8"
    ) as handle:
        futures = {executor.submit(run_one, row): row["sample_uid"] for row in todo}
        for completed, future in enumerate(as_completed(futures), 1):
            record = future.result()
            existing[record["sample_uid"]] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if completed == 1 or completed % 25 == 0 or completed == len(todo):
                print(
                    f"shard={args.shard_id}/{args.num_shards} completed={completed}/{len(todo)} "
                    f"uid={record['sample_uid']} finish={record['finish_reason']} "
                    f"tokens={record.get('completion_tokens')}",
                    flush=True,
                )
    ordered = [existing[row["sample_uid"]] for row in manifest if row["sample_uid"] in existing]
    if len(ordered) != len(manifest):
        raise RuntimeError(f"shard incomplete: expected={len(manifest)} rows={len(ordered)}")
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, raw_path)


if __name__ == "__main__":
    main()
