#!/usr/bin/env python3
"""Prepare exact MathVision-testmini and DynaMath-test manifests and images."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


DYNAMATH_REVISION = "c72be604052511aa6f2e94164139f8a2e801bd33"
SYSTEM_PROMPT = "You are a helpful assistant."
DYNAMATH_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in "
    "<answer>...</answer> tags.Please provide a clear, concise response within <answer> </answer> tags "
    "that directly addresses the question."
)
SHARDS = {"mathvision": 1, "dynamath": 8}
EXPECTED = {"mathvision": 304, "dynamath": 5010}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_image(image_record: dict[str, Any], fallback_name: str, root: Path) -> Path:
    payload = image_record["bytes"]
    source_name = image_record.get("path") or fallback_name
    suffix = Path(str(source_name)).suffix.lower() or Path(fallback_name).suffix.lower() or ".png"
    target = root / f"{Path(fallback_name).stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != len(payload):
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    return target.resolve()


def mathvision_prompt(question: str, choices: list[str]) -> str:
    query = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices:
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        choice_text = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
        query += (
            f"{question}\nChoices: {choice_text}\n"
            "Answer the question with the option's letter from the given choices directly."
        )
    else:
        query += question
    return query


def prepare_mathvision(parquet: Path, image_root: Path) -> list[dict[str, Any]]:
    source = pq.read_table(parquet).to_pylist()
    if len(source) != EXPECTED["mathvision"]:
        raise RuntimeError(f"MathVision rows={len(source)}")
    rows: list[dict[str, Any]] = []
    for row in source:
        uid = f"mathvision:{row['id']}"
        image_path = save_image(
            row["decoded_image"], f"mathvision_{row['id']}{Path(row['image']).suffix}", image_root
        )
        choices = list(row["options"] or [])
        rows.append(
            {
                "sample_uid": uid,
                "benchmark": "mathvision",
                "split": "testmini",
                "dataset_id": str(row["id"]),
                "question": mathvision_prompt(str(row["question"]), choices),
                "system_prompt": SYSTEM_PROMPT,
                "image_path": str(image_path),
                "gold": str(row["answer"]),
                "options": choices,
                "subject": str(row["subject"]),
                "level": int(row["level"]),
            }
        )
    return rows


def prepare_dynamath(parquet: Path, image_root: Path) -> list[dict[str, Any]]:
    source = pq.read_table(parquet).to_pylist()
    if len(source) != EXPECTED["dynamath"]:
        raise RuntimeError(f"DynaMath rows={len(source)}")
    group_counts = Counter(str(row["id"]) for row in source)
    if len(group_counts) != 501 or set(group_counts.values()) != {10}:
        raise RuntimeError(f"unexpected DynaMath groups: {Counter(group_counts.values())}")
    rows: list[dict[str, Any]] = []
    for row in source:
        question_id = int(row["question_id"])
        uid = f"dynamath:{question_id}"
        image_path = save_image(
            row["decoded_image"], f"dynamath_{question_id}{Path(row['image']).suffix}", image_root
        )
        rows.append(
            {
                "sample_uid": uid,
                "benchmark": "dynamath",
                "split": "test",
                "dataset_id": str(row["id"]),
                "question_id": question_id,
                "group_id": str(row["id"]),
                "question": str(row["question"]),
                "system_prompt": DYNAMATH_SYSTEM_PROMPT,
                "image_path": str(image_path),
                "gold": str(row["ground_truth"]),
                "answer_type": str(row["answer_type"]),
                "subject": str(row["subject"]),
                "knowledge_level": str(row["knowledge_level"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mathvision-parquet", required=True)
    parser.add_argument("--dynamath-parquet", required=True)
    parser.add_argument("--mathvision-image-root")
    parser.add_argument("--dynamath-image-root")
    parser.add_argument("--models", default="model")
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    mathvision_parquet = Path(args.mathvision_parquet).resolve()
    dynamath_parquet = Path(args.dynamath_parquet).resolve()
    mathvision_image_root = (
        Path(args.mathvision_image_root).resolve()
        if args.mathvision_image_root
        else mathvision_parquet.parent / "decoded_testmini_images"
    )
    dynamath_image_root = (
        Path(args.dynamath_image_root).resolve()
        if args.dynamath_image_root
        else dynamath_parquet.parent / "decoded_test_images"
    )
    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    if not models:
        raise RuntimeError("--models must contain at least one model key")
    manifests = {
        "mathvision": prepare_mathvision(mathvision_parquet, mathvision_image_root),
        "dynamath": prepare_dynamath(dynamath_parquet, dynamath_image_root),
    }
    for benchmark, rows in manifests.items():
        if len({row["sample_uid"] for row in rows}) != len(rows):
            raise RuntimeError(f"{benchmark}: duplicate sample_uid")
        if any(not Path(row["image_path"]).is_file() for row in rows):
            raise RuntimeError(f"{benchmark}: missing image")
        for model in models:
            write_jsonl(output / model / benchmark / "manifest.jsonl", rows)

    for model in models:
        smoke_rows = [manifests["mathvision"][0], manifests["dynamath"][0]]
        write_jsonl(output / "smoke" / model / "manifest.jsonl", smoke_rows)

    tasks: list[dict[str, Any]] = []
    task_id = 0
    for benchmark in ("mathvision", "dynamath"):
        for shard_id in range(SHARDS[benchmark]):
            for model in models:
                tasks.append(
                    {
                        "task_id": task_id,
                        "model": model,
                        "benchmark": benchmark,
                        "shard_id": shard_id,
                        "num_shards": SHARDS[benchmark],
                    }
                )
                task_id += 1
    (output / "task_map.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "task_map.tsv").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                f"{task['task_id']}\t{task['model']}\t{task['benchmark']}\t"
                f"{task['shard_id']}\t{task['num_shards']}\n"
            )

    report = {
        "complete": True,
        "protocol": {
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "seed": 42,
            "enable_thinking": False,
            "min_pixels": 3584,
            "max_pixels": 401408,
        },
        "datasets": {
            "mathvision": {
                "dataset": "MathLLMs/MathVision",
                "split": "testmini",
                "rows": len(manifests["mathvision"]),
                "parquet": str(mathvision_parquet),
                "sha256": sha256(mathvision_parquet),
                "prompt": "lmms-eval mathvision_testmini",
                "scoring": "lmms-eval mathvision_standard_eval",
            },
            "dynamath": {
                "dataset": "kcz358/DynaMath",
                "revision": DYNAMATH_REVISION,
                "split": "test",
                "rows": len(manifests["dynamath"]),
                "groups": 501,
                "variants_per_group": 10,
                "parquet": str(dynamath_parquet),
                "sha256": sha256(dynamath_parquet),
                "prompt": "lmms-eval dynamath_reasoning",
                "scoring": "lmms-eval dynamath_average and dynamath_worst with USE_LLM_JUDGE=False",
            },
        },
        "models": list(models),
        "task_count": len(tasks),
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
