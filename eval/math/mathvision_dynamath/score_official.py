#!/usr/bin/env python3
"""Score MathVision and DynaMath with the installed lmms-eval official rules."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


EXPECTED = {"mathvision": 304, "dynamath": 5010}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_predictions(
    root: Path, benchmark: str, allowed_max_tokens: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_root = root / benchmark
    manifest = read_jsonl(benchmark_root / "manifest.jsonl")
    raw_single = benchmark_root / "raw.jsonl"
    paths = [raw_single] if raw_single.is_file() else sorted((benchmark_root / "raw").glob("*.jsonl"))
    predictions = [row for path in paths for row in read_jsonl(path)]
    if len(manifest) != EXPECTED[benchmark] or len(predictions) != EXPECTED[benchmark]:
        raise RuntimeError(
            f"{root.name}/{benchmark}: manifest={len(manifest)} predictions={len(predictions)}"
        )
    if len({row["sample_uid"] for row in predictions}) != len(predictions):
        raise RuntimeError(f"{root.name}/{benchmark}: duplicate predictions")
    if {row["sample_uid"] for row in predictions} != {row["sample_uid"] for row in manifest}:
        raise RuntimeError(f"{root.name}/{benchmark}: prediction IDs differ from manifest")
    errors = [row for row in predictions if row.get("finish_reason") == "error"]
    bad_protocol = [
        row
        for row in predictions
        if row.get("max_tokens") not in allowed_max_tokens
        or row.get("enable_thinking") is not False
    ]
    if errors or bad_protocol:
        raise RuntimeError(
            f"{root.name}/{benchmark}: errors={len(errors)} bad_protocol={len(bad_protocol)}"
        )
    return manifest, predictions


def score_mathvision(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]], module):
    pred_by_uid = {row["sample_uid"]: row for row in predictions}
    details: list[dict[str, Any]] = []
    for source in manifest:
        prediction = pred_by_uid[source["sample_uid"]]
        model_answer = str(prediction["model_answer"]).strip()
        gold = str(source["gold"])
        choices = list(source.get("options") or [])
        gold_value = choices[ord(gold) - ord("A")] if choices and gold in "ABCDE" else ""
        for choice in "ABCDE":
            if (
                model_answer.endswith(f" {choice}.")
                or model_answer.endswith(f" ({choice}).")
                or model_answer.startswith(f"{choice}\n")
                or model_answer.startswith(f"({choice})\n")
                or model_answer.startswith(f"({choice}) {choice}\n")
            ):
                model_answer = choice
        if module.is_number(model_answer.split("is ")[-1].rstrip(".")):
            model_answer = model_answer.split("is ")[-1].rstrip(".")
        if "oxed{" not in model_answer:
            for flag in [
                "the final answer is",
                "the answer is",
                "the correct answer is",
                "the answer should be",
            ]:
                raw_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
                capitalized = flag.replace("the", "The")
                raw_answer = model_answer
                model_answer = model_answer.split(capitalized)[-1].strip()
                if capitalized in raw_answer:
                    model_answer = model_answer.split("\n")[0].split(". ")[0]
        elif model_answer.count("oxed{") > 1:
            model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]
        parsed = (
            module.find_math_answer(model_answer)
            .replace("(a)", "a").replace("(b)", "b").replace("(c)", "c")
            .replace("(d)", "d").replace("(e)", "e")
            .replace("{a}", "a").replace("{b}", "b").replace("{c}", "c")
            .replace("{d}", "d").replace("{e}", "e")
            .rstrip(".").lstrip(":").strip()
        )
        correct = bool(module.is_equal(gold, parsed) or module.is_equal(gold_value, parsed))
        details.append(
            {
                "sample_uid": source["sample_uid"],
                "gold": gold,
                "gold_value": gold_value,
                "parsed_answer": parsed,
                "correct": correct,
                "finish_reason": prediction["finish_reason"],
            }
        )
    return details, {
        "accuracy": mean(row["correct"] for row in details),
        "correct": sum(row["correct"] for row in details),
        "total": len(details),
        "truncated": sum(row["finish_reason"] == "length" for row in details),
        "protocol": "lmms-eval mathvision_standard_eval on MathVision testmini",
    }


def score_dynamath(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]], module):
    pred_by_uid = {row["sample_uid"]: row for row in predictions}
    details: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for source in manifest:
        prediction = pred_by_uid[source["sample_uid"]]
        score = module.compute_score(
            data_source="dynamath",
            solution_str=str(prediction["model_answer"]).strip(),
            ground_truth=str(source["gold"]),
            extra_info={"question": source["question"]},
        )
        accuracy = float(score["acc_score"])
        group_id = str(source["group_id"])
        grouped[group_id].append(accuracy)
        details.append(
            {
                "sample_uid": source["sample_uid"],
                "group_id": group_id,
                "gold": str(source["gold"]),
                "parsed_answer": score["predict_str"],
                "correct": accuracy,
                "format_score": float(score["format_reward_score"]),
                "finish_reason": prediction["finish_reason"],
            }
        )
    if len(grouped) != 501 or {len(values) for values in grouped.values()} != {10}:
        raise RuntimeError("DynaMath grouping differs from 501 groups x 10 variants")
    return details, {
        "average": mean(mean(values) for values in grouped.values()),
        "worst": mean(min(values) for values in grouped.values()),
        "row_accuracy": mean(row["correct"] for row in details),
        "format_accuracy": mean(row["format_score"] for row in details),
        "correct": sum(row["correct"] for row in details),
        "total": len(details),
        "groups": len(grouped),
        "variants_per_group": 10,
        "truncated": sum(row["finish_reason"] == "length" for row in details),
        "protocol": "lmms-eval dynamath_average and dynamath_worst with USE_LLM_JUDGE=False",
    }


def subtract(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(left[key]) - float(right[key])
        for key in left.keys() & right.keys()
        if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
        and key not in {"correct", "total", "groups", "variants_per_group", "truncated"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--models", default="model")
    parser.add_argument("--allowed-max-tokens", default="4096")
    parser.add_argument(
        "--lmms-eval-root",
        default=os.environ.get("LMMS_EVAL_ROOT", ""),
        help="Checkout root of lmms-eval (or set LMMS_EVAL_ROOT)",
    )
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    allowed_max_tokens = {
        int(value.strip()) for value in args.allowed_max_tokens.split(",") if value.strip()
    }
    if not models:
        raise RuntimeError("--models must contain at least one model key")
    if not allowed_max_tokens:
        raise RuntimeError("--allowed-max-tokens must contain at least one integer")
    if not args.lmms_eval_root:
        raise RuntimeError("--lmms-eval-root or LMMS_EVAL_ROOT is required")
    lmms_eval_root = Path(args.lmms_eval_root).resolve()
    mathvision_eval = lmms_eval_root / "lmms_eval/tasks/mathvision/eval_utils.py"
    dynamath_eval = lmms_eval_root / "lmms_eval/tasks/_task_utils/reasoning_utils.py"
    if not mathvision_eval.is_file() or not dynamath_eval.is_file():
        raise RuntimeError(f"lmms-eval task utilities not found under {lmms_eval_root}")
    os.environ["USE_LLM_JUDGE"] = "False"
    mathvision_module = load_module("mathvision_eval_utils", mathvision_eval)
    dynamath_module = load_module("dynamath_reasoning_utils", dynamath_eval)
    report: dict[str, Any] = {
        "complete": True,
        "max_tokens": sorted(allowed_max_tokens),
        "enable_thinking": False,
        "models": {},
    }
    for model in models:
        model_root = root / model
        mv_manifest, mv_predictions = collect_predictions(
            model_root, "mathvision", allowed_max_tokens
        )
        dm_manifest, dm_predictions = collect_predictions(
            model_root, "dynamath", allowed_max_tokens
        )
        mv_details, mv_accuracy = score_mathvision(mv_manifest, mv_predictions, mathvision_module)
        dm_details, dm_accuracy = score_dynamath(dm_manifest, dm_predictions, dynamath_module)
        write_jsonl(model_root / "mathvision" / "details.jsonl", mv_details)
        write_jsonl(model_root / "dynamath" / "details.jsonl", dm_details)
        (model_root / "mathvision" / "accuracy.json").write_text(
            json.dumps(mv_accuracy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (model_root / "dynamath" / "accuracy.json").write_text(
            json.dumps(dm_accuracy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report["models"][model] = {"mathvision": mv_accuracy, "dynamath": dm_accuracy}
        (model_root / "evaluation.complete").write_text("ok\n", encoding="utf-8")
    if {"base", "beta4"}.issubset(report["models"]):
        report["beta4_minus_base"] = {
            benchmark: subtract(
                report["models"]["beta4"][benchmark], report["models"]["base"][benchmark]
            )
            for benchmark in ("mathvision", "dynamath")
        }
    if {"base", "vision_opd"}.issubset(report["models"]):
        report["vision_opd_minus_base"] = {
            benchmark: subtract(
                report["models"]["vision_opd"][benchmark], report["models"]["base"][benchmark]
            )
            for benchmark in ("mathvision", "dynamath")
        }
    if {"beta4", "vision_opd"}.issubset(report["models"]):
        report["beta4_minus_vision_opd"] = {
            benchmark: subtract(
                report["models"]["beta4"][benchmark], report["models"]["vision_opd"][benchmark]
            )
            for benchmark in ("mathvision", "dynamath")
        }
    (root / "score_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "evaluation.complete").write_text("ok\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
