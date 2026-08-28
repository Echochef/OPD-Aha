#!/usr/bin/env python3
"""Run and deterministically score Qwen3-VL on the full WeMath testmini split.

The canonical parser reproduces VLMEvalKit's WeMath exact-matching path.  The
robust parser is a judge-free audit that accepts common explicit final-answer
formats while still failing closed when no answer marker can be found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import qwen3vl_mathverse_termination as shared


EXPECTED_ROWS = 1740
EXPECTED_GROUPS = 525
EXPECTED_KEYS = {
    "2steps_1": 360,
    "2steps_2": 360,
    "2steps_multi": 360,
    "3steps_1": 165,
    "3steps_2": 165,
    "3steps_3": 165,
    "3steps_multi": 165,
}
LETTERS = "ABCDEFGH"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - used only for dataset provenance
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    raise ValueError("WeMath row has no embedded image bytes")


def parse_options(value: str) -> list[tuple[str, str]]:
    text = str(value).strip()
    matches = list(re.finditer(r"(?:^|[;；]\s*)([A-H])[.．、]\s*", text))
    options: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        answer = text[start:end].strip().rstrip(";").strip()
        if not answer:
            raise ValueError(f"empty WeMath option in {value!r}")
        options.append((match.group(1), answer))
    labels = [letter for letter, _ in options]
    if len(options) < 2 or labels != sorted(set(labels)):
        raise ValueError(f"unable to parse ordered WeMath options: {value!r}")
    return options


def build_prompt(question: str, options: list[tuple[str, str]], hint: Any = None) -> str:
    prompt = ""
    if hint is not None and str(hint).strip() and str(hint).lower() != "nan":
        prompt += f"Hint: {str(hint).strip()}\n"
    prompt += f"Question: {str(question).strip()}\nOptions:\n"
    prompt += "".join(f"{letter}. {answer}\n" for letter, answer in options)
    return prompt


def command_prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    source = Path(args.wemath_parquet).resolve()
    output = Path(args.output_root).resolve()
    rows = pq.read_table(source).to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} WeMath testmini rows, found {len(rows)}")

    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        blob = image_bytes(row["image_path"])
        digest = hashlib.sha256(blob).hexdigest()
        image_path = image_dir / f"{digest}.png"
        if not image_path.is_file():
            image_path.write_bytes(blob)
        options = parse_options(str(row["option"]))
        gold = str(row["answer"]).strip().upper()
        if gold not in {letter for letter, _ in options}:
            raise ValueError(f"gold {gold!r} is not present in options at row {ordinal}")
        manifest.append(
            {
                "sample_uid": f"wemath:{ordinal:04d}",
                "ordinal": ordinal,
                "ID": str(row["ID"]),
                "key": str(row["key"]),
                "question_number": str(row["question number"]),
                "knowledge_concept": str(row["knowledge concept"]),
                "gold": gold,
                "choices": dict(options),
                "question": build_prompt(row["question"], options, row.get("hint")),
                "image_path": str(image_path),
            }
        )

    key_counts = Counter(row["key"] for row in manifest)
    if dict(key_counts) != EXPECTED_KEYS:
        raise ValueError(f"unexpected WeMath key counts: {dict(key_counts)}")
    if len({row["sample_uid"] for row in manifest}) != len(manifest):
        raise ValueError("duplicate WeMath sample_uid")
    if len({row["ID"] for row in manifest}) != EXPECTED_GROUPS:
        raise ValueError("WeMath must contain exactly 525 linked question groups")

    shared.write_jsonl(output / "manifest.jsonl", manifest)
    summary = {
        "dataset": "WeMath/testmini",
        "huggingface_dataset": "We-Math/We-Math",
        "huggingface_revision": "527c44d4d94c4e3c7c98157460146a8b18c8420a",
        "source": str(source),
        "source_md5": file_md5(source),
        "rows": len(manifest),
        "groups": len({row["ID"] for row in manifest}),
        "key_counts": dict(key_counts),
        "system_prompt": shared.SYSTEM_PROMPT,
        "prompt_scheme": "VLMEvalKit WeMath (non-COT): Question + Options",
        "scoring": "judge-free canonical exact parser plus robust deterministic audit parser",
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
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def canonical_answer(text: str) -> str | None:
    """Reproduce VLMEvalKit WeMath's built-in no-judge parser."""
    processed = str(text).split("Answer")[-1].strip()
    processed = re.sub(r"[>><<:.]", "", processed).strip()
    return processed[0] if processed and processed[0] in LETTERS else None


ROBUST_PATTERNS = (
    re.compile(
        r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?([A-H])(?=\s|[.}])",
        re.IGNORECASE,
    ),
    re.compile(r"<\s*answer\s*>\s*[:=]?\s*[*_`~>\(\[\{<]*([A-H])\b", re.IGNORECASE),
    re.compile(
        r"\b(?:final\s+answer|correct\s+answer|correct\s+option|correct\s+choice|answer)\s*"
        r"(?:(?:is|should\s+be|would\s+be)\s*)?[:=]?\s*(?:option\s*)?"
        r"[\s*_`~>\(\[\{<$✅✔️]*([A-H])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:choose|select|option)\s*(?:is|:|=)?\s*[*_`~>\(\[\{<]*([A-H])\b",
        re.IGNORECASE,
    ),
)


def robust_answer(text: str) -> str | None:
    candidates: list[tuple[int, str]] = []
    for pattern in ROBUST_PATTERNS:
        candidates.extend((match.start(), match.group(1).upper()) for match in pattern.finditer(str(text)))
    for match in re.finditer(r"(?m)^\s*[\(\[<]?([A-H])[\)\]>]?[\.!]?\s*$", str(text), re.IGNORECASE):
        candidates.append((match.start(), match.group(1).upper()))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def vlmevalkit_exact_answer(text: str, choices: dict[str, str]) -> str | None:
    """Reproduce VLMEvalKit's judge-free can_infer policy (commit 2b3f32d)."""
    answer = str(text)
    reject = (
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    )
    if any(message in answer for message in reject):
        return "Z"
    answer_mod = answer
    for char in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(char, " ")
    splits = [part.strip() for part in answer_mod.split()]
    present = [choice for choice in choices if choice in splits]
    if len(present) == 1 and splits.index(present[0]) > len(splits) - 5:
        return present[0]
    if not present and "Z" in splits:
        return "Z"
    verbose = re.search(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**", answer)
    if verbose and verbose.group(1).upper() in choices:
        return verbose.group(1).upper()
    lowered = answer.lower()
    values = {key: str(value).lower() for key, value in choices.items()}
    if len(lowered) <= 2 * sum(len(value) for value in values.values()):
        matching = [key for key, value in values.items() if value in lowered]
        if len(matching) == 1:
            return matching[0]
    return None


def choices_from_prompt(prompt: str) -> dict[str, str]:
    choices = {
        match.group(1): match.group(2).strip()
        for match in re.finditer(r"(?m)^([A-H])\.\s+(.*)$", str(prompt))
    }
    if len(choices) < 2:
        raise ValueError("unable to recover choices from prepared WeMath prompt")
    return choices


def score_rows(
    rows: list[dict[str, Any]], parser: Callable[[dict[str, Any]], str | None]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    groups: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        prediction = parser(row)
        hit = prediction == row["gold"]
        groups[row["ID"]][row["key"]] = hit
        details.append(
            {
                "sample_uid": row["sample_uid"],
                "ordinal": row["ordinal"],
                "ID": row["ID"],
                "key": row["key"],
                "gold": row["gold"],
                "prediction": prediction,
                "hit": hit,
            }
        )

    if len(groups) != EXPECTED_GROUPS:
        raise ValueError(f"expected {EXPECTED_GROUPS} groups, found {len(groups)}")
    counts = Counter()
    component_hits: list[bool] = []
    step2_hits: list[bool] = []
    step3_hits: list[bool] = []
    for group_id, values in groups.items():
        if "2steps_multi" in values:
            expected = {"2steps_1", "2steps_2", "2steps_multi"}
            component = [values["2steps_1"], values["2steps_2"]]
            multi = values["2steps_multi"]
            step2_hits.append(multi)
        elif "3steps_multi" in values:
            expected = {"3steps_1", "3steps_2", "3steps_3", "3steps_multi"}
            component = [values["3steps_1"], values["3steps_2"], values["3steps_3"]]
            multi = values["3steps_multi"]
            step3_hits.append(multi)
        else:
            raise ValueError(f"group {group_id!r} lacks a multi-step row")
        if set(values) != expected:
            raise ValueError(f"group {group_id!r} has keys {sorted(values)}, expected {sorted(expected)}")
        component_hits.extend(component)
        all_components = all(component)
        any_component = any(component)
        if all_components and not multi:
            counts["InadequateGeneralization"] += 1
        if not all_components and not multi:
            counts["InsufficientKnowledge"] += 1
        if all_components and multi:
            counts["CompleteMastery_strict"] += 1
        if any_component and multi:
            counts["CompleteMastery_loose"] += 1
        if not all_components and multi:
            counts["RoteMemorization_strict"] += 1
        if not any_component and multi:
            counts["RoteMemorization_loose"] += 1

    strict = (
        EXPECTED_GROUPS
        - 0.5 * counts["InadequateGeneralization"]
        - counts["RoteMemorization_strict"]
        - counts["InsufficientKnowledge"]
    ) / EXPECTED_GROUPS
    loose = (
        EXPECTED_GROUPS
        - 0.5 * counts["InadequateGeneralization"]
        - counts["RoteMemorization_loose"]
        - counts["InsufficientKnowledge"]
    ) / EXPECTED_GROUPS
    parsed = sum(item["prediction"] is not None for item in details)
    hits = sum(item["hit"] for item in details)
    report = {
        "rows": len(details),
        "parsed_rows": parsed,
        "parse_failures": len(details) - parsed,
        "row_accuracy": hits / len(details),
        "score_strict": strict,
        "score_loose": loose,
        "one_step_S1": sum(component_hits) / len(component_hits),
        "two_step_S2": sum(step2_hits) / len(step2_hits),
        "three_step_S3": sum(step3_hits) / len(step3_hits),
        "counts": dict(counts),
    }
    return report, details


def command_summarize(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve()
    manifest = shared.read_jsonl(output / "manifest.jsonl")
    raw_paths = (
        [output / "raw.jsonl"]
        if (output / "raw.jsonl").is_file()
        else sorted((output / "raw").glob("*.jsonl"))
    )
    records = [row for path in raw_paths for row in shared.read_jsonl(path)]
    by_uid = {row["sample_uid"]: row for row in records}
    missing = [row["sample_uid"] for row in manifest if row["sample_uid"] not in by_uid]
    if missing:
        raise ValueError(f"missing {len(missing)} inference rows; first={missing[:3]}")
    rows = [by_uid[row["sample_uid"]] for row in manifest]
    errors = [row for row in rows if row.get("finish_reason") == "error"]
    if errors:
        raise ValueError(f"inference contains {len(errors)} error rows")

    manifest_by_uid = {row["sample_uid"]: row for row in manifest}
    for row in rows:
        prepared = manifest_by_uid[row["sample_uid"]]
        row["choices"] = prepared.get("choices") or choices_from_prompt(prepared["question"])

    official_exact, official_details = score_rows(
        rows,
        lambda row: vlmevalkit_exact_answer(str(row.get("model_answer", "")), row["choices"]),
    )
    canonical, canonical_details = score_rows(
        rows, lambda row: canonical_answer(str(row.get("model_answer", "")))
    )
    robust, robust_details = score_rows(
        rows, lambda row: robust_answer(str(row.get("model_answer", "")))
    )
    official_by_uid = {row["sample_uid"]: row for row in official_details}
    canonical_by_uid = {row["sample_uid"]: row for row in canonical_details}
    robust_by_uid = {row["sample_uid"]: row for row in robust_details}
    details = []
    for row in rows:
        canonical_item = canonical_by_uid[row["sample_uid"]]
        robust_item = robust_by_uid[row["sample_uid"]]
        official_item = official_by_uid[row["sample_uid"]]
        details.append(
            {
                **{
                    key: value
                    for key, value in canonical_item.items()
                    if key not in {"prediction", "hit"}
                },
                "canonical_prediction": canonical_item["prediction"],
                "canonical_hit": canonical_item["hit"],
                "official_exact_prediction": official_item["prediction"],
                "official_exact_hit": official_item["hit"],
                "robust_prediction": robust_item["prediction"],
                "robust_hit": robust_item["hit"],
                "parser_disagreement": canonical_item["prediction"] != robust_item["prediction"],
                "finish_reason": row.get("finish_reason"),
                "completion_tokens": row.get("completion_tokens"),
                "model_answer": row.get("model_answer", ""),
            }
        )

    finish = Counter(row.get("finish_reason", "missing") for row in rows)
    tokens = [row["completion_tokens"] for row in rows if isinstance(row.get("completion_tokens"), int)]
    report = {
        "complete": len(rows) == EXPECTED_ROWS,
        "dataset": "WeMath/testmini",
        "model": rows[0].get("model_id") if rows else None,
        "judge": None,
        "judge_free": True,
        "official_vlmevalkit_exact_matching": official_exact,
        "wemath_builtin_canonical_parser": canonical,
        "robust_deterministic_matching": robust,
        "parser_disagreements": sum(item["parser_disagreement"] for item in details),
        "finish_reasons": dict(finish),
        "truncation_rate": finish.get("length", 0) / len(rows),
        "completion_tokens": {
            "mean": statistics.mean(tokens) if tokens else None,
            "median": statistics.median(tokens) if tokens else None,
            "min": min(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
    }
    shared.write_jsonl(output / "scored_details.jsonl", details)
    (output / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_selftest(_: argparse.Namespace) -> None:
    cases = {
        "Answer: B": ("B", "B"),
        "Reasoning. Final answer is (C).": (None, "C"),
        "Reasoning. ✅ Final Answer: **C. 2**": (None, "C"),
        "The correct answer is: **B. Green**": (None, "B"),
        "### Final Answer:\n$$\n\\boxed{\\text{A. } 16\\pi}\n$$": (None, "A"),
        "### Final Answer:\n✅ **D. 72°**": (None, "D"),
        "Reasoning\n\\boxed{D}": (None, "D"),
        "No explicit choice": (None, None),
        "Answer choices differ.\nAnswer: A": ("A", "A"),
    }
    for text, expected in cases.items():
        actual = (canonical_answer(text), robust_answer(text))
        if actual != expected:
            raise AssertionError(f"parser mismatch for {text!r}: {actual} != {expected}")
    choices = {"A": "one", "B": "two", "C": "three", "D": "four", "E": "No correct answer"}
    if vlmevalkit_exact_answer("Reasoning. The answer is B.", choices) != "B":
        raise AssertionError("VLMEvalKit exact matcher reproduction failed")
    parsed = parse_options("A. one; B. two; C. x; y; D. four; E. No correct answer")
    if parsed != [("A", "one"), ("B", "two"), ("C", "x; y"), ("D", "four"), ("E", "No correct answer")]:
        raise AssertionError(parsed)
    unusual = parse_options("A. one; B．two; C. three; F. No correct answer")
    if [letter for letter, _ in unusual] != ["A", "B", "C", "F"]:
        raise AssertionError(unusual)
    print("selftest: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--wemath-parquet", required=True)
    prepare.add_argument("--output-root", required=True)
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
    infer.set_defaults(func=shared.command_infer)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", required=True)
    summarize.set_defaults(func=command_summarize)
    selftest = subparsers.add_parser("selftest")
    selftest.set_defaults(func=command_selftest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
