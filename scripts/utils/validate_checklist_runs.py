#!/usr/bin/env python3
"""Validate post-fix CheckList GPU run directories (artifacts, metadata, suite totals).

Exits with code 1 if any check fails. Intended for CI or post-rsync verification.

Usage:
    python scripts/validate_checklist_runs.py
    python scripts/validate_checklist_runs.py --root /path/to/DLM_posix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPECTED = {
    "sentiment": {"tests": 38, "n_cases": 86420},
    "qqp": {"tests": 53, "n_cases": 63216},
    "squad": {"tests": 24, "n_cases": 13311},
}


def _model_id(task: str, model: str) -> str:
    if model == "llada":
        return "GSAI-ML_LLaDA-8B-Instruct"
    return "meta-llama_Llama-3.1-8B-Instruct"


def default_runs(root: Path) -> list[tuple[str, str, Path]]:
    """Task, model key, directory."""
    r = root / "results" / "checklist"
    return [
        ("sentiment", "llada", r / "sentiment" / "llada_rerun_fixed"),
        ("sentiment", "llama", r / "sentiment" / "llama_rerun_fixed"),
        ("qqp", "llada", r / "qqp" / "llada_rerun_fixed"),
        ("qqp", "llama", r / "qqp" / "llama_rerun_fixed"),
        ("squad", "llada", r / "squad" / "llada_rerun_fixed"),
        ("squad", "llama", r / "squad" / "llama_rerun_fixed"),
    ]


def validate_run(task: str, model: str, d: Path) -> list[str]:
    errors: list[str] = []
    tag = f"{task}/{model}"
    if not d.is_dir():
        return [f"{tag}: missing directory {d}"]

    mid = _model_id(task, model)
    files = {
        "checklist_summary.json",
        f"checklist_results_{mid}.json",
        f"examples_full_{mid}.jsonl",
        f"tests_full_{mid}.json",
        f"run_metadata_{mid}.json",
        f"suite_visual_snapshot_{mid}.json",
    }
    for name in files:
        if not (d / name).exists():
            errors.append(f"{tag}: missing {name}")

    if errors:
        return errors

    meta = json.loads((d / f"run_metadata_{mid}.json").read_text(encoding="utf-8"))
    if meta.get("demo_limit") is not None:
        errors.append(f"{tag}: expected demo_limit null, got {meta.get('demo_limit')!r}")

    suite = str(meta.get("suite_path", ""))
    if "suite_original_dump.json" not in suite.replace("\\", "/"):
        errors.append(f"{tag}: suite_path should reference *_suite_original_dump.json, got {suite!r}")

    summ = json.loads((d / "checklist_summary.json").read_text(encoding="utf-8"))
    keys = [k for k, v in summ.items() if isinstance(v, dict) and "tests" in v]
    if len(keys) != 1:
        errors.append(f"{tag}: expected one model key in checklist_summary, got {keys!r}")
    else:
        tests = summ[keys[0]]["tests"]
        exp = EXPECTED[task]
        nt = len(tests)
        nc = sum(t.get("n_cases", 0) for t in tests)
        if nt != exp["tests"]:
            errors.append(f"{tag}: test count {nt} != expected {exp['tests']}")
        if nc != exp["n_cases"]:
            errors.append(f"{tag}: sum(n_cases)={nc} != expected {exp['n_cases']}")

    tf = json.loads((d / f"tests_full_{mid}.json").read_text(encoding="utf-8"))
    tlist = tf.get("tests")
    if not isinstance(tlist, list):
        errors.append(f"{tag}: tests_full missing tests list")
    else:
        sum_ex = sum(len(t.get("examples", [])) for t in tlist)
        nlines = sum(1 for _ in (d / f"examples_full_{mid}.jsonl").open("rb"))
        if sum_ex != nlines:
            errors.append(
                f"{tag}: examples jsonl lines {nlines} != sum(len(examples)) in tests_full {sum_ex}"
            )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate CheckList rerun output directories.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Project root (default: repo root)",
    )
    args = ap.parse_args()
    root: Path = args.root

    all_errors: list[str] = []
    for task, model, d in default_runs(root):
        all_errors.extend(validate_run(task, model, d))

    if all_errors:
        print("VALIDATION FAILED:\n", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("All 6 CheckList run directories passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
