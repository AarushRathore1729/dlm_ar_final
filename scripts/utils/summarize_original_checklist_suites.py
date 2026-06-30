#!/usr/bin/env python3
"""Summarize recovered original CheckList suite JSON exports."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SUITES = {
    "sentiment": RESULTS_DIR / "checklist" / "suites" / "sentiment_suite_original_dump.json",
    "qqp": RESULTS_DIR / "checklist" / "suites" / "qqp_suite_original_dump.json",
    "squad": RESULTS_DIR / "checklist" / "suites" / "squad_suite_original_dump.json",
}


def label_kind(value):
    if value is None:
        return "none"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def summarize_suite(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tests = payload.get("tests", [])
    suite_summary = {
        "path": str(path),
        "num_tests": len(tests),
        "total_grouped_cases": sum(test.get("size", 0) for test in tests),
        "tests": [],
    }
    for test in tests:
        examples = test.get("examples") or []
        kinds = {}
        for ex in examples[: min(20, len(examples))]:
            kind = label_kind(ex.get("label"))
            kinds[kind] = kinds.get(kind, 0) + 1
        suite_summary["tests"].append(
            {
                "name": test.get("name"),
                "size": test.get("size"),
                "capability": test.get("capability"),
                "sample_label_kinds": kinds,
            }
        )
    return suite_summary


def main() -> None:
    output = {"suites": {}}
    for task, path in SUITES.items():
        if not path.exists():
            output["suites"][task] = {"missing": True, "path": str(path)}
            continue
        output["suites"][task] = summarize_suite(path)

    out_path = RESULTS_DIR / "original_suite_summary.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
