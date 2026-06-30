#!/usr/bin/env python3
"""Re-score Mistral QQP from saved responses after the leading-label parser fix.

Mistral frequently answered ``"Not_duplicate. <explanation...>"``. The original
run failed to parse the leading label and sent these responses to the LLM judge,
which sometimes mislabeled them as ``duplicate``. The fixed
``_parse_direct_label_only`` now reads the leading label directly.

This script re-parses every stored ``model_response`` with the corrected direct
parser, recomputes pass/fail with CheckList's own expectation functions, and
rewrites the run's ``examples_full`` jsonl and ``checklist_summary.json`` in
place. Originals are backed up with a ``.pre_parserfix`` suffix. CPU-only; no GPU
rerun is required.

Usage:
    python scripts/utils/rescore_mistral_qqp.py
    python scripts/utils/rescore_mistral_qqp.py --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts" / "utils"))

from dlm_safety.llm_judge import _parse_direct_label_only
from dlm_safety.checklist_suite_io import build_suite_from_exported_json, iter_suite_tests
from rescore_checklist import (
    _load_traces,
    _group_traces_by_test,
    _preds_confs_for_test,
    _flattened_count,
    _test_case_count,
    _json_safe,
)

RUN_DIR = project_root / "results/expansion/checklist/qqp/mistral"
MODEL = "mistralai_Mistral-7B-Instruct-v0.3"
SUITE_JSON = project_root / "results/checklist/suites/qqp_suite_original_dump.json"
LABELS = ("not_duplicate", "duplicate")
REPORT_PATH = project_root / "results/expansion/checklist/qqp/mistral/rescore_parserfix_report.json"


def reparse_rows(rows: list[dict]) -> tuple[int, int]:
    """Re-parse model_response with the fixed direct parser, in place.

    Returns (n_now_direct, n_label_changed): how many rows now parse directly
    (overriding the judge result) and how many of those changed the label.
    """
    n_now_direct = 0
    n_changed = 0
    for row in rows:
        resp = row.get("model_response") or ""
        idx, mode = _parse_direct_label_only(resp, LABELS)
        if idx is None:
            # Still needs the judge; keep the stored prediction untouched.
            continue
        n_now_direct += 1
        old_idx = row.get("predicted_label_idx")
        if idx != old_idx:
            n_changed += 1
        row["predicted_label_idx"] = idx
        row["predicted_label"] = LABELS[idx]
        row["probabilities"] = [1.0, 0.0] if idx == 0 else [0.0, 1.0]
        row["confidence"] = 1.0
        row["raw_prediction"] = idx
        row["parse_mode"] = mode
        # These came from the judge path before; clear stale judge artifacts.
        row["judge_output"] = None
    return n_now_direct, n_changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Recompute and report, but do not write any files.")
    args = ap.parse_args()

    trace_path = RUN_DIR / f"examples_full_{MODEL}.jsonl"
    summary_path = RUN_DIR / "checklist_summary.json"

    rows = _load_traces(trace_path)
    print(f"Loaded {len(rows)} trace rows from {trace_path.name}")

    n_direct, n_changed = reparse_rows(rows)
    print(f"Re-parsed: {n_direct} rows now parse directly "
          f"({n_changed} changed label vs the stored judge result)")

    grouped = _group_traces_by_test(rows)
    suite = build_suite_from_exported_json("qqp", SUITE_JSON)
    suite_tests = list(iter_suite_tests(suite))

    new_test_stats: dict[str, dict] = {}
    total_cases = 0
    total_fails = 0

    for test_name, test in suite_tests:
        trace_rows = grouped.get(test_name, [])
        flat_n = _flattened_count(test)
        if len(trace_rows) != flat_n:
            raise RuntimeError(
                f"count mismatch for {test_name!r}: suite={flat_n} trace={len(trace_rows)}"
            )

        examples, result_indexes = test.example_list_and_indices()
        preds, confs = _preds_confs_for_test("qqp", test_name, test, trace_rows)
        test.run_from_preds_confs(preds, confs, overwrite=True)

        passed = np.asarray(test.results.passed)
        fail_idxs = set(int(i) for i in test.fail_idxs())
        n_cases = _test_case_count(test)
        n_fails = len(fail_idxs)
        fail_rate = float(n_fails) / float(n_cases) if n_cases else None

        # Push per-case pass back onto each flattened row for consistency.
        for row, case_idx in zip(trace_rows, result_indexes):
            row_pass = bool(passed[case_idx])
            row["pass"] = row_pass
            row["expectation_score"] = row_pass
            row["test_n_fails"] = n_fails
            row["test_fail_rate"] = fail_rate
            row["test_n_cases"] = n_cases

        label_counts = Counter(LABELS[int(r["predicted_label_idx"])] for r in trace_rows)
        new_test_stats[test_name] = {
            "n_cases": n_cases,
            "n_fails": n_fails,
            "fail_rate": fail_rate,
            "predicted_label_counts": {lab: label_counts.get(lab, 0) for lab in LABELS},
        }
        total_cases += n_cases
        total_fails += n_fails

    # Load existing summary for old values + to update in place.
    summary_full = json.load(open(summary_path, encoding="utf-8"))
    model_block = summary_full[MODEL]
    old_total_fails = model_block["total_fails"]
    old_total_cases = model_block["total_cases"]

    print(f"\n{'Test':<70s} {'Type':>4s} {'Old%':>7s} {'New%':>7s} {'Δpp':>7s}")
    print("-" * 100)
    diff_report = []
    for test in model_block["tests"]:
        name = test["name"]
        new = new_test_stats.get(name)
        if new is None:
            continue
        old_fr = test.get("fail_rate")
        new_fr = new["fail_rate"]
        old_pct = old_fr * 100 if old_fr is not None else None
        new_pct = new_fr * 100 if new_fr is not None else None
        changed = old_fr != new_fr
        mark = " <-" if changed else ""
        op = f"{old_pct:.1f}" if old_pct is not None else "?"
        npc = f"{new_pct:.1f}" if new_pct is not None else "?"
        dl = f"{new_pct - old_pct:+.1f}" if (old_pct is not None and new_pct is not None) else "?"
        if changed:
            print(f"{name:<70.70s} {test['type']:>4s} {op:>7s} {npc:>7s} {dl:>7s}{mark}")
        diff_report.append({
            "name": name, "type": test["type"],
            "old_n_fails": test.get("n_fails"), "new_n_fails": new["n_fails"],
            "old_fail_rate": old_fr, "new_fail_rate": new_fr,
        })
        # Update the summary entry in place.
        test["n_fails"] = new["n_fails"]
        test["fail_rate"] = new["fail_rate"]
        test["predicted_label_counts"] = new["predicted_label_counts"]
        test["examples"] = grouped.get(name, test.get("examples"))

    overall_old = old_total_fails / old_total_cases if old_total_cases else None
    overall_new = total_fails / total_cases if total_cases else None
    print("-" * 100)
    print(f"TOTAL fails: {old_total_fails} -> {total_fails}   "
          f"cases: {old_total_cases} -> {total_cases}")
    print(f"Overall fail rate: {overall_old*100:.2f}% -> {overall_new*100:.2f}%")

    # Refresh judge fallback stats (some judge cases now resolve via direct parse).
    judge_fallback = len(rows) - n_direct
    model_block["total_fails"] = total_fails
    model_block["total_cases"] = total_cases
    model_block["judge_fallback_stats"] = {
        "total_examples": len(rows),
        "direct_parse_count": n_direct,
        "judge_fallback_count": judge_fallback,
        "judge_repair_count": 0,
    }

    report = {
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "leading-label parser fix (Mistral 'Not_duplicate. <prose>' responses)",
        "rows_now_direct_parse": n_direct,
        "rows_label_changed": n_changed,
        "old_total_fails": old_total_fails,
        "new_total_fails": total_fails,
        "old_overall_fail_rate": overall_old,
        "new_overall_fail_rate": overall_new,
        "per_test": diff_report,
    }

    if args.dry_run:
        print("\nDRY RUN — no files written.")
        return

    # Back up originals once, then write corrected files.
    for p in (trace_path, summary_path):
        bak = p.with_suffix(p.suffix + ".pre_parserfix")
        if not bak.exists():
            bak.write_bytes(p.read_bytes())
            print(f"Backed up {p.name} -> {bak.name}")

    with trace_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary_full), f, ensure_ascii=False)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, ensure_ascii=False)
    print(f"\nWrote corrected {trace_path.name}, {summary_path.name}, and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
