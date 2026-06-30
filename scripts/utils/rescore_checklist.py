#!/usr/bin/env python3
"""Re-score existing CheckList runs with corrected expectation functions.

Reads stored prediction traces from prior GPU runs, rebuilds suites with
fixed expectations (MFT_EXPECT_REGISTRY, INV_EXPECT_REGISTRY, updated DIR
expects), and re-evaluates pass/fail without requiring GPU.

Usage:
    python scripts/rescore_checklist.py
    python scripts/rescore_checklist.py --task sentiment
    python scripts/rescore_checklist.py --dry-run
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dlm_safety.checklist_suite_io import build_suite_from_exported_json, iter_suite_tests


RUNS = [
    {
        "task": "sentiment",
        "suite_json": "results/checklist/suites/sentiment_suite_original_dump.json",
        "runs": [
            {
                "model": "GSAI-ML_LLaDA-8B-Instruct",
                "run_dir": "results/checklist/sentiment/llada_rerun_fixed",
            },
            {
                "model": "meta-llama_Llama-3.1-8B-Instruct",
                "run_dir": "results/checklist/sentiment/llama_rerun_fixed",
            },
        ],
    },
    {
        "task": "qqp",
        "suite_json": "results/checklist/suites/qqp_suite_original_dump.json",
        "runs": [
            {
                "model": "GSAI-ML_LLaDA-8B-Instruct",
                "run_dir": "results/checklist/qqp/llada_rerun_fixed",
            },
            {
                "model": "meta-llama_Llama-3.1-8B-Instruct",
                "run_dir": "results/checklist/qqp/llama_rerun_fixed",
            },
        ],
    },
    {
        "task": "squad",
        "suite_json": "results/checklist/suites/squad_suite_original_dump.json",
        "runs": [
            {
                "model": "GSAI-ML_LLaDA-8B-Instruct",
                "run_dir": "results/checklist/squad/llada_rerun_fixed",
            },
            {
                "model": "meta-llama_Llama-3.1-8B-Instruct",
                "run_dir": "results/checklist/squad/llama_rerun_fixed",
            },
        ],
    },
]


def _load_traces(trace_path: Path) -> list[dict]:
    rows = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_traces_by_test(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        test_name = row.get("test_name")
        if test_name:
            grouped[test_name].append(row)
    return dict(grouped)


def _preds_confs_for_test(task: str, test_name: str, test, trace_rows: list[dict]):
    """Extract (preds, confs) lists from trace rows for a given test.

    Returns arrays matching the flattened example count that
    test.example_list_and_indices() produces.
    """
    if task == "sentiment":
        preds = []
        confs = []
        for row in trace_rows:
            pred_idx = row.get("predicted_label_idx")
            probs = row.get("probabilities")
            if pred_idx is None or probs is None:
                pred_idx = 1
                probs = [0.0, 1.0, 0.0]
            preds.append(int(pred_idx))
            confs.append(np.array(probs, dtype=float))
        return preds, confs

    if task == "qqp":
        preds = []
        confs = []
        for row in trace_rows:
            pred_idx = row.get("predicted_label_idx")
            probs = row.get("probabilities")
            if pred_idx is None or probs is None:
                pred_idx = 0
                probs = [1.0, 0.0]
            preds.append(int(pred_idx))
            confs.append(np.array(probs, dtype=float))
        return preds, confs

    if task == "squad":
        preds = []
        confs = []
        for row in trace_rows:
            pred = row.get("predicted_label") or row.get("model_response") or ""
            preds.append(str(pred))
            conf = row.get("confidence")
            if conf is None:
                conf = 1.0
            confs.append(conf)
        return preds, confs

    raise ValueError(f"Unsupported task: {task}")


def _test_case_count(test) -> int:
    return (
        getattr(test, "n", None)
        or (len(test.data) if getattr(test, "data", None) is not None else None)
        or getattr(test, "data_len", None)
        or 0
    )


def _flattened_count(test) -> int:
    """Count total flattened examples (accounts for grouped data)."""
    data = getattr(test, "data", None) or []
    return _flat_count_for_data(data)


def _flat_count_for_data(data: list) -> int:
    if not data:
        return 0
    if isinstance(data[0], (list, np.ndarray)):
        return sum(len(group) for group in data)
    return len(data)


def _truncate_test_to_flat_count(test, target_flat: int) -> None:
    """Slice test.data (and test.labels) so flattened count == target_flat.

    The original runs used demo_limit which sliced test.data[:N].  We
    reproduce that effect so example_list_and_indices produces the right
    result_indexes for the stored predictions.
    """
    data = getattr(test, "data", None) or []
    if not data:
        return
    is_grouped = isinstance(data[0], (list, np.ndarray))

    if is_grouped:
        cumulative = 0
        cut = 0
        for i, group in enumerate(data):
            cumulative += len(group)
            if cumulative >= target_flat:
                cut = i + 1
                break
        if cumulative != target_flat and cut > 0:
            cut = max(1, cut)
        test.data = data[:cut]
    else:
        test.data = data[:target_flat]

    labels = getattr(test, "labels", None)
    if isinstance(labels, list):
        test.labels = labels[:len(test.data)]
    meta = getattr(test, "meta", None)
    if isinstance(meta, list):
        test.meta = meta[:len(test.data)]


def _extract_test_results(test) -> dict:
    """Extract pass/fail stats from a scored test using CheckList's own counting."""
    n_cases = _test_case_count(test)
    results_obj = getattr(test, "results", None)
    if results_obj is None:
        return {"n_cases": n_cases, "n_fails": None, "fail_rate": None}

    try:
        n_fails = len(test.fail_idxs())
    except Exception:
        passed = getattr(results_obj, "passed", None)
        if passed is not None:
            try:
                n_fails = int(np.sum(passed == False))
            except Exception:
                n_fails = None
        else:
            n_fails = None

    fail_rate = float(n_fails) / float(n_cases) if n_fails is not None and n_cases else None
    return {"n_cases": n_cases, "n_fails": n_fails, "fail_rate": fail_rate}


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def rescore_run(task: str, suite_json: Path, run_dir: Path, model: str,
                output_dir: Path, dry_run: bool = False) -> dict | None:
    """Re-score a single (task, model) run."""
    trace_path = run_dir / f"examples_full_{model}.jsonl"
    if not trace_path.exists():
        print(f"  SKIP: trace file not found: {trace_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Re-scoring: task={task}  model={model}")
    print(f"  Trace: {trace_path}")
    print(f"  Suite: {suite_json}")

    traces = _load_traces(trace_path)
    print(f"  Loaded {len(traces)} trace rows")

    grouped = _group_traces_by_test(traces)
    print(f"  Tests in trace: {len(grouped)}")

    suite = build_suite_from_exported_json(task, suite_json)

    suite_tests = list(iter_suite_tests(suite))
    print(f"  Tests in suite: {len(suite_tests)}")

    mismatches = []
    matched = 0
    for test_name, test in suite_tests:
        trace_rows = grouped.get(test_name, [])
        flat_n = _flattened_count(test)
        if len(trace_rows) != flat_n:
            mismatches.append((test_name, flat_n, len(trace_rows)))
        else:
            matched += 1

    if mismatches:
        print(f"\n  WARNING: {len(mismatches)} test(s) have count mismatches:")
        for name, expected, got in mismatches[:10]:
            print(f"    {name}: suite expects {expected}, trace has {got}")

    print(f"  Matched: {matched}/{len(suite_tests)} tests")

    if dry_run:
        print("  DRY RUN — skipping actual re-scoring")
        return None

    old_results: dict[str, dict] = {}
    for row in traces:
        tn = row.get("test_name")
        if tn and tn not in old_results:
            old_results[tn] = {
                "n_cases": row.get("test_n_cases"),
                "n_fails": row.get("test_n_fails"),
                "fail_rate": row.get("test_fail_rate"),
                "test_type": row.get("test_type"),
            }

    new_results: dict[str, dict] = {}
    total_cases = 0
    total_fails = 0

    for test_name, test in suite_tests:
        trace_rows = grouped.get(test_name, [])
        flat_n = _flattened_count(test)

        if len(trace_rows) == 0:
            print(f"  SKIP (no traces): {test_name}")
            new_results[test_name] = {"n_cases": flat_n, "n_fails": None,
                                      "fail_rate": None, "status": "skipped_no_traces"}
            continue

        if len(trace_rows) != flat_n:
            _truncate_test_to_flat_count(test, len(trace_rows))
            new_flat = _flattened_count(test)
            if new_flat != len(trace_rows):
                print(f"  WARN: could not align {test_name} "
                      f"(suite={flat_n}→{new_flat}, trace={len(trace_rows)})")
                trace_rows = trace_rows[:new_flat]

        test.example_list_and_indices()

        preds, confs = _preds_confs_for_test(task, test_name, test, trace_rows)

        test.run_from_preds_confs(preds, confs, overwrite=True)

        stats = _extract_test_results(test)
        stats["test_type"] = test.__class__.__name__
        new_results[test_name] = stats

        if stats["n_cases"] and stats["n_fails"] is not None:
            total_cases += stats["n_cases"]
            total_fails += stats["n_fails"]

    print(f"\n  --- Results for {model} on {task} ---")
    print(f"  {'Test':<70s} {'Type':>4s} {'Old%':>7s} {'New%':>7s} {'Δ':>7s}")
    print(f"  {'-'*96}")

    for test_name, test in suite_tests:
        old = old_results.get(test_name, {})
        new = new_results.get(test_name, {})
        old_fr = old.get("fail_rate")
        new_fr = new.get("fail_rate")
        old_pct = f"{old_fr*100:.1f}" if old_fr is not None else "?"
        new_pct = f"{new_fr*100:.1f}" if new_fr is not None else "?"
        test_type = new.get("test_type", old.get("test_type", "?"))
        if old_fr is not None and new_fr is not None:
            delta = (new_fr - old_fr) * 100
            delta_str = f"{delta:+.1f}"
        else:
            delta_str = "?"
        changed = " ←" if old_pct != new_pct else ""
        print(f"  {test_name:<70s} {test_type:>4s} {old_pct:>7s} {new_pct:>7s} {delta_str:>7s}{changed}")

    overall_fr = float(total_fails) / float(total_cases) if total_cases else None
    print(f"\n  Overall: {total_fails}/{total_cases} fails "
          f"({overall_fr*100:.2f}%)" if overall_fr is not None else "")

    summary = {
        "task": task,
        "model": model,
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_trace": str(trace_path),
        "suite_json": str(suite_json),
        "total_cases": total_cases,
        "total_fails": total_fails,
        "overall_fail_rate": overall_fr,
        "tests": {},
    }
    for test_name, test in suite_tests:
        old = old_results.get(test_name, {})
        new = new_results.get(test_name, {})
        summary["tests"][test_name] = {
            "test_type": new.get("test_type", old.get("test_type")),
            "n_cases": new.get("n_cases"),
            "old_n_fails": old.get("n_fails"),
            "old_fail_rate": old.get("fail_rate"),
            "new_n_fails": new.get("n_fails"),
            "new_fail_rate": new.get("fail_rate"),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"rescored_{task}_{safe_model}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Re-score CheckList runs with corrected expectations (CPU-only)"
    )
    parser.add_argument("--task", choices=["sentiment", "qqp", "squad"],
                        default=None, help="Re-score only this task")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/checklist/rescored"),
                        help="Output directory for rescored results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify data alignment without re-scoring")
    args = parser.parse_args()

    all_summaries = []

    for task_block in RUNS:
        task = task_block["task"]
        if args.task and task != args.task:
            continue

        suite_json = project_root / task_block["suite_json"]
        if not suite_json.exists():
            print(f"Suite JSON not found: {suite_json}")
            continue

        for run_spec in task_block["runs"]:
            run_dir = project_root / run_spec["run_dir"]
            model = run_spec["model"]

            summary = rescore_run(
                task=task,
                suite_json=suite_json,
                run_dir=run_dir,
                model=model,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
            if summary:
                all_summaries.append(summary)

    if all_summaries and not args.dry_run:
        combined_path = args.output_dir / "rescore_summary.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(all_summaries), f, indent=2, ensure_ascii=False)
        print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
