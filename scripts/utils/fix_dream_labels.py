#!/usr/bin/env python3
"""Fix Dream label-parsing artifacts and rebuild checklist summaries.

Dream's masked diffusion generates the END of a label word with a leading period:
  entailment    -> '.ailment'    (drops first 3 chars)
  contradiction -> '.adiction'   (drops first 5 chars)
  not_duplicate -> '._duplicate' (replaces 'not' with '.')  or just 'not'
  duplicate     -> '._duplicate' -- also a truncation pattern; resolved by context

Corrected pass/fail is recomputed per test type:
  MFT: pass = (recovered_label == expected_label)
  INV: pairs are consecutive (even=original, odd=perturbed);
       pair fails iff recovered_label[2i] != recovered_label[2i+1]
  DIR: skipped (direction logic is complex; original pass field preserved)

Writes checklist_summary_fixed.json alongside originals (originals untouched).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Label recovery
# ---------------------------------------------------------------------------

NLI_LABELS    = {"entailment", "neutral", "contradiction"}
BINARY_LABELS = {"duplicate", "not_duplicate"}

_NLI_SUFFIXES = {
    "ailment":   "entailment",
    "adiction":  "contradiction",
    "ntral":     "neutral",
    "ntr":       "contradiction",
}


def _recover_nli(raw: str) -> str | None:
    s = raw.strip().lstrip(".''\"").lower().strip()
    if s in NLI_LABELS:
        return s
    for suffix, label in _NLI_SUFFIXES.items():
        if suffix in s:
            return label
    # prefix match (>= 3 chars)
    for label in NLI_LABELS:
        if len(s) >= 3 and label.startswith(s[:4]):
            return label
    return None


def _recover_binary(raw: str) -> str | None:
    s = raw.strip()
    if s in BINARY_LABELS:
        return s
    if "not_duplicate" in s:
        return "not_duplicate"
    # Dream: '._duplicate' = not_duplicate with 'not' -> '.'
    if s.startswith(".") or s.startswith("'"):
        tail = s.lstrip(".''\"")
        if tail.startswith("_duplicate"):
            return "not_duplicate"
        if tail in BINARY_LABELS:
            return tail
    # Dream 'not' (truncated not_duplicate before underscore)
    if s.lower() == "not":
        return "not_duplicate"
    if "not_duplicate" in s.lower():
        return "not_duplicate"
    if s.endswith("_duplicate") and "not" not in s.lower():
        return "duplicate"
    if "duplicate" in s.lower():
        return "duplicate"
    return None


def recover_label(raw: str, task: str) -> str | None:
    if task in ("anli", "mnli"):
        return _recover_nli(raw)
    if task in ("qqp", "paws"):
        return _recover_binary(raw)
    return None  # squad spans not recoverable


# ---------------------------------------------------------------------------
# Pass/fail recomputation
# ---------------------------------------------------------------------------

def _recompute_mft_pass(examples: list[dict], fixed_labels: list[str],
                        label_to_idx: dict[str, int]) -> list[bool]:
    """MFT: pass iff label_to_idx[predicted] == expected_label (int index)."""
    passes = []
    for ex, fl in zip(examples, fixed_labels):
        el = ex.get("expected_label")  # integer index
        pred_idx = label_to_idx.get(fl)
        passes.append(pred_idx is not None and pred_idx == el)
    return passes


def _recompute_inv_pass(examples: list[dict], fixed_labels: list[str]) -> list[bool]:
    """INV: consecutive pairs (even=original, odd=perturbed).
    Pair fails iff original_label != perturbed_label.
    Original always gets pass=True; perturbed gets pass=False on failure.
    """
    passes = [True] * len(fixed_labels)
    for i in range(0, len(fixed_labels) - 1, 2):
        if fixed_labels[i] != fixed_labels[i + 1]:
            passes[i + 1] = False  # perturbed fails; original stays True
    return passes


# ---------------------------------------------------------------------------
# Summary rebuilder
# ---------------------------------------------------------------------------

def rebuild_summary(jsonl_path: Path, task: str, model_key: str,
                    orig_summary_path: Path) -> dict:
    with orig_summary_path.open() as f:
        orig = json.load(f)
    orig_data = orig[model_key]

    # Build label->index map from run_metadata (labels field)
    meta_path = jsonl_path.parent.glob("run_metadata_*.json")
    label_to_idx: dict[str, int] = {}
    for mp in meta_path:
        md = json.load(open(mp))
        for i, lbl in enumerate(md.get("labels") or []):
            label_to_idx[str(lbl)] = i
        break

    # --- Read and recover labels ---
    all_rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    n_total = len(all_rows)
    n_fixed = 0
    recovered_labels: list[str] = []
    for row in all_rows:
        orig_label = row.get("predicted_label", "")
        raw = row.get("model_response", "")
        rec = recover_label(raw, task)
        if rec and rec != orig_label:
            recovered_labels.append(rec)
            n_fixed += 1
        else:
            recovered_labels.append(orig_label)

    print(f"    Recovered {n_fixed}/{n_total} labels ({100*n_fixed/n_total:.1f}%)")

    # --- Group by test name, preserving order ---
    test_order: list[str] = []
    by_test: dict[str, list[tuple[int, dict, str]]] = defaultdict(list)
    for idx, (row, fl) in enumerate(zip(all_rows, recovered_labels)):
        tn = row.get("test_name", "")
        if tn not in by_test:
            test_order.append(tn)
        by_test[tn].append((idx, row, fl))

    # --- Recompute per-test stats ---
    orig_test_map = {t["name"]: t for t in orig_data["tests"]}
    new_tests = []
    for orig_test in orig_data["tests"]:
        tn = orig_test["name"]
        test_type = orig_test["type"]  # MFT / INV / DIR
        rows_fl = by_test.get(tn, [])

        exs = [r for _, r, _ in rows_fl]
        fls = [fl for _, _, fl in rows_fl]
        n_rows = len(fls)
        n_cases = orig_test["n_cases"]  # keep original case count

        label_counts: Counter = Counter(fls)

        if test_type == "MFT":
            passes = _recompute_mft_pass(exs, fls, label_to_idx)
            n_fails = passes.count(False)

        elif test_type == "INV":
            passes = _recompute_inv_pass(exs, fls)
            n_fails = passes.count(False)
            # n_cases for INV = n_rows / 2
            n_cases = n_rows // 2

        else:  # DIR or other: keep original pass values
            orig_fails = orig_test.get("n_fails") or 0
            n_fails = orig_fails
            passes = None  # not recomputed

        fail_rate = n_fails / n_cases if n_cases else None

        new_test = dict(orig_test)
        new_test["n_cases"] = n_cases
        new_test["n_fails"] = n_fails
        new_test["fail_rate"] = fail_rate
        new_test["predicted_label_counts"] = dict(label_counts)
        new_tests.append(new_test)

    new_summary = {
        model_key: {
            "tests": new_tests,
            "_fix_note": (
                f"Labels recovered by fix_dream_labels.py; "
                f"{n_fixed}/{n_total} examples corrected."
            ),
        }
    }
    return new_summary


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS = [
    ("anli",  "dream",     "Dream-org_Dream-v0-Instruct-7B",
     "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
    ("mnli",  "dream",     "Dream-org_Dream-v0-Instruct-7B",
     "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
    ("qqp",   "dream",     "Dream-org_Dream-v0-Instruct-7B",
     "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
    ("paws",  "dream",     "Dream-org_Dream-v0-Instruct-7B",
     "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
]


def main():
    base = ROOT / "results" / "expansion" / "checklist"
    for task, model_dir, model_key, jsonl_name in TARGETS:
        run_dir = base / task / model_dir
        jsonl_path   = run_dir / jsonl_name
        summary_path = run_dir / "checklist_summary.json"

        if not jsonl_path.exists():
            print(f"SKIP (no JSONL): {task}/{model_dir}")
            continue
        if not summary_path.exists():
            print(f"SKIP (no summary): {task}/{model_dir}")
            continue

        print(f"\n=== {task} / {model_dir} ===")

        with summary_path.open() as f:
            orig = json.load(f)
        orig_tests = orig[model_key]["tests"]
        orig_total_cases = sum(t.get("n_cases") or 0 for t in orig_tests)
        orig_total_fails = sum(t.get("n_fails") or 0 for t in orig_tests)
        orig_rate = orig_total_fails / orig_total_cases if orig_total_cases else 0

        new_summary = rebuild_summary(jsonl_path, task, model_key, summary_path)
        new_tests = new_summary[model_key]["tests"]
        new_total_cases = sum(t.get("n_cases") or 0 for t in new_tests)
        new_total_fails = sum(t.get("n_fails") or 0 for t in new_tests)
        new_rate = new_total_fails / new_total_cases if new_total_cases else 0

        arrow = "↓" if new_rate < orig_rate else "↑"
        print(f"    Overall fail rate: {orig_rate*100:.1f}% -> {new_rate*100:.1f}%  "
              f"({arrow}{abs(new_rate-orig_rate)*100:.1f}pp)")

        print(f"    {'Test':<55s} {'Old%':>6s} {'New%':>6s} {'Δ':>7s}")
        for ot, nt in zip(orig_tests, new_tests):
            ofr = ot.get("fail_rate") or 0
            nfr = nt.get("fail_rate") or 0
            delta = (nfr - ofr) * 100
            marker = " ←" if abs(delta) > 2 else ""
            print(f"    {ot['name'][:55]:55s} {ofr*100:6.1f} {nfr*100:6.1f} {delta:+7.1f}{marker}")

        out_path = run_dir / "checklist_summary_fixed.json"
        with out_path.open("w") as f:
            json.dump(new_summary, f, indent=2, ensure_ascii=False)
        print(f"    Written: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
