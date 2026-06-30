#!/usr/bin/env python3
"""write_recovered_jsonl.py

Apply Dream label recovery (from fix_dream_labels.py) to the
per-row examples_full_*.jsonl, writing a new jsonl with the
predicted_label field replaced by the recovered label.

This makes the recovered labels available to *downstream* analyses (e.g.
compute_expansion_contraction.py) that read predicted_label directly.

Output goes to results/expansion/checklist/<task>/<model>_recovered/.

Usage:
    python scripts/utils/write_recovered_jsonl.py
    python scripts/utils/write_recovered_jsonl.py --targets dream_qqp
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fix_dream_labels import recover_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
BASE = ROOT / "results" / "expansion" / "checklist"

# (task, model_dir, jsonl_name)
TARGETS = {
    "dream_qqp":   ("qqp",  "dream",      "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
    "dream_anli":  ("anli", "dream",      "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
    "dream_mnli":  ("mnli", "dream",      "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
    "dream_paws":  ("paws", "dream",      "examples_full_Dream-org_Dream-v0-Instruct-7B.jsonl"),
}


def recover_one(task: str, model_dir: str, jsonl_name: str) -> dict:
    src_dir = BASE / task / model_dir
    src_jsonl = src_dir / jsonl_name
    if not src_jsonl.exists():
        return {"status": "skip-no-jsonl", "path": str(src_jsonl)}

    out_dir = BASE / task / f"{model_dir}_recovered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / jsonl_name

    raw_dist = Counter()      # raw response strings
    orig_label_dist = Counter()
    new_label_dist  = Counter()
    n = 0
    n_changed = 0
    n_recovered_from_default = 0
    samples_changed = []

    with src_jsonl.open() as f_in, out_jsonl.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            raw = row.get("model_response", "") or ""
            orig = row.get("predicted_label", "") or ""
            rec = recover_label(raw, task)
            new_label = rec if rec else orig

            raw_dist[raw[:30]] += 1
            orig_label_dist[orig] += 1
            new_label_dist[new_label] += 1

            if rec and rec != orig:
                n_changed += 1
                if len(samples_changed) < 10:
                    samples_changed.append((row.get("test_name","?")[:35], raw[:50], orig, rec))

            row["predicted_label"] = new_label
            if rec and rec != orig:
                row["_recovered_from"] = orig
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Copy ancillary files that downstream tools may look for (run_metadata, etc.)
    for fn in src_dir.iterdir():
        if fn.is_file() and fn.name != jsonl_name:
            dst = out_dir / fn.name
            if not dst.exists():
                try:
                    shutil.copy2(fn, dst)
                except Exception:
                    pass

    return {
        "status": "ok",
        "n": n,
        "n_changed": n_changed,
        "out_path": str(out_jsonl.relative_to(ROOT)),
        "orig_label_dist": orig_label_dist.most_common(),
        "new_label_dist": new_label_dist.most_common(),
        "raw_dist_top": raw_dist.most_common(15),
        "samples_changed": samples_changed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=None,
                    help=f"Subset of {list(TARGETS.keys())}")
    args = ap.parse_args()

    keys = args.targets or list(TARGETS.keys())
    for k in keys:
        if k not in TARGETS:
            print(f"unknown target: {k}")
            continue
        task, model_dir, jsonl_name = TARGETS[k]
        print(f"\n=== {k}  ({task} / {model_dir}) ===")
        r = recover_one(task, model_dir, jsonl_name)
        if r["status"] != "ok":
            print(f"  [skip] {r['status']}: {r.get('path','')}")
            continue
        print(f"  rows: {r['n']}")
        print(f"  recovered labels (changed from original parse): {r['n_changed']} ({r['n_changed']*100/max(r['n'],1):.1f}%)")
        print(f"  original label dist : {r['orig_label_dist']}")
        print(f"  recovered label dist: {r['new_label_dist']}")
        print(f"  top raw responses (first 30 chars):")
        for raw, c in r["raw_dist_top"]:
            print(f"    {raw!r:34} {c:6d}")
        print(f"  example changes (test, raw, was -> now):")
        for s in r["samples_changed"]:
            print(f"    {s}")
        print(f"  wrote: {r['out_path']}")


if __name__ == "__main__":
    main()
