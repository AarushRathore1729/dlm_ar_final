#!/usr/bin/env python3
"""One-shot regeneration for the retained expansion artifacts.

Dream cells are the validated steps=64 reruns; the rho/inv_rho pipeline is not
part of this artifact. This script refreshes every downstream artifact WITHOUT
re-embedding (replays the existing per-test CSV):

1. results/expansion/contraction/* — replayed from expansion_contraction_per_test.csv
   (aggregation identical to compute_expansion_contraction).
2. subtest appendix (build_subtest_appendix.main) — Dream folded back in.
3. capability split vs each AR (capability_split_per_ar.main) — for both
   llada_instruct and llada_moe.

Run: python3 scripts/analysis/regen_post_drop.py
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def replay_contraction() -> None:
    import compute_expansion_contraction as cec

    src = ROOT / "results/expansion/contraction/expansion_contraction_per_test.csv"
    rows = list(csv.DictReader(open(src)))
    records = []
    for r in rows:
        records.append({
            "model_key": r["model_key"], "display_name": r["display_name"],
            "model_type": r["model_type"], "task": r["task"],
            "test_name": r["test_name"], "test_type": r["test_type"],
            "capability": r.get("capability", ""),
            "n_pairs": int(r["n_pairs"]),
            "mean_ratio": float(r["mean_ratio"]),
            "median_ratio": float(r["median_ratio"]),
            "delta_perp": float(r["delta_perp"]),
            "mean_d_input": float(r["mean_d_input"]),
        })
    print(f"[contraction] replaying {len(records)} records")
    cec._write_outputs(records)


def run_subtest_appendix() -> None:
    import build_subtest_appendix
    sys.argv = ["build_subtest_appendix.py"]
    print("\n[subtest appendix]")
    build_subtest_appendix.main()


def run_capability_split() -> None:
    import capability_split_per_ar
    for dlm in ["llada_instruct", "llada_moe"]:
        sys.argv = ["capability_split_per_ar.py", "--dlm", dlm]
        print(f"\n[capability split] dlm={dlm}")
        capability_split_per_ar.main()


if __name__ == "__main__":
    replay_contraction()
    run_subtest_appendix()
    run_capability_split()
    print("\nAll retained artifacts regenerated.")
