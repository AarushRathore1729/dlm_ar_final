#!/usr/bin/env python3
"""pair_compare_dlm_vs_ar.py

Pairwise DLM × AR contraction/amplification comparison, mirroring v5's
LLaDA-vs-Llama format but extended to every DLM-AR combination.

Reads results/expansion/contraction/expansion_contraction_per_test.csv
(produced by compute_expansion_contraction.py) and computes, per task and
per (DLM, AR) pair:

  α_pair        = pair-weighted mean ratio over DLM's subtests
  γ_pair        = pair-weighted mean ratio over AR's subtests
  γ - α
  γ / α
  Δ⊥_DLM, Δ⊥_AR, Δ⊥ gap (= Δ⊥_DLM - Δ⊥_AR; sign per our script)
  n_subtests   = # subtests where BOTH models have data
  dlm_wins     = # subtests where α_subtest < γ_subtest (DLM more contractive)
  win_rate     = dlm_wins / n_subtests

Outputs:
  results/expansion/contraction/pair_matrix_<task>.csv  (rows=DLM, cols=AR)
  results/expansion/contraction/pair_long.csv           (one row per pair-task)

Usage:
    python scripts/analysis/pair_compare_dlm_vs_ar.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

PER_TEST_CSV = Path("results/expansion/contraction/expansion_contraction_per_test.csv")
OUT_DIR = Path("results/expansion/contraction")

DLM_KEYS = ["llada_instruct", "llada_moe", "dream"]
AR_KEYS  = ["llama_instruct", "mistral", "qwen", "gemma", "olmo"]


def load_records() -> list[dict]:
    if not PER_TEST_CSV.exists():
        raise FileNotFoundError(f"Run compute_expansion_contraction.py first; "
                                f"missing {PER_TEST_CSV}")
    out = []
    with PER_TEST_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                r["n_pairs"]      = int(r["n_pairs"])
                r["mean_ratio"]   = float(r["mean_ratio"])
                r["median_ratio"] = float(r["median_ratio"])
                r["delta_perp"]   = float(r["delta_perp"])
            except (KeyError, ValueError):
                continue
            out.append(r)
    return out


def by_model_test(records):
    """{(model_key, task, test_name): record}"""
    out = {}
    for r in records:
        out[(r["model_key"], r["task"], r["test_name"])] = r
    return out


def pair_compare(records, dlm_key: str, ar_key: str, task: str):
    """Compute pair-level metrics by joining on common subtests."""
    idx = by_model_test(records)
    # Find subtests for which BOTH models produced data
    dlm_tests = {tn for (mk, t, tn) in idx if mk == dlm_key and t == task}
    ar_tests  = {tn for (mk, t, tn) in idx if mk == ar_key  and t == task}
    common = sorted(dlm_tests & ar_tests)
    if not common:
        return None

    dlm_ratio_num = ar_ratio_num = 0.0
    dlm_dperp_num = ar_dperp_num = 0.0
    dlm_pairs = ar_pairs = 0
    wins = ties = total_subtests = 0
    per_subtest = []
    for tn in common:
        d = idx[(dlm_key, task, tn)]
        a = idx[(ar_key,  task, tn)]
        dlm_ratio_num += d["mean_ratio"] * d["n_pairs"]
        ar_ratio_num  += a["mean_ratio"] * a["n_pairs"]
        dlm_dperp_num += d["delta_perp"] * d["n_pairs"]
        ar_dperp_num  += a["delta_perp"] * a["n_pairs"]
        dlm_pairs += d["n_pairs"]
        ar_pairs  += a["n_pairs"]
        total_subtests += 1
        # Subtest win = DLM more contractive
        if d["mean_ratio"] < a["mean_ratio"]:
            wins += 1
        elif d["mean_ratio"] == a["mean_ratio"]:
            ties += 1
        per_subtest.append((tn, d["mean_ratio"], a["mean_ratio"]))

    alpha = dlm_ratio_num / dlm_pairs if dlm_pairs else None
    gamma = ar_ratio_num  / ar_pairs  if ar_pairs  else None
    if alpha is None or gamma is None:
        return None
    dp_d = dlm_dperp_num / dlm_pairs
    dp_a = ar_dperp_num  / ar_pairs
    return {
        "dlm": dlm_key, "ar": ar_key, "task": task,
        "alpha": alpha, "gamma": gamma,
        "gap_g_minus_a": gamma - alpha,
        "ratio_g_over_a": gamma / alpha,
        "delta_perp_dlm": dp_d, "delta_perp_ar": dp_a,
        "delta_perp_gap": dp_d - dp_a,
        "n_subtests": total_subtests,
        "dlm_wins": wins, "ties": ties,
        "win_rate": wins / total_subtests if total_subtests else 0.0,
        "per_subtest": per_subtest,
    }


def main():
    records = load_records()
    print(f"Loaded {len(records)} per-test records from {PER_TEST_CSV}")
    tasks = sorted({r["task"] for r in records})

    long_rows = []
    for task in tasks:
        print("\n" + "=" * 92)
        print(f"=== TASK: {task}  (rows = DLM, cols = AR)")
        print("=" * 92)
        # Build matrix; print 3 sub-tables: gap, ratio, win_rate
        matrices = {"gap (γ−α)": {}, "ratio (γ/α)": {}, "win_rate": {}}
        dlms_seen = []
        ars_seen = []
        for d in DLM_KEYS:
            for a in AR_KEYS:
                res = pair_compare(records, d, a, task)
                if res is None:
                    continue
                if d not in dlms_seen: dlms_seen.append(d)
                if a not in ars_seen:  ars_seen.append(a)
                matrices["gap (γ−α)"][(d, a)] = res["gap_g_minus_a"]
                matrices["ratio (γ/α)"][(d, a)] = res["ratio_g_over_a"]
                matrices["win_rate"][(d, a)] = res["win_rate"]
                long_rows.append({k: v for k, v in res.items() if k != "per_subtest"})

        for label, m in matrices.items():
            print(f"\n-- {label} --")
            label_corner = "DLM \\ AR"
            hdr = f"{label_corner:<16}" + "".join(f"{a:>14}" for a in ars_seen)
            print(hdr)
            for d in dlms_seen:
                cells = []
                for a in ars_seen:
                    v = m.get((d, a))
                    if v is None:
                        cells.append(f"{'--':>14}")
                    elif label == "win_rate":
                        cells.append(f"{v*100:>12.1f}%")
                    elif label == "ratio (γ/α)":
                        cells.append(f"{v:>14.3f}")
                    else:
                        cells.append(f"{v:>+14.3f}")
                print(f"{d:<16}" + "".join(cells))

        # Per-task pair matrix CSV (one CSV with α, γ, gap, ratio, win_rate side-by-side)
        mat_path = OUT_DIR / f"pair_matrix_{task}.csv"
        with mat_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dlm", "ar", "alpha", "gamma", "gap_g_minus_a",
                        "ratio_g_over_a", "delta_perp_dlm", "delta_perp_ar",
                        "delta_perp_gap", "n_subtests", "dlm_wins",
                        "ties", "win_rate"])
            for r in long_rows:
                if r["task"] != task: continue
                w.writerow([r["dlm"], r["ar"],
                            f"{r['alpha']:.4f}", f"{r['gamma']:.4f}",
                            f"{r['gap_g_minus_a']:+.4f}",
                            f"{r['ratio_g_over_a']:.4f}",
                            f"{r['delta_perp_dlm']:.4f}",
                            f"{r['delta_perp_ar']:.4f}",
                            f"{r['delta_perp_gap']:+.4f}",
                            r["n_subtests"], r["dlm_wins"], r["ties"],
                            f"{r['win_rate']:.4f}"])
        print(f"\nWrote {mat_path}")

    long_path = OUT_DIR / "pair_long.csv"
    with long_path.open("w", newline="") as f:
        cols = ["task", "dlm", "ar", "alpha", "gamma", "gap_g_minus_a",
                "ratio_g_over_a", "delta_perp_dlm", "delta_perp_ar",
                "delta_perp_gap", "n_subtests", "dlm_wins", "ties", "win_rate"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in long_rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})
    print(f"Wrote {long_path}")


if __name__ == "__main__":
    main()
