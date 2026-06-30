#!/usr/bin/env python3
"""ci_family_passrate.py

Bootstrap 95% CIs for the paper's headline behavioral family gaps
(LLaDA-family macro pass rate minus AR-family macro pass rate, per suite),
mirroring exactly how analyze_expansion_results.py computes the point
estimates: within-model case-pooled pass rate, then macro mean over models.

Resampling unit: SUBTEST, resampled jointly across all models of a task
(paired bootstrap — every model sees the same resampled test set).

Variants:
  - scope "overall" (all test types) and "INV" (INV tests only)
  - DLM side: "LLaDA family" (llada_instruct + llada_moe, the headline tables)
    and "Masked-DLM incl. Dream" (adds dream, supporting evidence)

Outputs:
  results/expansion/analysis/ci_family_passrate.csv
  results/expansion/analysis/ci_family_passrate.tex
  stdout summary

Usage:
  python scripts/analysis/ci_family_passrate.py [--boot 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

IN_CSV = Path("results/expansion/analysis/test_summary.csv")
OUT_DIR = Path("results/expansion/analysis")

TASK_ORDER = ["sentiment", "qqp", "squad", "mnli", "paws", "anli", "wildguard", "xstest"]

AR_FAM = ["llama_instruct", "mistral", "qwen", "gemma", "olmo"]
DLM_GROUPS = [
    ("LLaDA family", ["llada_instruct", "llada_moe"]),
    ("Masked-DLM incl. Dream", ["llada_instruct", "llada_moe", "dream"]),
]


def load_rows():
    with IN_CSV.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_cases"] = int(r["n_cases"])
        r["n_fails"] = int(r["n_fails"])
    return rows


def build_matrices(rows, task, models, types=None):
    """Return (fails, cases) arrays of shape [n_models, n_tests] over the
    subtests present (with n_cases>0) for every model in `models`."""
    per_model = defaultdict(dict)
    for r in rows:
        if r["task"] != task or r["model_key"] not in models:
            continue
        if types is not None and r["test_type"] not in types:
            continue
        if r["n_cases"] <= 0:
            continue
        per_model[r["model_key"]][r["test_name"]] = (r["n_fails"], r["n_cases"])
    if len(per_model) != len(models):
        return None, None, []
    common = sorted(set.intersection(*(set(d) for d in per_model.values())))
    if not common:
        return None, None, []
    fails = np.array([[per_model[m][t][0] for t in common] for m in models], dtype=float)
    cases = np.array([[per_model[m][t][1] for t in common] for m in models], dtype=float)
    return fails, cases, common


def macro_pass(fails, cases, idx):
    """Within-model case-pooled pass over subtests idx, macro mean over models."""
    f = fails[:, idx].sum(axis=1)
    c = cases[:, idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(c > 0, 1.0 - f / c, np.nan)
    return float(np.nanmean(pr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load_rows()
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in rows)]

    out = []
    print(f"{'scope':<8} {'DLM group':<24} {'task':<10} {'gap_pp':>8} "
          f"{'sub_lo':>8} {'sub_hi':>8} {'sig':>5} {'case_lo':>8} {'case_hi':>8} {'csig':>5} {'n_tests':>7}")
    print("-" * 110)
    for scope, types in [("overall", None), ("INV", {"INV"})]:
        for gname, dlm_models in DLM_GROUPS:
            models = dlm_models + AR_FAM
            n_dlm = len(dlm_models)
            for task in tasks:
                fails, cases, common = build_matrices(rows, task, models, types)
                if fails is None:
                    continue
                n = len(common)
                all_idx = np.arange(n)
                gap_pt = macro_pass(fails[:n_dlm], cases[:n_dlm], all_idx) - \
                         macro_pass(fails[n_dlm:], cases[n_dlm:], all_idx)
                boots = np.empty(args.boot)
                for b in range(args.boot):
                    idx = rng.integers(0, n, size=n)
                    boots[b] = macro_pass(fails[:n_dlm], cases[:n_dlm], idx) - \
                               macro_pass(fails[n_dlm:], cases[n_dlm:], idx)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                sig = "DLM*" if lo > 0 else ("AR*" if hi < 0 else "ns")

                # Case-level (binomial) bootstrap: fixed test set, resample each
                # model-test fail count ~ Binomial(n_cases, fail_rate). The right
                # unit when a suite has very few subtests (e.g. WildGuard INV = 1).
                rate = np.where(cases > 0, fails / cases, 0.0)
                cases_int = cases.astype(np.int64)
                cboots = np.empty(args.boot)
                for b in range(args.boot):
                    f_b = rng.binomial(cases_int, rate).astype(float)
                    cboots[b] = macro_pass(f_b[:n_dlm], cases[:n_dlm], all_idx) - \
                                macro_pass(f_b[n_dlm:], cases[n_dlm:], all_idx)
                clo, chi = np.percentile(cboots, [2.5, 97.5])
                csig = "DLM*" if clo > 0 else ("AR*" if chi < 0 else "ns")

                print(f"{scope:<8} {gname:<24} {task:<10} {gap_pt*100:>+8.1f} "
                      f"{lo*100:>+8.1f} {hi*100:>+8.1f} {sig:>5} "
                      f"{clo*100:>+8.1f} {chi*100:>+8.1f} {csig:>5} {n:>7d}")
                out.append({
                    "scope": scope, "dlm_group": gname, "task": task,
                    "n_subtests": n,
                    "gap_pp": gap_pt * 100, "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
                    "significance": sig,
                    "case_ci_lo_pp": clo * 100, "case_ci_hi_pp": chi * 100,
                    "case_significance": csig,
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ci_family_passrate.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "dlm_group", "task", "n_subtests",
                                          "gap_pp", "ci_lo_pp", "ci_hi_pp", "significance",
                                          "case_ci_lo_pp", "case_ci_hi_pp", "case_significance"])
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nWrote {csv_path}")

    # LaTeX: one table, LLaDA-family rows (headline), overall + INV columns
    def cell(scope, task):
        r = next((r for r in out if r["scope"] == scope and r["task"] == task
                  and r["dlm_group"] == "LLaDA family"), None)
        if r is None:
            return "--"
        # With very few subtests the subtest bootstrap is degenerate; report the
        # case-level (binomial) CI instead, marked with a dagger.
        if r["n_subtests"] < 5:
            s = f"{r['gap_pp']:+.1f}\\,[{r['case_ci_lo_pp']:+.1f},{r['case_ci_hi_pp']:+.1f}]$^\\dagger$"
            sig = r["case_significance"]
        else:
            s = f"{r['gap_pp']:+.1f}\\,[{r['ci_lo_pp']:+.1f},{r['ci_hi_pp']:+.1f}]"
            sig = r["significance"]
        if sig == "DLM*":
            s = r"\textbf{" + s + "}"
        elif sig == "AR*":
            s = r"\underline{" + s + "}"
        return s

    lines = [
        "% Auto-generated by scripts/analysis/ci_family_passrate.py",
        r"\begin{table}[H]\centering\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Suite & Overall gap (pp) & INV-only gap (pp) \\",
        r"\midrule",
    ]
    paper_tasks = [t for t in tasks if t != "xstest"]  # XSTest dropped from the paper
    for task in paper_tasks:
        lines.append(f"{task} & {cell('overall', task)} & {cell('INV', task)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{LLaDA-family minus AR-family macro pass-rate gap per suite, in"
        r" percentage points, with 95\% bootstrap CIs over subtests (10{,}000 paired"
        r" resamples). $^\dagger$For suites with fewer than five subtests the CI is a"
        r" case-level binomial bootstrap instead (subtest resampling is degenerate there)."
        r" \textbf{Bold}: LLaDA family significantly higher."
        r" \underline{Underlined}: AR family significantly higher. Plain: CI crosses 0.}",
        r"\label{tab:ci-family-passrate}",
        r"\end{table}",
        "",
    ]
    tex_path = OUT_DIR / "ci_family_passrate.tex"
    tex_path.write_text("\n".join(lines))
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
