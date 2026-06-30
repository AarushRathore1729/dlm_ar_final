#!/usr/bin/env python3
"""bootstrap_ci.py

Bootstrap 95% confidence intervals over subtests for:

  1. Per-(model, task) point estimates of pair-weighted alpha/gamma
  2. Per-(DLM, AR, task) pair estimates of:
        - gamma - alpha           (CI excludes 0 -> significant gap)
        - gamma / alpha           (CI excludes 1 -> significant amplification ratio)
        - DLM-win rate over subtests

Resampling unit: SUBTEST (i.e., one row of expansion_contraction_per_test.csv
per (model, task, test_name)). For pair statistics, we paired-bootstrap on the
intersection of subtests common to both models (mirrors how the gap is
computed).

Outputs:
  results/expansion/contraction/ci_per_model_task.csv
  results/expansion/contraction/ci_pair.csv
  results/expansion/contraction/ci_pair_matrix_<task>.csv        (gap CI matrix)
  results/expansion/contraction/ci_tables.tex                    (LaTeX-ready)
  stdout: per-task summary

Usage:
  python scripts/analysis/bootstrap_ci.py                 # default 10000 boot
  python scripts/analysis/bootstrap_ci.py --boot 50000
  python scripts/analysis/bootstrap_ci.py --seed 0
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/expansion/contraction")
PER_TEST_CSV = ROOT / "expansion_contraction_per_test.csv"
ROOT_SUFFIX = ""


def _out(name: str) -> Path:
    """Output path with the optional run suffix inserted before the extension."""
    stem, dot, ext = name.rpartition(".")
    return ROOT / f"{stem}{ROOT_SUFFIX}{dot}{ext}"

DLM_KEYS = ["llada_instruct", "llada_moe", "dream"]
AR_KEYS  = ["llama_instruct", "mistral", "qwen", "gemma", "olmo"]
MODEL_ORDER = DLM_KEYS + AR_KEYS

DISPLAY = {
    "llada_instruct": "LLaDA-8B",     "llada_moe": "LLaDA-MoE",
    "dream": "Dream-7B",
    "llama_instruct": "Llama-3.1-8B", "mistral": "Mistral-7B",
    "qwen": "Qwen2.5-7B",             "gemma": "Gemma-2-9B",
    "olmo": "OLMo-2-7B",
}


# ---------------------------------------------------------------------------
# Loaders / utilities
# ---------------------------------------------------------------------------

def load_records():
    if not PER_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing {PER_TEST_CSV}; run compute_expansion_contraction.py first.")
    rows = []
    with PER_TEST_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                r["n_pairs"]      = int(r["n_pairs"])
                r["mean_ratio"]   = float(r["mean_ratio"])
                r["delta_perp"]   = float(r["delta_perp"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def by_model_task(records):
    """{(model_key, task): list of (test_name, mean_ratio, n_pairs, delta_perp)}"""
    out = defaultdict(list)
    for r in records:
        key = (r["model_key"], r["task"])
        out[key].append((r["test_name"], r["mean_ratio"], r["n_pairs"], r["delta_perp"]))
    return out


def by_model_task_test(records):
    """{(model_key, task, test_name): (mean_ratio, n_pairs, delta_perp)}"""
    return {(r["model_key"], r["task"], r["test_name"]):
            (r["mean_ratio"], r["n_pairs"], r["delta_perp"])
            for r in records}


def weighted_mean(ratios, weights):
    return float(np.sum(ratios * weights) / max(np.sum(weights), 1))


# ---------------------------------------------------------------------------
# Bootstrap engines
# ---------------------------------------------------------------------------

def bootstrap_alpha(subtests, n_boot, rng):
    """Single-model bootstrap of pair-weighted mean ratio.

    subtests: list of (test_name, mean_ratio, n_pairs, delta_perp)
    """
    if not subtests:
        return None, None, None
    ratios  = np.array([r for _, r, _, _ in subtests], dtype=float)
    weights = np.array([w for _, _, w, _ in subtests], dtype=float)
    point = weighted_mean(ratios, weights)
    n = len(subtests)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = weighted_mean(ratios[idx], weights[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def bootstrap_pair(dlm_sub, ar_sub, n_boot, rng):
    """Paired bootstrap on the intersection of subtests.

    Returns dict with point estimates and CIs for:
      alpha, gamma, gap = gamma - alpha, ratio = gamma / alpha, win_rate
    """
    dlm_d = {tn: (r, w) for tn, r, w, _ in dlm_sub}
    ar_d  = {tn: (r, w) for tn, r, w, _ in ar_sub}
    common = sorted(set(dlm_d) & set(ar_d))
    if not common:
        return None
    dlm_r = np.array([dlm_d[t][0] for t in common])
    dlm_w = np.array([dlm_d[t][1] for t in common], dtype=float)
    ar_r  = np.array([ar_d[t][0]  for t in common])
    ar_w  = np.array([ar_d[t][1]  for t in common], dtype=float)
    alpha_pt = weighted_mean(dlm_r, dlm_w)
    gamma_pt = weighted_mean(ar_r,  ar_w)
    gap_pt   = gamma_pt - alpha_pt
    ratio_pt = gamma_pt / alpha_pt if alpha_pt else float("nan")
    win_pt   = float(np.mean(dlm_r < ar_r))

    n = len(common)
    a_boot = np.empty(n_boot); g_boot = np.empty(n_boot)
    win_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_boot[b] = weighted_mean(dlm_r[idx], dlm_w[idx])
        g_boot[b] = weighted_mean(ar_r[idx],  ar_w[idx])
        win_boot[b] = float(np.mean(dlm_r[idx] < ar_r[idx]))
    gap_boot   = g_boot - a_boot
    ratio_boot = g_boot / np.where(a_boot == 0, np.nan, a_boot)

    def pct(x): return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))
    return {
        "n_common": n,
        "alpha":      (alpha_pt,  *pct(a_boot)),
        "gamma":      (gamma_pt,  *pct(g_boot)),
        "gap":        (gap_pt,    *pct(gap_boot)),
        "ratio":      (ratio_pt,  *pct(ratio_boot[~np.isnan(ratio_boot)])),
        "win_rate":   (win_pt,    *pct(win_boot)),
    }


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------

def sig_gap(lo, hi):
    """Is the gap CI strictly above 0 or strictly below 0?"""
    if lo > 0: return "DLM-contract*"
    if hi < 0: return "AR-contract*"
    return "ns"


def sig_ratio(lo, hi):
    """Is ratio CI strictly above 1 or strictly below 1?"""
    if lo > 1: return "DLM-contract*"
    if hi < 1: return "AR-contract*"
    return "ns"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--types", nargs="+", default=None,
                        help="restrict to these test types (e.g. INV DIR); default = all")
    parser.add_argument("--suffix", default="",
                        help="suffix for output filenames (e.g. _inv)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    global ROOT_SUFFIX
    ROOT_SUFFIX = args.suffix

    records = load_records()
    if args.types:
        types = set(args.types)
        records = [r for r in records if r.get("test_type") in types]
        print(f"Filtered to test types {sorted(types)}: {len(records)} records")
    print(f"Loaded {len(records)} records from {PER_TEST_CSV}")
    grouped = by_model_task(records)
    tasks = sorted({r["task"] for r in records})

    # 1. Per-(model, task) alpha/gamma CIs
    per_model = []
    print("\n--- Per-(model, task) pair-weighted ratio with 95% CI ---")
    print(f"{'Model':<16} {'Task':<10} {'point':>7} {'CI_lo':>7} {'CI_hi':>7} {'n_tests':>7}")
    for mk in MODEL_ORDER:
        for task in tasks:
            sub = grouped.get((mk, task), [])
            if not sub: continue
            point, lo, hi = bootstrap_alpha(sub, args.boot, rng)
            per_model.append({
                "model_key": mk, "display_name": DISPLAY[mk],
                "model_type": "DLM" if mk in DLM_KEYS else "AR",
                "task": task,
                "point": point, "ci_lo": lo, "ci_hi": hi,
                "n_subtests": len(sub),
            })
            print(f"{DISPLAY[mk]:<16} {task:<10} {point:>7.3f} {lo:>7.3f} {hi:>7.3f} {len(sub):>7d}")

    with (_out("ci_per_model_task.csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_key","display_name","model_type",
                                          "task","point","ci_lo","ci_hi","n_subtests"])
        w.writeheader()
        for r in per_model:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {_out('ci_per_model_task.csv')}")

    # 2. Per-(DLM, AR, task) paired bootstrap
    pair_records = []
    print("\n--- Per-(DLM, AR, task) gap and ratio with 95% CI ---")
    print(f"{'DLM':<14} {'AR':<14} {'task':<10} {'gap':>8} {'gap_lo':>8} {'gap_hi':>8} {'ratio':>7} {'r_lo':>7} {'r_hi':>7} {'sig':<14}")
    for dlm in DLM_KEYS:
        for ar in AR_KEYS:
            for task in tasks:
                dlm_sub = grouped.get((dlm, task), [])
                ar_sub  = grouped.get((ar,  task), [])
                if not dlm_sub or not ar_sub: continue
                res = bootstrap_pair(dlm_sub, ar_sub, args.boot, rng)
                if res is None: continue
                gap_pt, gap_lo, gap_hi = res["gap"]
                ratio_pt, ratio_lo, ratio_hi = res["ratio"]
                sig = sig_gap(gap_lo, gap_hi)
                pair_records.append({
                    "dlm": dlm, "ar": ar, "task": task,
                    "n_subtests": res["n_common"],
                    "alpha":      res["alpha"][0],
                    "alpha_lo":   res["alpha"][1], "alpha_hi": res["alpha"][2],
                    "gamma":      res["gamma"][0],
                    "gamma_lo":   res["gamma"][1], "gamma_hi": res["gamma"][2],
                    "gap":        gap_pt, "gap_lo": gap_lo, "gap_hi": gap_hi,
                    "ratio":      ratio_pt, "ratio_lo": ratio_lo, "ratio_hi": ratio_hi,
                    "win_rate":   res["win_rate"][0],
                    "win_lo":     res["win_rate"][1], "win_hi": res["win_rate"][2],
                    "significance": sig,
                })
                print(f"{DISPLAY[dlm]:<14} {DISPLAY[ar]:<14} {task:<10} "
                      f"{gap_pt:>+8.3f} {gap_lo:>+8.3f} {gap_hi:>+8.3f} "
                      f"{ratio_pt:>7.3f} {ratio_lo:>7.3f} {ratio_hi:>7.3f} {sig:<14}")

    cols = ["dlm","ar","task","n_subtests",
            "alpha","alpha_lo","alpha_hi",
            "gamma","gamma_lo","gamma_hi",
            "gap","gap_lo","gap_hi",
            "ratio","ratio_lo","ratio_hi",
            "win_rate","win_lo","win_hi","significance"]
    with (_out("ci_pair.csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in pair_records:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {_out('ci_pair.csv')}")

    # 3. Per-task gap CI matrix (rows = DLM, cols = AR)
    for task in tasks:
        rows = [r for r in pair_records if r["task"] == task]
        if not rows: continue
        dlms = sorted({r["dlm"] for r in rows}, key=lambda k: MODEL_ORDER.index(k))
        ars  = sorted({r["ar"]  for r in rows}, key=lambda k: MODEL_ORDER.index(k))
        path = _out(f"ci_pair_matrix_{task}.csv")
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dlm"] + [DISPLAY[a] for a in ars])
            for d in dlms:
                row = [DISPLAY[d]]
                for a in ars:
                    cell = next((r for r in rows if r["dlm"] == d and r["ar"] == a), None)
                    if cell is None:
                        row.append("--")
                    else:
                        row.append(f"{cell['gap']:+.3f} [{cell['gap_lo']:+.3f}, {cell['gap_hi']:+.3f}] {cell['significance']}")
                w.writerow(row)
        print(f"Wrote {path}")

    # 4. Capability-level bootstrap (DLM family vs AR family, pooled per capability)
    cap_records = _capability_bootstrap(records, args.boot, rng)

    # 5. LaTeX snippet
    _write_latex(per_model, pair_records, tasks, cap_records)


def _capability_bootstrap(records, n_boot, rng):
    """Pool subtests across all tasks, group by CheckList capability,
    and bootstrap mean gap (DLM-mean - AR-mean) within each capability.

    Uses LLaDA family on the DLM side, matching the paper's headline
    "LLaDA family vs AR family" tables (Dream is reported separately as
    supporting evidence). AR side = mean across all 5 ARs.
    """
    DLM_FAM = {"llada_instruct", "llada_moe"}
    AR_FAM  = set(AR_KEYS)
    # Build: (task, test_name, capability) -> {model_key: mean_ratio}
    cell = defaultdict(dict)
    cap_of = {}
    for r in records:
        key = (r["task"], r["test_name"])
        cell[key][r["model_key"]] = r["mean_ratio"]
        cap_of[key] = r.get("capability") or "(unknown)"

    # For each capability, build the list of subtest-level (DLM_mean, AR_mean) pairs
    cap_subs = defaultdict(list)
    for key, mods in cell.items():
        dlm_vals = [mods[m] for m in DLM_FAM if m in mods]
        ar_vals  = [mods[m] for m in AR_FAM  if m in mods]
        if not dlm_vals or not ar_vals:
            continue
        dlm_mean = float(np.mean(dlm_vals))
        ar_mean  = float(np.mean(ar_vals))
        cap_subs[cap_of[key]].append((key, dlm_mean, ar_mean))

    print("\n--- Capability-level bootstrap CIs (DLM family vs AR family) ---")
    print(f"{'Capability':<20} {'n_tests':>8} {'gap':>8} {'gap_lo':>8} {'gap_hi':>8} {'sig':<14} {'class':<10}")
    print("-" * 80)

    # Semantic vs surface heuristic mapping (matches the LaTeX table we wrote)
    SEMANTIC = {"Vocabulary", "Negation", "Logic", "Temporal"}
    SURFACE  = {"NER", "SRL", "Taxonomy", "Robustness", "Fairness", "Coref"}

    out = []
    for cap, subs in sorted(cap_subs.items(), key=lambda x: -np.mean([d-a for _,d,a in x[1]]) if x[1] else 0):
        if not subs:
            continue
        dlm_arr = np.array([d for _, d, _ in subs])
        ar_arr  = np.array([a for _, _, a in subs])
        gaps = ar_arr - dlm_arr
        gap_pt = float(np.mean(gaps))
        n = len(subs)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(ar_arr[idx] - dlm_arr[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = sig_gap(lo, hi)
        cls = "semantic" if cap in SEMANTIC else ("surface" if cap in SURFACE else "other")
        print(f"{cap[:18]:<20} {n:>8d} {gap_pt:>+8.3f} {lo:>+8.3f} {hi:>+8.3f} {sig:<14} {cls:<10}")
        out.append({
            "capability": cap, "n_subtests": n,
            "gap": gap_pt, "gap_lo": lo, "gap_hi": hi,
            "significance": sig, "class": cls,
        })

    # Also: pooled semantic vs surface
    print()
    for label, members in [("ALL semantic", SEMANTIC), ("ALL surface", SURFACE)]:
        all_subs = []
        for cap, subs in cap_subs.items():
            if cap in members:
                all_subs += subs
        if not all_subs: continue
        dlm_arr = np.array([d for _, d, _ in all_subs])
        ar_arr  = np.array([a for _, _, a in all_subs])
        gaps = ar_arr - dlm_arr
        gap_pt = float(np.mean(gaps))
        n = len(all_subs)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots[b] = float(np.mean(ar_arr[idx] - dlm_arr[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = sig_gap(lo, hi)
        print(f"{label:<20} {n:>8d} {gap_pt:>+8.3f} {lo:>+8.3f} {hi:>+8.3f} {sig:<14}")
        out.append({
            "capability": label, "n_subtests": n,
            "gap": gap_pt, "gap_lo": lo, "gap_hi": hi,
            "significance": sig, "class": "pooled",
        })

    # Persist CSV
    cap_path = _out("ci_capability.csv")
    with cap_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["capability","n_subtests","gap","gap_lo","gap_hi","significance","class"])
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {cap_path}")
    return out


def _write_latex(per_model, pair_records, tasks, cap_records=None):
    lines = [
        "% Auto-generated by scripts/analysis/bootstrap_ci.py",
        "% Per-(model, task) alpha/gamma with 95% bootstrap CIs over subtests.",
        r"\begin{table}[H]\centering\small",
        r"\begin{tabular}{ll" + "c" * len(tasks) + "}",
        r"\toprule",
        r"Model & Type & " + " & ".join(t.capitalize() for t in tasks) + r" \\",
        r"\midrule",
    ]
    seen_dlm = False; seen_ar = False
    for mk in MODEL_ORDER:
        recs = {r["task"]: r for r in per_model if r["model_key"] == mk}
        if not recs: continue
        ty = "DLM" if mk in DLM_KEYS else "AR"
        if ty == "AR" and not seen_ar:
            lines.append(r"\midrule")
            seen_ar = True
        if ty == "DLM":
            seen_dlm = True
        cells = []
        for t in tasks:
            r = recs.get(t)
            if r is None:
                cells.append("--")
            else:
                cells.append(f"{r['point']:.3f}\\,[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")
        lines.append(f"{DISPLAY[mk]} & {ty} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Per-model pair-weighted mean ratio with 95\% bootstrap CIs over"
        r" subtests (10{,}000 resamples). Non-overlapping CIs indicate significant"
        r" differences between models on a task.}",
        r"\label{tab:ci-per-model}",
        r"\end{table}",
        "",
    ]

    # Pair gap table, one per task
    for task in tasks:
        prs = [r for r in pair_records if r["task"] == task]
        if not prs: continue
        dlms = sorted({r["dlm"] for r in prs}, key=lambda k: MODEL_ORDER.index(k))
        ars  = sorted({r["ar"]  for r in prs}, key=lambda k: MODEL_ORDER.index(k))
        lines += [
            r"\begin{table}[H]\centering\small",
            r"\begin{tabular}{l" + "c" * len(ars) + "}",
            r"\toprule",
            f"\\textit{{{task.capitalize()}}} $\\gamma-\\alpha$ & " +
            " & ".join(DISPLAY[a] for a in ars) + r" \\",
            r"\midrule",
        ]
        for d in dlms:
            cells = []
            for a in ars:
                cell = next((r for r in prs if r["dlm"] == d and r["ar"] == a), None)
                if cell is None:
                    cells.append("--")
                else:
                    s = f"{cell['gap']:+.2f}\\,[{cell['gap_lo']:+.2f},{cell['gap_hi']:+.2f}]"
                    if cell["significance"].startswith("DLM"):
                        s = r"\textbf{" + s + "}"
                    elif cell["significance"].startswith("AR"):
                        s = r"\underline{" + s + "}"
                    cells.append(s)
            lines.append(f"{DISPLAY[d]} & " + " & ".join(cells) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{task.capitalize()} pair-wise $\\gamma-\\alpha$ with 95\\% bootstrap CIs"
            r" (paired resample of common subtests, 10{,}000 iterations). \textbf{Bold}: DLM"
            r" significantly more contractive. \underline{Underlined}: AR significantly more"
            r" contractive. Plain: CI overlaps 0.}",
            f"\\label{{tab:ci-pair-{task}}}",
            r"\end{table}",
            "",
        ]

    if cap_records:
        lines += [
            r"\begin{table}[H]\centering\small",
            r"\begin{tabular}{lrrrl}",
            r"\toprule",
            r"Capability & $n$ & $\gamma_\mathrm{AR}-\alpha_\mathrm{Diff}$ & 95\% CI & Significance \\",
            r"\midrule",
        ]
        # Print semantic first, then surface, then pooled at bottom
        for cls in ["semantic", "surface", "other", "pooled"]:
            block = [r for r in cap_records if r["class"] == cls]
            if not block: continue
            for r in block:
                tag = r["significance"]
                if tag == "DLM-contract*": disp = r"\textbf{DLM*}"
                elif tag == "AR-contract*": disp = r"\textbf{AR*}"
                else: disp = "n.s."
                lines.append(f"{r['capability']} & {r['n_subtests']} & "
                             f"{r['gap']:+.3f} & [{r['gap_lo']:+.2f},\\,{r['gap_hi']:+.2f}] & {disp} \\\\")
            lines.append(r"\midrule")
        # remove last spurious midrule
        if lines[-1] == r"\midrule":
            lines.pop()
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Capability-level bootstrap CIs on $\gamma_\mathrm{AR}-\alpha_\mathrm{Diff}$"
            r" (95\% percentile, 10{,}000 resamples) with the LLaDA family on the DLM side and"
            r" the mean of all 5 ARs on the AR side. \textbf{DLM*}: significantly more contractive."
            r" \textbf{AR*}: significantly more amplifying than DLM (i.e.\ DLM amplifies more,"
            r" usually surface perturbations). \emph{Semantic} = vocabulary/negation/logic/temporal"
            r" perturbations; \emph{surface} = NER/SRL/taxonomy/robustness/fairness perturbations.}",
            r"\label{tab:ci-capability}",
            r"\end{table}",
            "",
        ]

    out = _out("ci_tables.tex")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
