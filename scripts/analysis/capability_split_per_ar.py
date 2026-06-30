#!/usr/bin/env python3
"""Capability-level gamma-alpha split: LLaDA vs EACH AR baseline (not just Llama).

Slide 12 reports gamma-alpha per capability for LLaDA vs the AR mean, using Llama
as the de-facto reference. But slide 8 already shows the effect depends on which
AR baseline is used, and Llama is a weak QQP baseline. This script recomputes the
capability split against every AR separately, so we can see whether the DLM
advantage survives the *strongest* AR, not only Llama.

Source: results/expansion/contraction/expansion_contraction_per_test.csv
  (per-(model, task, subtest) mean_ratio = amplification: alpha for DLMs, gamma
  for ARs). Pairs LLaDA vs each AR on (task, test_name); gamma-alpha > 0 means the
  DLM amplifies less (more stable).

Scope: INV-only by default. On INV a low ratio is unambiguously better (the
output *should not* move), so gamma-alpha>0 cleanly means "DLM more stable =
more robust". DIR is deliberately excluded here: on a should-change test the
ratio measures the *magnitude* of output change, not whether it changed in the
*correct* direction, so its sign is not directly a robustness signal -- DIR is
better read from pass rate (slide 13). MFT is out for the same slide-9 reason.
Dream cells are the validated steps=64 reruns, so no Dream exclusions remain.

Outputs (results/expansion/analysis/):
  capability_split_per_ar.csv  - capability x AR table of gamma-alpha (+ n, CI)
  capability_split_per_ar.tex  - LaTeX version
  capability_split_per_ar.png  - heatmap capability x AR
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "results/expansion/contraction/expansion_contraction_per_test.csv"
OUT = ROOT / "results/expansion/analysis"

AR_MODELS = ["llama_instruct", "mistral", "qwen", "gemma", "olmo"]
AR_LABEL = {"llama_instruct": "Llama-3.1", "mistral": "Mistral", "qwen": "Qwen2.5",
            "gemma": "Gemma-2", "olmo": "OLMo-2"}
VALID_TYPES = {"INV"}  # see module docstring: INV is the unambiguous-sign scope
# Capabilities highlighted on slide 12, in order.
SLIDE12_CAPS = ["Negation", "Logic", "NER", "Taxonomy", "SRL", "Fairness"]


def load_pairs(dlm_key: str):
    """Return cap -> ar_key -> list of paired (gamma - alpha) per subtest."""
    rows = list(csv.DictReader(open(SRC)))
    # (task, test_name) -> model_key -> (ratio, capability, test_type)
    idx = defaultdict(dict)
    for r in rows:
        if r["test_type"] not in VALID_TYPES:
            continue
        try:
            ratio = float(r["mean_ratio"])
        except ValueError:
            continue
        idx[(r["task"], r["test_name"])][r["model_key"]] = (
            ratio, r["capability"], r["test_type"])

    cap_ar = defaultdict(lambda: defaultdict(list))
    for key, permodel in idx.items():
        if dlm_key not in permodel:
            continue
        alpha, cap, _ = permodel[dlm_key]
        for ar in AR_MODELS:
            if ar in permodel:
                gamma = permodel[ar][0]
                cap_ar[cap][ar].append(gamma - alpha)  # >0 -> DLM more stable
    return cap_ar


def ci95(vals):
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    m = sum(vals) / n
    if n == 1:
        return (m, float("nan"), float("nan"), 1)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    # t critical (95%) approximated; use 1.96 for n>30, else a small-sample bump.
    tcrit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
             9: 2.31, 10: 2.26}.get(n, 2.09 if n <= 20 else 1.96)
    return (m, m - tcrit * se, m + tcrit * se, n)


def build(dlm_key: str, dlm_label: str):
    cap_ar = load_pairs(dlm_key)
    caps = SLIDE12_CAPS + sorted(c for c in cap_ar if c not in SLIDE12_CAPS)
    caps = [c for c in caps if c in cap_ar]

    # CSV / table rows
    table = []
    for cap in caps:
        row = {"capability": cap}
        per_ar_means = []
        for ar in AR_MODELS:
            vals = cap_ar[cap].get(ar, [])
            m, lo, hi, n = ci95(vals)
            row[ar] = m
            row[f"{ar}_n"] = n
            row[f"{ar}_lo"] = lo
            row[f"{ar}_hi"] = hi
            if n:
                per_ar_means.append(m)
        row["ar_min"] = min(per_ar_means) if per_ar_means else float("nan")
        row["ar_max"] = max(per_ar_means) if per_ar_means else float("nan")
        row["ar_mean"] = sum(per_ar_means) / len(per_ar_means) if per_ar_means else float("nan")
        # sign agreement: does DLM win (>0) against ALL ARs?
        row["dlm_wins_all"] = all(v > 0 for v in per_ar_means) if per_ar_means else False
        row["dlm_wins_any"] = any(v > 0 for v in per_ar_means) if per_ar_means else False
        table.append(row)

    # write CSV
    csv_path = OUT / f"capability_split_per_ar{'' if dlm_key=='llada_instruct' else '_'+dlm_key}.csv"
    cols = (["capability"] + AR_MODELS + ["ar_mean", "ar_min", "ar_max",
            "dlm_wins_all", "dlm_wins_any"])
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in table:
            w.writerow([row["capability"]] +
                       [f"{row[ar]:+.3f}" for ar in AR_MODELS] +
                       [f"{row['ar_mean']:+.3f}", f"{row['ar_min']:+.3f}",
                        f"{row['ar_max']:+.3f}", row["dlm_wins_all"], row["dlm_wins_any"]])

    # LaTeX
    tex = [
        r"% Auto-generated by scripts/analysis/capability_split_per_ar.py",
        rf"% gamma-alpha per capability: {dlm_label} vs each AR (INV-only). >0 = DLM more stable.",
        r"\begin{tabular}{@{}l" + "r" * len(AR_MODELS) + r"rc@{}}",
        r"\toprule",
        r"\textbf{Capability} & " +
        " & ".join(rf"\textbf{{{AR_LABEL[a]}}}" for a in AR_MODELS) +
        r" & \textbf{Range} & \textbf{DLM beats all?} \\",
        r"\midrule",
    ]
    for row in table:
        cells = " & ".join(f"${row[ar]:+.2f}$" for ar in AR_MODELS)
        rng = f"$[{row['ar_min']:+.2f}, {row['ar_max']:+.2f}]$"
        wins = r"\checkmark" if row["dlm_wins_all"] else (r"$\sim$" if row["dlm_wins_any"] else r"$\times$")
        tex.append(rf"{row['capability']} & {cells} & {rng} & {wins} \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_path = OUT / f"capability_split_per_ar{'' if dlm_key=='llada_instruct' else '_'+dlm_key}.tex"
    tex_path.write_text("\n".join(tex))

    # heatmap
    M = np.array([[row[ar] for ar in AR_MODELS] for row in table], dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(AR_MODELS) + 3, 0.5 * len(table) + 1.6), dpi=180)
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(AR_MODELS)), [AR_LABEL[a] for a in AR_MODELS], rotation=30, ha="right")
    ax.set_yticks(range(len(table)), [row["capability"] for row in table])
    for i in range(len(table)):
        for j in range(len(AR_MODELS)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.6 * vmax else "white")
    ax.set_title(f"$\\gamma-\\alpha$: {dlm_label} vs each AR (INV-only)\n"
                 "blue = DLM more stable, red = AR more stable", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png_path = OUT / f"capability_split_per_ar{'' if dlm_key=='llada_instruct' else '_'+dlm_key}.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return table, csv_path, tex_path, png_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dlm", default="llada_instruct",
                    choices=["llada_instruct", "llada_moe"])
    args = ap.parse_args()
    label = {"llada_instruct": "LLaDA-8B", "llada_moe": "LLaDA-MoE"}[args.dlm]

    table, csv_path, tex_path, png_path = build(args.dlm, label)
    print(f"{label} vs each AR -- gamma-alpha per capability (INV-only). "
          f">0 = DLM more stable.\n")
    hdr = f"{'capability':<12}" + "".join(f"{AR_LABEL[a]:>11}" for a in AR_MODELS) + f"{'range':>16}{'beats all':>10}"
    print(hdr)
    print("-" * len(hdr))
    for row in table:
        line = f"{row['capability']:<12}" + "".join(f"{row[a]:>+11.2f}" for a in AR_MODELS)
        line += f"   [{row['ar_min']:+.2f},{row['ar_max']:+.2f}]"
        line += f"{'YES' if row['dlm_wins_all'] else ('mixed' if row['dlm_wins_any'] else 'no'):>9}"
        print(line)
    for p in (csv_path, tex_path, png_path):
        print(f"Wrote: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
