#!/usr/bin/env python3
"""Generate ACL-style CheckList figures for the EMNLP empirical draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("paper/fig")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#D8D8D8"


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def passrate_gap() -> None:
    src = Path("results/expansion/analysis/ci_family_passrate.csv")
    df = pd.read_csv(src)
    tasks = ["sentiment", "qqp", "squad"]
    suite_labels = {"sentiment": "Sentiment", "qqp": "QQP", "squad": "SQuAD"}
    def series(scope: str, column: str) -> np.ndarray:
        vals = []
        for task in tasks:
            row = df[
                (df["scope"] == scope)
                & (df["dlm_group"] == "LLaDA family")
                & (df["task"] == task)
            ].iloc[0]
            vals.append(float(row[column]))
        return np.array(vals)

    suites = [suite_labels[t] for t in tasks]
    overall = series("overall", "gap_pp")
    overall_lo = series("overall", "ci_lo_pp")
    overall_hi = series("overall", "ci_hi_pp")
    inv = series("INV", "gap_pp")
    inv_lo = series("INV", "ci_lo_pp")
    inv_hi = series("INV", "ci_hi_pp")

    x = np.arange(len(suites))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(x - width / 2, overall, width, color=BLUE, label="Overall")
    ax.bar(x + width / 2, inv, width, color=ORANGE, label="INV only")
    ax.errorbar(
        x - width / 2,
        overall,
        yerr=[overall - overall_lo, overall_hi - overall],
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
    )
    ax.errorbar(
        x + width / 2,
        inv,
        yerr=[inv - inv_lo, inv_hi - inv],
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.set_ylabel("DLM - AR pass-rate gap (pp)")
    ax.set_ylim(-15, 36)
    ax.set_title("Family pass-rate gaps")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_passrate_gap")


def ratio_heatmap() -> None:
    ar_keys = ["llama_instruct", "mistral", "qwen", "gemma", "olmo"]
    ar = ["Llama", "Mistral", "Qwen", "Gemma", "OLMo"]
    dlm_keys = ["llada_instruct", "llada_moe", "dream"]
    dlm = ["LLaDA", "MoE", "Dream"]
    task_labels = [("sentiment", "Sentiment"), ("qqp", "QQP"), ("squad", "SQuAD")]
    panels = []
    for task, title in task_labels:
        df = pd.read_csv(f"results/expansion/contraction/pair_matrix_{task}.csv")
        vals = np.empty((len(dlm_keys), len(ar_keys)))
        for i, dlm_key in enumerate(dlm_keys):
            for j, ar_key in enumerate(ar_keys):
                row = df[(df["dlm"] == dlm_key) & (df["ar"] == ar_key)].iloc[0]
                vals[i, j] = float(row["ratio_g_over_a"])
        panels.append((title, vals))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)
    for ax, (title, vals) in zip(axes, panels):
        im = ax.imshow(vals, cmap="RdBu_r", vmin=0.35, vmax=1.65, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(ar)))
        ax.set_xticklabels(ar, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(dlm)))
        ax.set_yticklabels(dlm if ax is axes[0] else [])
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                mark = r"$^\dagger$" if title == "QQP" and ar[j] in {"Qwen", "Gemma"} else ""
                color = "white" if vals[i, j] < 0.65 or vals[i, j] > 1.35 else "black"
                ax.text(j, i, f"{vals[i, j]:.2f}{mark}", ha="center", va="center",
                        fontsize=7, color=color)
        ax.set_xticks(np.arange(-.5, len(ar), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(dlm), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="both", length=0)
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015)
    cbar.set_label(r"$\gamma_{\mathrm{AR}}/\alpha_{\mathrm{Diff}}$")
    fig.suptitle("Pairwise perturbation-coefficient ratios (>1 favors DLM stability)", y=1.06, fontsize=10)
    save(fig, "emnlp_ratio_heatmap")

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.15), constrained_layout=True)
    for ax, (title, vals) in zip(axes, panels):
        im = ax.imshow(vals, cmap="RdBu_r", vmin=0.35, vmax=1.65, aspect="auto")
        ax.set_title(title, pad=2)
        ax.set_xticks(np.arange(len(ar)))
        ax.set_xticklabels(ar if title == "SQuAD" else [], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(dlm)))
        ax.set_yticklabels(dlm)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                mark = r"$^\dagger$" if title == "QQP" and ar[j] in {"Qwen", "Gemma"} else ""
                color = "white" if vals[i, j] < 0.65 or vals[i, j] > 1.35 else "black"
                ax.text(j, i, f"{vals[i, j]:.2f}{mark}", ha="center", va="center",
                        fontsize=6.2, color=color)
        ax.set_xticks(np.arange(-.5, len(ar), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(dlm), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="both", length=0)
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.012)
    cbar.set_label(r"$\gamma_{\mathrm{AR}}/\alpha_{\mathrm{Diff}}$")
    save(fig, "emnlp_ratio_heatmap_column")


def reversal_summary() -> None:
    df = pd.read_csv("results/analysis/paper_figures/emnlp_reversal_summary.csv")
    labels = df["label"].tolist()
    dlm_a = df["llada_fail"].to_numpy(dtype=float)
    dlm_b = df["llada_moe_fail"].to_numpy(dtype=float)
    ar_ref = df["ar_ref_fail"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.bar(x - width, dlm_a, width, color=BLUE, label="LLaDA")
    ax.bar(x, dlm_b, width, color=GREEN, label="LLaDA-MoE")
    ax.bar(x + width, ar_ref, width, color=GRAY, label="AR ref.")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("fail rate (%)")
    ax.set_ylim(0, 68)
    ax.set_title("Sensitivity-correct perturbations reverse the advantage")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_reversal_summary")


def capability_gap() -> None:
    src = Path("results/expansion/contraction/ci_capability.csv")
    if src.exists() and src.stat().st_size > 0:
        df = pd.read_csv(src)
    else:
        df = pd.read_csv("results/analysis/paper_figures/emnlp_capability_gap.csv")
    keep = [
        "Temporal", "Robustness", "Negation", "Vocabulary", "Logic",
        "Fairness", "NER", "Taxonomy", "Coref", "SRL",
    ]
    df = df[df["capability"].isin(keep)].copy()
    order = keep[::-1]
    df["capability"] = pd.Categorical(df["capability"], categories=order, ordered=True)
    df = df.sort_values("capability")
    colors = [BLUE if c == "semantic" else ORANGE for c in df["class"]]
    sig = df["significance"].ne("ns").to_numpy()

    y = np.arange(len(df))
    x = df["gap"].to_numpy()
    lo = df["gap_lo"].to_numpy()
    hi = df["gap_hi"].to_numpy()
    fig, ax = plt.subplots(figsize=(3.35, 2.65))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.barh(y, x, color=colors, height=0.62)
    ax.errorbar(x, y, xerr=[x - lo, hi - x], fmt="none",
                ecolor="black", elinewidth=0.7, capsize=2)
    labels = [f"{c}{'*' if s else ''}" for c, s in zip(df["capability"], sig)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\gamma_{\mathrm{AR}}-\alpha_{\mathrm{Diff}}$")
    ax.set_title("Capability-level perturbation-coefficient gaps", pad=4)
    ax.set_xlim(-3.8, 2.35)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="semantic"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="surface"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right",
              handlelength=0.9, borderaxespad=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_capability_gap")


def main() -> None:
    passrate_gap()
    ratio_heatmap()
    reversal_summary()
    capability_gap()


if __name__ == "__main__":
    main()
