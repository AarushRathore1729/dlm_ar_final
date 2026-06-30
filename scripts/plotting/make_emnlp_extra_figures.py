#!/usr/bin/env python3
"""Generate compact supporting figures for the EMNLP CheckList draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def selective_stability_map() -> None:
    """Family-level behavioural win distribution by subtest."""
    df = pd.read_csv("results/expansion/analysis/test_summary.csv")
    wanted = [
        ("qqp", "INV", "QQP INV"),
        ("qqp", "DIR", "QQP DIR"),
        ("sentiment", "INV", "Sentiment INV"),
        ("sentiment", "DIR", "Sentiment DIR"),
        ("squad", "INV", "SQuAD INV"),
        ("squad", "MFT", "SQuAD MFT"),
    ]
    rows = []
    for task, typ, label in wanted:
        sub = df[(df["task"] == task) & (df["test_type"] == typ)]
        wins = []
        for _, g in sub.groupby("test_name"):
            dlm = g[g["family"] == "DLLM"]["pass_rate"].mean()
            ar = g[g["family"] == "AR"]["pass_rate"].mean()
            if pd.isna(dlm) or pd.isna(ar):
                continue
            wins.append({
                "winner": "dlm" if dlm > ar else "ar" if ar > dlm else "tie",
                "n_cases": float(g["n_cases"].iloc[0]),
            })
        n = float(len(wins))
        if n == 0:
            continue
        n_cases = sum(w["n_cases"] for w in wins)
        dlm_cases = sum(w["n_cases"] for w in wins if w["winner"] == "dlm")
        tie_cases = sum(w["n_cases"] for w in wins if w["winner"] == "tie")
        rows.append({
            "label": label,
            "dlm": 100.0 * sum(w["winner"] == "dlm" for w in wins) / n,
            "ar": 100.0 * sum(w["winner"] == "ar" for w in wins) / n,
            "tie": 100.0 * sum(w["winner"] == "tie" for w in wins) / n,
            "case": 100.0 * (dlm_cases + 0.5 * tie_cases) / n_cases,
            "n": int(n),
        })
    plot = pd.DataFrame(rows).iloc[::-1]

    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(3.35, 2.72))
    left = np.zeros(len(plot))
    for key, color, label in [
        ("dlm", BLUE, "DLM wins"),
        ("tie", LIGHT_GRAY, "ties"),
        ("ar", ORANGE, "AR wins"),
    ]:
        vals = plot[key].to_numpy()
        ax.barh(y, vals, left=left, height=0.62, color=color, label=label)
        left += vals
    ax.scatter(plot["case"], y, s=15, color="black", zorder=3, label="case-weighted")
    for yi, n in zip(y, plot["n"]):
        ax.text(101.0, yi, f"n={n}", va="center", fontsize=6.2, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("subtests won (%)")
    ax.set_title("Behavioural wins by test family", pad=5)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), handlelength=1.0,
              columnspacing=1.0)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_selective_stability_map")


def appendix_subtest_gap() -> None:
    """Representative subtests behind the selective-stability claim."""
    df = pd.read_csv("results/analysis/ratio_vs_failrate_concordance.csv")
    wanted = [
        ("qqp", "(q, paraphrase(q))", "QQP DIR: paraphrase to duplicate"),
        ("qqp", "Product of paraphrases(q1) * paraphrases(q2)", "QQP INV: paraphrase product"),
        ("qqp", "add one typo", "QQP INV: one typo"),
        ("qqp", "Change same number in both questions", "QQP INV: same number"),
        ("qqp", "Change first name in one of the questions", "QQP DIR: changed first name"),
        ("qqp", "Change numbers in one of the questions", "QQP DIR: changed number"),
        ("sentiment", "protected: sexual", "Sentiment INV: protected sexual"),
        ("sentiment", "protected: religion", "Sentiment INV: protected religion"),
        ("squad", "Add random sentence to context", "SQuAD INV: random sentence"),
        ("squad", "Question typo", "SQuAD INV: question typo"),
    ]
    rows = []
    for task, subtest, label in wanted:
        hit = df[(df["task"] == task) & (df["subtest"] == subtest)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append({
            "label": label,
            "gap": float(r["gap_gamma_minus_alpha"]),
            "dlm_fail": 100.0 * float(r["llada_fail"]),
            "ar_fail": 100.0 * float(r["llama_fail"]),
        })
    plot = pd.DataFrame(rows).iloc[::-1]
    colors = [BLUE if v > 0 else ORANGE for v in plot["gap"]]

    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(3.35, 3.6))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.barh(y, plot["gap"], color=colors, height=0.62)
    for yi, row in zip(y, plot.itertuples()):
        ax.text(
            0.03 if row.gap >= 0 else -0.03,
            yi,
            f"{row.dlm_fail:.0f}/{row.ar_fail:.0f}%",
            ha="left" if row.gap >= 0 else "right",
            va="center",
            fontsize=6.2,
            color="black",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel(r"$\gamma_{\mathrm{Llama}}-\alpha_{\mathrm{LLaDA}}$")
    ax.set_title("Representative subtest gaps", pad=8)
    ax.set_xlim(-3.4, 12.6)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_appendix_subtest_gap")


def qqp_validity_audit() -> None:
    """Small validity figure from retained QQP audit summaries."""
    df = pd.read_csv("results/analysis/paper_figures/emnlp_qqp_validity_audit.csv")
    labels = [label.replace(" ", "\n", 2) for label in df["label"]]
    vals = df["value"].to_numpy(dtype=float)
    colors = [ORANGE, ORANGE, ORANGE, BLUE]

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("QQP label-prior validity audit", pad=5)
    for xi, v in zip(x, vals):
        txt = r"$\leq$0.1" if v == 0.1 else f"{v:.0f}"
        ax.text(xi, v + 1.8, txt, ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_qqp_validity_audit")


def embedding_metric_check() -> None:
    """Compact output-embedding companion ratios."""
    df = pd.read_csv("results/analysis/v5_output_embedding_alpha_gamma_table.csv")
    df["ratio"] = df["gamma_over_alpha"]
    label = {"sentiment": "Sent.", "qqp": "QQP", "squad": "SQuAD"}
    df["label"] = df["task"].map(label)

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    x = np.arange(len(df))
    colors = [BLUE if r > 1 else ORANGE for r in df["ratio"]]
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.bar(x, df["ratio"], color=colors, width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel(r"$\gamma/\alpha$")
    ax.set_ylim(0, 1.35)
    ax.set_title("Output-embedding metric check")
    for xi, v in zip(x, df["ratio"]):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "emnlp_embedding_metric_check")


def main() -> None:
    selective_stability_map()
    appendix_subtest_gap()
    qqp_validity_audit()
    embedding_metric_check()


if __name__ == "__main__":
    main()
