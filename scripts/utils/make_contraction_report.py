#!/usr/bin/env python3
"""Assemble a one-PDF summary of the v5 contraction/amplification analysis.

Pulls CSVs from `contraction_analysis_v5/{tables,hierarchy,items_3to6}` and
key PNGs from the same tree, and writes a single PDF: title page, headline
tables, per-dataset breakdown, figures, remaining disagreements.

Usage:
    python scripts/make_contraction_report.py \\
        --v5-dir results/lightning/contraction_analysis_v5 \\
        --out    results/lightning/contraction_analysis_v5/report.pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.07, 0.95, title, fontsize=18, weight="bold", va="top")
    y = 0.89
    for ln in lines:
        ax.text(0.07, y, ln, fontsize=10, va="top", family="monospace",
                wrap=True)
        y -= 0.022
        if y < 0.05:
            break
    pdf.savefig(fig); plt.close(fig)


def _table_page(pdf: PdfPages, title: str, rows: list[dict[str, Any]],
                columns: list[tuple[str, str, str]],
                note: str | None = None) -> None:
    """columns: list of (csv_key, header, fmt) — fmt e.g. '{:.3f}' or '{}'."""
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.82]); ax.axis("off")
    ax.text(0.0, 1.02, title, fontsize=15, weight="bold",
            transform=ax.transAxes, va="bottom")
    if note:
        ax.text(0.0, -0.04, note, fontsize=9, style="italic",
                transform=ax.transAxes, va="top")
    headers = [h for _, h, _ in columns]
    cells = []
    for r in rows:
        row = []
        for k, _, fmt in columns:
            v = r.get(k, "")
            try:
                row.append(fmt.format(float(v)) if "{:" in fmt and v != "" else fmt.format(v))
            except (ValueError, TypeError):
                row.append(str(v))
        cells.append(row)
    tbl = ax.table(cellText=cells, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2d3e50")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    pdf.savefig(fig); plt.close(fig)


def _image_page(pdf: PdfPages, title: str, img_path: Path,
                caption: str | None = None) -> None:
    if not img_path.is_file():
        return
    fig = plt.figure(figsize=(11, 8.5))
    ax_title = fig.add_axes([0.05, 0.93, 0.9, 0.05]); ax_title.axis("off")
    ax_title.text(0, 0, title, fontsize=14, weight="bold")
    ax_img = fig.add_axes([0.05, 0.1, 0.9, 0.8]); ax_img.axis("off")
    ax_img.imshow(mpimg.imread(img_path))
    if caption:
        ax_cap = fig.add_axes([0.05, 0.03, 0.9, 0.05]); ax_cap.axis("off")
        ax_cap.text(0, 0, caption, fontsize=9, style="italic")
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    v5 = Path(args.v5_dir)
    h = v5 / "hierarchy"
    t = v5 / "tables"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds_summary = _read_csv(t / "dataset_summary.csv")
    ds_hier = _read_csv(h / "hierarchy_datasets.csv")
    tt_hier = _read_csv(h / "hierarchy_test_types.csv")
    disagree = _read_csv(h / "disagreement_subtests.csv")
    degenerate = _read_csv(h / "degenerate_subtests.csv")
    llama_win = _read_csv(h / "llama_winning_subtests.csv")

    with PdfPages(out) as pdf:
        # ---- Title / headline ----
        _text_page(pdf,
            "Contraction vs Amplification: v5 Report",
            [
                "Empirical test of CheckList Prop. 1:  gamma_AR > alpha_Diff",
                "   (diffusion contracts perturbations more than AR)",
                "",
                "Models:   LLaDA-8B-Instruct (diffusion)  vs  Llama-3.1-8B-Instruct (AR)",
                "Tasks:    sentiment, QQP, SQuAD   (CheckList behavioral suites)",
                "",
                "Pipeline: v4 numerics + v5 hierarchy with degenerate-MFT filter",
                "          (drop MFT subtests where min(alpha, gamma) < 1e-3;",
                "           those are cases where a model emits a constant label",
                "           and the ratio is meaningless).",
                "",
                "Headline (all datasets, pair-weighted):",
                "   gamma - alpha  > 0 on sentiment (+0.08), qqp (+0.21), squad (+0.17)",
                "   -> Prop. 1 (alpha < gamma) holds on every task.",
                "   Strict contraction (alpha < 1) is VIOLATED everywhere",
                "   (alpha in [1.6, 5.2]); only the weaker relative inequality holds.",
                "",
                "Subtest-level agreement (gap winner == accuracy winner):",
                "   Overall (filtered):  0.723   (pre-filter 0.693)",
                "   INV  (on-manifold):  0.846   <- clean Prop. 1 validation",
                "   DIR+MFT off-mfd:     0.641",
                "",
                "Interpretation: Prop. 1 is cleanest as an INVARIANCE claim.",
                "Paper's Delta_perp = L(f(T(x)), f(x)) is an invariance metric;",
                "INV tests are invariance tests. Natural match.",
            ],
        )

        # ---- Dataset-level numerics ----
        _table_page(
            pdf, "Dataset summary (v5 primary numerics, pair-weighted)",
            ds_summary,
            [
                ("dataset", "dataset", "{}"),
                ("winner", "winner", "{}"),
                ("alpha_Diff", "alpha_Diff", "{:.3f}"),
                ("gamma_AR", "gamma_AR", "{:.3f}"),
                ("gap_gamma_minus_alpha", "gap", "{:+.3f}"),
                ("Delta_perp_Diff", "Delta_perp_Diff", "{:.3f}"),
                ("Delta_perp_AR", "Delta_perp_AR", "{:.3f}"),
                ("llada_n_pairs", "pairs_llada", "{:.0f}"),
                ("n_subtests", "n_sub", "{:.0f}"),
            ],
            note=(
                "Positive gap = LLaDA has the lower perturbation coefficient. "
                "All three datasets are above the diagonal (LLaDA-favorable), "
                "but by thin margins (0.08 - 0.21)."
            ),
        )

        # ---- Hierarchy: per-dataset agreement ----
        _table_page(
            pdf, "Per-dataset subtest agreement (post degenerate filter)",
            ds_hier,
            [
                ("dataset", "dataset", "{}"),
                ("n_subtests", "n", "{:.0f}"),
                ("n_subtests_inv", "n_INV", "{:.0f}"),
                ("n_subtests_dir", "n_DIR", "{:.0f}"),
                ("n_subtests_mft", "n_MFT", "{:.0f}"),
                ("agreement_rate", "agree_all", "{:.3f}"),
                ("offmanifold_agreement_rate", "agree_off", "{:.3f}"),
                ("subtest_win_rate_llada_gap", "llada_win_gap", "{:.3f}"),
                ("subtest_win_rate_llada_acc", "llada_win_acc", "{:.3f}"),
                ("pair_weighted_mean_gap", "pw_mean_gap", "{:+.3f}"),
                ("pair_weighted_mean_acc_gap", "pw_acc_gap", "{:+.3f}"),
            ],
            note=(
                "qqp: cleanest Prop. 1 validation (0.852). "
                "sentiment: INV holds (0.846) but DIR weak (Fairness attribute swaps). "
                "squad: MFT residual noise from constant-output Llama cases."
            ),
        )

        # ---- Hierarchy: per test-type ----
        _table_page(
            pdf, "Per (dataset x test_type) agreement",
            tt_hier,
            [
                ("dataset", "dataset", "{}"),
                ("test_type", "test_type", "{}"),
                ("n_subtests", "n", "{:.0f}"),
                ("agreement_rate", "agree", "{:.3f}"),
                ("subtest_win_rate_llada_gap", "llada_win_gap", "{:.3f}"),
                ("subtest_win_rate_llada_acc", "llada_win_acc", "{:.3f}"),
                ("mean_alpha_Diff", "alpha", "{:.3f}"),
                ("mean_gamma_AR", "gamma", "{:.3f}"),
                ("pair_weighted_mean_gap", "pw_mean_gap", "{:+.3f}"),
            ],
            note=(
                "INV rows show the clean Prop. 1 story: agreement 0.80+ on all 3 datasets. "
                "DIR/MFT noisier - DIR expected-output should change, MFT within-group "
                "pairs are artefacts of CheckList structure."
            ),
        )

        # ---- Key figures ----
        _image_page(pdf, "Figure 1: Per-subtest gap vs accuracy gap (filtered)",
                    h / "subtest_gap_vs_acc.png",
                    caption=("Top-right / bottom-left quadrants = agreement. "
                             "Upper-right: LLaDA wins both. Lower-left: Llama wins both."))
        _image_page(pdf, "Figure 2: Per-capability strip, gap colored by acc winner",
                    h / "capability_strip.png",
                    caption=("Green = LLaDA wins accuracy, red = Llama wins. "
                             "Sentiment:Fairness (red cluster) is the honest weakness."))
        _image_page(pdf, "Figure 3: alpha vs gamma (per-subtest ratio scatter)",
                    v5 / "scatter_ratio_contraction_vs_amplification.png",
                    caption=("Above diagonal = LLaDA contracts more than Llama. "
                             "Axis convention: X=alpha_Diff, Y=gamma_AR."))
        _image_page(pdf, "Figure 4: Per-pair ratio histograms",
                    v5 / "histogram_pair_ratios.png")
        _image_page(pdf, "Figure 5: Capability-level gap bars",
                    v5 / "bar_capability_gap.png")
        _image_page(pdf, "Figure 6: Item-4 gap vs accuracy gap",
                    v5 / "items_3to6" / "item4_gap_vs_accuracy_gap.png")

        # ---- Llama-winning subtests (top 12) ----
        llama_top = sorted(llama_win,
                           key=lambda r: float(r["acc_gap_llada_minus_llama"]))[:12]
        _table_page(
            pdf, "Top Llama-winning subtests (by accuracy gap)",
            llama_top,
            [
                ("dataset", "dataset", "{}"),
                ("capability", "cap", "{}"),
                ("test_type", "tt", "{}"),
                ("subtest", "subtest", "{}"),
                ("alpha_Diff", "alpha", "{:.2f}"),
                ("gamma_AR", "gamma", "{:.2f}"),
                ("gap_gamma_minus_alpha", "gap", "{:+.3f}"),
                ("llada_acc", "LLaDA_acc", "{:.3f}"),
                ("llama_acc", "Llama_acc", "{:.3f}"),
                ("acc_gap_llada_minus_llama", "acc_gap", "{:+.3f}"),
            ],
            note=("Concentrated in sentiment:Fairness INV (protected-attribute "
                  "swaps: race/religion/sexuality) and squad MFT taxonomy/SRL."),
        )

        # ---- Remaining disagreements ----
        _table_page(
            pdf, f"Remaining disagreements after filter (n={len(disagree)})",
            disagree,
            [
                ("dataset", "dataset", "{}"),
                ("capability", "cap", "{}"),
                ("test_type", "tt", "{}"),
                ("subtest", "subtest", "{}"),
                ("alpha_Diff", "alpha", "{:.2f}"),
                ("gamma_AR", "gamma", "{:.2f}"),
                ("gap_gamma_minus_alpha", "gap", "{:+.3f}"),
                ("llada_acc", "LLaDA_acc", "{:.3f}"),
                ("llama_acc", "Llama_acc", "{:.3f}"),
                ("acc_gap_llada_minus_llama", "acc_gap", "{:+.3f}"),
            ],
            note=("Mostly squad MFT cases where Llama outputs a constant label "
                  "(acc = 0) but embeddings still vary -> alpha/gamma not < 1e-3. "
                  "Would need an output-entropy filter to catch these."),
        )

        # ---- Degenerate filter detail ----
        deg_sample = degenerate[:15]
        _table_page(
            pdf,
            f"Degenerate subtests dropped by filter (n={len(degenerate)}; showing 15)",
            deg_sample,
            [
                ("dataset", "dataset", "{}"),
                ("capability", "cap", "{}"),
                ("subtest", "subtest", "{}"),
                ("alpha_Diff", "alpha", "{:.4f}"),
                ("gamma_AR", "gamma", "{:.4f}"),
                ("llada_acc", "LLaDA_acc", "{:.3f}"),
                ("llama_acc", "Llama_acc", "{:.3f}"),
                ("degenerate_reason", "reason", "{}"),
            ],
            note=("All are MFT subtests where min(alpha, gamma) < 1e-3 - "
                  "one model's output is effectively constant, making the ratio "
                  "uninformative."),
        )

        # ---- Closing ----
        _text_page(pdf,
            "Takeaways & open items",
            [
                "Headline results (v5):",
                "  * Prop. 1 (alpha < gamma) holds on all 3 datasets, pair-weighted.",
                "  * INV test type = cleanest match to paper's Delta_perp invariance",
                "    metric. Agreement 0.846 across all datasets.",
                "  * qqp now the strongest off-manifold validation (agreement 0.842).",
                "",
                "Honest weaknesses:",
                "  * Strict contraction alpha < 1 (Assumption 3) violated everywhere;",
                "    only the relative inequality alpha < gamma holds.",
                "  * Sentiment:Fairness INV - LLaDA flips predictions more than Llama",
                "    on protected-attribute swaps (race/religion/sexuality). Real",
                "    finding, theory-consistent (LLaDA amplifies on those INV pairs),",
                "    worth surfacing as a separate robustness story.",
                "",
                "Remaining noise (10 disagreements):",
                "  * squad MFT where Llama emits constant label but embeddings vary;",
                "    not caught by alpha/gamma threshold. Needs output-entropy filter.",
                "",
                "Open items:",
                "  1. Output-entropy filter for MFT (complement to current filter).",
                "  2. Reframe writeup with INV as primary Prop. 1 test.",
                "  3. Sentiment Fairness as standalone story / footnote.",
                "  4. Off-manifold LM filter (paper Section 4 step 4) - still deferred.",
                "",
                "Data paths:",
                "  results/lightning/contraction_analysis_v5/",
                "    tables/{dataset,capability,subtest}_summary.csv",
                "    hierarchy/{hierarchy_*,disagreement_*,degenerate_*}.csv",
                "    items_3to6/item{3,4,5,6}_*",
            ],
        )

    print(f"Wrote report -> {out}")


if __name__ == "__main__":
    main()
