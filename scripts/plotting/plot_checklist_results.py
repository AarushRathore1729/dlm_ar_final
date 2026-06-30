#!/usr/bin/env python3
"""
Plot CheckList-style visualizations from saved JSON results (no Jupyter required).

Official CheckList uses `suite.visual_summary_table()` in a notebook (ipywidgets).
This script reproduces a similar *matrix* view from:

  - suite_visual_snapshot_<model>.json  (written by scripts/run_checklist.py)
  - checklist_results_<model>.json       (full results; uses embedded tests)
  - checklist_summary.json               (dict of model_name -> results)

Examples:

  python scripts/plot_checklist_results.py \\
    results/checklist/sentiment/llada_rerun_fixed/suite_visual_snapshot_GSAI-ML_LLaDA-8B-Instruct.json

  python scripts/plot_checklist_results.py snap_llada.json snap_llama.json \\
    --labels LLaDA Llama \\
    --out-dir results/analysis/plots/sentiment
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Same ordering as checklist.test_suite.TestSuite.visual_summary_table
CAPABILITY_ORDER = [
    "Vocabulary",
    "Taxonomy",
    "Robustness",
    "NER",
    "Fairness",
    "Temporal",
    "Negation",
    "Coref",
    "SRL",
    "Logic",
]


def _cap_sort_key(cap: str | None) -> int:
    if not cap:
        return 1000
    try:
        return CAPABILITY_ORDER.index(cap)
    except ValueError:
        return 500


def _safe_filename(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)[:120]


def load_tests_from_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (tests, run_metadata_or_none)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    meta = None
    if isinstance(data.get("suite"), dict) and "tests" in data["suite"]:
        meta = data.get("run_metadata")
        return data["suite"]["tests"], meta
    if "tests" in data and isinstance(data["tests"], list):
        meta = data.get("run_metadata")
        return data["tests"], meta
    raise ValueError(f"Unrecognized JSON schema in {path}")


def load_summary_models(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load checklist_summary.json -> {model_key: tests}."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at top level: {path}")
    for key, payload in data.items():
        if isinstance(payload, dict) and "tests" in payload:
            out[key] = payload["tests"]
    if not out:
        raise ValueError(f"No model entries with 'tests' in {path}")
    return out


def _prepare_rows(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort tests by capability order, then by fail_rate descending."""
    enriched = []
    for t in tests:
        fr = t.get("fail_rate")
        if fr is None and t.get("n_cases") and t.get("n_fails") is not None:
            fr = float(t["n_fails"]) / float(t["n_cases"])
        enriched.append({**t, "_fail_rate": float(fr) if fr is not None else 0.0})

    by_cap: dict[str, list[dict[str, Any]]] = {}
    for t in enriched:
        cap = t.get("capability") or "Other"
        by_cap.setdefault(cap, []).append(t)
    for cap in by_cap:
        by_cap[cap].sort(key=lambda x: x["_fail_rate"], reverse=True)

    ordered_caps = sorted(by_cap.keys(), key=_cap_sort_key)
    rows: list[dict[str, Any]] = []
    for cap in ordered_caps:
        rows.extend(by_cap[cap])
    return rows


def plot_fail_rate_heatmap(
    series: list[tuple[str, list[dict[str, Any]]]],
    out_path: Path,
    title: str | None = None,
) -> None:
    """series: [(label, tests), ...] — one column per label."""
    if not series:
        raise ValueError("No series to plot")

    base_rows = _prepare_rows(series[0][1])
    n = len(base_rows)
    m = len(series)

    # Align all series to same test order (by name)
    names = [r.get("name", "") for r in base_rows]
    name_to_idx = {n: i for i, n in enumerate(names)}

    mat = np.zeros((n, m))
    for j, (_label, tests) in enumerate(series):
        fr_by_name = {}
        for t in tests:
            nm = t.get("name", "")
            fr = t.get("fail_rate")
            if fr is None and t.get("n_cases") and t.get("n_fails") is not None:
                fr = float(t["n_fails"]) / float(t["n_cases"])
            fr_by_name[nm] = float(fr) if fr is not None else 0.0
        for i, nm in enumerate(names):
            mat[i, j] = fr_by_name.get(nm, np.nan)

    fig_h = max(8.0, 0.22 * n + 2.0)
    fig_w = max(4.0, 2.2 * m + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(m))
    ax.set_xticklabels([s[0] for s in series], rotation=25, ha="right")
    ax.set_yticks(np.arange(n))
    labels = []
    prev_cap = None
    for r in base_rows:
        cap = r.get("capability") or "?"
        name = (r.get("name") or "")[:52]
        if cap != prev_cap:
            labels.append(f"[{cap}] {name}")
            prev_cap = cap
        else:
            labels.append(f"  {name}")
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(n):
        for j in range(m):
            val = mat[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{100 * val:.0f}%", ha="center", va="center", color="black", fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Fail rate")

    ax.set_title(title or "CheckList-style fail-rate matrix (green=pass, red=fail)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capability_bars(
    tests: list[dict[str, Any]],
    out_path: Path,
    title: str | None = None,
) -> None:
    """One subplot per capability: horizontal bars = pass rate per test."""
    rows = _prepare_rows(tests)
    by_cap: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cap = r.get("capability") or "Other"
        by_cap.setdefault(cap, []).append(r)

    caps = sorted(by_cap.keys(), key=_cap_sort_key)
    n_caps = len(caps)
    fig, axes = plt.subplots(n_caps, 1, figsize=(10, max(6, 1.8 * n_caps)), squeeze=False)
    for ax, cap in zip(axes.flat, caps):
        items = by_cap[cap]
        names = [(it.get("name") or "")[:45] for it in items]
        pass_rates = [1.0 - it["_fail_rate"] for it in items]
        y = np.arange(len(names))
        colors = plt.cm.RdYlGn(np.array(pass_rates))
        ax.barh(y, pass_rates, color=colors, edgecolor="0.3", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Pass rate")
        ax.set_title(cap, fontsize=10, fontweight="bold")
        ax.axvline(1.0, color="0.5", linewidth=0.3)
    fig.suptitle(title or "Per-capability pass rates", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CheckList results from JSON snapshots")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="suite_visual_snapshot_*.json, checklist_results_*.json, or one checklist_summary.json",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Column labels for heatmap (must match number of inputs unless using summary)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: same dir as first input)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="checklist_viz",
        help="Base filename prefix for PNGs",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title override")
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.inputs]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else paths[0].parent

    # Single file: checklist_summary with multiple models
    if len(paths) == 1 and paths[0].name == "checklist_summary.json":
        models = load_summary_models(paths[0])
        labels = list(args.labels) if args.labels else list(models.keys())
        if len(labels) != len(models):
            labels = list(models.keys())
        series = [(labels[i] if i < len(labels) else k, models[k]) for i, k in enumerate(models)]
        stem = args.prefix + "_summary"
        plot_fail_rate_heatmap(
            series,
            out_dir / f"{stem}_heatmap.png",
            title=args.title or "CheckList fail-rate matrix (all models in summary)",
        )
        for key, tests in models.items():
            plot_capability_bars(
                tests,
                out_dir / f"{args.prefix}_{_safe_filename(key)}_by_capability.png",
                title=args.title or key,
            )
        print(f"Wrote heatmap and per-model bar charts under {out_dir}")
        return

    series: list[tuple[str, list[dict[str, Any]]]] = []
    for i, p in enumerate(paths):
        tests, meta = load_tests_from_json(p)
        label = None
        if args.labels and i < len(args.labels):
            label = args.labels[i]
        elif meta and meta.get("model"):
            label = meta["model"]
        else:
            label = p.stem.replace("suite_visual_snapshot_", "").replace("checklist_results_", "")
        series.append((label, tests))

    base_stem = args.prefix
    if len(series) == 1:
        safe = _safe_filename(series[0][0])
        heatmap_path = out_dir / f"{base_stem}_{safe}_heatmap.png"
        bars_path = out_dir / f"{base_stem}_{safe}_by_capability.png"
    else:
        heatmap_path = out_dir / f"{base_stem}_compare_heatmap.png"
        bars_path = None

    plot_fail_rate_heatmap(series, heatmap_path, title=args.title)
    print(f"Wrote {heatmap_path}")

    if len(series) == 1:
        plot_capability_bars(series[0][1], bars_path, title=args.title or series[0][0])
        print(f"Wrote {bars_path}")
    else:
        for label, tests in series:
            plot_capability_bars(
                tests,
                out_dir / f"{base_stem}_{_safe_filename(label)}_by_capability.png",
                title=label,
            )
            print(f"Wrote {out_dir / f'{base_stem}_{_safe_filename(label)}_by_capability.png'}")


if __name__ == "__main__":
    main()
