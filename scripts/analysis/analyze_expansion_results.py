#!/usr/bin/env python3
"""Aggregate expansion CheckList results into paper-ready tables and figures.

The expansion summaries include large per-example payloads. This script reads
them once, keeps only per-test aggregates, and writes compact CSV/LaTeX/Markdown
artifacts for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TASK_ORDER = ["sentiment", "qqp", "squad"]
TYPE_ORDER = ["MFT", "INV", "DIR"]

MODEL_META = {
    "llada_instruct": ("LLaDA-8B", "DLLM", "llada"),
    "llada_moe": ("LLaDA-MoE", "DLLM", "llada"),
    "dream": ("Dream-7B", "DLLM", "masked_diffusion"),
    "llama_instruct": ("Llama-3.1", "AR", "ar"),
    "mistral": ("Mistral-7B", "AR", "ar"),
    "qwen": ("Qwen2.5-7B", "AR", "ar"),
    "gemma": ("Gemma-2-9B", "AR", "ar"),
    "olmo": ("OLMo-2-7B", "AR", "ar"),
}

MODEL_ALIASES = {
    "dream_rerun": "dream",
}

DISPLAY_ORDER = [
    "LLaDA-8B",
    "LLaDA-MoE",
    "Dream-7B",
    "Llama-3.1",
    "Mistral-7B",
    "Qwen2.5-7B",
    "Gemma-2-9B",
    "OLMo-2-7B",
]


def _summary_key(data: dict[str, Any]) -> str:
    return next(k for k, v in data.items() if isinstance(v, dict) and "tests" in v)


def _read_run(path: Path, task: str, model_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = _summary_key(data)
    summary = data[key]
    tests: list[dict[str, Any]] = []
    for test in summary["tests"]:
        tests.append({k: v for k, v in test.items() if k != "examples"})

    display, family, subgroup = MODEL_META[model_key]
    total_cases = int(summary.get("total_cases") or sum(t.get("n_cases", 0) for t in tests))
    total_fails = int(summary.get("total_fails") or sum(t.get("n_fails", 0) for t in tests))
    fallback = summary.get("judge_fallback_stats") or {}
    fallback_total = fallback.get("total_examples") or 0
    fallback_count = fallback.get("judge_fallback_count") or 0
    run = {
        "task": task,
        "model_key": model_key,
        "model": display,
        "family": family,
        "subgroup": subgroup,
        "source_model": summary.get("model", ""),
        "summary_path": str(path),
        "total_cases": total_cases,
        "total_fails": total_fails,
        "pass_rate": 1.0 - total_fails / total_cases if total_cases else 0.0,
        "judge_fallback_rate": fallback_count / fallback_total if fallback_total else 0.0,
    }
    return run, tests


def load_all_runs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    summary_paths: dict[tuple[str, str], Path] = {}
    for path in sorted((root / "results" / "expansion" / "checklist").glob("*/*/checklist_summary.json")):
        task = path.parts[-3]
        raw_model_key = path.parts[-2]
        model_key = MODEL_ALIASES.get(raw_model_key, raw_model_key)
        if model_key not in MODEL_META:
            continue
        run_key = (task, model_key)
        if run_key not in summary_paths or raw_model_key.endswith("_rerun"):
            summary_paths[run_key] = path

    for task, model_key in sorted(summary_paths, key=lambda k: (TASK_ORDER.index(k[0]), DISPLAY_ORDER.index(MODEL_META[k[1]][0]))):
        path = summary_paths[(task, model_key)]
        run, tests = _read_run(path, task, model_key)
        runs.append(run)
        for test in tests:
            n_cases = int(test.get("n_cases", 0))
            n_fails = int(test.get("n_fails", 0))
            test_rows.append(
                {
                    **{k: run[k] for k in ("task", "model_key", "model", "family", "subgroup")},
                    "test_name": test.get("name", ""),
                    "test_type": test.get("type") or test.get("test_type") or "",
                    "capability": test.get("capability", ""),
                    "n_cases": n_cases,
                    "n_fails": n_fails,
                    "pass_rate": 1.0 - n_fails / n_cases if n_cases else 0.0,
                    "fail_rate": n_fails / n_cases if n_cases else 0.0,
                }
            )

    baseline = {
        ("sentiment", "llada_instruct"): root / "results" / "checklist" / "sentiment" / "llada_rerun_fixed" / "checklist_summary.json",
        ("sentiment", "llama_instruct"): root / "results" / "checklist" / "sentiment" / "llama_rerun_fixed" / "checklist_summary.json",
        ("qqp", "llada_instruct"): root / "results" / "checklist" / "qqp" / "llada_rerun_fixed" / "checklist_summary.json",
        ("qqp", "llama_instruct"): root / "results" / "checklist" / "qqp" / "llama_rerun_fixed" / "checklist_summary.json",
        ("squad", "llada_instruct"): root / "results" / "checklist" / "squad" / "llada_rerun_fixed" / "checklist_summary.json",
        ("squad", "llama_instruct"): root / "results" / "checklist" / "squad" / "llama_rerun_fixed" / "checklist_summary.json",
    }
    existing = {(r["task"], r["model_key"]) for r in runs}
    for (task, model_key), path in baseline.items():
        if (task, model_key) in existing or not path.exists():
            continue
        run, tests = _read_run(path, task, model_key)
        runs.append(run)
        for test in tests:
            n_cases = int(test.get("n_cases", 0))
            n_fails = int(test.get("n_fails", 0))
            test_rows.append(
                {
                    **{k: run[k] for k in ("task", "model_key", "model", "family", "subgroup")},
                    "test_name": test.get("name", ""),
                    "test_type": test.get("type") or test.get("test_type") or "",
                    "capability": test.get("capability", ""),
                    "n_cases": n_cases,
                    "n_fails": n_fails,
                    "pass_rate": 1.0 - n_fails / n_cases if n_cases else 0.0,
                    "fail_rate": n_fails / n_cases if n_cases else 0.0,
                }
            )
    return runs, test_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def family_dataset_rows(runs: list[dict[str, Any]], group_name: str, members: set[str]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["model_key"] in members:
            by_task[run["task"]].append(run)
    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        vals = by_task.get(task, [])
        if not vals:
            continue
        rows.append(
            {
                "task": task,
                "group": group_name,
                "n_models": len(vals),
                "macro_pass_rate": mean(v["pass_rate"] for v in vals),
                "min_pass_rate": min(v["pass_rate"] for v in vals),
                "max_pass_rate": max(v["pass_rate"] for v in vals),
                "models": ", ".join(v["model"] for v in sorted(vals, key=lambda r: DISPLAY_ORDER.index(r["model"]))),
            }
        )
    return rows


def type_summary_rows(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        grouped[(row["task"], row["model_key"], row["test_type"])].append(row)
    out: list[dict[str, Any]] = []
    for (task, model_key, test_type), rows in grouped.items():
        cases = sum(int(r["n_cases"]) for r in rows)
        fails = sum(int(r["n_fails"]) for r in rows)
        display, family, subgroup = MODEL_META[model_key]
        out.append(
            {
                "task": task,
                "model_key": model_key,
                "model": display,
                "family": family,
                "subgroup": subgroup,
                "test_type": test_type,
                "n_cases": cases,
                "n_fails": fails,
                "pass_rate": 1.0 - fails / cases if cases else 0.0,
            }
        )
    return sorted(out, key=lambda r: (TASK_ORDER.index(r["task"]), DISPLAY_ORDER.index(r["model"]), TYPE_ORDER.index(r["test_type"]) if r["test_type"] in TYPE_ORDER else 99))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def write_pass_matrix(runs: list[dict[str, Any]], out: Path) -> None:
    task_set = {r["task"] for r in runs}
    model_set = {r["model"] for r in runs}
    tasks = [t for t in TASK_ORDER if t in task_set]
    models = [m for m in DISPLAY_ORDER if m in model_set]
    lookup = {(r["task"], r["model"]): r for r in runs}

    fig_w = 1.25 * len(tasks) + 2.5
    fig_h = 0.48 * len(models) + 1.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.invert_yaxis()

    for y, model in enumerate(models):
        for x, task in enumerate(tasks):
            run = lookup.get((task, model))
            if not run:
                ax.text(x, y, "NA", ha="center", va="center", fontsize=7, color="#b8b8b8")
                continue
            val = run["pass_rate"]
            shade = 0.92 - 0.55 * val
            color = (shade, shade, shade)
            ax.scatter([x], [y], s=520, marker="s", color=color, edgecolor="none")
            ax.text(x, y, _pct(val), ha="center", va="center", fontsize=7, color="white" if val > 0.62 else "#222222")
            if run["judge_fallback_rate"] >= 0.5:
                ax.plot([x - 0.34, x + 0.34], [y + 0.31, y + 0.31], color="#2b6cb0", lw=1.0)

    ax.set_xticks(range(len(tasks)), [t.upper() if t in {"qqp"} else t.capitalize() for t in tasks], fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("CheckList pass rate by model and suite", loc="left", fontsize=10, pad=12)
    ax.text(
        1.0,
        1.02,
        "Blue underline: judge fallback on at least half of examples.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_family_gap_plot(family_rows: list[dict[str, Any]], out: Path) -> None:
    by_task_group = {(r["task"], r["group"]): r for r in family_rows}
    rows = []
    for task in TASK_ORDER:
        llada = by_task_group.get((task, "LLaDA family"))
        ar = by_task_group.get((task, "AR family"))
        if llada and ar:
            rows.append((task, llada["macro_pass_rate"], ar["macro_pass_rate"], llada["macro_pass_rate"] - ar["macro_pass_rate"]))
    rows.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(rows) + 1.2), dpi=180)
    yvals = list(range(len(rows)))
    for y, (task, llada, ar, gap) in zip(yvals, rows):
        ax.plot([ar, llada], [y, y], color="#cccccc", lw=1.0)
        ax.scatter([ar], [y], color="#555555", s=22, zorder=3)
        ax.scatter([llada], [y], color="#2b6cb0", s=24, zorder=3)
        ax.text(max(llada, ar) + 0.012, y, f"{gap * 100:+.1f} pp", va="center", fontsize=7, color="#333333")
    ax.set_yticks(yvals, [r[0].upper() if r[0] == "qqp" else r[0].capitalize() for r in rows], fontsize=8)
    ax.set_xlim(0.25, 1.0)
    ax.set_xlabel("Macro pass rate", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_title("LLaDA family versus AR family", loc="left", fontsize=10, pad=8)
    ax.text(0.25, len(rows) - 0.1, "Gray = AR mean; blue = LLaDA/LLaDA-MoE mean.", fontsize=7, color="#555555")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_inv_plot(type_rows: list[dict[str, Any]], out: Path) -> None:
    keep = [r for r in type_rows if r["test_type"] == "INV" and r["model_key"] in {"llada_instruct", "llada_moe", "llama_instruct", "mistral", "qwen", "gemma", "olmo"}]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in keep:
        group = "LLaDA family" if row["model_key"] in {"llada_instruct", "llada_moe"} else "AR family"
        grouped[(row["task"], group)].append(row["pass_rate"])
    rows = []
    for task in TASK_ORDER:
        if (task, "LLaDA family") in grouped and (task, "AR family") in grouped:
            llada = mean(grouped[(task, "LLaDA family")])
            ar = mean(grouped[(task, "AR family")])
            rows.append((task, llada, ar, llada - ar))
    rows.sort(key=lambda x: x[3], reverse=True)

    fig, ax = plt.subplots(figsize=(7.0, 0.45 * len(rows) + 1.2), dpi=180)
    for y, (task, llada, ar, gap) in enumerate(rows):
        ax.plot([ar, llada], [y, y], color="#cccccc", lw=1.0)
        ax.scatter([ar], [y], color="#555555", s=22, zorder=3)
        ax.scatter([llada], [y], color="#2b6cb0", s=24, zorder=3)
        ax.text(max(llada, ar) + 0.012, y, f"{gap * 100:+.1f} pp", va="center", fontsize=7, color="#333333")
    ax.set_yticks(range(len(rows)), [r[0].upper() if r[0] == "qqp" else r[0].capitalize() for r in rows], fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("INV pass rate", fontsize=8)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_title("Invariance tests: LLaDA family versus AR family", loc="left", fontsize=10, pad=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def expansion_family_gap_rows(type_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in type_rows:
        if row["model_key"] in {"llada_instruct", "llada_moe"}:
            group = "LLaDA family"
        elif row["family"] == "AR":
            group = "AR family"
        else:
            continue
        grouped[(row["task"], row["test_type"], group)].append(row["pass_rate"])

    rows: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        for test_type in TYPE_ORDER:
            lvals = grouped.get((task, test_type, "LLaDA family"))
            avals = grouped.get((task, test_type, "AR family"))
            if not lvals or not avals:
                continue
            llada = mean(lvals)
            ar = mean(avals)
            rows.append(
                {
                    "task": task,
                    "test_type": test_type,
                    "llada_family_pass": llada,
                    "ar_family_pass": ar,
                    "gap_pp": 100.0 * (llada - ar),
                    "supports_llda": llada > ar,
                }
            )
    return rows


def read_previous_v5(root: Path) -> list[dict[str, Any]]:
    path = root / "results/lightning" / "contraction_analysis_v5" / "tables" / "dataset_summary.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            alpha = float(row["alpha_Diff"])
            gamma = float(row["gamma_AR"])
            rows.append(
                {
                    "task": row["dataset"],
                    "alpha_diff": alpha,
                    "gamma_ar": gamma,
                    "gamma_over_alpha": gamma / alpha if alpha else 0.0,
                    "margin_pct": 100.0 * ((gamma / alpha) - 1.0) if alpha else 0.0,
                    "theory_match": row.get("theory_match") == "True",
                    "winner": row.get("winner", ""),
                }
            )
    return rows


def write_test_type_gap_plot(gap_rows: list[dict[str, Any]], out: Path) -> None:
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in gap_rows)]
    colors = {"MFT": "#777777", "INV": "#2b6cb0", "DIR": "#9a5b00"}
    offsets = {"MFT": -0.22, "INV": 0.0, "DIR": 0.22}

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=180)
    for row in gap_rows:
        x = tasks.index(row["task"]) + offsets.get(row["test_type"], 0.0)
        ax.bar(x, row["gap_pp"], width=0.18, color=colors.get(row["test_type"], "#555555"), label=row["test_type"])
    ax.axhline(0, color="#222222", lw=0.7)
    ax.set_xticks(range(len(tasks)), [t.upper() if t == "qqp" else t.capitalize() for t in tasks], fontsize=8)
    ax.set_ylabel("LLaDA family - AR family pass rate (pp)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = []
    labels = []
    for test_type in TYPE_ORDER:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[test_type]))
        labels.append(test_type)
    ax.legend(handles, labels, frameon=False, ncol=3, loc="upper right", fontsize=8)
    ax.set_title("Where the expansion gap comes from", loc="left", fontsize=10, pad=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_proposition_bridge_plot(prev_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]], out: Path) -> None:
    inv_lookup = {r["task"]: r["gap_pp"] for r in gap_rows if r["test_type"] == "INV"}
    prev_lookup = {r["task"]: r for r in prev_rows}
    tasks = [t for t in TASK_ORDER if t in inv_lookup]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.0, 5.4), dpi=180, sharex=False)

    old_tasks = [t for t in ["qqp", "sentiment", "squad"] if t in prev_lookup]
    old_vals = [prev_lookup[t]["margin_pct"] for t in old_tasks]
    ax0.barh(range(len(old_tasks)), old_vals, color=["#2b6cb0" if v > 0 else "#777777" for v in old_vals], height=0.46)
    for y, val in enumerate(old_vals):
        ax0.text(val + (0.5 if val >= 0 else -0.5), y, f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax0.axvline(0, color="#222222", lw=0.7)
    ax0.set_yticks(range(len(old_tasks)), [t.upper() if t == "qqp" else t.capitalize() for t in old_tasks], fontsize=8)
    ax0.set_xlabel("Previous v5: gamma/alpha margin (%)", fontsize=8)
    ax0.set_title("Old proposition metric", loc="left", fontsize=10, pad=6)

    inv_vals = [inv_lookup[t] for t in tasks]
    ax1.barh(range(len(tasks)), inv_vals, color=["#2b6cb0" if v > 0 else "#777777" for v in inv_vals], height=0.46)
    for y, val in enumerate(inv_vals):
        ax1.text(val + (0.7 if val >= 0 else -0.7), y, f"{val:+.1f} pp", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax1.axvline(0, color="#222222", lw=0.7)
    ax1.set_yticks(range(len(tasks)), [t.upper() if t == "qqp" else t.capitalize() for t in tasks], fontsize=8)
    left = min(inv_vals + [0]) - 4.0
    right = max(inv_vals + [0]) + 6.0
    ax1.set_xlim(left, right)
    ax1.set_xlabel("Expansion proxy: INV pass-rate gap (pp)", fontsize=8)
    ax1.set_title("New expansion proxy", loc="left", fontsize=10, pad=6)

    for ax in (ax0, ax1):
        ax.grid(axis="x", color="#eeeeee", lw=0.6)
        ax.tick_params(axis="both", length=0, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.invert_yaxis()
    fig.suptitle("How the expanded runs compare to the paper proposition", x=0.02, y=0.99, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def latex_table(rows: list[list[str]], headers: list[str]) -> str:
    body = ["\\begin{tabular}{" + "l" + "r" * (len(headers) - 1) + "}", "\\toprule"]
    body.append(" & ".join(headers) + " \\\\")
    body.append("\\midrule")
    for row in rows:
        body.append(" & ".join(row) + " \\\\")
    body.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(body)


def write_reports(
    root: Path,
    runs: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    type_gap_rows: list[dict[str, Any]],
    prev_rows: list[dict[str, Any]],
) -> None:
    paper_dir = root / "paper"
    paper_dir.mkdir(exist_ok=True)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)
    model_rank = sorted(
        (
            {
                "model": model,
                "n_tasks": len(vals),
                "macro_pass_rate": mean(v["pass_rate"] for v in vals),
                "fallback_tasks": sum(v["judge_fallback_rate"] >= 0.5 for v in vals),
            }
            for model, vals in by_model.items()
        ),
        key=lambda r: r["macro_pass_rate"],
        reverse=True,
    )

    by_task_group = {(r["task"], r["group"]): r for r in family_rows}
    overall_gap_rows = []
    for task in TASK_ORDER:
        llada = by_task_group.get((task, "LLaDA family"))
        ar = by_task_group.get((task, "AR family"))
        if llada and ar:
            overall_gap_rows.append([task, llada["macro_pass_rate"], ar["macro_pass_rate"], llada["macro_pass_rate"] - ar["macro_pass_rate"]])

    inv_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in type_rows:
        if row["test_type"] != "INV":
            continue
        if row["model_key"] in {"llada_instruct", "llada_moe"}:
            inv_grouped[(row["task"], "LLaDA family")].append(row["pass_rate"])
        elif row["family"] == "AR":
            inv_grouped[(row["task"], "AR family")].append(row["pass_rate"])
    inv_rows = []
    for task in TASK_ORDER:
        if (task, "LLaDA family") in inv_grouped and (task, "AR family") in inv_grouped:
            llada = mean(inv_grouped[(task, "LLaDA family")])
            ar = mean(inv_grouped[(task, "AR family")])
            inv_rows.append([task, llada, ar, llada - ar])

    top_model = model_rank[0]
    llada_wins = sum(1 for _, _, _, gap in overall_gap_rows if gap > 0)
    inv_wins = sum(1 for _, _, _, gap in inv_rows if gap > 0)
    high_fallback = [r for r in runs if r["judge_fallback_rate"] >= 0.5]

    md_lines = [
        "# Expansion results",
        "",
        f"Runs aggregated: {len(runs)} model-suite cells across {len(set(r['task'] for r in runs))} suites.",
        f"Best macro model: {top_model['model']} ({_pct(top_model['macro_pass_rate'])}% over {top_model['n_tasks']} suites).",
        f"LLaDA-family macro pass rate beats the AR-family mean on {llada_wins}/{len(overall_gap_rows)} comparable suites.",
        f"On INV-only tests, LLaDA-family beats the AR-family mean on {inv_wins}/{len(inv_rows)} comparable suites.",
        "",
        "## Model macro rank",
        "",
        "| model | suites | macro pass | high-fallback suites |",
        "|---|---:|---:|---:|",
    ]
    for row in model_rank:
        md_lines.append(f"| {row['model']} | {row['n_tasks']} | {_pct(row['macro_pass_rate'])}% | {row['fallback_tasks']} |")

    md_lines.extend(["", "## LLaDA family vs AR family", "", "| suite | LLaDA family | AR family | gap |", "|---|---:|---:|---:|"])
    for task, llada, ar, gap in overall_gap_rows:
        md_lines.append(f"| {task} | {_pct(llada)}% | {_pct(ar)}% | {gap * 100:+.1f} pp |")

    md_lines.extend(["", "## INV-only robustness", "", "| suite | LLaDA family | AR family | gap |", "|---|---:|---:|---:|"])
    for task, llada, ar, gap in inv_rows:
        md_lines.append(f"| {task} | {_pct(llada)}% | {_pct(ar)}% | {gap * 100:+.1f} pp |")

    md_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The matrix supports a narrower, stronger claim: the LLaDA-family runs are consistently competitive and often better than AR baselines overall, with large gains on QQP and SQuAD. On INV-only tests, sentiment and SQuAD remain counterexamples. The corrected Dream reruns now make Dream a calibrated supporting/stress-test model. A broad class-level diffusion claim still needs careful caveats.",
            "",
            "For a Core A* submission, the paper should foreground the LLaDA-family result, use corrected Dream as calibrated supporting evidence, and add paired perturbation metrics or confidence intervals before making an architecture-wide claim.",
        ]
    )
    if high_fallback:
        md_lines.extend(["", "High judge-fallback cells (>=50%) are flagged in the matrix figure and should be treated as less diagnostic."])

    (paper_dir / "expansion_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    inv_gap = {r["task"]: r["gap_pp"] for r in type_gap_rows if r["test_type"] == "INV"}
    old = {r["task"]: r for r in prev_rows}
    prop_lines = [
        "# Proposition comparison",
        "",
        "The paper proposition predicts relative stability on semantics-preserving perturbations: for comparable perturbation magnitude, the diffusion model should move less than the AR model. The earlier v5 result tested that directly with gamma/alpha. The expansion mostly gives a pass-rate proxy, so it should be read as evidence about the proposition, not as the same estimator.",
        "",
        "## Previous v5 metric",
        "",
        "| suite | gamma/alpha | margin | interpretation |",
        "|---|---:|---:|---|",
    ]
    for task in ["qqp", "sentiment", "squad"]:
        if task not in old:
            continue
        row = old[task]
        interpretation = "supports relative-stability ratio" if row["gamma_over_alpha"] > 1 else "opposes ratio"
        if task == "sentiment":
            interpretation += "; previous analysis marked it metric-sensitive"
        prop_lines.append(
            f"| {task} | {row['gamma_over_alpha']:.3f} | {row['margin_pct']:+.1f}% | {interpretation} |"
        )
    prop_lines.extend(
        [
            "",
            "## Expansion INV proxy",
            "",
            "| suite | LLaDA-family INV gap | stance vs proposition |",
            "|---|---:|---|",
        ]
    )
    for task in [t for t in TASK_ORDER if t in inv_gap]:
        gap = inv_gap[task]
        if gap >= 3:
            stance = "supports"
        elif gap <= -3:
            stance = "counterevidence"
        else:
            stance = "near tie / inconclusive"
        prop_lines.append(f"| {task} | {gap:+.1f} pp | {stance} |")
    prop_lines.extend(
        [
            "",
            "## Bottom line",
            "",
            "The pass-rate proxy does not upgrade the proposition into a broad DLLM class law. It strengthens a narrower LLaDA-family claim overall, while sentiment and SQuAD remain counterexamples under the INV pass-rate proxy. QQP stays strong overall, but the family-level INV proxy is close to parity because Qwen and OLMo are strong AR baselines.",
            "",
            "For the paper, the safest statement is: LLaDA-family models are often more robust overall and sometimes more invariant, but the proposition needs paired gamma/alpha-style estimators on the expanded suites before it can be claimed as architecture-level evidence.",
        ]
    )
    (paper_dir / "proposition_comparison.md").write_text("\n".join(prop_lines) + "\n", encoding="utf-8")

    rank_rows = [[r["model"], str(r["n_tasks"]), _pct(r["macro_pass_rate"]), str(r["fallback_tasks"])] for r in model_rank]
    gap_latex = [[task, _pct(llada), _pct(ar), f"{gap * 100:+.1f}"] for task, llada, ar, gap in overall_gap_rows]
    inv_latex = [[task, _pct(llada), _pct(ar), f"{gap * 100:+.1f}"] for task, llada, ar, gap in inv_rows]
    tex = "\n\n".join(
        [
            "% Auto-generated by scripts/analysis/analyze_expansion_results.py",
            "\\section{Expansion results}",
            "The matrix reports CheckList-style runs across the sentiment, QQP, and SQuAD task suites. Reported values are pass rates in percent; family rows use a macro average across available models to avoid case-count domination by the larger CheckList suites.",
            "\\begin{figure}[H]\\centering\n\\includegraphics[width=0.98\\linewidth]{expansion_pass_matrix.png}\n\\caption{Expansion CheckList pass-rate matrix. Values are pass rates in percent. Blue underlines mark model-suite cells where the LLM judge was used on at least half of examples.}\\end{figure}",
            "\\begin{figure}[H]\\centering\n\\includegraphics[width=0.82\\linewidth]{expansion_llada_vs_ar.png}\n\\caption{Overall macro pass rate: LLaDA-family mean versus AR-family mean.}\\end{figure}",
            "\\begin{figure}[H]\\centering\n\\includegraphics[width=0.82\\linewidth]{expansion_inv_llada_vs_ar.png}\n\\caption{INV-only pass rate: LLaDA-family mean versus AR-family mean.}\\end{figure}",
            "\\begin{figure}[H]\\centering\n\\includegraphics[width=0.88\\linewidth]{expansion_test_type_gap.png}\n\\caption{Expansion pass-rate gap by test type. Positive values favour the LLaDA family; negative values favour AR models.}\\end{figure}",
            "\\begin{figure}[H]\\centering\n\\includegraphics[width=0.88\\linewidth]{proposition_bridge.png}\n\\caption{Comparison between the paper's original v5 proposition metric and the expansion INV proxy. The two panels use different units: the original uses relative gamma/alpha margin, while the expansion uses pass-rate gap.}\\end{figure}",
            "\\begin{table}[H]\\centering\n" + latex_table(rank_rows, ["model", "suites", "macro pass", "fallback"]) + "\n\\caption{Model-level macro pass rate across available expansion suites. Fallback counts mark suites where the LLM judge was used on at least half of examples.}\\end{table}",
            "\\begin{table}[H]\\centering\n" + latex_table(gap_latex, ["suite", "LLaDA family", "AR family", "gap"]) + "\n\\caption{LLaDA-family (LLaDA-8B and LLaDA-MoE) macro pass rate versus AR-family macro pass rate. Gap is LLaDA family minus AR family, in percentage points.}\\end{table}",
            "\\begin{table}[H]\\centering\n" + latex_table(inv_latex, ["suite", "LLaDA family", "AR family", "gap"]) + "\n\\caption{INV-only robustness comparison.}\\end{table}",
            "\\paragraph{Takeaway.} The expanded matrix supports a LLaDA-family robustness claim more strongly than a broad diffusion-class claim. Corrected Dream reruns are now usable as calibrated supporting/stress-test evidence.",
        ]
    )
    (paper_dir / "expansion_results.tex").write_text(tex + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    root = args.root
    out_dir = root / "results" / "expansion" / "analysis"

    runs, test_rows = load_all_runs(root)
    runs = sorted(runs, key=lambda r: (TASK_ORDER.index(r["task"]), DISPLAY_ORDER.index(r["model"])))
    test_rows = sorted(test_rows, key=lambda r: (TASK_ORDER.index(r["task"]), DISPLAY_ORDER.index(r["model"]), r["test_type"], r["test_name"]))
    type_rows = type_summary_rows(test_rows)

    groups = [
        ("LLaDA family", {"llada_instruct", "llada_moe"}),
        ("Masked-DLM incl. Dream", {"llada_instruct", "llada_moe", "dream"}),
        ("AR family", {"llama_instruct", "mistral", "qwen", "gemma", "olmo"}),
    ]
    family_rows: list[dict[str, Any]] = []
    for name, members in groups:
        family_rows.extend(family_dataset_rows(runs, name, members))
    family_rows = sorted(family_rows, key=lambda r: (TASK_ORDER.index(r["task"]), r["group"]))

    write_csv(out_dir / "model_dataset_summary.csv", runs)
    write_csv(out_dir / "test_summary.csv", test_rows)
    write_csv(out_dir / "test_type_summary.csv", type_rows)
    write_csv(out_dir / "family_dataset_summary.csv", family_rows)
    gap_rows = expansion_family_gap_rows(type_rows)
    prev_rows = read_previous_v5(root)
    write_csv(out_dir / "family_test_type_gaps.csv", gap_rows)
    write_csv(out_dir / "previous_v5_dataset_summary.csv", prev_rows)

    write_pass_matrix(runs, root / "paper" / "fig" / "expansion_pass_matrix.png")
    write_family_gap_plot(family_rows, root / "paper" / "fig" / "expansion_llada_vs_ar.png")
    write_inv_plot(type_rows, root / "paper" / "fig" / "expansion_inv_llada_vs_ar.png")
    write_test_type_gap_plot(gap_rows, root / "paper" / "fig" / "expansion_test_type_gap.png")
    write_proposition_bridge_plot(prev_rows, gap_rows, root / "paper" / "fig" / "proposition_bridge.png")
    write_reports(root, runs, family_rows, type_rows, gap_rows, prev_rows)

    print(f"Wrote {len(runs)} run rows and {len(test_rows)} test rows to {out_dir}")
    print("Wrote paper analysis summaries and figures in paper/fig/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
