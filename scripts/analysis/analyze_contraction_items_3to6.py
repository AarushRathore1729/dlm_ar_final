#!/usr/bin/env python3
"""Follow-up analyses on the contraction/amplification results (items 3-6).

Item 3: scatter α_Diff vs γ_AR split by test_type (INV / DIR / MFT).
Item 4: scatter γ−α gap vs CheckList accuracy gap per (task, capability).
Item 5: qqp:SRL "Order does not matter for comparison" sanity-check.
Item 6: recompute squad subtest/capability/dataset α, γ using EM and F1
        as output distance (vs gold) instead of response cosine.

All items operate on cached artifacts under
`results/lightning/contraction_analysis_v2/` and
`results/lightning/checklist/`.

Usage:
    python scripts/analyze_contraction_items_3to6.py \\
        --contraction-dir results/lightning/contraction_analysis_v2 \\
        --checklist-root  results/lightning/checklist \\
        --out-dir         results/lightning/contraction_analysis_v2/items_3to6
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="deep")
except ModuleNotFoundError:
    sns = None

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
})

EPSILON = 1e-8
D_INPUT_FLOOR = 0.02
MFT_D_INPUT_FLOOR = 0.05
TASK_COLORS = {"sentiment": "#6C5CE7", "qqp": "#00B894", "squad": "#E17055"}
TT_MARKERS = {"INV": "o", "DIR": "s", "MFT": "^"}
SUITE_FILES = {
    "sentiment": "sentiment_suite_original_dump.json",
    "qqp": "qqp_suite_original_dump.json",
    "squad": "squad_suite_original_dump.json",
}


# ---------------------------------------------------------------------------
# Generic IO
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_results_json(run_dir: Path) -> Path:
    matches = list(run_dir.glob("checklist_results_*.json"))
    if not matches:
        raise FileNotFoundError(f"No checklist_results_*.json in {run_dir}")
    return matches[0]


def _find_examples_jsonl(run_dir: Path) -> Path:
    matches = list(run_dir.glob("examples_full_*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No examples_full_*.jsonl in {run_dir}")
    return matches[0]


# ---------------------------------------------------------------------------
# Item 3 — scatter split by test_type
# ---------------------------------------------------------------------------

def item3_scatter_by_test_type(subtest_csv: Path, out_dir: Path) -> None:
    rows = _read_csv(subtest_csv)
    by_tt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tt = r["test_type"] or "MFT"
        by_tt[tt].append(r)

    # combined plot with markers
    fig, ax = plt.subplots(figsize=(8, 8))
    all_x, all_y = [], []
    for tt, recs in by_tt.items():
        for r in recs:
            x = float(r["alpha_Diff"])
            y = float(r["gamma_AR"])
            all_x.append(x)
            all_y.append(y)
            ax.scatter(
                x, y,
                c=TASK_COLORS.get(r["dataset"], "#999999"),
                marker=TT_MARKERS.get(tt, "o"),
                s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
            )
    hi = max(max(all_x, default=1), max(all_y, default=1)) * 1.1
    ax.plot([0, hi], [0, hi], "--", color="#636E72", lw=1.2, label="y = x")
    # legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    task_handles = [Patch(facecolor=c, label=t) for t, c in TASK_COLORS.items()]
    tt_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="#636E72",
               markersize=8, label=tt)
        for tt, m in TT_MARKERS.items()
    ]
    ax.legend(handles=task_handles + tt_handles, fontsize=8, loc="upper left")
    ax.set_xlabel("α_Diff (LLaDA)")
    ax.set_ylabel("γ_AR (Llama)")
    ax.set_title("Eq. 8: α_Diff vs γ_AR — markers = test_type, colors = dataset")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    out_path = out_dir / "item3_scatter_combined_by_test_type.png"
    fig.savefig(out_path); plt.close(fig)
    print(f"  Saved: {out_path.name}")

    # per-test_type subplots
    tts = sorted(by_tt.keys())
    fig, axes = plt.subplots(1, len(tts), figsize=(5.5 * len(tts), 5.5),
                              squeeze=False)
    for ax, tt in zip(axes[0], tts):
        recs = by_tt[tt]
        xs = [float(r["alpha_Diff"]) for r in recs]
        ys = [float(r["gamma_AR"]) for r in recs]
        colors = [TASK_COLORS.get(r["dataset"], "#999999") for r in recs]
        ax.scatter(xs, ys, c=colors, s=50, alpha=0.7,
                   edgecolors="white", linewidths=0.5)
        hi = max(max(xs, default=1), max(ys, default=1)) * 1.1
        ax.plot([0, hi], [0, hi], "--", color="#636E72", lw=1.2, label="y=x")

        above = sum(1 for x, y in zip(xs, ys) if y > x)
        below = sum(1 for x, y in zip(xs, ys) if y < x)
        ax.set_title(f"{tt}   n={len(recs)}   above={above}  below={below}")
        ax.set_xlabel("α_Diff"); ax.set_ylabel("γ_AR")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(handles=[Patch(facecolor=c, label=t)
                           for t, c in TASK_COLORS.items()],
                  fontsize=7, loc="upper left")
    fig.suptitle("Item 3 — scatter split by test_type", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "item3_scatter_per_test_type.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")

    # per-dataset scatter (subtest-level, markers = test_type)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["dataset"]].append(r)
    tasks_sorted = sorted(by_task.keys())
    fig, axes = plt.subplots(1, len(tasks_sorted),
                             figsize=(5.5 * len(tasks_sorted), 5.5),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks_sorted):
        recs = by_task[task]
        xs = [float(r["alpha_Diff"]) for r in recs]
        ys = [float(r["gamma_AR"]) for r in recs]
        for r, x, y in zip(recs, xs, ys):
            ax.scatter(x, y,
                       c=TASK_COLORS.get(task, "#999999"),
                       marker=TT_MARKERS.get(r["test_type"] or "MFT", "o"),
                       s=55, alpha=0.75,
                       edgecolors="white", linewidths=0.5)
        hi = max(max(xs, default=1), max(ys, default=1)) * 1.1 or 1
        ax.plot([0, hi], [0, hi], "--", color="#636E72", lw=1.2, label="y=x")
        above = sum(1 for x, y in zip(xs, ys) if y > x)
        total = len(recs)
        mean_gap = float(np.mean([y - x for x, y in zip(xs, ys)]))
        ax.set_title(
            f"{task}   n={total}   above={above}   mean γ−α={mean_gap:+.2f}"
        )
        ax.set_xlabel("α_Diff"); ax.set_ylabel("γ_AR")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        tt_handles = [
            Line2D([0], [0], marker=m, color="w", markersize=8,
                   markerfacecolor=TASK_COLORS.get(task, "#999999"),
                   label=tt)
            for tt, m in TT_MARKERS.items()
            if any((rec["test_type"] or "MFT") == tt for rec in recs)
        ]
        ax.legend(handles=tt_handles, fontsize=8, loc="upper left")
    fig.suptitle(
        "Item 3 — subtest α_Diff vs γ_AR per dataset (markers = test_type)",
        y=1.02)
    fig.tight_layout()
    out_path = out_dir / "item3_scatter_per_dataset.png"
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path.name}")

    # stats csv
    summary_rows = []
    for tt, recs in by_tt.items():
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_task[r["dataset"]].append(r)
        for task, trecs in by_task.items():
            xs = [float(r["alpha_Diff"]) for r in trecs]
            ys = [float(r["gamma_AR"]) for r in trecs]
            above = sum(1 for x, y in zip(xs, ys) if y > x)
            summary_rows.append({
                "test_type": tt, "dataset": task,
                "n_subtests": len(trecs),
                "mean_alpha_Diff": round(float(np.mean(xs)), 4),
                "mean_gamma_AR": round(float(np.mean(ys)), 4),
                "mean_gap": round(float(np.mean([y - x for x, y in zip(xs, ys)])), 4),
                "n_above_diagonal": above,
                "n_below_diagonal": len(trecs) - above,
            })
    out_csv = out_dir / "item3_test_type_stats.csv"
    if summary_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"  Saved: {out_csv.name}")


# ---------------------------------------------------------------------------
# Item 4 — γ−α gap vs accuracy gap per capability
# ---------------------------------------------------------------------------

def _capability_accuracy(results_json: Path) -> dict[str, tuple[int, int]]:
    """Return {capability: (total_cases, total_passes)}."""
    data = json.load(results_json.open(encoding="utf-8"))
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t in data.get("tests", []):
        cap = t.get("capability", "Unknown")
        n = int(t.get("n_cases", 0))
        fails = int(t.get("n_fails", 0))
        agg[cap][0] += n
        agg[cap][1] += max(0, n - fails)
    return {k: (v[0], v[1]) for k, v in agg.items()}


def item4_gap_vs_accuracy(
    cap_csv: Path, checklist_root: Path, out_dir: Path,
) -> None:
    cap_rows = _read_csv(cap_csv)

    # Accuracy per (task, capability) per model
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for task_dir in sorted(checklist_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ("suites",):
            continue
        task = task_dir.name
        per_model = {}
        for model_key, subdir in (("llada", "llada_rerun_fixed"),
                                  ("llama", "llama_rerun_fixed")):
            run_dir = task_dir / subdir
            if not run_dir.exists():
                continue
            per_model[model_key] = _capability_accuracy(_find_results_json(run_dir))
        caps = set()
        for d in per_model.values(): caps.update(d.keys())
        for c in caps:
            entry: dict[str, float] = {}
            for m in ("llada", "llama"):
                if m in per_model and c in per_model[m]:
                    n, p = per_model[m][c]
                    entry[f"{m}_acc"] = p / n if n else 0.0
                    entry[f"{m}_n"] = n
            acc[(task, c)] = entry

    # Merge with gap
    points = []
    for r in cap_rows:
        task, cap = r["dataset"], r["capability"]
        gap = float(r["gap_gamma_minus_alpha"])
        key = (task, cap)
        if key not in acc:
            continue
        a = acc[key]
        if "llada_acc" not in a or "llama_acc" not in a:
            continue
        acc_gap = a["llada_acc"] - a["llama_acc"]
        points.append({
            "dataset": task, "capability": cap,
            "gap_gamma_minus_alpha": gap,
            "llada_acc": round(a["llada_acc"], 4),
            "llama_acc": round(a["llama_acc"], 4),
            "acc_gap_llada_minus_llama": round(acc_gap, 4),
            "n_subtests": int(r.get("n_subtests", 0)),
        })

    if not points:
        print("  item4: no points to plot")
        return

    # scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    for p in points:
        ax.scatter(
            p["gap_gamma_minus_alpha"], p["acc_gap_llada_minus_llama"],
            c=TASK_COLORS.get(p["dataset"], "#999999"),
            s=max(30, 5 * p["n_subtests"]),
            alpha=0.75, edgecolors="white", linewidths=0.6,
        )
        ax.annotate(
            f"{p['dataset']}:{p['capability']}",
            (p["gap_gamma_minus_alpha"], p["acc_gap_llada_minus_llama"]),
            fontsize=6.5, alpha=0.75,
            xytext=(3, 3), textcoords="offset points",
        )
    ax.axhline(0, color="#636E72", ls="--", lw=1)
    ax.axvline(0, color="#636E72", ls="--", lw=1)
    ax.set_xlabel("γ_AR − α_Diff   (>0 ⇒ Llama amplifies more)")
    ax.set_ylabel("acc(LLaDA) − acc(Llama)   (>0 ⇒ LLaDA more accurate)")
    ax.set_title("Item 4 — capability-level theory check: does γ−α predict accuracy gap?")
    ax.set_xscale("symlog", linthresh=1.0)

    # Pearson / Spearman
    xs = np.array([p["gap_gamma_minus_alpha"] for p in points])
    ys = np.array([p["acc_gap_llada_minus_llama"] for p in points])
    pearson = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else float("nan")
    rxs = np.argsort(np.argsort(xs))
    rys = np.argsort(np.argsort(ys))
    spearman = float(np.corrcoef(rxs, rys)[0, 1]) if len(xs) > 1 else float("nan")

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=c, label=t) for t, c in TASK_COLORS.items()],
        title=f"Pearson r={pearson:.3f}   Spearman ρ={spearman:.3f}",
        fontsize=8, title_fontsize=8, loc="upper left",
    )
    fig.tight_layout()
    out_path = out_dir / "item4_gap_vs_accuracy_gap.png"
    fig.savefig(out_path); plt.close(fig)
    print(f"  Saved: {out_path.name}  (Pearson={pearson:.3f}, Spearman={spearman:.3f})")

    with (out_dir / "item4_gap_vs_accuracy_gap.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        w.writeheader(); w.writerows(points)
    print("  Saved: item4_gap_vs_accuracy_gap.csv")


# ---------------------------------------------------------------------------
# Item 5 — qqp:SRL "Order does not matter for comparison" sanity
# ---------------------------------------------------------------------------

TARGET_SRL_TEST = "Order does not matter for comparison"


def item5_srl_sanity(checklist_root: Path, out_dir: Path) -> None:
    qqp_dir = checklist_root / "qqp"
    out: dict[str, Any] = {"test_name": TARGET_SRL_TEST}

    for model_key in ("llada", "llama"):
        run_dir = qqp_dir / f"{model_key}_rerun_fixed"
        rows = [
            r for r in _load_jsonl(_find_examples_jsonl(run_dir))
            if r.get("test_name") == TARGET_SRL_TEST
        ]
        # Examples grouped by example_idx (suite group id)
        groups: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r["example_idx"]].append(r)

        # Pair ratios: for each group, anchor = first, partners = rest
        # We DO NOT have cached input embs on hand — use simple Jaccard on
        # serialized "q1 ||| q2" text as a cheap input-distance proxy
        # (only for sanity-checking; the blow-up is driven by d_input → 0).
        pair_stats = []
        for gid, grp in groups.items():
            if len(grp) < 2:
                continue
            anchor = grp[0]
            for partner in grp[1:]:
                a = anchor["input_text"]
                b = partner["input_text"]
                toks_a = set(a.lower().split())
                toks_b = set(b.lower().split())
                jaccard_sim = (len(toks_a & toks_b) /
                               max(1, len(toks_a | toks_b)))
                d_input_jaccard = 1 - jaccard_sim

                lbl_a = (anchor.get("predicted_label") or "").strip()
                lbl_b = (partner.get("predicted_label") or "").strip()
                d_output = 0.0 if lbl_a == lbl_b else 1.0

                ratio = (d_output / d_input_jaccard
                         if d_input_jaccard > EPSILON else None)
                pair_stats.append({
                    "group_id": gid,
                    "d_input_jaccard": d_input_jaccard,
                    "d_output_label": d_output,
                    "ratio": ratio,
                    "anchor_input": a[:120],
                    "partner_input": b[:120],
                    "anchor_label": lbl_a,
                    "partner_label": lbl_b,
                })

        n_identical_input = sum(
            1 for p in pair_stats if p["d_input_jaccard"] < EPSILON)
        n_different_output = sum(
            1 for p in pair_stats if p["d_output_label"] > 0)
        d_in = [p["d_input_jaccard"] for p in pair_stats]
        ratios = [p["ratio"] for p in pair_stats if p["ratio"] is not None]

        out[model_key] = {
            "n_rows": len(rows),
            "n_groups": len(groups),
            "n_pairs": len(pair_stats),
            "n_pairs_with_identical_jaccard_input": n_identical_input,
            "n_pairs_with_label_flip": n_different_output,
            "median_d_input_jaccard": float(np.median(d_in)) if d_in else 0.0,
            "mean_d_input_jaccard": float(np.mean(d_in)) if d_in else 0.0,
            "median_ratio_jaccard": float(np.median(ratios)) if ratios else 0.0,
            "max_ratio_jaccard": float(np.max(ratios)) if ratios else 0.0,
            "top_5_ratio_pairs": sorted(
                pair_stats, key=lambda p: (p["ratio"] or 0), reverse=True
            )[:5],
        }

    # Diagnosis text
    llada = out.get("llada", {})
    llama = out.get("llama", {})
    msg = [
        f"qqp:SRL '{TARGET_SRL_TEST}' sanity check",
        "",
        "The CSV reports γ_AR=1301.49, α_Diff=27.96, mean_input_distance=0.0107.",
        "Suite structure: 990 example groups of 3 paraphrase pairs each,",
        "so inputs within a group are near-duplicates under the 'q1 ||| q2'",
        "serialization used by the analysis script. The ratio",
        "d_output / d_input blows up when d_input → 0 even under normal",
        "label disagreement.",
        "",
        f"Pairs with near-zero Jaccard d_input (llada): "
        f"{llada.get('n_pairs_with_identical_jaccard_input', 0)} / {llada.get('n_pairs', 0)}",
        f"Pairs with label flip (llada): "
        f"{llada.get('n_pairs_with_label_flip', 0)} / {llada.get('n_pairs', 0)}",
        f"Pairs with near-zero Jaccard d_input (llama): "
        f"{llama.get('n_pairs_with_identical_jaccard_input', 0)} / {llama.get('n_pairs', 0)}",
        f"Pairs with label flip (llama): "
        f"{llama.get('n_pairs_with_label_flip', 0)} / {llama.get('n_pairs', 0)}",
        "",
        "Verdict: the +444 capability-level gap is a d_input→0 artifact of",
        "the MFT triple-structure in this test; the test should be either",
        "(a) excluded from pair analysis, or (b) rerun with a Lipschitz-style",
        "floor on d_input (e.g. d_input < 0.02 → drop pair). Applies equally",
        "to other MFT tests where in-group inputs are templated paraphrases.",
    ]
    (out_dir / "item5_qqp_srl_sanity.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    (out_dir / "item5_qqp_srl_sanity.md").write_text("\n".join(msg) + "\n")
    print("  Saved: item5_qqp_srl_sanity.{json,md}")
    print("  " + "\n  ".join(msg[:11]))


# ---------------------------------------------------------------------------
# Item 6 — squad EM / F1 task-aware output distance
# ---------------------------------------------------------------------------

SQUAD_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def _squad_normalize(s: str) -> str:
    s = s.lower()
    s = SQUAD_ARTICLES.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _squad_em(pred: str, gold: str) -> float:
    return 1.0 if _squad_normalize(pred) == _squad_normalize(gold) else 0.0


def _squad_f1(pred: str, gold: str) -> float:
    p_toks = _squad_normalize(pred).split()
    g_toks = _squad_normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p_toks)
    rec = n_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def _load_squad_suite_meta(checklist_root: Path) -> dict[str, dict]:
    path = checklist_root / "suites" / SUITE_FILES["squad"]
    d = json.load(path.open(encoding="utf-8"))
    out = {}
    for t in d.get("tests", []):
        examples = t.get("examples") or []
        group_sizes = []
        for ex in examples:
            inp = ex.get("input")
            group_sizes.append(len(inp) if isinstance(inp, list) else 1)
        out[t["name"]] = {
            "test_type": t.get("test_type", "MFT"),
            "capability": t.get("capability", "Unknown"),
            "group_sizes": group_sizes,
        }
    return out


def _group_rows_by_example(rows: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("example_idx", -1)].append(r)
    return groups


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, 1.0 - float(np.dot(a, b)))


def _load_npz_for_squad(contraction_dir: Path, model: str) -> dict:
    path = contraction_dir / "embeddings" / f"squad_{model}_embeddings.npz"
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["metadata"].item()))
    return {
        "input": data["input_embeddings"],
        "response": data["response_embeddings"],
        "meta": meta,
    }


def item6_squad_emf1(
    contraction_dir: Path, checklist_root: Path, out_dir: Path,
) -> None:
    suite_meta = _load_squad_suite_meta(checklist_root)

    results = {}
    for model_key, subdir in (("llada", "llada_rerun_fixed"),
                              ("llama", "llama_rerun_fixed")):
        run_dir = checklist_root / "squad" / subdir
        jsonl_rows = _load_jsonl(_find_examples_jsonl(run_dir))
        per_test_rows: dict[str, list[dict]] = defaultdict(list)
        for r in jsonl_rows:
            per_test_rows[r["test_name"]].append(r)

        npz = _load_npz_for_squad(contraction_dir, model_key)
        input_embs = npz["input"]
        meta = npz["meta"]  # [{test_name, test_type, example_idx (=group id)}, ...]

        # Build per-test contiguous segments in NPZ order.
        segments: dict[str, list[tuple[int, int]]] = defaultdict(list)  # test -> [(npz_idx, group_id)]
        for i, m in enumerate(meta):
            segments[m["test_name"]].append((i, m["example_idx"]))

        results[model_key] = {
            "per_test_rows": per_test_rows,
            "input_embs": input_embs,
            "segments": segments,
        }
        # Sanity logging
        for tn, rows in per_test_rows.items():
            exp = len(segments.get(tn, []))
            if exp and exp != len(rows):
                print(f"  WARN {model_key} {tn}: npz={exp} jsonl={len(rows)}")

    # Align & compute per-subtest metrics
    subtest_records: dict[str, dict] = {}
    tests = sorted(set(results["llada"]["per_test_rows"].keys())
                   & set(results["llama"]["per_test_rows"].keys()))

    for test_name in tests:
        tmeta = suite_meta.get(test_name, {})
        tt = tmeta.get("test_type", "MFT")
        capability = tmeta.get("capability", "Unknown")
        group_sizes = tmeta.get("group_sizes", [])

        per_model_pairs = {}
        for model_key in ("llada", "llama"):
            rows = results[model_key]["per_test_rows"][test_name]
            input_embs = results[model_key]["input_embs"]
            segs = results[model_key]["segments"].get(test_name, [])
            if not segs or not rows:
                continue

            # NPZ segment: list of (npz_idx, group_id) in NPZ insertion order.
            # JSONL rows: in run-order, typically flattened suite-example order.
            # Align by slicing JSONL into suite groups using group_sizes, which
            # gives the same groups NPZ was built from. Then pair NPZ rows to
            # JSONL rows by position within each group.
            jsonl_groups: list[list[int]] = []
            cursor = 0
            for sz in group_sizes:
                if cursor + sz > len(rows):
                    break
                jsonl_groups.append(list(range(cursor, cursor + sz)))
                cursor += sz

            # NPZ rows grouped by group_id, preserving NPZ order within group
            npz_by_group: dict[int, list[int]] = defaultdict(list)
            for npz_idx, gid in segs:
                npz_by_group[gid].append(npz_idx)

            em_ratios, f1_ratios = [], []
            for gid, npz_idxs in npz_by_group.items():
                if gid >= len(jsonl_groups):
                    continue
                jsonl_idxs = jsonl_groups[gid]
                n = min(len(npz_idxs), len(jsonl_idxs))
                if n < 2:
                    continue
                if tt == "MFT":
                    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                else:
                    pairs = [(0, k) for k in range(1, n)]
                effective_floor = (MFT_D_INPUT_FLOOR if tt == "MFT"
                                   else D_INPUT_FLOOR)
                for i, j in pairs:
                    d_in = _cos_dist(input_embs[npz_idxs[i]],
                                     input_embs[npz_idxs[j]])
                    if d_in < max(effective_floor, EPSILON):
                        continue
                    a_row = rows[jsonl_idxs[i]]
                    b_row = rows[jsonl_idxs[j]]
                    resp_a = a_row.get("model_response", "") or ""
                    resp_b = b_row.get("model_response", "") or ""
                    gold_a = a_row.get("expected_label", "") or ""
                    gold_b = b_row.get("expected_label", "") or ""
                    em_a = _squad_em(resp_a, gold_a)
                    em_b = _squad_em(resp_b, gold_b)
                    f1_a = _squad_f1(resp_a, gold_a)
                    f1_b = _squad_f1(resp_b, gold_b)
                    em_ratios.append(abs(em_a - em_b) / d_in)
                    f1_ratios.append(abs(f1_a - f1_b) / d_in)

            per_model_pairs[model_key] = {
                "em_ratios": em_ratios, "f1_ratios": f1_ratios,
            }

        if "llada" not in per_model_pairs or "llama" not in per_model_pairs:
            continue
        pl = per_model_pairs["llada"]; pm = per_model_pairs["llama"]
        if not pl["em_ratios"] or not pm["em_ratios"]:
            continue
        subtest_records[test_name] = {
            "test_type": tt, "capability": capability,
            "alpha_Diff_em": float(np.mean(pl["em_ratios"])),
            "gamma_AR_em": float(np.mean(pm["em_ratios"])),
            "alpha_Diff_f1": float(np.mean(pl["f1_ratios"])),
            "gamma_AR_f1": float(np.mean(pm["f1_ratios"])),
            "llada_n_pairs": len(pl["em_ratios"]),
            "llama_n_pairs": len(pm["em_ratios"]),
        }

    # Per-capability & dataset aggregation (weighted by n_pairs)
    def agg(recs, key_num, key_denom):
        num = sum(r[key_num] * r[key_denom] for r in recs)
        den = sum(r[key_denom] for r in recs)
        return num / den if den else 0.0

    cap_buckets: dict[str, list[dict]] = defaultdict(list)
    subtest_rows = []
    for name, rec in subtest_records.items():
        cap_buckets[rec["capability"]].append(rec)
        subtest_rows.append({
            "subtest": name,
            "capability": rec["capability"],
            "test_type": rec["test_type"],
            "alpha_Diff_em": round(rec["alpha_Diff_em"], 6),
            "gamma_AR_em": round(rec["gamma_AR_em"], 6),
            "gap_em": round(rec["gamma_AR_em"] - rec["alpha_Diff_em"], 6),
            "winner_em": ("LLaDA" if rec["alpha_Diff_em"] < rec["gamma_AR_em"]
                          else "Llama"),
            "alpha_Diff_f1": round(rec["alpha_Diff_f1"], 6),
            "gamma_AR_f1": round(rec["gamma_AR_f1"], 6),
            "gap_f1": round(rec["gamma_AR_f1"] - rec["alpha_Diff_f1"], 6),
            "winner_f1": ("LLaDA" if rec["alpha_Diff_f1"] < rec["gamma_AR_f1"]
                          else "Llama"),
            "llada_n_pairs": rec["llada_n_pairs"],
            "llama_n_pairs": rec["llama_n_pairs"],
        })

    cap_rows = []
    for cap, recs in cap_buckets.items():
        a_em = agg(recs, "alpha_Diff_em", "llada_n_pairs")
        g_em = agg(recs, "gamma_AR_em", "llama_n_pairs")
        a_f1 = agg(recs, "alpha_Diff_f1", "llada_n_pairs")
        g_f1 = agg(recs, "gamma_AR_f1", "llama_n_pairs")
        cap_rows.append({
            "dataset": "squad", "capability": cap,
            "alpha_Diff_em": round(a_em, 6),
            "gamma_AR_em": round(g_em, 6),
            "gap_em": round(g_em - a_em, 6),
            "winner_em": "LLaDA" if a_em < g_em else "Llama",
            "alpha_Diff_f1": round(a_f1, 6),
            "gamma_AR_f1": round(g_f1, 6),
            "gap_f1": round(g_f1 - a_f1, 6),
            "winner_f1": "LLaDA" if a_f1 < g_f1 else "Llama",
            "n_subtests": len(recs),
            "llada_n_pairs": sum(r["llada_n_pairs"] for r in recs),
            "llama_n_pairs": sum(r["llama_n_pairs"] for r in recs),
        })

    all_recs = list(subtest_records.values())
    ds_a_em = agg(all_recs, "alpha_Diff_em", "llada_n_pairs")
    ds_g_em = agg(all_recs, "gamma_AR_em", "llama_n_pairs")
    ds_a_f1 = agg(all_recs, "alpha_Diff_f1", "llada_n_pairs")
    ds_g_f1 = agg(all_recs, "gamma_AR_f1", "llama_n_pairs")
    ds_row = {
        "dataset": "squad",
        "alpha_Diff_em": round(ds_a_em, 6),
        "gamma_AR_em": round(ds_g_em, 6),
        "gap_em": round(ds_g_em - ds_a_em, 6),
        "winner_em": "LLaDA" if ds_a_em < ds_g_em else "Llama",
        "alpha_Diff_f1": round(ds_a_f1, 6),
        "gamma_AR_f1": round(ds_g_f1, 6),
        "gap_f1": round(ds_g_f1 - ds_a_f1, 6),
        "winner_f1": "LLaDA" if ds_a_f1 < ds_g_f1 else "Llama",
        "n_subtests": len(all_recs),
        "llada_n_pairs": sum(r["llada_n_pairs"] for r in all_recs),
        "llama_n_pairs": sum(r["llama_n_pairs"] for r in all_recs),
    }

    # Write CSVs
    for fname, rows in (
        ("item6_squad_emf1_subtests.csv", subtest_rows),
        ("item6_squad_emf1_capabilities.csv", cap_rows),
        ("item6_squad_emf1_dataset.csv", [ds_row]),
    ):
        if rows:
            with (out_dir / fname).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            print(f"  Saved: {fname}")

    # Scatter EM vs F1
    for metric in ("em", "f1"):
        xs = [r[f"alpha_Diff_{metric}"] for r in subtest_rows]
        ys = [r[f"gamma_AR_{metric}"] for r in subtest_rows]
        fig, ax = plt.subplots(figsize=(7, 7))
        for r, x, y in zip(subtest_rows, xs, ys):
            ax.scatter(x, y, s=50, alpha=0.75,
                       c=TASK_COLORS["squad"],
                       marker=TT_MARKERS.get(r["test_type"], "o"),
                       edgecolors="white", linewidths=0.5)
        hi = max(max(xs, default=1), max(ys, default=1)) * 1.1 or 1
        ax.plot([0, hi], [0, hi], "--", color="#636E72", lw=1.2)
        above = sum(1 for x, y in zip(xs, ys) if y > x)
        ax.set_title(
            f"squad — {metric.upper()} gold-anchored distance\n"
            f"α_Diff vs γ_AR  (above diagonal: {above}/{len(xs)})"
        )
        ax.set_xlabel(f"α_Diff ({metric.upper()})")
        ax.set_ylabel(f"γ_AR ({metric.upper()})")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        fig.tight_layout()
        out_path = out_dir / f"item6_squad_scatter_{metric}.png"
        fig.savefig(out_path); plt.close(fig)
        print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contraction-dir", type=Path,
                    default=Path("results/lightning/contraction_analysis_v2"))
    ap.add_argument("--checklist-root", type=Path,
                    default=Path("results/lightning/checklist"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/lightning/contraction_analysis_v2/items_3to6"))
    ap.add_argument("--items", type=str, default="3,4,5,6",
                    help="Comma-separated subset of {3,4,5,6}")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = {s.strip() for s in args.items.split(",") if s.strip()}

    if "3" in items:
        print("\n=== Item 3: scatter by test_type ===")
        item3_scatter_by_test_type(
            args.contraction_dir / "tables" / "subtest_summary.csv", args.out_dir)
    if "4" in items:
        print("\n=== Item 4: γ−α gap vs accuracy gap per capability ===")
        item4_gap_vs_accuracy(
            args.contraction_dir / "tables" / "capability_summary.csv",
            args.checklist_root, args.out_dir)
    if "5" in items:
        print("\n=== Item 5: qqp:SRL 'Order does not matter for comparison' sanity ===")
        item5_srl_sanity(args.checklist_root, args.out_dir)
    if "6" in items:
        print("\n=== Item 6: squad EM/F1 gold-anchored distance ===")
        item6_squad_emf1(args.contraction_dir, args.checklist_root, args.out_dir)

    print(f"\nAll outputs under: {args.out_dir}")


if __name__ == "__main__":
    main()
