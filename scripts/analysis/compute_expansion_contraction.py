#!/usr/bin/env python3
"""compute_expansion_contraction.py

Multi-model contraction/amplification analysis for expansion CheckList runs,
using **v5's exact functions** (imported from analyze_contraction_amplification.py)
so methodology is byte-identical to the published v5 numbers.

Usage:
    python scripts/analysis/compute_expansion_contraction.py
    python scripts/analysis/compute_expansion_contraction.py --models llada_instruct llama_instruct
    python scripts/analysis/compute_expansion_contraction.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# SBERT model is already in the local HF cache; avoid hub metadata calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# --- Import v5's authoritative functions ----------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_contraction_amplification import (
    _compute_pair_metrics,
    _qqp_split,
    _rows_to_suite_groups,
    _load_suite_metadata,
    _select_group_rows,
    _row_input_text,
    _group_by_subtest,
    _load_jsonl,
    _embed,
)

EXPANSION_ROOT = Path("results/expansion/checklist")
PHASE2_ROOT = Path("results/checklist")
CHECKLIST_ROOT = PHASE2_ROOT  # for _load_suite_metadata (suites live under here)
OUT_DIR = Path("results/expansion/contraction")
EMB_CACHE = OUT_DIR / "emb_cache"

MODEL_META: dict[str, tuple[str, str]] = {
    "llada_instruct": ("LLaDA-8B",     "DLM"),
    "llada_moe":      ("LLaDA-MoE",    "DLM"),
    "dream":          ("Dream-7B",     "DLM"),
    "llama_instruct": ("Llama-3.1-8B", "AR"),
    "mistral":        ("Mistral-7B",   "AR"),
    "qwen":           ("Qwen2.5-7B",   "AR"),
    "gemma":          ("Gemma-2-9B",   "AR"),
    "olmo":           ("OLMo-2-7B",    "AR"),
}

MODEL_ALIASES = {
    "dream_rerun": "dream",
}

# Phase-2 fallback dirs for LLaDA-8B / Llama-3.1 (not under expansion root)
PHASE2_DIRS: dict[str, dict[str, str]] = {
    "llada_instruct": {"sentiment": "llada_rerun_fixed",
                       "qqp":       "llada_rerun_fixed",
                       "squad":     "llada_rerun_fixed"},
    "llama_instruct": {"sentiment": "llama_rerun_fixed",
                       "qqp":       "llama_rerun_fixed",
                       "squad":     "llama_rerun_fixed"},
}

# v5 defaults (from results/lightning/contraction_analysis_v5/.../config)
SAMPLES_PER_SUBTEST = 99999
D_INPUT_FLOOR = 0.02
MFT_D_INPUT_FLOOR = 0.05
SEMANTIC_SIM_THRESHOLD = 0.0
SEED = 42

MODEL_ORDER = ["llada_instruct", "llada_moe", "dream",
               "llama_instruct", "mistral", "qwen", "gemma", "olmo"]


# ---------------------------------------------------------------------------
# Cached embedding helper (avoids re-embedding identical texts across models)
# ---------------------------------------------------------------------------

class EmbCache:
    """Per-(task,model_key,field) embedding cache stored as {text: vec} npz."""

    def __init__(self, root: Path, model_name: str, device: str, batch_size: int):
        self.root = root
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, texts: list[str]) -> np.ndarray:
        """Return embeddings for `texts` (order preserved). Persists per-text vectors."""
        cache_path = self.root / f"{key}.npz"
        cached: dict[str, np.ndarray] = {}
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            if "texts" in z and "embeddings" in z:
                ct, ce = z["texts"], z["embeddings"]
                for i in range(len(ct)):
                    cached[str(ct[i])] = ce[i]

        missing_unique = list(dict.fromkeys(t for t in texts if t not in cached))
        if missing_unique:
            print(f"    [{key}] embedding {len(missing_unique)} new texts "
                  f"({len(cached)} cached) ...")
            new_embs = _embed(missing_unique, self.model_name, self.device, self.batch_size)
            for t, e in zip(missing_unique, new_embs):
                cached[t] = e
            all_texts = list(cached.keys())
            all_embs = np.stack([cached[t] for t in all_texts]).astype(np.float32)
            np.savez_compressed(cache_path,
                                texts=np.array(all_texts, dtype=object),
                                embeddings=all_embs)
        return np.stack([cached[t] for t in texts]).astype(np.float32)


# ---------------------------------------------------------------------------
# Run model directories discovery
# ---------------------------------------------------------------------------

def discover_model_dirs(task: str, models_filter: list[str] | None) -> dict[str, Path]:
    model_dir_map: dict[str, Path] = {}
    task_dir = EXPANSION_ROOT / task
    if task_dir.exists():
        for d in task_dir.iterdir():
            if not d.is_dir():
                continue
            raw_model_key = d.name
            model_key = MODEL_ALIASES.get(raw_model_key, raw_model_key)
            if model_key in MODEL_META:
                if model_key not in model_dir_map or raw_model_key.endswith("_rerun"):
                    model_dir_map[model_key] = d
    for mk, tmap in PHASE2_DIRS.items():
        if mk not in model_dir_map and task in tmap:
            fb = PHASE2_ROOT / task / tmap[task]
            if fb.exists():
                model_dir_map[mk] = fb
    if models_filter:
        model_dir_map = {k: v for k, v in model_dir_map.items() if k in models_filter}
    return model_dir_map


def find_jsonl(model_dir: Path) -> Path | None:
    fs = list(model_dir.glob("examples_full_*.jsonl"))
    return fs[0] if fs else None


# ---------------------------------------------------------------------------
# v5-style sample selection per (task, test_name) using suite group_sizes
# ---------------------------------------------------------------------------

def select_rows_per_test(
    rows_by_test: dict[str, list[dict[str, Any]]],
    task_test_meta: dict[str, dict[str, Any]],
    samples_per_subtest: int,
    seed: int,
) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    """Mirror v5's per-test row selection. Picks the first N group ids deterministically
    (sample_group_ids uses random.sample over [0..n_groups) with the given seed).
    """
    import random
    out: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    rng = random.Random(seed)
    for test_name, rows in rows_by_test.items():
        meta = task_test_meta.get(test_name, {})
        group_sizes = meta.get("group_sizes", [])
        tt = meta.get("test_type", "MFT")
        if not group_sizes:
            continue
        groups = _rows_to_suite_groups(rows, group_sizes)
        n_complete = len(groups)
        if n_complete == 0:
            continue
        ids = list(range(n_complete))
        if len(ids) > samples_per_subtest:
            ids = sorted(rng.sample(ids, samples_per_subtest))
        selected = _select_group_rows(groups, ids)
        out[test_name] = (tt, selected)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _embed_dedup(texts: list[str], model_name: str, device: str, batch_size: int
                 ) -> dict[str, np.ndarray]:
    """Embed unique texts in one big GPU call; return {text: vec} map."""
    uniq = list(dict.fromkeys(texts))
    if not uniq:
        return {}
    print(f"    [GPU] embedding {len(uniq)} unique texts (bs={batch_size}) ...")
    arr = _embed(uniq, model_name, device, batch_size)
    return {t: arr[i] for i, t in enumerate(uniq)}


def run(tasks: list[str], device: str, embedding_model: str, batch_size: int,
        models_filter: list[str] | None, samples_per_subtest: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suite_meta = _load_suite_metadata(CHECKLIST_ROOT)

    records: list[dict[str, Any]] = []

    for task in tasks:
        print(f"\n=== Task: {task} ===")
        if task not in suite_meta:
            print(f"  [skip] no suite for '{task}'")
            continue
        task_test_meta = suite_meta[task]
        model_dir_map = discover_model_dirs(task, models_filter)
        model_keys = [m for m in MODEL_ORDER if m in model_dir_map]
        print(f"  models found: {model_keys}")

        # Load + select rows for every model first so we know the full text pool.
        per_model_selected: dict[str, dict[str, tuple[str, list[dict[str, Any]]]]] = {}
        for mk in model_keys:
            jsonl_path = find_jsonl(model_dir_map[mk])
            if jsonl_path is None:
                print(f"  [{mk}] no jsonl, skip")
                continue
            print(f"  [{mk}] loading {jsonl_path.name}")
            rows = _load_jsonl(jsonl_path)
            rows_by_test = _group_by_subtest(rows)
            per_model_selected[mk] = select_rows_per_test(
                rows_by_test, task_test_meta, samples_per_subtest, seed)

        # ----- Phase 1: embed all unique input texts ONCE for the task -----
        all_input_texts: list[str] = []
        for mk, sel in per_model_selected.items():
            for _tn, (_tt, sampled) in sel.items():
                for r in sampled:
                    all_input_texts.append(_row_input_text(task, r))
        print(f"  -- input pool: {len(all_input_texts)} total, "
              f"{len(set(all_input_texts))} unique --")
        in_map = _embed_dedup(all_input_texts, embedding_model, device, batch_size)

        # QQP per-side: embed Q1/Q2 split texts once per task
        q1_map: dict[str, np.ndarray] = {}
        q2_map: dict[str, np.ndarray] = {}
        if task == "qqp":
            q1_texts = [_qqp_split(t)[0] or t for t in in_map.keys()]
            q2_texts = [_qqp_split(t)[1] or t for t in in_map.keys()]
            print(f"  -- QQP side pools: Q1 unique={len(set(q1_texts))}, "
                  f"Q2 unique={len(set(q2_texts))} --")
            q1_map = _embed_dedup(q1_texts, embedding_model, device, batch_size)
            q2_map = _embed_dedup(q2_texts, embedding_model, device, batch_size)

        # ----- Phase 2: per model, embed unique responses ONCE, then run pairs -----
        for mk, sel in per_model_selected.items():
            display_name, model_type = MODEL_META[mk]
            all_resp_texts: list[str] = []
            for _tn, (_tt, sampled) in sel.items():
                for r in sampled:
                    all_resp_texts.append(r.get("model_response", "") or "")
            print(f"  [{mk}] response pool: {len(all_resp_texts)} total, "
                  f"{len(set(all_resp_texts))} unique")
            resp_map = _embed_dedup(all_resp_texts, embedding_model, device, batch_size)

            for test_name, (tt, sampled) in sel.items():
                if not sampled:
                    continue
                input_texts = [_row_input_text(task, r) for r in sampled]
                response_texts = [r.get("model_response", "") or "" for r in sampled]
                inp_embs  = np.stack([in_map[t]  for t in input_texts]).astype(np.float32)
                resp_embs = np.stack([resp_map[t] for t in response_texts]).astype(np.float32)
                side_a = side_b = None
                if task == "qqp":
                    q1t = [_qqp_split(t)[0] or t for t in input_texts]
                    q2t = [_qqp_split(t)[1] or t for t in input_texts]
                    side_a = np.stack([q1_map[t] for t in q1t]).astype(np.float32)
                    side_b = np.stack([q2_map[t] for t in q2t]).astype(np.float32)

                pair_recs = _compute_pair_metrics(
                    task_name=task,
                    rows=sampled,
                    input_embs=inp_embs,
                    response_embs=resp_embs,
                    semantic_similarity_threshold=SEMANTIC_SIM_THRESHOLD,
                    test_type=tt,
                    d_input_floor=D_INPUT_FLOOR,
                    mft_d_input_floor=MFT_D_INPUT_FLOOR,
                    input_embs_side_a=side_a,
                    input_embs_side_b=side_b,
                )
                ratios = [p["ratio"] for p in pair_recs if p.get("ratio") is not None]
                d_outs = [p["d_output"] for p in pair_recs if p.get("d_output") is not None]
                d_ins  = [p["d_input"]  for p in pair_recs if p.get("d_input")  is not None]
                if not ratios:
                    continue
                capability = task_test_meta.get(test_name, {}).get("capability", "")
                records.append({
                    "model_key": mk,
                    "display_name": display_name,
                    "model_type": model_type,
                    "task": task,
                    "test_name": test_name,
                    "test_type": tt,
                    "capability": capability,
                    "n_pairs": len(ratios),
                    "mean_ratio": float(np.mean(ratios)),
                    "median_ratio": float(np.median(ratios)),
                    "delta_perp": float(np.mean(d_outs)),
                    "mean_d_input": float(np.mean(d_ins)) if d_ins else 0.0,
                })

    _write_outputs(records)


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:80]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_subtest_matrix(records: list[dict[str, Any]]) -> None:
    """v5-style subtest_summary: per task, one wide CSV with subtests as rows
    and each model contributing columns {mean, median, n_pairs}.
    Also a long-form CSV (one row per (task, test, model)) for easy filtering.
    """
    # Long form
    long_path = OUT_DIR / "expansion_contraction_subtests_long.csv"
    long_cols = ["task", "test_name", "test_type", "capability",
                 "model_key", "display_name", "model_type",
                 "n_pairs", "mean_ratio", "median_ratio",
                 "delta_perp", "mean_d_input"]
    with long_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=long_cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in long_cols})
    print(f"Wrote {long_path}")

    # Wide form per task (subtests × models matrices)
    tasks = sorted({r["task"] for r in records})
    models_in_order = [m for m in MODEL_ORDER
                       if any(r["model_key"] == m for r in records)]
    for task in tasks:
        task_rows = [r for r in records if r["task"] == task]
        if not task_rows:
            continue
        # index by (test_name) collecting per-model fields
        by_test: dict[str, dict[str, Any]] = {}
        for r in task_rows:
            test_name = r["test_name"]
            entry = by_test.setdefault(test_name, {
                "test_type": r["test_type"],
                "capability": r.get("capability", ""),
            })
            mk = r["model_key"]
            entry[f"{mk}_mean"]   = r["mean_ratio"]
            entry[f"{mk}_median"] = r["median_ratio"]
            entry[f"{mk}_npairs"] = r["n_pairs"]
            entry[f"{mk}_dperp"]  = r["delta_perp"]
        header = ["test_name", "test_type", "capability"]
        for mk in models_in_order:
            disp, _ = MODEL_META[mk]
            header += [f"{disp}_mean", f"{disp}_median",
                       f"{disp}_npairs", f"{disp}_dperp"]
        wide_path = OUT_DIR / f"expansion_contraction_subtests_{task}.csv"
        with wide_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for tn in sorted(by_test.keys()):
                e = by_test[tn]
                row = [tn, e["test_type"], e["capability"]]
                for mk in models_in_order:
                    row += [
                        f"{e.get(f'{mk}_mean', ''):.4f}" if isinstance(e.get(f'{mk}_mean'), float) else "",
                        f"{e.get(f'{mk}_median', ''):.4f}" if isinstance(e.get(f'{mk}_median'), float) else "",
                        e.get(f"{mk}_npairs", ""),
                        f"{e.get(f'{mk}_dperp', ''):.4f}" if isinstance(e.get(f'{mk}_dperp'), float) else "",
                    ]
                w.writerow(row)
        print(f"Wrote {wide_path}")


def _write_outputs(records: list[dict[str, Any]]) -> None:
    if not records:
        print("\nNo records.")
        return

    csv_path = OUT_DIR / "expansion_contraction_per_test.csv"
    fieldnames = ["model_key", "display_name", "model_type", "task",
                  "test_name", "test_type", "capability",
                  "n_pairs", "mean_ratio", "median_ratio",
                  "delta_perp", "mean_d_input"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"\nWrote {csv_path}")

    # Per-task subtest matrix: rows = subtests, columns = each model's mean_ratio
    # Plus test_type, capability, and per-model median_ratio + n_pairs side-cols.
    _write_subtest_matrix(records)

    # pair-weighted aggregation per (model, task)
    ratio_num:  dict[tuple[str, str], float] = defaultdict(float)
    dperp_num:  dict[tuple[str, str], float] = defaultdict(float)
    pair_count: dict[tuple[str, str], int]   = defaultdict(int)
    meta: dict[str, tuple[str, str]] = {}
    for r in records:
        key = (r["model_key"], r["task"])
        n = r["n_pairs"]
        ratio_num[key]  += r["mean_ratio"] * n
        dperp_num[key]  += r["delta_perp"] * n
        pair_count[key] += n
        meta[r["model_key"]] = (r["display_name"], r["model_type"])

    def agg_ratio(mk, t):
        n = pair_count.get((mk, t), 0)
        return ratio_num[(mk, t)] / n if n else None

    def agg_dperp(mk, t):
        n = pair_count.get((mk, t), 0)
        return dperp_num[(mk, t)] / n if n else None

    tasks_seen = [t for t in ["sentiment", "qqp", "squad"]
                  if any(r["task"] == t for r in records)]
    models_seen = [m for m in MODEL_ORDER if m in meta]

    sum_csv = OUT_DIR / "expansion_contraction_summary.csv"
    with sum_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_key", "display_name", "model_type"] + tasks_seen + ["avg"])
        for mk in models_seen:
            disp, mtype = meta[mk]
            row_vals = [agg_ratio(mk, t) for t in tasks_seen]
            valid = [v for v in row_vals if v is not None]
            avg = float(np.mean(valid)) if valid else None
            w.writerow([mk, disp, mtype]
                       + [f"{v:.4f}" if v is not None else "" for v in row_vals]
                       + [f"{avg:.4f}" if avg is not None else ""])
    print(f"Wrote {sum_csv}")

    # per-model table to stdout
    print("\n" + "=" * 70)
    print("Per-model pair-weighted mean ratio  (alpha=DLM, gamma=AR) — INV+DIR+MFT")
    print("=" * 70)
    hdr = f"{'Model':<16} {'Type':<5}"
    for t in tasks_seen:
        hdr += f"  {t.upper():>9}"
    hdr += f"  {'AVG':>9}"
    print(hdr)
    print("-" * len(hdr))
    for mk in models_seen:
        disp, mtype = meta[mk]
        row_vals = [agg_ratio(mk, t) for t in tasks_seen]
        valid = [v for v in row_vals if v is not None]
        avg = float(np.mean(valid)) if valid else None
        cells = [f"{v:.3f}" if v is not None else "--" for v in row_vals]
        avg_str = f"{avg:.3f}" if avg is not None else "--"
        print(f"{disp:<16} {mtype:<5}  " + "  ".join(f"{c:>9}" for c in cells)
              + f"  {avg_str:>9}")

    # Table-3 style
    dlm_keys = [k for k in models_seen if meta[k][1] == "DLM"]
    ar_keys  = [k for k in models_seen if meta[k][1] == "AR"]
    print("\n" + "=" * 80)
    print("Table-3 style: avg_alpha (DLMs) vs avg_gamma (ARs) per task")
    print(f"{'task':<12} {'a_Diff':>8} {'g_AR':>8} {'g-a':>8} {'g/a':>7}"
          f" {'Dp_Diff':>9} {'Dp_AR':>9} {'Dp_gap':>9}")
    print("-" * 80)
    summary_rows = []
    for t in tasks_seen:
        dlm_r = [v for v in (agg_ratio(k, t) for k in dlm_keys) if v is not None]
        ar_r  = [v for v in (agg_ratio(k, t) for k in ar_keys)  if v is not None]
        dlm_d = [v for v in (agg_dperp(k, t) for k in dlm_keys) if v is not None]
        ar_d  = [v for v in (agg_dperp(k, t) for k in ar_keys)  if v is not None]
        if not dlm_r or not ar_r:
            continue
        alpha = float(np.mean(dlm_r))
        gamma = float(np.mean(ar_r))
        dp_d  = float(np.mean(dlm_d)) if dlm_d else float("nan")
        dp_a  = float(np.mean(ar_d))  if ar_d  else float("nan")
        dp_gap = dp_d - dp_a  # v5's "Δ⊥ gap" = Δ⊥_DLM − Δ⊥_AR
        print(f"{t:<12} {alpha:>8.3f} {gamma:>8.3f} {gamma - alpha:>+8.3f}"
              f" {gamma / alpha:>7.3f} {dp_d:>9.3f} {dp_a:>9.3f} {dp_gap:>+9.3f}")
        summary_rows.append({
            "task": t, "alpha_Diff": alpha, "gamma_AR": gamma,
            "gamma_minus_alpha": gamma - alpha, "gamma_over_alpha": gamma / alpha,
            "Delta_perp_Diff": dp_d, "Delta_perp_AR": dp_a,
            "Delta_perp_gap": dp_gap,
        })

    # Persist task-level summary CSV
    task_csv = OUT_DIR / "expansion_contraction_task_summary.csv"
    if summary_rows:
        with task_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"Wrote {task_csv}")


# ---------------------------------------------------------------------------

def rerun_from_csv(per_test_csv: Path) -> None:
    """Re-aggregate all output tables from an existing per-test CSV, keeping only
    models still in MODEL_META.
    Per-pair statistics are model-local, so exclusion only changes the pooling."""
    records: list[dict[str, Any]] = []
    dropped: set[str] = set()
    with per_test_csv.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["model_key"] not in MODEL_META:
                dropped.add(r["model_key"])
                continue
            r["n_pairs"] = int(r["n_pairs"])
            for k in ("mean_ratio", "median_ratio", "delta_perp", "mean_d_input"):
                r[k] = float(r[k]) if r[k] != "" else 0.0
            records.append(r)
    if dropped:
        print(f"Dropped models not in MODEL_META: {sorted(dropped)}")
    _write_outputs(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "qqp", "squad"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-subtest", type=int, default=SAMPLES_PER_SUBTEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--models", nargs="+", default=None,
                        help="restrict to these model_keys, e.g. llada_instruct llama_instruct")
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="skip embedding; re-aggregate outputs from an existing per-test CSV")
    args = parser.parse_args()
    if args.from_csv:
        rerun_from_csv(args.from_csv)
        return
    run(args.tasks, args.device, args.embedding_model, args.batch_size,
        args.models, args.samples_per_subtest, args.seed)


if __name__ == "__main__":
    main()
