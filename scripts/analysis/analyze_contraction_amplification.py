#!/usr/bin/env python3
"""Contraction vs Amplification analysis across LLaDA and Llama.

Implements the prof's exact formula (Eq. 8):

    γ_AR(x_i)    = d(f_AR(x'_i),  f_AR(x_i))  / d(x'_i, x_i)
    α_Diff(x_i)  = d(f_Diff(x'_i), f_Diff(x_i)) / d(x'_i, x_i)

where d() is cosine distance and f() maps input → response embedding.

For INV/DIR tests (paired perturbations):
  - Embeds both inputs and responses
  - Computes per-pair perturbation coefficients
  - Aggregates per subtest (median)
  - Scatter: α_Diff vs γ_AR with y=x diagonal + scaling factor fit

For MFT tests (no natural pairs):
  - Computes intra-subtest response spread (complementary metric)

Usage (CPU is fine — embeddings are small texts):
    python scripts/analyze_contraction_amplification.py \\
        --checklist-root results/checklist \\
        --out-dir results/contraction_analysis \\
        --samples-per-subtest 10 \\
        --device cpu

Usage on GPU:
    python scripts/analyze_contraction_amplification.py \\
        --checklist-root results/checklist \\
        --out-dir results/contraction_analysis \\
        --samples-per-subtest 50 \\
        --embedding-model /path/to/local/all-MiniLM-L6-v2 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
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
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

EPSILON = 1e-8

SUITE_FILES = {
    "sentiment": "sentiment_suite_original_dump.json",
    "qqp": "qqp_suite_original_dump.json",
    "squad": "squad_suite_original_dump.json",
}


def _example_input_width(ex: dict[str, Any]) -> int:
    inp = ex.get("input")
    if isinstance(inp, list):
        return len(inp)
    return 1


def _load_suite_metadata(checklist_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load authoritative test metadata from suite dumps.

    Returns {task: {test_name: {"test_type": str, "group_sizes": list[int]}}}.
    """
    suites_dir = checklist_root / "suites"
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for task_name, fname in SUITE_FILES.items():
        suite_path = suites_dir / fname
        if not suite_path.exists():
            continue
        with suite_path.open(encoding="utf-8") as f:
            suite = json.load(f)
        out[task_name] = {
            t["name"]: {
                "test_type": t.get("test_type", "MFT"),
                "capability": t.get("capability", "Unknown"),
                "group_sizes": [
                    _example_input_width(ex) for ex in (t.get("examples") or [])
                ],
            }
            for t in suite.get("tests", [])
        }
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _detect_model_key(run_dir_name: str) -> str | None:
    low = run_dir_name.lower()
    if "llada" in low:
        return "llada"
    if "llama" in low:
        return "llama"
    return None


def _find_run_pairs(checklist_root: Path) -> dict[str, dict[str, Path]]:
    """Find matching llada/llama run dirs per task.

    Returns {task: {"llada": Path, "llama": Path}}.
    """
    tasks: dict[str, dict[str, Path]] = {}
    for task_dir in sorted(checklist_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if task_name in ("suites",):
            continue
        models: dict[str, Path] = {}
        collisions: dict[str, list[Path]] = defaultdict(list)
        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            key = _detect_model_key(run_dir.name)
            if key:
                jsonl_files = list(run_dir.glob("examples_full_*.jsonl"))
                if jsonl_files:
                    collisions[key].append(jsonl_files[0])
        for key, candidates in collisions.items():
            if len(candidates) > 1:
                joined = ", ".join(str(p.parent.name) for p in candidates)
                raise ValueError(
                    f"Multiple {key} runs found for task '{task_name}': {joined}. "
                    "Keep one run per model/task or make the selection explicit."
                )
            models[key] = candidates[0]
        if "llada" in models and "llama" in models:
            tasks[task_name] = models
    return tasks


# ---------------------------------------------------------------------------
# Sampling — preserves paired structure for INV/DIR
# ---------------------------------------------------------------------------

def _group_by_subtest(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get("test_name", "unknown")
        groups[key].append(row)
    return dict(groups)


def _sample_example_groups(
    group_keys: list[int], n_groups: int, seed: int
) -> list[int]:
    """Sample up to n_groups suite example ids."""
    rng = random.Random(seed)
    if len(group_keys) > n_groups:
        group_keys = rng.sample(group_keys, n_groups)
    return sorted(group_keys)


def _sample_rows_flat(
    rows: list[dict], n: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def _sample_group_ids(n_groups: int, n_pick: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    group_ids = list(range(n_groups))
    if len(group_ids) > n_pick:
        group_ids = rng.sample(group_ids, n_pick)
    return sorted(group_ids)


def _rows_to_suite_groups(rows: list[dict], group_sizes: list[int]) -> list[list[dict]]:
    """Slice flattened JSONL rows into suite example groups using suite metadata."""
    groups: list[list[dict]] = []
    cursor = 0
    for size in group_sizes:
        if cursor + size > len(rows):
            break
        groups.append(rows[cursor : cursor + size])
        cursor += size
    return groups


def _select_group_rows(groups: list[list[dict]], picked_ids: list[int]) -> list[dict]:
    out: list[dict] = []
    for group_id in picked_ids:
        if 0 <= group_id < len(groups):
            for row in groups[group_id]:
                row_copy = dict(row)
                row_copy["suite_example_idx"] = group_id
                row_copy["example_idx"] = group_id
                out.append(row_copy)
    return out


_QQP_Q1_PREFIX = "Q1:"
_QQP_Q2_PREFIX = "Q2:"


def _qqp_split(text: str) -> tuple[str, str]:
    """Parse a qqp input into (q1, q2). Returns ('', '') on failure.

    Recognises both `Q1:\\n...Q2:\\n...` (current suite format) and the
    legacy `q1 ||| q2` serialization.
    """
    if not text:
        return "", ""
    if "|||" in text:
        a, _, b = text.partition("|||")
        return a.strip(), b.strip()
    lo = text.find(_QQP_Q1_PREFIX)
    hi = text.find(_QQP_Q2_PREFIX)
    if lo != -1 and hi != -1 and hi > lo:
        q1 = text[lo + len(_QQP_Q1_PREFIX):hi].strip()
        q2 = text[hi + len(_QQP_Q2_PREFIX):].strip()
        return q1, q2
    return "", ""


_LABEL_STRIP = " \t\n\r.,!?;:\"'"


def _normalize_label(response: str) -> str:
    return (response or "").strip().strip(_LABEL_STRIP).lower()


def _serialize_input(task_name: str, item: Any) -> str:
    if task_name == "sentiment":
        return str(item)
    if task_name == "qqp":
        if isinstance(item, dict):
            q1 = item.get("question1", item.get("q1", ""))
            q2 = item.get("question2", item.get("q2", ""))
            return f"{q1} ||| {q2}"
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return f"{item[0]} ||| {item[1]}"
        return str(item)
    if task_name == "squad":
        if isinstance(item, dict):
            return f"Passage: {item.get('passage', '')}\nQuestion: {item.get('question', '')}"
        return str(item)
    return str(item)


def _row_input_text(task_name: str, row: dict[str, Any]) -> str:
    text = (row.get("input_text") or "").strip()
    if text:
        return text
    structured = row.get("input_structured")
    if structured is not None:
        return _serialize_input(task_name, structured)
    return ""


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(texts: list[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing 'sentence-transformers'. Install with:\n"
            "  pip install sentence-transformers"
        ) from exc
    model = SentenceTransformer(model_name, device=device)
    if device != "cpu":
        model.half()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, 1.0 - float(np.dot(a, b)))


# ---------------------------------------------------------------------------
# INV/DIR: per-pair ratio computation (prof's Eq. 8)
# ---------------------------------------------------------------------------

def _compute_pair_ratios(
    rows: list[dict],
    input_embs: np.ndarray,
    response_embs: np.ndarray,
) -> list[dict]:
    """Compute perturbation coefficient per (original, perturbed) pair.

    ratio = d(f(x'), f(x)) / d(x', x)
    """
    by_example: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_example[row.get("example_idx", i)].append(i)

    pair_records: list[dict] = []
    for ex_idx, idxs in by_example.items():
        if len(idxs) < 2:
            continue
        orig_idx = idxs[0]
        for pert_pos in idxs[1:]:
            d_input = _cosine_dist(input_embs[orig_idx], input_embs[pert_pos])
            d_output = _cosine_dist(response_embs[orig_idx], response_embs[pert_pos])

            if d_input < EPSILON:
                ratio = None
            else:
                ratio = d_output / d_input

            pair_records.append({
                "example_idx": ex_idx,
                "d_input": d_input,
                "d_output": d_output,
                "ratio": ratio,
            })
    return pair_records


def _task_output_distance(
    task_name: str,
    response_a: str,
    response_b: str,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    if task_name == "sentiment":
        mapping = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        a = mapping.get(_normalize_label(response_a))
        b = mapping.get(_normalize_label(response_b))
        if a is not None and b is not None:
            return abs(a - b)
    elif task_name == "qqp":
        mapping = {"not_duplicate": 0.0, "duplicate": 1.0,
                   "no": 0.0, "yes": 1.0, "0": 0.0, "1": 1.0}
        a = mapping.get(_normalize_label(response_a))
        b = mapping.get(_normalize_label(response_b))
        if a is not None and b is not None:
            return abs(a - b)

    return _cosine_dist(emb_a, emb_b)


def _compute_pair_metrics(
    task_name: str,
    rows: list[dict],
    input_embs: np.ndarray,
    response_embs: np.ndarray,
    semantic_similarity_threshold: float,
    test_type: str = "MFT",
    d_input_floor: float = 0.0,
    mft_d_input_floor: float | None = None,
    input_embs_side_a: np.ndarray | None = None,
    input_embs_side_b: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute pair metrics per suite example.

    For INV/DIR: anchor pairing (first row vs each subsequent row) — matches
      CheckList semantics where perturbations are defined *relative to an
      original*.
    For MFT (symmetric variant groups): all C(k, 2) unordered pairs — no
      designated original, so every variant pair is equally valid.

    Pairs with d_input < d_input_floor are dropped (Lipschitz floor to guard
    against d_input → 0 blow-up in templated MFT tests).
    """
    by_example: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_example[row.get("example_idx", i)].append(i)

    pair_records: list[dict[str, Any]] = []
    for ex_idx, idxs in by_example.items():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs)

        if test_type == "MFT":
            pair_iter = [(idxs[i], idxs[j])
                         for i in range(len(idxs))
                         for j in range(i + 1, len(idxs))]
        else:
            pair_iter = [(idxs[0], idxs[k]) for k in range(1, len(idxs))]

        effective_floor = (mft_d_input_floor
                           if test_type == "MFT" and mft_d_input_floor is not None
                           else d_input_floor)

        for a_idx, b_idx in pair_iter:
            input_sim = float(np.dot(input_embs[a_idx], input_embs[b_idx]))
            if input_sim < semantic_similarity_threshold:
                continue

            if (input_embs_side_a is not None
                    and input_embs_side_b is not None):
                d_a = _cosine_dist(
                    input_embs_side_a[a_idx], input_embs_side_a[b_idx])
                d_b = _cosine_dist(
                    input_embs_side_b[a_idx], input_embs_side_b[b_idx])
                d_input = max(d_a, d_b)
            else:
                d_input = _cosine_dist(input_embs[a_idx], input_embs[b_idx])
            if d_input < max(effective_floor, EPSILON):
                continue

            response_a = rows[a_idx].get("model_response", "") or ""
            response_b = rows[b_idx].get("model_response", "") or ""
            d_output = _task_output_distance(
                task_name,
                response_a,
                response_b,
                response_embs[a_idx],
                response_embs[b_idx],
            )

            ratio = d_output / d_input

            pair_records.append({
                "example_idx": ex_idx,
                "d_input": d_input,
                "input_similarity": input_sim,
                "d_output": d_output,
                "ratio": ratio,
                "orig_input": _row_input_text(task_name, rows[a_idx]),
                "pert_input": _row_input_text(task_name, rows[b_idx]),
                "orig_response": response_a,
                "pert_response": response_b,
            })

    return pair_records


# ---------------------------------------------------------------------------
# MFT: intra-subtest spread (complementary)
# ---------------------------------------------------------------------------

def _mft_spread(embs: np.ndarray) -> float:
    if len(embs) < 2:
        return 0.0
    sim = embs @ embs.T
    n = len(embs)
    mask = ~np.eye(n, dtype=bool)
    return max(0.0, 1.0 - float(sim[mask].mean()))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _scatter_with_diagonal(
    x: list[float],
    y: list[float],
    labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    task_colors: dict[str, str] | None = None,
    tasks: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    if task_colors and tasks:
        for task_name, color in task_colors.items():
            mask = [i for i, t in enumerate(tasks) if t == task_name]
            if mask:
                ax.scatter(
                    [x[i] for i in mask],
                    [y[i] for i in mask],
                    c=color, label=task_name, s=40, alpha=0.7,
                    edgecolors="white", linewidths=0.5,
                )
    else:
        ax.scatter(x, y, s=40, alpha=0.7, c="#6C5CE7",
                   edgecolors="white", linewidths=0.5)

    lo = 0
    hi = max(max(x) if x else 0.01, max(y) if y else 0.01) * 1.15
    ax.plot([lo, hi], [lo, hi], "--", color="#636E72", lw=1.2,
            label="y = x (equal sensitivity)")

    if len(x) >= 3:
        coeffs = np.polyfit(x, y, 1)
        fit_x = np.linspace(lo, hi, 100)
        fit_y = np.polyval(coeffs, fit_x)
        ax.plot(fit_x, fit_y, "-", color="#E17055", lw=1.5, alpha=0.7,
                label=f"fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.4f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _bar_comparison(
    subtests: list[str],
    llada_vals: list[float],
    llama_vals: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(subtests) * 0.4), 6))
    x_pos = np.arange(len(subtests))
    w = 0.35
    ax.bar(x_pos - w / 2, llada_vals, w, label="LLaDA (Diff)", color="#6C5CE7", alpha=0.8)
    ax.bar(x_pos + w / 2, llama_vals, w, label="Llama (AR)", color="#00B894", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subtests, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _ratio_histogram(
    llada_ratios: list[float],
    llama_ratios: list[float],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(
        np.percentile(llada_ratios, 95) if llada_ratios else 1,
        np.percentile(llama_ratios, 95) if llama_ratios else 1,
    ) * 1.2, 50)
    ax.hist(llada_ratios, bins=bins, alpha=0.6, color="#6C5CE7",
            label=f"LLaDA α_Diff (med={np.median(llada_ratios):.3f})", density=True)
    ax.hist(llama_ratios, bins=bins, alpha=0.6, color="#00B894",
            label=f"Llama γ_AR (med={np.median(llama_ratios):.3f})", density=True)
    ax.axvline(1.0, color="#636E72", ls="--", lw=1, label="ratio=1 (isometry)")
    ax.set_xlabel("Ratio: d(f(x'), f(x)) / d(x', x)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contraction vs amplification analysis (Eq. 8) from CheckList responses."
    )
    parser.add_argument("--checklist-root", type=str, default="results/checklist")
    parser.add_argument("--out-dir", type=str, default="results/contraction_analysis")
    parser.add_argument("--samples-per-subtest", type=int, default=10,
                        help="Max example *groups* sampled per INV/DIR subtest, "
                             "or flat rows per MFT subtest.")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic-sim-threshold", type=float, default=0.0,
                        help="Retain only pairs with cosine similarity >= threshold.")
    parser.add_argument("--d-input-floor", type=float, default=0.02,
                        help="Drop pairs with d_input < floor (Lipschitz guard "
                             "against ratio blow-up when inputs are near-identical).")
    parser.add_argument("--mft-d-input-floor", type=float, default=0.05,
                        help="Stronger floor applied to MFT tests only. MFT groups "
                             "are often templated paraphrase-triples where d_input "
                             "stays small even after the base floor; a higher floor "
                             "drops those pairs from the ratio analysis.")
    args = parser.parse_args()

    checklist_root = Path(args.checklist_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    task_pairs = _find_run_pairs(checklist_root)
    if not task_pairs:
        raise FileNotFoundError(
            f"No matched llada/llama run pairs under {checklist_root}")
    print(f"Found tasks: {list(task_pairs.keys())}")

    # -----------------------------------------------------------------------
    # Phase 1: collect all texts to embed (inputs + responses, both models)
    # -----------------------------------------------------------------------
    texts_to_embed: list[str] = []
    # embed_plan tracks where each slice of embeddings belongs
    # (task, model_key, test_name, test_type, field, count)
    embed_plan: list[tuple[str, str, str, str, str, int]] = []

    # Store sampled rows per (task, model_key, test_name)
    sampled_store: dict[tuple[str, str, str], list[dict]] = {}

    # Load suite metadata from suite dump JSON files (authoritative source)
    suite_meta = _load_suite_metadata(checklist_root)

    for task_name, model_paths in task_pairs.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        task_test_meta = suite_meta.get(task_name, {})
        task_rows_by_model: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for model_key in ("llada", "llama"):
            path = model_paths[model_key]
            print(f"  Loading {model_key}: {path.name}")
            task_rows_by_model[model_key] = _group_by_subtest(_load_jsonl(path))

        test_names = sorted(
            set(task_rows_by_model["llada"].keys()) & set(task_rows_by_model["llama"].keys())
        )
        for test_name in test_names:
            meta = task_test_meta.get(test_name, {})
            tt = meta.get("test_type", "MFT")
            sampled_by_model: dict[str, list[dict[str, Any]]] = {}

            group_sizes = meta.get("group_sizes", [])
            llada_groups = _rows_to_suite_groups(
                task_rows_by_model["llada"][test_name], group_sizes
            )
            llama_groups = _rows_to_suite_groups(
                task_rows_by_model["llama"][test_name], group_sizes
            )
            n_complete = min(len(llada_groups), len(llama_groups))
            if n_complete == 0:
                continue
            picked_ids = _sample_group_ids(
                n_complete, args.samples_per_subtest, args.seed
            )
            sampled_by_model["llada"] = _select_group_rows(llada_groups, picked_ids)
            sampled_by_model["llama"] = _select_group_rows(llama_groups, picked_ids)

            for model_key in ("llada", "llama"):
                sampled = sampled_by_model.get(model_key, [])
                if not sampled:
                    continue
                sampled_store[(task_name, model_key, test_name)] = sampled

                input_texts = [_row_input_text(task_name, r) for r in sampled]
                texts_to_embed.extend(input_texts)
                embed_plan.append((
                    task_name, model_key, test_name, tt, "input", len(input_texts)))

                if task_name == "qqp":
                    q1_texts = [_qqp_split(t)[0] or t for t in input_texts]
                    q2_texts = [_qqp_split(t)[1] or t for t in input_texts]
                    texts_to_embed.extend(q1_texts)
                    embed_plan.append((
                        task_name, model_key, test_name, tt, "input_q1",
                        len(q1_texts)))
                    texts_to_embed.extend(q2_texts)
                    embed_plan.append((
                        task_name, model_key, test_name, tt, "input_q2",
                        len(q2_texts)))

                response_texts = [r.get("model_response", "") or "" for r in sampled]
                texts_to_embed.extend(response_texts)
                embed_plan.append((
                    task_name, model_key, test_name, tt, "response", len(response_texts)))

                resp_save_dir = out_dir / "response_texts" / task_name / model_key
                resp_save_dir.mkdir(parents=True, exist_ok=True)
                safe_name = test_name.replace("/", "_").replace(" ", "_")[:80]
                resp_save_path = resp_save_dir / f"{safe_name}.json"
                with resp_save_path.open("w", encoding="utf-8") as rf:
                    json.dump({
                        "test_name": test_name,
                        "test_type": tt,
                        "model": model_key,
                        "task": task_name,
                        "n_sampled": len(sampled),
                        "examples": [
                            {"input": inp, "response": resp}
                            for inp, resp in zip(input_texts, response_texts)
                        ],
                    }, rf, indent=2, ensure_ascii=False)

        for model_key in ("llada", "llama"):
            n_sampled = sum(
                len(v) for (t, m, _), v in sampled_store.items()
                if t == task_name and m == model_key
            )
            print(f"    {len(test_names)} subtests, {n_sampled} sampled rows for {model_key}")

    # -----------------------------------------------------------------------
    # Phase 2: embed everything in one batch
    # -----------------------------------------------------------------------
    print(f"\nEmbedding {len(texts_to_embed)} texts (inputs + responses, both models)...")
    all_embeddings = _embed(
        texts_to_embed, args.embedding_model, device, args.batch_size)

    # Distribute embeddings back
    # emb_store[(task, model, test_name, field)] = np.ndarray
    emb_store: dict[tuple[str, str, str, str], np.ndarray] = {}
    cursor = 0
    for task_name, model_key, test_name, tt, field, count in embed_plan:
        emb_store[(task_name, model_key, test_name, field)] = \
            all_embeddings[cursor:cursor + count]
        cursor += count

    # Save all embeddings as NPZ per task/model
    emb_save_dir = out_dir / "embeddings"
    emb_save_dir.mkdir(parents=True, exist_ok=True)
    for task_name, model_paths in task_pairs.items():
        for model_key in ("llada", "llama"):
            task_input_embs = []
            task_q1_embs = []
            task_q2_embs = []
            task_resp_embs = []
            task_meta = []
            for (t, m, tn, field), emb in emb_store.items():
                if t != task_name or m != model_key:
                    continue
                if field == "input":
                    task_input_embs.append(emb)
                    rows_key = (t, m, tn)
                    for i, row in enumerate(sampled_store.get(rows_key, [])):
                        tt_actual = suite_meta.get(t, {}).get(tn, {}).get("test_type", "MFT")
                        task_meta.append({
                            "test_name": tn,
                            "test_type": tt_actual,
                            "example_idx": row.get("example_idx", i),
                        })
                elif field == "input_q1":
                    task_q1_embs.append(emb)
                elif field == "input_q2":
                    task_q2_embs.append(emb)
                elif field == "response":
                    task_resp_embs.append(emb)
            if task_input_embs and task_resp_embs:
                inp_arr = np.concatenate(task_input_embs, axis=0)
                resp_arr = np.concatenate(task_resp_embs, axis=0)
                npz_payload: dict[str, Any] = {
                    "input_embeddings": inp_arr,
                    "response_embeddings": resp_arr,
                    "metadata": np.array(json.dumps(task_meta), dtype=object),
                }
                if task_q1_embs and task_q2_embs:
                    npz_payload["input_q1_embeddings"] = np.concatenate(
                        task_q1_embs, axis=0)
                    npz_payload["input_q2_embeddings"] = np.concatenate(
                        task_q2_embs, axis=0)
                np.savez_compressed(
                    emb_save_dir / f"{task_name}_{model_key}_embeddings.npz",
                    **npz_payload,
                )
                print(f"  Saved embeddings: {task_name}_{model_key}_embeddings.npz "
                      f"(input: {inp_arr.shape}, response: {resp_arr.shape})")

    # -----------------------------------------------------------------------
    # Phase 3: compute metrics
    # -----------------------------------------------------------------------
    task_colors = {"sentiment": "#6C5CE7", "qqp": "#00B894", "squad": "#E17055"}

    # Collect per-subtest median ratios for scatter
    ratio_scatter_llada: dict[str, float] = {}
    ratio_scatter_llama: dict[str, float] = {}
    ratio_scatter_tasks: dict[str, str] = {}

    # Collect all individual pair ratios for histogram
    all_pair_ratios_llada: list[float] = []
    all_pair_ratios_llama: list[float] = []

    # MFT spread (complementary)
    mft_spread_llada: dict[str, float] = {}
    mft_spread_llama: dict[str, float] = {}
    mft_spread_tasks: dict[str, str] = {}

    summary: dict[str, Any] = {}
    detailed_summary: dict[str, Any] = {
        "config": {
            "embedding_model": args.embedding_model,
            "device": device,
            "batch_size": args.batch_size,
            "samples_per_subtest": args.samples_per_subtest,
            "semantic_similarity_threshold": args.semantic_sim_threshold,
            "d_input_floor": args.d_input_floor,
            "mft_d_input_floor": args.mft_d_input_floor,
        },
        "tasks": {},
    }

    for task_name, model_paths in task_pairs.items():
        print(f"\nAnalyzing: {task_name}")

        # Collect all test_names for this task
        test_names = set()
        for (t, m, tn) in sampled_store:
            if t == task_name:
                test_names.add(tn)

        task_ratio_data: dict[str, dict[str, Any]] = {}
        task_mft_data: dict[str, dict[str, float]] = {}

        for test_name in sorted(test_names):
            llada_rows = sampled_store.get((task_name, "llada", test_name), [])
            llama_rows = sampled_store.get((task_name, "llama", test_name), [])
            if not llada_rows or not llama_rows:
                continue

            tt = suite_meta.get(task_name, {}).get(test_name, {}).get("test_type", "MFT")

            capability = suite_meta.get(task_name, {}).get(
                test_name, {}).get("capability", "Unknown")

            if tt in ("INV", "DIR", "MFT"):
                # --- Ratio computation (Eq. 8) ---
                llada_input_embs = emb_store.get(
                    (task_name, "llada", test_name, "input"))
                llada_resp_embs = emb_store.get(
                    (task_name, "llada", test_name, "response"))
                llama_input_embs = emb_store.get(
                    (task_name, "llama", test_name, "input"))
                llama_resp_embs = emb_store.get(
                    (task_name, "llama", test_name, "response"))

                if any(e is None for e in [
                    llada_input_embs, llada_resp_embs,
                    llama_input_embs, llama_resp_embs,
                ]):
                    continue

                llada_q1 = emb_store.get(
                    (task_name, "llada", test_name, "input_q1"))
                llada_q2 = emb_store.get(
                    (task_name, "llada", test_name, "input_q2"))
                llama_q1 = emb_store.get(
                    (task_name, "llama", test_name, "input_q1"))
                llama_q2 = emb_store.get(
                    (task_name, "llama", test_name, "input_q2"))

                llada_pairs = _compute_pair_metrics(
                    task_name,
                    llada_rows,
                    llada_input_embs,
                    llada_resp_embs,
                    args.semantic_sim_threshold,
                    test_type=tt,
                    d_input_floor=args.d_input_floor,
                    mft_d_input_floor=args.mft_d_input_floor,
                    input_embs_side_a=llada_q1,
                    input_embs_side_b=llada_q2,
                )
                llama_pairs = _compute_pair_metrics(
                    task_name,
                    llama_rows,
                    llama_input_embs,
                    llama_resp_embs,
                    args.semantic_sim_threshold,
                    test_type=tt,
                    d_input_floor=args.d_input_floor,
                    mft_d_input_floor=args.mft_d_input_floor,
                    input_embs_side_a=llama_q1,
                    input_embs_side_b=llama_q2,
                )

                llada_valid = [p["ratio"] for p in llada_pairs if p["ratio"] is not None]
                llama_valid = [p["ratio"] for p in llama_pairs if p["ratio"] is not None]

                if llada_valid and llama_valid:
                    key = f"{task_name}::{test_name}"
                    med_llada = float(np.median(llada_valid))
                    med_llama = float(np.median(llama_valid))
                    ratio_scatter_llada[key] = med_llada
                    ratio_scatter_llama[key] = med_llama
                    ratio_scatter_tasks[key] = task_name

                    all_pair_ratios_llada.extend(llada_valid)
                    all_pair_ratios_llama.extend(llama_valid)

                    mean_alpha = float(np.mean(llada_valid))
                    mean_gamma = float(np.mean(llama_valid))
                    task_ratio_data[test_name] = {
                        "test_type": tt,
                        "capability": capability,
                        "llada_median_ratio": med_llada,
                        "llama_median_ratio": med_llama,
                        "llada_mean_ratio": mean_alpha,
                        "llama_mean_ratio": mean_gamma,
                        "gap_gamma_minus_alpha": mean_gamma - mean_alpha,
                        "winner": "LLaDA" if mean_alpha < mean_gamma else "Llama",
                        "llada_mean_conditional_degradation": float(
                            np.mean([p["d_output"] for p in llada_pairs])
                        ),
                        "llama_mean_conditional_degradation": float(
                            np.mean([p["d_output"] for p in llama_pairs])
                        ),
                        "mean_input_distance": float(np.mean(
                            [p["d_input"] for p in llada_pairs] +
                            [p["d_input"] for p in llama_pairs]
                        )),
                        "llada_n_pairs": len(llada_valid),
                        "llama_n_pairs": len(llama_valid),
                        "llada_pairs": [
                            {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in p.items()}
                            for p in llada_pairs if p["ratio"] is not None
                        ],
                        "llama_pairs": [
                            {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in p.items()}
                            for p in llama_pairs if p["ratio"] is not None
                        ],
                    }

                # Complementary response spread
                if len(llada_resp_embs) >= 2 and len(llama_resp_embs) >= 2:
                    key = f"{task_name}::{test_name}"
                    sp_llada = _mft_spread(llada_resp_embs)
                    sp_llama = _mft_spread(llama_resp_embs)
                    mft_spread_llada[key] = sp_llada
                    mft_spread_llama[key] = sp_llama
                    mft_spread_tasks[key] = task_name
                    task_mft_data[test_name] = {
                        "capability": capability,
                        "llada_spread": sp_llada,
                        "llama_spread": sp_llama,
                    }

        summary[task_name] = {
            "inv_dir_ratios": task_ratio_data,
            "mft_spread": task_mft_data,
        }

        # Structured summaries for dataset / capability / subtest
        capability_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for test_name, rec in task_ratio_data.items():
            capability_buckets[rec["capability"]].append(rec)

        capability_summary: dict[str, Any] = {}
        for capability_name, recs in capability_buckets.items():
            llada_pairs = sum(r["llada_n_pairs"] for r in recs)
            llama_pairs = sum(r["llama_n_pairs"] for r in recs)
            llada_ratio_num = sum(r["llada_mean_ratio"] * r["llada_n_pairs"] for r in recs)
            llama_ratio_num = sum(r["llama_mean_ratio"] * r["llama_n_pairs"] for r in recs)
            llada_deg_num = sum(
                r["llada_mean_conditional_degradation"] * r["llada_n_pairs"] for r in recs
            )
            llama_deg_num = sum(
                r["llama_mean_conditional_degradation"] * r["llama_n_pairs"] for r in recs
            )
            alpha = llada_ratio_num / max(llada_pairs, 1)
            gamma = llama_ratio_num / max(llama_pairs, 1)
            gap = gamma - alpha
            capability_summary[capability_name] = {
                "n_subtests": len(recs),
                "llada_n_pairs": llada_pairs,
                "llama_n_pairs": llama_pairs,
                "llada_mean_alpha_diff": alpha,
                "llama_mean_gamma_ar": gamma,
                "gap_gamma_minus_alpha": gap,
                "winner": "LLaDA" if alpha < gamma else "Llama",
                "llada_mean_conditional_degradation": llada_deg_num / max(llada_pairs, 1),
                "llama_mean_conditional_degradation": llama_deg_num / max(llama_pairs, 1),
                "relative_position": "above diagonal" if gamma > alpha else "below diagonal",
            }

        task_llada_pairs = sum(r["llada_n_pairs"] for r in task_ratio_data.values())
        task_llama_pairs = sum(r["llama_n_pairs"] for r in task_ratio_data.values())
        task_llada_ratio_num = sum(
            r["llada_mean_ratio"] * r["llada_n_pairs"] for r in task_ratio_data.values()
        )
        task_llama_ratio_num = sum(
            r["llama_mean_ratio"] * r["llama_n_pairs"] for r in task_ratio_data.values()
        )
        task_llada_deg_num = sum(
            r["llada_mean_conditional_degradation"] * r["llada_n_pairs"]
            for r in task_ratio_data.values()
        )
        task_llama_deg_num = sum(
            r["llama_mean_conditional_degradation"] * r["llama_n_pairs"]
            for r in task_ratio_data.values()
        )
        task_alpha = task_llada_ratio_num / max(task_llada_pairs, 1)
        task_gamma = task_llama_ratio_num / max(task_llama_pairs, 1)
        task_gap = task_gamma - task_alpha
        # Theory: sentiment should be below diagonal, qqp/squad above
        theory_expected = {
            "sentiment": "below diagonal",
            "qqp": "above diagonal",
            "squad": "above diagonal",
        }
        actual_position = "above diagonal" if task_gamma > task_alpha else "below diagonal"
        theory_match = (
            actual_position == theory_expected.get(task_name, "")
            if task_name in theory_expected
            else None
        )
        detailed_summary["tasks"][task_name] = {
            "dataset_summary": {
                "n_subtests": len(task_ratio_data),
                "llada_n_pairs": task_llada_pairs,
                "llama_n_pairs": task_llama_pairs,
                "llada_mean_alpha_diff": task_alpha,
                "llama_mean_gamma_ar": task_gamma,
                "gap_gamma_minus_alpha": task_gap,
                "winner": "LLaDA" if task_alpha < task_gamma else "Llama",
                "llada_mean_conditional_degradation": task_llada_deg_num / max(task_llada_pairs, 1),
                "llama_mean_conditional_degradation": task_llama_deg_num / max(task_llama_pairs, 1),
                "relative_position": actual_position,
                "theory_match": theory_match,
            },
            "capability_summary": capability_summary,
            "subtest_summary": task_ratio_data,
        }

        # --- Per-task bar charts ---
        inv_tests = sorted(task_ratio_data.keys())
        if inv_tests:
            _bar_comparison(
                [t[:40] for t in inv_tests],
                [task_ratio_data[t]["llada_median_ratio"] for t in inv_tests],
                [task_ratio_data[t]["llama_median_ratio"] for t in inv_tests],
                ylabel="Median Ratio: d(f(x'),f(x)) / d(x',x)",
                title=f"{task_name.upper()} INV/DIR — Amplification/Contraction Factor",
                out_path=out_dir / f"{task_name}_inv_dir_ratio_bars.png",
            )

        mft_tests = sorted(task_mft_data.keys())
        if mft_tests:
            _bar_comparison(
                [t[:40] for t in mft_tests],
                [task_mft_data[t]["llada_spread"] for t in mft_tests],
                [task_mft_data[t]["llama_spread"] for t in mft_tests],
                ylabel="Mean Pairwise Cosine Distance",
                title=f"{task_name.upper()} MFT — Response Spread per Subtest",
                out_path=out_dir / f"{task_name}_mft_spread_bars.png",
            )

    # -----------------------------------------------------------------------
    # Phase 4: cross-task plots
    # -----------------------------------------------------------------------

    # --- Main scatter: α_Diff vs γ_AR (prof's plot) ---
    if ratio_scatter_llada:
        keys = sorted(ratio_scatter_llada.keys())
        _scatter_with_diagonal(
            x=[ratio_scatter_llada[k] for k in keys],
            y=[ratio_scatter_llama[k] for k in keys],
            labels=keys,
            xlabel="α_Diff (LLaDA contraction factor)",
            ylabel="γ_AR (Llama amplification factor)",
            title="Eq. 8: Contraction (α_Diff) vs Amplification (γ_AR) per Subtest",
            out_path=out_dir / "scatter_ratio_contraction_vs_amplification.png",
            task_colors=task_colors,
            tasks=[ratio_scatter_tasks[k] for k in keys],
        )

    # --- Ratio histogram ---
    if all_pair_ratios_llada and all_pair_ratios_llama:
        _ratio_histogram(
            all_pair_ratios_llada,
            all_pair_ratios_llama,
            title="Distribution of Per-Pair Ratios (All Tasks)",
            out_path=out_dir / "histogram_pair_ratios.png",
        )

    # --- MFT spread scatter ---
    if mft_spread_llada:
        keys = sorted(mft_spread_llada.keys())
        _scatter_with_diagonal(
            x=[mft_spread_llada[k] for k in keys],
            y=[mft_spread_llama[k] for k in keys],
            labels=keys,
            xlabel="LLaDA Response Spread",
            ylabel="Llama Response Spread",
            title="MFT: Response Diversity (LLaDA vs Llama)",
            out_path=out_dir / "scatter_mft_spread.png",
            task_colors=task_colors,
            tasks=[mft_spread_tasks[k] for k in keys],
        )

    # -----------------------------------------------------------------------
    # Phase 5: summary
    # -----------------------------------------------------------------------
    summary_path = out_dir / "contraction_amplification_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")

    detailed_summary_path = out_dir / "contraction_amplification_detailed.json"
    with detailed_summary_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed summary: {detailed_summary_path}")

    # --- Print key stats ---
    if ratio_scatter_llada:
        keys = sorted(ratio_scatter_llada.keys())
        x_vals = [ratio_scatter_llada[k] for k in keys]
        y_vals = [ratio_scatter_llama[k] for k in keys]

        above = sum(1 for x, y in zip(x_vals, y_vals) if y > x)
        below = sum(1 for x, y in zip(x_vals, y_vals) if y < x)
        on = len(x_vals) - above - below

        print(f"\n--- Ratio-based analysis (Eq. 8) ---")
        print(f"Subtests analyzed: {len(keys)}")
        print(f"Above diagonal (γ_AR > α_Diff, AR amplifies more): {above}")
        print(f"Below diagonal (α_Diff > γ_AR, Diff less stable): {below}")
        print(f"On diagonal: {on}")

        if len(x_vals) >= 3:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            print(f"Linear fit: γ_AR = {coeffs[0]:.3f} * α_Diff + {coeffs[1]:.5f}")
            print(f"  slope > 1 → AR amplifies proportionally more")
            print(f"  slope < 1 → Diff contracts proportionally less")

        if all_pair_ratios_llada and all_pair_ratios_llama:
            print(f"\n--- Per-pair ratio stats ---")
            print(f"LLaDA α_Diff:  median={np.median(all_pair_ratios_llada):.4f}, "
                  f"mean={np.mean(all_pair_ratios_llada):.4f}, "
                  f"n={len(all_pair_ratios_llada)}")
            print(f"Llama  γ_AR:   median={np.median(all_pair_ratios_llama):.4f}, "
                  f"mean={np.mean(all_pair_ratios_llama):.4f}, "
                  f"n={len(all_pair_ratios_llama)}")
            R = np.median(all_pair_ratios_llama) / max(
                np.median(all_pair_ratios_llada), EPSILON)
            print(f"R = median(γ_AR) / median(α_Diff) = {R:.4f}")
            if R > 1:
                print("  → AR amplifies more than Diff (diffusion wins overall)")
            else:
                print("  → Diff contracts less (AR wins overall)")

    print(f"\nAll outputs in: {out_dir}")

    # -----------------------------------------------------------------------
    # Phase 6: CSV export (Tables A/B/C for slides/report)
    # -----------------------------------------------------------------------
    _export_csvs(detailed_summary, out_dir)

    # -----------------------------------------------------------------------
    # Phase 7: capability gap bar chart
    # -----------------------------------------------------------------------
    _plot_capability_gap(detailed_summary, out_dir, task_colors)


def _export_csvs(detailed: dict[str, Any], out_dir: Path) -> None:
    """Write dataset_summary.csv, capability_summary.csv, subtest_summary.csv."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Table A: Dataset Summary ---
    ds_rows: list[dict[str, Any]] = []
    for task_name, task_data in detailed.get("tasks", {}).items():
        ds = task_data.get("dataset_summary", {})
        ds_rows.append({
            "dataset": task_name,
            "winner": ds.get("winner", ""),
            "Delta_perp_Diff": round(ds.get("llada_mean_conditional_degradation", 0), 6),
            "Delta_perp_AR": round(ds.get("llama_mean_conditional_degradation", 0), 6),
            "alpha_Diff": round(ds.get("llada_mean_alpha_diff", 0), 6),
            "gamma_AR": round(ds.get("llama_mean_gamma_ar", 0), 6),
            "gap_gamma_minus_alpha": round(ds.get("gap_gamma_minus_alpha", 0), 6),
            "relative_position": ds.get("relative_position", ""),
            "theory_match": ds.get("theory_match", ""),
            "n_subtests": ds.get("n_subtests", 0),
            "llada_n_pairs": ds.get("llada_n_pairs", 0),
            "llama_n_pairs": ds.get("llama_n_pairs", 0),
        })
    if ds_rows:
        _write_csv(tables_dir / "dataset_summary.csv", ds_rows)

    # --- Table B: Capability Summary ---
    cap_rows: list[dict[str, Any]] = []
    for task_name, task_data in detailed.get("tasks", {}).items():
        for cap_name, cs in task_data.get("capability_summary", {}).items():
            cap_rows.append({
                "dataset": task_name,
                "capability": cap_name,
                "alpha_Diff": round(cs.get("llada_mean_alpha_diff", 0), 6),
                "gamma_AR": round(cs.get("llama_mean_gamma_ar", 0), 6),
                "gap_gamma_minus_alpha": round(cs.get("gap_gamma_minus_alpha", 0), 6),
                "winner": cs.get("winner", ""),
                "Delta_perp_Diff": round(cs.get("llada_mean_conditional_degradation", 0), 6),
                "Delta_perp_AR": round(cs.get("llama_mean_conditional_degradation", 0), 6),
                "relative_position": cs.get("relative_position", ""),
                "n_subtests": cs.get("n_subtests", 0),
                "llada_n_pairs": cs.get("llada_n_pairs", 0),
                "llama_n_pairs": cs.get("llama_n_pairs", 0),
            })
    if cap_rows:
        _write_csv(tables_dir / "capability_summary.csv", cap_rows)

    # --- Table C: Subtest Summary ---
    sub_rows: list[dict[str, Any]] = []
    for task_name, task_data in detailed.get("tasks", {}).items():
        for sub_name, ss in task_data.get("subtest_summary", {}).items():
            sub_rows.append({
                "dataset": task_name,
                "subtest": sub_name,
                "capability": ss.get("capability", ""),
                "test_type": ss.get("test_type", ""),
                "alpha_Diff": round(ss.get("llada_mean_ratio", 0), 6),
                "gamma_AR": round(ss.get("llama_mean_ratio", 0), 6),
                "gap_gamma_minus_alpha": round(ss.get("gap_gamma_minus_alpha", 0), 6),
                "winner": ss.get("winner", ""),
                "Delta_perp_Diff": round(ss.get("llada_mean_conditional_degradation", 0), 6),
                "Delta_perp_AR": round(ss.get("llama_mean_conditional_degradation", 0), 6),
                "mean_input_distance": round(ss.get("mean_input_distance", 0), 6),
                "llada_n_pairs": ss.get("llada_n_pairs", 0),
                "llama_n_pairs": ss.get("llama_n_pairs", 0),
            })
    if sub_rows:
        _write_csv(tables_dir / "subtest_summary.csv", sub_rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV: {path}")


def _plot_capability_gap(
    detailed: dict[str, Any], out_dir: Path,
    task_colors: dict[str, str],
) -> None:
    """Bar chart of gamma_AR - alpha_Diff per capability, colored by dataset."""
    caps: list[str] = []
    gaps: list[float] = []
    colors: list[str] = []
    for task_name, task_data in detailed.get("tasks", {}).items():
        for cap_name, cs in task_data.get("capability_summary", {}).items():
            label = f"{task_name}:{cap_name}"
            caps.append(label)
            gaps.append(cs.get("gap_gamma_minus_alpha", 0))
            colors.append(task_colors.get(task_name, "#999999"))

    if not caps:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(caps) * 0.5), 6))
    x_pos = np.arange(len(caps))
    ax.bar(x_pos, gaps, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#636E72", ls="--", lw=1, label="equal (γ_AR = α_Diff)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(caps, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("γ_AR − α_Diff")
    ax.set_title("Capability-Level Gap: γ_AR − α_Diff (>0 = AR amplifies more)")

    # Legend for task colors
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=t) for t, c in task_colors.items()
               if t in detailed.get("tasks", {})]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "bar_capability_gap.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


if __name__ == "__main__":
    main()
