"""Build a Table-2-style alpha/gamma table for cosine-on-label v5.

The headline Table 2 uses categorical output distance for sentiment and QQP.
This companion table keeps the same pair-weighted alpha/gamma estimator but
uses cosine distance between normalized label embeddings as d_output.
"""

from __future__ import annotations

import csv
import json
import string
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(".")
CHECKLIST = ROOT / "results" / "checklist"
EMB_DIR = ROOT / "results/lightning" / "contraction_analysis_v5" / "embeddings"
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["sentiment", "qqp"]
MODELS = {
    "llada": "llada_rerun_fixed",
    "llama": "llama_rerun_fixed",
}
D_INPUT_FLOOR = 0.02
MFT_D_INPUT_FLOOR = 0.05
EPSILON = 1e-12


def normalize_label(label: str | None) -> str:
    value = (label or "").strip().lower()
    return value.translate(str.maketrans("", "", string.punctuation)).strip()


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def load_rows(task: str, model_key: str) -> list[dict]:
    run_dir = CHECKLIST / task / MODELS[model_key]
    path = next(run_dir.glob("examples_full_*.jsonl"))
    with path.open() as f:
        return [json.loads(line) for line in f]


def load_embeddings(task: str, model_key: str) -> dict[str, np.ndarray]:
    path = EMB_DIR / f"{task}_{model_key}_embeddings.npz"
    return dict(np.load(path, allow_pickle=True))


def label_embeddings(rows_by_task: dict[str, dict[str, list[dict]]]) -> dict[str, np.ndarray]:
    """Recover label embeddings from the cached response embeddings.

    The v5 run already embedded model responses with all-MiniLM-L6-v2. For the
    label-cosine companion table we need embeddings of the normalized labels.
    Exact lower-case label responses exist in the cached runs, so we reuse those
    vectors instead of loading sentence-transformers again.
    """
    wanted = sorted({
        normalize_label(row.get("predicted_label"))
        for task_rows in rows_by_task.values()
        for rows in task_rows.values()
        for row in rows
        if normalize_label(row.get("predicted_label"))
    })
    found: dict[str, np.ndarray] = {}
    for task, task_rows in rows_by_task.items():
        for model_key, rows in task_rows.items():
            embs = load_embeddings(task, model_key)["response_embeddings"]
            for idx, row in enumerate(rows):
                response = (row.get("model_response") or "").strip()
                label = normalize_label(row.get("predicted_label"))
                if label in wanted and label not in found and response == label:
                    found[label] = embs[idx]

    missing = [label for label in wanted if label not in found]
    if missing:
        raise RuntimeError(f"Could not recover exact cached embeddings for labels: {missing}")
    return found


def pair_records_for_model(
    task: str,
    rows: list[dict],
    embs: dict[str, np.ndarray],
    label_embs: dict[str, np.ndarray],
) -> tuple[list[float], list[float]]:
    by_test_example: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_test_example[(row.get("test_name", ""), row.get("example_idx", idx))].append(idx)

    ratios: list[float] = []
    d_outputs: list[float] = []

    input_embs = embs["input_embeddings"]
    q1_embs = embs.get("input_q1_embeddings")
    q2_embs = embs.get("input_q2_embeddings")

    for (_, _), idxs in by_test_example.items():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs)
        test_type = rows[idxs[0]].get("test_type", "MFT")
        if test_type == "MFT":
            pair_iter = [
                (idxs[i], idxs[j])
                for i in range(len(idxs))
                for j in range(i + 1, len(idxs))
            ]
            floor = MFT_D_INPUT_FLOOR
        else:
            pair_iter = [(idxs[0], idxs[k]) for k in range(1, len(idxs))]
            floor = D_INPUT_FLOOR

        for a_idx, b_idx in pair_iter:
            if q1_embs is not None and q2_embs is not None:
                d_input = max(
                    cos_dist(q1_embs[a_idx], q1_embs[b_idx]),
                    cos_dist(q2_embs[a_idx], q2_embs[b_idx]),
                )
            else:
                d_input = cos_dist(input_embs[a_idx], input_embs[b_idx])
            if d_input < max(floor, EPSILON):
                continue

            label_a = normalize_label(rows[a_idx].get("predicted_label"))
            label_b = normalize_label(rows[b_idx].get("predicted_label"))
            if not label_a or not label_b:
                continue
            if label_a == label_b:
                d_output = 0.0
            else:
                d_output = cos_dist(label_embs[label_a], label_embs[label_b])

            ratios.append(d_output / d_input)
            d_outputs.append(d_output)

    return ratios, d_outputs


def fmt_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    rows_by_task = {
        task: {model_key: load_rows(task, model_key) for model_key in MODELS}
        for task in TASKS
    }
    label_embs = label_embeddings(rows_by_task)

    rows = []
    for task in TASKS:
        model_stats = {}
        for model_key in MODELS:
            embs = load_embeddings(task, model_key)
            ratios, d_outputs = pair_records_for_model(
                task, rows_by_task[task][model_key], embs, label_embs
            )
            model_stats[model_key] = {
                "ratio": float(np.mean(ratios)),
                "delta_perp": float(np.mean(d_outputs)),
                "n_pairs": len(ratios),
            }

        alpha = model_stats["llada"]["ratio"]
        gamma = model_stats["llama"]["ratio"]
        rows.append({
            "task": task,
            "alpha_Diff_cos_label": alpha,
            "gamma_AR_cos_label": gamma,
            "gap_gamma_minus_alpha": gamma - alpha,
            "gamma_over_alpha": gamma / alpha if alpha else float("nan"),
            "Delta_perp_Diff_cos_label": model_stats["llada"]["delta_perp"],
            "Delta_perp_AR_cos_label": model_stats["llama"]["delta_perp"],
            "llada_n_pairs": model_stats["llada"]["n_pairs"],
            "llama_n_pairs": model_stats["llama"]["n_pairs"],
        })

    csv_path = OUT_DIR / "v5_cos_label_alpha_gamma_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = OUT_DIR / "v5_cos_label_alpha_gamma_table.tex"
    with tex_path.open("w") as f:
        f.write("\\begin{table}[H]\\centering\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "task & $\\alpha_{\\mathrm{Diff}}$ & $\\gamma_{\\mathrm{AR}}$ "
            "& $\\gamma-\\alpha$ & $\\gamma/\\alpha$ "
            "& $\\Delta_\\perp^{\\mathrm{Diff}}$ "
            "& $\\Delta_\\perp^{\\mathrm{AR}}$ \\\\ \\midrule\n"
        )
        for row in rows:
            f.write(
                f"{row['task']} & "
                f"{fmt_float(row['alpha_Diff_cos_label'])} & "
                f"{fmt_float(row['gamma_AR_cos_label'])} & "
                f"{row['gap_gamma_minus_alpha']:+.3f} & "
                f"{fmt_float(row['gamma_over_alpha'])} & "
                f"{fmt_float(row['Delta_perp_Diff_cos_label'])} & "
                f"{fmt_float(row['Delta_perp_AR_cos_label'])} \\\\ \n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Pair-weighted $\\alpha_{\\mathrm{Diff}}$ and "
            "$\\gamma_{\\mathrm{AR}}$ using cosine distance between "
            "normalised label embeddings as the output distance. This "
            "companion table applies only to sentiment and QQP; SQuAD "
            "already uses response-embedding cosine in the main table.}\n"
        )
        f.write("\\end{table}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    for row in rows:
        print(
            row["task"],
            fmt_float(row["alpha_Diff_cos_label"]),
            fmt_float(row["gamma_AR_cos_label"]),
            f"{row['gap_gamma_minus_alpha']:+.3f}",
            fmt_float(row["gamma_over_alpha"]),
        )


if __name__ == "__main__":
    main()
