"""Build a Table-2-style alpha/gamma table for v5 response embeddings.

This companion table forces output distance to be cosine distance between
cached sentence embeddings of the model responses:

    d_output = 1 - cos(emb(y), emb(y'))

This is intentionally different from the main v5 table, which uses task-aware
label-space distances for sentiment/QQP when labels can be parsed, and response
embedding cosine as a fallback / for SQuAD.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(".")
EMB_DIR = (
    ROOT
    / "results/lightning"
    / "contraction_analysis_v5"
    / "embeddings"
)
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ORDER = ["sentiment", "qqp", "squad"]
MODELS = ("llada", "llama")
D_INPUT_FLOOR = 0.02
MFT_D_INPUT_FLOOR = 0.05
EPSILON = 1e-12


def fmt_float(value: float) -> str:
    return f"{value:.3f}"


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPSILON
    return float(1.0 - (np.dot(a, b) / denom))


def load_embeddings(task: str, model_key: str) -> dict:
    path = EMB_DIR / f"{task}_{model_key}_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached embeddings: {path}")
    data = np.load(path, allow_pickle=True)
    out = {key: data[key] for key in data.files}
    out["metadata"] = json.loads(data["metadata"].item())
    return out


def pair_stats(task: str, model_key: str) -> dict[str, float | int]:
    data = load_embeddings(task, model_key)
    meta = data["metadata"]
    input_embs = data["input_embeddings"]
    response_embs = data["response_embeddings"]
    q1_embs = data.get("input_q1_embeddings")
    q2_embs = data.get("input_q2_embeddings")

    by_test_example: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        key = (row.get("test_name", ""), row.get("example_idx", idx))
        by_test_example[key].append(idx)

    ratios: list[float] = []
    d_outputs: list[float] = []

    for idxs in by_test_example.values():
        if len(idxs) < 2:
            continue
        idxs = sorted(idxs)
        test_type = meta[idxs[0]].get("test_type", "MFT")
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

            d_output = cos_dist(response_embs[a_idx], response_embs[b_idx])
            ratios.append(d_output / d_input)
            d_outputs.append(d_output)

    return {
        "ratio": float(np.mean(ratios)),
        "delta_perp": float(np.mean(d_outputs)),
        "n_pairs": len(ratios),
    }


def build_rows() -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for task in TASK_ORDER:
        stats = {model_key: pair_stats(task, model_key) for model_key in MODELS}
        alpha = float(stats["llada"]["ratio"])
        gamma = float(stats["llama"]["ratio"])
        out.append({
            "task": task,
            "alpha_Diff_output_emb": alpha,
            "gamma_AR_output_emb": gamma,
            "gap_gamma_minus_alpha": gamma - alpha,
            "gamma_over_alpha": gamma / alpha if alpha else float("nan"),
            "Delta_perp_Diff_output_emb": float(stats["llada"]["delta_perp"]),
            "Delta_perp_AR_output_emb": float(stats["llama"]["delta_perp"]),
            "winner": "LLaDA" if alpha < gamma else "Llama",
            "llada_n_pairs": int(stats["llada"]["n_pairs"]),
            "llama_n_pairs": int(stats["llama"]["n_pairs"]),
        })
    return out


def write_csv(rows: list[dict[str, float | str]]) -> Path:
    path = OUT_DIR / "v5_output_embedding_alpha_gamma_table.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_tex(rows: list[dict[str, float | str]]) -> Path:
    path = OUT_DIR / "v5_output_embedding_alpha_gamma_table.tex"
    with path.open("w") as f:
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
                f"{fmt_float(float(row['alpha_Diff_output_emb']))} & "
                f"{fmt_float(float(row['gamma_AR_output_emb']))} & "
                f"{float(row['gap_gamma_minus_alpha']):+.3f} & "
                f"{fmt_float(float(row['gamma_over_alpha']))} & "
                f"{fmt_float(float(row['Delta_perp_Diff_output_emb']))} & "
                f"{fmt_float(float(row['Delta_perp_AR_output_emb']))} \\\\ \n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Pair-weighted $\\alpha_{\\mathrm{Diff}}$ and "
            "$\\gamma_{\\mathrm{AR}}$ using response/output embedding cosine "
            "distance as $d_{\\mathrm{output}}$. Here "
            "$\\Delta_\\perp$ is the mean output-embedding shift on paired "
            "perturbations.}\n"
        )
        f.write("\\end{table}\n")
    return path


def main() -> None:
    rows = build_rows()
    csv_path = write_csv(rows)
    tex_path = write_tex(rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    for row in rows:
        print(
            row["task"],
            fmt_float(float(row["alpha_Diff_output_emb"])),
            fmt_float(float(row["gamma_AR_output_emb"])),
            f"{float(row['gap_gamma_minus_alpha']):+.3f}",
            fmt_float(float(row["gamma_over_alpha"])),
        )


if __name__ == "__main__":
    main()
