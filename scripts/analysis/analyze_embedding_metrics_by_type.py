"""Per-test-type embedding metrics, broken down by capability and subtest.

Uses cached contraction_analysis_v5 embeddings (input/response) + tests_full
JSONs (capability + fail_rate) to produce:

  results/analysis/embedding_by_type/
    subtest/{task}.csv        - per (test_name, type, capability) x model
    capability/{task}.csv     - rolled up by capability x type x model
    summary.csv               - task x type x model top-line
    by_type_bars.png          - grouped bars: mean delta by capability/type

Metrics per row-group (rows sharing (test_name, example_idx)):
  eps = 1 - cos(input[0], input[k])    for k>=1
  delta = 1 - cos(response[0], response[k])

Per test_type aggregation:
  MFT: primary fail_rate (from tests_full); secondary mean_delta (spread).
  INV: primary mean/median delta (lower = better); rho = median(delta/eps).
  DIR: primary mean_delta (higher ~ output moved); secondary fail_rate
       (direction correctness per suite's expectation fn).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("qqp", "sentiment", "squad")
MODELS = {
    "llada": "GSAI-ML_LLaDA-8B-Instruct",
    "llama": "meta-llama_Llama-3.1-8B-Instruct",
}
EPS_FLOOR = 0.02  # rho denominator guard


def cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine similarity. `a` shape (d,) or (n,d); `b` shape (n,d)."""
    if a.ndim == 1:
        a = np.broadcast_to(a, b.shape)
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return 1.0 - (np.sum(a * b, axis=1) / (na * nb))


def load_tests_full(task: str, model_key: str) -> dict[str, dict]:
    run = f"results/checklist/{task}/{model_key}_rerun_fixed"
    p = Path(run) / f"tests_full_{MODELS[model_key]}.json"
    j = json.loads(p.read_text())
    return {t["name"]: t for t in j["tests"]}


def load_embeddings(task: str, model_key: str):
    p = Path("results/lightning/contraction_analysis_v5/embeddings") / \
        f"{task}_{model_key}_embeddings.npz"
    d = np.load(p, allow_pickle=True)
    meta = json.loads(d["metadata"].item())
    return d["input_embeddings"], d["response_embeddings"], meta


def group_rows(meta: list[dict]) -> dict[tuple[str, int], list[int]]:
    """(test_name, example_idx) -> row indices, in order."""
    g: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, x in enumerate(meta):
        g[(x["test_name"], x["example_idx"])].append(i)
    return g


def per_subtest_metrics(
    inp: np.ndarray, resp: np.ndarray, meta: list[dict],
    tests_info: dict[str, dict],
) -> list[dict]:
    groups = group_rows(meta)
    # collect per (test_name) lists of eps, delta
    by_test: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"eps": [], "delta": []})
    for (tname, _), idxs in groups.items():
        if len(idxs) < 2:
            continue
        anchor = idxs[0]
        rest = idxs[1:]
        eps = cos_dist(inp[anchor], inp[rest])
        dlt = cos_dist(resp[anchor], resp[rest])
        by_test[tname]["eps"].extend(eps.tolist())
        by_test[tname]["delta"].extend(dlt.tolist())

    rows = []
    for tname, info in tests_info.items():
        arrs = by_test.get(tname, {"eps": [], "delta": []})
        eps = np.asarray(arrs["eps"], dtype=float)
        dlt = np.asarray(arrs["delta"], dtype=float)
        rec = {
            "test_name": tname,
            "test_type": info["type"],
            "capability": info["capability"],
            "n_cases": info["n_cases"],
            "fail_rate": info["fail_rate"],
            "n_pairs": int(dlt.size),
            "mean_delta": float(dlt.mean()) if dlt.size else np.nan,
            "median_delta": float(np.median(dlt)) if dlt.size else np.nan,
            "mean_eps": float(eps.mean()) if eps.size else np.nan,
        }
        mask = eps > EPS_FLOOR
        if mask.any():
            ratios = dlt[mask] / eps[mask]
            rec["rho_median"] = float(np.median(ratios))
            rec["rho_mean"] = float(np.mean(ratios))
            rec["n_pairs_rho"] = int(mask.sum())
        else:
            rec["rho_median"] = np.nan
            rec["rho_mean"] = np.nan
            rec["n_pairs_rho"] = 0
        rows.append(rec)
    return rows


def roll_up_capability(sub_df: pd.DataFrame) -> pd.DataFrame:
    # weighted means by n_cases (for fail_rate) and n_pairs (for delta/eps)
    def agg(g: pd.DataFrame) -> pd.Series:
        w_cases = g["n_cases"].to_numpy()
        w_pairs = g["n_pairs"].to_numpy()
        out = {
            "n_subtests": len(g),
            "n_cases": int(w_cases.sum()),
            "n_pairs": int(w_pairs.sum()),
        }
        if w_cases.sum() > 0:
            out["fail_rate"] = float(
                np.average(g["fail_rate"], weights=w_cases))
        else:
            out["fail_rate"] = np.nan
        if w_pairs.sum() > 0:
            out["mean_delta"] = float(
                np.average(g["mean_delta"].fillna(0), weights=w_pairs))
            out["mean_eps"] = float(
                np.average(g["mean_eps"].fillna(0), weights=w_pairs))
        else:
            out["mean_delta"] = np.nan
            out["mean_eps"] = np.nan
        out["rho_median"] = float(g["rho_median"].median(skipna=True))
        return pd.Series(out)

    return (sub_df
            .groupby(["capability", "test_type", "model"], dropna=False)
            .apply(agg, include_groups=False)
            .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/analysis/embedding_by_type")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "subtest").mkdir(parents=True, exist_ok=True)
    (out / "capability").mkdir(parents=True, exist_ok=True)

    all_sub = []
    for task in args.tasks:
        per_model_rows = []
        for mk in MODELS:
            try:
                tests_info = load_tests_full(task, mk)
                inp, resp, meta = load_embeddings(task, mk)
            except FileNotFoundError as e:
                print(f"[skip] {task}/{mk}: {e}")
                continue
            rows = per_subtest_metrics(inp, resp, meta, tests_info)
            for r in rows:
                r["model"] = mk
                r["task"] = task
            per_model_rows.extend(rows)

        if not per_model_rows:
            continue
        sub_df = pd.DataFrame(per_model_rows)
        sub_df.to_csv(out / "subtest" / f"{task}.csv", index=False)

        cap_df = roll_up_capability(sub_df)
        cap_df["task"] = task
        cap_df.to_csv(out / "capability" / f"{task}.csv", index=False)

        all_sub.append(sub_df)
        print(f"[{task}] wrote {len(sub_df)} subtest rows, "
              f"{len(cap_df)} capability rows")

    if all_sub:
        full = pd.concat(all_sub, ignore_index=True)
        # task x type x model summary
        summary = (full
                   .groupby(["task", "test_type", "model"], dropna=False)
                   .apply(lambda g: pd.Series({
                       "n_subtests": len(g),
                       "n_cases": int(g["n_cases"].sum()),
                       "n_pairs": int(g["n_pairs"].sum()),
                       "fail_rate": float(np.average(
                           g["fail_rate"],
                           weights=g["n_cases"].clip(lower=1))),
                       "mean_delta": float(np.average(
                           g["mean_delta"].fillna(0),
                           weights=g["n_pairs"].clip(lower=1))),
                       "median_delta": float(g["median_delta"].median(
                           skipna=True)),
                       "rho_median": float(g["rho_median"].median(
                           skipna=True)),
                   }), include_groups=False)
                   .reset_index())
        summary.to_csv(out / "summary.csv", index=False)
        print(f"wrote summary.csv ({len(summary)} rows)")
        _plot(full, out / "by_type_bars.png")


def _plot(full: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    types = ["MFT", "INV", "DIR"]
    tasks = sorted(full["task"].unique())
    fig, axes = plt.subplots(
        len(tasks), len(types),
        figsize=(5 * len(types), 3.2 * len(tasks)),
        squeeze=False)

    for i, task in enumerate(tasks):
        for j, tt in enumerate(types):
            ax = axes[i][j]
            sub = full[(full["task"] == task) & (full["test_type"] == tt)]
            if sub.empty:
                ax.set_visible(False)
                continue
            pivot = (sub.groupby(["capability", "model"])["mean_delta"]
                     .mean().unstack("model"))
            pivot.plot.bar(ax=ax, width=0.8)
            ax.set_title(f"{task} / {tt}")
            ax.set_ylabel("mean delta (1 - cos)")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote plot -> {path}")


if __name__ == "__main__":
    main()
