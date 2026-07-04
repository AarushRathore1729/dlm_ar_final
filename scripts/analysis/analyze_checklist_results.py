#!/usr/bin/env python3
"""Post-hoc comparative analysis of CheckList reruns (LLaDA vs Llama).

Reads examples_full JSONL traces + suite dumps + checklist_summary.json for each
run, then computes:

  - Fail-rate deltas and two-proportion z-scores per test
  - PSR (perturbation sensitivity): INV and DIR tests, fraction of (anchor, pert) pairs where
    predictions differ
  - DCR (directional compliance proxy): for suite DIR tests, 1 - fail_rate from summary
  - Delta-perp: among INV pairs, P(perturbed fails | anchor passes)
  - ConfShift: mean L1 distance between probability vectors (sentiment/QQP only)
  - Gamma (amplification): PSR(cap) / MFT fail rate for same model and capability

Writes:
  - results/analysis/comparative_summary.json
  - results/analysis/comparative_tables.tex
  - results/analysis/plots/*.png

Usage:
    python scripts/analyze_checklist_results.py
    python scripts/analyze_checklist_results.py --root /path/to/project
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


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


def load_suite_types(suite_path: Path) -> dict[str, str]:
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for t in data.get("tests", []):
        out[t["name"]] = t.get("test_type", "MFT")
    return out


def _example_input_width(ex: dict[str, Any]) -> int:
    """Number of flattened trace rows one suite example expands to."""
    inp = ex.get("input")
    if isinstance(inp, list):
        return len(inp)
    return 1


def load_inv_dir_group_sizes(suite_data: dict[str, Any], test_name: str) -> list[int] | None:
    """Per-suite-example row counts (list inputs flatten to len(list) rows each)."""
    for t in suite_data.get("tests", []):
        if t.get("name") != test_name:
            continue
        exs = t.get("examples") or []
        return [_example_input_width(ex) for ex in exs]
    return None


def iter_inv_dir_groups(
    rows: list[dict[str, Any]], sizes: list[int] | None
) -> Iterator[list[dict[str, Any]]]:
    """Yield groups of trace rows: original first, perturbations follow (suite order)."""
    if not rows:
        return
    if sizes:
        total = sum(sizes)
        if total == len(rows):
            cursor = 0
            for sz in sizes:
                yield rows[cursor : cursor + sz]
                cursor += sz
            return
        if total > len(rows):
            # Partial run: only complete groups that fit.
            cursor = 0
            for sz in sizes:
                if cursor + sz > len(rows):
                    break
                yield rows[cursor : cursor + sz]
                cursor += sz
            return
        if total < len(rows):
            # Extra trailing rows (unexpected): analyze the prefix that matches the suite.
            cursor = 0
            for sz in sizes:
                yield rows[cursor : cursor + sz]
                cursor += sz
            return

    # Legacy: multiple JSONL lines per example_idx (anchor + perturbations).
    by_ex: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ex[int(r.get("example_idx", 0))].append(r)
    for k in sorted(by_ex.keys()):
        grp = by_ex[k]
        if len(grp) >= 2:
            yield grp

    if any(len(by_ex[k]) >= 2 for k in by_ex):
        return

    # Flattened traces: one row per global example_idx — pair consecutive rows.
    n_cases = rows[0].get("test_n_cases")
    if n_cases is not None and len(rows) == 2 * int(n_cases):
        for i in range(0, len(rows), 2):
            yield rows[i : i + 2]
        return
    if len(rows) >= 2 and len(rows) % 2 == 0:
        for i in range(0, len(rows), 2):
            yield rows[i : i + 2]


def load_summary_by_name(summary_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    key = next(k for k, v in data.items() if isinstance(v, dict) and "tests" in v)
    return {t["name"]: t for t in data[key]["tests"]}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pred_key(task: str, row: dict[str, Any]) -> str:
    if task in ("sentiment", "qqp"):
        idx = row.get("predicted_label_idx")
        if idx is not None:
            return str(int(idx))
        pl = row.get("predicted_label")
        return str(pl) if pl is not None else ""
    pred = row.get("predicted_label")
    if pred is not None and str(pred).strip():
        return str(pred).strip()
    return str(row.get("raw_prediction") or row.get("model_response") or "").strip()


def conf_vec_flat(row: dict[str, Any]) -> np.ndarray | None:
    """Single probability vector for this row, or None."""
    c = row.get("probabilities")
    if c is None:
        c = row.get("confidence")
    if c is None:
        return None
    if isinstance(c, (int, float)):
        return None
    if isinstance(c, list):
        if not c:
            return None
        if isinstance(c[0], list):
            return None
        try:
            return np.array([float(x) for x in c], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    return None


def conf_pair_vecs(row0: dict[str, Any], row1: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """For nested confidence [[orig],[pert]] on a single row, or separate rows."""
    c0 = row0.get("confidence")
    c1 = row1.get("confidence")
    if isinstance(c0, list) and len(c0) == 2 and isinstance(c0[0], list) and isinstance(c0[1], list):
        try:
            a = np.array([float(x) for x in c0[0]], dtype=np.float64)
            b = np.array([float(x) for x in c0[1]], dtype=np.float64)
            if a.shape == b.shape:
                return a, b
        except (TypeError, ValueError):
            pass
    v0 = conf_vec_flat(row0)
    v1 = conf_vec_flat(row1)
    if v0 is not None and v1 is not None and v0.shape == v1.shape:
        return v0, v1
    return None


def two_proportion_z(n1: int, k1: int, n2: int, k2: int) -> float | None:
    if n1 <= 0 or n2 <= 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    if p <= 0 or p >= 1:
        return None
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    return (p1 - p2) / se


def analyze_traces(
    task: str,
    suite_types: dict[str, str],
    suite_path: Path,
    jsonl_path: Path,
) -> dict[str, Any]:
    """INV/DIR pair statistics from one model's examples_full JSONL."""
    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    rows_by_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(jsonl_path):
        rows_by_test[row["test_name"]].append(row)

    inv_psr_pairs = 0
    inv_psr_mismatch = 0
    inv_delta_denom = 0
    inv_delta_num = 0
    conf_shifts: list[float] = []
    per_test: dict[str, dict[str, Any]] = {}

    for test_name, rows in rows_by_test.items():
        st = suite_types.get(test_name, "MFT")
        if st not in ("INV", "DIR"):
            continue

        sizes = load_inv_dir_group_sizes(suite_data, test_name)
        t_pairs = 0
        t_mis = 0
        t_dd = 0
        t_dn = 0
        for grp in iter_inv_dir_groups(rows, sizes):
            if len(grp) < 2:
                continue
            anchor = grp[0]
            pa = pred_key(task, anchor)
            ap = anchor.get("pass")
            for pert in grp[1:]:
                pp = pred_key(task, pert)
                t_pairs += 1
                if pa != pp:
                    t_mis += 1
                if ap is True:
                    t_dd += 1
                    if pert.get("pass") is not True:
                        t_dn += 1
                if task in ("sentiment", "qqp"):
                    vecs = conf_pair_vecs(anchor, pert)
                    if vecs is not None:
                        a, b = vecs
                        conf_shifts.append(float(np.sum(np.abs(a - b))))
        if t_pairs:
            per_test[test_name] = {
                "suite_type": st,
                "capability": rows[0].get("test_capability"),
                "n_pairs": t_pairs,
                "n_mismatch": t_mis,
                "psr": t_mis / t_pairs,
                "delta_perp": (t_dn / t_dd) if t_dd else None,
            }
        inv_psr_pairs += t_pairs
        inv_psr_mismatch += t_mis
        inv_delta_denom += t_dd
        inv_delta_num += t_dn

    return {
        "inv_pair_count": inv_psr_pairs,
        "inv_psr": (inv_psr_mismatch / inv_psr_pairs) if inv_psr_pairs else None,
        "delta_perp": (inv_delta_num / inv_delta_denom) if inv_delta_denom else None,
        "mean_conf_shift": (float(np.mean(conf_shifts)) if conf_shifts else None),
        "per_test_inv": per_test,
    }


def aggregate_capability(
    task: str,
    suite_types: dict[str, str],
    by_name_llada: dict[str, dict[str, Any]],
    by_name_llama: dict[str, dict[str, Any]],
    trace_llada: dict[str, Any],
    trace_llama: dict[str, Any],
) -> dict[str, Any]:
    """Per-capability rollups."""
    caps: set[str] = set()
    for t in by_name_llada.values():
        caps.add(t.get("capability") or "Other")

    mft_fail_llada: dict[str, list[tuple[int, int]]] = defaultdict(list)
    mft_fail_llama: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for name, t in by_name_llada.items():
        if suite_types.get(name) != "MFT":
            continue
        cap = t.get("capability") or "Other"
        mft_fail_llada[cap].append((t["n_cases"], t["n_fails"]))
    for name, t in by_name_llama.items():
        if suite_types.get(name) != "MFT":
            continue
        cap = t.get("capability") or "Other"
        mft_fail_llama[cap].append((t["n_cases"], t["n_fails"]))

    def mft_rate(pairs: list[tuple[int, int]]) -> float | None:
        if not pairs:
            return None
        nc = sum(a for a, _ in pairs)
        nf = sum(b for _, b in pairs)
        return nf / nc if nc else None

    inv_pairs_llada: dict[str, int] = defaultdict(int)
    inv_mis_llada: dict[str, float] = defaultdict(float)
    for name, info in trace_llada["per_test_inv"].items():
        cap = info.get("capability") or "Other"
        inv_pairs_llada[cap] += info["n_pairs"]
        inv_mis_llada[cap] += float(info.get("n_mismatch", info["psr"] * info["n_pairs"]))

    inv_pairs_llama: dict[str, int] = defaultdict(int)
    inv_mis_llama: dict[str, float] = defaultdict(float)
    for name, info in trace_llama["per_test_inv"].items():
        cap = info.get("capability") or "Other"
        inv_pairs_llama[cap] += info["n_pairs"]
        inv_mis_llama[cap] += float(info.get("n_mismatch", info["psr"] * info["n_pairs"]))

    out: dict[str, Any] = {}
    for cap in sorted(caps, key=_cap_sort_key):
        r_a = mft_rate(mft_fail_llada[cap])
        r_b = mft_rate(mft_fail_llama[cap])
        psr_a = (
            inv_mis_llada[cap] / inv_pairs_llada[cap] if inv_pairs_llada[cap] else None
        )
        psr_b = (
            inv_mis_llama[cap] / inv_pairs_llama[cap] if inv_pairs_llama[cap] else None
        )
        gamma_a = (psr_a / r_a) if (psr_a is not None and r_a and r_a > 0) else None
        gamma_b = (psr_b / r_b) if (psr_b is not None and r_b and r_b > 0) else None
        ratio = None
        if gamma_a and gamma_b:
            ratio = gamma_b / gamma_a
        out[cap] = {
            "mft_fail_rate_llada": r_a,
            "mft_fail_rate_llama": r_b,
            "inv_psr_llada": psr_a,
            "inv_psr_llama": psr_b,
            "gamma_llada": gamma_a,
            "gamma_llama": gamma_b,
            "gamma_ratio_llama_over_llada": ratio,
        }
    return out


def plot_psr_caps(task: str, cap_data: dict[str, Any], out_path: Path) -> None:
    caps = sorted(cap_data.keys(), key=_cap_sort_key)
    x = np.arange(len(caps))
    w = 0.35
    a = [cap_data[c].get("inv_psr_llada") or 0.0 for c in caps]
    b = [cap_data[c].get("inv_psr_llama") or 0.0 for c in caps]
    fig, ax = plt.subplots(figsize=(max(8, len(caps) * 0.5), 4))
    ax.bar(x - w / 2, a, width=w, label="LLaDA", color="tab:blue")
    ax.bar(x + w / 2, b, width=w, label="Llama", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(caps, rotation=30, ha="right")
    ax.set_ylabel("INV PSR (pred mismatch rate)")
    ax.set_title(f"{task.upper()} — perturbation sensitivity by capability")
    ax.legend()
    ax.set_ylim(0, max(1e-6, max(a + b) * 1.15))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_delta_perp_caps(task: str, per_inv_llada: dict, per_inv_llama: dict, out_path: Path) -> None:
    """Bar chart of delta-perp per test (top 15 by average delta)."""
    names = sorted(
        set(per_inv_llada.keys()) & set(per_inv_llama.keys()),
        key=lambda n: max(
            per_inv_llada[n].get("delta_perp") or 0,
            per_inv_llama[n].get("delta_perp") or 0,
        ),
        reverse=True,
    )[:15]
    if not names:
        return
    x = np.arange(len(names))
    w = 0.35
    da = [per_inv_llada[n].get("delta_perp") or 0.0 for n in names]
    db = [per_inv_llama[n].get("delta_perp") or 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.25)))
    ax.barh(x - w / 2, da, height=w, label="LLaDA", color="tab:blue")
    ax.barh(x + w / 2, db, height=w, label="Llama", color="tab:orange")
    ax.set_yticks(x)
    ax.set_yticklabels([n[:48] for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Delta-perp (fail pert | anchor pass)")
    ax.set_title(f"{task.upper()} — top INV tests by conditional degradation")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_tex_summary(payload: dict[str, Any]) -> str:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{booktabs}",
        r"\begin{document}",
        "",
        r"\section*{CheckList comparative summary (post-fix reruns)}",
        "",
    ]
    for task, block in payload["tasks"].items():
        lines.append(rf"\subsection*{{{task.upper()}}}")
        oa = block["overall"]
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(
            r"Model & Grouped cases & Fails & Fail rate & INV PSR \\ \midrule"
        )
        for model in ("llada", "llama"):
            mo = oa[model]
            lines.append(
                rf"{model} & {mo['n_cases']:,d} & {mo['n_fails']:,d} & {mo['fail_rate']:.4f} & {mo.get('inv_psr') or 0:.4f} \\"
            )
        lines.append(r"\bottomrule\end{tabular}")
        lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def run_task(
    task: str,
    root: Path,
    suite_rel: str,
    llada_dir: Path,
    llama_dir: Path,
) -> dict[str, Any]:
    suite_path = root / suite_rel
    suite_types = load_suite_types(suite_path)
    mid_l = "GSAI-ML_LLaDA-8B-Instruct"
    mid_m = "meta-llama_Llama-3.1-8B-Instruct"

    sum_l = load_summary_by_name(llada_dir / "checklist_summary.json")
    sum_m = load_summary_by_name(llama_dir / "checklist_summary.json")

    trace_l = analyze_traces(
        task, suite_types, suite_path, llada_dir / f"examples_full_{mid_l}.jsonl"
    )
    trace_m = analyze_traces(
        task, suite_types, suite_path, llama_dir / f"examples_full_{mid_m}.jsonl"
    )

    per_test_rows: list[dict[str, Any]] = []
    for name in sorted(set(sum_l.keys()) & set(sum_m.keys())):
        a, b = sum_l[name], sum_m[name]
        n = a["n_cases"]
        if b["n_cases"] != n:
            continue
        fa, fb = a["n_fails"], b["n_fails"]
        ra, rb = fa / n, fb / n
        st = suite_types.get(name, "MFT")
        dcr_a = (1.0 - ra) if st == "DIR" else None
        dcr_b = (1.0 - rb) if st == "DIR" else None
        z = two_proportion_z(n, fa, n, fb)
        per_test_rows.append(
            {
                "name": name,
                "suite_type": st,
                "capability": a.get("capability"),
                "n_cases": n,
                "n_fails_llada": fa,
                "n_fails_llama": fb,
                "fail_rate_llada": ra,
                "fail_rate_llama": rb,
                "fail_rate_delta_llada_minus_llama": ra - rb,
                "two_proportion_z": z,
                "dcr_llada_proxy": dcr_a,
                "dcr_llama_proxy": dcr_b,
            }
        )

    cap_block = aggregate_capability(
        task, suite_types, sum_l, sum_m, trace_l, trace_m
    )

    n_cases = sum(t["n_cases"] for t in sum_l.values())
    fails_l = sum(t["n_fails"] for t in sum_l.values())
    fails_m = sum(t["n_fails"] for t in sum_m.values())

    return {
        "suite_path": str(suite_path),
        "per_test": per_test_rows,
        "per_capability": cap_block,
        "overall": {
            "llada": {
                "n_cases": n_cases,
                "n_fails": fails_l,
                "fail_rate": fails_l / n_cases if n_cases else 0,
                "inv_psr": trace_l["inv_psr"],
                "delta_perp": trace_l["delta_perp"],
                "mean_conf_shift": trace_l["mean_conf_shift"],
            },
            "llama": {
                "n_cases": n_cases,
                "n_fails": fails_m,
                "fail_rate": fails_m / n_cases if n_cases else 0,
                "inv_psr": trace_m["inv_psr"],
                "delta_perp": trace_m["delta_perp"],
                "mean_conf_shift": trace_m["mean_conf_shift"],
            },
        },
        "trace_detail_llada": trace_l,
        "trace_detail_llama": trace_m,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparative CheckList analysis (LLaDA vs Llama).")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    args = ap.parse_args()
    root: Path = args.root
    rc = root / "results" / "checklist"

    out_dir = root / "results" / "analysis"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks_payload: dict[str, Any] = {}
    for task, suite_rel, ldir, mdir in (
        (
            "sentiment",
            "results/checklist/suites/sentiment_suite_original_dump.json",
            rc / "sentiment" / "llada_rerun_fixed",
            rc / "sentiment" / "llama_rerun_fixed",
        ),
        (
            "qqp",
            "results/checklist/suites/qqp_suite_original_dump.json",
            rc / "qqp" / "llada_rerun_fixed",
            rc / "qqp" / "llama_rerun_fixed",
        ),
        (
            "squad",
            "results/checklist/suites/squad_suite_original_dump.json",
            rc / "squad" / "llada_rerun_fixed",
            rc / "squad" / "llama_rerun_fixed",
        ),
    ):
        tasks_payload[task] = run_task(task, root, suite_rel, ldir, mdir)
        plot_psr_caps(task, tasks_payload[task]["per_capability"], plot_dir / f"{task}_psr_by_capability.png")
        plot_delta_perp_caps(
            task,
            tasks_payload[task]["trace_detail_llada"]["per_test_inv"],
            tasks_payload[task]["trace_detail_llama"]["per_test_inv"],
            plot_dir / f"{task}_delta_perp_top_inv.png",
        )
        # Drop heavy nested per-example detail from JSON output
        tasks_payload[task].pop("trace_detail_llada", None)
        tasks_payload[task].pop("trace_detail_llama", None)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks_payload,
    }
    (out_dir / "comparative_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "comparative_tables.tex").write_text(
        build_tex_summary(payload), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'comparative_summary.json'}")
    print(f"Wrote {out_dir / 'comparative_tables.tex'}")
    print(f"Wrote plots under {plot_dir}")


if __name__ == "__main__":
    main()
