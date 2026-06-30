#!/usr/bin/env python3
"""Emit LaTeX full-result tables (overall + per-subtest) from two checklist_summary.json files.

Matches the style of results/tables/qqp_full_tables.tex and sentiment_full_tables.tex:
  - One model per summary file (typical for large runs saved separately).
  - Rows aligned by subtest name; cases must match between models.

Example:

  python scripts/export_checklist_full_tables.py \\
  --llada-summary results/checklist/squad/llada_rerun_fixed/checklist_summary.json \\
  --llama-summary results/checklist/squad/llama_rerun_fixed/checklist_summary.json \\
    --dataset-name SQuAD \\
    --out results/tables/squad_full_tables.tex
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Same ordering as scripts/plot_checklist_results.py
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


def _prepare_rows(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _latex_escape(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        else:
            out.append(ch)
    return "".join(out)


def _load_single_model_tests(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected non-empty object in {path}")
    keys = [k for k in data if isinstance(data[k], dict) and "tests" in data[k]]
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one model entry in {path}, found {keys!r}")
    k = keys[0]
    return k, data[k]["tests"]


def build_tables(
    llada_path: Path,
    llama_path: Path,
    dataset_name: str,
) -> str:
    _lk, llada_tests = _load_single_model_tests(llada_path)
    _mk, llama_tests = _load_single_model_tests(llama_path)
    by_name_a = {t["name"]: t for t in llada_tests}
    by_name_b = {t["name"]: t for t in llama_tests}
    if set(by_name_a) != set(by_name_b):
        only_a = set(by_name_a) - set(by_name_b)
        only_b = set(by_name_b) - set(by_name_a)
        raise ValueError(f"Subtest names differ: only in llada {only_a!r}, only in llama {only_b!r}")

    ordered = _prepare_rows(list(by_name_a.values()))

    n_tests = len(ordered)
    tot_a = sum(t["n_fails"] for t in llada_tests)
    tot_b = sum(t["n_fails"] for t in llama_tests)
    cases_a = sum(t["n_cases"] for t in llada_tests)
    cases_b = sum(t["n_cases"] for t in llama_tests)
    if cases_a != cases_b:
        raise ValueError(f"Total cases mismatch: llada {cases_a}, llama {cases_b}")
    total_cases = cases_a
    rate_a = tot_a / total_cases if total_cases else 0.0
    rate_b = tot_b / total_cases if total_cases else 0.0

    lines: list[str] = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{pdflscape}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        "",
        r"\begin{document}",
        "",
        r"\begin{center}",
        rf"{{\Large CheckList {dataset_name} Full Results}}",
        r"\end{center}",
        "",
        r"\vspace{0.5em}",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Tests & Total Cases & Total Fails & Fail Rate \\",
        r"\midrule",
        rf"LLaDA & {n_tests} & {total_cases} & {tot_a} & {rate_a:.6f} \\",
        rf"Llama & {n_tests} & {total_cases} & {tot_b} & {rate_b:.6f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Overall full-run CheckList {dataset_name} results.}}",
        r"\end{table}",
        "",
        r"\begin{landscape}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\small",
        r"\begin{longtable}{p{7.5cm}p{2.0cm}rrrrr}",
        rf"\caption{{Per-subtest full-run CheckList {dataset_name} results for LLaDA and Llama.}} \\",
        r"\toprule",
        r"Subtest & Capability & Cases & LLaDA Fails & LLaDA Fail Rate & Llama Fails & Llama Fail Rate \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Subtest & Capability & Cases & LLaDA Fails & LLaDA Fail Rate & Llama Fails & Llama Fail Rate \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    for t in ordered:
        name = t["name"]
        a = by_name_a[name]
        b = by_name_b[name]
        cap = t.get("capability") or ""
        n = a["n_cases"]
        if b["n_cases"] != n:
            raise ValueError(f"Case count mismatch for {name!r}: {a['n_cases']} vs {b['n_cases']}")
        fa, fb = a["n_fails"], b["n_fails"]
        ra = float(fa) / float(n) if n else 0.0
        rb = float(fb) / float(n) if n else 0.0
        lines.append(
            rf"{_latex_escape(name)} & {_latex_escape(cap)} & {n} & {fa} & {ra:.6f} & {fb} & {rb:.6f} \\"
        )

    lines.extend(
        [
            r"\end{longtable}",
            r"\end{landscape}",
            "",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Export CheckList full LaTeX tables from two summaries.")
    p.add_argument("--llada-summary", type=Path, required=True)
    p.add_argument("--llama-summary", type=Path, required=True)
    p.add_argument(
        "--dataset-name",
        type=str,
        default="SQuAD",
        help="Title/caption name (e.g. SQuAD, QQP, Sentiment)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .tex path")
    p.add_argument(
        "--pdf",
        action="store_true",
        help="Run pdflatex on the .tex file (writes PDF next to --out)",
    )
    args = p.parse_args()

    tex = build_tables(args.llada_summary, args.llama_summary, args.dataset_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")
    print(f"Wrote {args.out}")

    if args.pdf:
        exe = "pdflatex"
        cwd = str(args.out.resolve().parent)
        tex_name = args.out.name
        try:
            for _ in range(2):
                r = subprocess.run(
                    [exe, "-interaction=nonstopmode", tex_name],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                )
                if r.returncode != 0:
                    print(r.stdout[-3000:] if r.stdout else "", file=sys.stderr)
                    print(r.stderr[-3000:] if r.stderr else "", file=sys.stderr)
                    sys.exit(r.returncode)
            pdf = args.out.with_suffix(".pdf")
            print(f"Wrote {pdf}")
        except FileNotFoundError:
            print(f"{exe} not found; wrote .tex only", file=sys.stderr)


if __name__ == "__main__":
    main()
