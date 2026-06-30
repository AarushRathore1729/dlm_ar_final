# DLM Safety: CheckList Robustness Analysis

This repository contains the anonymous-review artifact for a paper on the
relative robustness of diffusion and autoregressive language models under the
original CheckList sentiment, QQP, and SQuAD behavioral suites. The paper
evaluates eight 7--9B instruction-tuned models: three diffusion models
(LLaDA-8B, LLaDA-MoE, Dream-7B) and five autoregressive baselines
(Llama-3.1, Mistral, Qwen2.5, Gemma-2, OLMo-2).

The artifact is intentionally compact. It retains the code, configuration
files, aggregate result tables, paper figures, and paper source needed to
inspect and reproduce the reported pass-rate gaps, perturbation-coefficient
ratios, capability-level gaps, selective-stability analysis, and QQP
label-prior audit. Full raw model outputs and complete run directories are not
uploaded because the full runs are large; the retained aggregate files are the
minimal materials needed to verify the submitted figures and claims.

The active review PDF is `paper/emnlp2026/main.pdf`; its LaTeX source is
`paper/emnlp2026/main.tex`.

## Repository Layout

- `src/dlm_safety/`: reusable model wrappers, CheckList adapters, suite I/O,
  judges, and task utilities.
- `configs/`: experiment configuration files for the retained CheckList and
  expansion runs.
- `scripts/analysis/`: aggregation, auditing, bootstrap, and
  perturbation-coefficient analysis scripts.
- `scripts/plotting/`: scripts that regenerate the paper figures from retained
  CSV summaries.
- `scripts/utils/`: utility scripts for suite validation, rescoring, recovery,
  and exports.
- `results/`: compact aggregate tables and figure inputs used by the paper.
- `paper/fig/`: retained PDF/PNG figure assets.
- `paper/emnlp2026/`: final anonymous paper source, bibliography, ACL style
  files, retained `main.bbl`, and compiled `main.pdf`.
- `tests/`: focused unit tests for SQuAD utilities, LLM judging, and Dream
  decoding-label behavior.

## Setup

Python 3.10 is recommended.

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[analysis]"
```

If `uv` is unavailable, use a standard Python 3.10 virtual environment and
install the package with:

```bash
pip install -e ".[analysis]"
```

The optional `checklist` extra is only needed for workflows that rebuild or
inspect original CheckList suites. The paper figures and retained aggregate
checks use the compact files already stored under `results/`.

## Quick Verification

Run the unit tests from the repository root:

```bash
PYTHONPATH=src pytest -q
```

The tests cover the local judging/parsing utilities that are most likely to
affect the retained aggregate analyses. They do not rerun the full model
evaluation, which requires external model weights and the raw CheckList outputs
that are intentionally excluded from this artifact.

## Regenerate Paper Figures

The paper figures are generated from compact CSV summaries retained under
`results/`.

```bash
python scripts/plotting/make_emnlp_checklist_figures.py
python scripts/plotting/make_emnlp_extra_figures.py
```

These commands write PDF and PNG figures to `paper/fig/`. The four figures
directly included by `main.tex` are:

- `paper/fig/emnlp_ratio_heatmap_column.pdf`
- `paper/fig/emnlp_capability_gap.pdf`
- `paper/fig/emnlp_selective_stability_map.pdf`
- `paper/fig/emnlp_qqp_validity_audit.pdf`

Additional retained figures in `paper/fig/` support checks and appendix-style
views used during analysis.

## Rebuild Paper

```bash
cd paper/emnlp2026
latexmk -g -pdf -interaction=nonstopmode main.tex
```

The compiled review PDF is written to `paper/emnlp2026/main.pdf`. The retained
`main.bbl` is included so the paper can also be inspected in environments where
running BibTeX is inconvenient.

## Result Provenance

The compact retained result folders are:

- `results/analysis/`
- `results/analysis/paper_figures/`
- `results/analysis/embedding_by_type/`
- `results/expansion/analysis/`
- `results/expansion/contraction/`

These files support the paper's aggregate tables, figure inputs, bootstrap
intervals, label-prior audits, and reported perturbation-coefficient
comparisons. The scripts under `scripts/analysis/` document how the retained
tables were derived from full CheckList run outputs.

Full raw model outputs, complete run directories, model checkpoints, local
caches, old drafts, and machine-specific files are intentionally excluded from
this anonymous artifact because the full runs are large and not needed to verify
the submitted figures and aggregate claims.

## Notes for Anonymous Review

This repository is prepared for anonymous review. It should not contain local
virtual environments, `.pytest_cache`, `__pycache__`, LaTeX build byproducts,
raw archives, or non-anonymous drafts. The paper includes a data/code
availability statement and an acknowledgment of AI-assistant use.
