# Robustness of Diffusion and Autoregressive Language Models to CheckList-Style Perturbations

📄 **Paper:** [OpenReview](https://openreview.net/forum?id=UYuYrUrtUw)

**Aarush Rathore, Tirtharaj Dash, Lovekesh Vig, and Ashwin Srinivasan**  
*Grounding Language Models: Learning Faithfully and Efficiently @ EMNLP 2026*

This repository contains the code and analysis artifact for work on the
relative robustness of diffusion and autoregressive language models under the
original CheckList sentiment, QQP, and SQuAD behavioral suites.
This repository contains the code and analysis artifact for work on the
relative robustness of diffusion and autoregressive language models under the
original CheckList sentiment, QQP, and SQuAD behavioral suites. The study
evaluates eight 7--9B instruction-tuned models: three diffusion models
(LLaDA-8B, LLaDA-MoE, Dream-7B) and five autoregressive baselines
(Llama-3.1, Mistral, Qwen2.5, Gemma-2, OLMo-2).

The artifact is intentionally compact. It retains the code, configuration
files, and aggregate result tables needed to inspect and reproduce the reported
pass-rate gaps, perturbation-coefficient ratios, capability-level gaps,
selective-stability analysis, and QQP label-prior audit, along with the scripts
that regenerate the figures. Full raw model outputs and complete run
directories are not uploaded because the full runs are large; the retained
aggregate files are the minimal materials needed to verify the figures and
claims. The manuscript source and compiled PDF are not part of this repository.

## Repository Layout

- `src/dlm_safety/`: reusable model wrappers, CheckList adapters, suite I/O,
  judges, and task utilities.
- `configs/`: experiment configuration files for the CheckList sentiment, QQP,
  and SQuAD runs.
- `scripts/run_checklist.py`: the evaluation entry point that runs a config's
  models over a CheckList suite and writes per-test outputs.
- `data/checklist/`: instructions for obtaining the original CheckList
  release data needed to rerun evaluations (the data itself is not shipped).
- `scripts/analysis/`: aggregation, auditing, bootstrap, and
  perturbation-coefficient analysis scripts.
- `scripts/plotting/`: scripts that regenerate the figures from retained
  CSV summaries.
- `scripts/utils/`: utility scripts for suite validation, rescoring, recovery,
  and exports.
- `results/`: compact aggregate tables and figure inputs.
- `tests/`: focused unit tests for SQuAD utilities, LLM judging, and Dream
  decoding-label behavior.

The plotting and analysis scripts write generated figures to a `paper/fig/`
directory, which they create on first run.

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
evaluation, which requires external model weights and the CheckList release
data that are intentionally excluded from this artifact (see "Rerunning
Evaluations" below).

## Regenerate Figures

The figures are generated from compact CSV summaries retained under
`results/`.

```bash
python scripts/plotting/make_emnlp_checklist_figures.py
python scripts/plotting/make_emnlp_extra_figures.py
```

These commands write PDF and PNG figures to `paper/fig/` (created on first
run). The four primary figures are:

- `paper/fig/emnlp_ratio_heatmap_column.pdf`
- `paper/fig/emnlp_capability_gap.pdf`
- `paper/fig/emnlp_selective_stability_map.pdf`
- `paper/fig/emnlp_qqp_validity_audit.pdf`

Additional generated figures support checks and appendix-style views used
during analysis.

## Rerunning Evaluations (Optional)

The retained aggregates are sufficient to verify the figures and claims, so
rerunning the model evaluations is not required. To rerun them anyway, install
the `checklist` extra (`pip install -e ".[analysis,checklist]"`), download the
CheckList release data following `data/checklist/README.md`, and run the
evaluation entry point with one of the retained configs, e.g.:

```bash
python scripts/run_checklist.py --config configs/checklist/phase2_checklist.yaml
```

Model weights are pulled from the Hugging Face Hub as referenced in each
config; a GPU is required.

## Result Provenance

The compact retained result folders are:

- `results/analysis/`
- `results/analysis/paper_figures/`
- `results/analysis/embedding_by_type/`
- `results/expansion/analysis/`
- `results/expansion/contraction/`

These files support the aggregate tables, figure inputs, bootstrap
intervals, label-prior audits, and reported perturbation-coefficient
comparisons. The scripts under `scripts/analysis/` document how the retained
tables were derived from full CheckList run outputs.

Full raw model outputs, complete run directories, model checkpoints, local
caches, old drafts, and machine-specific files are intentionally excluded from
this artifact because the full runs are large and not needed to verify the
figures and aggregate claims.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{rathore2026robustness,
  title     = {Robustness of Diffusion and Autoregressive Language Models to {CheckList}-Style Perturbations},
  author    = {Aarush Rathore and Tirtharaj Dash and Lovekesh Vig and Ashwin Srinivasan},
  booktitle = {Grounding Language Models: Learning Faithfully and Efficiently @ EMNLP 2026},
  year      = {2026},
  url       = {https://openreview.net/forum?id=UYuYrUrtUw}
}
```

## Notes

This repository holds code and analysis artifacts only. It should not contain
local virtual environments, `.pytest_cache`, `__pycache__`, LaTeX build
byproducts, raw archives, or manuscript drafts.
