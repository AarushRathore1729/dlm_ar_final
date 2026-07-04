# CheckList release data

CheckList (ACL 2020) test suites and test inputs are not shipped in the library; they live in the repo's release archive.

## Get release data

1. Clone or download the CheckList repo and unpack the release data:

   ```bash
   # From project root
   cd data/checklist
   wget https://github.com/marcotcr/checklist/raw/master/release_data.tar.gz
   tar xvf release_data.tar.gz
   ```

   Or clone the repo and copy:

   ```bash
   git clone https://github.com/marcotcr/checklist.git /tmp/checklist
   cp /tmp/checklist/release_data.tar.gz data/checklist/
   cd data/checklist && tar xvf release_data.tar.gz
   ```

2. Expected layout after unpacking:

   - `release_data/sentiment/sentiment_suite.pkl` — sentiment TestSuite
   - `release_data/sentiment/tests_n500` — test texts (or similar)
   - `release_data/qqp/` — QQP suite and tests (optional)

   **Python version:** The official `sentiment_suite.pkl` was built with Python &lt;3.11. It will not load on Python 3.12 (dill/code object change). Use **Python 3.10** for `run_checklist.py`, or re-generate the suite with Python 3.12 using the CheckList repo.

## Re-pickled `*_py310.pkl` suites

The configs under `configs/` reference `sentiment_suite_py310.pkl`,
`qqp_suite_py310.pkl`, and `squad_suite_py310.pkl`. These are the official
suites re-saved under Python 3.10 so they load reliably in this environment.
To produce them after downloading the release data, load each official suite
(or rebuild it from raw release data) and re-save it with:

```bash
python scripts/utils/use_checklist_suite_api.py \
    --dataset squad \
    --release-data-dir data/checklist/release_data \
    --save-suite data/checklist/release_data/squad/squad_suite_py310.pkl
```

Alternatively, point `checklist.suite_path` at the official pickle:
`run_checklist.py` falls back to rebuilding the suite from raw release data
when a pickle cannot be loaded (`checklist.raw_fallback`).

## Config

Point `checklist.release_data_dir` in your config to the directory that contains `sentiment/` (and optionally `qqp/`), e.g. `data/checklist/release_data` or `release_data` if you unpack in project root.
