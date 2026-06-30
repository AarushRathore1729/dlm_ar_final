"""
Produce a side-by-side table of two v5 metric implementations:

  (A) Categorical  — 1 if predicted_label changed, 0 otherwise
  (B) Cosine       — 1 - cos(embed(norm(label)), embed(norm(label')))
                     all-MiniLM-L6-v2; labels normalised (lowercase, strip punct)

Per-subtest fail rates shown for INV and DIR on sentiment and QQP.
Saves:
  results/analysis/v5_cat_vs_cos_inv.csv
  results/analysis/v5_cat_vs_cos_dir.csv
  results/analysis/v5_cat_vs_cos_inv.tex   (LaTeX table)
  results/analysis/v5_cat_vs_cos_dir.tex

Run on remote:
  python scripts/analysis/v5_cosine_vs_categorical_table.py
"""

import json, os, re, collections
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

RESULTS  = Path("results/checklist")
OUT_DIR  = Path("results/analysis")
OUT_DIR.mkdir(exist_ok=True)
TASKS    = ["sentiment", "qqp"]
MODELS   = {"llada": "llada_rerun_fixed", "llama": "llama_rerun_fixed"}
SBERT    = "all-MiniLM-L6-v2"


# ── helpers ──────────────────────────────────────────────────────────────────

def norm_label(s):
    """Lowercase and strip trailing punctuation/whitespace."""
    return re.sub(r"[^\w]+$", "", s.strip().lower())


def load_examples(task, model_key):
    d = RESULTS / task / MODELS[model_key]
    f = next(d.glob("examples_full_*.jsonl"))
    with open(f) as fp:
        return [json.loads(l) for l in fp]


def build_label_embs(all_examples, sbert_model):
    """Embed every normalised unique label across all examples."""
    labels = sorted({
        norm_label(e["predicted_label"])
        for ex_list in all_examples
        for e in ex_list
        if e["predicted_label"]
    })
    print(f"  Label vocabulary (normalised): {labels}")
    embs = sbert_model.encode(labels, normalize_embeddings=True)

    # Show pairwise distances so you can inspect the scale
    print("  Pairwise cosine distances:")
    dist = {}
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            d = float(1 - np.dot(embs[i], embs[j]))
            dist[(l1, l2)] = d
            print(f"    dist({l1!r}, {l2!r}) = {d:.4f}")
    return {l: e for l, e in zip(labels, embs)}, dist


def per_subtest_metrics(examples, label_embs, test_type):
    """
    Returns dict: test_name -> {"n": int, "cat": float, "cos": float}

    cat = 1 - mean(pass)          (categorical fail rate)
    cos = mean cosine delta        (0 for same label, c_ij for different)

    'pass' already encodes label change; cosine is computed as:
      pass=True  → delta = 0
      pass=False → delta = dist(embed(pred_label), embed(modal_label))
    where modal_label is the most common label in the subtest
    (proxy for the original label, exact when only 2 label values exist).
    """
    ex = [e for e in examples
          if e["test_type"] == test_type and e["pass"] is not None]

    by_test = collections.defaultdict(list)
    for e in ex:
        by_test[e["test_name"]].append(e)

    rows = {}
    for test_name, group in by_test.items():
        n = len(group)
        passes   = [bool(g["pass"]) for g in group]
        cat_fail = 1 - sum(passes) / n

        # Modal label = most common predicted_label in this subtest
        modal = norm_label(
            collections.Counter(g["predicted_label"] for g in group)
            .most_common(1)[0][0]
        )

        cos_deltas = []
        for g in group:
            if bool(g["pass"]):
                cos_deltas.append(0.0)
            else:
                pred = norm_label(g["predicted_label"])
                if pred == modal:       # degenerate: failed but same label as modal
                    cos_deltas.append(0.0)
                else:
                    cos_deltas.append(
                        float(1 - np.dot(label_embs[pred], label_embs[modal]))
                    )

        rows[test_name] = {
            "n": n,
            "cat": round(cat_fail, 4),
            "cos": round(float(np.mean(cos_deltas)), 4),
        }
    return rows


def build_table(task, test_type, sbert_model):
    ex = {mk: load_examples(task, mk) for mk in ["llada", "llama"]}
    label_embs, _ = build_label_embs(list(ex.values()), sbert_model)

    metrics = {
        mk: per_subtest_metrics(ex[mk], label_embs, test_type)
        for mk in ["llada", "llama"]
    }

    all_tests = sorted(set(metrics["llada"]) & set(metrics["llama"]))
    rows = []
    for t in all_tests:
        ll = metrics["llada"][t]
        lm = metrics["llama"][t]
        rows.append({
            "subtest":      t,
            "n":            ll["n"],
            "LLaDA_cat":    ll["cat"],
            "Llama_cat":    lm["cat"],
            "gap_cat":      round(ll["cat"] - lm["cat"], 4),
            "LLaDA_cos":    ll["cos"],
            "Llama_cos":    lm["cos"],
            "gap_cos":      round(ll["cos"] - lm["cos"], 4),
            "rank_agrees":  (ll["cat"] < lm["cat"]) == (ll["cos"] < lm["cos"])
                            or (ll["cat"] == lm["cat"] and ll["cos"] == lm["cos"]),
        })

    return pd.DataFrame(rows)


def to_latex(df, task, test_type):
    lines = [
        r"\begin{table}[H]\centering",
        r"{\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Subtest & $n$ & \multicolumn{2}{c}{Categorical fail rate} & "
        r"\multicolumn{2}{c}{Cosine fail rate} & Rank \\",
        r" & & LLaDA & Llama & LLaDA & Llama & agrees? \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        agree = r"\checkmark" if row["rank_agrees"] else r"\textbf{!}"
        lines.append(
            f"{row['subtest'][:55]} & {row['n']} & "
            f"{row['LLaDA_cat']:.3f} & {row['Llama_cat']:.3f} & "
            f"{row['LLaDA_cos']:.4f} & {row['Llama_cos']:.4f} & "
            f"{agree} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        f"\\caption{{v5 metric comparison: categorical vs cosine on normalised "
        f"labels. {task.upper()} {test_type} subtests. "
        r"``Rank agrees'' = both methods agree on which model has lower fail rate.}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    print("Loading sentence embedding model...")
    sbert_model = SentenceTransformer(SBERT)

    for task in TASKS:
        for test_type in ["INV", "DIR"]:
            print(f"\n{'='*60}")
            print(f"  {task.upper()} — {test_type}")
            print(f"{'='*60}")

            df = build_table(task, test_type, sbert_model)
            if df.empty:
                print("  (no data)")
                continue

            # Print to console
            print(df.to_string(index=False, float_format="{:.4f}".format))

            # Ranking summary
            agree_n  = df["rank_agrees"].sum()
            total_n  = len(df)
            print(f"\n  Ranking agreement: {agree_n}/{total_n} subtests")
            if agree_n < total_n:
                print("  DISAGREEMENTS:")
                print(df[~df["rank_agrees"]][["subtest","gap_cat","gap_cos"]].to_string())

            # Save CSV
            csv_path = OUT_DIR / f"v5_cat_vs_cos_{task}_{test_type.lower()}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

            # Save LaTeX
            tex = to_latex(df, task, test_type)
            tex_path = OUT_DIR / f"v5_cat_vs_cos_{task}_{test_type.lower()}.tex"
            tex_path.write_text(tex)
            print(f"  Saved: {tex_path}")


if __name__ == "__main__":
    main()
