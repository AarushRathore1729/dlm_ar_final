#!/usr/bin/env python3
"""option_order_probe.py — does label-option order in the prompt drive the
Qwen/Gemma QQP priors?

Background: on the QQP reflexivity test (two IDENTICAL questions, ground
truth 'duplicate'), greedy-decoded Qwen2.5-7B answers 'not_duplicate' 67%
of the time and Gemma-2-9B answers 'duplicate' 100% — content-insensitive,
opposite-direction priors. The prompt ends "Reply with exactly one label:
duplicate or not_duplicate." Hypothesis: Qwen picks the LAST-mentioned
option, Gemma the FIRST (option-order/recency bias).

Probe: rerun the same 1000 verbatim prompts in (a) original order and
(b) reversed order ("not_duplicate or duplicate"), greedy, chat template
applied — identical to the original pipeline. If per-prompt labels flip
with the order, the prior is option-order bias; if they stay put, it is a
content/format prior.

Run (remote GPU):
  python scripts/analysis/option_order_probe.py --model qwen
  python scripts/analysis/option_order_probe.py --model gemma

Outputs: results/expansion/analysis/option_order_probe_<model>.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Gemma-2 triggers torch.compile/inductor; the remote lacks Python dev headers
# so Triton kernel compilation fails (known issue from the May expansion runs).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo": "allenai/OLMo-2-1124-7B-Instruct",
}
# DLMs go through the dlm_safety classes with the same generation params as
# their canonical QQP runs (LLaDA/MoE: steps=8, block 32; Dream: steps=64;
# max_new_tokens=16 for all), mirroring the checklist pipeline call path.
DLM_MODELS = {"llada", "llada_moe", "dream"}
MANIFEST = Path("results/expansion/analysis/option_order_probe_manifest.json")
OUT_DIR = Path("results/expansion/analysis")

ORIG = "duplicate or not_duplicate"
REVERSED = "not_duplicate or duplicate"
# Third arm: same task, label-free phrasing. Tests whether the deficit is
# bound to the label format or is a capability gap.
INSTR_ORIG = "Reply with exactly one label: duplicate or not_duplicate."
INSTR_YESNO = "Answer with exactly one word: yes or no."


def classify(text: str) -> str:
    """Mirror the pipeline's exact + minimal-normalization parsing
    (DLMs occasionally emit 'not' or 'duplicate_duplicate')."""
    t = text.strip().lower().strip(".:'\" ")
    if t.startswith("not_duplicate") or t.startswith("not duplicate") or t == "not":
        return "not_duplicate"
    if t.startswith("duplicate"):
        return "duplicate"
    return f"OTHER:{t[:30]}"


def classify_yesno(text: str) -> str:
    t = text.strip().lower().strip(".:,'\" ")
    if t.startswith("yes"):
        return "duplicate"
    if t.startswith("no"):
        return "not_duplicate"
    return f"OTHER:{t[:30]}"


def _build_dlm(key: str):
    """Instantiate a DLM via dlm_safety with its canonical QQP params."""
    import sys
    sys.path.insert(0, "src")
    if key == "llada":
        from dlm_safety.models.llada_model import LLaDAModel
        return LLaDAModel(
            model_name_or_path="GSAI-ML/LLaDA-8B-Instruct", device_str="auto",
            sampling_steps=8, block_length=32, torch_dtype="bfloat16",
        ), "GSAI-ML/LLaDA-8B-Instruct"
    if key == "llada_moe":
        from dlm_safety.models.llada_moe_model import LLaDAMoEModel
        return LLaDAMoEModel(
            model_name_or_path="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
            device_str="auto", sampling_steps=8, block_length=32,
            torch_dtype="bfloat16",
        ), "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    from dlm_safety.models.dream_model import DreamModel
    return DreamModel(
        model_name_or_path="Dream-org/Dream-v0-Instruct-7B", device_str="auto",
        steps=64, torch_dtype="bfloat16",
    ), "Dream-org/Dream-v0-Instruct-7B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(set(MODELS) | DLM_MODELS), required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    prompts = json.load(open(MANIFEST))
    assert all(ORIG in p for p in prompts), "manifest prompts missing the option phrase"

    if args.model in DLM_MODELS:
        dlm, name = _build_dlm(args.model)
        args.max_new_tokens = 16  # match the canonical QQP runs

        def generate(batch_prompts: list[str]) -> list[str]:
            _, texts = dlm.get_responses(
                batch_prompts, batched=True,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
            return texts
    else:
        name = MODELS[args.model]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        def generate(batch_prompts: list[str]) -> list[str]:
            formatted = [
                tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
            enc = tok(formatted, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            return [
                tok.decode(out[i][enc.input_ids.shape[1]:], skip_special_tokens=True)
                for i in range(len(batch_prompts))
            ]

    results = []
    arms = [
        ("original", lambda p: p, classify),
        ("reversed", lambda p: p.replace(ORIG, REVERSED), classify),
        ("yesno", lambda p: p.replace(INSTR_ORIG, INSTR_YESNO), classify_yesno),
    ]
    for order, transform, clf in arms:
        labels = []
        for i in range(0, len(prompts), args.batch_size):
            batch = [transform(p) for p in prompts[i:i + args.batch_size]]
            labels += [clf(t) for t in generate(batch)]
            if i % 160 == 0:
                print(f"{order}: {i + len(batch)}/{len(prompts)}")
        results.append((order, labels))

    res = dict(results)
    orig_labels, rev_labels, yn_labels = res["original"], res["reversed"], res["yesno"]
    summary = {
        "model": name,
        "n": len(prompts),
        "original_counts": {l: orig_labels.count(l) for l in set(orig_labels)},
        "reversed_counts": {l: rev_labels.count(l) for l in set(rev_labels)},
        "yesno_counts": {l: yn_labels.count(l) for l in set(yn_labels)},
        "n_flipped": sum(1 for a, b in zip(orig_labels, rev_labels)
                         if a != b and not a.startswith("OTHER") and not b.startswith("OTHER")),
        "per_prompt": [
            {"orig": a, "rev": b, "yesno": c}
            for a, b, c in zip(orig_labels, rev_labels, yn_labels)
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"option_order_probe_{args.model}.json"
    json.dump(summary, open(out, "w"))
    print(f"\n{name}")
    print("  original:", summary["original_counts"])
    print("  reversed:", summary["reversed_counts"])
    print("  yes/no  :", summary["yesno_counts"])
    print(f"  per-prompt flips (orig vs rev): {summary['n_flipped']}/{summary['n']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
