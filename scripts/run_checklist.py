#!/usr/bin/env python3
"""
Run CheckList behavioral tests (ACL 2020) on LLaDA and Llama with LLM-as-judge.

Usage:
    python scripts/run_checklist.py --config configs/checklist/phase2_checklist.yaml
    python scripts/run_checklist.py --config configs/checklist/phase2_checklist_squad_original.yaml --demo 5 --model-filter Llama

Requires:
  - checklist package and release_data (see data/checklist/README.md)
  - Config with models (under test), checklist.task, checklist.suite_path, checklist.judge
"""

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from dlm_safety.utils.config import load_config, ModelConfig
from dlm_safety.models.base import LanguageModel
from dlm_safety.models.ar_model import ARModel
from dlm_safety.models.llada_model import LLaDAModel
from dlm_safety.utils.squad_utils import (
    extract_squad_span,
    is_valid_squad_span as _shared_is_valid_squad_span,
    SQUAD_SENTINEL,
)
from dlm_safety.llm_judge import LLMJudge, SentimentPipelineScorer, RegexRefusalScorer
from dlm_safety.checklist_adapter import LMPredictorForCheckList
from dlm_safety.checklist_suite_io import build_suite_from_exported_json


SENTIMENT_TASK_PROMPT = """You are a strict sentiment classifier.
Read the text and decide whether the sentiment is negative, neutral, or positive.
Reply with exactly one lowercase label: negative, neutral, or positive.
Do not explain your answer.
Do not repeat or quote the text.
If the sentiment is mixed, ambiguous, or unclear, reply with neutral.

{text}"""

QQP_TASK_PROMPT = """Determine whether these two questions are duplicates. Reply with exactly one label: duplicate or not_duplicate.

Question 1: {question1}
Question 2: {question2}"""

SQUAD_TASK_PROMPT = """Read the passage and answer the question.
Output exactly one answer span copied verbatim from the passage.
Do not explain.
Do not answer in a full sentence.
Do not repeat the question.
Do not add any words that are not in the passage.
Do not use quotes or prefixes like "Answer:".
If the answer is a short phrase, output only that phrase.

Passage: {passage}
Question: {question}
Answer:"""


def create_model(model_cfg: ModelConfig) -> LanguageModel:
    """Instantiate a model from config (same as run_experiment)."""
    if model_cfg.type == "ar":
        return ARModel(
            model_name_or_path=model_cfg.name,
            device_str=model_cfg.device,
            **model_cfg.model_kwargs,
        )
    if model_cfg.type == "llada":
        return LLaDAModel(
            model_name_or_path=model_cfg.name,
            device_str=model_cfg.device,
            sampling_steps=model_cfg.sampling_steps,
            **model_cfg.model_kwargs,
        )
    if model_cfg.type == "dream":
        from dlm_safety.models.dream_model import DreamModel
        return DreamModel(
            model_name_or_path=model_cfg.name,
            device_str=model_cfg.device,
            steps=model_cfg.sampling_steps,
            **model_cfg.model_kwargs,
        )
    if model_cfg.type == "llada_moe":
        from dlm_safety.models.llada_moe_model import LLaDAMoEModel
        return LLaDAMoEModel(
            model_name_or_path=model_cfg.name,
            device_str=model_cfg.device,
            sampling_steps=model_cfg.sampling_steps,
            **model_cfg.model_kwargs,
        )
    if model_cfg.type == "diffucoder":
        from dlm_safety.models.diffucoder_model import DiffuCoderModel
        return DiffuCoderModel(
            model_name_or_path=model_cfg.name,
            device_str=model_cfg.device,
            steps=model_cfg.sampling_steps,
            **model_cfg.model_kwargs,
        )
    raise ValueError(f"Unsupported model type for CheckList: {model_cfg.type!r}")


def _model_config_from_dict(d: dict) -> ModelConfig:
    return ModelConfig(
        name=d["name"],
        type=d.get("type", "ar"),
        device=d.get("device", "auto"),
        tokenizer=d.get("tokenizer"),
        sampling_steps=d.get("sampling_steps", 100),
        seq_length=d.get("seq_length", 128),
        model_kwargs=d.get("model_kwargs", {}),
    )


def _get_suite_tests(suite):
    """Iterable of (test_name, test) — supports both .tests (current) and .test_list (legacy)."""
    if hasattr(suite, "tests") and suite.tests:
        for name, t in suite.tests.items():
            yield name, t
        return
    for t in getattr(suite, "test_list", []):
        yield getattr(t, "name", str(t)), t


def _limit_suite_to_demo(suite, limit: int) -> None:
    """Slice each test's data to first `limit` examples for a quick demo run."""
    for _name, t in _get_suite_tests(suite):
        if hasattr(t, "data") and t.data is not None:
            t.data = t.data[:limit]
        if hasattr(t, "labels") and isinstance(t.labels, list):
            t.labels = t.labels[:limit]
        if hasattr(t, "meta") and isinstance(t.meta, list):
            t.meta = t.meta[:limit]
        if hasattr(t, "run_idxs"):
            t.run_idxs = None
        if hasattr(t, "result_indexes"):
            t.result_indexes = None
        if hasattr(t, "data_len"):
            t.data_len = len(t.data) if t.data is not None else 0
        if hasattr(t, "n"):
            t.n = len(t.data) if t.data is not None else 0


def _test_case_count(test) -> int:
    return (
        getattr(test, "n", None)
        or (len(test.data) if getattr(test, "data", None) is not None else None)
        or getattr(test, "data_len", None)
        or 0
    )


def _print_suite_tests(suite) -> None:
    print("Suite tests to run:")
    for test_name, test in _get_suite_tests(suite):
        print(f"  - {test_name}: {_test_case_count(test)} cases")


def _normalize_pass_value(value):
    if value is None:
        return None
    try:
        return bool(value)
    except Exception:
        return None


def _checklist_score_to_pass(value) -> bool | None:
    """Match CheckList semantics: True / numbers > 0 mean pass; <= 0 means fail."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value > 0
    try:
        return bool(value)
    except Exception:
        return None


def _flat_coords(ri: list[int], flat_idx: int) -> tuple[int, int] | None:
    """Map flattened example index → (testcase_index, variant_index within testcase)."""
    if flat_idx < 0 or flat_idx >= len(ri):
        return None
    tc_idx = int(ri[flat_idx])
    variant_idx = sum(1 for j in range(flat_idx) if int(ri[j]) == tc_idx)
    return tc_idx, variant_idx


def _synthetic_flat_result_indexes_from_data(test, n_flat: int) -> list[int] | None:
    """Rebuild CheckList-style flat→testcase mapping when ``result_indexes`` was cleared.

    Demo mode sets ``result_indexes`` to None; grouped MFT still flattens as
    ``[g0_v0, g0_v1, g1_v0, g1_v1, ...]``.
    """
    data = getattr(test, "data", None) or []
    if not data or n_flat <= 0:
        return None
    el0 = data[0]
    if isinstance(el0, str):
        return None
    if not isinstance(el0, (list, tuple, np.ndarray)):
        return None
    if isinstance(el0, np.ndarray) and el0.ndim == 0:
        return None
    out: list[int] = []
    for ti, group in enumerate(data):
        try:
            glen = len(group)
        except TypeError:
            return None
        for _ in range(glen):
            out.append(ti)
    if len(out) != n_flat:
        return None
    return out


def _effective_flat_result_indexes(test, n_flat: int) -> list[int] | None:
    ri = getattr(test, "result_indexes", None)
    if ri is not None and len(ri) == n_flat:
        try:
            return [int(x) for x in ri]
        except (TypeError, ValueError):
            pass
    return _synthetic_flat_result_indexes_from_data(test, n_flat)


def _expected_label_for_flat(test, flat_idx: int, ri: list[int] | None) -> Any:
    labels = getattr(test, "labels", None)
    if labels is None:
        return None
    if ri is None:
        if isinstance(labels, (list, tuple, np.ndarray)) and flat_idx < len(labels):
            return labels[flat_idx]
        return labels
    coords = _flat_coords(ri, flat_idx)
    if coords is None:
        return None
    tc_idx, variant_idx = coords
    if not isinstance(labels, (list, tuple, np.ndarray)):
        return labels
    if tc_idx >= len(labels):
        return None
    label_row = labels[tc_idx]
    if isinstance(label_row, str):
        return label_row if variant_idx == 0 else None
    if isinstance(label_row, (list, tuple, np.ndarray)):
        if variant_idx < len(label_row):
            return label_row[variant_idx]
    return label_row


def _result_cell_at_variant(seq, tc_idx: int, variant_idx: int):
    """Index preds/confs/expect row for grouped MFT (one row per testcase)."""
    if seq is None or tc_idx >= len(seq):
        return None
    cell = seq[tc_idx]
    if isinstance(cell, str):
        return cell if variant_idx == 0 else None
    if isinstance(cell, np.ndarray):
        cell = cell.tolist()
    if isinstance(cell, (list, tuple)):
        if variant_idx < len(cell):
            return cell[variant_idx]
        return None
    return cell if variant_idx == 0 else None


def _pass_for_flat_row(
    test, flat_idx: int, ri: list[int] | None, passed: list, expect_results: list
):
    if ri is not None:
        coords = _flat_coords(ri, flat_idx)
        if coords is None:
            return None
        tc_idx, variant_idx = coords
        if expect_results and tc_idx < len(expect_results):
            row = expect_results[tc_idx]
            if row is not None and variant_idx < len(row):
                return _checklist_score_to_pass(row[variant_idx])
        if tc_idx < len(passed):
            return _normalize_pass_value(passed[tc_idx])
        return None
    if flat_idx < len(passed):
        return _normalize_pass_value(passed[flat_idx])
    return None


def _as_list(value):
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    # NumPy scalars (np.bool_, np.int64, …) are not handled by json.dumps
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist", None)):
        try:
            return _json_safe(value.tolist())
        except TypeError:
            pass
    return str(value)


def _serialize_task_input(task: str, item):
    if task == "sentiment":
        return {"text": str(item)}
    if task == "qqp":
        if isinstance(item, dict):
            return {
                "question1": item.get("question1", item.get("q1", "")),
                "question2": item.get("question2", item.get("q2", "")),
            }
        q1, q2 = item
        return {"question1": q1, "question2": q2}
    if task == "squad":
        return {
            "passage": item.get("passage", ""),
            "question": item.get("question", ""),
        }
    return {"input": _json_safe(item)}


def _extract_suite_results(suite, example_traces: dict | None = None) -> dict:
    """Extract failure info from CheckList suite after run."""
    out = {"tests": [], "total_fails": 0, "total_cases": 0, "example_count": 0}
    example_traces = example_traces or {}
    for name, t in _get_suite_tests(suite):
        name = (name or getattr(t, "name", str(t)))[:200]
        n_cases = _test_case_count(t)
        n_fails = getattr(t, "fail_count", None) or getattr(t, "n_fails", None)
        if n_fails is None and hasattr(t, "fail_idxs"):
            try:
                n_fails = len(t.fail_idxs())
            except Exception:
                n_fails = None
        if n_fails is None and hasattr(t, "results") and hasattr(t.results, "passed"):
            try:
                n_fails = int((~t.results.passed).sum()) if hasattr(t.results.passed, "sum") else sum(1 for r in t.results.passed if r is False)
            except Exception:
                n_fails = None
        examples = []
        trace_rows = example_traces.get(name, [])
        results_obj = getattr(t, "results", None)
        passed = _as_list(getattr(results_obj, "passed", None))
        preds = _as_list(getattr(results_obj, "preds", None))
        confs = _as_list(getattr(results_obj, "confs", None))
        expect_results = _as_list(getattr(results_obj, "expect_results", None))
        n_flat = len(trace_rows)
        ri = _effective_flat_result_indexes(t, n_flat)
        use_coords = ri is not None
        for idx in range(n_flat):
            expected_label = _expected_label_for_flat(t, idx, ri)
            pass_value = _pass_for_flat_row(t, idx, ri, passed, expect_results)
            if use_coords:
                coords = _flat_coords(ri, idx)
                raw_pred = (
                    _json_safe(_result_cell_at_variant(preds, coords[0], coords[1]))
                    if coords is not None
                    else None
                )
                conf_cell = (
                    _result_cell_at_variant(confs, coords[0], coords[1])
                    if coords is not None
                    else None
                )
                exp_cell = (
                    _result_cell_at_variant(expect_results, coords[0], coords[1])
                    if coords is not None and expect_results
                    else None
                )
            else:
                raw_pred = (
                    _json_safe(preds[idx]) if idx < len(preds) and preds[idx] is not None else None
                )
                conf_cell = confs[idx] if idx < len(confs) else None
                exp_cell = expect_results[idx] if idx < len(expect_results) else None
            if hasattr(conf_cell, "tolist"):
                conf_out = conf_cell.tolist()
            else:
                conf_out = conf_cell
            if hasattr(exp_cell, "tolist"):
                exp_out = exp_cell.tolist()
            else:
                exp_out = exp_cell
            examples.append(
                {
                    "example_idx": idx,
                    "input_text": trace_rows[idx].get("input_text"),
                    "input_structured": trace_rows[idx].get("input_structured"),
                    "prompt": trace_rows[idx].get("prompt"),
                    "model_response": trace_rows[idx].get("model_response"),
                    "predicted_label": trace_rows[idx].get("predicted_label"),
                    "predicted_label_idx": trace_rows[idx].get("predicted_label_idx"),
                    "expected_label": expected_label,
                    "pass": pass_value,
                    "parse_mode": trace_rows[idx].get("parse_mode"),
                    "repair_output": trace_rows[idx].get("repair_output"),
                    "raw_prediction": raw_pred,
                    "confidence": conf_out,
                    "expectation_score": exp_out,
                    "judge_output": trace_rows[idx].get("judge_output"),
                    "probabilities": trace_rows[idx].get("probabilities"),
                    "raw_prediction_line": trace_rows[idx].get("raw_prediction_line"),
                    "raw_response_before_repair": trace_rows[idx].get("raw_response_before_repair"),
                    "extraction_trace": trace_rows[idx].get("extraction_trace"),
                    "normalization": trace_rows[idx].get("normalization"),
                }
            )
        fail_rate = (float(n_fails) / float(n_cases)) if (n_fails is not None and n_cases) else None
        predicted_counter = Counter(
            str(example.get("predicted_label"))
            for example in examples
            if example.get("predicted_label") is not None
        )
        out["tests"].append({
            "name": name,
            "type": t.__class__.__name__,
            "capability": getattr(t, "capability", None),
            "description": getattr(t, "description", None),
            "n_cases": n_cases,
            "n_fails": n_fails,
            "fail_rate": fail_rate,
            "predicted_label_counts": dict(predicted_counter),
            "examples": examples,
        })
        out["example_count"] += len(examples)
        if n_cases and n_fails is not None:
            out["total_cases"] += n_cases
            out["total_fails"] += n_fails
    return out


def _build_test_trace_map(suite, flat_rows: list[dict]) -> dict[str, list[dict]]:
    tests = list(_get_suite_tests(suite))
    test_ranges = getattr(suite, "test_ranges", None) or {}
    out = {}

    if test_ranges:
        for test_name, test in tests:
            row_range = test_ranges.get(test_name)
            if row_range and len(row_range) == 2:
                start, end = row_range
                out[test_name] = flat_rows[start:end]
            else:
                out[test_name] = []
        return out

    cursor = 0
    for test_name, test in tests:
        n = _test_case_count(test)
        out[test_name] = flat_rows[cursor:cursor + n]
        cursor += n
    return out


def _build_examples_full_records(results: dict, run_metadata: dict) -> list[dict]:
    records = []
    base_meta = {
        "run_name": run_metadata.get("run_name"),
        "task": run_metadata.get("task"),
        "model": run_metadata.get("model"),
        "judge_type": run_metadata.get("judge_type"),
        "official_run_from_file": run_metadata.get("official_run_from_file"),
    }
    for test in results.get("tests", []):
        for example in test.get("examples", []):
            row = dict(base_meta)
            row.update(
                {
                    "test_name": test.get("name"),
                    "test_capability": test.get("capability"),
                    "test_description": test.get("description"),
                    "test_type": test.get("type"),
                    "test_n_cases": test.get("n_cases"),
                    "test_n_fails": test.get("n_fails"),
                    "test_fail_rate": test.get("fail_rate"),
                }
            )
            row.update(example)
            records.append(_json_safe(row))
    return records


def _build_suite_visual_snapshot(results: dict) -> dict:
    return {
        "total_cases": results.get("total_cases"),
        "total_fails": results.get("total_fails"),
        "fail_rate": (
            float(results["total_fails"]) / float(results["total_cases"])
            if results.get("total_cases")
            else None
        ),
        "tests": [
            {
                "name": test.get("name"),
                "capability": test.get("capability"),
                "description": test.get("description"),
                "n_cases": test.get("n_cases"),
                "n_fails": test.get("n_fails"),
                "fail_rate": test.get("fail_rate"),
                "predicted_label_counts": test.get("predicted_label_counts", {}),
            }
            for test in results.get("tests", [])
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def _write_detailed_result_bundle(out_dir: Path, safe_name: str, results: dict, run_metadata: dict) -> None:
    examples_full = _build_examples_full_records(results, run_metadata)
    tests_full = {
        "run_metadata": _json_safe(run_metadata),
        "summary": _build_suite_visual_snapshot(results),
        "tests": results.get("tests", []),
    }
    suite_visual_snapshot = {
        "run_metadata": _json_safe(run_metadata),
        "suite": _build_suite_visual_snapshot(results),
    }

    _write_jsonl(out_dir / f"examples_full_{safe_name}.jsonl", examples_full)
    _write_json(out_dir / f"tests_full_{safe_name}.json", tests_full)
    _write_json(out_dir / f"suite_visual_snapshot_{safe_name}.json", suite_visual_snapshot)
    _write_json(out_dir / f"run_metadata_{safe_name}.json", run_metadata)


def _read_sentiment_predictions(pred_path: Path) -> list[tuple[int, list[float]]]:
    rows = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            probs = [float(x) for x in parts[1:]]
            rows.append((label, probs))
    return rows


def _resolve_release_dir(release_dir: str | None) -> Path | None:
    if not release_dir:
        return None
    path = Path(release_dir)
    if not path.is_absolute():
        path = project_root / path
    return path


def _load_official_task_inputs(task: str, release_dir: Path, limit: int | None = None):
    if task == "sentiment":
        items = (release_dir / "sentiment" / "tests_n500").read_text(encoding="utf-8").splitlines()
        return items[:limit] if limit is not None else items

    if task == "qqp":
        rows = []
        with (release_dir / "qqp" / "tests_n500").open(encoding="utf-8") as f:
            header = f.readline()
            if not header:
                return rows
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                rows.append((parts[1], parts[2]))
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    if task == "squad":
        rows = []
        with (release_dir / "squad" / "squad.jsonl").open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    raise ValueError(f"Unsupported official task input loading for task={task!r}")


def _task_prompt_template(task: str, configured_template: str | None) -> str:
    if configured_template:
        return configured_template
    if task == "sentiment":
        return SENTIMENT_TASK_PROMPT
    if task == "qqp":
        return QQP_TASK_PROMPT
    if task == "squad":
        return SQUAD_TASK_PROMPT
    raise ValueError(f"No default prompt template for task={task!r}")


def _build_task_prompts(task: str, items: list, task_prompt_template: str) -> list[str]:
    prompts = []
    if task == "sentiment":
        for text in items:
            prompts.append(task_prompt_template.format(text=text))
        return prompts
    if task == "qqp":
        for q1, q2 in items:
            prompts.append(task_prompt_template.format(text=f"Q1: {q1}\nQ2: {q2}", question1=q1, question2=q2))
        return prompts
    if task == "squad":
        for record in items:
            prompts.append(
                task_prompt_template.format(
                    text=f"Passage: {record.get('passage', '')}\nQuestion: {record.get('question', '')}",
                    passage=record.get("passage", ""),
                    question=record.get("question", ""),
                )
            )
        return prompts
    raise ValueError(f"Unsupported task for prompt generation: {task!r}")


def _write_official_predictions(
    task: str,
    pred_path: Path,
    responses: list[str],
    judge,
    judge_inputs: list[str],
    qqp_file_format: str = "binary_conf",
) -> list[dict]:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    if task == "squad":
        with pred_path.open("w", encoding="utf-8") as f:
            for response in responses:
                answer = (response or "").strip().replace("\n", " ")
                f.write(answer + "\n")
                trace_rows.append(
                    {
                        "predicted_label": answer,
                        "predicted_label_idx": None,
                        "judge_output": None,
                        "probabilities": None,
                        "parse_mode": "pred_only",
                        "raw_prediction_line": answer,
                    }
                )
        return trace_rows

    preds, probs = judge.evaluate(judge_inputs, responses)
    judge_trace = getattr(judge, "last_trace", [])

    if task == "sentiment":
        label_names = ["negative", "neutral", "positive"]
        with pred_path.open("w", encoding="utf-8") as f:
            for idx, (pred, row) in enumerate(zip(preds, probs)):
                line = f"{int(pred)} {float(row[0]):.6f} {float(row[1]):.6f} {float(row[2]):.6f}"
                f.write(line + "\n")
                judge_info = judge_trace[idx] if idx < len(judge_trace) else {}
                trace_rows.append(
                    {
                        "predicted_label": label_names[int(pred)],
                        "predicted_label_idx": int(pred),
                        "judge_output": judge_info.get("judge_output"),
                        "probabilities": row.tolist() if hasattr(row, "tolist") else list(row),
                        "parse_mode": judge_info.get("parse_mode", "pipeline"),
                        "repair_output": judge_info.get("repair_output"),
                        "raw_prediction_line": line,
                    }
                )
        return trace_rows

    if task == "qqp":
        label_names = ["not_duplicate", "duplicate"]
        duplicate_idx = 1
        with pred_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(probs):
                conf = float(row[duplicate_idx])
                pred = 1 if conf >= 0.5 else 0
                if qqp_file_format == "pred_and_conf":
                    line = f"{pred} {conf:.6f}"
                else:
                    line = f"{conf:.6f}"
                f.write(line + "\n")
                judge_info = judge_trace[idx] if idx < len(judge_trace) else {}
                trace_rows.append(
                    {
                        "predicted_label": label_names[pred],
                        "predicted_label_idx": pred,
                        "judge_output": judge_info.get("judge_output"),
                        "probabilities": row.tolist() if hasattr(row, "tolist") else list(row),
                        "parse_mode": judge_info.get("parse_mode", "judge_binary_conf"),
                        "repair_output": judge_info.get("repair_output"),
                        "raw_prediction_line": line,
                    }
                )
        return trace_rows

    raise ValueError(f"Unsupported task for writing predictions: {task!r}")


def _is_valid_squad_span(answer: str, record: dict, max_answer_chars: int = 128) -> bool:
    passage = (record.get("passage") or "").strip()
    return _shared_is_valid_squad_span(answer, passage, max_answer_chars=max_answer_chars)


def _repair_squad_responses(
    task_inputs: list,
    responses: list[str],
    primary_model,
    max_new_tokens: int,
    max_answer_chars: int = 128,
    retry_max: int = 1,
    fallback_model=None,
) -> tuple[list[str], dict, list[dict]]:
    fixed = [((r or "").strip().replace("\n", " ")) for r in responses]
    per_example_traces = [
        {
            "final_mode": "direct_valid",
            "extraction_trace": None,
            "primary_repair_output": None,
            "primary_repair_trace": None,
            "fallback_repair_output": None,
            "fallback_repair_trace": None,
        }
        for _ in responses
    ]
    repaired_extraction = 0
    repaired_primary = 0
    repaired_fallback = 0
    still_invalid = []

    # --- Step 1: deterministic extraction for all invalid responses ---
    for idx, (record, answer) in enumerate(zip(task_inputs, fixed)):
        if _is_valid_squad_span(answer, record, max_answer_chars=max_answer_chars):
            per_example_traces[idx]["extraction_trace"] = {
                "extraction_mode": "already_valid",
                "stripped_cue": None,
                "base_extraction_mode": None,
                "question_rule": None,
            }
            continue
        # Try deterministic span extraction
        passage = (record.get("passage", "") or "").strip()
        question = (record.get("question", "") or "").strip()
        extracted, trace = extract_squad_span(
            answer,
            passage,
            question=question,
            max_answer_chars=max_answer_chars,
        )
        per_example_traces[idx]["extraction_trace"] = trace
        if extracted != SQUAD_SENTINEL and _is_valid_squad_span(
            extracted, record, max_answer_chars=max_answer_chars
        ):
            fixed[idx] = extracted
            per_example_traces[idx]["final_mode"] = "deterministic_extraction"
            repaired_extraction += 1
        else:
            per_example_traces[idx]["final_mode"] = "still_invalid_after_extraction"
            still_invalid.append(idx)

    # --- Step 2: primary model repair ---
    if still_invalid and retry_max > 0:
        prompts = []
        for idx in still_invalid:
            record = task_inputs[idx]
            prompts.append(
                "Extract the shortest answer span from the passage for the question. "
                "Output only the answer span and nothing else.\n\n"
                f"Passage: {record.get('passage', '')}\n"
                f"Question: {record.get('question', '')}\n"
                f"Current answer: {fixed[idx]}\n"
                "Answer:"
            )
        _, repaired = primary_model.get_responses(
            prompts,
            batched=False,
            max_new_tokens=max(8, max_new_tokens),
            do_sample=False,
        )
        next_invalid = []
        for idx, new_answer in zip(still_invalid, repaired):
            candidate = (new_answer or "").strip().replace("\n", " ")
            per_example_traces[idx]["primary_repair_output"] = candidate
            passage = (task_inputs[idx].get("passage", "") or "").strip()
            question = (task_inputs[idx].get("question", "") or "").strip()
            normalized_candidate, repair_trace = extract_squad_span(
                candidate,
                passage,
                question=question,
                max_answer_chars=max_answer_chars,
            )
            if normalized_candidate != SQUAD_SENTINEL and _is_valid_squad_span(
                normalized_candidate,
                task_inputs[idx],
                max_answer_chars=max_answer_chars,
            ):
                fixed[idx] = normalized_candidate
                per_example_traces[idx]["final_mode"] = "primary_repair"
                per_example_traces[idx]["primary_repair_trace"] = repair_trace
                repaired_primary += 1
            else:
                next_invalid.append(idx)
        still_invalid = next_invalid

    # --- Step 3: fallback model repair ---
    if still_invalid and fallback_model is not None:
        prompts = []
        for idx in still_invalid:
            record = task_inputs[idx]
            prompts.append(
                "You are a strict answer normalizer for extractive QA. "
                "Return exactly one short span copied from the passage that answers the question. "
                "Output only the span.\n\n"
                f"Passage: {record.get('passage', '')}\n"
                f"Question: {record.get('question', '')}\n"
                f"Model answer: {fixed[idx]}\n"
                "Answer span:"
            )
        _, repaired = fallback_model.get_responses(
            prompts,
            batched=False,
            max_new_tokens=max(8, max_new_tokens),
            do_sample=False,
        )
        for idx, new_answer in zip(still_invalid, repaired):
            candidate = (new_answer or "").strip().replace("\n", " ")
            per_example_traces[idx]["fallback_repair_output"] = candidate
            passage = (task_inputs[idx].get("passage", "") or "").strip()
            question = (task_inputs[idx].get("question", "") or "").strip()
            normalized_candidate, repair_trace = extract_squad_span(
                candidate,
                passage,
                question=question,
                max_answer_chars=max_answer_chars,
            )
            if normalized_candidate != SQUAD_SENTINEL and _is_valid_squad_span(
                normalized_candidate,
                task_inputs[idx],
                max_answer_chars=max_answer_chars,
            ):
                fixed[idx] = normalized_candidate
                per_example_traces[idx]["final_mode"] = "fallback_repair"
                per_example_traces[idx]["fallback_repair_trace"] = repair_trace
                repaired_fallback += 1

    # --- Step 4: sentinel for remaining invalid answers ---
    sentinel_count = 0
    for idx, record in enumerate(task_inputs):
        if not _is_valid_squad_span(fixed[idx], record, max_answer_chars=max_answer_chars):
            fixed[idx] = SQUAD_SENTINEL
            per_example_traces[idx]["final_mode"] = "sentinel_fallback"
            sentinel_count += 1

    stats = {
        "deterministic_extraction": repaired_extraction,
        "primary_repaired": repaired_primary,
        "fallback_repaired": repaired_fallback,
        "sentinel_fallback": sentinel_count,
    }
    return fixed, stats, per_example_traces


def _validate_official_prediction_file(
    task: str,
    pred_path: Path,
    expected_lines: int,
    qqp_file_format: str = "binary_conf",
) -> None:
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    lines = pred_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_lines:
        raise ValueError(
            f"Prediction line count mismatch for task={task!r}: "
            f"got {len(lines)}, expected {expected_lines}."
        )

    if task == "sentiment":
        for i, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid sentiment format at line {i}: expected 4 fields, got {len(parts)}"
                )
            try:
                pred_idx = int(parts[0])
                probs = [float(x) for x in parts[1:]]
            except Exception as e:
                raise ValueError(f"Invalid sentiment values at line {i}: {line}") from e
            if pred_idx not in (0, 1, 2):
                raise ValueError(f"Invalid sentiment class index at line {i}: {pred_idx}")
            if any(p < 0.0 or p > 1.0 for p in probs):
                raise ValueError(f"Sentiment probability out of range at line {i}: {probs}")
            if abs(sum(probs) - 1.0) > 0.05:
                raise ValueError(f"Sentiment probabilities do not sum to ~1 at line {i}: {probs}")
        return

    if task == "qqp":
        for i, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if qqp_file_format == "pred_and_conf":
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid qqp pred_and_conf format at line {i}: expected 2 fields, got {len(parts)}"
                    )
                try:
                    pred = int(parts[0])
                    conf = float(parts[1])
                except Exception as e:
                    raise ValueError(f"Invalid qqp pred_and_conf values at line {i}: {line}") from e
                if pred not in (0, 1):
                    raise ValueError(f"Invalid qqp class index at line {i}: {pred}")
                if conf < 0.0 or conf > 1.0:
                    raise ValueError(f"QQP confidence out of range at line {i}: {conf}")
            else:
                if len(parts) != 1:
                    raise ValueError(
                        f"Invalid qqp binary_conf format at line {i}: expected 1 field, got {len(parts)}"
                    )
                try:
                    conf = float(parts[0])
                except Exception as e:
                    raise ValueError(f"Invalid qqp confidence value at line {i}: {line}") from e
                if conf < 0.0 or conf > 1.0:
                    raise ValueError(f"QQP confidence out of range at line {i}: {conf}")
        return

    if task == "squad":
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(f"Invalid squad pred_only format at line {i}: empty answer span")
        return

    raise ValueError(f"Unsupported task for prediction validation: {task!r}")


def _bucket_sentiment_text(text: str) -> str:
    text = text.strip()
    if not text:
        return "empty"
    tokens = text.split()
    if len(tokens) == 1:
        return "single_token"
    if len(tokens) <= 3 and not any(ch in text for ch in ".!?"):
        return "short_phrase"
    return "sentence"


def _build_sentiment_suite_from_raw(test_suite_cls, release_dir: Path, min_agreement: int = 4):
    """Reconstruct a sentiment MFT suite from raw release data.

    The historical `sentiment_suite.pkl` is brittle across Python/dill versions.
    This fallback rebuilds a simpler suite from raw texts plus consensus labels
    from the bundled reference model predictions.
    """
    from checklist.test_types import MFT

    sentiment_dir = release_dir / "sentiment"
    tests_path = sentiment_dir / "tests_n500"
    pred_dir = sentiment_dir / "predictions"
    pred_files = ["amazon", "bert", "google", "microsoft", "roberta"]

    if not tests_path.exists():
        raise FileNotFoundError(f"Raw sentiment test file not found: {tests_path}")
    missing = [name for name in pred_files if not (pred_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing sentiment prediction files needed for raw suite fallback: "
            + ", ".join(str(pred_dir / name) for name in missing)
        )

    texts = tests_path.read_text(encoding="utf-8").splitlines()
    prediction_rows = [_read_sentiment_predictions(pred_dir / name) for name in pred_files]
    lengths = {len(texts), *(len(rows) for rows in prediction_rows)}
    if len(lengths) != 1:
        raise ValueError("Raw sentiment files have inconsistent lengths.")

    label_name = {0: "negative", 1: "neutral", 2: "positive"}
    grouped = defaultdict(list)
    skipped = 0

    for idx, text in enumerate(texts):
        votes = [rows[idx][0] for rows in prediction_rows]
        counts = Counter(votes)
        label, agreement = counts.most_common(1)[0]
        if agreement < min_agreement:
            skipped += 1
            continue
        bucket = _bucket_sentiment_text(text)
        grouped[(bucket, label)].append(text)

    suite = test_suite_cls()
    descriptions = {
        "single_token": "Single-token sentiment cues with strong label consensus.",
        "short_phrase": "Short sentiment phrases with strong label consensus.",
        "sentence": "Sentence-level sentiment examples with strong label consensus.",
    }
    added = 0
    for bucket in ["single_token", "short_phrase", "sentence"]:
        for label in [0, 1, 2]:
            data = grouped.get((bucket, label), [])
            if not data:
                continue
            suite.add(
                MFT(
                    data=data,
                    labels=label,
                    name=f"{bucket}_{label_name[label]}",
                    capability="sentiment",
                    description=descriptions[bucket],
                ),
                overwrite=True,
            )
            added += 1

    if added == 0:
        raise ValueError("Could not build any sentiment tests from raw CheckList data.")

    print(
        "Rebuilt sentiment suite from raw data "
        f"({sum(len(v) for v in grouped.values())} examples kept, {skipped} skipped, "
        f"{added} tests)."
    )
    return suite


def _build_qqp_suite_from_raw(test_suite_cls, release_dir: Path):
    """Reconstruct a basic QQP MFT suite from raw release data and reference probs."""
    from checklist.test_types import MFT

    tests_path = release_dir / "qqp" / "tests_n500"
    pred_path = release_dir / "qqp" / "predictions" / "bert"

    if not tests_path.exists():
        raise FileNotFoundError(f"Raw QQP test file not found: {tests_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"QQP reference predictions not found: {pred_path}")

    pairs = []
    with tests_path.open(encoding="utf-8") as f:
        _header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pairs.append((parts[1], parts[2]))

    probs = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            probs.append(float(line))

    n = min(len(pairs), len(probs))
    pairs = pairs[:n]
    labels = [1 if p >= 0.5 else 0 for p in probs[:n]]  # 1=duplicate, 0=not_duplicate

    suite = test_suite_cls()
    suite.add(
        MFT(
            data=pairs,
            labels=labels,
            name="qqp_raw_reference",
            capability="qqp",
            description="QQP pairs with pseudo labels from BERT reference probabilities (>=0.5 duplicate).",
        ),
        overwrite=True,
    )
    print(f"Rebuilt QQP suite from raw data ({n} examples, 1 test).")
    return suite


def _build_squad_suite_from_raw(test_suite_cls, release_dir: Path):
    """Reconstruct a basic SQuAD-like MFT suite from raw jsonl and reference answers."""
    from checklist.test_types import MFT

    data_path = release_dir / "squad" / "squad.jsonl"
    pred_path = release_dir / "squad" / "predictions" / "bert"

    if not data_path.exists():
        raise FileNotFoundError(f"Raw SQuAD jsonl not found: {data_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"SQuAD reference predictions not found: {pred_path}")

    records = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({"passage": obj.get("passage", ""), "question": obj.get("question", "")})

    answers = [line.rstrip("\n") for line in pred_path.open(encoding="utf-8")]

    n = min(len(records), len(answers))
    records = records[:n]
    answers = answers[:n]

    suite = test_suite_cls()
    suite.add(
        MFT(
            data=records,
            labels=answers,
            name="squad_raw_reference",
            capability="squad",
            description="SQuAD passage-question pairs with pseudo labels from BERT reference answers.",
        ),
        overwrite=True,
    )
    print(f"Rebuilt SQuAD suite from raw data ({n} examples, 1 test).")
    return suite


def _load_suite(test_suite_cls, suite_path: Path, checklist_cfg: dict):
    """Load a CheckList suite with a clearer compatibility error."""
    task = checklist_cfg.get("task")
    if suite_path.suffix.lower() == ".json":
        suite = build_suite_from_exported_json(task, suite_path, test_suite_cls=test_suite_cls)
        setattr(suite, "_raw_fallback_built", False)
        print(f"Loaded suite from stable JSON export: {suite_path.name}")
        return suite

    candidate_paths = [suite_path]
    if task == "sentiment" and suite_path.name == "sentiment_suite.pkl":
        py310_path = suite_path.with_name("sentiment_suite_py310.pkl")
        if py310_path.exists():
            candidate_paths = [py310_path, suite_path]

    last_error = None
    for candidate in candidate_paths:
        try:
            suite = test_suite_cls.from_file(str(candidate))
            setattr(suite, "_raw_fallback_built", False)
            if candidate != suite_path:
                print(f"Loaded suite via compatibility fallback: {candidate.name}")
            return suite
        except Exception as e:
            last_error = e

    try:
        raise last_error if last_error is not None else RuntimeError("Unknown suite load failure")
    except Exception as e:
        release_dir = checklist_cfg.get("release_data_dir")
        enable_raw_fallback = checklist_cfg.get("raw_fallback", True)
        release_dir_path = None
        if release_dir:
            release_dir_path = Path(release_dir)
            if not release_dir_path.is_absolute():
                release_dir_path = project_root / release_dir_path
        if enable_raw_fallback and task == "sentiment" and release_dir:
            print(
                "Could not load suite pickle directly; falling back to raw sentiment "
                f"release data. Original error: {type(e).__name__}: {e}"
            )
            suite = _build_sentiment_suite_from_raw(
                test_suite_cls,
                release_dir=release_dir_path,
                min_agreement=int(checklist_cfg.get("raw_min_agreement", 4)),
            )
            setattr(suite, "_raw_fallback_built", True)
            return suite
        if enable_raw_fallback and task == "qqp" and release_dir:
            print(
                "Could not load suite pickle directly; falling back to raw QQP "
                f"release data. Original error: {type(e).__name__}: {e}"
            )
            suite = _build_qqp_suite_from_raw(
                test_suite_cls,
                release_dir=release_dir_path,
            )
            setattr(suite, "_raw_fallback_built", True)
            return suite
        if enable_raw_fallback and task == "squad" and release_dir:
            print(
                "Could not load suite pickle directly; falling back to raw SQuAD "
                f"release data. Original error: {type(e).__name__}: {e}"
            )
            suite = _build_squad_suite_from_raw(
                test_suite_cls,
                release_dir=release_dir_path,
            )
            setattr(suite, "_raw_fallback_built", True)
            return suite
        if task == "sentiment" and not enable_raw_fallback:
            raise RuntimeError(
                "Could not load official sentiment suite and raw_fallback is disabled. "
                "To stay fully official, use a Python/dill environment compatible with "
                "sentiment_suite.pkl (typically Python 3.10) or regenerate the suite with "
                "the CheckList repository in your environment. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        raise


def _run_suite_from_prediction_file(suite, pred_file: Path, task: str, qqp_file_format: str = "binary_conf") -> None:
    file_format = None
    if task == "qqp":
        file_format = qqp_file_format
    elif task == "squad":
        file_format = "pred_only"

    suite_test_names = {name for name, _ in _get_suite_tests(suite)}
    suite_range_names = set(getattr(suite, "test_ranges", {}).keys())
    has_range_mismatch = bool(suite_range_names) and suite_test_names != suite_range_names
    force_per_test = task in {"qqp", "squad"}

    if force_per_test or getattr(suite, "_raw_fallback_built", False) or has_range_mismatch:
        if force_per_test:
            print("Using per-test run_from_file mode for task requiring stable file replay.")
        if has_range_mismatch:
            print(
                "Suite test_ranges mismatch detected; switching to per-test run_from_file mode. "
                f"tests={sorted(suite_test_names)}, test_ranges={sorted(suite_range_names)}"
            )
        for test_name, test in _get_suite_tests(suite):
            print(f"Running {test_name} from predictions file")
            test.example_list_and_indices()
            if file_format is None:
                test.run_from_file(str(pred_file), overwrite=True)
            else:
                test.run_from_file(str(pred_file), overwrite=True, file_format=file_format)
        return

    if file_format is None:
        suite.run_from_file(str(pred_file), overwrite=True)
    else:
        suite.run_from_file(str(pred_file), overwrite=True, file_format=file_format)


def main():
    parser = argparse.ArgumentParser(description="Run CheckList with LLM-as-judge on LLaDA/Llama")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--demo", type=int, default=None, metavar="N", help="Demo mode: run only first N examples per test (overrides config checklist.demo_limit)")
    parser.add_argument("--no-demo-limit", action="store_true", help="Ignore any config demo_limit and run the full suite")
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.checklist:
        print("Config has no 'checklist' section. Add checklist.task, suite_path, release_data_dir, judge.")
        sys.exit(1)

    cl = config.checklist
    task = cl.get("task", "sentiment")
    suite_path = cl.get("suite_path")
    judge_spec = cl.get("judge")
    max_new_tokens = cl.get("max_new_tokens", 64)
    batch_size = int(cl.get("batch_size", 1))
    max_examples_per_test = cl.get("max_examples_per_test", None)
    if max_examples_per_test is not None:
        max_examples_per_test = int(max_examples_per_test)
    demo_limit = None if args.no_demo_limit else (args.demo if args.demo is not None else cl.get("demo_limit"))
    use_regex_refusal = cl.get("use_regex_refusal", False)
    use_sentiment_pipeline = cl.get("use_sentiment_pipeline", False)
    sentiment_pipeline_parse_direct_labels = cl.get(
        "sentiment_pipeline_parse_direct_labels", True
    )
    task_prompt_template = cl.get("task_prompt_template")
    judge_prompt_template = cl.get("judge_prompt_template")
    _default_labels = {
        "sentiment": ["negative", "neutral", "positive"],
        "qqp": ["not_duplicate", "duplicate"],
    }
    labels = tuple(cl.get("labels", _default_labels.get(task, ["negative", "neutral", "positive"])))
    fallback_label = cl.get("fallback_label", labels[0] if labels else None)
    squad_repair_max_retries = int(cl.get("squad_repair_max_retries", 1))
    squad_max_answer_chars = int(cl.get("squad_max_answer_chars", 128))
    squad_use_judge_fallback = bool(cl.get("squad_use_judge_fallback", False))
    release_dir = _resolve_release_dir(cl.get("release_data_dir"))
    official_run_from_file = cl.get("official_run_from_file", task in {"qqp", "squad"})

    if not suite_path:
        print("checklist.suite_path is required.")
        sys.exit(1)
    if task != "squad" and not labels:
        print("checklist.labels must contain at least one label.")
        sys.exit(1)
    if task != "squad" and not use_sentiment_pipeline and not use_regex_refusal and not judge_spec:
        print("Either checklist.judge, checklist.use_sentiment_pipeline, or checklist.use_regex_refusal is required.")
        sys.exit(1)
    if use_sentiment_pipeline and task != "sentiment":
        print("Sentiment pipeline scorer only supports checklist.task='sentiment'.")
        sys.exit(1)
    if official_run_from_file and not release_dir:
        print("checklist.release_data_dir is required when official_run_from_file is enabled.")
        sys.exit(1)
    if task == "squad" and not official_run_from_file:
        print("SQuAD: using predictor-wrapper mode (model response = prediction, no judge).")

    # Resolve paths relative to project root
    suite_path = Path(suite_path)
    if not suite_path.is_absolute():
        suite_path = project_root / suite_path
    if not suite_path.exists():
        print(f"Suite not found: {suite_path}. Download release_data (see data/checklist/README.md)")
        sys.exit(1)

    out_dir = Path(args.output_dir or config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CheckList (optional dependency: pip install -e ".[checklist]" or pip install jupyter checklist)
    try:
        from checklist.test_suite import TestSuite
        from checklist.pred_wrapper import PredictorWrapper
    except ImportError as e:
        print("CheckList support requires the 'checklist' package. Install with:")
        print('  pip install -e ".[checklist]"')
        print("If that fails, try (notebook 7+ breaks checklist's setup):")
        print("  pip install 'notebook<7'")
        print("  pip install checklist --no-build-isolation")
        print(f"(Import error: {e})")
        sys.exit(1)

    suite = _load_suite(TestSuite, suite_path, cl)
    print(f"Loaded suite request: {suite_path.name}")

    if demo_limit is not None:
        _limit_suite_to_demo(suite, demo_limit)
        print(f"Demo mode: limited to first {demo_limit} examples per test")
    _print_suite_tests(suite)
    print(f"Execution mode: {'official_run_from_file' if official_run_from_file else 'direct_predictor'}")

    # Scorer: regex refusal, HF sentiment pipeline, or LLM judge
    # For squad in predictor-wrapper mode, judge is None (model response IS the prediction)
    if task == "squad" and not official_run_from_file:
        judge = None
        print("SQuAD predictor-wrapper mode: model response is the prediction (no judge).")
    elif task == "squad":
        judge = None
    elif use_regex_refusal:
        judge = RegexRefusalScorer()
        print("Using regex refusal scorer (no LLM judge).")
    elif use_sentiment_pipeline:
        pipeline_model = cl.get("pipeline_model", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        print(
            "Using HF sentiment pipeline: "
            f"{pipeline_model} "
            f"(parse_direct_labels={sentiment_pipeline_parse_direct_labels})"
        )
        judge = SentimentPipelineScorer(
            model_id=pipeline_model,
            parse_direct_labels=sentiment_pipeline_parse_direct_labels,
        )
    else:
        judge_cfg = _model_config_from_dict(judge_spec)
        print(f"Loading judge: {judge_cfg.name}")
        judge_model = create_model(judge_cfg)
        judge = LLMJudge(
            judge_model,
            task=task,
            prompt_template=judge_prompt_template,
            labels=labels,
            fallback_label=fallback_label,
            max_new_tokens=16,
        )
    print("Scorer ready.")

    all_results = {}

    for model_cfg in config.models:
        suite = _load_suite(TestSuite, suite_path, cl)
        if demo_limit is not None:
            _limit_suite_to_demo(suite, demo_limit)
        _print_suite_tests(suite)
        safe_name = model_cfg.name.replace("/", "_").replace(" ", "_")
        print(f"\n--- Model under test: {model_cfg.name} ---")
        model = create_model(model_cfg)
        run_started_at = datetime.now(timezone.utc).isoformat()
        test_traces = {}

        if official_run_from_file:
            task_inputs = _load_official_task_inputs(task, release_dir=release_dir, limit=demo_limit)
            prompt_template = _task_prompt_template(task, task_prompt_template)
            prompts = _build_task_prompts(task, task_inputs, prompt_template)
            print(f"Generating predictions for {len(prompts)} official {task} inputs")
            _, responses = model.get_responses(
                prompts,
                batched=False,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            raw_responses = list(responses)
            repair_stats = None

            if task == "squad":
                squad_fallback_model = None
                if squad_use_judge_fallback and judge_spec:
                    judge_cfg = _model_config_from_dict(judge_spec)
                    print(f"Loading SQuAD fallback judge: {judge_cfg.name}")
                    squad_fallback_model = create_model(judge_cfg)

                responses, repair_stats, squad_example_traces = _repair_squad_responses(
                    task_inputs=task_inputs,
                    responses=responses,
                    primary_model=model,
                    max_new_tokens=max_new_tokens,
                    max_answer_chars=squad_max_answer_chars,
                    retry_max=squad_repair_max_retries,
                    fallback_model=squad_fallback_model,
                )
                print(
                    "SQuAD normalization stats: "
                    f"deterministic_extraction={repair_stats['deterministic_extraction']}, "
                    f"primary_repaired={repair_stats['primary_repaired']}, "
                    f"fallback_repaired={repair_stats['fallback_repaired']}, "
                    f"sentinel_fallback={repair_stats['sentinel_fallback']}"
                )
                if squad_fallback_model is not None:
                    del squad_fallback_model
                    torch.cuda.empty_cache()
                    gc.collect()

            pred_file = out_dir / f"preds_{safe_name}_{task}.txt"
            if task == "sentiment":
                judge_inputs = [str(x) for x in task_inputs]
            elif task == "qqp":
                judge_inputs = [f"Q1: {q1}\nQ2: {q2}" for q1, q2 in task_inputs]
            else:
                judge_inputs = [f"Passage: {r.get('passage','')}\nQuestion: {r.get('question','')}" for r in task_inputs]

            qqp_file_format = "pred_and_conf" if task == "qqp" and getattr(suite, "_raw_fallback_built", False) else "binary_conf"
            flat_trace_rows = _write_official_predictions(
                task=task,
                pred_path=pred_file,
                responses=responses,
                judge=judge,
                judge_inputs=judge_inputs,
                qqp_file_format=qqp_file_format,
            )
            for idx, row in enumerate(flat_trace_rows):
                squad_trace = squad_example_traces[idx] if task == "squad" else None
                row.update(
                    {
                        "input_text": judge_inputs[idx],
                        "input_structured": _serialize_task_input(task, task_inputs[idx]),
                        "prompt": prompts[idx],
                        "model_response": responses[idx],
                        "raw_response_before_repair": raw_responses[idx],
                        "extraction_trace": (
                            squad_trace.get("extraction_trace") if squad_trace else None
                        ),
                        "normalization": {
                            "response_changed": raw_responses[idx] != responses[idx],
                            "final_mode": squad_trace.get("final_mode") if squad_trace else None,
                            "primary_repair_output": (
                                squad_trace.get("primary_repair_output") if squad_trace else None
                            ),
                            "primary_repair_trace": (
                                squad_trace.get("primary_repair_trace") if squad_trace else None
                            ),
                            "fallback_repair_output": (
                                squad_trace.get("fallback_repair_output") if squad_trace else None
                            ),
                            "fallback_repair_trace": (
                                squad_trace.get("fallback_repair_trace") if squad_trace else None
                            ),
                            "repair_stats": repair_stats,
                        },
                    }
                )
            test_traces = _build_test_trace_map(suite, flat_trace_rows)
            print(f"Wrote official predictions: {pred_file}")

            _validate_official_prediction_file(
                task=task,
                pred_path=pred_file,
                expected_lines=len(task_inputs),
                qqp_file_format=qqp_file_format,
            )
            print("Prediction file format validation passed.")

            _run_suite_from_prediction_file(
                suite=suite,
                pred_file=pred_file,
                task=task,
                qqp_file_format=qqp_file_format,
            )
        else:
            adapter = LMPredictorForCheckList(
                model,
                judge,
                task=task,
                task_prompt_template=task_prompt_template,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                max_examples_per_test=max_examples_per_test,
            )
            if task == "squad":
                wrapped = adapter.predict_squad
            else:
                wrapped = PredictorWrapper.wrap_softmax(adapter.predict_proba)
            adapter.reset_trace_log()
            judge_stats_accum = {
                "total_examples": 0,
                "direct_parse_count": 0,
                "judge_fallback_count": 0,
                "judge_repair_count": 0,
            }
            for test_name, test in _get_suite_tests(suite):
                n_cases = _test_case_count(test)
                print(f"Running {test_name} ({n_cases} cases)")
                test.run(wrapped, verbose=True, overwrite=True)
                if adapter.trace_log:
                    test_traces[test_name] = adapter.trace_log.pop(0)
                if adapter.stats_log:
                    batch_stats = adapter.stats_log.pop(0) or {}
                    for key in judge_stats_accum:
                        judge_stats_accum[key] += int(batch_stats.get(key, 0) or 0)
            if judge_stats_accum["total_examples"]:
                print(
                    "Judge fallback stats: "
                    f"direct={judge_stats_accum['direct_parse_count']}, "
                    f"fallbacks={judge_stats_accum['judge_fallback_count']}, "
                    f"repairs={judge_stats_accum['judge_repair_count']}"
                )
        suite.summary(n=2)

        results = _extract_suite_results(suite, example_traces=test_traces)
        results["model"] = model_cfg.name
        results["task"] = task
        results["labels"] = list(labels)
        if task == "squad" and not official_run_from_file:
            results["judge_type"] = "direct_generation"
        elif use_sentiment_pipeline:
            results["judge_type"] = "sentiment_pipeline"
        else:
            results["judge_type"] = "llm_judge"
        if not official_run_from_file and not use_sentiment_pipeline and task != "squad":
            results["judge_fallback_stats"] = judge_stats_accum
        run_metadata = {
            "run_name": safe_name,
            "model": model_cfg.name,
            "task": task,
            "labels": list(labels),
            "judge_type": results["judge_type"],
            "official_run_from_file": official_run_from_file,
            "config_path": args.config,
            "output_dir": str(out_dir),
            "suite_path": str(suite_path),
            "release_data_dir": str(release_dir) if release_dir else None,
            "demo_limit": demo_limit,
            "max_new_tokens": max_new_tokens,
            "task_prompt_template": task_prompt_template,
            "judge_prompt_template": judge_prompt_template,
            "fallback_label": fallback_label,
            "sentiment_pipeline_parse_direct_labels": (
                sentiment_pipeline_parse_direct_labels if use_sentiment_pipeline else None
            ),
            "judge_fallback_stats": (
                judge_stats_accum if not official_run_from_file and not use_sentiment_pipeline and task != "squad" else None
            ),
            "raw_fallback_built": bool(getattr(suite, "_raw_fallback_built", False)),
            "started_at_utc": run_started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "squad_repair_max_retries": squad_repair_max_retries if task == "squad" else None,
            "squad_max_answer_chars": squad_max_answer_chars if task == "squad" else None,
            "squad_use_judge_fallback": squad_use_judge_fallback if task == "squad" else None,
        }
        results["run_metadata"] = run_metadata
        all_results[safe_name] = results

        out_file = out_dir / f"checklist_results_{safe_name}.json"
        _write_json(out_file, results)
        _write_detailed_result_bundle(out_dir, safe_name, results, run_metadata)
        print(f"Wrote {out_file}")

        # Cleanup: free GPU memory before loading next model
        del model
        if not official_run_from_file:
            del adapter
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Cleaned up GPU memory after {model_cfg.name}")

    # Summary comparison (must use _json_safe: preds/pass flags may be NumPy scalars)
    summary_path = out_dir / "checklist_summary.json"
    _write_json(summary_path, all_results)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
