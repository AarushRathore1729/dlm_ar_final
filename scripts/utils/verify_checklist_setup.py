#!/usr/bin/env python3
"""
Comprehensive verification of CheckList setup:
- Config file consistency
- Suite files loadable
- Prompt templates vs input keys match
- Labels are correct types
"""

import json
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from checklist.test_suite import TestSuite


def verify_configs():
    """Check all GPU configs have valid suite paths and settings."""
    print("=" * 70)
    print("VERIFYING CONFIG FILES")
    print("=" * 70)

    configs = list(project_root.glob("configs/checklist/phase2_checklist_*_gpu_*.yaml"))
    if not configs:
        print("ERROR: No phase2_checklist_*_gpu_*.yaml configs found!")
        return False

    all_valid = True
    for cfg_path in sorted(configs):
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)

        name = cfg_path.stem
        task = cfg.get("checklist", {}).get("task")
        suite_path = cfg.get("checklist", {}).get("suite_path")
        demo_limit = cfg.get("checklist", {}).get("demo_limit")
        official_run = cfg.get("checklist", {}).get("official_run_from_file")

        suite_full = project_root / suite_path if suite_path else None
        suite_exists = suite_full and suite_full.exists()

        status = "✓" if suite_exists else "✗"
        print(f"\n{status} {name}")
        print(f"  task: {task}, demo_limit: {demo_limit}, official_run_from_file: {official_run}")
        print(f"  suite: {suite_path}")
        print(f"  exists: {suite_exists}")

        if not suite_exists:
            all_valid = False

    return all_valid


def verify_suites():
    """Load and inspect all suite files."""
    print("\n" + "=" * 70)
    print("VERIFYING SUITE FILES")
    print("=" * 70)

    suite_files = [
        "data/checklist/release_data/sentiment/sentiment_suite_py310.pkl",
        "data/checklist/release_data/qqp/qqp_suite_py310.pkl",
        "data/checklist/release_data/squad/squad_suite_py310.pkl",
    ]

    all_valid = True
    for suite_rel in suite_files:
        suite_path = project_root / suite_rel
        if not suite_path.exists():
            print(f"\n✗ {suite_rel} NOT FOUND")
            all_valid = False
            continue

        try:
            suite = TestSuite.from_file(str(suite_path))
            tests = list(suite.tests.keys())
            test = list(suite.tests.values())[0] if suite.tests else None

            n_examples = len(test.data) if test and test.data else 0
            label_sample = test.labels[0] if test and test.labels else None
            label_type = type(label_sample).__name__ if label_sample is not None else "None"

            print(f"\n✓ {suite_rel}")
            print(f"  tests: {tests}")
            print(f"  examples: {n_examples}")
            print(f"  label type: {label_type}")
            if isinstance(label_sample, (int, float)):
                print(f"  label samples: {test.labels[:3]}")
            else:
                print(f"  WARNING: label is {type(label_sample)}, expected int/float")
        except Exception as e:
            print(f"\n✗ {suite_rel}: {type(e).__name__}: {str(e)[:100]}")
            all_valid = False

    return all_valid


def verify_adapters():
    """Check adapter supports all input formats."""
    print("\n" + "=" * 70)
    print("VERIFYING CHECKLIST ADAPTER INPUT FORMATS")
    print("=" * 70)

    from dlm_safety.checklist_adapter import LMPredictorForCheckList

    test_cases = {
        "sentiment": {
            "inputs": ["great movie", "not great"],
            "template": "Classify: {text}",
        },
        "qqp": {
            "inputs": [
                {"q1": "What is AI?", "q2": "What is artificial intelligence?"},
                {"question1": "How old?", "question2": "What is your age?"},
            ],
            "template": "Q1: {question1}\nQ2: {question2}",
        },
        "squad": {
            "inputs": [
                {"passage": "Paris is in France.", "question": "Where is Paris?"},
            ],
            "template": "Passage: {passage}\nQ: {question}",
        },
    }

    all_valid = True
    for task, test in test_cases.items():
        print(f"\n{task}:")
        adapter = LMPredictorForCheckList.__new__(LMPredictorForCheckList)
        adapter.task = task
        adapter.task_prompt_template = test["template"]

        for idx, inp in enumerate(test["inputs"]):
            try:
                prompt = adapter._format_prompt(inp)
                print(f"  ✓ input {idx}: formatted successfully")
            except Exception as e:
                print(f"  ✗ input {idx}: {type(e).__name__}: {str(e)[:80]}")
                all_valid = False

    return all_valid


def verify_prompt_templates():
    """Check config prompt templates match expected task keys."""
    print("\n" + "=" * 70)
    print("VERIFYING PROMPT TEMPLATES VS TASK KEYS")
    print("=" * 70)

    required_keys = {
        "sentiment": {"text"},
        "qqp": {"question1", "question2"},
        "squad": {"passage", "question"},
    }

    configs = {
        "sentiment": [
            "configs/checklist/phase2_checklist_sentiment_rerun_llama.yaml",
            "configs/checklist/phase2_checklist_sentiment_rerun_llada.yaml",
        ],
        "qqp": [
            "configs/checklist/phase2_checklist_qqp_gpu_llama_only.yaml",
            "configs/checklist/phase2_checklist_qqp_gpu_llada_only.yaml",
        ],
        "squad": [
            "configs/checklist/phase2_checklist_squad_gpu_llama_only.yaml",
            "configs/checklist/phase2_checklist_squad_gpu_llada_only.yaml",
        ],
    }

    all_valid = True
    for task, required in required_keys.items():
        print(f"\n{task.upper()} (requires {required}):")
        for cfg_rel in configs[task]:
            cfg_path = project_root / cfg_rel
            if not cfg_path.exists():
                print(f"  ✗ {cfg_rel} NOT FOUND")
                all_valid = False
                continue

            with cfg_path.open() as f:
                cfg = yaml.safe_load(f)

            tpl = cfg.get("checklist", {}).get("task_prompt_template", "")
            import string
            template_keys = set(
                k for _, k, _, _ in string.Formatter().parse(tpl)
                if k is not None
            )

            if required <= template_keys:
                print(f"  ✓ {cfg_rel}: {template_keys}")
            else:
                missing = required - template_keys
                print(f"  ✗ {cfg_rel}: missing keys {missing}, has {template_keys}")
                all_valid = False

    return all_valid


def main():
    results = {
        "configs": verify_configs(),
        "suites": verify_suites(),
        "adapter_formats": verify_adapters(),
        "prompt_templates": verify_prompt_templates(),
    }

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 70)
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    all_pass = all(results.values())
    print("\n" + ("✓ ALL CHECKS PASSED - Ready to scale up!" if all_pass else "✗ Some checks failed"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
