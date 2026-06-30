#!/usr/bin/env python3
"""Load and inspect CheckList suites using the official Suite API.

This utility focuses on SQuAD release data in this repository and supports:
1) loading an existing suite pickle via ``TestSuite.from_file``
2) rebuilding a SQuAD suite from raw jsonl + reference answers via ``TestSuite`` + ``MFT``
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlm_safety.checklist_visual_compat import attach_visual_api


def build_suite_from_raw(release_dir: Path, dataset: str = "squad"):
    """Build a suite from raw data for sentiment, squad, or qqp."""
    from checklist.test_suite import TestSuite
    from checklist.test_types import MFT

    suite = TestSuite()
    
    if dataset == "squad":
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

        suite.add(
            MFT(
                data=records,
                labels=answers,
                name="squad",
                capability="squad",
                description="SQuAD passage-question pairs with pseudo labels from BERT reference answers.",
            ),
            overwrite=True,
        )

    elif dataset == "sentiment":
        data_path = release_dir / "sentiment" / "tests_n500"
        pred_path = release_dir / "sentiment" / "predictions" / "bert"

        if not data_path.exists():
            raise FileNotFoundError(f"Raw sentiment tests not found: {data_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"Sentiment reference predictions not found: {pred_path}")

        # Load sentiment test data (newline-separated text)
        records = []
        with data_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(line)

        # Load predictions from bert file (format: class_id conf0 conf1 conf2)
        # Extract just the class ID (first number)
        # BERT outputs 0=negative, 1=neutral, 2=positive (already 0-indexed)
        labels = []
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        labels.append(int(parts[0]))  # Already 0-indexed from BERT

        suite.add(
            MFT(
                data=records,
                labels=labels,
                name="sentiment",
                capability="sentiment",
                description="Sentiment test cases with reference labels.",
            ),
            overwrite=True,
        )

    elif dataset == "qqp":
        data_path = release_dir / "qqp" / "tests_n500"
        pred_path = release_dir / "qqp" / "predictions" / "bert"

        if not data_path.exists():
            raise FileNotFoundError(f"Raw QQP tests not found: {data_path}")
        if not pred_path.exists():
            raise FileNotFoundError(f"QQP reference predictions not found: {pred_path}")

        # Load QQP test data (TSV with header; columns include id, question1, question2)
        records = []
        with data_path.open(encoding="utf-8") as f:
            _header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    records.append({"q1": parts[1], "q2": parts[2]})

        # Load reference QQP probabilities and convert to binary labels
        probs = []
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    probs.append(float(line))

        n = min(len(records), len(probs))
        records = records[:n]
        labels = [1 if p >= 0.5 else 0 for p in probs[:n]]

        suite.add(
            MFT(
                data=records,
                labels=labels,
                name="qqp",
                capability="qqp",
                description="QQP (Quora Question Pairs) test cases with reference labels.",
            ),
            overwrite=True,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: sentiment, squad, qqp")

    return suite


def build_squad_suite_from_raw(release_dir: Path):
    """Backward compatibility wrapper."""
    return build_suite_from_raw(release_dir, dataset="squad")


def load_suite_from_pickle(suite_path: Path):
    from checklist.test_suite import TestSuite

    return TestSuite.from_file(str(suite_path))


def iter_suite_tests(suite):
    if hasattr(suite, "tests") and suite.tests:
        for name, test in suite.tests.items():
            yield name, test
        return
    for test in getattr(suite, "test_list", []):
        yield getattr(test, "name", str(test)), test


def preview_suite(suite, limit: int):
    print("Suite tests:")
    for test_name, test in iter_suite_tests(suite):
        data = getattr(test, "data", []) or []
        labels = getattr(test, "labels", None)
        print(f"- {test_name}: {len(data)} cases")
        for idx, item in enumerate(data[:limit]):
            print(f"  Example {idx + 1}:")
            if isinstance(item, dict):
                passage = item.get("passage", "").replace("\n", " ")
                question = item.get("question", "")
                print(f"    passage: {passage[:220]}{'...' if len(passage) > 220 else ''}")
                print(f"    question: {question}")
            else:
                print(f"    input: {str(item)[:220]}")

            if labels is not None:
                if isinstance(labels, list):
                    label = labels[idx] if idx < len(labels) else None
                else:
                    label = labels
                print(f"    label: {label}")


def export_suite_to_json(suite, output_path: Path):
    payload = {"tests": []}
    for test_name, test in iter_suite_tests(suite):
        data = getattr(test, "data", []) or []
        labels = getattr(test, "labels", None)
        serialized = []
        for idx, item in enumerate(data):
            row = {"input": item}
            if labels is not None:
                if isinstance(labels, list):
                    row["label"] = labels[idx] if idx < len(labels) else None
                else:
                    row["label"] = labels
            serialized.append(row)

        payload["tests"].append(
            {
                "name": test_name,
                "capability": getattr(test, "capability", None),
                "description": getattr(test, "description", None),
                "size": len(data),
                "examples": serialized,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Exported suite JSON to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Use CheckList Suite API with CheckList data.")
    parser.add_argument(
        "--suite-path",
        type=Path,
        default=None,
        help="Path to suite pickle (auto-inferred from dataset if not provided)",
    )
    parser.add_argument(
        "--dataset",
        choices=["sentiment", "squad", "qqp"],
        default="squad",
        help="Dataset to load: sentiment, squad, or qqp (default: squad)",
    )
    parser.add_argument(
        "--release-data-dir",
        type=Path,
        default=Path("data/checklist/release_data"),
        help="Path to CheckList release_data root",
    )
    parser.add_argument(
        "--source",
        choices=["pickle", "raw", "auto"],
        default="auto",
        help="Source for constructing suite",
    )
    parser.add_argument("--preview", type=int, default=3, help="Number of examples to preview per test")
    parser.add_argument("--export-json", type=Path, default=None, help="Optional path to export full suite content")
    parser.add_argument("--save-suite", type=Path, default=None, help="Optional path to save suite pickle in current env")
    parser.add_argument(
        "--visual-suite",
        action="store_true",
        help="Call visualization compatibility API: suite.visual.suite()",
    )
    parser.add_argument(
        "--visual-test",
        type=str,
        default=None,
        help="Call visualization compatibility API: suite.visual.test(<test_name>)",
    )
    args = parser.parse_args()

    # Auto-infer suite path if not provided
    if args.suite_path is None:
        args.suite_path = args.release_data_dir / args.dataset / f"{args.dataset}_suite.pkl"

    if args.source == "pickle":
        suite = load_suite_from_pickle(args.suite_path)
        source_used = "pickle"
    elif args.source == "raw":
        suite = build_suite_from_raw(args.release_data_dir, dataset=args.dataset)
        source_used = "raw"
    else:
        try:
            suite = load_suite_from_pickle(args.suite_path)
            source_used = "pickle"
        except Exception as exc:
            print(f"Pickle load failed ({type(exc).__name__}: {exc}). Falling back to raw data.")
            suite = build_suite_from_raw(args.release_data_dir, dataset=args.dataset)
            source_used = "raw"

    print(f"Loaded suite from: {source_used}")
    suite = attach_visual_api(suite)
    preview_suite(suite, args.preview)

    if args.export_json is not None:
        export_suite_to_json(suite, args.export_json)

    if args.save_suite is not None:
        args.save_suite.parent.mkdir(parents=True, exist_ok=True)
        had_visual = hasattr(suite, "visual")
        visual_obj = getattr(suite, "visual", None)
        if had_visual:
            delattr(suite, "visual")
        suite.save(str(args.save_suite))
        if had_visual:
            setattr(suite, "visual", visual_obj)
        print(f"Saved suite pickle to: {args.save_suite}")

    if args.visual_suite:
        print("Calling compatibility visualization API: suite.visual.suite()")
        suite.visual.suite()

    if args.visual_test:
        print(f"Calling compatibility visualization API: suite.visual.test({args.visual_test!r})")
        suite.visual.test(args.visual_test)


if __name__ == "__main__":
    main()
