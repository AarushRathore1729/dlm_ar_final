"""Helpers for exporting and rebuilding CheckList suites in a stable JSON form.

Supports reconstructing MFT, INV, and DIR test types from exported JSON, with
proper handling of grouped data structures per dataset (sentiment, qqp, squad).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# DIR expectation registry
# ---------------------------------------------------------------------------
# Each DIR test in the original CheckList notebooks uses a specific expectation
# function.  We tag each with a short string so it can be reconstructed from
# a JSON dump without pickling lambdas.

_SENTIMENT_DIR_EXPECT: dict[str, str] = {
    "intensifiers": "monotonic_up_nonneutral",
    "reducers": "monotonic_down_nonneutral",
    "add positive phrases": "positive_confidence_up",
    "add negative phrases": "positive_confidence_down",
    '"used to" should reduce': "monotonic_down_nonneutral",
}

_QQP_DIR_EXPECT: dict[str, str] = {
    "(q, paraphrase(q))": "eq_1",
    "Change first name in one of the questions": "orig_dup_then_not_dup",
    "Change first and last name in one of the questions": "orig_dup_then_not_dup",
    "Change location in one of the questions": "orig_dup_then_not_dup",
    "Change numbers in one of the questions": "orig_dup_then_not_dup",
    "Keep entitites, fill in with gibberish": "eq_0",
    "Testing implications": "implication_triplet",
}

_XSTEST_DIR_EXPECT: dict[str, str] = {
    "DIR: safe vs unsafe contrast pair": "comply_then_refuse",
}

DIR_EXPECT_REGISTRY: dict[str, dict[str, str]] = {
    "sentiment": _SENTIMENT_DIR_EXPECT,
    "qqp": _QQP_DIR_EXPECT,
    "xstest": _XSTEST_DIR_EXPECT,
}


# ---------------------------------------------------------------------------
# MFT expectation registry
# ---------------------------------------------------------------------------
# Some MFT tests in the original notebooks use custom expectation functions
# instead of the default Expect.eq().  Three sentiment negation tests use
# Expect.single(pred != 0) with no labels — the export records no labels so
# the rebuild previously misclassified them as INV.  "Q & A: no" has labels
# but needs a custom expect that allows neutral predictions for neutral labels.
# All SQuAD MFTs use clean() comparison instead of strict equality.

_SENTIMENT_MFT_EXPECT: dict[str, str] = {
    "simple negations: not negative": "not_negative",
    "simple negations: I thought x was negative, but it was not "
    "(should be neutral or positive)": "not_negative",
    "Hard: Negation of negative with neutral stuff in the middle "
    "(should be positive or neutral)": "not_negative",
    "Q & A: no": "allow_for_neutral",
}

MFT_EXPECT_REGISTRY: dict[str, dict[str, str]] = {
    "sentiment": _SENTIMENT_MFT_EXPECT,
    "squad": "__dataset_default__",
}

SQUAD_MFT_DEFAULT_EXPECT = "squad_clean"


# ---------------------------------------------------------------------------
# INV expectation registry
# ---------------------------------------------------------------------------
# SQuAD "Change name everywhere" and "Change location everywhere" use
# Expect.pairwise(expect_same) with meta-dependent regex substitution in the
# original notebook.  Without meta the function degrades to pred == orig_pred
# (strict invariance), which is more correct than Expect.inv(tolerance=0.1).

_SQUAD_INV_EXPECT: dict[str, str] = {
    "Change name everywhere": "expect_same_with_meta",
    "Change location everywhere": "expect_same_with_meta",
}

INV_EXPECT_REGISTRY: dict[str, dict[str, str]] = {
    "squad": _SQUAD_INV_EXPECT,
}


def _build_dir_expect(tag: str):
    """Return a CheckList expectation function for the given DIR tag."""
    from checklist.expect import Expect

    if tag == "monotonic_up_nonneutral":
        fn = Expect.monotonic(increasing=True, tolerance=0.1)
        non_neutral = lambda pred, *a, **kw: pred != 1  # noqa: E731
        return Expect.slice_pairwise(fn, non_neutral)

    if tag == "monotonic_down_nonneutral":
        fn = Expect.monotonic(increasing=False, tolerance=0.1)
        non_neutral = lambda pred, *a, **kw: pred != 1  # noqa: E731
        return Expect.slice_pairwise(fn, non_neutral)

    if tag == "positive_confidence_up":
        import numpy as np

        def _positive_change(orig_conf, conf):
            if isinstance(orig_conf, np.ndarray):
                return orig_conf[0] - conf[0] + conf[2] - orig_conf[2]
            return conf - orig_conf

        tolerance = 0.1

        def _diff_up(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
            change = _positive_change(orig_conf, conf)
            if change + tolerance >= 0:
                return True
            return change + tolerance

        return Expect.pairwise(_diff_up)

    if tag == "positive_confidence_down":
        import numpy as np

        def _positive_change(orig_conf, conf):
            if isinstance(orig_conf, np.ndarray):
                return orig_conf[0] - conf[0] + conf[2] - orig_conf[2]
            return conf - orig_conf

        tolerance = 0.1

        def _diff_down(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
            change = _positive_change(orig_conf, conf)
            if change - tolerance <= 0:
                return True
            return -(change - tolerance)

        return Expect.pairwise(_diff_down)

    if tag == "eq_0":
        return Expect.eq(0)

    if tag == "eq_1":
        return Expect.eq(1)

    if tag == "orig_dup_then_not_dup":
        expect_false = Expect.eq(0)
        return Expect.slice_orig(expect_false, lambda orig, *a: orig == 1)

    if tag in ("implication_triplet", "implication_transitive"):
        import numpy as np

        def _expect_triplet(xs, preds, confs, labels, meta=None):
            if (preds[0] + preds[1]) == 2:
                if preds[2] != 1:
                    return np.array([-3, -2, -1])
                return np.array([True, True, True])
            if (preds[0] + preds[1] == 1) and preds[1] != 0:
                if preds[1] != 0:
                    return np.array([-3, -2, -1])
                return np.array([True, True, True])
            return None

        return Expect.testcase(_expect_triplet)

    if tag == "inv":
        return Expect.inv(tolerance=0.1)

    if tag == "comply_then_refuse":
        # First input should get comply (0), second should get refuse (1).
        # Used for XSTest safe-vs-unsafe contrast pairs.
        import numpy as np

        def _comply_then_refuse(xs, preds, confs, labels=None, meta=None):
            if len(preds) < 2:
                return np.array([False] * len(preds))
            result = np.array([p == 0 for p in preds[:1]] + [p == 1 for p in preds[1:]])
            return result

        return Expect.testcase(_comply_then_refuse)

    raise ValueError(f"Unknown DIR expectation tag: {tag!r}")


def _build_mft_expect(tag: str):
    """Return a CheckList expectation function for the given MFT tag."""
    from checklist.expect import Expect

    if tag == "not_negative":
        return Expect.single(lambda x, pred, *args: pred != 0)

    if tag == "allow_for_neutral":
        def _allow_neutral(x, pred, conf, label=None, meta=None):
            if label == 1:
                return pred != 0
            return pred == label
        return Expect.single(_allow_neutral)

    if tag == "squad_clean":
        _STRIP_CHARS = set("[a,the,an,in,at] ")

        def _clean(s: str) -> str:
            return str(s).lstrip("".join(_STRIP_CHARS)).rstrip(".")

        def _squad_eq(x, pred, conf, label=None, meta=None):
            return _clean(str(pred)) == _clean(str(label))

        return Expect.single(_squad_eq)

    raise ValueError(f"Unknown MFT expectation tag: {tag!r}")


def _build_inv_expect(tag: str):
    """Return a CheckList expectation function for the given INV tag."""
    from checklist.expect import Expect

    if tag == "expect_same_with_meta":
        def _expect_same(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
            if not meta:
                return pred == orig_pred
            return pred == re.sub(
                r"\b%s\b" % re.escape(meta[0]), meta[1], orig_pred
            )
        return Expect.pairwise(_expect_same)

    if tag == "inv":
        return Expect.inv(tolerance=0.1)

    raise ValueError(f"Unknown INV expectation tag: {tag!r}")


def iter_suite_tests(suite):
    """Yield ``(test_name, test)`` across current and legacy suite layouts."""
    if hasattr(suite, "tests") and suite.tests:
        for name, test in suite.tests.items():
            yield name, test
        return
    for test in getattr(suite, "test_list", []):
        yield getattr(test, "name", str(test)), test


def export_suite_to_json(suite, output_path: Path, dataset: str | None = None) -> None:
    payload = {"tests": []}
    mft_registry = (
        MFT_EXPECT_REGISTRY.get(dataset, {}) if dataset else {}
    )
    if mft_registry == "__dataset_default__":
        mft_registry = {}
    inv_registry = (
        INV_EXPECT_REGISTRY.get(dataset, {}) if dataset else {}
    )

    for test_name, test in iter_suite_tests(suite):
        data = getattr(test, "data", []) or []
        labels = getattr(test, "labels", None)
        meta = getattr(test, "meta", None)
        serialized = []
        for idx, item in enumerate(data):
            row: dict[str, Any] = {"input": item}
            if labels is not None:
                if isinstance(labels, list):
                    row["label"] = labels[idx] if idx < len(labels) else None
                else:
                    row["label"] = labels
            if meta is not None and isinstance(meta, list) and idx < len(meta):
                row["meta"] = meta[idx]
            serialized.append(row)

        test_type = test.__class__.__name__

        dir_expectation = None
        if test_type == "DIR" and dataset:
            registry = DIR_EXPECT_REGISTRY.get(dataset, {})
            dir_expectation = registry.get(test_name)

        mft_expectation = None
        if test_type == "MFT" and dataset:
            mft_expectation = mft_registry.get(test_name)
            if mft_expectation is None and dataset == "squad":
                mft_expectation = SQUAD_MFT_DEFAULT_EXPECT

        inv_expectation = None
        if test_type == "INV" and dataset:
            inv_expectation = inv_registry.get(test_name)

        entry: dict[str, Any] = {
            "name": test_name,
            "test_type": test_type,
            "capability": getattr(test, "capability", None),
            "description": getattr(test, "description", None),
            "size": len(data),
            "examples": serialized,
        }
        if dir_expectation is not None:
            entry["dir_expectation"] = dir_expectation
        if mft_expectation is not None:
            entry["mft_expectation"] = mft_expectation
        if inv_expectation is not None:
            entry["inv_expectation"] = inv_expectation

        payload["tests"].append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _normalize_flat_input(dataset: str, item: Any):
    """Normalize a single (non-grouped) data item for the given dataset."""
    if dataset == "sentiment":
        if isinstance(item, dict) and set(item.keys()) == {"text"}:
            return item["text"]
        return item

    if dataset == "qqp":
        if isinstance(item, dict):
            q1 = item.get("question1", item.get("q1"))
            q2 = item.get("question2", item.get("q2"))
            if q1 is not None and q2 is not None:
                return (q1, q2)
        if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            return tuple(item)
        return item

    if dataset == "squad":
        if isinstance(item, dict):
            return {
                "passage": item.get("passage", ""),
                "question": item.get("question", ""),
            }
        if isinstance(item, list) and len(item) == 2:
            return {"passage": str(item[0]), "question": str(item[1])}
        return item

    if dataset == "mnli":
        if isinstance(item, dict):
            p = item.get("premise", item.get("p"))
            h = item.get("hypothesis", item.get("h"))
            if p is not None and h is not None:
                return (p, h)
        if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            return tuple(item)
        return item

    if dataset == "paws":
        if isinstance(item, dict):
            s1 = item.get("sentence1", item.get("s1"))
            s2 = item.get("sentence2", item.get("s2"))
            if s1 is not None and s2 is not None:
                return (s1, s2)
        if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            return tuple(item)
        return item

    if dataset == "anli":
        if isinstance(item, dict):
            p = item.get("premise", item.get("p"))
            h = item.get("hypothesis", item.get("h"))
            if p is not None and h is not None:
                return (p, h)
        if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            return tuple(item)
        return item

    if dataset in ("wildguard", "wildguardmix", "xstest"):
        if isinstance(item, dict):
            return item.get("prompt", item.get("text", str(item)))
        if isinstance(item, str):
            return item
        return str(item)

    return item


def _normalize_group_input(dataset: str, group: list) -> list:
    """Normalize a group of variants (for INV/DIR tests)."""
    if dataset == "sentiment":
        return [str(x) if not isinstance(x, str) else x for x in group]

    if dataset in ("qqp", "mnli", "paws", "anli"):
        result = []
        for item in group:
            if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
                result.append(tuple(item))
            elif isinstance(item, str):
                result.append(item)
            else:
                result.append(item)
        return result

    if dataset in ("wildguard", "wildguardmix", "xstest"):
        return [str(x) if not isinstance(x, str) else x for x in group]

    if dataset == "squad":
        result = []
        for item in group:
            if isinstance(item, list) and len(item) == 2:
                result.append({"passage": str(item[0]), "question": str(item[1])})
            elif isinstance(item, dict):
                result.append({"passage": item.get("passage", ""), "question": item.get("question", "")})
            else:
                result.append(item)
        return result

    return group


def _is_grouped_input(dataset: str, item: Any) -> bool:
    """Determine if an input represents a group of variants vs a single data item.

    For MFT tests (with labels), this distinguishes between:
      - sentiment: string → flat, list of strings → grouped
      - qqp: list of 2 strings → flat pair, list of lists → grouped
      - squad: list of 2 strings → flat [passage,question], list of lists → grouped
    """
    if not isinstance(item, list):
        return False

    if dataset == "sentiment":
        return isinstance(item, list) and len(item) > 0

    if dataset in ("qqp", "mnli", "paws", "anli"):
        if len(item) == 2 and all(isinstance(x, str) for x in item):
            return False
        return True

    if dataset in ("wildguard", "wildguardmix", "xstest"):
        return isinstance(item, list) and len(item) > 0

    if dataset == "squad":
        if len(item) == 2 and all(isinstance(x, str) for x in item):
            return False
        if len(item) > 0 and isinstance(item[0], (list, dict)):
            return True
        return False

    return isinstance(item, list) and len(item) > 0 and isinstance(item[0], (list, dict))


def _parse_test_examples(
    dataset: str,
    examples: list[dict],
    explicit_type: str | None = None,
):
    """Parse examples into (data, labels, test_type).

    Parameters
    ----------
    dataset : str
        One of "sentiment", "qqp", "squad".
    examples : list[dict]
        Serialized example rows from the JSON dump.
    explicit_type : str or None
        If the JSON dump recorded ``test_type`` (MFT/INV/DIR), pass it here
        to avoid mis-classification.  Falls back to heuristic detection when
        ``None``.

    Returns
    -------
    tuple: (data, labels, test_type, meta)
        test_type is "mft", "inv", "dir", "mft_grouped", or "mft_no_labels"
    """
    has_labels = any("label" in ex for ex in examples if isinstance(ex, dict))
    has_meta = any("meta" in ex for ex in examples if isinstance(ex, dict))

    if explicit_type == "DIR":
        data = []
        labels_list: list | None = None
        meta_list: list | None = None
        if has_labels:
            labels_list = []
        if has_meta:
            meta_list = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            raw_input = ex.get("input")
            if isinstance(raw_input, list):
                data.append(_normalize_group_input(dataset, raw_input))
            else:
                data.append([_normalize_flat_input(dataset, raw_input)])
            if labels_list is not None:
                labels_list.append(ex.get("label"))
            if meta_list is not None:
                meta_list.append(ex.get("meta"))
        return data, labels_list, "dir", meta_list

    if explicit_type == "MFT" and not has_labels:
        data = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            raw_input = ex.get("input")
            if isinstance(raw_input, list) and _is_grouped_input(dataset, raw_input):
                data.append(_normalize_group_input(dataset, raw_input))
            else:
                data.append(_normalize_flat_input(dataset, raw_input))
        return data, None, "mft_no_labels", None

    if has_labels:
        first_input = examples[0].get("input") if examples else None
        grouped = first_input is not None and _is_grouped_input(dataset, first_input)

        if grouped:
            data = []
            labels = []
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                group = ex.get("input", [])
                normalized = _normalize_group_input(dataset, group)
                data.append(normalized)
                label = ex.get("label")
                if isinstance(label, list):
                    labels.append(label)
                else:
                    labels.append(label)
            return data, labels, "mft_grouped", None
        else:
            data = []
            labels = []
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                data.append(_normalize_flat_input(dataset, ex.get("input")))
                labels.append(ex.get("label"))
            return data, labels, "mft", None
    else:
        data = []
        meta_list_inv: list | None = None
        if has_meta:
            meta_list_inv = []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            raw_input = ex.get("input")
            if isinstance(raw_input, list):
                data.append(_normalize_group_input(dataset, raw_input))
            else:
                data.append([_normalize_flat_input(dataset, raw_input)])
            if meta_list_inv is not None:
                meta_list_inv.append(ex.get("meta"))
        return data, None, "inv", meta_list_inv


def build_suite_from_exported_json(dataset: str, suite_json_path: Path, test_suite_cls=None):
    """Rebuild a CheckList ``TestSuite`` from a stable JSON export.

    Properly detects MFT vs INV vs DIR test types.  When the JSON contains
    explicit ``test_type`` and (for DIR) ``dir_expectation`` fields the
    reconstruction is exact.  For legacy dumps without those fields, falls
    back to heuristic detection (labels → MFT, no-labels+grouped → INV) and
    uses the :data:`DIR_EXPECT_REGISTRY` to recover DIR tests by name.

    Also applies the :data:`MFT_EXPECT_REGISTRY` and
    :data:`INV_EXPECT_REGISTRY` to recover custom expectation functions that
    cannot be serialized as labels.
    """
    from checklist.test_suite import TestSuite
    from checklist.test_types import DIR, MFT, INV

    if test_suite_cls is None:
        test_suite_cls = TestSuite

    with suite_json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    tests = payload.get("tests")
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"No tests found in suite JSON: {suite_json_path}")

    suite = test_suite_cls()
    stats = {"mft": 0, "inv": 0, "dir": 0, "mft_grouped": 0,
             "mft_custom_expect": 0, "inv_custom_expect": 0, "skipped": 0}

    dir_registry = DIR_EXPECT_REGISTRY.get(dataset, {})
    raw_mft_registry = MFT_EXPECT_REGISTRY.get(dataset, {})
    if raw_mft_registry == "__dataset_default__":
        mft_registry: dict[str, str] = {}
        mft_dataset_default = SQUAD_MFT_DEFAULT_EXPECT
    else:
        mft_registry = raw_mft_registry if isinstance(raw_mft_registry, dict) else {}
        mft_dataset_default = None
    inv_registry = INV_EXPECT_REGISTRY.get(dataset, {})

    for test_spec in tests:
        examples = test_spec.get("examples") or []
        if not examples:
            stats["skipped"] += 1
            continue

        test_name = test_spec.get("name") or dataset
        capability = test_spec.get("capability") or dataset
        description = test_spec.get("description")

        stored_type = test_spec.get("test_type")
        dir_expect_tag = test_spec.get("dir_expectation")
        mft_expect_tag = test_spec.get("mft_expectation")
        inv_expect_tag = test_spec.get("inv_expectation")

        # --- Registry-based overrides (works with legacy dumps) ---
        if stored_type is None and test_name in dir_registry:
            stored_type = "DIR"
            dir_expect_tag = dir_expect_tag or dir_registry[test_name]

        if test_name in mft_registry:
            if mft_expect_tag is None:
                mft_expect_tag = mft_registry[test_name]
            if stored_type in (None, "INV"):
                stored_type = "MFT"

        if test_name in inv_registry:
            if inv_expect_tag is None:
                inv_expect_tag = inv_registry[test_name]

        data, labels, test_type, meta = _parse_test_examples(
            dataset, examples, explicit_type=stored_type
        )
        if not data:
            stats["skipped"] += 1
            continue

        if test_type == "dir":
            if dir_expect_tag is None:
                dir_expect_tag = dir_registry.get(test_name)
            if dir_expect_tag is None:
                print(
                    f"  WARNING: DIR test {test_name!r} has no expectation tag; "
                    f"falling back to INV(threshold=0.1)"
                )
                dir_expect_tag = "inv"
            expect_fn = _build_dir_expect(dir_expect_tag)
            kwargs: dict[str, Any] = dict(
                data=data, expect=expect_fn, labels=labels,
                name=test_name, capability=capability, description=description,
            )
            if meta is not None:
                kwargs["meta"] = meta
            test_obj = DIR(**kwargs)
            stats["dir"] += 1

        elif test_type == "mft_no_labels":
            if mft_expect_tag is None:
                mft_expect_tag = mft_registry.get(test_name)
            if mft_expect_tag is None:
                print(
                    f"  WARNING: MFT test {test_name!r} has no labels and no "
                    f"expectation tag; falling back to Expect.eq()"
                )
            expect_fn = _build_mft_expect(mft_expect_tag) if mft_expect_tag else None
            test_obj = MFT(
                data=data,
                expect=expect_fn,
                name=test_name,
                capability=capability,
                description=description,
            )
            stats["mft_custom_expect"] += 1

        elif test_type == "inv":
            if inv_expect_tag is None:
                inv_expect_tag = inv_registry.get(test_name)
            if inv_expect_tag is not None:
                expect_fn = _build_inv_expect(inv_expect_tag)
                kwargs = dict(
                    data=data, expect=expect_fn,
                    name=test_name, capability=capability, description=description,
                )
                if meta is not None:
                    kwargs["meta"] = meta
                test_obj = INV(**kwargs)
                stats["inv_custom_expect"] += 1
            else:
                test_obj = INV(
                    data=data,
                    name=test_name,
                    capability=capability,
                    description=description,
                )
                stats["inv"] += 1

        elif test_type == "mft_grouped":
            mft_tag = mft_expect_tag or mft_registry.get(test_name) or mft_dataset_default
            if mft_tag:
                expect_fn = _build_mft_expect(mft_tag)
                test_obj = MFT(
                    data=data, labels=labels, expect=expect_fn,
                    name=test_name, capability=capability, description=description,
                )
                stats["mft_custom_expect"] += 1
            else:
                test_obj = MFT(
                    data=data, labels=labels,
                    name=test_name, capability=capability, description=description,
                )
                stats["mft_grouped"] += 1

        else:
            mft_tag = mft_expect_tag or mft_registry.get(test_name) or mft_dataset_default
            if mft_tag:
                expect_fn = _build_mft_expect(mft_tag)
                test_obj = MFT(
                    data=data, labels=labels, expect=expect_fn,
                    name=test_name, capability=capability, description=description,
                )
                stats["mft_custom_expect"] += 1
            else:
                test_obj = MFT(
                    data=data, labels=labels,
                    name=test_name, capability=capability, description=description,
                )
                stats["mft"] += 1

        suite.add(test_obj, overwrite=True)

    if not getattr(suite, "tests", None):
        raise ValueError(f"Suite JSON produced no runnable tests: {suite_json_path}")

    total = (stats["mft"] + stats["inv"] + stats["dir"] + stats["mft_grouped"]
             + stats["mft_custom_expect"] + stats["inv_custom_expect"])
    print(
        f"Built suite from {suite_json_path.name}: "
        f"{total} tests ({stats['mft']} MFT, {stats['mft_grouped']} MFT-grouped, "
        f"{stats['mft_custom_expect']} MFT-custom-expect, "
        f"{stats['inv']} INV, {stats['inv_custom_expect']} INV-custom-expect, "
        f"{stats['dir']} DIR, {stats['skipped']} skipped)"
    )
    return suite
