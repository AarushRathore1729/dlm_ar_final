"""
Data loading utilities for safety evaluation datasets.

Supports:
  - BIPIA (Benchmark for Indirect Prompt Injection Attacks)
  - NotInject (benign samples with injection trigger words)
  - Local JSON prompt sets (alpaca/mmlu style)
  - Custom injection prompt set construction
"""

import json
from pathlib import Path
from typing import Optional

def load_bipia_dataset(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load the BIPIA dataset from HuggingFace.

    Each sample contains:
      - task: the task description
      - content: the external content (with/without injection)
      - injection: the injected instruction
      - label: whether it's an attack

    Args:
        split: Dataset split ('train', 'test').
        max_samples: Limit number of samples loaded.

    Returns:
        List of sample dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the `datasets` library: uv pip install datasets"
        )

    # BIPIA is available via the microsoft/bipia GitHub repo
    # We use the HF datasets version if available, otherwise fall back
    # to a local/cached version
    try:
        ds = load_dataset("yjcHH/bipia", split=split)
    except Exception:
        print(
            "BIPIA not found on HuggingFace. Trying alternative sources..."
        )
        try:
            ds = load_dataset(
                "protectai/prompt-injection-validation", split=split
            )
        except Exception:
            print(
                "Could not load BIPIA. Please download manually from "
                "https://github.com/microsoft/BIPIA and place in data/bipia/"
            )
            return []

    samples = [dict(row) for row in ds]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_notinject_dataset(
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load the NotInject dataset (benign samples with trigger words).

    These are used as negative controls -- benign prompts that
    should NOT trigger injection defenses.

    Args:
        max_samples: Limit number of samples.

    Returns:
        List of sample dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the `datasets` library: uv pip install datasets"
        )

    try:
        ds = load_dataset("notinject/notinject", split="test")
    except Exception:
        print(
            "NotInject not found. Trying alternative sources..."
        )
        return []

    samples = [dict(row) for row in ds]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_local_prompt_sets_json(
    json_path: str,
    max_sets: Optional[int] = None,
    min_variants: int = 2,
    max_variants_per_set: Optional[int] = None,
) -> list[list[str]]:
    """Load grouped prompt sets from a local JSON file.

    Expected schema:
      - A JSON list of objects
      - Each object should contain a `prompts` field that is a list[str]
      - Example: [{"id": 0, "prompts": ["p1", "p2", ...]}, ...]

    Args:
        json_path: Path to the local JSON file.
        max_sets: Optional cap on number of prompt sets loaded.
        min_variants: Minimum number of prompt variants required per set.
        max_variants_per_set: If set, use only the first N variants per set
            (reduces cost: fewer generations and smaller log-prob matrix).

    Returns:
        List of prompt sets, each prompt set being list[str].
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Local prompt JSON not found: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {json_path}, got {type(data).__name__}"
        )

    prompt_sets: list[list[str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        prompts = item.get("prompts")
        if not isinstance(prompts, list):
            continue

        clean_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
        if len(clean_prompts) < min_variants:
            continue

        if max_variants_per_set is not None:
            clean_prompts = clean_prompts[:max_variants_per_set]
        prompt_sets.append(clean_prompts)

        if max_sets is not None and len(prompt_sets) >= max_sets:
            break

    if not prompt_sets:
        raise ValueError(
            f"No valid prompt sets found in {json_path}. "
            "Expected entries with non-empty `prompts` lists."
        )

    return prompt_sets
