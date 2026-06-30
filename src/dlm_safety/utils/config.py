"""
Configuration loading and management.

Loads YAML config files and provides typed access to experiment parameters.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    type: str  # 'ar', 'llada'
    device: str = "auto"
    tokenizer: Optional[str] = None
    sampling_steps: int = 100
    seq_length: int = 128
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class SensitivityExpConfig:
    """Configuration for sensitivity measurement."""
    max_new_tokens: int = 32
    batched: bool = False
    batch_size: int = 4


@dataclass
class DataConfig:
    """Configuration for data loading."""
    sources: list = field(default_factory=list)


@dataclass
class InjectionConfig:
    """Configuration for injection experiments."""
    payloads: list = field(default_factory=list)
    positions: list = field(default_factory=lambda: ["begin", "mid", "end"])


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "experiment"
    output_dir: str = "results"
    models: list = field(default_factory=list)
    sensitivity: SensitivityExpConfig = field(default_factory=SensitivityExpConfig)
    data: DataConfig = field(default_factory=DataConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    diffusion_sweep: Optional[dict] = None
    checklist: Optional[dict] = None  # checklist.task, suite_path, release_data_dir, judge (model spec)


def load_config(config_path: str) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        ExperimentConfig with all fields populated.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Parse experiment metadata
    exp = raw.get("experiment", {})

    # Parse model configs
    models = []
    for m in raw.get("models", []):
        models.append(ModelConfig(
            name=m["name"],
            type=m["type"],
            device=m.get("device", "auto"),
            tokenizer=m.get("tokenizer"),
            sampling_steps=m.get("sampling_steps", 100),
            seq_length=m.get("seq_length", 128),
            model_kwargs=m.get("model_kwargs", {}),
        ))

    # Parse sensitivity config
    sens_raw = raw.get("sensitivity", {})
    sensitivity = SensitivityExpConfig(
        max_new_tokens=sens_raw.get("max_new_tokens", 32),
        batched=sens_raw.get("batched", False),
        batch_size=sens_raw.get("batch_size", 4),
    )

    # Parse data config
    data_raw = raw.get("data", {})
    data = DataConfig(sources=data_raw.get("sources", []))

    # Parse injection config
    inj_raw = raw.get("injection", {})
    injection = InjectionConfig(
        payloads=inj_raw.get("payloads", []),
        positions=inj_raw.get("positions", ["begin", "mid", "end"]),
    )

    return ExperimentConfig(
        name=exp.get("name", "experiment"),
        output_dir=exp.get("output_dir", "results"),
        models=models,
        sensitivity=sensitivity,
        data=data,
        injection=injection,
        diffusion_sweep=raw.get("diffusion_sweep"),
        checklist=raw.get("checklist"),
    )
