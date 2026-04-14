import yaml
from pathlib import Path


def load_yaml_config(filepath):
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def should_save_raw_output(config: dict) -> bool:
    """Return whether per-seed raw arrays should be written to disk."""
    output_cfg = config.get("output", {})
    return bool(output_cfg.get("save_raw", False))
