import yaml
from pathlib import Path


def load_yaml_config(filepath):
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config