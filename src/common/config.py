from pathlib import Path
import yaml

VALID_MODES = {"quickstart", "single_use_case", "selected_use_cases", "full_poc", "advanced"}

def load_yaml(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data

def merged_config(global_path, use_case_path, overrides=None):
    config = load_yaml(global_path)
    config.update(load_yaml(use_case_path))
    config.update(overrides or {})
    validate_config(config)
    return config

def validate_config(config):
    required = ["seed", "device", "output_dir", "use_case", "dataset", "task", "split", "models"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required configuration fields: {missing}")
    if config["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of auto, cpu, cuda, mps")
    split = config["split"]
    for key in ("validation_fraction", "test_fraction"):
        value = float(split.get(key, 0))
        if not 0 < value < 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if split["validation_fraction"] + split["test_fraction"] >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
