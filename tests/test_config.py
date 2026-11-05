from src.common.config import merged_config

def test_config_loads():
    cfg = merged_config("configs/global.yaml", "configs/use_cases/synthetic_binary.yaml")
    assert cfg["use_case"] == "synthetic_binary"
