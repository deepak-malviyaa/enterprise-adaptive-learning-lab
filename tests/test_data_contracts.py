from src.data.loaders import load_dataset
from src.data.validation import validate_frame

def test_synthetic_contract():
    frame, target = load_dataset({"dataset":"synthetic_binary", "seed":42})
    summary = validate_frame(frame, target)
    assert summary["rows"] == 2500
    assert 0 < summary["positive_rate"] < 0.2
