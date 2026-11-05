from src.data.loaders import load_dataset
from src.models.baselines import build_model

def test_logistic_smoke():
    frame, target = load_dataset({"dataset":"synthetic_binary", "seed":42})
    X, y = frame.drop(columns=target).head(200), frame[target].head(200)
    model = build_model("logistic_regression", X, 42).fit(X, y)
    assert model.predict_proba(X[:3]).shape == (3, 2)
