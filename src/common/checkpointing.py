from pathlib import Path
import joblib

def save_sklearn_model(model, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_sklearn_model(path):
    return joblib.load(path)
