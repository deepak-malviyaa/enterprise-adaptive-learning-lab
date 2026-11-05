from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification

DATASET_FILES = {
    "ulb_creditcard": "creditcard.csv",
    "ibm_telco_churn": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "uci_ai4i_2020": "ai4i2020.csv",
}

def load_dataset(config):
    name = config["dataset"]
    if name == "synthetic_binary":
        X, y = make_classification(n_samples=2500, n_features=20, n_informative=8,
            n_redundant=4, weights=[0.94, 0.06], class_sep=1.0,
            random_state=int(config["seed"]))
        frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        frame["target"] = y
        return frame, "target"
    path = Path(config.get("data_path", f"data/raw/{DATASET_FILES.get(name, name + '.csv')}"))
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' was not found at {path}. Place the approved source file there, "
            "complete its data card and licence review, then rerun. Raw data must not be committed."
        )
    frame = pd.read_csv(path)
    targets = {"ulb_creditcard": "Class", "ibm_telco_churn": "Churn", "uci_ai4i_2020": "Machine failure"}
    return frame, targets[name]
