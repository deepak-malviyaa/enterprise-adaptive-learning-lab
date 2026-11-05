import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml
from .common.config import merged_config, VALID_MODES
from .common.device import select_device
from .common.paths import ensure_output_tree
from .common.seeds import set_global_seed
from .data.loaders import load_dataset
from .data.validation import validate_frame
from .data.splits import make_splits
from .models.baselines import build_model
from .evaluation.classification import classification_metrics
from .reporting.markdown_report import write_report

def run_one(use_case, mode, requested_device):
    config = merged_config("configs/global.yaml", f"configs/use_cases/{use_case}.yaml", {"mode": mode, "device": requested_device})
    config["resolved_device"] = select_device(config["device"])
    set_global_seed(int(config["seed"]))
    paths = ensure_output_tree(config["output_dir"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{use_case}"
    run_dir = paths["runs"] / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    frame, target = load_dataset(config)
    max_rows = int(config.get("runtime", {}).get("max_rows", 0))
    if max_rows > 0: frame = frame.head(max_rows).copy()
    audit = validate_frame(frame, target)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(frame, target, config)
    audit.update({"train_rows": len(X_train), "validation_rows": len(X_val), "test_rows": len(X_test)})
    results = {}
    pred_frame = pd.DataFrame({"y_true": y_test.reset_index(drop=True)})
    for model_name in config["models"]:
        if model_name not in {"logistic_regression", "random_forest", "gradient_boosting"}:
            print(f"SKIP {model_name}: planned for a later phase")
            continue
        model = build_model(model_name, X_train, int(config["seed"]))
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        results[model_name] = classification_metrics(y_test, probs)
        pred_frame[f"probability_{model_name}"] = probs
    if not results:
        raise RuntimeError("No implemented model was selected")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pred_frame.to_csv(run_dir / "predictions.csv", index=False)
    report = paths["reports"] / f"{run_id}.md"
    write_report(report, run_id, config, audit, results, [
        "This is a research POC, not a production decision system.",
        "Public or synthetic data cannot establish regulatory compliance or real-world savings.",
        "Threshold 0.5 is a starter operating point and must be selected on validation data for a real use case.",
    ])
    print(json.dumps({"run_id": run_id, "device": config["resolved_device"], "metrics": results, "report": str(report)}, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-case", default="synthetic_binary", help="One use case for the starter runner")
    parser.add_argument("--mode", default="quickstart", choices=sorted(VALID_MODES))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    run_one(args.use_case, args.mode, args.device)
if __name__ == "__main__": main()
