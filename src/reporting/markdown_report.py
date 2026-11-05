from pathlib import Path

def write_report(path, run_id, config, data_summary, results, limitations):
    lines = [f"# Run report: {run_id}", "", "## Configuration", "```yaml"]
    import yaml
    lines += [yaml.safe_dump(config, sort_keys=False).strip(), "```", "", "## Dataset audit"]
    lines += [f"- **{k}:** {v}" for k, v in data_summary.items()]
    lines += ["", "## Model results"]
    for model, metrics in results.items():
        lines += [f"### {model}"] + [f"- **{k}:** {v:.6f}" for k, v in metrics.items()]
    lines += ["", "## Interpretation", "Higher PR-AUC indicates better ranking of the minority class. Review calibration and operating-point metrics before making decisions.", "", "## Limitations"]
    lines += [f"- {item}" for item in limitations]
    lines += ["", "## Next experiment", "Validate the dataset contract and add one controlled model family without changing the split or test set."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
