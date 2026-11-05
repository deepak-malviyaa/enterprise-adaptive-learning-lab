from pathlib import Path

def ensure_output_tree(base):
    base = Path(base)
    paths = {name: base / name for name in ["metrics", "figures", "checkpoints", "reports", "predictions", "runs"]}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
