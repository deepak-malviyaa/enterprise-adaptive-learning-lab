import json, os, platform, sys
from pathlib import Path
import psutil
from .common.device import select_device

def environment_details():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "selected_device": select_device("auto"),
    }

def main():
    details = environment_details()
    print(json.dumps(details, indent=2))
    Path("outputs/runs").mkdir(parents=True, exist_ok=True)
    Path("outputs/runs/environment.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
if __name__ == "__main__": main()
