"""Start the private matched E2B pilot on one Kaggle T4 session."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv/bin/kaggle"
KERNEL_SLUG = "gemma-e2b-pilot"


def main():
    owner = json.loads((Path.home() / ".kaggle/credentials.json").read_text())["username"]
    kernel = ROOT / "artifacts/kaggle-pilot/kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "pilot.py").write_text((ROOT / "scripts/kaggle_e2b_pilot.py").read_text())
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Gemma E2B matched pilot",
        "code_file": "pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{owner}/gemma-e2b-smoke-files"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2) + "\n")
    completed = subprocess.run([
        str(KAGGLE), "kernels", "push", "-p", str(kernel),
        "--accelerator", "NvidiaTeslaT4", "--timeout", "21600",
    ])
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(json.dumps({
        "kernel": f"{owner}/{KERNEL_SLUG}",
        "url": f"https://www.kaggle.com/code/{owner}/{KERNEL_SLUG}",
        "timeout_seconds": 21600,
    }, indent=2))


if __name__ == "__main__":
    main()
