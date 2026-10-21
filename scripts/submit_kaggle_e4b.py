"""Start the private E4B price-head run on one Kaggle T4."""

import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv/bin/kaggle"
KERNEL_SLUG = "gemma-e4b-price-head"


def main():
    owner = json.loads((Path.home() / ".kaggle/credentials.json").read_text())["username"]
    kernel = ROOT / "artifacts/kaggle-e4b/kernel"
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    template = (ROOT / "scripts/kaggle_e4b_regression.py").read_text()
    train_source = json.dumps((ROOT / "src/gemma_tabular/train.py").read_text())
    cli_source = json.dumps((ROOT / "src/gemma_tabular/cli.py").read_text())
    template = template.replace("TRAIN_SOURCE = None", "TRAIN_SOURCE = " + train_source, 1)
    template = template.replace("CLI_SOURCE = None", "CLI_SOURCE = " + cli_source, 1)
    if "TRAIN_SOURCE = None" in template or "CLI_SOURCE = None" in template:
        raise RuntimeError("Could not pack the training code into the notebook")
    (kernel / "e4b.py").write_text(template)
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Gemma E4B price head",
        "code_file": "e4b.py",
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
