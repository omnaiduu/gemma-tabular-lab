"""Kaggle entry for the E4B direct price head.

Same 1,250-step recipe as the finished E2B price head. The per-layer embedding
table stays on CPU because the text weights do not fit beside it on one T4.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/kaggle/working/gemma-tabular-lab")
STATUS = Path("/kaggle/working/e4b-regression-status.json")
# Filled by scripts/submit_kaggle_e4b.py. Kaggle runs only this one file.
TRAIN_SOURCE = None
CLI_SOURCE = None


def status(payload):
    print(json.dumps(payload), flush=True)
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    markers = list(Path("/kaggle/input").rglob("processed-checksums.json"))
    if len(markers) != 1:
        raise RuntimeError(f"Expected one prepared source tree, found {len(markers)}")
    source = markers[0].parents[1]
    if ROOT.exists():
        shutil.rmtree(ROOT)
    shutil.copytree(source, ROOT)
    if not TRAIN_SOURCE or not CLI_SOURCE:
        raise RuntimeError("Training code was not packed into this notebook")
    (ROOT / "src/gemma_tabular/train.py").write_text(TRAIN_SOURCE)
    (ROOT / "src/gemma_tabular/cli.py").write_text(CLI_SOURCE)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["HF_HOME"] = "/kaggle/working/hf-home"
    fetch = subprocess.run([sys.executable, "scripts/fetch_mercari.py"], cwd=ROOT, env=env)
    if fetch.returncode != 0:
        raise RuntimeError("Mercari download failed")
    prepare = subprocess.run(
        [sys.executable, "-m", "gemma_tabular.cli", "prepare",
         "--source", "data/raw/train.tsv", "--output", "data/processed",
         "--train-limit", "10000", "--holdout-limit", "1000", "--seed", "42"],
        cwd=ROOT, env=env)
    if prepare.returncode != 0:
        raise RuntimeError("Split preparation failed")
    expected = json.loads((ROOT / "results/processed-checksums.json").read_text())["files"]
    actual = {name: sha256(ROOT / "data/processed" / name) for name in expected}
    if actual != expected:
        raise RuntimeError("Prepared splits do not match the recorded checksums")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This Kaggle session has no GPU. Training was not started.")
    status({
        "event": "cuda_check",
        "cuda_available": True,
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    })
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
    output = Path("/kaggle/working/e4b-regression")
    command = [
        sys.executable, "-m", "gemma_tabular.cli", "train",
        "--data", "data/processed",
        "--output", str(output),
        "--head", "regression",
        "--adaptation", "lora",
        "--model-id", "google/gemma-4-E4B",
        "--revision", "411aa17b749aa952df1359d2dcea73917a544d9a",
        "--steps", "1250",
        "--batch-size", "1",
        "--accumulation-steps", "8",
        "--max-length", "256",
        "--max-seconds", "18000",
        "--precision", "fp16",
        "--eval-batch-size", "4",
        "--validation-limit", "256",
        "--seed", "42",
        "--cpu-per-layer-embeddings",
    ]
    status({"event": "training_started", "head": "regression", "model": "google/gemma-4-E4B"})
    completed = subprocess.run(command, cwd=ROOT, env=env)
    report = output / "training-report.json"
    if report.is_file():
        shutil.copyfile(report, Path("/kaggle/working/e4b-regression-report.json"))
    if completed.returncode != 0:
        raise RuntimeError(f"regression training exited {completed.returncode}")
    status({"event": "head_finished", "head": "regression", "report": str(report)})


if __name__ == "__main__":
    main()
