"""Kaggle entry for the digit head only.

The matched pilot already finished the regression head. This session repeats
the same 1,250-step settings for the four-digit head and then scores the
held-out test rows.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/kaggle/working/gemma-tabular-lab")
STATUS = Path("/kaggle/working/e2b-digit-status.json")


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
    output = Path("/kaggle/working/e2b-digits")
    command = [
        sys.executable, "-m", "gemma_tabular.cli", "train",
        "--data", "data/processed",
        "--output", str(output),
        "--head", "digits",
        "--adaptation", "lora",
        "--model-id", "google/gemma-4-E2B",
        "--revision", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "--steps", "1250",
        "--batch-size", "1",
        "--accumulation-steps", "8",
        "--max-length", "256",
        "--max-seconds", "9000",
        "--precision", "fp16",
        "--eval-batch-size", "4",
        "--validation-limit", "256",
        "--seed", "42",
    ]
    status({"event": "training_started", "head": "digits"})
    completed = subprocess.run(command, cwd=ROOT, env=env)
    report = output / "training-report.json"
    if report.is_file():
        shutil.copyfile(report, Path("/kaggle/working/e2b-digits-report.json"))
    if completed.returncode != 0:
        raise RuntimeError(f"digits training exited {completed.returncode}")
    status({"event": "head_finished", "head": "digits", "report": str(report)})


if __name__ == "__main__":
    main()
