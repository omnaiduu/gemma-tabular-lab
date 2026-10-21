"""Kaggle entry for one bounded E2B regression smoke test.

Run only on a Kaggle GPU kernel. It rebuilds the fixed Mercari splits, checks
their checksums, then trains the regression head for 20 steps. It does not
run the final test set.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/kaggle/working/gemma-tabular-lab")
REPORT = Path("/kaggle/working/e2b-smoke-report.json")


def status(payload):
    print(json.dumps(payload), flush=True)
    REPORT.write_text(json.dumps(payload, indent=2) + "\n")


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
    status({"event": "payload_ready", "source": str(source)})

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
        status({"event": "checksum_mismatch", "expected": expected, "actual": actual})
        raise RuntimeError("Prepared splits do not match the recorded checksums")
    status({"event": "checksums_ok"})

    import torch
    cuda = {
        "event": "cuda_check",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    status(cuda)
    if not cuda["cuda_available"]:
        raise RuntimeError("This Kaggle session has no GPU. Training was not started.")
    # Kaggle's image has torchao 0.10. The installed peft refuses to add LoRA
    # unless torchao is absent or newer than 0.16. Removing it avoids pulling a
    # different PyTorch build.
    removed = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    status({"event": "torchao_removed", "returncode": removed.returncode})

    command = [
        sys.executable, "-m", "gemma_tabular.cli", "train",
        "--data", "data/processed",
        "--output", "/kaggle/working/e2b-regression-smoke",
        "--head", "regression",
        "--adaptation", "lora",
        "--model-id", "google/gemma-4-E2B",
        "--revision", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "--steps", "20",
        "--batch-size", "1",
        "--max-length", "256",
        "--max-seconds", "5400",
        "--precision", "fp16",
        "--eval-batch-size", "1",
        "--validation-limit", "8",
        "--skip-test",
    ]
    status({"event": "training_started", "command": command[3:]})
    completed = subprocess.run(command, cwd=ROOT, env=env)
    report_path = Path("/kaggle/working/e2b-regression-smoke/training-report.json")
    if report_path.is_file():
        shutil.copyfile(report_path, REPORT)
    if completed.returncode != 0:
        raise RuntimeError(f"Training command exited {completed.returncode}")
    status({"event": "smoke_finished", "report": str(report_path)})


if __name__ == "__main__":
    main()
