"""Kaggle entry: train the E2B price head on the made-up sentence shop.

The listing files are the attached dataset. This script does not download
Mercari and does not train the digit head.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path("/kaggle/working/gemma-tabular-lab")
STATUS = Path("/kaggle/working/synthetic-sentence-status.json")


def status(payload):
    print(json.dumps(payload), flush=True)
    STATUS.write_text(json.dumps(payload, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slices(test_rows, predictions):
    if len(test_rows) != len(predictions):
        raise RuntimeError("Prediction file is not aligned with the test file")
    grouped = {"seen": [], "new": []}
    for row, item in zip(test_rows, predictions):
        if abs(row["target"] - item["actual"]) > 1e-6:
            raise RuntimeError("Prediction file is not in test-file order")
        grouped[row["wording"]].append((row["target"], item["predicted"], row["meaning"]))
    report = {}
    for name, pairs in grouped.items():
        errors = [abs(actual - predicted) for actual, predicted, _ in pairs]
        means = {}
        for meaning in ("good", "bad"):
            chosen = [predicted for _, predicted, item_meaning in pairs if item_meaning == meaning]
            truth = [actual for actual, _, item_meaning in pairs if item_meaning == meaning]
            means[meaning] = {"true_mean": sum(truth) / len(truth), "predicted_mean": sum(chosen) / len(chosen)}
        report[name] = {
            "n": len(pairs),
            "mae": sum(errors) / len(errors),
            "true_gap": means["good"]["true_mean"] - means["bad"]["true_mean"],
            "model_gap": means["good"]["predicted_mean"] - means["bad"]["predicted_mean"],
        }
    return report


def main():
    markers = [path for path in Path("/kaggle/input").rglob("manifest.json") if path.parent.name == "processed"]
    if len(markers) != 1:
        raise RuntimeError(f"Expected one synthetic manifest, found {len(markers)}")
    source = markers[0].parents[2]
    if ROOT.exists():
        shutil.rmtree(ROOT)
    shutil.copytree(source, ROOT, ignore=shutil.ignore_patterns("dataset-metadata.json"))
    expected = json.loads((ROOT / "results/synthetic-sentence-checksums.json").read_text())
    actual = {name: sha256(ROOT / "data/processed" / name) for name in expected}
    if actual != expected:
        raise RuntimeError("Synthetic files do not match the recorded checksums")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This Kaggle session has no GPU. Training was not started.")
    status({
        "event": "cuda_check",
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    })
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["HF_HOME"] = "/kaggle/working/hf-home"
    output = Path("/kaggle/working/synthetic-sentence")
    command = [
        sys.executable, "-m", "gemma_tabular.cli", "train",
        "--data", "data/processed",
        "--output", str(output),
        "--head", "regression",
        "--adaptation", "lora",
        "--model-id", "google/gemma-4-E2B",
        "--revision", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "--steps", "500",
        "--batch-size", "1",
        "--accumulation-steps", "8",
        "--max-length", "128",
        "--max-seconds", "10800",
        "--precision", "fp16",
        "--eval-batch-size", "4",
        "--validation-limit", "200",
        "--seed", "42",
    ]
    status({"event": "training_started", "steps": 500, "train_rows": 1000})
    completed = subprocess.run(command, cwd=ROOT, env=env)
    report_path = output / "training-report.json"
    if report_path.is_file():
        shutil.copyfile(report_path, Path("/kaggle/working/synthetic-sentence-report.json"))
    predictions_path = output / "predictions.jsonl"
    if predictions_path.is_file():
        shutil.copyfile(predictions_path, Path("/kaggle/working/synthetic-sentence-predictions.jsonl"))
    if completed.returncode != 0:
        raise RuntimeError(f"training exited {completed.returncode}")
    test_rows = [json.loads(line) for line in (ROOT / "data/processed/test.jsonl").read_text().splitlines()]
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    sliced = slices(test_rows, predictions)
    status({"event": "head_finished", "slices": sliced})


if __name__ == "__main__":
    main()
