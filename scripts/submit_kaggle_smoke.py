"""Upload the private source payload and start one Kaggle GPU smoke run."""

import json
import os
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv/bin/kaggle"
DATASET_SLUG = "gemma-e2b-smoke-files"
KERNEL_SLUG = "gemma-e2b-regression-smoke"
PAYLOAD_FILES = [
    "scripts/fetch_mercari.py",
    "scripts/kaggle_e2b_smoke.py",
    "results/processed-checksums.json",
]


def username():
    document = json.loads((Path.home() / ".kaggle/credentials.json").read_text())
    name = document.get("username")
    if not name:
        raise RuntimeError("Kaggle username is missing. Run: .venv/bin/kaggle auth login")
    return name


def run(command):
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command)
    return completed.returncode


def build(owner):
    folder = ROOT / "artifacts/kaggle-smoke"
    dataset = folder / "dataset"
    kernel = folder / "kernel"
    dataset.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    payload = dataset / "payload.tar"
    with tarfile.open(payload, "w") as archive:
        for relative in PAYLOAD_FILES + sorted(path.relative_to(ROOT).as_posix()
                                                for path in (ROOT / "src/gemma_tabular").glob("*.py")):
            archive.add(ROOT / relative, arcname=f"gemma-tabular-lab/{relative}")
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "Gemma E2B smoke files",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")
    (kernel / "smoke.py").write_text((ROOT / "scripts/kaggle_e2b_smoke.py").read_text())
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Gemma E2B regression smoke",
        "code_file": "smoke.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{owner}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2) + "\n")
    return dataset, kernel


def main():
    owner = username()
    dataset, kernel = build(owner)
    created = run([str(KAGGLE), "datasets", "create", "-p", str(dataset), "-t", "-q"])
    if created != 0:
        versioned = run([str(KAGGLE), "datasets", "version", "-p", str(dataset), "-t", "-q",
                         "-m", "Update E2B smoke source"])
        if versioned != 0:
            raise SystemExit("Could not upload the private Kaggle dataset")
    pushed = run([str(KAGGLE), "kernels", "push", "-p", str(kernel),
                  "--accelerator", "NvidiaTeslaT4", "--timeout", "7200"])
    if pushed != 0:
        raise SystemExit("Kaggle rejected the kernel push")
    print(json.dumps({
        "kernel": f"{owner}/{KERNEL_SLUG}",
        "url": f"https://www.kaggle.com/code/{owner}/{KERNEL_SLUG}",
        "private": True,
        "accelerator": "NvidiaTeslaT4",
        "timeout_seconds": 7200,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
