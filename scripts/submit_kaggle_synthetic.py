"""Upload the made-up listings and start the E2B price-head kernel."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / ".venv/bin/kaggle"
DATASET_SLUG = "gemma-synthetic-sentence"
KERNEL_SLUG = "gemma-e2b-synthetic-price"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    owner = json.loads((Path.home() / ".kaggle/credentials.json").read_text())["username"]
    dataset = ROOT / "artifacts/kaggle-synthetic-data"
    if dataset.exists():
        shutil.rmtree(dataset)
    processed = dataset / "data/processed"
    processed.mkdir(parents=True)
    source_processed = ROOT / "data/synthetic-sentence/processed"
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        shutil.copyfile(source_processed / name, processed / name)
    package = dataset / "src/gemma_tabular"
    package.mkdir(parents=True)
    for path in (ROOT / "src/gemma_tabular").glob("*.py"):
        shutil.copyfile(path, package / path.name)
    checksums = {
        name: sha256(processed / name)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json")
    }
    (dataset / "results").mkdir()
    (dataset / "results/synthetic-sentence-checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (ROOT / "results/synthetic-sentence-checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "Gemma synthetic sentence",
        "id": f"{owner}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")
    created = subprocess.run([str(KAGGLE), "datasets", "create", "-p", str(dataset), "--dir-mode", "zip", "--keep-tabular"])
    if created.returncode != 0:
        versioned = subprocess.run([
            str(KAGGLE), "datasets", "version", "-p", str(dataset),
            "-m", "Made-up sentence listings and training code", "--dir-mode", "zip", "--keep-tabular",
        ])
        if versioned.returncode != 0:
            raise SystemExit(versioned.returncode)

    kernel = ROOT / "artifacts/kaggle-synthetic-kernel"
    if kernel.exists():
        shutil.rmtree(kernel)
    kernel.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/kaggle_synthetic_sentence.py", kernel / "synthetic.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "Gemma synthetic sentence",
        "code_file": "synthetic.py",
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
    pushed = subprocess.run([
        str(KAGGLE), "kernels", "push", "-p", str(kernel),
        "--accelerator", "NvidiaTeslaT4", "--timeout", "21600",
    ])
    if pushed.returncode != 0:
        raise SystemExit(pushed.returncode)
    print(json.dumps({
        "kernel": f"{owner}/{KERNEL_SLUG}",
        "url": f"https://www.kaggle.com/code/{owner}/{KERNEL_SLUG}",
        "dataset": f"{owner}/{DATASET_SLUG}",
    }, indent=2))


if __name__ == "__main__":
    main()
