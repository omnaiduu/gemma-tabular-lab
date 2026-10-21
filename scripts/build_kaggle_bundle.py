"""Build a credential-free source bundle and a standalone Kaggle check notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAME = "Gemma_E2B_Kaggle_Check.ipynb"
BUNDLE_NAME = "Gemma_Tabular_Kaggle_Source.zip"


def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(destination):
    destination.mkdir(parents=True, exist_ok=True)
    preflight_source = (ROOT / "scripts/kaggle_preflight.py").read_text()
    notebook_source = preflight_source.rsplit('\nif __name__ == "__main__":', 1)[0] + "\nmain([])\n"
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell("markdown", """# Check the Kaggle runtime

This checks the runtime only. It does not install packages, download model weights,
or start training. The CUDA check can use a little GPU quota. No password or API
token is needed.

1. Sign in to Kaggle and import this notebook.
2. Keep the notebook private, and enable Internet in the session options.
3. Select one GPU T4. Two T4 chips do not share memory. Leave TPU unselected.
4. Run the cell. The JSON is saved as `/kaggle/working/kaggle-preflight.json`.

The JSON has package versions and whether a GPU is present. It does not include
passwords or listing rows.
"""),
            cell("code", notebook_source),
        ],
    }
    for index, notebook_cell in enumerate(notebook["cells"]):
        notebook_cell["id"] = f"kaggle-check-{index}"
    notebook_path = destination / NOTEBOOK_NAME
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")

    # Explicit inclusion avoids accidentally packing local credentials, dataset
    # rows, caches, model weights, or unrelated user files.
    included = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "docs/blog.md",
        ROOT / "scripts/fetch_mercari.py",
        ROOT / "scripts/kaggle_preflight.py",
        ROOT / "scripts/run_kaggle_preflight.py",
        ROOT / "scripts/build_kaggle_bundle.py",
        ROOT / "results/data-provenance.json",
        ROOT / "results/data-manifest.json",
        ROOT / "results/processed-checksums.json",
        ROOT / "results/baseline-comparison.csv",
    ] + sorted((ROOT / "src/gemma_tabular").glob("*.py"))
    bundle_path = destination / BUNDLE_NAME
    manifest = {}
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in included:
            name = "gemma-tabular-lab/" + path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            archive.writestr(name, content)
            manifest[name] = hashlib.sha256(content).hexdigest()
        name = "gemma-tabular-lab/notebooks/" + NOTEBOOK_NAME
        content = notebook_path.read_bytes()
        archive.writestr(name, content)
        manifest[name] = hashlib.sha256(content).hexdigest()
        archive.writestr("gemma-tabular-lab/bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    return {"notebook": str(notebook_path), "source_bundle": str(bundle_path), "source_files": len(manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/kaggle")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
