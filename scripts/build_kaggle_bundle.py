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
            cell("markdown", """# Check Kaggle for the Gemma E2B experiment

This checks the runtime only. It does not install packages, download model weights,
or start training. The tiny CUDA check and runtime startup can consume a small
amount of your Kaggle GPU quota. No password or API token is needed.

1. Sign in to Kaggle and import this notebook.
2. Keep the notebook private. In Settings / Session options, enable Internet.
3. Select GPU T4 x2 if available; otherwise P100. Complete Kaggle's own account
   verification if it asks. Leave TPU unselected for this PyTorch experiment.
4. Run the cell below and copy its JSON output back to Codex.
5. Also report the remaining GPU quota shown in Kaggle, then stop the session.

Two T4 GPUs do not automatically combine their memory. We will choose the
training setup after seeing the allocated hardware and installed CUDA software.
"""),
            cell("code", notebook_source),
            cell("markdown", """## What comes next

The JSON is also saved as `/kaggle/working/kaggle-preflight.json` on Kaggle.
Share that output or file; it contains versions, GPU information, and public
metadata connection checks. It does not intentionally collect account IDs,
environment variables, passwords, tokens, or dataset rows.

The source bundle is for the next training step; you do not need to attach it to
run this check. It contains code and aggregate reports, without credentials,
model weights, or source dataset rows. It is not yet a validated T4/P100 training
package. The current training code still needs low-memory training support and
a real GPU memory/gradient smoke test before full E2B runs.
"""),
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
        ROOT / "scripts/probe_kaggle_session.py",
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
