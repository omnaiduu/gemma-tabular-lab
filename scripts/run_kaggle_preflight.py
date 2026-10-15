"""Submit one private, bounded Kaggle GPU check after Kaggle CLI sign-in."""

import argparse
import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import re
import time
import uuid

from build_kaggle_bundle import build, NOTEBOOK_NAME


ROOT = Path(__file__).resolve().parents[1]
# Public Kaggle GPU image, resolved from its official GCR manifest on 2026-10-15.
GPU_IMAGE = "gcr.io/kaggle-gpu-images/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def signed_in_username():
    # Read only the username; never copy the credential document into artifacts.
    for path in (Path.home() / ".kaggle/credentials.json", Path.home() / ".kaggle/kaggle.json"):
        if path.is_file():
            value = json.loads(path.read_text()).get("username")
            if value:
                return value
    return os.environ.get("KAGGLE_USERNAME")


def prepare_job(destination, username):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid Kaggle username")
    built = build(destination)
    notebook_path = Path(built["notebook"])
    notebook = json.loads(notebook_path.read_text())
    notebook["metadata"]["accelerator"] = "GPU"
    notebook["metadata"]["kaggle"] = {
        "accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
        "isInternetEnabled": True, "language": "python", "sourceType": "notebook",
    }
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    title = "Gemma E2B GPU Check " + destination.name
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    metadata = {
        "id": f"{username}/{slug}", "title": title,
        "code_file": Path(built["notebook"]).name,
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "docker_image": GPU_IMAGE, "docker_image_pinning_type": "original",
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def submit(api, folder, runtime_seconds):
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    if (metadata.get("is_private") is not True or metadata.get("enable_gpu") is not True
            or metadata.get("enable_tpu") is not False
            or metadata.get("machine_shape") != "NvidiaTeslaT4"
            or not 120 <= runtime_seconds <= 600):
        raise ValueError("Expected one private T4 check bounded to at most 600 seconds")
    # One submission, without retries or fallback to another accelerator.
    response = api.kernels_push(str(folder), timeout=str(runtime_seconds), acc="NvidiaTeslaT4")
    if response is None or response.error:
        raise RuntimeError("Kaggle rejected the notebook submission; inspect the private API log")
    return metadata["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", help="Defaults to the locally authenticated Kaggle username")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-seconds", type=int, default=600)
    parser.add_argument("--wait-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 120 <= args.runtime_seconds <= 600 or not 30 <= args.wait_seconds <= 1800:
        parser.error("Use a 120–600 second runtime and a 30–1800 second polling window")
    username = args.user or signed_in_username()
    if not username:
        parser.error("Kaggle sign-in is required: .venv/bin/kaggle auth login")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = ROOT / "artifacts/kaggle-runs" / run_id
    metadata = prepare_job(folder, username)
    print(json.dumps({"prepared": str(folder), "kernel": metadata["id"],
                      "private": True, "accelerator": "NvidiaTeslaT4",
                      "maximum_runtime_seconds": args.runtime_seconds,
                      "dry_run": args.dry_run}), flush=True)
    if args.dry_run:
        return
    fd = os.open(folder / "private-api.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            kernel = submit(api, folder, args.runtime_seconds)
        print(json.dumps({"submitted": kernel, "url": "https://www.kaggle.com/code/" + kernel}), flush=True)
        deadline = time.monotonic() + args.wait_seconds
        last_status = None
        while time.monotonic() < deadline:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                response = api.kernels_status(kernel)
            status = str(response.status).lower().rsplit(".", 1)[-1]
            if status != last_status:
                print(json.dumps({"kernel_status": status}), flush=True)
                last_status = status
            if status in ("complete", "completed", "error", "failed", "cancelled", "canceled"):
                output = folder / "output"
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    api.kernels_output(kernel, str(output), file_pattern=r"^kaggle-preflight\.json$", force=True, quiet=True)
                report_path = output / "kaggle-preflight.json"
                if status not in ("complete", "completed") or not report_path.is_file():
                    raise RuntimeError("Kaggle check did not produce a completed report; inspect downloaded logs")
                print(report_path.read_text(), flush=True)
                if not json.loads(report_path.read_text()).get("runtime_checks_passed"):
                    raise RuntimeError("Kaggle executed the check, but GPU/network checks failed; training was not started")
                return
            time.sleep(min(15, max(0, deadline - time.monotonic())))
    print(json.dumps({"status": "polling deadline reached", "kernel": metadata["id"],
                      "note": "No second job was submitted; inspect this run before any retry"}), flush=True)


if __name__ == "__main__":
    main()
