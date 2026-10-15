"""Check a Kaggle runtime without installing packages or loading model weights."""

import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request


MODEL_ID = "google/gemma-4-E2B"
MODEL_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"


def collect_report(check_network=True):
    report = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Environment check only; no model training or accuracy measurement",
        "training_started": False,
        "model_weights_downloaded": False,
        "python_version": platform.python_version(),
        "packages": {},
        "cuda_available": False,
        "gpus": [],
        "network_checks": {},
        "gpu_quota": "Check remaining quota in the Kaggle notebook interface",
    }
    report["nvidia_device_files"] = [path.name for path in Path("/dev").glob("nvidia*")]
    report["kaggle_working_directory_exists"] = Path("/kaggle/working").is_dir()
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            report["nvidia_smi"] = {"return_code": result.returncode, "gpu_summary": result.stdout.strip()[:1000]}
        except Exception as error:
            report["nvidia_smi"] = {"error_type": type(error).__name__}
    else:
        report["nvidia_smi"] = {"available": False}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy", "pandas", "pyarrow"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = "not installed"
    try:
        import torch

        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            report["torch_cuda_architectures"] = torch.cuda.get_arch_list()
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                gpu = {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": [properties.major, properties.minor],
                    "native_bfloat16": properties.major >= 8,
                }
                try:
                    # A two-element operation tests that this CUDA wheel can
                    # actually execute a kernel on the allocated GPU.
                    probe = torch.ones(2, device=f"cuda:{index}")
                    result = float(probe.sum().item())
                    del probe
                    gpu["kernel_check"] = {"ok": result == 2.0}
                except Exception as error:
                    gpu["kernel_check"] = {"ok": False, "error_type": type(error).__name__}
                report["gpus"].append(gpu)
    except Exception as error:
        report["torch_check_error_type"] = type(error).__name__

    if check_network:
        checks = {
            "gemma_e2b_config": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/config.json",
            "mercari_metadata": "https://www.openml.org/api/v1/json/data/46660",
        }
        for name, url in checks.items():
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("Unexpectedly large metadata response")
                metadata = json.loads(raw)
                if name == "gemma_e2b_config":
                    text_config = metadata["text_config"]
                    valid = metadata["model_type"] == "gemma4" and text_config["hidden_size"] == 1536
                    report["network_checks"][name] = {
                        "ok": valid,
                        "model_id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "hidden_size": text_config["hidden_size"],
                        "text_layers": text_config["num_hidden_layers"],
                    }
                else:
                    dataset = metadata["data_set_description"]
                    valid = dataset["name"] == "mercari_price_suggestion100K" and str(dataset["version"]) == "1"
                    report["network_checks"][name] = {"ok": valid, "dataset": dataset["name"]}
            except Exception as error:
                # These requests use fixed public URLs and no credentials.
                report["network_checks"][name] = {
                    "ok": False, "error_type": type(error).__name__,
                    "error_detail": str(getattr(error, "reason", error))[:300],
                }
    else:
        report["network_checks"] = {"status": "skipped"}
    report["runtime_checks_passed"] = bool(
        report["cuda_available"] and report["gpus"]
        and all(gpu["kernel_check"]["ok"] for gpu in report["gpus"])
        and check_network and all(item.get("ok") for item in report["network_checks"].values())
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_folder = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    parser.add_argument("--output", type=Path, default=default_folder / "kaggle-preflight.json")
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args(argv)
    report = collect_report(check_network=not args.skip_network)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
