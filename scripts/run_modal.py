"""Launch one bounded Modal job after data and authentication preflights.

Run with PYTHONPATH=src and MODAL_PROFILE=gemma-tabular-lab.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemma_tabular.budget import Ledger, worst_case_cost


def record(message):
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with (ROOT / "docs" / "findings.md").open("a") as handle:
        handle.write(f"\n## {timestamp} — Run event\n\n{message}\n")


def preflight():
    try:
        result = subprocess.run(["modal", "token", "info"], capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise RuntimeError("Modal authentication preflight failed or timed out; no compute job was launched") from None
    if result.returncode:
        # Do not expose token-info output, authentication metadata, or tracebacks.
        raise RuntimeError("Modal authentication preflight did not succeed; no compute job was launched")


def make_function(stage, volume, timeout):
    import modal
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "numpy>=2.2,<3", "pandas>=2.2,<3", "scikit-learn>=1.6,<2", "xgboost>=3,<4", "joblib>=1.4,<2")
    if stage != "baselines":
        image = image.pip_install("torch>=2.10,<3", "transformers>=5.5,<6", "peft>=0.19,<1",
                                  "accelerate>=1.10,<2", "safetensors>=0.6,<1")
    image = image.add_local_dir(ROOT / "src", "/opt/experiment").env({"PYTHONPATH": "/opt/experiment", "HF_HOME": "/work/model-cache"})
    app = modal.App("gemma-tabular-lab")

    @app.function(image=image, gpu=None if stage == "baselines" else "L40S",
                  cpu=4, memory=32768, timeout=timeout, startup_timeout=300,
                  retries=0, max_containers=1, scaledown_window=0, volumes={"/work": volume})
    def execute(options, run_id):
        import json
        from pathlib import Path
        from gemma_tabular.baselines import run_baselines
        from gemma_tabular.train import train
        output = Path("/work/runs") / run_id
        output.mkdir(parents=True, exist_ok=True)
        try:
            if options["stage"] == "baselines":
                report = run_baselines("/work/data", output, options["seed"])
            else:
                report = train("/work/data", output, head=options["stage"], adaptation=options["adaptation"],
                               model_id=options["model_id"], revision=options["revision"], seed=options["seed"],
                               steps=options["steps"], max_length=options["max_length"],
                               numerical_weight=options["numerical_weight"], max_seconds=options["max_seconds"])
            return {"report": report, "files": [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()]}
        finally:
            # Preserve checkpoints and partial reports even if a later stage fails.
            volume.commit()

    return app, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baselines", "digits", "regression"))
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--adaptation", choices=("frozen", "lora"), default="lora")
    parser.add_argument("--model-id", default="google/gemma-4-E4B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--numerical-weight", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 600 <= args.timeout <= 3600:
        parser.error("A job must be bounded between 600 and 3600 seconds")
    timeout = min(args.timeout, 1200) if args.stage == "baselines" else args.timeout
    maximum = worst_case_cost(gpu=None if args.stage == "baselines" else "L40S", seconds=timeout)
    ledger = Ledger(ROOT / "artifacts" / "budget.sqlite")
    print(json.dumps({"stage": args.stage, "worst_case_job_reservation_usd": maximum, "ledger": ledger.summary()}, indent=2))
    if args.dry_run:
        return
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        if not (args.data / name).is_file():
            raise RuntimeError("Prepared dataset is missing; run data preparation before allocating compute")
    os.environ["MODAL_PROFILE"] = "gemma-tabular-lab"
    preflight()
    import modal
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + args.stage + "-" + uuid.uuid4().hex[:8]
    reservation = ledger.reserve(run_id, maximum, {"gpu": None if args.stage == "baselines" else "L40S", "timeout": timeout})
    try:
        volume = modal.Volume.from_name("gemma-tabular-lab-artifacts", create_if_missing=True)
        with volume.batch_upload(force=True) as upload:
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
                upload.put_file(args.data / name, f"/data/{name}")
        app, function = make_function(args.stage, volume, timeout)
        options = {"stage": args.stage, "adaptation": args.adaptation, "model_id": args.model_id,
                   "revision": args.revision, "seed": args.seed, "steps": args.steps,
                   "max_length": args.max_length, "numerical_weight": args.numerical_weight,
                   "max_seconds": timeout - 60}
        record(f"Reserved ${maximum:.2f} worst-case job charges for `{run_id}`. Actual billed cost is not yet known.")
        with app.run():
            result = function.remote(options, run_id)
        destination = ROOT / "artifacts" / run_id
        destination.mkdir(parents=True, exist_ok=True)
        for relative in result["files"]:
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("Unexpected artifact path")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                for chunk in volume.read_file(f"/runs/{run_id}/{relative}"):
                    handle.write(chunk)
        # Summary is safe to commit; weights, raw data, and per-row predictions remain ignored artifacts.
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        (results_dir / f"{run_id}.json").write_text(json.dumps(result["report"], indent=2) + "\n")
        record(f"Completed `{run_id}`. Metrics: `results/{run_id}.json`. Worst-case allocation remains ${maximum:.2f}; reconcile with actual billing.")
        ledger.complete(reservation, "completed")
    except Exception:
        # Conservative allocation is retained after ambiguous failures; it is never silently refunded.
        ledger.complete(reservation, "failed-allocation-retained")
        record(f"Run `{run_id}` failed; allocation retained. Inspect Modal logs and billing before another launch.")
        raise


if __name__ == "__main__":
    main()
