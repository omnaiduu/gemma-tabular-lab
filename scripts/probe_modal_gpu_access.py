"""Check cheaper-GPU eligibility; optionally run a bounded hardware query."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemma_tabular.budget import Ledger, worst_case_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("T4", "L4"), action="append")
    parser.add_argument("--run", action="store_true", help="If registration succeeds, query the GPU once")
    args = parser.parse_args()
    os.environ["MODAL_PROFILE"] = "gemma-tabular-lab"
    import modal

    ledger = Ledger(ROOT / "artifacts/budget.sqlite")
    report = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "profile": "gemma-tabular-lab", "checks": [],
              "sources": ["https://modal.com/docs/guide/gpu", "https://modal.com/pricing"]}
    destination = ROOT / "results/gpu-access-recheck.json"
    destination.parent.mkdir(exist_ok=True)
    for gpu in args.gpu or ["T4", "L4"]:
        maximum = worst_case_cost(gpu, seconds=30, startup_seconds=60, cpu=0.25, memory_gib=0.5)
        name = "gpu-access-" + gpu + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        reservation = ledger.reserve(name, maximum, {"gpu": gpu, "purpose": "Account eligibility recheck",
                                                    "timeout": 30, "startup_timeout": 60})
        item = {"gpu": gpu, "maximum_allocation_usd": maximum, "function_registered": False,
                "worker_executed": False}
        try:
            app = modal.App("gemma-tabular-gpu-access-check")

            @app.function(image=modal.Image.debian_slim(python_version="3.12"), gpu=gpu,
                          cpu=(0.25, 0.25), memory=(512, 512), timeout=30,
                          startup_timeout=60, min_containers=0, buffer_containers=0,
                          max_containers=1, scaledown_window=2, retries=0,
                          serialized=True, include_source=False)
            def hardware_query():
                import subprocess
                result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                         "--format=csv,noheader"], capture_output=True,
                                        text=True, timeout=10)
                if result.returncode:
                    raise RuntimeError("GPU hardware query failed")
                return result.stdout.strip()

            with modal.enable_output():
                with app.run():
                    item["function_registered"] = True
                    if args.run:
                        item["hardware"] = hardware_query.remote()
                        item["worker_executed"] = True
            item["status"] = "hardware-query-succeeded" if args.run else "registered-without-execution"
            ledger.complete(reservation, "completed")
        except Exception as error:
            message = re.sub(r"\b(?:ak|as)-[A-Za-z0-9]+", "[REDACTED]", str(error))
            item.update(status="rejected-or-failed", error_type=type(error).__name__, error=message[:1200])
            ledger.complete(reservation, "failed-allocation-retained")
        report["checks"].append(item)
        report["budget"] = ledger.summary()
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(item, indent=2), flush=True)


if __name__ == "__main__":
    main()
