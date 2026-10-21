"""Join the saved Gemma guesses to the made-up test file, in file order."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemma_tabular.metrics import price_metrics

TEST = ROOT / "data/synthetic-sentence/processed/test.jsonl"
PREDICTIONS = ROOT / "results/kaggle-synthetic-output/synthetic-sentence-predictions.jsonl"
OUTPUT = ROOT / "results/synthetic-sentence-gemma.json"


def block(rows):
    metrics = price_metrics([row["target"] for row in rows], [row["gemma"] for row in rows])
    means = {}
    for meaning in ("good", "bad"):
        chosen = [row for row in rows if row["meaning"] == meaning]
        means[meaning] = {
            "n": len(chosen),
            "true_mean": sum(row["target"] for row in chosen) / len(chosen),
            "gemma_mean": sum(row["gemma"] for row in chosen) / len(chosen),
        }
    metrics["true_gap"] = means["good"]["true_mean"] - means["bad"]["true_mean"]
    metrics["gemma_gap"] = means["good"]["gemma_mean"] - means["bad"]["gemma_mean"]
    metrics["by_meaning"] = means
    return metrics


def main():
    test = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    guesses = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    if len(test) != len(guesses):
        raise SystemExit(f"length mismatch {len(test)} vs {len(guesses)}")
    joined = []
    for row, guess in zip(test, guesses):
        if abs(row["target"] - guess["actual"]) > 1e-6:
            raise SystemExit(f"order mismatch at {row['row_id']}")
        joined.append({**row, "gemma": float(guess["predicted"])})
    groups = {
        "all": block(joined),
        "seen_wording": block([row for row in joined if row["wording"] == "seen"]),
        "new_wording": block([row for row in joined if row["wording"] == "new"]),
    }
    OUTPUT.write_text(json.dumps({"made_up": True, "checkpoint": "lowest validation RMSLE, step 400", "groups": groups}, indent=2) + "\n")
    print(json.dumps({
        name: {"n": item["n"], "mae": round(item["mae"], 2), "true_gap": round(item["true_gap"], 2), "gemma_gap": round(item["gemma_gap"], 2)}
        for name, item in groups.items()
    }, indent=2))


if __name__ == "__main__":
    main()
