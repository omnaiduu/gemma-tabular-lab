"""Fit the Mercari Ridge recipe on the made-up sentence shop and score it."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemma_tabular.data import CATEGORICAL, FEATURES, NUMERIC, TEXT, read_split
from gemma_tabular.metrics import price_metrics

DATA = ROOT / "data/synthetic-sentence/processed"
OUTPUT = ROOT / "results/synthetic-sentence-ridge.json"


def frame(rows):
    table = pd.DataFrame([row["features"] for row in rows], columns=FEATURES)
    for column in NUMERIC:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    for column in CATEGORICAL + TEXT:
        table[column] = table[column].fillna("").astype(str)
    return table


def model(alpha):
    transforms = [
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
        ]), list(NUMERIC)),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ]
    for column in TEXT:
        transforms.append((column, TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b",
        ), column))
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("features", ColumnTransformer(transforms, sparse_threshold=1.0)),
            ("regressor", Ridge(alpha=alpha, solver="lsqr")),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def block(rows, predicted):
    metrics = price_metrics([row["target"] for row in rows], predicted)
    by_meaning = {}
    for meaning in ("good", "bad"):
        chosen = [(row, guess) for row, guess in zip(rows, predicted) if row["meaning"] == meaning]
        by_meaning[meaning] = {
            "n": len(chosen),
            "true_mean": float(np.mean([row["target"] for row, _ in chosen])),
            "ridge_mean": float(np.mean([guess for _, guess in chosen])),
        }
    metrics["true_gap"] = by_meaning["good"]["true_mean"] - by_meaning["bad"]["true_mean"]
    metrics["ridge_gap"] = by_meaning["good"]["ridge_mean"] - by_meaning["bad"]["ridge_mean"]
    metrics["by_meaning"] = by_meaning
    return metrics


def main():
    rows = {name: read_split(DATA, name) for name in ("train", "validation", "test")}
    best, best_error, chosen = None, float("inf"), None
    trials = []
    for alpha in (1.0, 10.0):
        fitted = model(alpha)
        fitted.fit(frame(rows["train"]), np.array([row["target"] for row in rows["train"]]))
        guesses = np.maximum(0, fitted.predict(frame(rows["validation"])))
        metrics = price_metrics([row["target"] for row in rows["validation"]], guesses)
        trials.append({"alpha": alpha, "validation": metrics})
        if metrics["rmsle"] < best_error:
            best, best_error, chosen = fitted, metrics["rmsle"], alpha
    test_guesses = [float(value) for value in np.maximum(0, best.predict(frame(rows["test"])))]
    groups = {
        "all": block(rows["test"], test_guesses),
        "seen_wording": block(
            [row for row in rows["test"] if row["wording"] == "seen"],
            [guess for row, guess in zip(rows["test"], test_guesses) if row["wording"] == "seen"],
        ),
        "new_wording": block(
            [row for row in rows["test"] if row["wording"] == "new"],
            [guess for row, guess in zip(rows["test"], test_guesses) if row["wording"] == "new"],
        ),
    }
    predictions = [
        {"row_id": row["row_id"], "actual": row["target"], "predicted": guess, "wording": row["wording"], "meaning": row["meaning"]}
        for row, guess in zip(rows["test"], test_guesses)
    ]
    report = {"made_up": True, "alpha": chosen, "validation_trials": trials, "groups": groups}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT.with_name("synthetic-sentence-ridge-predictions.jsonl")).write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    print(json.dumps({
        "alpha": chosen,
        "groups": {name: {"n": item["n"], "mae": round(item["mae"], 2), "true_gap": round(item["true_gap"], 2), "ridge_gap": round(item["ridge_gap"], 2)} for name, item in groups.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
