"""Score the saved E2B guesses and a fresh Ridge fit on text-only slices.

Group labels use the listing text and the training vocabulary. They do not
use either model's error. Rules were fixed before the test scores were read.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gemma_tabular.data import CATEGORICAL, FEATURES, NUMERIC, TEXT, read_split
from gemma_tabular.metrics import price_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/kaggle-pilot-output/gemma-tabular-lab/data/processed"
GEMMA = ROOT / "results/kaggle-pilot-output/e2b-regression/predictions.jsonl"
CHECKSUMS = ROOT / "results/processed-checksums.json"
OUTPUT = ROOT / "results/mercari-sentence-slice.json"

TOKEN = re.compile(r"(?u)\b\w+\b")
CONDITION = re.compile(
    r"(?i)\b(?:no|not|never|without|n'?t)\b(?:\W+\w+){0,5}\W+"
    r"\b(?:stains?|cracks?|scratches?|rips?|holes?|damages?|tears?|flaws?|wear|broken|works?|working|tags?|box)\b"
    r"|\b(?:but|however)\b(?:\W+\w+){0,8}\W+"
    r"\b(?:stains?|cracks?|scratches?|damages?|wear|broken|dies|dead|works?|missing)\b"
    r"|\b(?:missing|broken|does not work|doesn't work|not working|for parts)\b",
)
BUNDLE = re.compile(r"(?i)\b(?:bundle|bundled|lot of|set of|comes with)\b")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def words(row):
    features = row["features"]
    return TOKEN.findall(f"{features['name']} {features['item_description']}".lower())


def labels(row, train_words):
    text = f"{row['features']['name']} {row['features']['item_description']}"
    listing_words = words(row)
    reasons = []
    if CONDITION.search(text):
        reasons.append("condition_sentence")
    if BUNDLE.search(text):
        reasons.append("bundle")
    if len(listing_words) > 80:
        reasons.append("long_description")
    if sum(word not in train_words for word in set(listing_words)) >= 5:
        reasons.append("unseen_words")
    return reasons


def frames(records):
    frame = pd.DataFrame([row["features"] for row in records], columns=FEATURES)
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL + TEXT:
        frame[column] = frame[column].fillna("").astype(str)
        frame.loc[frame[column].str.strip() == "", column] = "missing"
    return frame


def ridge():
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
    pipeline = Pipeline([
        ("features", ColumnTransformer(transforms, sparse_threshold=1.0)),
        ("regressor", Ridge(alpha=1.0, solver="lsqr")),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def subset(rows, mask):
    chosen = [row for row, keep in zip(rows, mask) if keep]
    return {
        "n": len(chosen),
        "ridge": price_metrics([row["actual"] for row in chosen], [row["ridge"] for row in chosen]),
        "gemma_e2b": price_metrics([row["actual"] for row in chosen], [row["gemma"] for row in chosen]),
    }


def main():
    expected = json.loads(CHECKSUMS.read_text())["files"]
    for name, digest in expected.items():
        actual = sha256(DATA / name)
        if actual != digest:
            raise SystemExit(f"{name} checksum {actual} does not match the scored split {digest}")

    records = {split: read_split(DATA, split) for split in ("train", "validation", "test")}
    train_words = set()
    for row in records["train"]:
        train_words.update(words(row))

    gemma = {}
    with GEMMA.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            gemma[str(row["row_id"])] = float(row["predicted"])
    missing = [row["row_id"] for row in records["test"] if str(row["row_id"]) not in gemma]
    if missing:
        raise SystemExit(f"{len(missing)} test rows have no saved Gemma prediction")

    best, best_error, chosen_alpha = None, float("inf"), None
    train_x, train_y = frames(records["train"]), np.array([row["target"] for row in records["train"]])
    valid_x = frames(records["validation"])
    valid_y = np.array([row["target"] for row in records["validation"]])
    for alpha in (1.0, 10.0):
        model = ridge()
        model.regressor.set_params(regressor__alpha=alpha)
        model.fit(train_x, train_y)
        metrics = price_metrics(valid_y, np.maximum(0, model.predict(valid_x)))
        if metrics["rmsle"] < best_error:
            best, best_error, chosen_alpha = model, metrics["rmsle"], alpha

    predicted = np.maximum(0, best.predict(frames(records["test"])))
    scored = []
    for row, prediction in zip(records["test"], predicted):
        reasons = labels(row, train_words)
        scored.append({
            "row_id": row["row_id"],
            "actual": row["target"],
            "ridge": float(prediction),
            "gemma": gemma[str(row["row_id"])],
            "reasons": reasons,
        })

    groups = {
        "all": subset(scored, [True] * len(scored)),
        "ordinary": subset(scored, [not row["reasons"] for row in scored]),
        "sentence": subset(scored, [bool(row["reasons"]) for row in scored]),
    }
    for reason in ("condition_sentence", "bundle", "long_description", "unseen_words"):
        groups[reason] = subset(scored, [reason in row["reasons"] for row in scored])

    def gap(block):
        return {
            "mae_gemma_minus_ridge": block["gemma_e2b"]["mae"] - block["ridge"]["mae"],
            "rmsle_gemma_minus_ridge": block["gemma_e2b"]["rmsle"] - block["ridge"]["rmsle"],
        }

    report = {
        "data_checksums": expected,
        "gemma_predictions": str(GEMMA.relative_to(ROOT)),
        "ridge": {"alpha": chosen_alpha, "validation_rmsle": best_error},
        "rules": {
            "condition_sentence": "negation or contrast near a defect word, or missing/broken/not working/for parts",
            "bundle": "bundle, bundled, lot of, set of, or comes with",
            "long_description": "more than 80 words in the title plus description",
            "unseen_words": "at least five words absent from the 10,000 training listings",
            "sentence": "any of the four rules",
            "ordinary": "none of the four rules",
            "label_source": "text and training vocabulary only; model error was not used",
        },
        "groups": groups,
        "gap": {name: gap(block) for name, block in groups.items()},
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"alpha": chosen_alpha, "groups": {
        name: {"n": block["n"], "ridge_mae": round(block["ridge"]["mae"], 2), "gemma_mae": round(block["gemma_e2b"]["mae"], 2),
               "gap": round(report["gap"][name]["mae_gemma_minus_ridge"], 2)}
        for name, block in groups.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
