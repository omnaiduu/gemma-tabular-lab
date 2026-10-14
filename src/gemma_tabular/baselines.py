"""CPU baselines with validation-only selection and comparable input fields."""

import importlib.metadata
import json
from pathlib import Path
import time

from .data import CATEGORICAL, FEATURES, NUMERIC, TEXT, read_split
from .metrics import price_metrics


def run_baselines(data_dir, output_dir, seed=42):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBRegressor
    from .reduction import AdaptiveSVD

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = {split: read_split(data_dir, split) for split in ("train", "validation", "test")}
    frames = {split: pd.DataFrame([row["features"] for row in rows], columns=FEATURES) for split, rows in records.items()}
    targets = {split: np.array([row["target"] for row in rows]) for split, rows in records.items()}
    for frame in frames.values():
        for column in NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in CATEGORICAL + TEXT:
            frame[column] = frame[column].fillna("").astype(str)
            frame.loc[frame[column].str.strip() == "", column] = "missing"

    def preprocess(text=False, reduced=False):
        transforms = [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)), ("scale", StandardScaler())]), list(NUMERIC)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ]
        if text:
            for column in TEXT:
                vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b\w+\b")
                transform = Pipeline([("tfidf", vectorizer), ("svd", AdaptiveSVD(n_components=16, random_state=seed))]) if reduced else vectorizer
                transforms.append((column, transform, column))
        return ColumnTransformer(transforms, sparse_threshold=1.0)

    def estimator(name, trial):
        if name == "text_ridge":
            learner = Ridge(alpha=(1.0, 10.0)[trial], solver="lsqr")
        else:
            learner = XGBRegressor(n_estimators=300, max_depth=(4, 8)[trial], learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, n_jobs=4, tree_method="hist", random_state=seed)
        transform = preprocess(text=name != "structured_tree", reduced=name == "text_tree")
        pipeline = Pipeline([("features", transform), ("regressor", learner)])
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)

    report = {"seed": seed, "models": {}, "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")}}
    median = float(np.median(targets["train"]))
    report["models"]["median"] = {"test": price_metrics(targets["test"], np.full(len(targets["test"]), median))}
    for name in ("structured_tree", "text_ridge", "text_tree"):
        start = time.perf_counter()
        trials, best, best_error = [], None, float("inf")
        for trial in range(2):
            model = estimator(name, trial)
            model.fit(frames["train"], targets["train"])
            predictions = np.maximum(0, model.predict(frames["validation"]))
            metrics = price_metrics(targets["validation"], predictions)
            trials.append({"trial": trial, "validation": metrics})
            if metrics["rmsle"] < best_error:
                best, best_error = model, metrics["rmsle"]
        training_seconds = time.perf_counter() - start
        # The test set is consulted only after selecting this family's best validation trial.
        predicted = np.maximum(0, best.predict(frames["test"]))
        best.predict(frames["test"].iloc[:1])
        durations = []
        for position in range(min(50, len(frames["test"]))):
            begin = time.perf_counter()
            best.predict(frames["test"].iloc[position:position+1])
            durations.append((time.perf_counter() - begin) * 1000)
        begin = time.perf_counter()
        best.predict(frames["test"])
        batch_seconds = time.perf_counter() - begin
        report["models"][name] = {
            "validation_trials": trials, "test": price_metrics(targets["test"], predicted),
            "tuning_and_training_seconds": training_seconds,
            "warm_batch1_p50_ms": float(np.percentile(durations, 50)),
            "warm_batch1_p95_ms": float(np.percentile(durations, 95)),
            "batch_rows_per_second": len(predicted) / max(batch_seconds, 1e-9),
            "device": "CPU", "requested_cores": 4,
        }
        joblib.dump(best, output / f"{name}.joblib")
        with (output / f"{name}-predictions.jsonl").open("w") as handle:
            for row, prediction in zip(records["test"], predicted):
                handle.write(json.dumps({"row_id": row["row_id"], "actual": row["target"], "predicted": float(prediction),
                                         "category": row["features"]["category_name"]}) + "\n")
        (output / "baseline-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
