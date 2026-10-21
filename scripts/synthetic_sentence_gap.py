"""Made-up listings where the sentence, not the brand, sets the price.

Ridge is the same word-and-word-pair model used on Mercari. It is trained on
one set of phrasings and tested on two sets: those same phrasings, and new
phrasings of the same meaning. This is not Mercari and not a Gemma score.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/synthetic-sentence-gap.json"

BRANDS = ("Apple", "Samsung", "Google", "Sony")
GOOD_TRAIN = (
    "no cracks on the screen",
    "screen is not cracked",
    "no scratches at all",
    "flawless screen",
    "glass is perfect",
    "not a single crack",
)
BAD_TRAIN = (
    "has cracks on the screen",
    "screen is cracked",
    "deep scratches",
    "shattered glass",
    "cracked screen",
    "the screen is broken",
)
GOOD_NEW = (
    "the display is unmarked",
    "glass has zero damage",
    "screen looks untouched",
)
BAD_NEW = (
    "the display is smashed",
    "glass is in pieces",
    "screen looks destroyed",
)


def listing(brand, phrase, price, split, meaning):
    return {
        "split": split,
        "meaning": meaning,
        "brand_name": brand,
        "name": f"{brand} phone",
        "item_description": f"{phrase}. Unlocked. Includes a case.",
        "price": price,
    }


def build(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for meaning, phrases, center in (("good", GOOD_TRAIN, 200), ("bad", BAD_TRAIN, 40)):
        for phrase in phrases:
            for brand in BRANDS:
                for _ in range(8):
                    price = float(np.clip(rng.normal(center, 8), 5, 400))
                    rows.append(listing(brand, phrase, price, "train", meaning))
    for meaning, phrases, center in (("good", GOOD_TRAIN, 200), ("bad", BAD_TRAIN, 40)):
        for phrase in phrases:
            for brand in BRANDS:
                price = float(np.clip(rng.normal(center, 8), 5, 400))
                rows.append(listing(brand, phrase, price, "seen_wording", meaning))
    for meaning, phrases, center in (("good", GOOD_NEW, 200), ("bad", BAD_NEW, 40)):
        for phrase in phrases:
            for brand in BRANDS:
                price = float(np.clip(rng.normal(center, 8), 5, 400))
                rows.append(listing(brand, phrase, price, "new_wording", meaning))
    return rows


def fit_predict(rows):
    frame = pd.DataFrame(rows)
    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("features", ColumnTransformer([
                ("brand", OneHotEncoder(handle_unknown="ignore"), ["brand_name"]),
                ("name", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1), "name"),
                ("description", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=1), "item_description"),
            ])),
            ("regressor", Ridge(alpha=1.0, solver="lsqr")),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    train = frame["split"] == "train"
    model.fit(frame.loc[train, ["brand_name", "name", "item_description"]], frame.loc[train, "price"])
    predicted = np.maximum(0, model.predict(frame[["brand_name", "name", "item_description"]]))
    frame["predicted"] = predicted
    return frame


def summarize(frame, split):
    part = frame[frame["split"] == split]
    out = {"n": int(len(part))}
    for meaning in ("good", "bad"):
        block = part[part["meaning"] == meaning]
        out[meaning] = {
            "n": int(len(block)),
            "true_mean": float(block["price"].mean()),
            "ridge_mean": float(block["predicted"].mean()),
            "mae": float((block["price"] - block["predicted"]).abs().mean()),
        }
    out["true_gap"] = out["good"]["true_mean"] - out["bad"]["true_mean"]
    out["ridge_gap"] = out["good"]["ridge_mean"] - out["bad"]["ridge_mean"]
    return out


def main():
    frame = fit_predict(build())
    examples = []
    for split in ("seen_wording", "new_wording"):
        for meaning in ("good", "bad"):
            row = frame[(frame["split"] == split) & (frame["meaning"] == meaning)].iloc[0]
            examples.append({
                "split": split,
                "meaning": meaning,
                "description": row["item_description"],
                "true_price": round(float(row["price"]), 2),
                "ridge_price": round(float(row["predicted"]), 2),
            })
    report = {
        "made_up": True,
        "what_this_is": "Same brand and title. Price is high only when the description says the screen is fine. Ridge trains on one wording and is tested on that wording and on a new wording.",
        "seen_wording": summarize(frame, "seen_wording"),
        "new_wording": summarize(frame, "new_wording"),
        "examples": examples,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
