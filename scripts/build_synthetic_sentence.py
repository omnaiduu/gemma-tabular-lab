"""Build a made-up shop where only the sentence sets the price.

1,000 training rows, 200 validation rows, and 500 test rows. The test set is
half wording the models have seen and half wording they have not. Brand,
product, condition, and shipping are the same for a fine item and a broken
item. Prices are drawn, not copied from Mercari.
"""

import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gemma_tabular.data import serialize

OUT = ROOT / "data/synthetic-sentence/processed"
PRODUCTS = ("phone", "laptop", "camera", "headphones", "watch")
BRANDS = ("Apple", "Samsung", "Google", "Sony")
STOP = {"a", "an", "the", "is", "are", "on", "in", "at", "of", "and", "with", "to", "it", "for", "all"}
FILLER = "No box included."
GOOD_TRAIN = (
    "no cracks on the screen",
    "screen is not cracked",
    "no scratches at all",
    "flawless screen",
    "glass is perfect",
    "not a single crack",
    "mint condition screen",
    "undamaged glass",
    "clean screen",
    "like new screen",
)
BAD_TRAIN = (
    "has cracks on the screen",
    "screen is cracked",
    "deep scratches",
    "shattered glass",
    "cracked screen",
    "the screen is broken",
    "spiderweb crack",
    "chipped glass",
    "fractured screen",
    "split glass",
)
GOOD_NEW = (
    "the display is unmarked",
    "panel shows zero damage",
    "face looks untouched",
    "immaculate front",
    "spotless and blemish free",
)
BAD_NEW = (
    "the display is smashed",
    "panel is in pieces",
    "face looks destroyed",
    "ruined front",
    "wrecked and crushed",
)


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def assert_disjoint():
    seen = set()
    for phrase in GOOD_TRAIN + BAD_TRAIN:
        seen.update(tokens(phrase))
    fresh = set()
    for phrase in GOOD_NEW + BAD_NEW:
        fresh.update(tokens(phrase))
    overlap = sorted(seen & fresh)
    if overlap:
        raise SystemExit(f"New wording reuses training words: {overlap}")


def price(rng, meaning):
    center, spread = (200.0, 8.0) if meaning == "good" else (40.0, 6.0)
    return float(round(float(np.clip(rng.normal(center, spread), 5, 400))))


def row(rng, index, split, product, phrase, meaning, wording):
    brand = BRANDS[index % len(BRANDS)]
    features = {
        "name": f"{brand} {product}",
        "item_description": f"{phrase}. {FILLER}",
        "category_name": product,
        "brand_name": brand,
        "item_condition_id": "3",
        "shipping": "0",
    }
    return {
        "features": features,
        "target": price(rng, meaning),
        "group": f"{split}-{index}",
        "row_id": f"{split}-{index:04d}",
        "text": serialize(features),
        "meaning": meaning,
        "wording": wording,
        "phrase": phrase,
    }


def expand(rng, split, phrases, meaning, wording, copies, start):
    rows = []
    index = start
    for _ in range(copies):
        for product in PRODUCTS:
            for phrase in phrases:
                rows.append(row(rng, index, split, product, phrase, meaning, wording))
                index += 1
    return rows, index


def write(name, rows):
    path = OUT / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main():
    assert_disjoint()
    if (OUT / "train.jsonl").exists() and "--force" not in sys.argv:
        raise SystemExit("Refusing to overwrite the scored synthetic split. Pass --force to rebuild.")
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    train, cursor = expand(rng, "train", GOOD_TRAIN, "good", "seen", 10, 0)
    more, cursor = expand(rng, "train", BAD_TRAIN, "bad", "seen", 10, cursor)
    train += more
    validation, cursor = expand(rng, "validation", GOOD_TRAIN, "good", "seen", 2, 0)
    more, cursor = expand(rng, "validation", BAD_TRAIN, "bad", "seen", 2, cursor)
    validation += more
    seen_test, cursor = expand(rng, "test-seen", GOOD_TRAIN[:5], "good", "seen", 5, 0)
    more, cursor = expand(rng, "test-seen", BAD_TRAIN[:5], "bad", "seen", 5, cursor)
    seen_test += more
    new_test, cursor = expand(rng, "test-new", GOOD_NEW, "good", "new", 5, 0)
    more, _ = expand(rng, "test-new", BAD_NEW, "bad", "new", 5, cursor)
    new_test += more
    test = seen_test + new_test
    counts = {"train": len(train), "validation": len(validation), "test": len(test)}
    if counts != {"train": 1000, "validation": 200, "test": 500}:
        raise SystemExit(counts)
    write("train", train)
    write("validation", validation)
    write("test", test)
    manifest = {
        "made_up": True,
        "counts": counts,
        "test_seen": sum(row["wording"] == "seen" for row in test),
        "test_new": sum(row["wording"] == "new" for row in test),
        "good_train_phrases": GOOD_TRAIN,
        "bad_train_phrases": BAD_TRAIN,
        "good_new_phrases": GOOD_NEW,
        "bad_new_phrases": BAD_NEW,
        "filler": FILLER,
        "condition_and_shipping": "identical on every row, so they cannot explain the price",
        "seed": 42,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
