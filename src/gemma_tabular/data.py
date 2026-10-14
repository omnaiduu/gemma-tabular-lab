"""Allowlisted inputs and duplicate-grouped train/validation/test splits."""

import csv
import hashlib
import html
import json
import math
from pathlib import Path
import random
import re

FEATURES = ("name", "item_description", "category_name", "brand_name", "item_condition_id", "shipping")
NUMERIC = ("item_condition_id", "shipping")
CATEGORICAL = ("category_name", "brand_name")
TEXT = ("name", "item_description")


def clean(value):
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def feature_record(row):
    return {key: clean(row.get(key, "")) for key in FEATURES}


def fingerprint(row):
    normalized = {key: value.casefold() for key, value in feature_record(row).items()}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def serialize(row):
    # Only the allowlist is serialized. Prices and arbitrary extra columns never enter the prompt.
    features = feature_record(row)
    structured = [f"{key}: {features[key] or '[missing]'}" for key in FEATURES if key not in TEXT]
    prose = [f"{key}: {features[key] or '[missing]'}" for key in TEXT]
    return "Estimate the listing sale price in USD.\n" + "\n".join(structured + prose) + "\nPredicted price:"


def assign_split(group_key, seed=42):
    # Hashing groups before sampling keeps the holdout fixed when training size increases.
    value = int(hashlib.sha256(f"{seed}:{group_key}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


def prepare(source, destination, train_limit=10000, holdout_limit=2000, seed=42):
    source, destination = Path(source), Path(destination)
    if train_limit <= 0 or holdout_limit <= 0:
        raise ValueError("Sample limits must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    partitions = {name: [] for name in ("train", "validation", "test")}
    seen = {name: 0 for name in partitions}
    limits = {"train": train_limit, "validation": holdout_limit, "test": holdout_limit}
    rng = {name: random.Random(f"{seed}:{name}") for name in partitions}
    skipped = total = 0
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {*FEATURES, "price"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            total += 1
            try:
                target = float(raw["price"])
            except (ValueError, TypeError):
                skipped += 1
                continue
            if not math.isfinite(target) or target < 0:
                skipped += 1
                continue
            row = feature_record(raw)
            group = fingerprint(row)
            partition = assign_split(group, seed)
            record = {"features": row, "target": target, "group": group,
                      "row_id": str(raw.get("train_id", total)), "text": serialize(row)}
            seen[partition] += 1
            bucket, limit = partitions[partition], limits[partition]
            if len(bucket) < limit:
                bucket.append(record)
            else:
                position = rng[partition].randrange(seen[partition])
                if position < limit:
                    bucket[position] = record
    if any(not rows for rows in partitions.values()):
        raise ValueError("Each split must contain at least one valid example")
    groups = {name: {row["group"] for row in rows} for name, rows in partitions.items()}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Duplicate groups overlap across splits")
    for name, rows in partitions.items():
        rows.sort(key=lambda row: row["row_id"])
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    manifest = {"source_sha256": digest.hexdigest(), "seed": seed,
                "total_source_rows": total, "skipped_invalid_targets": skipped,
                "eligible_split_counts": seen, "sampled_split_counts": {k: len(v) for k, v in partitions.items()},
                "feature_allowlist": FEATURES, "target": "price", "duplicate_policy": "normalized-feature hash groups"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_split(directory, split):
    with (Path(directory) / f"{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
