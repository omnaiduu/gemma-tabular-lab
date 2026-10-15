"""Acquire the published Mercari 100K subset without redistributing its rows."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = "https://www.openml.org/api/v1/json/data/46660"
PARQUET = "https://data.openml.org/datasets/0004/46660/dataset_46660.pq"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    import numpy as np
    import pandas as pd

    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DESCRIPTION, timeout=30) as response:
        metadata = json.load(response)["data_set_description"]
    if metadata["name"] != "mercari_price_suggestion100K" or str(metadata["version"]) != "1":
        raise RuntimeError("Unexpected dataset identity")
    parquet = raw / "mercari-openml-46660.parquet"
    if not parquet.exists():
        temporary = parquet.with_suffix(".partial")
        try:
            with urllib.request.urlopen(PARQUET, timeout=30) as response, temporary.open("wb") as output:
                size = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size > 200 * 1024 * 1024:
                        raise RuntimeError("Dataset exceeds the expected acquisition size")
                    output.write(block)
            temporary.replace(parquet)
        finally:
            temporary.unlink(missing_ok=True)
    frame = pd.read_parquet(parquet)
    fields = ["train_id", "name", "item_description", "category_name", "brand_name",
              "item_condition_id", "shipping", "price"]
    if len(frame) != 100000 or not set(fields + ["log_price"]).issubset(frame.columns):
        raise RuntimeError("Unexpected dataset schema or row count")
    price = pd.to_numeric(frame["price"]).to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price < 0).any():
        raise RuntimeError("Invalid original price values")
    log_error = float(np.max(np.abs(np.log1p(price) - frame["log_price"].to_numpy(dtype=float))))
    if log_error > 1e-5:
        raise RuntimeError("Original price and published log1p target disagree")
    # Keep the original price. Neither the published log target nor derived
    # category columns are allowed into the serialized prediction input.
    table = raw / "train.tsv"
    frame[fields].to_csv(table, sep="\t", index=False)
    provenance = {
        "dataset": metadata["name"], "openml_id": 46660, "version": 1,
        "description_url": DESCRIPTION, "download_url": PARQUET,
        "original_source": metadata.get("original_data_url"),
        "declared_licence": metadata.get("licence"),
        "usage": "Research pilot; original Kaggle terms apply; source rows remain outside Git",
        "rows": len(frame), "columns_exported": fields,
        "target": "Original price column; no inverse log transformation applied",
        "max_abs_log1p_consistency_error": log_error,
        "source_parquet_sha256": digest(parquet), "exported_tsv_sha256": digest(table),
        "subset_limit": "Published 100K research subset, not the full Kaggle competition training set",
    }
    (raw / "source-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    destination = ROOT / "results/data-provenance.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
