import csv
import math
from pathlib import Path
import tempfile
import unittest

from gemma_tabular.budget import Ledger, worst_case_cost
from gemma_tabular.codec import SLOT_SIZES, decode, encode, format_slots
from gemma_tabular.data import FEATURES, fingerprint, prepare, read_split, serialize
from gemma_tabular.metrics import price_metrics


class CodecTests(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(encode(18), (0, 0, 1, 8))
        self.assertEqual(format_slots(encode(18)), "0018")
        self.assertEqual(decode(encode(400)), 400)
        self.assertEqual(encode(1100), (1, 1, 0, 0))
        self.assertEqual(decode(encode(10.5)), 11)

    def test_zero(self):
        self.assertEqual(encode(0), (0, 0, 0, 0))
        self.assertEqual(encode(-0.0), (0, 0, 0, 0))
        self.assertEqual(decode((0, 0, 0, 0)), 0.0)

    def test_prepared_price_range_round_trips(self):
        for value in (0, 3, 10, 18, 99, 100, 400, 409, 999, 1009, 1100, 9999):
            self.assertEqual(decode(encode(value)), value)

    def test_invalid_numbers_and_overflow_after_rounding(self):
        for value in (math.nan, math.inf, -math.inf, -1, 10000, 9999.5):
            with self.assertRaises(ValueError):
                encode(value)

    def test_invalid_slots(self):
        for slots in ((0, 0, 0), (10, 0, 0, 0), (-1, 0, 0, 0), (0, 0, 0, True)):
            with self.assertRaises(ValueError):
                decode(slots)
        self.assertEqual(sum(SLOT_SIZES), 40)


class DataTests(unittest.TestCase):
    def test_target_and_unknown_columns_are_excluded(self):
        row = {"name": "chair", "price": "987654321", "future_revenue": "secret-target"}
        text = serialize(row)
        self.assertNotIn("987654321", text)
        self.assertNotIn("secret-target", text)
        self.assertEqual(fingerprint(row), fingerprint({"name": "CHAIR"}))

    def test_group_splits_and_holdouts_remain_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.tsv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[*FEATURES, "price"], delimiter="\t")
                writer.writeheader()
                for index in range(500):
                    row = {key: "" for key in FEATURES}
                    row.update(name=f"item {index//2}", category_name="test", price=index+1)
                    writer.writerow(row)
            first = prepare(source, root / "first", 20, 20)
            second = prepare(source, root / "second", 50, 20)
            groups = {split: {r["group"] for r in read_split(root / "second", split)} for split in ("train", "validation", "test")}
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["test"] & groups["validation"])
            for split in ("validation", "test"):
                self.assertEqual(read_split(root / "first", split), read_split(root / "second", split))
            self.assertEqual(first["source_sha256"], second["source_sha256"])

    def test_missing_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.csv"
            source.write_text("name,price\nchair,20\n")
            with self.assertRaises(ValueError):
                prepare(source, Path(directory) / "out")


class BudgetTests(unittest.TestCase):
    def test_ceiling_and_no_automatic_refund(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.sqlite")
            job = ledger.reserve("first", 20)
            ledger.complete(job, "completed")
            with self.assertRaises(ValueError):
                ledger.reserve("second", 6)
            self.assertEqual(ledger.summary()["allocated_worst_case_usd"], 20)
            self.assertIsNone(ledger.summary()["actual_billed_spend_usd"])

    def test_unresolved_launch_blocks_concurrent_job(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.sqlite")
            ledger.reserve("first", 1)
            with self.assertRaises(ValueError):
                ledger.reserve("parallel", 1)

    def test_invalid_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.sqlite")
            for amount in (-1, 0, math.nan, math.inf, 26):
                with self.assertRaises(ValueError):
                    ledger.reserve("invalid", amount)

    def test_gpu_pricing_includes_resources_and_margin(self):
        self.assertGreater(worst_case_cost("L40S", 3600), 0.000542 * 3600)
        with self.assertRaises(ValueError):
            worst_case_cost("unpriced", 3600)


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        metrics = price_metrics([0, 10, 100], [0, 10, 100])
        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["rmsle"], 0)

    def test_large_mistakes(self):
        metrics = price_metrics([100, 200], [100, 500])
        self.assertEqual(metrics["mae"], 150)
        self.assertEqual(metrics["factor_two_error_rate"], 0.5)

    def test_invalid_predictions(self):
        for predictions in ([], [-1], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                price_metrics([1], predictions)


if __name__ == "__main__":
    unittest.main()
