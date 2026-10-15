import unittest

from gemma_tabular.train import training_batches


class TrainingCoverageTests(unittest.TestCase):
    def test_partial_batch_preserves_every_row_once_per_epoch(self):
        rows = list(range(10))
        batches = training_batches(rows, 4, 42)
        epochs = [[next(batches) for _ in range(3)] for _ in range(2)]
        for epoch in epochs:
            self.assertEqual([len(batch) for batch in epoch], [4, 4, 2])
            self.assertEqual(sorted(row for batch in epoch for row in batch), rows)
        self.assertNotEqual(epochs[0], epochs[1])

    def test_batch_order_is_reproducible(self):
        first = training_batches(list(range(20)), 8, 17)
        second = training_batches(list(range(20)), 8, 17)
        self.assertEqual([next(first) for _ in range(6)], [next(second) for _ in range(6)])

    def test_invalid_inputs_are_rejected(self):
        for rows, size in (([], 1), ([1], 0), ([1], -1)):
            with self.subTest(rows=rows, size=size), self.assertRaises(ValueError):
                next(training_batches(rows, size, 42))
