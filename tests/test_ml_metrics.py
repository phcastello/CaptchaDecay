"""Greedy CTC ordering and dataset-wide string metrics."""

import unittest

import torch

from ml.charset import Charset
from ml.decode import greedy_decode
from ml.metrics import MetricAccumulator, levenshtein


class GreedyDecodeTests(unittest.TestCase):
    def setUp(self):
        self.charset = Charset("ab7")

    def logits_for_paths(self, paths):
        ids = torch.tensor(paths, dtype=torch.long).transpose(0, 1)
        return torch.nn.functional.one_hot(ids, num_classes=self.charset.num_classes).float() * 10

    def test_collapse_before_blank_removal(self):
        paths = [[1, 1, 0, 2, 2], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [3, 3, 3, 3, 3]]
        self.assertEqual(greedy_decode(self.logits_for_paths(paths), self.charset), ["ab", "aa", "", "7"])

    def test_blanks_at_boundaries_and_separated_repeated_characters(self):
        paths = [[0, 1, 0, 1, 1, 0, 1, 0], [0, 2, 1, 3, 0, 2, 1, 0]]
        self.assertEqual(greedy_decode(self.logits_for_paths(paths), self.charset), ["aaa", "ba7ba"])

    def test_decoding_validates_shape_classes_and_finite_values(self):
        for logits in (torch.zeros(2, 3), torch.zeros(0, 1, 4), torch.zeros(2, 0, 4)):
            with self.subTest(shape=tuple(logits.shape)), self.assertRaisesRegex(ValueError, "Decoder expects"):
                greedy_decode(logits, self.charset)
        with self.assertRaisesRegex(ValueError, "classes"):
            greedy_decode(torch.zeros(2, 1, 5), self.charset)
        with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
            greedy_decode(torch.full((2, 1, 4), float("nan")), self.charset)


class MetricTests(unittest.TestCase):
    def test_levenshtein_substitutions_deletions_insertions_and_empty(self):
        for expected, predicted, distance in (("abc", "abc", 0), ("abc", "axc", 1),
                                              ("abc", "ac", 1), ("abc", "abxc", 1),
                                              ("kitten", "sitting", 3), ("", "abc", 3),
                                              ("abc", "", 3), ("", "", 0), ("aA", "aa", 1)):
            with self.subTest(expected=expected, predicted=predicted):
                self.assertEqual(levenshtein(expected, predicted), distance)
                self.assertEqual(levenshtein(predicted, expected), distance)

    def test_weighted_cer_and_exact_match_cover_all_batches(self):
        metrics = MetricAccumulator()
        metrics.update(["abcd", "abcde"], ["abcd", "abXde"])
        metrics.update(["abcdef"], ["abcde"])
        result = metrics.compute()
        self.assertEqual(result["num_images"], 3)
        self.assertAlmostEqual(result["exact_match"], 1 / 3)
        self.assertAlmostEqual(result["cer"], 2 / 15)
        self.assertNotAlmostEqual(result["cer"], (0 + 1 / 5 + 1 / 6) / 3)
        self.assertEqual(result["by_length"], {
            "4": {"exact_match": 1.0, "num_images": 1},
            "5": {"exact_match": 0.0, "num_images": 1},
            "6": {"exact_match": 0.0, "num_images": 1},
        })
        metrics.update(["zzzz"], ["zzzz"])
        self.assertEqual(metrics.compute()["by_length"]["4"]["num_images"], 2)

    def test_insertions_can_produce_cer_above_one(self):
        metrics = MetricAccumulator()
        metrics.update(["a"], ["abcd"])
        self.assertEqual(metrics.compute()["cer"], 3.0)
        self.assertEqual(metrics.compute()["exact_match"], 0.0)

    def test_empty_predictions_are_deletions_and_case_is_preserved(self):
        metrics = MetricAccumulator()
        metrics.update(["aA", "abc"], ["aa", ""])
        self.assertAlmostEqual(metrics.compute()["cer"], 4 / 5)
        self.assertEqual(metrics.compute()["exact_match"], 0.0)

    def test_invalid_updates_do_not_partially_change_counters(self):
        metrics = MetricAccumulator()
        self.assertEqual(metrics.compute(), {"exact_match": 0.0, "cer": 0.0, "num_images": 0, "by_length": {}})
        for expected, predicted in ((["a"], []), ("a", "a"), (["a", 2], ["a", "b"])):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                metrics.update(expected, predicted)
            self.assertEqual(metrics.num_images, 0)
        metrics.update([""], [""])
        self.assertEqual(metrics.compute()["exact_match"], 1.0)
        metrics.update([""], ["a"])
        with self.assertRaisesRegex(ValueError, "CER is undefined"):
            metrics.compute()


if __name__ == "__main__":
    unittest.main()
