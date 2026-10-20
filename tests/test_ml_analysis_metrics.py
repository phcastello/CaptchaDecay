"""Alignment and diagnostic aggregation use artificial in-memory strings only."""

from copy import deepcopy
from itertools import product
import json
import unittest

from ml.analysis_metrics import analyze_predictions, has_consecutive_repeats
from ml.metrics import MetricAccumulator, align_edits, levenshtein


class EditAlignmentTests(unittest.TestCase):
    def test_exact_empty_and_single_operation_alignments(self):
        self.assertEqual(align_edits("", ""), [])
        self.assertEqual(align_edits("a", "a"), [{"operation": "match", "expected": "a", "predicted": "a"}])
        self.assertEqual(align_edits("a", "b"), [{"operation": "substitution", "expected": "a", "predicted": "b"}])
        self.assertEqual(align_edits("a", ""), [{"operation": "deletion", "expected": "a", "predicted": ""}])
        self.assertEqual(align_edits("", "b"), [{"operation": "insertion", "expected": "", "predicted": "b"}])

    def test_documented_backtrace_tie_break_is_deterministic(self):
        self.assertEqual(align_edits("ab", "ba"), [
            {"operation": "substitution", "expected": "a", "predicted": "b"},
            {"operation": "substitution", "expected": "b", "predicted": "a"},
        ])
        self.assertEqual(align_edits("aa", "a"), [
            {"operation": "deletion", "expected": "a", "predicted": ""},
            {"operation": "match", "expected": "a", "predicted": "a"},
        ])
        self.assertEqual(align_edits("aba", "bab"), [
            {"operation": "insertion", "expected": "", "predicted": "b"},
            {"operation": "match", "expected": "a", "predicted": "a"},
            {"operation": "match", "expected": "b", "predicted": "b"},
            {"operation": "deletion", "expected": "a", "predicted": ""},
        ])

    def test_exhaustive_short_strings_reconstruct_text_and_match_existing_distance(self):
        strings = ["".join(chars) for length in range(5) for chars in product("ab", repeat=length)]
        for expected, predicted in product(strings, repeat=2):
            with self.subTest(expected=expected, predicted=predicted):
                aligned = align_edits(expected, predicted)
                self.assertEqual("".join(op["expected"] for op in aligned), expected)
                self.assertEqual("".join(op["predicted"] for op in aligned), predicted)
                self.assertEqual(sum(op["operation"] != "match" for op in aligned), levenshtein(expected, predicted))
                self.assertEqual(aligned, align_edits(expected, predicted))
                for operation in aligned:
                    self.assertEqual(set(operation), {"operation", "expected", "predicted"})
                    name, left, right = operation["operation"], operation["expected"], operation["predicted"]
                    if name == "match":
                        self.assertEqual(left, right)
                        self.assertEqual(len(left), 1)
                    elif name == "substitution":
                        self.assertNotEqual(left, right)
                        self.assertEqual((len(left), len(right)), (1, 1))
                    elif name == "deletion":
                        self.assertEqual((len(left), right), (1, ""))
                    else:
                        self.assertEqual(name, "insertion")
                        self.assertEqual((left, len(right)), ("", 1))

    def test_alignment_preserves_unicode_case_and_validates_types(self):
        aligned = align_edits("aAé", "aaê")
        self.assertEqual([operation["operation"] for operation in aligned], ["match", "substitution", "substitution"])
        self.assertEqual("".join(operation["expected"] for operation in aligned), "aAé")
        for expected, predicted in ((None, "a"), ("a", 1), (["a"], "a")):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "strings"):
                align_edits(expected, predicted)


class PredictionAnalysisTests(unittest.TestCase):
    def setUp(self):
        pairs = (("abcd", "abcd"), ("abcd", "abXd"), ("aabc", "abc"),
                 ("abde", "abXde"), ("abcde", "abXde"), ("abcdef", "abdeQ"),
                 ("aa", ""), ("b", "xxxx"))
        self.predictions = [
            {"filename": f"{index}.png", "expected": expected, "predicted": predicted,
             "correct": expected == predicted}
            for index, (expected, predicted) in enumerate(pairs)
        ]

    def test_rows_have_exact_schema_and_input_is_unchanged(self):
        original = deepcopy(self.predictions)
        rows, summary = analyze_predictions(self.predictions)
        self.assertEqual(original, self.predictions)
        schema = {"filename", "expected", "predicted", "correct", "expected_length", "predicted_length",
                  "edit_distance", "sample_cer", "substitutions", "deletions", "insertions"}
        for row, source in zip(rows, self.predictions):
            self.assertEqual(set(row), schema)
            self.assertEqual(row["filename"], source["filename"])
            self.assertIsInstance(row["correct"], bool)
            self.assertEqual(row["edit_distance"], row["substitutions"] + row["deletions"] + row["insertions"])
            self.assertEqual(row["sample_cer"], row["edit_distance"] / len(source["expected"]))
        self.assertEqual([row["edit_distance"] for row in rows], [0, 1, 1, 1, 1, 2, 2, 4])
        json.dumps({"rows": rows, "summary": summary}, allow_nan=False)

    def test_weighted_global_metrics_match_existing_accumulator(self):
        rows, summary = analyze_predictions(self.predictions)
        total = summary["global"]
        self.assertEqual(total, {"num_images": 8, "num_correct": 1, "num_incorrect": 7,
                                 "exact_match": 1 / 8, "cer": 12 / 30,
                                 "total_reference_characters": 30, "total_edit_distance": 12,
                                 "substitutions": 4, "deletions": 4, "insertions": 4,
                                 "mean_edit_distance": 12 / 8})
        existing = MetricAccumulator()
        existing.update([row["expected"] for row in rows], [row["predicted"] for row in rows])
        self.assertEqual(total["exact_match"], existing.compute()["exact_match"])
        self.assertEqual(total["cer"], existing.compute()["cer"])
        self.assertNotAlmostEqual(total["cer"], sum(row["sample_cer"] for row in rows) / len(rows))

    def test_reference_length_and_repeat_groups_use_the_same_statistics(self):
        _, summary = analyze_predictions(self.predictions)
        self.assertEqual(list(summary["by_length"]), ["1", "2", "4", "5", "6"])
        length_four = summary["by_length"]["4"]
        self.assertEqual(length_four["num_images"], 4)
        self.assertEqual(length_four["exact_match"], 1 / 4)
        self.assertEqual(length_four["cer"], 3 / 16)
        self.assertEqual(summary["by_repeats"]["repeated"]["num_images"], 2)
        self.assertEqual(summary["by_repeats"]["repeated"]["cer"], 3 / 6)
        self.assertEqual(summary["by_repeats"]["non_repeated"]["num_images"], 6)
        self.assertEqual(summary["by_repeats"]["non_repeated"]["cer"], 9 / 24)
        for group in (*summary["by_length"].values(), *summary["by_repeats"].values()):
            self.assertEqual(set(group), set(summary["global"]))

    def test_error_type_images_operations_shares_and_combinations(self):
        _, summary = analyze_predictions(self.predictions)
        self.assertEqual(summary["error_types"], {
            "substitutions": {"operations": 4, "images": 4, "share": 1 / 3},
            "deletions": {"operations": 4, "images": 3, "share": 1 / 3},
            "insertions": {"operations": 4, "images": 2, "share": 1 / 3},
        })
        self.assertEqual(summary["error_combinations"], {
            "substitution": 2, "deletion": 2, "insertion": 1,
            "substitution+deletion": 1, "substitution+insertion": 1,
            "deletion+insertion": 0, "substitution+deletion+insertion": 0,
        })
        self.assertEqual(sum(summary["error_combinations"].values()), summary["global"]["num_incorrect"])
        self.assertEqual(summary["length_comparison"], {"shorter": 3, "longer": 2, "same_correct": 1, "same_incorrect": 2})

    def test_character_confusions_include_matches_and_exclude_gaps(self):
        _, summary = analyze_predictions(self.predictions)
        self.assertEqual(summary["substitutions"], [
            {"expected": "c", "predicted": "X", "count": 2},
            {"expected": "b", "predicted": "x", "count": 1},
            {"expected": "f", "predicted": "Q", "count": 1},
        ])
        self.assertEqual(summary["deletions"], {"a": 3, "c": 1})
        self.assertEqual(summary["insertions"], {"X": 1, "x": 3})
        confusion = summary["confusion_matrix"]
        characters, matrix = confusion["characters"], confusion["counts"]
        self.assertEqual(characters, sorted(set("".join(row["expected"] + row["predicted"] for row in self.predictions))))
        self.assertNotIn("", characters)
        self.assertEqual(len(matrix), len(characters))
        self.assertTrue(all(len(row) == len(characters) for row in matrix))
        self.assertEqual(sum(sum(row) for row in matrix), 26)
        self.assertEqual(sum(matrix[index][index] for index in range(len(characters))), 22)
        self.assertEqual(matrix[characters.index("c")][characters.index("X")], 2)
        self.assertEqual(matrix[characters.index("X")][characters.index("c")], 0)

    def test_all_seven_error_combinations_are_independent_image_counts(self):
        pairs = (("ab", "ac"), ("ab", "a"), ("a", "ab"), ("abc", "ax"),
                 ("ab", "axc"), ("abc", "bca"), ("abcd", "bceX"))
        records = [{"filename": f"{index}.png", "expected": left, "predicted": right}
                   for index, (left, right) in enumerate(pairs)]
        rows, summary = analyze_predictions(records)
        self.assertEqual(summary["global"]["num_incorrect"], 7)
        self.assertTrue(all(count == 1 for count in summary["error_combinations"].values()))
        self.assertEqual((rows[-1]["substitutions"], rows[-1]["deletions"], rows[-1]["insertions"]), (1, 1, 1))
        self.assertEqual(analyze_predictions(list(reversed(records)))[1], summary)

    def test_insertion_and_deletion_only_characters_do_not_invent_confusions(self):
        _, summary = analyze_predictions([
            {"filename": "insert.png", "expected": "a", "predicted": "ax"},
            {"filename": "delete.png", "expected": "ya", "predicted": "a"},
        ])
        self.assertEqual(summary["confusion_matrix"], {
            "characters": ["a", "x", "y"], "counts": [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
        })
        self.assertEqual(summary["insertions"], {"x": 1})
        self.assertEqual(summary["deletions"], {"y": 1})

    def test_all_correct_and_empty_repeat_bucket_produce_finite_zero_errors(self):
        rows, summary = analyze_predictions([{"filename": "one.png", "expected": "Ab", "predicted": "Ab"}])
        self.assertTrue(rows[0]["correct"])
        self.assertEqual(summary["global"]["cer"], 0.0)
        self.assertEqual(summary["global"]["exact_match"], 1.0)
        self.assertEqual(summary["by_repeats"]["repeated"]["num_images"], 0)
        self.assertEqual(summary["by_repeats"]["repeated"]["cer"], 0.0)
        self.assertTrue(all(value["share"] == 0 for value in summary["error_types"].values()))
        self.assertTrue(all(count == 0 for count in summary["error_combinations"].values()))
        self.assertEqual(summary["substitutions"], [])
        self.assertEqual(summary["deletions"], {})
        self.assertEqual(summary["insertions"], {})
        json.dumps(summary, allow_nan=False)

    def test_empty_predictions_are_valid_all_deletions_and_cer_can_exceed_one(self):
        rows, summary = analyze_predictions(iter([
            {"filename": "blank.png", "expected": "aa", "predicted": ""},
            {"filename": "long.png", "expected": "a", "predicted": "aaaaa"},
        ]))
        self.assertEqual(rows[0]["deletions"], 2)
        self.assertEqual(rows[0]["sample_cer"], 1)
        self.assertEqual(rows[1]["sample_cer"], 4)
        self.assertEqual(summary["global"]["cer"], 2)

    def test_repeats_are_consecutive_reference_characters_without_case_folding(self):
        for text in ("aa", "aab", "baac", "777", "AA", "éé"):
            with self.subTest(text=text):
                self.assertTrue(has_consecutive_repeats(text))
        for text in ("", "a", "aba", "aA", "éê"):
            with self.subTest(text=text):
                self.assertFalse(has_consecutive_repeats(text))
        with self.assertRaises(ValueError):
            has_consecutive_repeats(None)

    def test_malformed_and_inconsistent_prediction_records_fail_clearly(self):
        valid = {"filename": "one.png", "expected": "ab", "predicted": "ab"}
        invalid = (
            ([], "at least one"), (None, "collection"), ({}, "collection"), ("bad", "collection"),
            ([None], "mapping"), ([{}], "missing fields"),
            ([dict(valid, filename="")], "nonempty filename"),
            ([dict(valid, filename=42)], "nonempty filename"),
            ([dict(valid, expected="")], "reference.*nonempty"),
            ([dict(valid, expected=None)], "reference.*nonempty"),
            ([dict(valid, predicted=None)], "predicted text.*string"),
            ([dict(valid, correct=False)], "correct.*consistent"),
            ([dict(valid, correct="True")], "correct.*boolean"),
            ([valid, dict(valid)], "Duplicate"),
            ([valid, dict(valid, filename="ONE.PNG")], "Duplicate"),
        )
        for records, message in invalid:
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, message):
                analyze_predictions(records)


if __name__ == "__main__":
    unittest.main()
