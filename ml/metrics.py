"""Exact string accuracy and dataset-weighted character error rate."""

from collections import Counter
from collections.abc import Sequence


def levenshtein(expected: str, predicted: str) -> int:
    """Unit-cost edit distance using memory proportional to the shorter text."""
    if len(expected) < len(predicted):
        expected, predicted = predicted, expected
    previous = list(range(len(predicted) + 1))
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, predicted_char in enumerate(predicted, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_char != predicted_char),
            ))
        previous = current
    return previous[-1]


def align_edits(expected: str, predicted: str) -> list[dict]:
    """Return a deterministic minimum-edit alignment in reading order.

    Backtracking prefers an exact match, then substitution, deletion and
    insertion, in that order. Equal-cost alignments can attribute errors to
    different characters, so this rule is part of the diagnostic contract.
    Empty strings are valid here; a deletion/insertion uses an empty predicted/
    expected character respectively. Case and all original characters remain
    unchanged. The existing distance-only implementation is unaffected.
    """
    if not isinstance(expected, str) or not isinstance(predicted, str):
        raise ValueError("Alignment requires expected and predicted strings")
    costs = [list(range(len(predicted) + 1))]
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, predicted_char in enumerate(predicted, start=1):
            current.append(min(
                costs[row - 1][column - 1] + (expected_char != predicted_char),
                costs[row - 1][column] + 1,
                current[column - 1] + 1,
            ))
        costs.append(current)
    row, column = len(expected), len(predicted)
    alignment = []
    while row or column:
        if (row and column and expected[row - 1] == predicted[column - 1]
                and costs[row][column] == costs[row - 1][column - 1]):
            operation, left, right = "match", expected[row - 1], predicted[column - 1]
            row -= 1
            column -= 1
        elif row and column and costs[row][column] == costs[row - 1][column - 1] + 1:
            operation, left, right = "substitution", expected[row - 1], predicted[column - 1]
            row -= 1
            column -= 1
        elif row and costs[row][column] == costs[row - 1][column] + 1:
            operation, left, right = "deletion", expected[row - 1], ""
            row -= 1
        else:
            operation, left, right = "insertion", "", predicted[column - 1]
            column -= 1
        alignment.append({"operation": operation, "expected": left, "predicted": right})
    alignment.reverse()
    return alignment


class MetricAccumulator:
    """Aggregate all images, edits and reference characters across batches."""

    def __init__(self):
        self.num_images = 0
        self.num_correct = 0
        self.num_characters = 0
        self.num_errors = 0
        self.length_counts: Counter[int] = Counter()
        self.length_correct: Counter[int] = Counter()

    def update(self, expected: Sequence[str], predicted: Sequence[str]) -> None:
        if isinstance(expected, str) or isinstance(predicted, str) or len(expected) != len(predicted):
            raise ValueError("Metrics require equally sized sequences of expected and predicted strings")
        if any(not isinstance(value, str) for value in (*expected, *predicted)):
            raise ValueError("Every expected and predicted value must be a string")
        for truth, prediction in zip(expected, predicted):
            correct = truth == prediction
            length = len(truth)
            self.num_images += 1
            self.num_correct += correct
            self.num_characters += length
            self.num_errors += levenshtein(truth, prediction)
            self.length_counts[length] += 1
            self.length_correct[length] += correct

    def compute(self) -> dict:
        if self.num_characters == 0 and self.num_errors:
            raise ValueError("CER is undefined when all reference strings are empty and predictions contain characters")
        return {
            "exact_match": self.num_correct / self.num_images if self.num_images else 0.0,
            "cer": self.num_errors / self.num_characters if self.num_characters else 0.0,
            "num_images": self.num_images,
            "by_length": {
                str(length): {
                    "exact_match": self.length_correct[length] / self.length_counts[length],
                    "num_images": self.length_counts[length],
                }
                for length in sorted(self.length_counts)
            },
        }
