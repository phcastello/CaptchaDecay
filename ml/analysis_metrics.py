"""Deterministic diagnostics over already decoded, local prediction records.

This module has no inference, training, model, dataset, file or network access.
CER uses total edit operations divided by total reference characters. An error
type's share uses all edit operations as its denominator, while its image count
counts an image once even if it has several operations of that type.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import combinations

from .metrics import align_edits, levenshtein


ERROR_OPERATIONS = ("substitution", "deletion", "insertion")
ERROR_FIELDS = ("substitutions", "deletions", "insertions")


def has_consecutive_repeats(text: str) -> bool:
    """Check neighboring characters, preserving case and character identity."""
    if not isinstance(text, str):
        raise ValueError("Repeat analysis requires a string")
    return any(left == right for left, right in zip(text, text[1:]))


def _statistics(rows: list[dict]) -> dict:
    num_images = len(rows)
    num_correct = sum(row["correct"] for row in rows)
    num_characters = sum(row["expected_length"] for row in rows)
    distance = sum(row["edit_distance"] for row in rows)
    return {
        "num_images": num_images,
        "num_correct": num_correct,
        "num_incorrect": num_images - num_correct,
        "exact_match": num_correct / num_images if num_images else 0.0,
        "cer": distance / num_characters if num_characters else 0.0,
        "total_reference_characters": num_characters,
        "total_edit_distance": distance,
        "substitutions": sum(row["substitutions"] for row in rows),
        "deletions": sum(row["deletions"] for row in rows),
        "insertions": sum(row["insertions"] for row in rows),
        "mean_edit_distance": distance / num_images if num_images else 0.0,
    }


def analyze_predictions(predictions: Iterable[Mapping]) -> tuple[list[dict], dict]:
    """Enrich prediction rows and aggregate error counts without mutating input.

    Each input requires filename, expected and predicted strings. References
    must be nonempty; an empty decoded prediction is a valid all-deletions
    result. If supplied, ``correct`` must be a boolean agreeing with the strings.
    Filename uniqueness is case-insensitive to match the Windows dataset.

    Error combinations use singular operation names in substitution/deletion/
    insertion order, joined with ``+``. All seven combinations are included,
    including zero counts. Substitutions are sorted by descending count, then
    expected and predicted character. The confusion matrix has reference rows
    and predicted columns; it counts matches and substitutions only. Insertions
    and deletions have separate character counters, without an invented blank.
    """
    if isinstance(predictions, (str, bytes, Mapping)):
        raise ValueError("Predictions must be a nonempty collection of prediction records")
    try:
        source_rows = list(predictions)
    except TypeError as error:
        raise ValueError("Predictions must be a nonempty collection of prediction records") from error
    if not source_rows:
        raise ValueError("Predictions must contain at least one record")
    rows = []
    seen_filenames = set()
    substitution_counts = Counter()
    deletion_counts = Counter()
    insertion_counts = Counter()
    confusion_counts = Counter()
    characters = set()
    combinations_counts = {
        "+".join(names): 0
        for count in range(1, len(ERROR_OPERATIONS) + 1)
        for names in combinations(ERROR_OPERATIONS, count)
    }
    length_comparison = {key: 0 for key in ("shorter", "longer", "same_correct", "same_incorrect")}
    for index, source in enumerate(source_rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"Prediction record {index} must be a mapping")
        missing = {"filename", "expected", "predicted"}.difference(source)
        if missing:
            raise ValueError(f"Prediction record {index} is missing fields: {', '.join(sorted(missing))}")
        filename, expected, predicted = source["filename"], source["expected"], source["predicted"]
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Prediction record {index} must have a nonempty filename string")
        if filename.casefold() in seen_filenames:
            raise ValueError(f"Duplicate prediction filename: {filename!r}")
        seen_filenames.add(filename.casefold())
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"{filename}: expected/reference text must be a nonempty string")
        if not isinstance(predicted, str):
            raise ValueError(f"{filename}: predicted text must be a string (an empty prediction is allowed)")
        correct = expected == predicted
        if "correct" in source:
            if not isinstance(source["correct"], bool) or source["correct"] != correct:
                raise ValueError(f"{filename}: correct must be a boolean consistent with expected and predicted text")
        aligned = align_edits(expected, predicted)
        counts = Counter(operation["operation"] for operation in aligned)
        edit_distance = levenshtein(expected, predicted)
        if sum(counts[name] for name in ERROR_OPERATIONS) != edit_distance:
            raise RuntimeError(f"{filename}: alignment operation count disagrees with Levenshtein distance")
        row = {
            "filename": filename,
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "expected_length": len(expected),
            "predicted_length": len(predicted),
            "edit_distance": edit_distance,
            "sample_cer": edit_distance / len(expected),
            "substitutions": counts["substitution"],
            "deletions": counts["deletion"],
            "insertions": counts["insertion"],
        }
        rows.append(row)
        if not correct:
            combination = "+".join(name for name in ERROR_OPERATIONS if counts[name])
            combinations_counts[combination] += 1
        if len(predicted) < len(expected):
            length_comparison["shorter"] += 1
        elif len(predicted) > len(expected):
            length_comparison["longer"] += 1
        else:
            length_comparison["same_correct" if correct else "same_incorrect"] += 1
        characters.update(expected)
        characters.update(predicted)
        for operation in aligned:
            name, left, right = operation["operation"], operation["expected"], operation["predicted"]
            if name == "substitution":
                substitution_counts[left, right] += 1
            elif name == "deletion":
                deletion_counts[left] += 1
            elif name == "insertion":
                insertion_counts[right] += 1
            if name in ("match", "substitution"):
                confusion_counts[left, right] += 1
    global_statistics = _statistics(rows)
    ordered_characters = sorted(characters)
    summary = {
        "global": global_statistics,
        "by_length": {
            str(length): _statistics([row for row in rows if row["expected_length"] == length])
            for length in sorted({row["expected_length"] for row in rows})
        },
        "by_repeats": {
            name: _statistics([row for row in rows if has_consecutive_repeats(row["expected"]) == repeated])
            for name, repeated in (("repeated", True), ("non_repeated", False))
        },
        "error_types": {
            field: {
                "operations": global_statistics[field],
                "images": sum(row[field] > 0 for row in rows),
                "share": (global_statistics[field] / global_statistics["total_edit_distance"]
                          if global_statistics["total_edit_distance"] else 0.0),
            }
            for field in ERROR_FIELDS
        },
        "error_combinations": combinations_counts,
        "length_comparison": length_comparison,
        "substitutions": [
            {"expected": left, "predicted": right, "count": count}
            for (left, right), count in sorted(
                substitution_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "deletions": dict(sorted(deletion_counts.items())),
        "insertions": dict(sorted(insertion_counts.items())),
        "confusion_matrix": {
            "characters": ordered_characters,
            "counts": [[confusion_counts[left, right] for right in ordered_characters] for left in ordered_characters],
        },
    }
    return rows, summary
