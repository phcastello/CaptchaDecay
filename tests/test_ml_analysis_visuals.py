import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from ml.analysis_visuals import generate_gallery, generate_plots, select_gallery


def results():
    rows = []
    for index in range(45):
        length = 4 + index % 3
        expected = "a" * length
        rows.append({"filename": f"correct-{index:03d}.png", "expected": expected, "predicted": expected,
                     "expected_length": length, "edit_distance": 0, "correct": True,
                     "substitutions": 0, "deletions": 0, "insertions": 0, "extra_metadata": index})
    for index in range(60):
        length = 4 + index % 3
        operation = (index // 3) % 3
        expected = "a" * length
        predicted = ("b" + expected[1:], expected[:-1], expected + "b")[operation]
        rows.append({"filename": f"incorrect-{index:03d}.png", "expected": expected, "predicted": predicted,
                     "expected_length": length, "edit_distance": 1, "correct": False,
                     "substitutions": int(operation == 0), "deletions": int(operation == 1),
                     "insertions": int(operation == 2), "extra_metadata": index})
    for index in range(4):
        rows.append({"filename": f"extreme-{index:03d}.png", "expected": "aaaaaa", "predicted": "b" * (index + 8),
                     "expected_length": 6, "edit_distance": index + 8, "correct": False,
                     "substitutions": 6, "deletions": 0, "insertions": index + 2,
                     "extra_metadata": "extreme"})
    return rows


class GallerySelectionTests(unittest.TestCase):
    def test_seeded_deterministic_order_independent_selection_and_caps(self):
        rows = results()
        original = [dict(row) for row in rows]
        shuffled = list(rows)
        random.Random(8).shuffle(shuffled)
        first = select_gallery(rows, seed=42)
        self.assertEqual(first, select_gallery(shuffled, seed=42))
        self.assertEqual(rows, original)
        self.assertEqual(len(first), 40)
        self.assertEqual(sum(row["correct"] for row in first), 20)
        self.assertEqual(sum(not row["correct"] for row in first), 20)
        self.assertEqual(len({row["filename"] for row in first}), 40)
        self.assertTrue(all("extra_metadata" in row for row in first))
        self.assertNotEqual(first, select_gallery(rows, seed=99))
        self.assertTrue(all("selection" not in row for row in rows))

    def test_stratified_representative_rows_cover_lengths_and_edit_types(self):
        selected = select_gallery(results())
        correct = [row for row in selected if row["correct"]]
        incorrect = [row for row in selected if not row["correct"] and row["selection"] == "amostra"]
        self.assertEqual({row["expected_length"] for row in correct}, {4, 5, 6})
        observed = {(row["expected_length"], bool(row["substitutions"]), bool(row["deletions"]), bool(row["insertions"]))
                    for row in incorrect}
        self.assertEqual(len(observed), 9)

    def test_extremes_are_highest_distance_and_explicitly_marked(self):
        selected = select_gallery(results())
        extremes = [row for row in selected if row["selection"] == "extremo"]
        self.assertEqual([row["edit_distance"] for row in extremes], [11, 10, 9, 8])
        self.assertTrue(all(not row["correct"] for row in extremes))
        self.assertEqual(len(extremes), 4)

    def test_duplicate_rows_deduplicated_and_conflicts_rejected(self):
        row = results()[0]
        self.assertEqual(len(select_gallery([row, dict(row)])), 1)
        conflict = dict(row, predicted="other")
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            select_gallery([row, conflict])

    def test_empty_inputs_small_quotas_and_limit_validation(self):
        self.assertEqual(select_gallery([]), [])
        self.assertEqual(select_gallery(results(), max_correct=0, max_incorrect=0), [])
        self.assertEqual(len(select_gallery(results(), max_correct=1, max_incorrect=1)), 2)
        selected = select_gallery(results(), max_correct=100, max_incorrect=100)
        self.assertEqual(len(selected), 40)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            select_gallery(results(), max_correct=-1)
        with self.assertRaisesRegex(ValueError, "require string"):
            select_gallery([{"filename": "x.png", "expected": "ab", "predicted": None}])


class DiagnosticArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture_images(self, rows):
        paths = {}
        for index, row in enumerate(rows):
            # Neutral geometry is a fixture, not generated CAPTCHA content.
            image = Image.new("RGB", (215, 80), (230 - index * 10, 245, 250))
            draw = ImageDraw.Draw(image)
            draw.rectangle((10 + index, 12, 90, 64), fill=(30, 90, 160))
            path = self.root / row["filename"]
            image.save(path)
            paths[row["filename"]] = path
        return paths

    def test_gallery_png_captions_native_images_and_original_hashes(self):
        rows = [results()[0], results()[46]]
        paths = self.fixture_images(rows)
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        output = self.root / "artifacts" / "gallery.png"
        original_text = ImageDraw.ImageDraw.text
        captured = []

        def recording_text(draw, position, text, *args, **kwargs):
            captured.append(text)
            return original_text(draw, position, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", recording_text):
            selected = generate_gallery(rows, paths, output)
        self.assertEqual(len(selected), 2)
        self.assertTrue(any("esperado:" in text for text in captured))
        self.assertTrue(any("previsto:" in text for text in captured))
        self.assertTrue(any("distância:" in text for text in captured))
        self.assertTrue(any("seleção:" in text for text in captured))
        self.assertTrue(any(rows[0]["filename"] in text for text in captured))
        with Image.open(output) as gallery:
            self.assertEqual(gallery.format, "PNG")
            gallery.verify()
        with Image.open(output) as gallery, Image.open(paths[selected[0]["filename"]]) as original:
            # First native image is centered in the first 310px card, at y=98.
            crop = gallery.crop((59, 98, 59 + 215, 98 + 80))
            self.assertEqual(crop.tobytes(), original.convert("RGB").tobytes())
        self.assertEqual(hashes, {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()})

    def test_gallery_refuses_overwriting_originals_missing_images_and_wrong_dimensions(self):
        rows = [results()[0]]
        paths = self.fixture_images(rows)
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            generate_gallery(rows, paths, paths[rows[0]["filename"]])
        with self.assertRaisesRegex(ValueError, "Missing original image path"):
            generate_gallery(rows, {}, self.root / "gallery.png")
        Image.new("RGB", (10, 10)).save(paths[rows[0]["filename"]])
        with self.assertRaisesRegex(ValueError, "215 x 80 PNG"):
            generate_gallery(rows, paths, self.root / "gallery.png")

    def test_empty_gallery_writes_explicit_empty_png(self):
        output = self.root / "empty-gallery.png"
        self.assertEqual(generate_gallery([], {}, output), [])
        with Image.open(output) as image:
            image.verify()

    def test_pillow_plots_use_actual_summary_and_leave_summary_unmodified(self):
        summary = {
            "confusion_matrix": {"characters": ["a", "b"], "counts": [[40, 3], [2, 27]]},
            "by_length": {
                "4": {"num_images": 12, "num_correct": 9, "exact_match": 0.75, "cer": 0.125,
                      "substitutions": 3, "deletions": 2, "insertions": 1},
                "6": {"num_images": 1, "num_correct": 0, "exact_match": 0, "cer": 1.5,
                      "substitutions": 2, "deletions": 1, "insertions": 6},
            },
        }
        with patch("ml.analysis_visuals.importlib.util.find_spec", return_value=None):
            paths = generate_plots(SimpleNamespace(**summary), self.root / "plots")
        self.assertEqual([path.name for path in paths], ["confusion-matrix.png", "by-length.png"])
        for path in paths:
            with self.subTest(path=path), Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                image.verify()
        self.assertEqual(summary["confusion_matrix"]["counts"], [[40, 3], [2, 27]])
        self.assertEqual(summary["by_length"]["6"]["cer"], 1.5)

    def test_empty_plots_and_invalid_data(self):
        with patch("ml.analysis_visuals.importlib.util.find_spec", return_value=None):
            paths = generate_plots({}, self.root / "empty-plots")
        self.assertTrue(all(path.is_file() for path in paths))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            generate_plots({"confusion_matrix": {"characters": ["a"], "counts": []}}, self.root)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            generate_plots({"confusion_matrix": {"characters": ["a"], "counts": [[-1]]}}, self.root)
        with self.assertRaisesRegex(ValueError, "Invalid cer"):
            generate_plots({"by_length": {4: {"cer": float("nan")}}}, self.root)


if __name__ == "__main__":
    unittest.main()
