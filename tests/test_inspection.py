import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from inspection.compare import compare, fit_size, image_paths, select_samples
from test_labeler import create_dataset


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.real = self.root / "real"
        create_dataset(self.real)
        self.synthetic = self.root / "synthetic/pilot"
        train = self.synthetic / "train"
        train.mkdir(parents=True)
        rows = []
        for i, label in enumerate(["AB", "AbC", "AB", "AbC123", "x", "", "YZ"]):
            name = f"{i:08d}.png"
            Image.new("RGB", (60, 20), (20 * i, 40, 50)).save(train / name)
            rows.append([name, label, "train", "xoshiro256starstar-sha256-v2"])
        with (self.synthetic / "labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "label", "split", "rng_scheme"])
            writer.writerows(rows)

    def run_compare(self, name="comparison.png", **options):
        return compare(self.real / "train", self.synthetic / "train", output=self.root / name, **options)

    def test_deterministic_sampling_and_seed_changes_without_global_rng(self):
        paths = image_paths(self.synthetic / "train")
        first = select_samples(paths, 4, 42, "synthetic")
        self.assertEqual(first, select_samples(list(reversed(paths)), 4, 42, "synthetic"))
        self.assertNotEqual(first, select_samples(paths, 4, 7, "synthetic"))
        a = self.run_compare(count=2, seed=42)
        b = self.run_compare("again.png", count=2, seed=42)
        for domain in ("real_train", "synthetic"):
            self.assertEqual(a[domain]["selected_files"], b[domain]["selected_files"])

    def test_valid_png_json_counts_labels_and_originals_unchanged(self):
        original = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = self.run_compare(count=20)
        with Image.open(self.root / "comparison.png") as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreater(image.width, 1500)
            image.load()
        saved = json.loads((self.root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["pairs"], 3)
        self.assertEqual(report["real_train"]["available"], 3)
        self.assertEqual(report["real_train"]["sampled"], 3)
        self.assertEqual(report["synthetic"]["available"], 7)
        self.assertEqual(report["synthetic"]["sampled"], 3)
        self.assertEqual(report["real_train"]["labeled"], 2)
        self.assertEqual(report["real_train"]["unlabeled_or_unknown"], 1)
        self.assertEqual(report["real_train"]["label_lengths"], {3: 1, 18: 1})
        self.assertEqual(report["synthetic"]["label_lengths"], {1: 1, 2: 3, 3: 1, 6: 1})
        self.assertIn("a", report["real_train"]["characters"])
        self.assertIn("B", report["real_train"]["characters"])
        self.assertEqual(report["synthetic"]["characters"], "123ABCYZbx")
        self.assertTrue(all(v is None for v in report["manual_checklist"].values()))
        self.assertEqual({p: p.read_bytes() for p in original}, original)

    def test_only_train_labels_and_resolutions_enter_statistics(self):
        report = self.run_compare(count=2)
        self.assertEqual(report["real_train"]["resolutions"], {"120x40": 1, "80x30": 2})
        self.assertEqual(report["synthetic"]["resolutions"], {"60x20": 7})
        self.assertTrue(all(p.startswith("sample-") for p in report["real_train"]["selected_files"]))
        self.assertNotIn("sample-3.png", report["real_train"]["selected_files"])
        self.assertNotIn("sample-4.png", report["real_train"]["selected_files"])

    def test_proportions_and_shared_scale_are_preserved(self):
        report = self.run_compare(count=3)
        self.assertEqual(report["display_scale"], 3)
        self.assertEqual(fit_size((120, 40), report["display_scale"]), (360, 120))
        self.assertEqual(fit_size((60, 20), report["display_scale"]), (180, 60))
        # Confirm the drawn rectangular fixture has the expected width/height.
        selected = report["real_train"]["selected_files"].index("sample-1.png")
        x = 24 + (800 - 360) // 2
        y = 100 + selected * 355 + 12 + (280 - 120) // 2
        with Image.open(self.root / "comparison.png") as image:
            self.assertEqual(image.getpixel((x, y)), (30, 100, 200))
            self.assertEqual(image.getpixel((x + 359, y + 119)), (30, 100, 200))
            self.assertEqual(image.getpixel((x - 1, y)), (255, 255, 255))
            self.assertEqual(image.getpixel((x, y + 120)), (255, 255, 255))

    def test_test_validation_aliases_and_wrong_real_directory_are_rejected(self):
        for directory in (self.real / "test", self.real / "validation",
                          self.real / "train/../test", self.root / "arbitrary"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                compare(directory, self.synthetic / "train", output=self.root / "forbidden.png")
        with self.assertRaises(ValueError):
            compare(self.real / "train", self.real / "test", output=self.root / "forbidden.png")
        self.assertFalse((self.root / "forbidden.png").exists())

    def test_empty_missing_invalid_count_and_existing_output_are_reported(self):
        empty = self.root / "empty/train"
        empty.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "vazio"):
            compare(empty, self.synthetic / "train", output=self.root / "empty.png")
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_compare(count=count)
        self.run_compare(count=1)
        original = (self.root / "comparison.png").read_bytes()
        with self.assertRaisesRegex(ValueError, "Saída já existe"):
            self.run_compare(count=1)
        self.assertEqual((self.root / "comparison.png").read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "ausente"):
            compare(self.root / "missing/train", self.synthetic / "train", output=self.root / "missing.png")

    def test_missing_csv_means_unknown_labels_without_inference(self):
        (self.synthetic / "labels.csv").unlink()
        report = self.run_compare(count=1)
        self.assertEqual(report["synthetic"]["labeled"], 0)
        self.assertEqual(report["synthetic"]["label_lengths"], {})
        self.assertIsNone(report["synthetic"]["labels_csv"])


if __name__ == "__main__":
    unittest.main()
