import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml.charset import Charset
from ml.dataset import SyntheticDataset, SyntheticManifest, collate_ctc


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "synthetic"
        self.root.mkdir()
        for split in ("train", "validation"):
            (self.root / split).mkdir()
        self.rows = [
            dict(filename="002.png", label="bA44", split="train", width=215, height=80, length=4),
            dict(filename="001.png", label="44Aba", split="train", width=215, height=80, length=5),
            dict(filename="003.png", label="Aabb44", split="validation", width=215, height=80, length=6),
        ]
        for index, row in enumerate(self.rows):
            Image.new("RGB", (215, 80), (index * 100, 255, 0)).save(self.root / row["split"] / row["filename"])
        self.write_csv()

    def write_csv(self, fields=None):
        with (self.root / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(self.rows[0]), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def test_labels_come_from_csv_and_splits_are_separate(self):
        manifest = SyntheticManifest(self.root)
        train = SyntheticDataset(self.root, "train", manifest=manifest)
        validation = SyntheticDataset(self.root, "validation", train.charset, manifest=manifest)
        self.assertEqual([record.filename for record in train.records], ["001.png", "002.png"])
        self.assertEqual(train[0]["label"], "44Aba")
        self.assertEqual(train.charset.decode(train[0]["target"]), "44Aba")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertEqual(validation[0]["label"], "Aabb44")
        self.assertFalse({record.path for record in train.records} & {record.path for record in validation.records})

    def test_full_image_grayscale_float32_and_normalization(self):
        Image.new("RGB", (215, 80), "white").save(self.root / "train" / "001.png")
        dataset = SyntheticDataset(self.root, "train")
        image = dataset[0]["image"]
        self.assertEqual(image.shape, (1, 80, 215))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(torch.all(image == 1.0))
        Image.new("RGB", (215, 80), "black").save(self.root / "train" / "001.png")
        self.assertTrue(torch.all(dataset[0]["image"] == -1.0))

    def test_collate_variable_targets_and_windows_worker_default(self):
        dataset = SyntheticDataset(self.root, "train")
        loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_ctc)
        batch = next(iter(loader))
        self.assertEqual(batch["images"].shape, (2, 1, 80, 215))
        self.assertEqual(batch["targets"].dtype, torch.long)
        self.assertEqual(batch["target_lengths"].tolist(), [5, 4])
        self.assertEqual(batch["targets"].numel(), 9)
        self.assertEqual(batch["labels"], ["44Aba", "bA44"])
        self.assertEqual(batch["filenames"], ["001.png", "002.png"])

    def test_missing_manifest_prints_manual_generation_command(self):
        (self.root / "labels.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "php generator/generate.php --count 3000"):
            SyntheticDataset(self.root, "train")

    def test_missing_or_duplicate_columns_rejected(self):
        self.write_csv(fields=["filename", "label", "split"])
        with self.assertRaisesRegex(ValueError, "columns.*height.*width"):
            SyntheticManifest(self.root)
        (self.root / "labels.csv").write_text("filename,label,split,width,height,label\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate headers"):
            SyntheticManifest(self.root)

    def test_malformed_row_and_csv_rejected(self):
        with (self.root / "labels.csv").open("a", encoding="utf-8") as stream:
            stream.write("bad.png,ab,train,215,80,2,unexpected\n")
        with self.assertRaisesRegex(ValueError, "Malformed CSV row"):
            SyntheticManifest(self.root)
        (self.root / "labels.csv").write_text(
            'filename,label,split,width,height\n"unterminated', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Invalid CSV"):
            SyntheticManifest(self.root)

    def test_missing_png_identifies_filename(self):
        (self.root / "train" / "002.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing PNG.*002.png"):
            SyntheticDataset(self.root, "validation")

    def test_non_png_and_corrupted_png_rejected(self):
        path = self.root / "train" / "002.png"
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Invalid or corrupted PNG.*002.png"):
            SyntheticManifest(self.root)
        Image.new("RGB", (215, 80)).save(path, format="JPEG")
        with self.assertRaisesRegex(ValueError, "not a PNG.*002.png"):
            SyntheticManifest(self.root)

    def test_wrong_dimensions_in_image_or_csv_rejected(self):
        Image.new("RGB", (214, 80)).save(self.root / "train" / "002.png")
        with self.assertRaisesRegex(ValueError, "dimensions.*002.png"):
            SyntheticManifest(self.root)
        self.rows[0]["width"] = 214
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "dimensions.*width=214"):
            SyntheticManifest(self.root)

    def test_empty_label_rejected(self):
        for label in ("", "  "):
            self.rows[0]["label"] = label
            self.write_csv()
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "Empty label.*002.png"):
                SyntheticManifest(self.root)

    def test_duplicate_entries_and_filenames_across_splits_rejected(self):
        self.rows.append(dict(self.rows[0]))
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "Duplicate filename.*002.png"):
            SyntheticManifest(self.root)
        self.rows[-1]["split"] = "validation"
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "Duplicate filename.*002.png"):
            SyntheticManifest(self.root)

    def test_unknown_validation_character_does_not_extend_training_vocabulary(self):
        self.rows[2]["label"] = "Aabb4X"
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "Unknown character 'X'.*validation/003.png"):
            SyntheticDataset(self.root, "train")

    def test_given_charset_checks_train_labels_too(self):
        with self.assertRaisesRegex(ValueError, "Unknown character '4'.*train/001.png"):
            SyntheticDataset(self.root, "validation", Charset("Ab"))

    def test_invalid_split_and_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid synthetic split"):
            SyntheticDataset(self.root, "test")
        self.rows[0]["split"] = "test"
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "Invalid split.*test"):
            SyntheticManifest(self.root)
        self.rows[0]["split"] = "train"
        for filename in ("../other.png", "..\\other.png", "http://example.png", "file.jpg"):
            self.rows[0]["filename"] = filename
            self.write_csv()
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "Invalid PNG filename"):
                SyntheticManifest(self.root)

    def test_length_metadata_is_checked(self):
        self.rows[0]["length"] = 6
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "length metadata mismatch.*002.png"):
            SyntheticManifest(self.root)

    def test_manifest_identity_hashes_csv_and_image_bytes(self):
        original = SyntheticManifest(self.root).identity
        self.assertEqual(original["counts"], {"train": 2, "validation": 1})
        self.assertEqual(original["image_count"], 3)
        self.assertEqual(original, SyntheticManifest(self.root).identity)
        Image.new("RGB", (215, 80), "black").save(self.root / "train" / "002.png")
        changed = SyntheticManifest(self.root).identity
        self.assertEqual(original["manifest_sha256"], changed["manifest_sha256"])
        self.assertNotEqual(original["images_sha256"], changed["images_sha256"])
        self.assertNotEqual(original["fingerprint"], changed["fingerprint"])
        self.rows[0]["label"] = "Ab44"
        self.write_csv()
        self.assertNotEqual(changed["manifest_sha256"], SyntheticManifest(self.root).identity["manifest_sha256"])

    def test_real_paths_remote_paths_and_real_source_metadata_rejected(self):
        for root in (self.root / "dataset" / "real", "https://example.invalid/data"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, "Synthetic-only|exclusively local"):
                SyntheticManifest(root)
        self.rows[0]["source"] = "real"
        self.write_csv()
        with self.assertRaisesRegex(ValueError, "refuses real/remote source"):
            SyntheticManifest(self.root)

    def test_empty_manifest_and_empty_batches_rejected(self):
        self.rows = []
        self.write_csv(fields=["filename", "label", "split", "width", "height"])
        with self.assertRaisesRegex(ValueError, "no records"):
            SyntheticManifest(self.root)
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate_ctc([])


if __name__ == "__main__":
    unittest.main()
