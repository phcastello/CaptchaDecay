import copy
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from generator import prepare_experiment_d as preparation


def png_bytes(color):
    stream = io.BytesIO()
    Image.new("RGB", (215, 80), color).save(stream, format="PNG")
    return stream.getvalue()


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=preparation.COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class HistoricalValidationTests(unittest.TestCase):
    def test_all_300_versioned_images_labels_and_metadata_are_intact(self):
        snapshot = preparation.verify_snapshot(preparation.HISTORICAL)
        self.assertEqual(len(snapshot["records"]), 300)
        self.assertEqual(sum(len(row["label"]) for row in snapshot["rows"]), 1507)
        self.assertEqual(preparation.digest(snapshot["labels_data"]),
                         preparation.HISTORICAL_LABELS_SHA256)


@unittest.skipUnless(sys.platform == "linux", "Preparação/publicação atômica usa renameat2 do Linux.")
class ExperimentDDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "historical"
        (self.source / "validation").mkdir(parents=True)
        template = preparation.read_rows(preparation.HISTORICAL / "labels.csv")[1][0]
        self.rows = []
        records = []
        for number, label in ((8, "abcd"), (13, "2345")):
            row = {**template, "filename": f"{number:08d}.png", "index": str(number),
                   "label": label, "length": str(len(label))}
            data = png_bytes((number, 20, 50))
            (self.source / "validation" / row["filename"]).write_bytes(data)
            records.append({"filename": row["filename"], "label": label,
                            "sha256": preparation.digest(data), "bytes": len(data)})
            self.rows.append(row)
        labels_data = csv_bytes(self.rows)
        (self.source / "labels.csv").write_bytes(labels_data)
        manifest = {"schema_version": 1, "split": "validation", "num_images": 2,
                    "labels_file": "labels.csv", "charset": preparation.CHARSET,
                    "image_dimensions": {"width": 215, "height": 80},
                    "total_reference_characters": 8, "counts_by_length": {"4": 2},
                    "labels_sha256": preparation.digest(labels_data), "images": records,
                    "image_records_sha256": preparation.canonical_hash(records)}
        manifest_data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        (self.source / "manifest.json").write_bytes(manifest_data)
        self.manifest_sha = preparation.digest(manifest_data)
        self.labels_sha = preparation.digest(labels_data)
        self.snapshot = self.verify_fixture()
        self.destination = self.root / "experiment-d"

    def verify_fixture(self, *unused_args, **unused_options):
        return preparation.verify_snapshot(
            self.source, expected_count=2, expected_manifest_sha256=self.manifest_sha,
            expected_labels_sha256=self.labels_sha)

    def recipe(self, php, destination, count, seed):
        return {"config": {"count": count, "seed": seed, "output": str(destination)},
                "visual_config_sha256": preparation.VISUAL_CONFIG_SHA256,
                "securimage": {"source_path": str(self.source),
                               "source_tree_sha256": preparation._dependency_tree_hash(self.source)},
                "files_sha256": {name: preparation.digest((preparation.ROOT / name).read_bytes())
                                   for name in ("generator/generate.php", "generator/config.php", "composer.lock")}}

    def fake_generator(self, command):
        output = Path(command[command.index("--output") + 1])
        count = int(command[command.index("--count") + 1])
        seed = command[command.index("--seed") + 1]
        rows = copy.deepcopy(self.rows)
        (output / "train").mkdir()
        for index in range(14, 14 + count):
            row = {**self.rows[0], "filename": f"{index:08d}.png", "index": str(index),
                   "split": "train", "seed": seed}
            (output / "train" / row["filename"]).write_bytes(png_bytes((index, 60, 70)))
            rows.append(row)
        # The real generator appends to the preserved historical CSV.
        (output / "labels.csv").write_bytes(csv_bytes(rows))
        return f"Concluído: train={count}, validation=0.\n"

    def prepare_fixture(self):
        real_verify = preparation.verify_snapshot
        def check_fixture(source):
            return real_verify(source, expected_count=2, expected_manifest_sha256=self.manifest_sha,
                               expected_labels_sha256=self.labels_sha)
        with patch.object(preparation, "verify_snapshot", side_effect=check_fixture), \
                patch.object(preparation, "_recipe", side_effect=self.recipe), \
                patch.object(preparation, "_run_php", side_effect=self.fake_generator):
            return preparation.prepare_dataset(self.destination, source=self.source, train_count=2)

    def test_fixture_verifies_but_public_default_requires_300(self):
        self.assertEqual(len(self.verify_fixture()["rows"]), 2)
        with self.assertRaisesRegex(ValueError, "Quantidade histórica"):
            preparation.verify_snapshot(self.source, expected_manifest_sha256=self.manifest_sha,
                                        expected_labels_sha256=self.labels_sha)

    def test_historical_pixel_corruption_is_rejected(self):
        (self.source / "validation/00000008.png").write_bytes(png_bytes((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "PNG histórico alterado"):
            self.verify_fixture()

    def test_historical_metadata_drift_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["render_seed"] = "17"
        (self.source / "labels.csv").write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(ValueError, "CSV histórico alterado"):
            self.verify_fixture()

    def test_historical_manifest_drift_is_rejected(self):
        (self.source / "manifest.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Manifesto histórico alterado"):
            self.verify_fixture()

    def test_unexpected_historical_file_is_rejected(self):
        (self.source / "validation/extra.png").write_bytes(png_bytes((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "inesperados"):
            self.verify_fixture()

    def test_invalid_source_does_not_create_destination_or_parent(self):
        destination = self.root / "absent-parent" / "experiment-d"
        with self.assertRaisesRegex(ValueError, "Manifesto histórico alterado"):
            preparation.prepare_dataset(destination, source=self.source, train_count=2)
        self.assertFalse(destination.parent.exists())

    def test_prepare_preserves_all_historical_bytes_and_writes_integrity_records(self):
        result = self.prepare_fixture()
        self.assertEqual(result["status"], "created")
        self.assertEqual(result["counts"], {"train": 2, "validation": 2})
        actual = preparation.verify_dataset(self.destination, self.snapshot, 2, 20261020)
        self.assertEqual(len(actual["images"]), 4)
        for row in self.rows:
            self.assertEqual((self.destination / "validation" / row["filename"]).read_bytes(),
                             (self.source / "validation" / row["filename"]).read_bytes())
        for name in ("labels.csv", "manifest.json"):
            self.assertEqual((self.destination / "historical-validation" / name).read_bytes(),
                             (self.source / name).read_bytes())
        self.assertEqual(self.verify_fixture(), self.snapshot)

    def test_second_preparation_verifies_without_appending_or_running_php(self):
        self.prepare_fixture()
        before = (self.destination / "labels.csv").read_bytes()
        with patch.object(preparation, "verify_snapshot", return_value=self.snapshot), \
                patch.object(preparation, "_recipe") as recipe, \
                patch.object(preparation, "_run_php") as generator:
            result = preparation.prepare_dataset(self.destination, source=self.source, train_count=2)
        self.assertEqual(result["status"], "verified_existing")
        recipe.assert_not_called()
        generator.assert_not_called()
        self.assertEqual(before, (self.destination / "labels.csv").read_bytes())

    def test_existing_corrupt_dataset_is_not_repaired_or_appended(self):
        self.prepare_fixture()
        corrupt = self.destination / "train/00000014.png"
        corrupt.write_bytes(png_bytes((1, 2, 3)))
        before = (self.destination / "labels.csv").read_bytes()
        with patch.object(preparation, "verify_snapshot", return_value=self.snapshot), \
                patch.object(preparation, "_run_php") as generator:
            with self.assertRaisesRegex(ValueError, "integridade diverge"):
                preparation.prepare_dataset(self.destination, source=self.source, train_count=2)
        generator.assert_not_called()
        self.assertEqual(corrupt.read_bytes(), png_bytes((1, 2, 3)))
        self.assertEqual(before, (self.destination / "labels.csv").read_bytes())

    def test_exact_train_validation_leakage_is_rejected(self):
        self.prepare_fixture()
        shutil.copyfile(self.source / "validation/00000008.png", self.destination / "train/00000014.png")
        with self.assertRaisesRegex(ValueError, "Duplicata exata/leakage"):
            preparation.inspect_dataset(self.destination, self.snapshot, 2, 20261020)

    def test_missing_png_is_rejected(self):
        self.prepare_fixture()
        (self.destination / "train/00000014.png").unlink()
        with self.assertRaisesRegex(ValueError, "PNG ausente"):
            preparation.inspect_dataset(self.destination, self.snapshot, 2, 20261020)

    def test_wrong_png_dimensions_are_rejected(self):
        self.prepare_fixture()
        Image.new("RGB", (80, 215)).save(self.destination / "train/00000014.png")
        with self.assertRaisesRegex(ValueError, "PNG inválido"):
            preparation.inspect_dataset(self.destination, self.snapshot, 2, 20261020)

    def test_invalid_train_charset_is_rejected(self):
        self.prepare_fixture()
        path = self.destination / "labels.csv"
        _, rows = preparation.read_rows(path)
        rows[-1]["label"] = "abc0"
        path.write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(ValueError, "charset"):
            preparation.inspect_dataset(self.destination, self.snapshot, 2, 20261020)

    def test_validation_metadata_change_is_rejected_even_with_same_png(self):
        self.prepare_fixture()
        path = self.destination / "labels.csv"
        _, rows = preparation.read_rows(path)
        rows[0]["render_seed"] = "17"
        path.write_bytes(csv_bytes(rows))
        with self.assertRaisesRegex(ValueError, "CSV histórico foi reescrito"):
            preparation.inspect_dataset(self.destination, self.snapshot, 2, 20261020)

    def test_reusing_historical_seed_is_rejected_before_generation(self):
        with patch.object(preparation, "verify_snapshot", return_value=self.snapshot), \
                patch.object(preparation, "_run_php") as generator:
            with self.assertRaisesRegex(ValueError, "seed distinta"):
                preparation.prepare_dataset(self.destination, source=self.source, train_count=2, seed=42)
        generator.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_failed_generation_keeps_staging_and_never_publishes(self):
        with patch.object(preparation, "verify_snapshot", return_value=self.snapshot), \
                patch.object(preparation, "_recipe", side_effect=self.recipe), \
                patch.object(preparation, "_run_php", side_effect=RuntimeError("generation failed")):
            with self.assertRaisesRegex(RuntimeError, "arquivos parciais preservados"):
                preparation.prepare_dataset(self.destination, source=self.source, train_count=2)
        self.assertFalse(self.destination.exists())
        self.assertEqual(len(list(self.root.glob(".experiment-d.staging-*"))), 1)
        self.assertEqual(self.verify_fixture(), self.snapshot)

    def test_atomic_publish_does_not_replace_an_existing_empty_directory(self):
        self.destination.mkdir()
        staging = self.root / "staging"
        staging.mkdir()
        (staging / "marker").write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            preparation._publish_without_replacement(staging, self.destination)
        self.assertTrue((staging / "marker").exists())
        self.assertEqual(list(self.destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
