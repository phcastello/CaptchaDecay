import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from inspection.compare import allowed_directories, compare, fit_size, image_paths, select_samples
from labeling.labeler import LabelStore
from test_labeler import create_dataset


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.real = self.root / "real"
        create_dataset(self.real)
        self.synthetic = self.root / "synthetic/pilot"
        self.roots = {"real_root": self.real, "synthetic_root": self.synthetic.parent}
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
        return compare(self.real / "train", self.synthetic / "train", output=self.root / name,
                       **self.roots, **options)

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
                compare(directory, self.synthetic / "train", output=self.root / "forbidden.png", **self.roots)
        with self.assertRaises(ValueError):
            compare(self.real / "train", self.real / "test", output=self.root / "forbidden.png", **self.roots)
        self.assertFalse((self.root / "forbidden.png").exists())

    def test_empty_missing_invalid_count_and_existing_output_are_reported(self):
        empty = self.root / "empty/train"
        empty.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "vazio"):
            compare(empty, self.synthetic / "train", output=self.root / "empty.png",
                    real_root=empty.parent, synthetic_root=self.synthetic.parent)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_compare(count=count)
        self.run_compare(count=1)
        original = (self.root / "comparison.png").read_bytes()
        with self.assertRaisesRegex(ValueError, "Saída já existe"):
            self.run_compare(count=1)
        self.assertEqual((self.root / "comparison.png").read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "ausente"):
            compare(self.root / "missing/train", self.synthetic / "train", output=self.root / "missing.png",
                    real_root=self.root / "missing", synthetic_root=self.synthetic.parent)

    def test_missing_csv_means_unknown_labels_without_inference(self):
        (self.synthetic / "labels.csv").unlink()
        report = self.run_compare(count=1)
        self.assertEqual(report["synthetic"]["labeled"], 0)
        self.assertEqual(report["synthetic"]["label_lengths"], {})
        self.assertIsNone(report["synthetic"]["labels_csv"])

    def assert_counts_consistent(self, stats):
        self.assertEqual(stats["labeled"] + stats["labeled_in_review"] + stats["unlabeled_or_unknown"],
                         stats["available"])
        self.assertLessEqual(stats["unlabeled_in_review"], stats["unlabeled_or_unknown"])

    def test_confirmed_and_empty_labels_without_state_keep_previous_statistics(self):
        self.assertFalse((self.real / "labeling_state.json").exists())
        stats = self.run_compare(count=1)["real_train"]
        self.assertEqual(stats["labeled"], 2)
        self.assertEqual(stats["labeled_in_review"], 0)
        self.assertEqual(stats["unlabeled_or_unknown"], 1)
        self.assertEqual(stats["unlabeled_in_review"], 0)
        self.assertEqual(stats["label_lengths"], {3: 1, 18: 1})
        self.assert_counts_consistent(stats)

    def test_filled_and_empty_review_labels_are_separate_and_excluded_from_ground_truth(self):
        baseline = self.run_compare("baseline.png", count=1)
        store = LabelStore(self.real)
        store.set_review("sample-0.png")  # Filled, but no longer confirmed.
        store.set_review("sample-1.png")  # Empty; still part of the unlabeled count.
        originals = {p: p.read_bytes() for p in self.real.rglob("*") if p.is_file()}
        report = self.run_compare(count=1)
        stats = report["real_train"]
        self.assertEqual(stats["labeled"], 1)
        self.assertEqual(stats["labeled_in_review"], 1)
        self.assertEqual(stats["unlabeled_or_unknown"], 1)
        self.assertEqual(stats["unlabeled_in_review"], 1)
        self.assertEqual(stats["label_lengths"], {18: 1})
        self.assertEqual(set(stats["characters"]), set("LongLabel123456789"))
        self.assertNotIn("B", stats["characters"])
        self.assertNotIn("0", stats["characters"])
        self.assert_counts_consistent(stats)
        self.assertEqual(report["synthetic"], baseline["synthetic"])
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_only_empty_review_label_does_not_reduce_confirmed_or_double_count(self):
        LabelStore(self.real).set_review("sample-1.png")
        stats = self.run_compare(count=1)["real_train"]
        self.assertEqual((stats["labeled"], stats["labeled_in_review"], stats["unlabeled_or_unknown"],
                          stats["unlabeled_in_review"]), (2, 0, 1, 1))
        self.assertEqual(stats["label_lengths"], {3: 1, 18: 1})
        self.assert_counts_consistent(stats)

    def test_review_requires_same_split_filename_and_hash_as_labeler(self):
        store = LabelStore(self.real)
        row = store.selected[0]
        wrong = [dict(row, split="validation"), dict(row, sha256="0" * 64), dict(row, filename="other.png")]
        state = {"version": 1, "review": {LabelStore.identity(r): True for r in wrong}}
        (self.real / "labeling_state.json").write_text(json.dumps(state), encoding="utf-8")
        stats = self.run_compare(count=1)["real_train"]
        self.assertEqual(stats["labeled"], 2)
        self.assertEqual(stats["labeled_in_review"], 0)
        self.assert_counts_consistent(stats)

    def test_corrupt_review_state_is_reported_without_writing_outputs(self):
        state = self.real / "labeling_state.json"
        for invalid in (b"{", b'[]', b'{"version": 2, "review": {}}',
                        b'{"version": 1, "review": []}', b'{"version": 1, "review": {"x": false}}'):
            with self.subTest(invalid=invalid):
                state.write_bytes(invalid)
                with self.assertRaisesRegex(ValueError, "labeling_state.json"):
                    self.run_compare(count=1)
                self.assertEqual(state.read_bytes(), invalid)
                self.assert_no_outputs_or_temps()

    def test_default_and_pilot_under_authorized_synthetic_root_are_accepted(self):
        default_train = self.synthetic.parent / "train"
        default_train.mkdir()
        Image.new("RGB", (60, 20), "blue").save(default_train / "fixture.png")
        for index, directory in enumerate((default_train, self.synthetic / "train")):
            with self.subTest(directory=directory):
                report = compare(self.real / "train", directory, count=1,
                                 output=self.root / f"allowed-{index}.png", **self.roots)
                self.assertEqual(report["pairs"], 1)
                self.assertEqual(Path(report["synthetic"]["directory"]), directory.resolve())

    def test_forbidden_synthetic_origins_fail_before_any_image_enumeration(self):
        directories = [self.real / split for split in ("train", "validation", "test")]
        directories += [self.root / split for split in ("train", "validation", "test")]
        directories += [self.synthetic / "../../test", self.synthetic / "../../real/train",
                        self.root / "synthetic-other/train"]
        for directory in directories:
            for candidate in (directory.absolute(), Path(os.path.relpath(directory))):
                with self.subTest(directory=candidate), patch("inspection.compare.image_paths") as read:
                    with self.assertRaises(ValueError):
                        compare(self.real / "train", candidate, output=self.root / "forbidden.png", **self.roots)
                    read.assert_not_called()

    def test_real_input_requires_authorized_train_root_not_just_folder_name(self):
        with patch("inspection.compare.image_paths") as read:
            with self.assertRaisesRegex(ValueError, "raiz real autorizada"):
                compare(self.root / "other/train", self.synthetic / "train",
                        output=self.root / "forbidden.png", **self.roots)
            read.assert_not_called()

    def test_symlink_escape_from_synthetic_tree_is_rejected_before_reading(self):
        alias = self.synthetic.parent / "alias"
        for target in (self.real / "test", self.real / "train", self.root / "test", self.root / "validation"):
            with self.subTest(target=target):
                target.mkdir(exist_ok=True)
                try:
                    alias.symlink_to(target.resolve(), target_is_directory=True)
                except OSError as error:
                    self.skipTest(f"Ambiente sem suporte/permissão para criar symlinks: {error}")
                try:
                    with patch("inspection.compare.image_paths") as read:
                        with self.assertRaises(ValueError):
                            compare(self.real / "train", alias, output=self.root / "forbidden.png", **self.roots)
                        read.assert_not_called()
                finally:
                    alias.unlink()

    def test_alias_to_legitimate_pilot_and_relative_dot_paths_are_accepted(self):
        alias = self.synthetic.parent / "alias"
        try:
            alias.symlink_to((self.synthetic / "train").resolve(), target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Ambiente sem suporte/permissão para criar symlinks: {error}")
        for directory in (alias, Path(os.path.relpath(alias)), self.synthetic / "../pilot/train"):
            with self.subTest(directory=directory):
                real, synthetic = allowed_directories(Path(os.path.relpath(self.real / "train")),
                                                      directory, **self.roots)
                self.assertEqual(real, (self.real / "train").resolve())
                self.assertEqual(synthetic, (self.synthetic / "train").resolve())

    def assert_no_outputs_or_temps(self):
        self.assertFalse((self.root / "comparison.png").exists())
        self.assertFalse((self.root / "comparison.json").exists())
        self.assertEqual(list(self.root.glob(".comparison-*.tmp")), [])

    def test_png_write_failure_cleans_temporary_and_allows_retry(self):
        def fail(image, stream, **options):
            stream.write(b"partial PNG")
            raise OSError("PNG write failure")
        with patch("inspection.compare.Image.Image.save", new=fail):
            with self.assertRaisesRegex(OSError, "PNG write failure"):
                self.run_compare(count=1)
        self.assert_no_outputs_or_temps()
        self.assertEqual(self.run_compare(count=1)["pairs"], 1)

    def test_json_staging_failure_cleans_both_temporaries_and_allows_retry(self):
        def fail(report, stream, **options):
            self.assertEqual(len(list(self.root.glob(".comparison-*.tmp"))), 2)
            stream.write('{"partial":')
            raise OSError("JSON write failure")
        with patch("inspection.compare.json.dump", side_effect=fail):
            with self.assertRaisesRegex(OSError, "JSON write failure"):
                self.run_compare(count=1)
        self.assert_no_outputs_or_temps()
        self.assertEqual(self.run_compare(count=1)["pairs"], 1)

    def test_partial_png_publish_failure_removes_own_file_and_allows_retry(self):
        def fail(source, destination):
            destination.write(b"partial PNG")
            raise OSError("PNG publish failure")
        with patch("inspection.compare.shutil.copyfileobj", side_effect=fail):
            with self.assertRaisesRegex(OSError, "PNG publish failure"):
                self.run_compare(count=1)
        self.assert_no_outputs_or_temps()
        self.assertEqual(self.run_compare(count=1)["pairs"], 1)

    def test_json_publish_failure_removes_published_png_partial_json_and_temporaries(self):
        copy = shutil.copyfileobj
        calls = []
        def fail(source, destination):
            calls.append(destination.name)
            if len(calls) == 2:
                with Image.open(self.root / "comparison.png") as image:
                    image.verify()
                destination.write(b"partial JSON")
                raise OSError("JSON publish failure")
            copy(source, destination)
        with patch("inspection.compare.shutil.copyfileobj", side_effect=fail):
            with self.assertRaisesRegex(OSError, "JSON publish failure"):
                self.run_compare(count=1)
        self.assertEqual(len(calls), 2)
        self.assert_no_outputs_or_temps()
        self.assertEqual(self.run_compare(count=1)["pairs"], 1)

    def test_each_preexisting_output_is_preserved_without_creating_other_files(self):
        for suffix in (".png", ".json"):
            with self.subTest(suffix=suffix):
                output = self.root / ("previous" + suffix)
                output.write_bytes(b"preexisting output")
                before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(ValueError, "Saída já existe"):
                    self.run_compare("previous.png", count=1)
                self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)
                output.unlink()

    def test_json_created_by_another_run_after_preflight_is_not_removed(self):
        copy = shutil.copyfileobj
        def concurrent_output(source, destination):
            copy(source, destination)
            (self.root / "comparison.json").write_bytes(b"other execution")
        with patch("inspection.compare.shutil.copyfileobj", side_effect=concurrent_output):
            with self.assertRaises(FileExistsError):
                self.run_compare(count=1)
        self.assertFalse((self.root / "comparison.png").exists())
        self.assertEqual((self.root / "comparison.json").read_bytes(), b"other execution")
        self.assertEqual(list(self.root.glob(".comparison-*.tmp")), [])

    def test_rollback_does_not_delete_a_final_file_replaced_by_another_run(self):
        copy = shutil.copyfileobj
        calls = []
        def replace_then_fail(source, destination):
            calls.append(destination.name)
            if len(calls) == 2:
                png = self.root / "comparison.png"
                png.rename(self.root / "moved-by-other.png")
                png.write_bytes(b"replacement from another run")
                raise OSError("publication failed")
            copy(source, destination)
        with patch("inspection.compare.shutil.copyfileobj", side_effect=replace_then_fail):
            with self.assertRaisesRegex(OSError, "publication failed"):
                self.run_compare(count=1)
        self.assertEqual((self.root / "comparison.png").read_bytes(), b"replacement from another run")
        self.assertFalse((self.root / "comparison.json").exists())
        self.assertEqual(list(self.root.glob(".comparison-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
