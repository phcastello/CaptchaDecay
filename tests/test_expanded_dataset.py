"""Small synthetic fixtures exercise preparation without generating a large dataset."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
import zlib

from scripts import prepare_expanded_dataset as preparation


ROOT = Path(__file__).resolve().parents[1]
PHP = os.environ.get("PHP_BINARY") or shutil.which("php")
COLUMNS = [
    "filename", "label", "split", "length", "width", "height", "perturbation", "num_lines",
    "noise_level", "font_ratio", "image_bg_color", "text_color", "line_color", "seed", "index",
    "render_seed", "config_sha256", "securimage_version", "rng_scheme",
]


def fixture_png(number):
    def chunk(kind, data):
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data)))
    scanline = b"\0" + bytes([number, 50, 70]) * 215
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!2I5B", 215, 80, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanline * 80)) + chunk(b"IEND", b""))


def recipe(index, split="train", seed=42):
    return dict(zip(COLUMNS, [f"{index:08d}.png", f"ab{index}", split, 3, 215, 80,
        0.95, 5, 2, 0.45, "#ffffff", "#606060", "#808080", seed, index,
        seed * 1000 + index, "visual-fixture-sha", "4.0.2", "xoshiro256starstar-sha256-v2"]))


def append_fixture(root, row):
    (root / row["split"] / row["filename"]).write_bytes(fixture_png(int(row["index"])))
    with (root / "labels.csv").open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, COLUMNS).writerow(row)


def create_dataset(root, extra=0):
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True)
    with (root / "labels.csv").open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, COLUMNS).writeheader()
    for index in range(1, 4 + extra):
        append_fixture(root, recipe(index, "validation" if index == 3 else "train"))
    return root


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): preparation.file_hash(path)
            for path in root.rglob("*") if path.is_file()}


class ExpandedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = create_dataset(self.base / "source")
        self.destination = self.base / "expanded"
        self.experiment = self.base / "experiment"
        self.run = self.base / "run"
        self.run.mkdir()
        self.repo = self.base / "repo"
        for relative in ("generator/generate.php", "generator/config.php", "composer.lock",
                         "vendor/dapphp/securimage/securimage.php", "vendor/dapphp/securimage/AHGBold.ttf"):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture asset {relative}", encoding="utf-8")
        for name in ("best.pt", "last.pt", "history.csv"):
            (self.run / name).write_bytes(f"immutable fixture {name}".encode())
        (self.run / "run_config.json").write_text(json.dumps({
            "dataset": preparation.inspect_dataset(self.source).identity(3),
        }), encoding="utf-8")
        self.info_patch = mock.patch.object(preparation, "_generator_info", side_effect=self.generator_info)
        self.info_patch.start()
        self.addCleanup(self.info_patch.stop)
        self.generator_patch = mock.patch.object(preparation, "_run_generation", side_effect=self.generate)
        self.generator = self.generator_patch.start()
        self.addCleanup(self.generator_patch.stop)

    def generator_info(self, repo, php, source, baseline_count, target_count, seed, destination):
        rows = [recipe(index, "validation" if index == 3 else "train", seed)
                for index in range(1, target_count + 1)]
        return {
            "config": {"count": target_count - baseline_count, "seed": seed,
                       "train_ratio": 1, "validation_ratio": 0, "output": str(destination)},
            "visual": {"width": 215, "height": 80}, "runtime": {"php": "fixture"},
            "rng_scheme": "xoshiro256starstar-sha256-v2", "columns": COLUMNS, "recipes": rows,
        }

    def generate(self, repo, php, destination, count, seed):
        first = len(preparation.inspect_dataset(destination).rows) + 1
        for index in range(first, first + count):
            append_fixture(destination, recipe(index, seed=seed))

    def prepare(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return preparation.prepare_dataset(self.source, self.destination, self.experiment, self.run,
                repo=self.repo, baseline_train=2, baseline_validation=1, target_train=5, **kwargs)

    def test_independent_copy_counts_labels_metadata_hashes_and_originals_preserved(self):
        original = tree_hashes(self.source)
        checkpoint_hashes = tree_hashes(self.run)
        report = self.prepare()
        expanded = preparation.inspect_dataset(self.destination)
        self.assertEqual(expanded.counts, {"train": 5, "validation": 1})
        self.assertEqual(len(expanded.rows), 6)
        self.assertEqual(report["original_pngs_verified"], 3)
        self.assertEqual(report["validation_pngs_byte_identical"], 1)
        self.assertEqual(report["exact_duplicate_count"], 0)
        self.assertEqual(tree_hashes(self.source), original)
        self.assertEqual(tree_hashes(self.run), checkpoint_hashes)
        snapshot = self.experiment / "baseline-dataset"
        self.assertEqual(tree_hashes(snapshot), original)
        for relative in original:
            self.assertEqual((self.source / relative).read_bytes(), (snapshot / relative).read_bytes())
            self.assertFalse(os.path.samefile(self.source / relative, snapshot / relative))
            self.assertFalse(os.path.samefile(self.source / relative, self.destination / relative))
        self.assertTrue((self.destination / "labels.csv").read_bytes().startswith(
            (self.source / "labels.csv").read_bytes()))
        self.assertEqual(expanded.rows[:3], preparation.inspect_dataset(self.source).rows)
        manifest = json.loads((self.experiment / "baseline-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["baseline_artifacts_sha256"]["best.pt"], checkpoint_hashes["best.pt"])
        self.assertEqual(manifest["generation"]["rng_scheme"], "xoshiro256starstar-sha256-v2")

    def test_complete_rerun_and_verify_only_are_read_only_and_do_not_generate(self):
        first = self.prepare()
        hashes = tree_hashes(self.base)
        self.assertEqual(self.prepare(), first)
        self.assertEqual(self.prepare(verify_only=True), first)
        self.assertEqual(tree_hashes(self.base), hashes)
        self.assertEqual(self.generator.call_count, 1)

    def test_verify_only_does_not_create_missing_preparation(self):
        before = tree_hashes(self.base)
        with self.assertRaisesRegex(preparation.PreparationError, "Verificação requer"):
            self.prepare(verify_only=True)
        self.assertEqual(tree_hashes(self.base), before)
        self.assertFalse(self.experiment.exists())
        self.assertFalse(self.destination.exists())
        self.generator.assert_not_called()

    def test_recovers_proven_append_only_baseline_without_copying_extra_source_images(self):
        for index in range(4, 7):
            append_fixture(self.source, recipe(index))
        source_hashes = tree_hashes(self.source)
        self.prepare()
        self.generator.assert_called_once()
        self.assertEqual(self.generator.call_args.args[3], 3)
        self.assertEqual(tree_hashes(self.source), source_hashes)
        snapshot = preparation.inspect_dataset(self.experiment / "baseline-dataset")
        self.assertEqual(snapshot.counts, {"train": 2, "validation": 1})
        self.assertEqual(tree_hashes(self.source), tree_hashes(self.destination))

    def test_unproven_csv_prefix_is_rejected_before_copy_or_generation(self):
        path = self.source / "labels.csv"
        # Even harmless CSV newline changes invalidate the original run identity.
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        with self.assertRaisesRegex(preparation.PreparationError, "identidade registrada"):
            self.prepare()
        self.assertFalse(self.destination.exists())
        self.generator.assert_not_called()

    def test_reference_manifest_is_immutable_when_source_or_checkpoint_changes(self):
        self.prepare()
        manifest = (self.experiment / "baseline-manifest.json").read_bytes()
        (self.run / "best.pt").write_bytes(b"different checkpoint")
        before = tree_hashes(self.base)
        with self.assertRaisesRegex(preparation.PreparationError, "Manifesto existente diverge"):
            self.prepare()
        self.assertEqual((self.experiment / "baseline-manifest.json").read_bytes(), manifest)
        self.assertEqual(tree_hashes(self.base), before)

    def test_interrupted_generation_is_diagnosed_and_never_automatically_appended(self):
        def interrupted(repo, php, destination, count, seed):
            append_fixture(destination, recipe(4))
            raise preparation.PreparationError("interrupção simulada")
        self.generator.side_effect = interrupted
        with self.assertRaisesRegex(preparation.PreparationError, "interrupção simulada"):
            self.prepare()
        before = tree_hashes(self.base)
        self.generator.side_effect = self.generate
        with self.assertRaisesRegex(preparation.PreparationError, "Expansão interrompida: 4/6"):
            self.prepare()
        self.assertEqual(tree_hashes(self.base), before)
        self.assertEqual(self.generator.call_count, 1)

    def test_incomplete_copy_and_orphan_png_are_preserved(self):
        self.destination.mkdir()
        (self.destination / "train").mkdir()
        orphan = self.destination / "train" / "00000004.png"
        orphan.write_bytes(fixture_png(4))
        before = tree_hashes(self.destination)
        with self.assertRaisesRegex(preparation.PreparationError, "Preparação incompleta"):
            self.prepare()
        self.assertEqual(tree_hashes(self.destination), before)
        self.generator.assert_not_called()

    def test_fully_copied_baseline_can_start_generation_safely(self):
        shutil.copytree(self.source, self.destination)
        self.prepare()
        self.generator.assert_called_once()
        self.assertEqual(preparation.inspect_dataset(self.destination).counts, {"train": 5, "validation": 1})

    def test_byte_changed_validation_fails_even_with_valid_png_and_matching_label(self):
        self.prepare()
        validation = self.destination / "validation" / "00000003.png"
        validation.write_bytes(fixture_png(120))
        before = tree_hashes(self.base)
        with self.assertRaisesRegex(preparation.PreparationError, "byte a byte idênticos"):
            self.prepare()
        self.assertEqual(tree_hashes(self.base), before)

    def test_modified_original_training_png_and_validation_label_are_rejected(self):
        self.prepare()
        path = self.destination / "train" / "00000001.png"
        path.write_bytes(fixture_png(121))
        with self.assertRaisesRegex(preparation.PreparationError, "byte a byte idênticos"):
            self.prepare()
        path.write_bytes((self.source / "train" / "00000001.png").read_bytes())
        csv_path = self.destination / "labels.csv"
        csv_path.write_bytes(csv_path.read_bytes().replace(b",ab3,validation,", b",zz3,validation,"))
        with self.assertRaisesRegex(preparation.PreparationError, "byte a byte idênticos"):
            self.prepare()

    def test_duplicate_hashes_missing_references_blank_labels_and_unlisted_png_rejected(self):
        self.prepare()
        pristine = (self.destination / "labels.csv").read_bytes()
        path = self.destination / "train" / "00000006.png"
        original = path.read_bytes()
        variants = (
            ("Duplicata exata", lambda: path.write_bytes((self.destination / "validation" / "00000003.png").read_bytes())),
            ("Arquivo ausente", lambda: path.unlink()),
            ("Nome/split/label", lambda: (self.destination / "labels.csv").write_bytes(pristine.replace(b",ab6,train,", b",,train,"))),
            ("PNGs sem referência", lambda: (self.destination / "train" / "00000007.png").write_bytes(fixture_png(7))),
        )
        for expected, corrupt in variants:
            with self.subTest(expected=expected):
                corrupt()
                with self.assertRaisesRegex(preparation.PreparationError, expected):
                    preparation.inspect_dataset(self.destination)
                path.write_bytes(original)
                (self.destination / "labels.csv").write_bytes(pristine)
                extra = self.destination / "train" / "00000007.png"
                if extra.exists():
                    extra.unlink()

    def test_new_sample_visual_or_rng_metadata_change_is_rejected(self):
        self.prepare()
        path = self.destination / "labels.csv"
        contents = path.read_bytes()
        path.write_bytes(contents.replace(b"ab6,train,3,215,80,0.95", b"ab6,train,3,215,80,0.91"))
        with self.assertRaisesRegex(preparation.PreparationError, "configuração visual divergente"):
            self.prepare()
        path.write_bytes(contents.replace(b"42006,visual-fixture-sha", b"42007,visual-fixture-sha"))
        with self.assertRaisesRegex(preparation.PreparationError, "Receita RNG"):
            self.prepare()

    def test_repeated_expansion_in_separate_directories_has_identical_bytes(self):
        self.prepare()
        first = tree_hashes(self.destination)
        self.destination = self.base / "expanded_second"
        self.experiment = self.base / "experiment_second"
        self.prepare()
        self.assertEqual(tree_hashes(self.destination), first)
        self.assertEqual(self.generator.call_count, 2)

    def test_hardlinks_are_rejected(self):
        original = self.source / "train" / "00000001.png"
        link = self.base / "linked.png"
        try:
            os.link(original, link)
        except OSError as error:
            self.skipTest(f"Hardlinks indisponíveis neste ambiente: {error}")
        with self.assertRaisesRegex(preparation.PreparationError, "Hardlink"):
            self.prepare()
        self.generator.assert_not_called()

    def test_source_destination_overlap_and_real_paths_are_rejected(self):
        self.destination = self.source / "child"
        with self.assertRaisesRegex(preparation.PreparationError, "sobrepor a fonte"):
            self.prepare()
        self.source = self.base / "dataset" / "real" / "test"
        with self.assertRaisesRegex(preparation.PreparationError, "dataset/real"):
            self.prepare()
        self.generator.assert_not_called()

    def test_no_output_path_may_overlap_protected_baseline_run(self):
        original = tree_hashes(self.run)
        for output in ("destination", "experiment"):
            for protected_path in (self.run, self.run / "child", self.run.parent):
                with self.subTest(output=output, protected_path=protected_path):
                    original_output = getattr(self, output)
                    setattr(self, output, protected_path)
                    try:
                        with self.assertRaisesRegex(preparation.PreparationError, "sobrepor"):
                            self.prepare()
                    finally:
                        setattr(self, output, original_output)
        self.assertEqual(tree_hashes(self.run), original)
        self.generator.assert_not_called()

    def test_split_link_is_rejected_before_opening_any_png(self):
        read_bytes = Path.read_bytes
        def reject_png_read(path):
            if path.suffix == ".png":
                self.fail("Um PNG foi lido antes de rejeitar o split com link.")
            return read_bytes(path)
        with mock.patch.object(Path, "is_symlink", autospec=True,
                               side_effect=lambda path: path.name == "train"), \
             mock.patch.object(Path, "read_bytes", autospec=True, side_effect=reject_png_read):
            with self.assertRaisesRegex(preparation.PreparationError, "link/junction"):
                self.prepare()
        self.generator.assert_not_called()


@unittest.skipUnless(PHP, "PHP CLI não encontrado; configure PHP_BINARY.")
class ExpandedDatasetPhpTests(unittest.TestCase):
    def test_php_arguments_recipe_and_render_reproduce_after_independent_baseline_copies(self):
        # Three originals plus two new images in each copy: never a full dataset.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            first, second = base / "first", base / "second"
            code = r'''
require $argv[1]; require $argv[2];
$config = parse_options(['--count','3','--seed','42','--output',$argv[3],
                         '--train-ratio','0.6666666666666666','--validation-ratio','0.3333333333333333']);
generate_dataset($config);
'''
            result = subprocess.run([PHP, "-r", code, str(ROOT / "generator/generate.php"),
                                     str(ROOT / "vendor/autoload.php"), str(source)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = preparation.inspect_dataset(source)
            shutil.copytree(source, first)
            shutil.copytree(source, second)
            info = preparation._generator_info(ROOT, PHP, original, 3, 5, 42, first)
            self.assertEqual(info["config"]["train_ratio"], 1)
            self.assertEqual(info["config"]["validation_ratio"], 0)
            self.assertEqual(info["recipes"][3]["index"], 4)
            self.assertEqual(info["recipes"][4]["index"], 5)
            for destination in (first, second):
                with contextlib.redirect_stdout(io.StringIO()):
                    preparation._run_generation(ROOT, PHP, destination, 2, 42)
                expanded = preparation.inspect_dataset(destination)
                preparation._check_originals(expanded, original, 3)
                preparation._check_recipes(expanded, info["recipes"], info["columns"])
                self.assertEqual(expanded.counts, {"train": 4, "validation": 1})
            self.assertEqual(tree_hashes(first), tree_hashes(second))
            self.assertEqual(preparation.inspect_dataset(source).identity(), original.identity())


if __name__ == "__main__":
    unittest.main()
