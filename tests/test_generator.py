import base64
import csv
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHP = os.environ.get("PHP_BINARY") or shutil.which("php")
RNG_SCHEME = "xoshiro256starstar-sha256-v2"
RECIPE_FIELDS = ("label", "length", "width", "height", "perturbation", "num_lines",
                 "noise_level", "font_ratio", "image_bg_color", "text_color", "line_color", "render_seed")


def fixture_png(number):
    def chunk(kind, data):
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data)))
    scanline = b"\0" + bytes([number, 50, 70]) * 215
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!2I5B", 215, 80, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanline * 80)) + chunk(b"IEND", b""))


@unittest.skipUnless(PHP, "Instale PHP CLI com GD/FreeType para testar o gerador.")
class GeneratorTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run([PHP, str(ROOT / "generator/generate.php"), *map(str, arguments)],
                              cwd=ROOT, capture_output=True, text=True, timeout=30)

    def php_json(self, code, *arguments):
        result = subprocess.run([PHP, "-r", "require $argv[1]; " + code,
                                 str(ROOT / "generator/generate.php"), *map(str, arguments)],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_parsing_options_and_defaults(self):
        code = "echo json_encode(parse_options(array_slice($argv, 2)));"
        defaults = self.php_json(code)
        self.assertEqual(defaults["count"], 1000)
        self.assertEqual(defaults["seed"], 42)
        self.assertEqual(defaults["train_ratio"], 0.9)
        self.assertEqual(defaults["charset"], "abcdefghiklmnprstuvwyz23456789")
        self.assertEqual(defaults["charset"], defaults["charset"].lower())
        self.assertNotIn("0", defaults["charset"])
        self.assertNotIn("1", defaults["charset"])
        self.assertEqual(defaults["length_distribution"], {"4": 27, "5": 46, "6": 27})
        config = self.php_json(code, "--count=20", "--seed", 7, "--output", "custom",
                               "--train-ratio", 0.75, "--validation-ratio=0.25")
        self.assertEqual((config["count"], config["seed"], config["output"]), (20, 7, "custom"))
        self.assertEqual(config["validation_ratio"], 0.25)

    def test_help_and_invalid_cli_options(self):
        self.assertEqual(self.run_cli("--help").returncode, 0)
        for arguments in (("--count", 0), ("--count", -1), ("--count", "1.5"),
                          ("--count", "99999999999999999999999999"), ("--count",),
                          ("--seed", -1), ("--seed", "NaN"), ("--seed", 2147483648),
                          ("--train-ratio", "NaN"), ("--train-ratio", 1.2),
                          ("--train-ratio", 0.8, "--validation-ratio", 0.3),
                          ("--count", 1, "--count", 2), ("--unknown", 1)):
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 1)
                self.assertIn("Erro:", result.stderr)

    def test_seed_metadata_lengths_charset_and_visual_ranges(self):
        code = """
            $c = parse_options(array_slice($argv, 2));
            $samples = [];
            for ($i = 1; $i <= 32; ++$i) { $samples[] = sample_metadata($c, $i, 'train'); }
            echo json_encode(['config' => $c, 'samples' => $samples]);
        """
        first = self.php_json(code, "--seed", 42)
        self.assertEqual(first, self.php_json(code, "--seed", 42))
        self.assertNotEqual(first["samples"], self.php_json(code, "--seed", 7)["samples"])
        config = first["config"]
        self.assertGreater(len({row["length"] for row in first["samples"]}), 1)
        for row in first["samples"]:
            self.assertEqual(len(row["label"]), row["length"])
            self.assertTrue(config["min_length"] <= row["length"] <= config["max_length"])
            self.assertTrue(set(row["label"]) <= set(config["charset"]))
            self.assertRegex(row["filename"], r"^\d{8}\.png$")
            self.assertNotIn(row["label"], row["filename"])
            for name in ("perturbation", "num_lines", "noise_level", "font_ratio"):
                self.assertTrue(config[name][0] <= row[name] <= config[name][1])
            for name in ("image_bg_color", "text_color", "line_color"):
                self.assertIn(row[name], config[name])

    def test_split_distribution_is_deterministic_and_configurable(self):
        code = "$c = parse_options(array_slice($argv, 2)); echo json_encode(batch_splits($c, 1));"
        first = self.php_json(code, "--count", 20)
        self.assertEqual(first.count("train"), 18)
        self.assertEqual(first.count("validation"), 2)
        self.assertEqual(first, self.php_json(code, "--count", 20))
        self.assertNotEqual(first, self.php_json(code, "--count", 20, "--seed", 7))
        other = self.php_json(code, "--count", 20, "--train-ratio", 0.6, "--validation-ratio", 0.4)
        self.assertEqual(other.count("train"), 12)
        self.assertEqual(other.count("validation"), 8)

    def test_collision_regression_and_independence_from_output_split_and_mt_rand(self):
        result = self.php_json("""
            $c = parse_options([]);
            $a = sample_metadata($c, 3057, 'validation');
            $b = sample_metadata($c, 21576, 'train');
            mt_srand(7); for ($i = 0; $i < 1000; ++$i) { mt_rand(); }
            $again = sample_metadata($c, 3057, 'validation');
            $c['output'] = '/another/output';
            $other = sample_metadata($c, 3057, 'train');
            echo json_encode([$a, $b, $again, $other]);
        """)
        a, b, again, other = result
        recipe = lambda row: {key: row[key] for key in RECIPE_FIELDS}
        self.assertNotEqual(recipe(a), recipe(b))
        self.assertEqual(a, again)
        self.assertEqual(recipe(a), recipe(other))
        self.assertEqual(a["config_sha256"], other["config_sha256"])
        self.assertEqual(a["rng_scheme"], RNG_SCHEME)
        # Calibrated default recipes also tie generation to the full-digest v2 engine contract.
        self.assertEqual(tuple(a[key] for key in RECIPE_FIELDS),
                         ("p6a2i7", 6, 215, 80, 0.95753, 7, 4, 0.44851,
                          "#f6f6f6", "#808080", "#808080", 407070361))
        self.assertEqual(tuple(b[key] for key in RECIPE_FIELDS),
                         ("kis8", 4, 215, 80, 0.91457, 7, 2, 0.51288,
                          "#f6f6f6", "#606060", "#606060", 1141260154))

    def test_rng_engine_uses_all_32_binary_digest_bytes(self):
        for context in ("sample:3057", "sample:21576", "splits:1"):
            digest = hashlib.sha256(f"captchadecay:rng-v2:42:{context}".encode()).hexdigest()
            result = self.php_json(r"""
                $actual = make_rng(42, $argv[2])->engine;
                $expected = new Random\Engine\Xoshiro256StarStar(hex2bin($argv[3]));
                $streams = [[], []];
                for ($i = 0; $i < 8; ++$i) {
                    $streams[0][] = bin2hex($actual->generate());
                    $streams[1][] = bin2hex($expected->generate());
                }
                echo json_encode([get_class($actual), $streams]);
            """, context, digest)
            self.assertEqual(result[0], "Random\\Engine\\Xoshiro256StarStar")
            self.assertEqual(result[1][0], result[1][1])

    def test_thousands_of_render_recipes_do_not_repeat(self):
        samples = self.php_json("""
            $c = parse_options([]); $samples = [];
            for ($i = 1; $i <= 25000; ++$i) {
                $s = sample_metadata($c, $i, 'train');
                unset($s['filename'], $s['index']); $samples[] = $s;
            }
            echo json_encode($samples);
        """)
        recipes = {tuple(row[key] for key in RECIPE_FIELDS) for row in samples}
        self.assertEqual(len(recipes), 25000)

    def test_invalid_configuration_has_clear_error(self):
        cases = ({"min_length": 7, "max_length": 3}, {"width": 0}, {"height": 999999},
                 {"charset": ""}, {"charset": "AAB"}, {"charset": "á"},
                 {"perturbation": [0.8, 0.2]}, {"num_lines": [-1, 5]},
                 {"noise_level": [0, 11]}, {"font_ratio": [0.1, 1.5]},
                 {"text_color": ["bad"]}, {"line_color": []}, {"unknown_property": 1},
                 {"length_distribution": []}, {"length_distribution": "uniform"},
                 {"length_distribution": {4: 0, 5: 0, 6: 0}},
                 {"length_distribution": {3: 1}}, {"length_distribution": {7: 1}},
                 {"length_distribution": {4: -1, 5: 2}},
                 {"length_distribution": {4: 0.27, 5: 0.46, 6: 0.27}},
                 {"length_distribution": {4: True}}, {"length_distribution": {4: 1000001}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.php"
            for config in cases:
                with self.subTest(config=config):
                    encoded = base64.b64encode(json.dumps(config).encode()).decode()
                    path.write_text(f"<?php return json_decode(base64_decode('{encoded}'), true);", encoding="utf-8")
                    result = self.run_cli("--config", path)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("Erro:", result.stderr)
                    self.assertNotIn("Fatal error", result.stderr)

    def test_configuration_overrides_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.php"
            path.write_text("<?php return ['charset' => 'AB2', 'min_length' => 2, "
                            "'max_length' => 3, 'length_distribution' => null, 'seed' => 7, 'count' => 5];", encoding="utf-8")
            config = self.php_json("echo json_encode(parse_options(array_slice($argv, 2)));",
                                   "--config", path, "--count", 10)
            self.assertEqual(config["charset"], "AB2")
            self.assertEqual(config["count"], 10)
            self.assertEqual(config["seed"], 7)
            self.assertEqual((config["min_length"], config["max_length"]), (2, 3))
            self.assertIsNone(config["length_distribution"])

    def test_weighted_default_length_distribution_favors_five_characters(self):
        counts = self.php_json("""
            $c = parse_options([]); $counts = [4 => 0, 5 => 0, 6 => 0];
            for ($i = 1; $i <= 10000; ++$i) {
                ++$counts[sample_metadata($c, $i, 'train')['length']];
            }
            echo json_encode($counts);
        """)
        self.assertEqual(sum(counts.values()), 10000)
        self.assertGreater(counts["5"], counts["4"])
        self.assertGreater(counts["5"], counts["6"])
        for length, expected in (("4", 0.27), ("5", 0.46), ("6", 0.27)):
            self.assertAlmostEqual(counts[length] / 10000, expected, delta=0.02)

    def test_length_distribution_is_configurable_and_recorded_in_config_hash(self):
        result = self.php_json("""
            $c = parse_options([]); $original = sample_metadata($c, 1, 'train');
            $c['min_length'] = 2; $c['max_length'] = 3;
            $c['length_distribution'] = [2 => 0, 3 => 10]; $c = validate_config($c);
            $samples = [];
            for ($i = 1; $i <= 100; ++$i) { $samples[] = sample_metadata($c, $i, 'train'); }
            echo json_encode([$original, $samples, sample_metadata($c, 1, 'train')]);
        """)
        original, samples, repeated = result
        self.assertEqual({sample["length"] for sample in samples}, {3})
        self.assertTrue(all(len(sample["label"]) == 3 for sample in samples))
        self.assertEqual(samples[0], repeated)
        self.assertNotEqual(original["config_sha256"], samples[0]["config_sha256"])

    def test_null_distribution_keeps_legacy_uniform_selection_and_override_ranges(self):
        result = self.php_json("""
            $c = parse_options([]); $c['length_distribution'] = null;
            $c['min_length'] = 2; $c['max_length'] = 3; $c = validate_config($c);
            $actual = []; $expected = [];
            for ($i = 1; $i <= 200; ++$i) {
                $actual[] = sample_metadata($c, $i, 'train')['length'];
                $expected[] = make_rng($c['seed'], "sample:$i")->getInt(2, 3);
            }
            echo json_encode([$actual, $expected]);
        """)
        self.assertEqual(result[0], result[1])
        self.assertEqual(set(result[0]), {2, 3})

    def assert_png(self, path):
        data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack("!II", data[16:24]), (215, 80))
        position = 8
        kinds = []
        while position < len(data):
            size = struct.unpack("!I", data[position:position + 4])[0]
            kind = data[position + 4:position + 8]
            payload = data[position + 8:position + 8 + size]
            checksum = struct.unpack("!I", data[position + 8 + size:position + 12 + size])[0]
            self.assertEqual(zlib.crc32(kind + payload), checksum)
            kinds.append(kind)
            position += size + 12
        self.assertEqual(position, len(data))
        self.assertIn(b"IDAT", kinds)
        self.assertEqual(kinds[-1], b"IEND")

    def test_png_csv_incremental_generation_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "first"
            result = self.run_cli("--count", 20, "--seed", 42, "--output", root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "train").is_dir())
            self.assertTrue((root / "validation").is_dir())
            self.assertFalse((root / "test").exists())
            original_csv = (root / "labels.csv").read_bytes()
            rows = list(csv.DictReader(io.StringIO(original_csv.decode())))
            self.assertEqual(len(rows), 20)
            self.assertEqual(sum(row["split"] == "train" for row in rows), 18)
            images = {path: path.read_bytes() for path in root.glob("*/*.png")}
            self.assertEqual(len(images), 20)
            self.assertEqual(len({row["filename"] for row in rows}), 20)
            for row in rows:
                path = root / row["split"] / row["filename"]
                self.assert_png(path)
                self.assertEqual(int(row["length"]), len(row["label"]))
                self.assertNotIn(row["label"], row["filename"])
                self.assertEqual(row["securimage_version"], "4.0.2")
                self.assertEqual(row["rng_scheme"], RNG_SCHEME)
            self.assertEqual(len({hashlib.sha256(data).digest() for data in images.values()}), 20)
            result = self.run_cli("--count", 5, "--seed", 42, "--output", root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "labels.csv").read_bytes().startswith(original_csv))
            new_rows = list(csv.DictReader((root / "labels.csv").read_text().splitlines()))
            self.assertEqual(len(new_rows), 25)
            self.assertEqual([row["filename"] for row in new_rows[-5:]],
                             [f"{index:08d}.png" for index in range(21, 26)])
            self.assertEqual(len(list(root.glob("*/*.png"))), 25)
            self.assertEqual(len({hashlib.sha256(path.read_bytes()).digest()
                                  for path in root.glob("*/*.png")}), 25)
            for path, data in images.items():
                self.assertEqual(path.read_bytes(), data)
            second = Path(folder) / "second"
            self.assertEqual(self.run_cli("--count", 20, "--seed", 42, "--output", second).returncode, 0)
            self.assertEqual((second / "labels.csv").read_bytes(), original_csv)

    def test_orphan_png_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train" / "00000001.png"
            path.parent.mkdir()
            path.write_bytes(b"original orphan")
            result = self.run_cli("--count", 1, "--output", folder)
            self.assertEqual(result.returncode, 1)
            self.assertIn("PNG sem metadata", result.stderr)
            self.assertEqual(path.read_bytes(), b"original orphan")

    def test_incomplete_or_duplicate_csv_is_preserved_and_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.run_cli("--count", 1, "--output", folder).returncode, 0)
            path = Path(folder) / "labels.csv"
            original = path.read_bytes()
            for suffix, message in ((b"00000002.png,ABC", "Linha incompleta"),
                                    (original.splitlines(keepends=True)[1], "duplicada")):
                with self.subTest(message=message):
                    changed = original + suffix
                    path.write_bytes(changed)
                    result = self.run_cli("--count", 1, "--output", folder)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(message, result.stderr)
                    self.assertEqual(path.read_bytes(), changed)
                    self.assertEqual(len(list(Path(folder).glob("*/*.png"))), 1)

    def render_fixtures(self, root, numbers, *, train_ratio=1, same_label=False):
        encoded = json.dumps([base64.b64encode(fixture_png(n)).decode() for n in numbers])
        # Only the rendering boundary is injected; validation and persistence are real.
        return self.php_json("""
            $c = parse_options(['--output', $argv[2], '--count', $argv[3],
                '--train-ratio', $argv[4], '--validation-ratio', $argv[5]]);
            if ($argv[7] === 'yes') {
                $c['charset'] = 'A'; $c['min_length'] = $c['max_length'] = 4;
                $c['length_distribution'] = null;
            }
            $images = json_decode($argv[6], true); $n = 0;
            try {
                $saved = generate_dataset($c, function ($sample) use (&$n, $images) {
                    return base64_decode($images[$n++]);
                });
                echo json_encode(['saved' => $saved]);
            } catch (Throwable $e) { echo json_encode(['error' => $e->getMessage()]); }
        """, root, len(numbers), train_ratio, 1 - train_ratio, encoded, "yes" if same_label else "no")

    def snapshot(self, root):
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def test_new_duplicate_against_existing_same_and_other_split_preserves_data(self):
        for existing_ratio in (0, 1):
            with self.subTest(existing_ratio=existing_ratio), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.assertIn("saved", self.render_fixtures(root, [1], train_ratio=existing_ratio))
                before = self.snapshot(root)
                result = self.render_fixtures(root, [1])
                self.assertIn("Duplicata sintética", result["error"])
                self.assertIn("índice 2, split train", result["error"])
                self.assertIn(("train" if existing_ratio else "validation") + "/00000001.png", result["error"])
                self.assertIn("labels diferentes", result["error"])
                self.assertEqual(self.snapshot(root), before)

    def test_duplicate_in_one_run_preserves_progress_without_duplicate_row_or_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = self.render_fixtures(root, [1, 2, 2])
            self.assertIn("Duplicata sintética", result["error"])
            rows = list(csv.DictReader((root / "labels.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(list(root.glob("*/*.png"))), 2)
            for row, number in zip(rows, [1, 2]):
                self.assertEqual((root / row["split"] / row["filename"]).read_bytes(), fixture_png(number))
            before = self.snapshot(root)
            self.assertIn("error", self.render_fixtures(root, [2]))
            self.assertEqual(self.snapshot(root), before)

    def test_existing_duplicates_fail_before_rendering_or_migrating(self):
        for second_split in ("train", "validation"):
            with self.subTest(second_split=second_split), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.render_fixtures(root, [1, 2])
                path = root / "labels.csv"
                rows = list(csv.DictReader(path.read_text().splitlines()))
                old_path = root / rows[1]["split"] / rows[1]["filename"]
                data = fixture_png(1)
                old_path.unlink()
                (root / second_split / rows[1]["filename"]).write_bytes(data)
                rows[1]["split"] = second_split
                # Exercise the legacy reader too: inconsistent data must never migrate.
                columns = [key for key in rows[0] if key != "rng_scheme"]
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                before = self.snapshot(root)
                result = self.render_fixtures(root, [3])
                self.assertIn("train/00000001.png", result["error"])
                self.assertIn(second_split, result["error"])
                self.assertEqual(self.snapshot(root), before)

    def test_same_label_with_distinct_pngs_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = self.render_fixtures(root, [1, 2], same_label=True)
            self.assertEqual(result, {"saved": {"train": 2, "validation": 0}})
            rows = list(csv.DictReader((root / "labels.csv").read_text().splitlines()))
            self.assertEqual([row["label"] for row in rows], ["AAAA", "AAAA"])
            self.assertNotEqual((root / "train/00000001.png").read_bytes(),
                                (root / "train/00000002.png").read_bytes())

    def test_legacy_csv_migration_preserves_values_and_images_with_exact_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.render_fixtures(root, [1, 2])
            path = root / "labels.csv"
            rows = list(csv.DictReader(path.read_text().splitlines()))
            columns = [key for key in rows[0] if key != "rng_scheme"]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            original_csv = path.read_bytes()
            images = {p: p.read_bytes() for p in root.glob("*/*.png")}
            self.assertIn("saved", self.render_fixtures(root, [3]))
            backups = list(root.glob("labels.csv.pre-rng-v2-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original_csv)
            updated = list(csv.DictReader(path.read_text().splitlines()))
            self.assertEqual(len(updated), 3)
            for old, new in zip(rows, updated):
                self.assertEqual({k: new[k] for k in columns}, {k: old[k] for k in columns})
                self.assertEqual(new["rng_scheme"], "legacy-unknown")
            self.assertEqual(updated[2]["rng_scheme"], RNG_SCHEME)
            for p, data in images.items():
                self.assertEqual(p.read_bytes(), data)
            self.assertIn("saved", self.render_fixtures(root, [4]))
            self.assertEqual(list(root.glob("labels.csv.pre-rng-v2-*")), backups)

    def test_mixed_unknown_or_interrupted_csv_formats_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.render_fixtures(root, [1])
            path = root / "labels.csv"
            original = path.read_bytes()
            header, row = original.splitlines(keepends=True)
            legacy_header = header.replace(b",rng_scheme", b"")
            for changed, message in (
                (legacy_header + row, "Linha incompleta"),
                (header + row.replace(RNG_SCHEME.encode(), b""), "rng_scheme ausente"),
                (header.replace(b"rng_scheme", b"unexpected") + row, "Cabeçalho incompatível"),
                (original.rstrip(b"\r\n"), "interrompido"),
            ):
                with self.subTest(message=message):
                    path.write_bytes(changed)
                    before = self.snapshot(root)
                    result = self.render_fixtures(root, [2])
                    self.assertIn(message, result["error"])
                    self.assertEqual(self.snapshot(root), before)

    def test_display_value_and_rendering_do_not_change_pipeline_metadata(self):
        result = self.php_json("""
            class Securimage {
                const SI_IMAGE_PNG = 3;
                public static array $options;
                public function __construct($options) { self::$options = $options; }
                public function show() { for ($i = 0; $i < 100; ++$i) { mt_rand(); } echo 'stub'; }
            }
            $c = parse_options([]); $a = sample_metadata($c, 3057, 'train');
            render_png(sample_metadata($c, 2, 'train'));
            render_png($a); $options = Securimage::$options;
            $again = sample_metadata($c, 3057, 'train');
            echo json_encode([$a, $again, $options]);
        """)
        sample, again, options = result
        self.assertEqual(sample, again)
        self.assertEqual(options["display_value"], sample["label"])
        self.assertEqual(options["image_width"], sample["width"])
        self.assertEqual(options["image_height"], sample["height"])
        for key in ("perturbation", "num_lines", "noise_level", "font_ratio",
                    "image_bg_color", "text_color", "line_color"):
            self.assertEqual(options[key], sample[key])


if __name__ == "__main__":
    unittest.main()
