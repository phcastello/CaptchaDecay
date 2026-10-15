import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import captcha_dataset as dataset
from generator.organize_real import COLUMNS, organize_real
from test_captcha_dataset import png, response


class OrganizeRealTests(unittest.TestCase):
    def create_originals(self, root, count=10):
        paths = []
        for index in range(count):
            path = root / ("train" if index < 7 else "validation") / f"source_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(png(index))
            paths.append(path)
        return paths

    def read_labels(self, root):
        with (root / "labels.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_deterministic_split_preserves_originals_and_blank_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            originals = {path: path.read_bytes() for path in self.create_originals(source)}
            self.assertEqual(organize_real(source, output, seed=42), {"train": 6, "validation": 2, "test": 2})
            rows = self.read_labels(output)
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["label"] == "" for row in rows))
            self.assertEqual(len({row["filename"] for row in rows}), 10)
            self.assertEqual(len({row["sha256"] for row in rows}), 10)
            self.assertEqual({row["sha256"] for row in rows},
                             {hashlib.sha256(data).hexdigest() for data in originals.values()})
            for row in rows:
                path = output / row["split"] / row["filename"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
                self.assertEqual(row["split_seed"], "42")
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data)
            second = Path(folder) / "real_again"
            organize_real(source, second, seed=42)
            self.assertEqual(self.read_labels(second), rows)
            different = Path(folder) / "different"
            organize_real(source, different, seed=7)
            self.assertNotEqual([row["split"] for row in self.read_labels(different)],
                                [row["split"] for row in rows])

    def test_incremental_organization_preserves_manual_labels_and_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            self.create_originals(source)
            organize_real(source, output)
            rows = self.read_labels(output)
            rows[0]["label"] = "Manual42"
            with (output / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            previous = {path: path.read_bytes() for path in output.glob("*/*.png")}
            (source / "train" / "new.png").write_bytes(png(99))
            organize_real(source, output, seed=7)
            updated = self.read_labels(output)
            self.assertEqual(len(updated), 11)
            mapping = {row["filename"]: row for row in updated}
            added = next(row for row in updated if row["filename"] not in {r["filename"] for r in rows})
            self.assertEqual(added["split_seed"], "7")
            for row in rows:
                self.assertEqual(mapping[row["filename"]], row)
            for path, data in previous.items():
                self.assertEqual(path.read_bytes(), data)

    def test_duplicate_originals_remain_but_have_one_canonical_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            paths = self.create_originals(source)
            duplicate = source / "validation" / "duplicate.png"
            duplicate.write_bytes(paths[0].read_bytes())
            organize_real(source, output)
            self.assertTrue(duplicate.exists())
            self.assertEqual(len(self.read_labels(output)), 10)
            self.assertEqual(len(list(output.glob("*/*.png"))), 10)

    def test_new_collected_image_can_be_indexed_without_moving_existing_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            self.create_originals(source)
            organize_real(source, output)
            data = png(99)
            path = output / "test" / (hashlib.sha256(data).hexdigest() + ".png")
            path.write_bytes(data)
            organize_real(source, output)
            row = next(row for row in self.read_labels(output) if row["filename"] == path.name)
            self.assertEqual(row["label"], "")
            self.assertEqual(row["split"], "test")
            self.assertEqual(row["split_seed"], "")
            self.assertEqual(path.read_bytes(), data)

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_collection_seed_7_is_not_falsely_recorded_as_organizer_seed_42(self, build, sleep):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="captcha.php">'),
            *[response(png(number), "image/png") for number in range(5)],
        ]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            dataset.collect_captchas(5, output, seed=7, delay=0)
            before = {p: p.read_bytes() for p in output.glob("*/*.png")}
            organize_real(source, output, seed=42)
            rows = self.read_labels(output)
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["split_seed"] for r in rows}, {""})
            for row in rows:
                self.assertEqual((output / row["split"] / row["filename"]).read_bytes(),
                                 before[output / row["split"] / row["filename"]])
            rows[0]["label"] = "Manual7"
            with (output / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            organize_real(source, output, seed=99)
            self.assertEqual(self.read_labels(output), rows)
            self.assertEqual({p: p.read_bytes() for p in output.glob("*/*.png")}, before)

    def test_conflicts_fail_without_changing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            self.create_originals(source)
            organize_real(source, output)
            previous = (output / "labels.csv").read_bytes()
            original = next((output / "train").glob("*.png"))
            duplicate = output / "test" / original.name
            duplicate.write_bytes(original.read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicada"):
                organize_real(source, output)
            self.assertEqual((output / "labels.csv").read_bytes(), previous)
            self.assertEqual(duplicate.read_bytes(), original.read_bytes())

    def test_invalid_seed_source_and_png(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "legacy", Path(folder) / "real"
            for seed in (-1, True, 1.5, 2147483648):
                with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                    organize_real(source, output, seed=seed)
            with self.assertRaisesRegex(ValueError, "Nenhuma imagem"):
                organize_real(source, output)
            with self.assertRaisesRegex(ValueError, "diferentes"):
                organize_real(source, source)
            (source / "train").mkdir(parents=True)
            (source / "train" / "bad.png").write_bytes(b"HTML error")
            with self.assertRaisesRegex(ValueError, "não-PNG"):
                organize_real(source, output)


if __name__ == "__main__":
    unittest.main()
