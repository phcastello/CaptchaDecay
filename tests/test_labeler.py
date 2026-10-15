import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from labeling.labeler import COLUMNS, ConcurrentEditError, LabelStore, main, parse_csv


def create_dataset(root):
    rows = []
    for index, (split, label, size) in enumerate([
        ("train", "aB0", (80, 30)), ("train", "", (120, 40)),
        ("train", "LongLabel123456789", (80, 30)), ("validation", "", (100, 30)),
        ("test", "", (80, 30)),
    ]):
        path = root / split / f"sample-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (30 * index, 100, 200)).save(path)
        rows.append(dict(zip(COLUMNS, [path.name, label, split, hashlib.sha256(path.read_bytes()).hexdigest(),
                                       f"original/{path.name}", "7" if index % 2 else ""])))
    write_rows(root / "labels.csv", rows)
    return rows


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class LabelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = (Path(self.temporary.name) / "real").resolve()
        self.rows = create_dataset(self.root)
        self.store = LabelStore(self.root)

    def test_load_default_excludes_test_and_starts_at_first_pending(self):
        self.assertEqual(self.store.rows, self.rows)
        self.assertEqual({r["split"] for r in self.store.selected}, {"train", "validation"})
        self.assertEqual(self.store.filtered()[0]["filename"], "sample-1.png")
        # A missing test PNG must not prevent development labeling.
        (self.root / "test/sample-4.png").unlink()
        LabelStore(self.root)

    def test_save_preserves_columns_metadata_other_labels_and_all_pngs(self):
        originals = {p: p.read_bytes() for p in self.root.glob("*/*.png")}
        self.store.save_label("sample-1.png", "  Aa 0z!  ")
        updated = parse_csv((self.root / "labels.csv").read_bytes())
        expected = [dict(r) for r in self.rows]
        expected[1]["label"] = "Aa 0z!"
        self.assertEqual(updated, expected)
        self.assertEqual(list(updated[0]), COLUMNS)
        self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_variable_lengths_case_restart_and_edit_old_label(self):
        for label in ("a", "A", "aA01234567890123456789", "a b", "Áß7"):
            with self.subTest(label=label):
                self.store.save_label("sample-0.png", label)
                restarted = LabelStore(self.root)
                self.assertEqual(restarted.selected[0]["label"], label)
                self.assertEqual(restarted.status(restarted.selected[0]), "rotulada")
        self.store.save_label("sample-0.png", "")
        restarted = LabelStore(self.root)
        self.assertEqual(restarted.selected[0]["label"], "")
        self.assertEqual(restarted.status(restarted.selected[0]), "pendente")

    def test_review_is_separate_persistent_and_does_not_create_ground_truth(self):
        original = (self.root / "labels.csv").read_bytes()
        self.store.set_review("sample-1.png")
        restarted = LabelStore(self.root)
        row = restarted.filtered("Revisão")[0]
        self.assertEqual(row["filename"], "sample-1.png")
        self.assertEqual(row["label"], "")
        self.assertEqual((self.root / "labels.csv").read_bytes(), original)
        self.assertNotIn(row, restarted.filtered())
        restarted.set_review(row["filename"], False)
        self.assertEqual(LabelStore(self.root).filtered("Revisão"), [])

    def test_review_identity_includes_hash_and_split(self):
        self.store.set_review("sample-1.png")
        path = self.root / "train/sample-1.png"
        Image.new("RGB", (120, 40), "black").save(path)
        self.rows[1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_rows(self.root / "labels.csv", self.rows)
        restarted = LabelStore(self.root)
        self.assertEqual(restarted.status(restarted.selected[1]), "pendente")

    def test_labeling_test_requires_explicit_authorization_in_store_and_cli(self):
        with self.assertRaisesRegex(ValueError, "allow-test-labeling"):
            LabelStore(self.root, ["test"])
        allowed = LabelStore(self.root, ["test"], allow_test=True)
        self.assertEqual(len(allowed.selected), 1)
        allowed.save_label("sample-4.png", "FixtureOnly")
        with self.assertRaisesRegex(ValueError, "autorizados"):
            self.store.save_label("sample-4.png", "Forbidden")
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as raised:
                main(["--dataset", str(self.root), "--split", "test"])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("allow-test-labeling", stderr.getvalue())

    def test_atomic_save_validates_before_replacing_in_same_directory(self):
        original = self.store.csv_path.read_bytes()
        replace = os.replace
        seen = []
        def observe(source, target):
            self.assertEqual(Path(source).parent, self.root)
            self.assertEqual(Path(target), self.store.csv_path)
            self.assertEqual(self.store.csv_path.read_bytes(), original)
            self.assertEqual(parse_csv(Path(source).read_bytes())[1]["label"], "NewLabel")
            seen.append(source)
            replace(source, target)
        with patch("labeling.labeler.os.replace", side_effect=observe), patch("labeling.labeler.os.fsync", wraps=os.fsync) as sync:
            self.store.save_label("sample-1.png", "NewLabel")
        self.assertEqual(len(seen), 1)
        sync.assert_called_once()
        self.assertEqual(list(self.root.glob(".labeling-*.tmp")), [])

    def test_failed_replace_preserves_original_and_memory_and_removes_own_temp(self):
        original = self.store.csv_path.read_bytes()
        with patch("labeling.labeler.os.replace", side_effect=PermissionError("fixture failure")):
            with self.assertRaises(PermissionError):
                self.store.save_label("sample-1.png", "NotSaved")
        self.assertEqual(self.store.csv_path.read_bytes(), original)
        self.assertEqual(self.store.selected[1]["label"], "")
        self.assertEqual(list(self.root.glob(".labeling*")), [])
        self.store.save_label("sample-1.png", "Retry")

    def test_external_csv_edit_is_not_overwritten_and_can_reload(self):
        self.rows[0]["label"] = "External"
        write_rows(self.store.csv_path, self.rows)
        changed = self.store.csv_path.read_bytes()
        with self.assertRaises(ConcurrentEditError):
            self.store.save_label("sample-1.png", "Local")
        self.assertEqual(self.store.csv_path.read_bytes(), changed)
        self.store.reload()
        self.store.save_label("sample-1.png", "Local")
        self.assertEqual(parse_csv(self.store.csv_path.read_bytes())[0]["label"], "External")

    def test_external_edit_while_temp_is_written_is_detected(self):
        validate = self.store.validate_written_csv
        self.rows[0]["label"] = "Concurrent"
        def external_edit(data, expected):
            validate(data, expected)
            write_rows(self.store.csv_path, self.rows)
        with patch.object(self.store, "validate_written_csv", side_effect=external_edit):
            with self.assertRaises(ConcurrentEditError):
                self.store.save_label("sample-1.png", "Local")
        self.assertEqual(parse_csv(self.store.csv_path.read_bytes()), self.rows)

    def test_external_state_edit_and_other_writer_lock_are_preserved(self):
        state = {"version": 1, "review": {self.store.identity(self.store.selected[0]): True}}
        self.store.state_path.write_text(json.dumps(state), encoding="utf-8")
        original = self.store.state_path.read_bytes()
        with self.assertRaises(ConcurrentEditError):
            self.store.set_review("sample-1.png")
        self.assertEqual(self.store.state_path.read_bytes(), original)
        lock = self.root / ".labeling.lock"
        lock.write_text("other writer", encoding="utf-8")
        with self.assertRaises(ConcurrentEditError):
            self.store.save_label("sample-1.png", "Local")
        self.assertEqual(lock.read_text(), "other writer")

    def test_missing_or_changed_png_invalid_csv_and_invalid_state_are_reported(self):
        original = self.store.csv_path.read_bytes()
        for bad in (b"filename,label\nx,A\n", original + b"broken,row\n",
                    original.replace(b"sample-0.png", b"../escape.png"),
                    original + original.splitlines(keepends=True)[1]):
            with self.subTest(csv=bad[:40]):
                self.store.csv_path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    LabelStore(self.root)
                self.assertEqual(self.store.csv_path.read_bytes(), bad)
        self.store.csv_path.write_bytes(original)
        path = self.root / "train/sample-1.png"
        path.write_bytes(b"invalid PNG")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.store.reload()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.reload()
        # Restore fixtures, then check state parsing independently.
        create_dataset(self.root)
        self.store.state_path.write_text('{"version": 1, "review": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "estado preservado"):
            self.store.reload()
        self.store.csv_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store.reload()


if __name__ == "__main__":
    unittest.main()
