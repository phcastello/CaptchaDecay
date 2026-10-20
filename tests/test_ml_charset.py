import copy
import unittest

from ml.charset import Charset


class CharsetTests(unittest.TestCase):
    def test_charset_deterministic_independent_of_label_order(self):
        first = Charset.from_labels(["bA2", "aaa", "24b"])
        second = Charset.from_labels(["24b", "bA2", "aaa"])
        self.assertEqual(first.characters, "24Aab")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_blank_reserved_and_class_count(self):
        charset = Charset("ab")
        self.assertEqual(charset.blank, 0)
        self.assertEqual(charset.num_classes, 3)
        self.assertEqual(charset.encode("ab"), [1, 2])
        self.assertEqual(charset.decode([0, 1, 0, 2]), "ab")

    def test_exact_roundtrip_variable_lengths_and_repeated_characters(self):
        charset = Charset.from_labels(["Aba4", "aaaa4", "4bbA44"])
        for label in ("Aba4", "aaaa4", "4bbA44"):
            with self.subTest(label=label):
                self.assertEqual(charset.decode(charset.encode(label)), label)

    def test_unknown_character_identifies_image_and_character(self):
        with self.assertRaisesRegex(ValueError, "Unknown character 'A'.*sample.png"):
            Charset("abc").encode("abA", context="sample.png")

    def test_no_implicit_case_conversion(self):
        with self.assertRaisesRegex(ValueError, "Unknown character 'A'"):
            Charset("a").encode("A")

    def test_invalid_charsets_and_empty_labels_rejected(self):
        for characters in ("", "aa", ["ab"], [1]):
            with self.subTest(characters=characters), self.assertRaises(ValueError):
                Charset(characters)
        with self.assertRaisesRegex(ValueError, "Empty"):
            Charset("ab").encode("")
        for labels in ([], ["a", ""], [None]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                Charset.from_labels(labels)

    def test_unknown_index_rejected(self):
        for index in (-1, 3):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Unknown character index"):
                Charset("ab").decode([index])
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            Charset("ab").decode([1.5])

    def test_checkpoint_roundtrip_preserves_mapping(self):
        charset = Charset("zA2")
        restored = Charset.from_dict(charset.to_dict())
        self.assertEqual(restored.characters, "zA2")
        self.assertEqual(restored.encode("Az2"), [2, 1, 3])

    def test_checkpoint_rejects_changed_mapping_or_blank(self):
        charset = Charset("ab")
        state = copy.deepcopy(charset.to_dict())
        state["char_to_index"]["a"] = 0
        with self.assertRaisesRegex(ValueError, "mappings"):
            Charset.from_dict(state)
        state = charset.to_dict()
        state["blank"] = 2
        with self.assertRaisesRegex(ValueError, "blank"):
            Charset.from_dict(state)


if __name__ == "__main__":
    unittest.main()
