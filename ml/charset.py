"""Deterministic character vocabulary, with CTC blank reserved at index zero."""

from collections.abc import Iterable, Mapping, Sequence
from operator import index as integer_index


class Charset:
    """Encode exact labels without case folding or unknown-character fallback."""

    blank = 0

    def __init__(self, characters: Sequence[str] | str):
        values = list(characters)
        if not values:
            raise ValueError("Charset must contain at least one character.")
        if any(not isinstance(char, str) or len(char) != 1 for char in values):
            raise ValueError("Each charset entry must be exactly one character.")
        if len(set(values)) != len(values):
            raise ValueError("Charset contains duplicate characters.")
        self.characters = "".join(values)
        self.char_to_index = {char: index for index, char in enumerate(values, start=1)}
        self.index_to_char = {index: char for char, index in self.char_to_index.items()}

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> "Charset":
        characters: set[str] = set()
        for label in labels:
            if not isinstance(label, str) or not label:
                raise ValueError("Cannot build charset from an empty or non-string label.")
            characters.update(label)
        return cls(sorted(characters))

    @property
    def num_classes(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str, context: str | None = None) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError(f"Empty or non-string label{self._context(context)}.")
        result = []
        for char in text:
            if char not in self.char_to_index:
                raise ValueError(f"Unknown character {char!r}{self._context(context)} in label {text!r}.")
            result.append(self.char_to_index[char])
        return result

    def decode(self, indices: Iterable[int]) -> str:
        """Decode character indices, omitting blank without collapsing repetitions."""
        result = []
        for value in indices:
            try:
                index = integer_index(value)
            except TypeError as error:
                raise ValueError(f"Character index must be an integer, got {value!r}.") from error
            if index == self.blank:
                continue
            if index not in self.index_to_char:
                raise ValueError(f"Unknown character index {index}.")
            result.append(self.index_to_char[index])
        return "".join(result)

    @staticmethod
    def _context(context: str | None) -> str:
        return f" for image {context!r}" if context is not None else ""

    def to_dict(self) -> dict:
        return {
            "characters": self.characters,
            "blank": self.blank,
            "char_to_index": dict(self.char_to_index),
            "index_to_char": {str(index): char for index, char in self.index_to_char.items()},
        }

    @classmethod
    def from_dict(cls, state: Mapping) -> "Charset":
        if (not isinstance(state, Mapping) or type(state.get("blank")) is not int
                or state.get("blank") != 0):
            raise ValueError("Invalid checkpoint charset: CTC blank must be zero.")
        characters = state.get("characters")
        if not isinstance(characters, str):
            raise ValueError("Invalid checkpoint charset: characters must be a string.")
        result = cls(characters)
        if dict(state) != result.to_dict():
            raise ValueError("Invalid checkpoint charset: index mappings do not match characters.")
        return result
