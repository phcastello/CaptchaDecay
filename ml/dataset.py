"""Strict local loading of the synthetic generator's manifest and PNG files."""

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .charset import Charset


IMAGE_WIDTH = 215
IMAGE_HEIGHT = 80
SPLITS = ("train", "validation")
GENERATION_COMMAND = (
    "php generator/generate.php --count 3000 --seed 42 "
    "--output dataset/synthetic/v2_main --train-ratio 0.9 --validation-ratio 0.1"
)


def read_image_tensor(path):
    """Training-compatible pixels, shared with frozen real-domain evaluation."""
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(f"Expected original 215 x 80 PNG: {path}.")
            pixels = np.array(image.convert("L"), dtype=np.float32, copy=True)
            return torch.from_numpy(pixels).unsqueeze(0).div_(127.5).sub_(1.0)
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError(f"Cannot read PNG {path}: {error}") from error


@dataclass(frozen=True)
class SyntheticRecord:
    filename: str
    label: str
    split: str
    path: Path


def _is_real_path(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    return any(parts[index:index + 2] == ["dataset", "real"] for index in range(len(parts) - 1))


def _check_source(row: dict, context: str) -> None:
    for key in ("source", "source_type", "dataset_type"):
        value = row.get(key, "").casefold().replace("\\", "/")
        if value and ("dataset/real" in value or "correios" in value
                      or value in {"real", "real-world", "manual"} or "://" in value):
            raise ValueError(f"Synthetic-only pipeline refuses real/remote source metadata for {context}.")


class SyntheticManifest:
    """Validate every manifest row and image once, reusable across both splits.

    Identity hashes include CSV bytes and all image bytes. They do not depend on
    the absolute directory, so a verified dataset can move to another machine.
    """

    def __init__(self, root: str | Path, *, image_splits=None):
        if "://" in str(root):
            raise ValueError("Dataset must be an exclusively local synthetic directory.")
        self.root = Path(root).resolve()
        self.image_splits = tuple(SPLITS if image_splits is None else image_splits)
        if not self.image_splits or any(split not in SPLITS for split in self.image_splits):
            raise ValueError("Image validation scope must contain train and/or validation.")
        if _is_real_path(self.root):
            raise ValueError("Synthetic-only pipeline refuses dataset/real, including real/test.")
        manifest_path = self.root / "labels.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Synthetic manifest not found: {manifest_path}. "
                f"Generate the main dataset manually from the repository root:\n{GENERATION_COMMAND}"
            )
        manifest_bytes = manifest_path.read_bytes()
        try:
            stream = io.StringIO(manifest_bytes.decode("utf-8-sig"), newline="")
        except UnicodeDecodeError as error:
            raise ValueError(f"CSV must be UTF-8: {manifest_path}.") from error
        reader = csv.DictReader(stream, strict=True)
        required = {"filename", "label", "split", "width", "height"}
        try:
            columns = reader.fieldnames
        except csv.Error as error:
            raise ValueError(f"Invalid CSV header in {manifest_path}: {error}") from error
        if not columns or len(columns) != len(set(columns)) or not required.issubset(columns):
            missing = sorted(required.difference(columns or []))
            raise ValueError(f"Invalid CSV columns in {manifest_path}; missing={missing}, duplicate headers forbidden.")

        records = []
        seen_names: set[str] = set()
        image_hashes = []
        self.image_hashes = {}
        counts = dict.fromkeys(SPLITS, 0)
        try:
            for line_number, row in enumerate(reader, start=2):
                context = f"labels.csv line {line_number}"
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row at {context}: incorrect number of fields.")
                filename, label, split = row["filename"], row["label"], row["split"]
                context = f"{context}, image {filename!r}"
                if split not in SPLITS:
                    raise ValueError(f"Invalid split {split!r} at {context}; expected train or validation.")
                if (not filename or Path(filename).name != filename or "/" in filename
                        or "\\" in filename or ":" in filename or filename in {".", ".."}
                        or Path(filename).suffix.casefold() != ".png"):
                    raise ValueError(f"Invalid PNG filename at {context}; use a plain basename.")
                if filename.casefold() in seen_names:
                    raise ValueError(f"Duplicate filename {filename!r} across manifest entries/splits.")
                seen_names.add(filename.casefold())
                if not label or not label.strip():
                    raise ValueError(f"Empty label at {context}.")
                _check_source(row, context)
                for key, expected in (("width", IMAGE_WIDTH), ("height", IMAGE_HEIGHT)):
                    try:
                        actual = int(row[key])
                    except ValueError as error:
                        raise ValueError(f"Invalid {key} metadata at {context}.") from error
                    if actual != expected:
                        raise ValueError(f"Invalid dimensions at {context}: {key}={actual}, expected {expected}.")
                if "length" in row:
                    try:
                        label_length = int(row["length"])
                    except ValueError as error:
                        raise ValueError(f"Invalid label length metadata at {context}.") from error
                    if label_length != len(label):
                        raise ValueError(f"Label length metadata mismatch at {context}.")
                split_root = (self.root / split).resolve()
                path = (split_root / filename).resolve()
                if (_is_real_path(path) or not split_root.is_relative_to(self.root)
                        or not path.is_relative_to(split_root)):
                    raise ValueError(f"Image path escapes synthetic dataset at {context}.")
                records.append(SyntheticRecord(filename, label, split, path))
                counts[split] += 1
                if split not in self.image_splits:
                    continue
                if not path.is_file():
                    raise FileNotFoundError(f"Missing PNG for {context}: {path}.")
                png_bytes = path.read_bytes()
                try:
                    with Image.open(io.BytesIO(png_bytes)) as image:
                        if image.format != "PNG":
                            raise ValueError(f"Image is not a PNG at {context}.")
                        if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                            raise ValueError(f"Invalid PNG dimensions {image.size} at {context}; expected (215, 80).")
                        image.verify()
                    with Image.open(io.BytesIO(png_bytes)) as image:
                        image.load()
                except (UnidentifiedImageError, OSError, SyntaxError) as error:
                    raise ValueError(f"Invalid or corrupted PNG at {context}: {error}") from error
                digest = hashlib.sha256(png_bytes).hexdigest()
                self.image_hashes[(split, filename)] = digest
                image_hashes.append((split, filename, digest))
        except csv.Error as error:
            raise ValueError(f"Invalid CSV {manifest_path}: {error}") from error
        if not records:
            raise ValueError(f"Synthetic manifest contains no records: {manifest_path}.")
        self.records = sorted(records, key=lambda record: (record.split, record.filename))
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        images_hash = hashlib.sha256(json.dumps(sorted(image_hashes), ensure_ascii=False).encode("utf-8")).hexdigest()
        fingerprint = hashlib.sha256(f"{manifest_hash}:{images_hash}".encode("ascii")).hexdigest()
        self.identity = {
            "manifest_sha256": manifest_hash,
            "images_sha256": images_hash,
            "fingerprint": fingerprint,
            "counts": counts,
            "image_count": len(records),
        }
        if set(self.image_splits) != set(SPLITS):
            self.identity["validated_image_splits"] = sorted(set(self.image_splits))
            self.identity["validated_image_count"] = len(image_hashes)


class SyntheticDataset(Dataset):
    def __init__(self, root: str | Path, split: str, charset: Charset | None = None,
                 *, manifest: SyntheticManifest | None = None):
        if split not in SPLITS:
            raise ValueError(f"Invalid synthetic split {split!r}; expected train or validation.")
        self.manifest = manifest or SyntheticManifest(root)
        if self.manifest.root != Path(root).resolve():
            raise ValueError("Provided manifest belongs to a different dataset directory.")
        if split not in self.manifest.image_splits:
            raise ValueError(f"Split {split!r} is outside the manifest's validated image scope.")
        self.root = self.manifest.root
        self.split = split
        self.charset = charset or Charset.from_labels(
            record.label for record in self.manifest.records if record.split == "train"
        )
        for record in self.manifest.records:
            if record.split in self.manifest.image_splits:
                self.charset.encode(record.label, context=f"{record.split}/{record.filename}")
        self.records = [record for record in self.manifest.records if record.split == split]
        if not self.records:
            raise ValueError(f"Synthetic dataset has no images in split {split!r}.")
        self.identity = self.manifest.identity

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "image": read_image_tensor(record.path),
            "target": torch.tensor(self.charset.encode(record.label, record.filename), dtype=torch.long),
            "label": record.label,
            "filename": record.filename,
        }


def collate_ctc(samples: list[dict]) -> dict:
    """Stack equal-size images and concatenate variable-length CTC targets."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "images": torch.stack([sample["image"] for sample in samples]),
        "targets": torch.cat([sample["target"] for sample in samples]),
        "target_lengths": torch.tensor([sample["target"].numel() for sample in samples], dtype=torch.long),
        "labels": [sample["label"] for sample in samples],
        "filenames": [sample["filename"] for sample in samples],
    }
