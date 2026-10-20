"""Prepare independent synthetic datasets without changing the existing baseline.

The reference manifest is written before copying or generating any image. An
append-only source is accepted only when its original CSV prefix and PNG hashes
reproduce the dataset identity recorded by the baseline run. Partial preparations
are preserved and diagnosed; the script never deletes files or guesses repairs.
Only the existing PHP generator renders the additional training images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation")
NUMERIC_FIELDS = {
    "length", "width", "height", "perturbation", "num_lines", "noise_level",
    "font_ratio", "seed", "index", "render_seed",
}


class PreparationError(ValueError):
    """A preparation that requires inspection, with all existing files preserved."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _plain_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"Arquivo ausente ou link não permitido: {path}")
    if path.stat().st_nlink != 1:
        raise PreparationError(f"Hardlink não permitido; use cópia independente: {path}")


def _local_path(path: str | Path) -> Path:
    if "://" in str(path):
        raise PreparationError("A preparação aceita somente diretórios sintéticos locais.")
    absolute = Path(path).absolute()
    for parent in (absolute, *absolute.parents):
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise PreparationError(f"Diretório com link/junction não permitido: {parent}")
    absolute = absolute.resolve()
    parts = [part.casefold() for part in absolute.parts]
    if any(parts[index:index + 2] == ["dataset", "real"] for index in range(len(parts) - 1)):
        raise PreparationError("A preparação não acessa dataset/real.")
    return absolute


def _png_dimensions(data: bytes, path: Path) -> tuple[int, int]:
    """Validate PNG framing and CRC with the standard library, without rendering."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PreparationError(f"Assinatura PNG inválida: {path}")
    offset, dimensions, ended, idat = 8, None, False, False
    while offset < len(data):
        if offset + 12 > len(data):
            raise PreparationError(f"PNG truncado: {path}")
        length = struct.unpack("!I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise PreparationError(f"PNG truncado: {path}")
        payload = data[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack("!I", data[end - 4:end])[0]
        if zlib.crc32(kind + payload) != expected_crc:
            raise PreparationError(f"CRC PNG inválido: {path}")
        if dimensions is None:
            if kind != b"IHDR" or length != 13:
                raise PreparationError(f"IHDR PNG inválido: {path}")
            dimensions = struct.unpack("!II", payload[:8])
        elif kind == b"IHDR":
            raise PreparationError(f"IHDR PNG duplicado: {path}")
        idat = idat or kind == b"IDAT"
        offset = end
        if kind == b"IEND":
            ended = length == 0 and offset == len(data)
            break
    if not dimensions or not ended or not idat:
        raise PreparationError(f"PNG incompleto: {path}")
    return dimensions


@dataclass
class Inventory:
    root: Path
    csv_bytes: bytes
    columns: list[str]
    rows: list[dict[str, str]]
    images: list[dict]

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(row["split"] for row in self.rows)
        return {split: counts[split] for split in SPLITS}

    def identity(self, count: int | None = None) -> dict:
        count = len(self.rows) if count is None else count
        prefix = self.prefix_bytes(count)
        selected = self.images[:count]
        image_rows = sorted((item["split"], item["filename"], item["sha256"]) for item in selected)
        images_hash = sha256(json.dumps(image_rows, ensure_ascii=False).encode("utf-8"))
        manifest_hash = sha256(prefix)
        counts = Counter(row["split"] for row in self.rows[:count])
        return {
            "manifest_sha256": manifest_hash,
            "images_sha256": images_hash,
            "fingerprint": sha256(f"{manifest_hash}:{images_hash}".encode("ascii")),
            "counts": {split: counts[split] for split in SPLITS},
            "image_count": count,
        }

    def prefix_bytes(self, count: int) -> bytes:
        # Generator labels and metadata contain no embedded line breaks. Refuse
        # such a manifest rather than reconstructing or rewriting CSV records.
        lines = self.csv_bytes.splitlines(keepends=True)
        if len(lines) != len(self.rows) + 1:
            raise PreparationError("CSV com campos multilinha não é compatível com a cópia de prefixo.")
        return b"".join(lines[:count + 1])


def inspect_dataset(root: str | Path) -> Inventory:
    """Check every CSV reference, PNG, metadata and exact duplicate globally."""
    root = _local_path(root)
    csv_path = root / "labels.csv"
    _plain_file(csv_path)
    contents = csv_path.read_bytes()
    if not contents.endswith(b"\n"):
        raise PreparationError(f"CSV possivelmente interrompido, sem newline final: {csv_path}")
    try:
        reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig"), newline=""), strict=True)
        columns = reader.fieldnames
        required = {"filename", "label", "split", "width", "height", "index"}
        if not columns or len(set(columns)) != len(columns) or not required.issubset(columns):
            raise PreparationError(f"Cabeçalho CSV inválido: {csv_path}")
        rows = list(reader)
    except (csv.Error, UnicodeError) as error:
        raise PreparationError(f"CSV inválido: {csv_path}: {error}") from error
    if not rows:
        raise PreparationError(f"CSV sem amostras: {csv_path}")
    # Validate both split directories before opening any image. In particular,
    # reject a junction to dataset/real before following the first CSV reference.
    for split in SPLITS:
        split_root = _local_path(root / split)
        if not split_root.is_dir():
            raise PreparationError(f"Split ausente: {split_root}")
    images, names, hashes, references = [], set(), {}, set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise PreparationError(f"Linha incompleta no CSV: {csv_path}")
        filename, split, label = row["filename"], row["split"], row["label"]
        if (not re.fullmatch(r"[0-9]{8}\.png", filename) or split not in SPLITS
                or filename in names or not label.strip()):
            raise PreparationError(f"Nome/split/label inválido ou referência duplicada: {row}")
        if any(row.get(key, "").casefold() in {"real", "manual", "real-world"}
               or "://" in row.get(key, "") or "correios" in row.get(key, "").casefold()
               or "dataset/real" in row.get(key, "").casefold().replace("\\", "/")
               for key in ("source", "source_type", "dataset_type")):
            raise PreparationError("Metadata não sintética recusada.")
        path = root / split / filename
        _plain_file(path)
        try:
            dimensions = (int(row["width"]), int(row["height"]))
            if int(row["index"]) != int(filename[:-4]):
                raise ValueError("index e filename diferentes")
            if "length" in row and int(row["length"]) != len(label):
                raise ValueError("length e label diferentes")
        except ValueError as error:
            raise PreparationError(f"Metadata numérica inválida: {path}: {error}") from error
        data = path.read_bytes()
        if _png_dimensions(data, path) != dimensions:
            raise PreparationError(f"Dimensões do PNG não correspondem à metadata: {path}")
        digest = sha256(data)
        if digest in hashes:
            raise PreparationError(f"Duplicata exata SHA-256: {hashes[digest]} e {path}")
        hashes[digest] = str(path)
        names.add(filename)
        references.add(f"{split}/{filename}")
        images.append({"filename": filename, "split": split, "label": label, "sha256": digest})
    actual = set()
    for split in SPLITS:
        split_root = _local_path(root / split)
        if not split_root.is_dir():
            raise PreparationError(f"Split ausente: {split_root}")
        for path in split_root.rglob("*"):
            if path.is_symlink() or path.is_dir():
                raise PreparationError(f"Estrutura inesperada/link dentro do split: {path}")
            _plain_file(path)
            if path.suffix.casefold() == ".png":
                actual.add(f"{split}/{path.name}")
    if actual != references:
        raise PreparationError(f"PNGs sem referência no CSV: {sorted(actual - references)[:5]}")
    result = Inventory(root, contents, columns, rows, images)
    result.prefix_bytes(len(rows))
    return result


def _generator_info(repo: Path, php: str, source: Inventory, baseline_count: int,
                    target_count: int, seed: int, destination: Path) -> dict:
    entries = [{"index": int(row["index"]), "split": row["split"]}
               for row in source.rows[:baseline_count]]
    entries.extend({"index": index, "split": "train"}
                   for index in range(baseline_count + 1, target_count + 1))
    code = r'''
require $argv[1];
$spec = json_decode(stream_get_contents(STDIN), true, 512, JSON_THROW_ON_ERROR);
$config = parse_options(['--count', (string)$spec['count'], '--seed', (string)$spec['seed'],
                         '--output', $spec['output'], '--train-ratio', '1', '--validation-ratio', '0']);
$visual = array_diff_key($config, array_flip(['count','output','seed','train_ratio','validation_ratio']));
$recipes = [];
foreach ($spec['entries'] as $entry) {
    $recipes[] = sample_metadata($config, $entry['index'], $entry['split']);
}
echo json_encode(['config' => $config, 'visual' => $visual, 'columns' => CSV_COLUMNS,
                  'rng_scheme' => RNG_SCHEME, 'recipes' => $recipes,
                  'runtime' => ['php' => PHP_VERSION, 'gd' => extension_loaded('gd') ? gd_info() : null]],
                 JSON_THROW_ON_ERROR);
'''
    payload = {"count": target_count - baseline_count, "seed": seed,
               "output": str(destination), "entries": entries}
    try:
        result = subprocess.run([php, "-r", code, str(repo / "generator/generate.php")],
                                input=json.dumps(payload), capture_output=True, text=True,
                                cwd=repo, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PreparationError(f"Não foi possível inspecionar o gerador PHP: {error}") from error
    if result.returncode:
        raise PreparationError(f"Gerador PHP recusou os argumentos: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PreparationError("O gerador não retornou metadata JSON válida.") from error


def _check_recipes(inventory: Inventory, recipes: list[dict], columns: list[str]) -> None:
    if inventory.columns != columns or len(inventory.rows) > len(recipes):
        raise PreparationError("Cabeçalho/quantidade incompatível com o gerador atual; dados preservados.")
    for row, expected in zip(inventory.rows, recipes):
        for key, value in expected.items():
            try:
                matches = (Decimal(row[key]) == Decimal(str(value)) if key in NUMERIC_FIELDS
                           else row[key] == str(value))
            except (KeyError, ArithmeticError):
                matches = False
            if not matches:
                raise PreparationError(f"Receita RNG/configuração visual divergente: {row['filename']}, {key}.")


def _asset_hashes(repo: Path) -> dict[str, str]:
    assets = [repo / path for path in ("generator/generate.php", "generator/config.php", "composer.lock",
              "vendor/dapphp/securimage/securimage.php", "vendor/dapphp/securimage/AHGBold.ttf")]
    result = {}
    for path in assets:
        _plain_file(path)
        result[path.relative_to(repo).as_posix()] = file_hash(path)
    return result


def _run_hashes(run: Path) -> dict[str, str]:
    result = {}
    for name in ("best.pt", "last.pt", "history.csv", "run_config.json"):
        path = run / name
        _plain_file(path)
        result[name] = file_hash(path)
    return result


def _immutable_json(path: Path, value: dict) -> None:
    if path.exists():
        _plain_file(path)
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise PreparationError(f"Manifesto existente inválido, preservado: {path}") from error
        if previous != value:
            raise PreparationError(f"Manifesto existente diverge do estado atual; não foi sobrescrito: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _copy_baseline(source: Inventory, destination: Path, count: int) -> None:
    destination.mkdir(parents=True)
    for split in SPLITS:
        (destination / split).mkdir()
    for row in source.rows[:count]:
        original = source.root / row["split"] / row["filename"]
        target = destination / row["split"] / row["filename"]
        with original.open("rb") as input_stream, target.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
    with (destination / "labels.csv").open("xb") as stream:
        stream.write(source.prefix_bytes(count))


def _check_originals(candidate: Inventory, source: Inventory, count: int) -> None:
    if (len(candidate.rows) < count or candidate.prefix_bytes(count) != source.prefix_bytes(count)
            or candidate.rows[:count] != source.rows[:count]
            or candidate.images[:count] != source.images[:count]):
        raise PreparationError(f"Os PNGs/labels/metadata originais não são byte a byte idênticos: {candidate.root}")


def _run_generation(repo: Path, php: str, destination: Path, count: int, seed: int) -> None:
    command = [php, str(repo / "generator/generate.php"), "--count", str(count), "--seed", str(seed),
               "--train-ratio", "1", "--validation-ratio", "0", "--output", str(destination)]
    try:
        result = subprocess.run(command, cwd=repo, text=True, capture_output=True)
    except OSError as error:
        raise PreparationError(f"Falha ao iniciar PHP; cópia preservada: {error}") from error
    if result.returncode:
        raise PreparationError(f"Expansão interrompida; arquivos preservados. Diagnóstico PHP: {result.stderr.strip()}")
    print(result.stdout.strip(), flush=True)


def prepare_dataset(source: str | Path, destination: str | Path, experiment_dir: str | Path,
                    baseline_run: str | Path, *, php: str = "php", repo: Path = ROOT,
                    baseline_train: int = 2700, baseline_validation: int = 300,
                    target_train: int = 9000, seed: int = 42, verify_only: bool = False) -> dict:
    """Prepare once, verify on every repeat, and leave interrupted files untouched."""
    source_path, destination, experiment_dir, baseline_run = map(
        _local_path, (source, destination, experiment_dir, baseline_run))
    snapshot = experiment_dir / "baseline-dataset"
    for output_path in (destination, experiment_dir):
        for protected in (source_path, baseline_run):
            if output_path.is_relative_to(protected) or protected.is_relative_to(output_path):
                raise PreparationError("Destinos de escrita não podem sobrepor a fonte ou o diretório do baseline.")
    data_paths = (source_path, destination, snapshot)
    for index, left in enumerate(data_paths):
        for right in data_paths[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise PreparationError("Fonte, snapshot e destino precisam ser diretórios independentes.")
    if baseline_train < 1 or baseline_validation < 1 or target_train <= baseline_train:
        raise PreparationError("Quantidades de preparação inválidas.")
    baseline_count = baseline_train + baseline_validation
    total = target_train + baseline_validation
    source_inventory = inspect_dataset(source_path)
    if len(source_inventory.rows) < baseline_count:
        raise PreparationError("Fonte não contém todos os exemplos do baseline.")
    if [int(row["index"]) for row in source_inventory.rows] != list(range(1, len(source_inventory.rows) + 1)):
        raise PreparationError("Fonte não é uma sequência append-only de índices iniciada em 1.")
    if any(row["split"] != "train" for row in source_inventory.rows[baseline_count:]):
        raise PreparationError("Fonte contém validação adicional após o baseline; revise o histórico.")
    baseline_identity = source_inventory.identity(baseline_count)
    if baseline_identity["counts"] != {"train": baseline_train, "validation": baseline_validation}:
        raise PreparationError("Prefixo da fonte não corresponde aos splits originais do baseline.")
    run_hashes = _run_hashes(baseline_run)
    run_config = json.loads((baseline_run / "run_config.json").read_text(encoding="utf-8"))
    if run_config.get("dataset") != baseline_identity:
        raise PreparationError("CSV original/prefixo e PNGs não reproduzem a identidade registrada pelo baseline.")
    print(f"Fonte observada: {source_inventory.counts}; baseline comprovado pelo prefixo CSV e hashes PNG: "
          f"{baseline_identity['counts']}. A fonte será preservada integralmente.", flush=True)
    info = _generator_info(repo, php, source_inventory, baseline_count, total, seed, destination)
    baseline_view = Inventory(source_path, source_inventory.prefix_bytes(baseline_count), source_inventory.columns,
                              source_inventory.rows[:baseline_count], source_inventory.images[:baseline_count])
    _check_recipes(baseline_view, info["recipes"], info["columns"])
    if (info["config"]["count"] != target_train - baseline_train
            or info["config"]["train_ratio"] != 1 or info["config"]["validation_ratio"] != 0
            or info["config"]["seed"] != seed):
        raise PreparationError("Argumentos de expansão não foram aceitos conforme o protocolo.")
    assets = _asset_hashes(repo)
    manifest = {
        "schema_version": 1,
        "source": str(source_path), "destination": str(destination), "baseline_snapshot": str(snapshot),
        "source_observed": {"identity": source_inventory.identity(), "files_per_split": source_inventory.counts,
                            "images": source_inventory.images},
        "baseline": {"identity": baseline_identity, "columns": source_inventory.columns,
                     "images": [dict(image, metadata=row) for image, row in
                                zip(baseline_view.images, baseline_view.rows)],
                     "recovery": "unchanged CSV byte prefix and PNG bytes match baseline run_config dataset identity"},
        "baseline_run": str(baseline_run), "baseline_artifacts_sha256": run_hashes,
        "generation": {key: value for key, value in info.items() if key != "recipes"},
        "generation_assets_sha256": assets,
        "protocol": {"baseline_train": baseline_train, "baseline_validation": baseline_validation,
                     "target_train": target_train, "additional_train": target_train - baseline_train,
                     "seed": seed, "new_indices": [baseline_count + 1, total]},
    }
    if verify_only and not (experiment_dir / "baseline-manifest.json").is_file():
        raise PreparationError("Verificação requer baseline-manifest.json existente; nada foi criado.")
    _immutable_json(experiment_dir / "baseline-manifest.json", manifest)
    print(f"Referência preservada: {experiment_dir / 'baseline-manifest.json'}", flush=True)

    def ensure_copy(path: Path) -> Inventory:
        if not path.exists():
            if verify_only:
                raise PreparationError(f"Verificação requer cópia existente: {path}; nada foi criado.")
            _copy_baseline(source_inventory, path, baseline_count)
        try:
            copied = inspect_dataset(path)
        except (PreparationError, OSError) as error:
            raise PreparationError(f"Preparação incompleta/inconsistente em {path}; preserve e revise os dados. {error}") from error
        _check_originals(copied, source_inventory, baseline_count)
        return copied

    snapshot_inventory = ensure_copy(snapshot)
    if snapshot_inventory.identity() != baseline_identity:
        raise PreparationError(f"Snapshot baseline possui arquivos extras ou divergentes, preservados: {snapshot}")
    expanded = ensure_copy(destination)
    if len(expanded.rows) == baseline_count:
        if verify_only:
            raise PreparationError("Expansão ainda não executada; --verify-only não gera imagens.")
        print(f"Cópia independente verificada; gerando somente {target_train - baseline_train} novos PNGs de treino.", flush=True)
        _run_generation(repo, php, destination, target_train - baseline_train, seed)
        expanded = inspect_dataset(destination)
    elif len(expanded.rows) != total:
        raise PreparationError(
            f"Expansão interrompida: {len(expanded.rows)}/{total} linhas, splits={expanded.counts}. "
            "Dados preservados; não há retomada automática. Revise a causa e os arquivos antes de continuar.")
    if expanded.counts != {"train": target_train, "validation": baseline_validation}:
        raise PreparationError(f"Quantidades finais incorretas: {expanded.counts}")
    if [int(row["index"]) for row in expanded.rows] != list(range(1, total + 1)):
        raise PreparationError("Índices finais incorretos; expansão preservada para inspeção.")
    _check_originals(expanded, source_inventory, baseline_count)
    _check_recipes(expanded, info["recipes"], info["columns"])
    after = inspect_dataset(source_path)
    if (after.identity() != source_inventory.identity() or after.rows != source_inventory.rows
            or after.images != source_inventory.images):
        raise PreparationError("A fonte foi alterada durante a preparação; investigue antes de usar o resultado.")
    if _run_hashes(baseline_run) != run_hashes or _asset_hashes(repo) != assets:
        raise PreparationError("Checkpoint/configuração/gerador mudou durante a preparação.")
    report = {
        "dataset": str(destination), "identity": expanded.identity(),
        "baseline_snapshot": str(snapshot), "baseline_identity": snapshot_inventory.identity(),
        "original_pngs_verified": baseline_count, "validation_pngs_byte_identical": baseline_validation,
        "validation_labels_identical": baseline_validation, "exact_duplicate_count": 0,
        "source_unchanged": True, "baseline_artifacts_unchanged": True, "generation_assets_unchanged": True,
    }
    if verify_only and not (experiment_dir / "preparation-report.json").is_file():
        raise PreparationError("Verificação requer preparation-report.json existente; nada foi criado.")
    _immutable_json(experiment_dir / "preparation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/synthetic/v2_main")
    parser.add_argument("--destination", type=Path, default=ROOT / "dataset/synthetic/v2_9k_train")
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "outputs/experiments/dataset-expansion")
    parser.add_argument("--baseline-run", type=Path, default=ROOT / "outputs/runs/crnn-v2-baseline")
    parser.add_argument("--php", default=os.environ.get("PHP_BINARY") or shutil.which("php") or "php")
    parser.add_argument("--verify-only", action="store_true", help="Verificar preparação completa sem criar ou gerar arquivos.")
    args = parser.parse_args(argv)
    try:
        prepare_dataset(args.source, args.destination, args.experiment_dir, args.baseline_run,
                        php=args.php, verify_only=args.verify_only)
    except (PreparationError, OSError, ValueError) as error:
        print(f"Erro: {error}\nNenhum arquivo foi apagado ou reparado automaticamente.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
