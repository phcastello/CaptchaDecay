"""Prepare experiment D from synthetic data only; never append to an existing run.

The 20,000 training images are a new deterministic sample, rather than an
extension of experiment B. Comparisons with B therefore also change sampling.
"""

import argparse
import csv
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "dataset/synthetic/validation-300"
DESTINATION = ROOT / "dataset/synthetic/v2_20k_train"
HISTORICAL_MANIFEST_SHA256 = "0bbdc3f7a7b74206466444f98855bab13d28d8016e1615bef3382fc65abad597"
HISTORICAL_LABELS_SHA256 = "2158f8746674f893770245220177634d69cd77a6298373e8b74d0f9a9b1d7b95"
VISUAL_CONFIG_SHA256 = "f3dd425af956b5ac232759861220258f3d339ccccba60c8604d1a0387245951b"
CHARSET = "23456789abcdefghiklmnprstuvwyz"
RNG_SCHEME = "xoshiro256starstar-sha256-v2"
COLUMNS = ("filename", "label", "split", "length", "width", "height",
           "perturbation", "num_lines", "noise_level", "font_ratio",
           "image_bg_color", "text_color", "line_color", "seed", "index",
           "render_seed", "config_sha256", "securimage_version", "rng_scheme")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    data = path.read_bytes()
    require(data.endswith(b"\n"), f"CSV incompleto: {path}")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    require(tuple(reader.fieldnames or ()) == COLUMNS, f"Cabeçalho CSV inesperado: {path}")
    rows = list(reader)
    require(all(set(row) == set(COLUMNS) and all(value is not None for value in row.values())
                for row in rows), f"CSV contém linhas incompletas: {path}")
    return data, rows


def validate_row(row):
    require(re.fullmatch(r"[0-9]{8}\.png", row["filename"]), "Nome de PNG inválido")
    require(row["split"] in ("train", "validation"), "Split inválido")
    require(4 <= len(row["label"]) <= 6 and set(row["label"]) <= set(CHARSET),
            "Label fora do charset/comprimentos v2")
    require(int(row["length"]) == len(row["label"]), "Comprimento de label inconsistente")
    require((int(row["width"]), int(row["height"])) == (215, 80), "Dimensões CSV inválidas")
    require(int(row["index"]) == int(row["filename"][:-4]), "Índice inconsistente")
    require(0 <= int(row["seed"]) <= 2147483647 and
            0 <= int(row["render_seed"]) <= 2147483647, "Seed inválida")
    for name, low, high in (("perturbation", .90, 1.0), ("num_lines", 5, 8),
                            ("noise_level", 2, 4), ("font_ratio", .42, .52)):
        value = int(row[name]) if name in ("num_lines", "noise_level") else float(row[name])
        require(low <= value <= high, f"Parâmetro visual v2 inválido: {name}")
    for name, allowed in (("image_bg_color", ("#ffffff", "#f6f6f6")),
                          ("text_color", ("#606060", "#707070", "#808080")),
                          ("line_color", ("#606060", "#808080"))):
        require(row[name] in allowed, f"Cor v2 inválida: {name}")
    require(row["config_sha256"] == VISUAL_CONFIG_SHA256, "Configuração visual diverge de v2")
    require(row["rng_scheme"] == RNG_SCHEME and row["securimage_version"] == "4.0.2",
            "RNG ou versão Securimage divergente")


def image_record(root, row):
    path = root / row["split"] / row["filename"]
    require(not path.is_symlink() and path.is_file(), f"PNG ausente ou link simbólico: {path}")
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == "PNG" and image.size == (215, 80), f"PNG inválido: {path}")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()  # Decode the complete image too, including compressed pixel data.
    return {"filename": row["filename"], "label": row["label"],
            "sha256": digest(data), "bytes": len(data)}


def verify_snapshot(source, *, expected_manifest_sha256=HISTORICAL_MANIFEST_SHA256,
                    expected_labels_sha256=HISTORICAL_LABELS_SHA256, expected_count=300):
    """Verify the immutable historical source before creating any directories."""
    source = Path(source)
    manifest_data = (source / "manifest.json").read_bytes()
    require(digest(manifest_data) == expected_manifest_sha256, "Manifesto histórico alterado")
    manifest = json.loads(manifest_data)
    labels_data, rows = read_rows(source / "labels.csv")
    require(digest(labels_data) == expected_labels_sha256 == manifest["labels_sha256"],
            "CSV histórico alterado (labels ou metadata)")
    require(manifest["schema_version"] == 1 and manifest["split"] == "validation" and
            manifest["labels_file"] == "labels.csv", "Manifesto histórico incompatível")
    require(manifest["num_images"] == len(rows) == expected_count,
            "Quantidade histórica de validação incorreta")
    require(manifest["image_dimensions"] == {"width": 215, "height": 80} and
            manifest["charset"] == CHARSET, "Dimensões ou charset histórico divergente")
    require(all(row["split"] == "validation" for row in rows), "Histórico contém treino")
    require(len({row["filename"] for row in rows}) == len(rows), "Filename histórico duplicado")
    for row in rows:
        validate_row(row)
    records = [image_record(source, row) for row in rows]
    require(records == manifest["images"] and
            canonical_hash(records) == manifest["image_records_sha256"],
            "Hash/label/tamanho de PNG histórico alterado")
    require(len({record["sha256"] for record in records}) == len(rows),
            "Duplicata exata na validação histórica")
    require(set(path.name for path in (source / "validation").iterdir()) ==
            {row["filename"] for row in rows}, "Arquivos históricos ausentes ou inesperados")
    require(sum(len(row["label"]) for row in rows) == manifest["total_reference_characters"] and
            dict(Counter(str(len(row["label"])) for row in rows)) == manifest["counts_by_length"],
            "Distribuição/comprimentos históricos divergentes")
    return {"manifest": manifest, "manifest_data": manifest_data, "labels_data": labels_data,
            "rows": rows, "records": records}


def inspect_dataset(destination, snapshot, train_count, seed):
    """Read all PNGs and metadata; reject missing data, duplicates, and leakage."""
    destination = Path(destination)
    labels_data, rows = read_rows(destination / "labels.csv")
    require(len(rows) == train_count + len(snapshot["rows"]), "Quantidade total incorreta")
    require(labels_data.startswith(snapshot["labels_data"]), "CSV histórico foi reescrito")
    require([row for row in rows if row["split"] == "validation"] == snapshot["rows"],
            "Metadata/labels históricos divergentes no dataset D")
    train_rows = [row for row in rows if row["split"] == "train"]
    require(len(train_rows) == train_count, "Quantidade de treino incorreta")
    require(len({row["filename"] for row in rows}) == len(rows), "Filename duplicado entre splits")
    historical_indices = {int(row["index"]) for row in snapshot["rows"]}
    expected_indices = set(range(max(historical_indices) + 1, max(historical_indices) + 1 + train_count))
    require({int(row["index"]) for row in train_rows} == expected_indices,
            "Índices de treino fora da sequência nova")
    require(all(int(row["seed"]) == seed for row in train_rows), "Seed de treino inconsistente")
    require(seed not in {int(row["seed"]) for row in snapshot["rows"]},
            "Seed de treino reutiliza a seed histórica")
    records = []
    hashes = set()
    for row in rows:
        validate_row(row)
        record = image_record(destination, row)
        require(record["sha256"] not in hashes, "Duplicata exata/leakage entre imagens")
        hashes.add(record["sha256"])
        records.append({"split": row["split"], **record})
    for split in ("train", "validation"):
        require(set(path.name for path in (destination / split).iterdir()) ==
                {row["filename"] for row in rows if row["split"] == split},
                f"Arquivos ausentes ou inesperados em {split}")
    require([{key: record[key] for key in ("filename", "label", "sha256", "bytes")}
             for record in records if record["split"] == "validation"] == snapshot["records"],
            "Validação D difere dos PNGs históricos")
    return {"schema_version": 1, "counts": {"train": train_count, "validation": len(snapshot["rows"])},
            "image_dimensions": {"width": 215, "height": 80}, "charset": CHARSET,
            "labels_sha256": digest(labels_data), "rows_sha256": canonical_hash(rows),
            "historical_manifest_sha256": digest(snapshot["manifest_data"]),
            "historical_labels_sha256": digest(snapshot["labels_data"]),
            "historical_image_records_sha256": snapshot["manifest"]["image_records_sha256"],
            "image_records_sha256": canonical_hash(records), "images": records}


def verify_dataset(destination, snapshot, train_count, seed):
    destination = Path(destination)
    manifest = json.loads((destination / "integrity-manifest.json").read_bytes())
    generation_data = (destination / "generation.json").read_bytes()
    require(digest(generation_data) == manifest.pop("generation_sha256"),
            "Registro de geração foi alterado")
    generation = json.loads(generation_data)
    require(generation["config"]["count"] == train_count and generation["config"]["seed"] == seed and
            generation["visual_config_sha256"] == VISUAL_CONFIG_SHA256,
            "Configuração existente difere da geração solicitada")
    for name, expected in (("manifest.json", snapshot["manifest_data"]),
                           ("labels.csv", snapshot["labels_data"])):
        require((destination / "historical-validation" / name).read_bytes() == expected,
                f"Cópia histórica de {name} alterada")
    actual = inspect_dataset(destination, snapshot, train_count, seed)
    require(actual == manifest, "Manifesto de integridade diverge dos arquivos atuais")
    return actual


def _run_php(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    require(not result.stderr.strip(), f"PHP emitiu mensagens: {result.stderr}")
    require(not re.search(r"\b(?:Warning|Notice|Deprecated|Fatal error):", result.stdout),
            f"PHP emitiu diagnóstico: {result.stdout}")
    return result.stdout


def _generator_arguments(output, count, seed):
    return ["--count", str(count), "--seed", str(seed), "--train-ratio", "1",
            "--validation-ratio", "0", "--output", str(output)]


def _resolve_php(php):
    if php:
        return str(php)
    system_php = shutil.which("php")
    if system_php:
        return system_php
    local_php = ROOT / "scripts/php-local"
    require(local_php.is_file() and os.access(local_php, os.X_OK),
            "PHP CLI ausente; instale PHP/GD ou prepare scripts/php-local")
    return str(local_php)


def _dependency_tree_hash(directory):
    return canonical_hash([{ "path": str(path.relative_to(directory)),
                             "sha256": digest(path.read_bytes())}
                           for path in sorted(directory.rglob("*")) if path.is_file()])


def _recipe(php, output, count, seed):
    code = ("require $argv[1]; require dirname($argv[1]).'/../vendor/autoload.php'; "
            "$c=parse_options(array_slice($argv,2)); "
            "$v=array_diff_key($c,array_flip(['count','output','seed','train_ratio','validation_ratio'])); "
            "ob_start(); phpinfo(INFO_MODULES); $info=ob_get_clean(); "
            "preg_match('/^FreeType Version => (.+)$/m',$info,$ft); "
            "$font=Securimage::getPath().'/AHGBold.ttf'; "
            "echo json_encode(['config'=>$c,'visual_config_sha256'=>hash('sha256',json_encode($v)),"
            "'runtime'=>['php_version'=>PHP_VERSION,'php_binary'=>PHP_BINARY,'os_family'=>PHP_OS_FAMILY,"
            "'gd'=>gd_info(),'freetype_version'=>$ft[1]??null],"
            "'securimage'=>['version'=>Composer\\InstalledVersions::getPrettyVersion('dapphp/securimage'),"
            "'reference'=>Composer\\InstalledVersions::getReference('dapphp/securimage'),"
            "'source_path'=>Securimage::getPath(),'font_path'=>$font,'font_sha256'=>hash_file('sha256',$font)]]);")
    recipe = json.loads(_run_php([php, "-r", code, str(ROOT / "generator/generate.php"),
                                  *_generator_arguments(output, count, seed)]))
    require(recipe["visual_config_sha256"] == VISUAL_CONFIG_SHA256,
            "Configuração atual do gerador diverge da configuração visual v2 histórica")
    require(recipe["runtime"]["gd"]["FreeType Support"] and recipe["runtime"]["gd"]["PNG Support"],
            "GD precisa de suporte PNG/FreeType")
    locked = next(package for package in json.loads((ROOT / "composer.lock").read_bytes())["packages"]
                  if package["name"] == "dapphp/securimage")
    require(recipe["securimage"]["version"] == locked["version"] == "4.0.2" and
            recipe["securimage"]["reference"] == locked["source"]["reference"],
            "Securimage instalado diverge do lockfile")
    dependency = Path(recipe["securimage"]["source_path"])
    recipe["securimage"]["source_tree_sha256"] = _dependency_tree_hash(dependency)
    recipe["files_sha256"] = {name: digest((ROOT / name).read_bytes()) for name in
                              ("generator/generate.php", "generator/config.php", "composer.lock")}
    recipe["methodology"] = ("All training images are newly sampled locally with the official v2 "
                              "Securimage generator. Comparison with experiment B changes both "
                              "training quantity and generator sampling; no prior training PNGs reused. "
                              "PNG rendering may vary across PHP/GD/FreeType/platform versions; the "
                              "exact local runtime, dependency tree and font hashes are recorded here.")
    return recipe


def _publish_without_replacement(staging, destination):
    # Linux renameat2 publishes the complete directory atomically and refuses even
    # an empty existing destination. os.rename alone could overwrite that directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    require(rename is not None, "Publicação segura requer Linux com renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def prepare_dataset(destination=DESTINATION, *, source=HISTORICAL, train_count=20000,
                    seed=20261020, php=None, verify_only=False):
    require(1 <= train_count <= 1000000 and 0 <= seed <= 2147483647, "Contagem ou seed inválida")
    source = Path(source).resolve()
    destination = Path(destination).absolute()
    require(source != destination.resolve() and source not in destination.resolve().parents and
            destination.resolve() not in source.parents, "Destino sobrepõe o histórico")
    snapshot = verify_snapshot(source)
    require(seed not in {int(row["seed"]) for row in snapshot["rows"]},
            "Utilize uma seed distinta da validação histórica")
    require(not destination.is_symlink(), "Destino não pode ser um link simbólico")
    if destination.exists():
        result = verify_dataset(destination, snapshot, train_count, seed)
        return {"status": "verified_existing", "destination": str(destination), "counts": result["counts"]}
    require(not verify_only, "Dataset ainda não existe; --verify-only não gera arquivos")
    php = _resolve_php(php)
    recipe = _recipe(php, destination, train_count, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        (staging / "validation").mkdir()
        (staging / "historical-validation").mkdir()
        for row in snapshot["rows"]:
            shutil.copy2(source / "validation" / row["filename"], staging / "validation" / row["filename"])
        shutil.copy2(source / "labels.csv", staging / "labels.csv")
        for name in ("labels.csv", "manifest.json"):
            shutil.copy2(source / name, staging / "historical-validation" / name)
        command = [php, str(ROOT / "generator/generate.php"),
                   *_generator_arguments(staging, train_count, seed)]
        print(f"Gerando {train_count} imagens novas em {staging}", flush=True)
        recipe["generator_stdout"] = _run_php(command).strip()
        recipe["command"] = [php, "generator/generate.php", *_generator_arguments(destination, train_count, seed)]
        require(recipe["files_sha256"] == {name: digest((ROOT / name).read_bytes())
                                           for name in recipe["files_sha256"]},
                "Gerador/configuração/lockfile mudou durante a geração")
        require(recipe["securimage"]["source_tree_sha256"] ==
                _dependency_tree_hash(Path(recipe["securimage"]["source_path"])),
                "Dependência Securimage mudou durante a geração")
        manifest = inspect_dataset(staging, snapshot, train_count, seed)
        # Re-read the original source after generation: it must remain byte-identical.
        verify_snapshot(source)
        generation_data = (json.dumps(recipe, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
        (staging / "generation.json").write_bytes(generation_data)
        manifest["generation_sha256"] = digest(generation_data)
        (staging / "integrity-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_dataset(staging, snapshot, train_count, seed)
        _publish_without_replacement(staging, destination)
    except Exception as error:
        raise RuntimeError(f"Preparação interrompida; arquivos parciais preservados em {staging}: {error}") from error
    return {"status": "created", "destination": str(destination), "counts": manifest["counts"]}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--source", type=Path, default=HISTORICAL)
    parser.add_argument("--train-count", type=int, default=20000,
                        help="Use quantidades menores somente em testes isolados; D exige 20000.")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--php", help="Executável PHP; detecta PHP do sistema ou scripts/php-local.")
    parser.add_argument("--verify-only", action="store_true")
    options = parser.parse_args(arguments)
    try:
        result = prepare_dataset(options.output, source=options.source, train_count=options.train_count,
                                 seed=options.seed, php=options.php, verify_only=options.verify_only)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Erro: {error}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
