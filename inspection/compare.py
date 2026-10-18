"""Comparação local de imagens reais de treino e sintéticas, sem OCR."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from labeling.labeler import LabelStore, parse_state, read_optional


def allowed_directories(real_dir, synthetic_dir, real_root, synthetic_root):
    """Validate both origins before enumerating or opening any input files."""
    reject_evaluation_path(real_dir)
    reject_evaluation_path(synthetic_dir)
    real_root, synthetic_root = Path(real_root).resolve(), Path(synthetic_root).resolve()
    real_dir, synthetic_dir = Path(real_dir).resolve(), Path(synthetic_dir).resolve()
    if real_dir != real_root / "train":
        raise ValueError("A entrada real deve ser o train da raiz real autorizada.")
    forbidden = [real_root, ROOT / "dataset/real"]
    for dataset_root in (real_root.parent, ROOT / "dataset"):
        forbidden.extend(dataset_root / split for split in ("train", "validation", "test"))
    if (not synthetic_dir.is_relative_to(synthetic_root)
            or any(synthetic_dir.is_relative_to(path.resolve()) for path in forbidden)):
        raise ValueError("A entrada sintética deve pertencer à raiz sintética autorizada; reais e backups são proibidos.")
    return real_dir, synthetic_dir


def reject_evaluation_path(path):
    for candidate in (Path(path).absolute(), Path(path).resolve()):
        parts = [p.casefold() for p in candidate.parts]
        if any(a == "real" and b in {"test", "validation"} for a, b in zip(parts, parts[1:])):
            raise ValueError("Comparação usa somente real/train; real/test e real/validation são isolados.")
        for split in ("test", "validation"):
            if candidate.is_relative_to((ROOT / "dataset/real" / split).resolve()):
                raise ValueError("Conjuntos de avaliação não podem entrar na comparação.")


def image_paths(directory, *, real=False):
    directory = Path(directory).resolve()
    reject_evaluation_path(directory)
    if real and directory.name.casefold() != "train":
        raise ValueError("A entrada real deve ser o diretório train. Não use validation/test para calibração.")
    if not directory.is_dir():
        raise ValueError(f"Diretório de imagens ausente: {directory}")
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png"),
                   key=lambda p: p.name)
    if not paths:
        raise ValueError(f"Dataset vazio (nenhum PNG): {directory}")
    for path in paths:
        reject_evaluation_path(path)
        if path.resolve().parent != directory:
            raise ValueError(f"Imagem aponta para fora do diretório selecionado: {path}")
    return paths


def select_samples(paths, count, seed, domain):
    return random.Random(f"captchadecay:comparison:{seed}:{domain}").sample(sorted(paths), min(count, len(paths)))


def labels_for(directory, paths, *, allowed_root=None):
    csv_path = directory.parent / "labels.csv"
    reject_evaluation_path(csv_path)
    if not csv_path.exists():
        return {}, None
    if allowed_root is not None and not csv_path.resolve().is_relative_to(allowed_root):
        raise ValueError(f"CSV fora da raiz autorizada: {csv_path}")
    labels = {}
    names = {p.name for p in paths}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not {"filename", "label", "split"} <= set(reader.fieldnames or []):
            raise ValueError(f"CSV sem filename,label,split: {csv_path}")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Linha incompleta: {csv_path}")
            if row["split"] != directory.name or row["filename"] not in names:
                continue
            if row["filename"] in labels:
                raise ValueError(f"Registro duplicado: {csv_path}: {row['filename']}")
            labels[row["filename"]] = row["label"]
    return labels, str(csv_path)


def dataset_stats(directory, paths, sampled, *, real=False, allowed_root=None):
    labels, csv_path = labels_for(directory, paths, allowed_root=allowed_root)
    reviewing = set()
    if real:
        state_path = directory.parent / "labeling_state.json"
        if allowed_root is not None and not state_path.resolve().is_relative_to(allowed_root):
            raise ValueError(f"Estado fora da raiz autorizada: {state_path}")
        try:
            review = parse_state(read_optional(state_path))["review"]
        except ValueError as error:
            raise ValueError(f"labeling_state.json inválido: {error}") from error
        if review:
            for path in paths:
                identity = LabelStore.identity({"split": directory.name, "filename": path.name,
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if identity in review:
                    reviewing.add(path.name)
    filled = {name: value for name, value in labels.items() if value != ""}
    known = [value for name, value in filled.items() if name not in reviewing]
    resolutions = Counter()
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Arquivo não é PNG: {path}")
            image.verify()
            resolutions[f"{image.width}x{image.height}"] += 1
    stats = {
        "directory": str(directory), "available": len(paths), "sampled": len(sampled),
        "selected_files": [p.name for p in sampled], "resolutions": dict(sorted(resolutions.items())),
        "labels_csv": csv_path, "labeled": len(known), "unlabeled_or_unknown": len(paths) - len(filled),
        "label_lengths": dict(sorted(Counter(map(len, known)).items())),
        "characters": "".join(sorted(set("".join(known)))),
    }
    if real:
        stats["labeled_in_review"] = len(reviewing & filled.keys())
        # A subset of unlabeled_or_unknown, not another disjoint category.
        stats["unlabeled_in_review"] = len(reviewing - filled.keys())
    return stats


def fit_size(size, scale):
    return tuple(max(1, round(length * scale)) for length in size)


def remove_owned(files):
    for path, identity in reversed(files):
        try:
            if os.path.samestat(path.lstat(), identity):
                path.unlink()
        except FileNotFoundError:
            pass


def save_comparison(canvas, report, output, report_path):
    """Stage and validate the pair; roll back only files created by this call."""
    temporary, created = [], []
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".comparison-", suffix=".tmp",
                                         delete=False) as stream:
            png_temp = Path(stream.name)
            temporary.append((png_temp, os.fstat(stream.fileno())))
            canvas.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        with Image.open(png_temp) as image:
            image.verify()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".comparison-", suffix=".tmp", delete=False) as stream:
            json_temp = Path(stream.name)
            temporary.append((json_temp, os.fstat(stream.fileno())))
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(json_temp.read_text(encoding="utf-8"))
        # Exclusive creation preserves outputs from other runs, even after preflight.
        # Publishing two files is not an indivisible transaction.
        for source, target in ((png_temp, output), (json_temp, report_path)):
            with target.open("xb") as destination:
                created.append((target, os.fstat(destination.fileno())))
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, destination)
                destination.flush()
                os.fsync(destination.fileno())
    except BaseException:
        remove_owned(created)
        raise
    finally:
        remove_owned(temporary)


def compare(real_dir=ROOT / "dataset/real/train", synthetic_dir=ROOT / "dataset/synthetic/train",
            *, count=20, seed=42, output=ROOT / "inspection/output/comparison.png",
            real_root=ROOT / "dataset/real", synthetic_root=ROOT / "dataset/synthetic"):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count deve ser inteiro positivo.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed deve ser inteiro.")
    real_dir, synthetic_dir = allowed_directories(real_dir, synthetic_dir, real_root, synthetic_root)
    real_paths = image_paths(real_dir, real=True)
    synthetic_paths = image_paths(synthetic_dir)
    paired_count = min(count, len(real_paths), len(synthetic_paths))
    real_sample = select_samples(real_paths, paired_count, seed, "real")
    synthetic_sample = select_samples(synthetic_paths, paired_count, seed, "synthetic")
    output = Path(output).absolute()
    report_path = output.with_suffix(".json")
    if output.suffix.lower() != ".png":
        raise ValueError("A saída deve ter extensão .png; o relatório usa o mesmo nome com .json.")
    if output.exists() or report_path.exists():
        raise ValueError("Saída já existe; escolha outro --output para preservar os arquivos.")
    if any(output.resolve().is_relative_to(directory) or report_path.resolve().is_relative_to(directory)
           for directory in (real_dir, synthetic_dir, ROOT / "dataset/real")):
        raise ValueError("Grave a comparação fora dos diretórios de imagens de entrada/dataset real.")
    reject_evaluation_path(output)
    stats = {"real_train": dataset_stats(real_dir, real_paths, real_sample, real=True,
                                        allowed_root=Path(real_root).resolve()),
             "synthetic": dataset_stats(synthetic_dir, synthetic_paths, synthetic_sample,
                                        allowed_root=Path(synthetic_root).resolve())}
    images = []
    for real, synthetic in zip(real_sample, synthetic_sample):
        pair = []
        for path in (real, synthetic):
            with Image.open(path) as image:
                image.load()
                pair.append(image.convert("RGB"))
        images.append(pair)
    max_width = max(image.width for pair in images for image in pair)
    max_height = max(image.height for pair in images for image in pair)
    # One shared scale for both domains preserves apparent size and aspect ratio.
    scale = min(3, 760 / max_width, 280 / max_height)
    column_width, row_height, margin, header = 800, 355, 24, 100
    canvas = Image.new("RGB", (2 * column_width + 3 * margin,
                               header + paired_count * row_height + 68), "#eef1f5")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=28)
    font = ImageFont.load_default(size=18)
    draw.text((margin, 18), "REAIS - somente train", fill="#172033", font=title)
    draw.text((2 * margin + column_width, 18), "SINTETICAS", fill="#172033", font=title)
    draw.text((margin, 60), f"Seed {seed} | {paired_count} pares | escala comum {scale:.3f}x | sem pareamento de labels",
              fill="#34445c", font=font)
    for index, pair in enumerate(images):
        y = header + index * row_height
        for column, image in enumerate(pair):
            x = margin + column * (column_width + margin)
            draw.rectangle((x, y, x + column_width, y + row_height - 12), fill="white")
            resized = image.resize(fit_size(image.size, scale), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x + (column_width - resized.width) // 2, y + 12 + (280 - resized.height) // 2))
            path = (real_sample, synthetic_sample)[column][index]
            draw.text((x + 12, y + 296), f"{index + 1:02d}. {path.name[:70]}", fill="#172033", font=font)
            draw.text((x + 12, y + 320), f"Original: {image.width} x {image.height} px", fill="#34445c", font=font)
    draw.text((margin, canvas.height - 50), "Compare manualmente: caracteres, ocupacao, espacamento, cores, contraste, ruido, linhas e distorcao.",
              fill="#34445c", font=font)
    report = {
        "seed": seed, "requested": count, "pairs": paired_count, "display_scale": scale,
        "comparison_png": str(output), **stats,
        "notes": ["As estatísticas abrangem todos os PNGs disponíveis nas entradas, não só a amostra.",
                  "Comprimentos e caracteres vêm exclusivamente de labels conhecidos do CSV; não há OCR.",
                  "Nenhuma imagem real de validation/test participa da comparação.",
                  "Labels reais em revisão não contribuem para labeled, label_lengths ou characters.",
                  "Não há estimativa de parâmetros de distorção nem calibração automática."],
        "manual_checklist": {name: None for name in [
            "apparent_character_size", "horizontal_vertical_occupancy", "spacing", "character_count_of_unlabeled_images",
            "colors", "contrast", "noise", "interference_lines", "distortion_intensity"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_comparison(canvas, report, output, report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", "--real", type=Path, default=ROOT / "dataset/real/train")
    parser.add_argument("--synthetic-dir", "--synthetic", type=Path, default=ROOT / "dataset/synthetic/train")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "inspection/output/comparison.png")
    args = parser.parse_args(argv)
    try:
        report = compare(args.real_dir, args.synthetic_dir, count=args.count, seed=args.seed, output=args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Erro: {error}\n")
    print(f"Comparação: {args.output}\nRelatório: {args.output.with_suffix('.json')}\nPares: {report['pairs']}")


if __name__ == "__main__":
    main()
