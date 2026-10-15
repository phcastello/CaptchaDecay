"""Organiza cópias das amostras reais em 60/20/20 sem alterar as aquisições originais."""

import argparse
import csv
import hashlib
import os
import random
import tempfile
from pathlib import Path

SPLITS = {"train": 60, "validation": 20, "test": 20}
COLUMNS = ["filename", "label", "split", "sha256", "source", "split_seed"]


def split_destinations(count, seed):
    counts = {name: count * weight // 100 for name, weight in SPLITS.items()}
    order = sorted(SPLITS, key=lambda name: count * SPLITS[name] % 100, reverse=True)
    for name in order[:count - sum(counts.values())]:
        counts[name] += 1
    destinations = [name for name, total in counts.items() for _ in range(total)]
    random.Random(seed).shuffle(destinations)
    return destinations


def organize_real(source="dataset", output="dataset/real", *, seed=42):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2147483647:
        raise ValueError("seed deve ser inteiro entre 0 e 2147483647.")
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("source e output devem ser diferentes; os originais serão preservados.")
    originals = {}
    for split in SPLITS:
        for path in sorted((source / split).glob("*.png")):
            data = path.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Arquivo não-PNG encontrado: {path}")
            digest = hashlib.sha256(data).hexdigest()
            originals.setdefault(digest, []).append(path)

    previous = {}
    labels = output / "labels.csv"
    if labels.exists():
        with labels.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != COLUMNS:
                raise ValueError("Cabeçalho incompatível em labels.csv; arquivo preservado.")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Linha incompleta em labels.csv; arquivo preservado.")
                if (row["filename"] != Path(row["filename"]).name or
                        row["split"] not in SPLITS or row["filename"] in previous):
                    raise ValueError("Referência inválida ou duplicada em labels.csv.")
                previous[row["filename"]] = row

    records = {}
    digests = set()
    for split in SPLITS:
        for path in sorted((output / split).glob("*.png")):
            data = path.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Arquivo não-PNG encontrado: {path}")
            digest = hashlib.sha256(data).hexdigest()
            if digest in digests or path.name in records:
                raise ValueError("Imagem real duplicada entre splits; nenhum arquivo foi alterado.")
            digests.add(digest)
            row = previous.get(path.name)
            if row and (row["split"] != split or row["sha256"] != digest):
                raise ValueError("Imagem não corresponde ao labels.csv; arquivo preservado.")
            origin = ";".join(path.as_posix() for path in originals.get(digest, []))
            # This split predates this run; its original decision seed is unknown.
            records[path.name] = row or dict(zip(COLUMNS, [path.name, "", split, digest, origin, ""]))
    if previous.keys() - records.keys():
        raise ValueError("labels.csv referencia imagem ausente; arquivo preservado.")

    new = sorted(set(originals) - digests)
    for digest, split in zip(new, split_destinations(len(new), seed)):
        name = f"{digest}.png"
        if name in records:
            raise ValueError(f"Nome já utilizado para conteúdo diferente: {name}")
        records[name] = dict(zip(COLUMNS, [name, "", split, digest,
                                          ";".join(path.as_posix() for path in originals[digest]), str(seed)]))
    if not records:
        raise ValueError("Nenhuma imagem real encontrada nas pastas train/validation/test.")

    for split in SPLITS:
        (output / split).mkdir(parents=True, exist_ok=True)
    for digest in new:
        record = records[f"{digest}.png"]
        data = originals[digest][0].read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Imagem original mudou durante a organização; execute novamente.")
        # Exclusive creation: existing images can never be overwritten.
        with (output / record["split"] / record["filename"]).open("xb") as stream:
            stream.write(data)

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=output, suffix=".tmp", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records[name] for name in sorted(records))
        temporary = stream.name
    os.replace(temporary, labels)
    return {split: sum(row["split"] == split for row in records.values()) for split in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/real"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        counts = organize_real(args.source, args.output, seed=args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Erro: {error}\n")
    print("Organizado: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
