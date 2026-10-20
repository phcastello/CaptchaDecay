# Validação sintética fixa — 300 imagens

Este snapshot contém exatamente as mesmas 300 imagens de validação utilizadas
nos experimentos CRNN A (2.700 imagens de treino) e B/C (9.000 imagens de treino).
Os PNGs existentes foram copiados sem regeneração, transformação ou mudança de
nome. Este conjunto deve permanecer fixo para comparações entre treinamentos;
não deve ser incorporado ao treino. É validação sintética, não teste independente.

## Conteúdo

- `validation/`: 300 PNGs originais de 215 × 80 pixels, total de 1.227.375 bytes.
- `labels.csv`: somente as 300 linhas de validação, com todos os 19 campos
  originais do gerador, incluindo labels, seeds e parâmetros visuais.
- `manifest.json`: hashes SHA-256 de cada PNG e do CSV, contagens e proveniência.
- `.gitattributes`: preserva os bytes dos PNGs e do CSV entre plataformas.

São 1.507 caracteres de referência: 84 labels de comprimento 4, 125 de
comprimento 5 e 91 de comprimento 6. Não há PNGs duplicados por SHA-256.
Labels são preservados literalmente, inclusive `l` minúsculo. A exclusão de
ambiguidades na avaliação real não altera este benchmark sintético.

## Proveniência e integridade

Nomes, labels, todas as colunas de metadata e bytes das imagens foram comparados
entre as cópias locais de:

- `dataset/synthetic/v2_main`;
- `dataset/synthetic/v2_9k_train`;
- `outputs/experiments/dataset-expansion/baseline-dataset`.

Os hashes individuais também coincidiram com os 300 registros de validação do
manifesto histórico `baseline-manifest.json`, criado na preparação da expansão.
O `manifest.json` deste snapshot registra essa proveniência sem depender desses
diretórios para verificar os arquivos versionados.

O SHA-256 do CSV **deste subconjunto** é
`2158f8746674f893770245220177634d69cd77a6298373e8b74d0f9a9b1d7b95`.
Ele difere dos CSVs completos de treino/validação registrados nos checkpoints:
este snapshot contém exclusivamente as 300 linhas de validação. Essa diferença
não representa mudança nas imagens ou nos labels.

## Verificação após clonar

Na raiz do repositório, execute no PowerShell com Python 3:

```powershell
@'
import csv, hashlib, json
from pathlib import Path

root = Path("dataset/synthetic/validation-300")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha256(root / "labels.csv") == manifest["labels_sha256"]
with (root / "labels.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
images = manifest["images"]
assert len(rows) == len(images) == 300
assert {p.name for p in (root / "validation").glob("*.png")} == {r["filename"] for r in images}
assert len({r["sha256"] for r in images}) == 300
assert all(r["split"] == "validation" for r in rows)
assert {r["filename"]: r["label"] for r in rows} == {r["filename"]: r["label"] for r in images}
assert all(sha256(root / "validation" / r["filename"]) == r["sha256"] for r in images)
canonical = json.dumps(images, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == manifest["image_records_sha256"]
assert sum(len(r["label"]) for r in rows) == 1507
print("OK: 300 PNGs e labels íntegros; hashes conferidos.")
'@ | python -
```

Os diretórios de trabalho dos treinamentos permanecem separados e não são
modificados por este snapshot. Ele não contém imagens reais, treino, pesos ou
saídas de inferência.
