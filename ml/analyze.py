"""Offline error diagnosis of a frozen CRNN on synthetic validation only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time

from .analysis_metrics import analyze_predictions
from .analysis_visuals import generate_gallery, generate_plots
from .evaluate import evaluate_checkpoint, export_csv, file_sha256
from .train import atomic_write, positive, nonnegative, versions


PREDICTION_FIELDS = ("filename", "expected", "predicted", "correct", "expected_length",
                     "predicted_length", "edit_distance", "sample_cer", "substitutions",
                     "deletions", "insertions")


def export_predictions(rows, path):
    export_csv(path, PREDICTION_FIELDS, rows, lowercase_booleans=True)


def verify_metrics(evaluation, summary, checkpoint_path):
    """Compare measured metrics to the checkpoint and available historical rows."""
    measured = {**summary["global"], "val_loss": evaluation["metrics"]["loss"]}
    checkpoint = evaluation["checkpoint"]
    sources = {"checkpoint.metrics": checkpoint.get("metrics", {})}
    for row in checkpoint["history"]:
        if row.get("epoch") == checkpoint["epoch"]:
            sources["checkpoint.history"] = row
    history_path = Path(checkpoint_path).parent / "history.csv"
    completed_epochs = checkpoint["epoch"]
    if history_path.is_file():
        with history_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                epoch = int(row["epoch"])
                completed_epochs = max(completed_epochs, epoch)
                if epoch == checkpoint["epoch"]:
                    sources["history.csv"] = row
    differences = []
    comparisons = []
    for source, reference in sources.items():
        for key in ("num_images", "exact_match", "cer", "val_loss"):
            if key not in reference:
                if source == "checkpoint.metrics":
                    differences.append(f"{source}: missing reference metric {key}")
                continue
            expected = float(reference[key])
            actual = float(measured[key])
            tolerance = 1e-6 if key == "val_loss" else 1e-12
            matches = math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
            comparisons.append({"source": source, "metric": key, "recorded": expected,
                                "measured": actual, "matches": matches, "tolerance": tolerance})
            if not matches:
                differences.append(f"{source} {key}: recorded={expected}, measured={actual}")
    for key in ("num_images", "exact_match", "cer"):
        if not math.isclose(float(measured[key]), float(evaluation["metrics"][key]),
                            rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Aligned metrics disagree with existing evaluation for {key}.")
    return {"matches_checkpoint": not differences, "differences": differences,
            "comparisons": comparisons, "completed_epochs_from_history": completed_epochs,
            "preservation_verified": evaluation["preservation_verified"]}


def export_tables(summary, output):
    def table(name, fields, rows):
        def writer(stream):
            csv_writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        atomic_write(output / name, writer)
    stats = ("num_images", "num_correct", "num_incorrect", "exact_match", "cer",
             "mean_edit_distance", "total_reference_characters", "total_edit_distance",
             "substitutions", "deletions", "insertions")
    table("by-length.csv", ("length", *stats),
          ({"length": key, **value} for key, value in summary["by_length"].items()))
    table("by-repeats.csv", ("group", *stats),
          ({"group": key, **value} for key, value in summary["by_repeats"].items()))
    table("substitutions.csv", ("expected", "predicted", "count"), summary["substitutions"])
    matrix = summary["confusion_matrix"]
    def writer(stream):
        csv_writer = csv.writer(stream)
        csv_writer.writerow(["expected/predicted", *matrix["characters"]])
        for char, counts in zip(matrix["characters"], matrix["counts"]):
            csv_writer.writerow([char, *counts])
    atomic_write(output / "confusion-matrix.csv", writer)


def percent(value):
    return f"{value * 100:.4f}%"


def render_report(summary, rows, selected):
    g = summary["global"]
    exp = summary["experiment"]
    verification = summary["verification"]
    lines = ["# Análise de erros da CRNN — validação sintética", "",
             "## Experimento e preservação", "",
             f"- Arquitetura: CNN 1→32→64→128→256, média vertical, BiLSTM de {exp['model_config']['num_layers']} camadas "
             f"(hidden size {exp['model_config']['hidden_size']} por direção), Linear + CTC.",
             f"- Parâmetros: {exp['parameter_count']:,}. Entrada [B,1,80,215]; logits [53,B,{exp['model_config']['num_classes']}].",
             f"- Checkpoint: `{exp['checkpoint']}`; época **{exp['epoch']}**, verificada no conteúdo do arquivo.",
             f"- Épocas concluídas registradas no histórico: {verification['completed_epochs_from_history']}.",
             f"- SHA-256 do checkpoint: `{exp['checkpoint_sha256']}`.",
             f"- Dataset: `{exp['data']}`; split `{exp['split']}`; {g['num_images']} imagens; dispositivo `{exp['device']}`.",
             f"- Charset preservado: `{exp['charset']}`; blank=0. Decoder greedy CTC original, sem correção de previsões.",
             "- Inferência em eval() + torch.inference_mode(), parâmetros congelados; nenhum otimizador foi instanciado na análise.",
             "- Os hashes do checkpoint, CSV e PNGs de validação foram conferidos antes/depois; o state_dict em memória permaneceu idêntico.",
             "- Somente PNGs de validation foram abertos. O CSV compartilhado foi lido para validar metadata e sua identidade; PNGs de train e conjuntos reais não foram consultados.",
             f"- Versões: `{json.dumps(exp['versions'], ensure_ascii=False)}`.", "",
             "## Métricas globais e conferência do histórico", "",
             "| Medida | Valor |", "|---|---:|",
             f"| Imagens | {g['num_images']} |", f"| Sequências corretas | {g['num_correct']} |",
             f"| Sequências incorretas | {g['num_incorrect']} |",
             f"| Exact match | {percent(g['exact_match'])} |", f"| CER global | {g['cer']:.12f} |",
             f"| Caracteres de referência | {g['total_reference_characters']} |",
             f"| Erros de edição | {g['total_edit_distance']} |",
             f"| Validation loss recalculada | {exp['validation_loss']:.12f} |",
             f"| Train loss registrada no checkpoint (não recalculada) | {exp['recorded_metrics'].get('train_loss', 'não disponível')} |", "",
             f"CER = {g['total_edit_distance']} / {g['total_reference_characters']}; "
             "os CERs individuais não foram simplesmente promediados.", ""]
    if verification["matches_checkpoint"]:
        lines.append("**CONFERÊNCIA APROVADA:** métricas recalculadas coincidem com o checkpoint e as referências históricas disponíveis.")
    else:
        lines.extend(["**DIVERGÊNCIA: os resultados NÃO coincidem com as referências. Nenhum número foi ajustado.**", "",
                      *[f"- {item}" for item in verification["differences"]]])
    lines.extend(["", "| Referência | Métrica | Registrado | Recalculado | Confere |",
                  "|---|---|---:|---:|---|"])
    for item in verification["comparisons"]:
        lines.append(f"| {item['source']} | {item['metric']} | {item['recorded']:.12g} | {item['measured']:.12g} | {'sim' if item['matches'] else 'NÃO'} |")
    lines.extend(["", "Tolerância absoluta/relativa: 1e-12 para contagens, exact match e CER; 1e-6 para loss.",
                  "", "## Operações e imagens afetadas", "",
                  "O alinhamento Levenshtein usa custo unitário. No backtrace, empates são resolvidos na ordem "
                  "match exato → substituição → deleção → inserção, mantendo a orientação esperado→previsto. "
                  "A soma S+D+I é verificada contra a distância mínima em cada imagem. Alinhamentos mínimos "
                  "alternativos podem mudar a atribuição das operações/confusões, mas não a distância total.", "",
                  "| Tipo | Operações | Participação nos erros de edição | Imagens afetadas |",
                  "|---|---:|---:|---:|"])
    names = {"substitutions": "Substituições", "deletions": "Deleções", "insertions": "Inserções"}
    for key, name in names.items():
        item = summary["error_types"][key]
        lines.append(f"| {name} | {item['operations']} | {percent(item['share'])} | {item['images']} |")
    lines.extend(["", "Uma imagem pode aparecer em mais de um tipo; as quantidades de imagens não são somadas como operações.", "",
                  "| Combinação de tipos por imagem incorreta | Imagens |", "|---|---:|"])
    for key, value in summary["error_combinations"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "| Distância de edição | Imagens |", "|---|---:|"])
    distances = Counter(row["edit_distance"] for row in rows)
    for key, value in sorted(distances.items()):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Confusões entre caracteres", "",
                  f"Total de substituições alinhadas: **{g['substitutions']}**. Abaixo, até 15 pares com maior frequência "
                  "(empates ordenados pelo caractere esperado e depois pelo previsto).", "",
                  "| Esperado | Previsto | Substituições |", "|---|---|---:|"])
    for item in summary["substitutions"][:15]:
        lines.append(f"| `{item['expected']}` | `{item['predicted']}` | {item['count']} |")
    lines.extend(["", "A matriz completa inclui matches na diagonal e substituições fora dela. Inserções e deleções "
                  "estão separadas e não recebem caracteres fictícios. O painel de substituições facilita ver "
                  "erros sem a predominância da diagonal. As frequências são contagens absolutas, sem normalizar "
                  "pela exposição de cada caractere; não medem dificuldade intrínseca.", "",
                  "![Matriz de confusão](confusion-matrix.png)", "",
                  "Caracteres deletados: `" + json.dumps(summary["deletions"], ensure_ascii=False) + "`.",
                  "Caracteres inseridos: `" + json.dumps(summary["insertions"], ensure_ascii=False) + "`.", "",
                  "## Comprimento de referência", "",
                  "| Comprimento | Imagens | Corretas | Exact match | CER | Edição média | S | D | I |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for length, item in summary["by_length"].items():
        lines.append(f"| {length} | {item['num_images']} | {item['num_correct']} | {percent(item['exact_match'])} | {item['cer']:.6f} | {item['mean_edit_distance']:.6f} | {item['substitutions']} | {item['deletions']} | {item['insertions']} |")
    lines.extend(["", "![Métricas por comprimento](by-length.png)", "",
                  "| Relação entre comprimentos | Imagens | Participação no split |", "|---|---:|---:|"])
    relation_names = {"shorter": "Previsão mais curta", "longer": "Previsão mais longa",
                      "same_correct": "Mesmo comprimento e texto correto", "same_incorrect": "Mesmo comprimento e texto incorreto"}
    for key, name in relation_names.items():
        count = summary["length_comparison"][key]
        lines.append(f"| {name} | {count} | {percent(count / g['num_images'])} |")
    lines.extend(["", "Mesmo comprimento não implica apenas substituições: inserções e deleções podem se compensar.", "",
                  "## Repetições consecutivas", "",
                  "Um label pertence ao grupo repetido quando ao menos dois vizinhos são idênticos (ex.: aa). "
                  "Não basta haver o mesmo caractere em posições distantes.", "",
                  "| Grupo | Imagens | Corretas | Exact match | CER | S | D | I |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for key, item in summary["by_repeats"].items():
        lines.append(f"| {'Com repetições' if key == 'repeated' else 'Sem repetições'} | {item['num_images']} | {item['num_correct']} | {percent(item['exact_match'])} | {item['cer']:.6f} | {item['substitutions']} | {item['deletions']} | {item['insertions']} |")
    repeated = summary["by_repeats"]["repeated"]
    plain = summary["by_repeats"]["non_repeated"]
    lines.extend(["", f"O grupo repetido contém {repeated['num_images']} exemplos; "
                  + ("a amostra é pequena para conclusões estáveis." if repeated["num_images"] <= 50 else "a comparação permanece exploratória e sem controle das demais condições visuais."),
                  "A presença de repetições não demonstra que CTC causou um erro. Não foi realizado teste causal "
                  "ou comparação com outro decoder/modelo.", "", "## Exemplos rastreáveis", "",
                  "| Categoria | Arquivo | Esperado | Previsto | Distância | S/D/I |", "|---|---|---|---|---:|---|"])
    predicates = [("Acerto", lambda r: r["correct"]),
                  ("Substituição", lambda r: r["substitutions"] and not r["deletions"] and not r["insertions"]),
                  ("Deleção", lambda r: r["deletions"] and not r["substitutions"] and not r["insertions"]),
                  ("Inserção", lambda r: r["insertions"] and not r["substitutions"] and not r["deletions"]),
                  ("Múltiplos erros", lambda r: r["edit_distance"] > 1)]
    for category, predicate in predicates:
        candidates = [row for row in rows if predicate(row)]
        if candidates:
            row = sorted(candidates, key=lambda r: (-r["edit_distance"], r["filename"]))[0]
            lines.append(f"| {category} | {row['filename']} | `{row['expected']}` | `{row['predicted']}` | {row['edit_distance']} | {row['substitutions']}/{row['deletions']}/{row['insertions']} |")
        else:
            lines.append(f"| {category} | Não observado isoladamente | | | | |")
    lines.extend(["", "![Galeria de previsões](error-gallery.png)", "",
                  f"Galeria: {sum(row['correct'] for row in selected)} acertos e {sum(not row['correct'] for row in selected)} erros. "
                  f"Seed {exp['selection_seed']}. A amostra é estratificada por comprimento e combinação de tipos de erro; "
                  "até quatro casos extremos são reservados e marcados explicitamente. Isso dá cobertura de condições, "
                  "mas não torna a proporção de erros da galeria igual à frequência no dataset. O CSV contém todas as previsões.", "",
                  "## Diagnóstico: observações, hipóteses e limites", ""])
    relations = summary["length_comparison"]
    equal = relations["same_correct"] + relations["same_incorrect"]
    largest_type = max(summary["error_types"], key=lambda key: summary["error_types"][key]["operations"])
    lines.extend([f"1. **Comprimento:** {equal}/{g['num_images']} ({percent(equal/g['num_images'])}) têm comprimento correto. "
                  f"Há {relations['shorter']} strings curtas demais e {relations['longer']} longas demais; "
                  f"{relations['same_incorrect']} acertam o comprimento, mas erram o texto.",
                  f"2. **Tipo predominante:** {names[largest_type].lower()}, com {summary['error_types'][largest_type]['operations']} "
                  f"operações ({percent(summary['error_types'][largest_type]['share'])} dos erros de edição)."])
    if summary["substitutions"]:
        top = summary["substitutions"][0]
        tied = [item for item in summary["substitutions"] if item["count"] == top["count"]]
        pairs = ", ".join(f"`{item['expected']}→{item['predicted']}`" for item in tied)
        lines.append(f"3. **Confusões:** maior frequência nos pares {pairs}, com {top['count']} ocorrências cada. "
                     "Os pares acima descrevem alinhamentos observados; similaridade visual é uma hipótese, não uma causa demonstrada.")
    else:
        lines.append("3. **Confusões:** nenhuma substituição foi observada.")
    lengths = sorted(summary["by_length"], key=int)
    accuracies = [summary["by_length"][key]["exact_match"] for key in lengths]
    monotonic = len(lengths) > 1 and all(a >= b for a, b in zip(accuracies, accuracies[1:]))
    lines.append("4. **Sequências maiores:** " + ("o exact match diminui monotonicamente nos comprimentos observados, mas isso é uma associação descritiva, sem controle das condições visuais."
                  if monotonic else "o exact match não apresenta queda monotônica com o comprimento. Esses dados não sustentam uma regra simples de que sequências maiores são sempre mais difíceis.")
                  + " CER e edição média constam da tabela; os tamanhos amostrais diferem.")
    if repeated["num_images"] and plain["num_images"]:
        lines.append(f"5. **Repetições:** exact match de {percent(repeated['exact_match'])} com repetições versus "
                     f"{percent(plain['exact_match'])} sem repetições; CER {repeated['cer']:.6f} versus {plain['cer']:.6f}. "
                     "É uma diferença observada, sem atribuição causal à CTC e sem ajuste por comprimento/ruído.")
    else:
        lines.append("5. **Repetições:** falta um dos grupos para realizar comparação.")
    one = distances.get(1, 0)
    lines.extend([f"6. **Extensão dos erros:** {one}/{g['num_incorrect']} sequências incorretas têm uma única edição, "
                  f"e a maior distância observada é {max(distances)}. Há {g['num_correct']} reconhecimentos completos corretos. "
                  "Essas contagens distinguem erros textuais pontuais de sequências com vários erros; não localizam o problema em uma camada da rede.", ""])
    if g["num_incorrect"] and one > g["num_incorrect"] / 2:
        lines.extend(["Predominam erros textuais de uma edição entre as sequências incorretas, com casos residuais de múltiplas edições; "
                      "o padrão não caracteriza falha de reconhecimento em todas as imagens.", ""])
    if summary["deletions"]:
        char = max(summary["deletions"], key=summary["deletions"].get)
        count = summary["deletions"][char]
        exposure = sum(row["expected"].count(char) for row in rows)
        lines.extend([f"**Concentração das deleções:** `{char}` responde por {count}/{g['deletions']} deleções "
                      f"({percent(count/g['deletions'])}). O caractere ocorre {exposure} vezes nas referências, "
                      f"com taxa de deleção alinhada de {percent(count/exposure)}. "
                      "É um padrão descritivo de omissão, sem atribuição causal.", ""])
    lines.extend([
                  "**Hipóteses a investigar futuramente:** sobreposição/distorção e caracteres visualmente similares podem "
                  "contribuir para alguns erros. Nenhuma dessas condições foi anotada ou controlada nesta análise; "
                  "a galeria permite inspeção, sem transformar impressão visual em explicação causal.", "",
                  f"**Limitações:** somente este checkpoint, decoder greedy e {g['num_images']} exemplos sintéticos; nenhuma medida de "
                  "generalização real ou teste independente. A decomposição depende do desempate do alinhamento; confusões "
                  "raras e pequenos subgrupos têm alta incerteza. A loss de treino foi apenas lida do checkpoint. "
                  "A identidade do CSV coincide com a salva no checkpoint; o formato antigo não armazena hashes individuais "
                  "dos PNGs de validação. Por isso, não se afirma uma comparação isolada desses PNGs com hashes históricos: "
                  "verificou-se sua preservação durante esta execução e a reprodução das métricas históricas. "
                  "Os hashes agregados atuais cobrem somente validation; os históricos cobrem train e validation, "
                  "portanto seus valores não são diretamente comparáveis. "
                  "Os gráficos foram gerados com Pillow, pois Matplotlib não está instalado; nenhuma dependência foi baixada.", "",
                  "## Protocolo para comparar 2.700 e aproximadamente 9.000 imagens de treino", "",
                  "Nenhuma imagem adicional foi gerada. Para um experimento futuro:", "",
                  "- Manter a arquitetura inicial, o vocabulário e o mesmo protocolo de preprocessing, greedy decoding e métricas.",
                  "- Congelar exatamente estas 300 imagens/labels de validação, identificados por hashes no summary.json; não misturá-los ao novo treino.",
                  "- Impedir sobreposição de arquivos/conteúdo entre treino ampliado e validação anterior, verificando SHA-256; preservar a configuração sintética e registrar a composição dos lotes.",
                  "- Reservar o mesmo conjunto de teste futuro, sem consultá-lo durante seleção e desenvolvimento.",
                  "- Controlar tanto quanto possível quantidade de atualizações, learning rate e scheduler, além de batch size, seeds, critério de seleção e orçamento computacional.",
                  "- O mesmo número de épocas em um dataset maior implica mais atualizações: registrar e distinguir o efeito de dados adicionais do efeito de orçamento adicional.",
                  "- Comparar os erros pareados nos mesmos 300 exemplos e, se possível, repetir com múltiplas seeds. A ampliação para 10.000 imagens totais não garante suficiência nem melhora.", "",
                  "## Artefatos e reprodução", "",
                  "- `predictions.csv`: todas as previsões com distância, CER individual e S/D/I.",
                  "- `summary.json`: métricas completas, hashes, protocolo e conferência histórica.",
                  "- `by-length.csv`, `by-repeats.csv`, `substitutions.csv`, `confusion-matrix.csv`.",
                  "- `error-gallery.png`, `confusion-matrix.png`, `by-length.png`.",
                  "- `gallery-selection.csv`: arquivos e motivo de seleção para rastrear a galeria.", "",
                  "Na raiz do repositório, escolhendo uma saída nova ou vazia:", "", "```powershell",
                  f".\\.venv-ml\\Scripts\\python.exe -m ml.analyze --data {exp['data']} --split validation --checkpoint {exp['checkpoint']} --device cpu --output outputs/analysis/crnn-v2-baseline-rerun",
                  "```", ""])
    return "\n".join(lines)


def check_output(output, data, checkpoint):
    output, data, checkpoint = Path(output).resolve(), Path(data).resolve(), Path(checkpoint).resolve()
    for source in (data, checkpoint.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Analysis output must be separate from dataset and checkpoint directories.")
    if any(part.casefold() == "real" for part in output.parts):
        raise ValueError("Analysis output cannot target a real dataset directory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Analysis output is not empty: {output}. Choose a new --output.")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose a frozen CRNN using local synthetic validation only.")
    parser.add_argument("--data", type=Path, default=Path("dataset/synthetic/v2_main"))
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--threads", type=positive, default=6)
    parser.add_argument("--seed", type=nonnegative, default=42, help="gallery selection only")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        started = time.perf_counter()
        output = check_output(args.output, args.data, args.checkpoint)
        evaluation = evaluate_checkpoint(args.data, args.split, args.checkpoint, args.device,
                                         args.batch_size, 0, args.threads, strict_dataset_identity=True)
        rows, summary = analyze_predictions(evaluation["metrics"]["predictions"])
        summary["verification"] = verify_metrics(evaluation, summary, args.checkpoint)
        dataset = evaluation["dataset"]
        checkpoint = evaluation["checkpoint"]
        summary["experiment"] = {"data": str(args.data), "split": args.split,
            "checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"], "device": evaluation["device"],
            "model_config": evaluation["model_config"], "parameter_count": evaluation["parameter_count"],
            "charset": evaluation["charset"].characters, "selection_seed": args.seed,
            "validation_loss": evaluation["metrics"]["loss"], "recorded_metrics": checkpoint.get("metrics", {}),
            "versions": versions(), "batch_size": args.batch_size, "threads": args.threads}
        summary["input_identity"] = {"dataset": evaluation["dataset_identity"],
            "checkpoint_dataset": checkpoint["dataset_identity"],
            "validation_png_sha256": {record.filename: dataset.manifest.image_hashes[(args.split, record.filename)]
                                       for record in dataset.records}}
        output.mkdir(parents=True, exist_ok=True)
        export_predictions(rows, output / "predictions.csv")
        # Check the serialized CSV, not just the in-memory aggregation.
        with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            exported = list(csv.DictReader(stream))
        if (len(exported) != summary["global"]["num_images"]
                or sum(int(row["edit_distance"]) for row in exported) != summary["global"]["total_edit_distance"]
                or sum(row["correct"] == "true" for row in exported) != summary["global"]["num_correct"]):
            raise RuntimeError("Exported CSV metrics are inconsistent.")
        export_tables(summary, output)
        selected = generate_gallery(rows, {r.filename: r.path for r in dataset.records},
                                     output / "error-gallery.png", seed=args.seed)
        def write_selection(stream):
            writer = csv.DictWriter(stream, fieldnames=("filename", "correct", "selection", "edit_distance"))
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in selected)
        atomic_write(output / "gallery-selection.csv", write_selection)
        generate_plots(summary, output)
        # Include gallery/plot generation in source preservation verification.
        for record in dataset.records:
            if file_sha256(record.path) != summary["input_identity"]["validation_png_sha256"][record.filename]:
                raise RuntimeError(f"PNG changed during artifact generation: {record.filename}")
        if (file_sha256(args.checkpoint) != evaluation["checkpoint_sha256"]
                or file_sha256(dataset.root / "labels.csv") != dataset.manifest.identity["manifest_sha256"]):
            raise RuntimeError("Checkpoint or labels changed during artifact generation.")
        summary["seconds"] = time.perf_counter() - started
        atomic_write(output / "summary.json", lambda stream: json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False))
        atomic_write(output / "REPORT.md", lambda stream: stream.write(render_report(summary, rows, selected)))
        print(json.dumps({"output": str(output), "epoch": checkpoint["epoch"], **summary["global"],
                          "validation_loss": evaluation["metrics"]["loss"],
                          "matches_checkpoint": summary["verification"]["matches_checkpoint"],
                          "preservation_verified": True}, indent=2))
        if not summary["verification"]["matches_checkpoint"]:
            print("METRIC DIVERGENCE: inspect REPORT.md; no recorded values were adjusted.", file=sys.stderr)
            return 1
        return 0
    except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
