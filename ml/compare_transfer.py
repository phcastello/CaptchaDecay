"""Paired CPU evaluation of frozen synthetic CRNNs on real/validation only.

No training, remote access, label normalization or test-image reads. Eligibility
and exact references are exported before either model produces predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.utils.data import Dataset

from labeling.labeler import parse_csv, parse_state, read_optional, LabelStore
from .analysis_metrics import analyze_predictions
from .analysis_visuals import generate_gallery
from .analyze import check_output, percent
from .charset import Charset
from .dataset import read_image_tensor
from .evaluate import export_csv, file_sha256, frozen_inference
from .train import atomic_write, load_checkpoint, positive, versions


PREDICTION_FIELDS = ("filename", "expected", "predicted_a", "predicted_b", "correct_a",
                     "correct_b", "edit_distance_a", "edit_distance_b", "expected_length",
                     "predicted_length_a", "predicted_length_b")
EXCLUSION_FIELDS = ("filename", "reason", "label_status", "charset_ambiguity")
AUDIT_POLICY = (
    "Saved, nonempty, unflagged labels in both checkpoint vocabularies are eligible. "
    "All members of duplicate SHA-256 groups are excluded (other splits: CSV metadata only). "
    "Review flags and out-of-vocabulary labels are excluded before inference. "
    "Any empty validation label blocks the entire evaluation. No case conversion or substitution. "
    "Per human instruction before inference, exclude every label containing lowercase l, "
    "because it is visually ambiguous with uppercase L/I; never convert these characters. "
    "Lowercase i alone is not an ambiguity flag."
)


def json_write(path, value):
    atomic_write(path, lambda stream: json.dump(value, stream, ensure_ascii=False,
                                              indent=2, allow_nan=False))


def audit_validation(data, charsets):
    """Read shared metadata, but inspect/open PNGs exclusively in validation."""
    if "://" in str(data):
        raise ValueError("Only local files are permitted.")
    root = Path(data).resolve()
    directory = root / "validation"
    if directory.is_symlink() or directory.is_junction() or directory.resolve() != directory:
        raise ValueError("Validation directory must not redirect to another split.")
    csv_path, state_path = root / "labels.csv", root / "labeling_state.json"
    if any(path.is_symlink() or path.is_junction() for path in (csv_path, state_path)):
        raise ValueError("Label metadata must not redirect to another directory.")
    csv_bytes = csv_path.read_bytes()
    all_rows = parse_csv(csv_bytes)
    if len({r['filename'].casefold() for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate case-insensitive CSV filenames.")
    state_bytes = read_optional(state_path)
    state = parse_state(state_bytes)
    selected = sorted((r for r in all_rows if r["split"] == "validation"),
                      key=lambda r: r["filename"])
    files = sorted(directory.iterdir())
    if any(path.is_symlink() or path.is_junction() for path in files):
        raise ValueError("Validation files must not redirect to other splits.")
    images = {path.name for path in files if path.is_file() and path.suffix.lower() == ".png"}
    names = {r["filename"] for r in selected}
    if images != names:
        raise ValueError(f"Validation CSV/image mismatch: unlisted={sorted(images-names)}, missing={sorted(names-images)}")
    if not images:
        raise ValueError("Validation contains no images.")
    sha_counts = Counter(r["sha256"] for r in all_rows)
    records, excluded, eligible, review = [], [], [], []
    for row in selected:
        path = directory / row["filename"]
        digest = file_sha256(path)
        if digest != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {row['filename']}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (215, 80):
                raise ValueError(f"Expected original 215 x 80 PNG: {row['filename']}")
            image.verify()
        marked = LabelStore.identity(row) in state["review"]
        missing = not row["label"].strip()
        unknown = {key: ''.join(sorted(set(row['label']) - set(charset.characters)))
                   for key, charset in charsets.items()}
        reasons = []
        if missing:
            reasons.append("missing_label")
        if marked:
            reasons.append("pending_human_review")
        if 'l' in row['label']:
            reasons.append("human_declared_visual_ambiguity_lowercase_l")
        if any(unknown.values()):
            reasons.append("out_of_vocabulary")
        if sha_counts[digest] > 1:
            reasons.append("duplicate_sha256_in_csv")
        status = "review" if marked else "missing" if missing else "saved_unflagged"
        info = json.dumps({"unknown": unknown, "review_flag": marked,
                           "contains_lowercase_l_or_i": bool(set(row['label']) & set('li')),
                           "human_excluded_lowercase_l": 'l' in row['label']},
                          ensure_ascii=False, sort_keys=True)
        record = {"filename": row["filename"], "expected": row["label"], "sha256": digest,
                  "label_status": status, "eligible": not reasons, "reasons": reasons,
                  "charset_ambiguity": info}
        records.append(record)
        if reasons:
            excluded.append({"filename": row["filename"], "reason": ";".join(reasons),
                             "label_status": status, "charset_ambiguity": info})
        else:
            eligible.append(record)
        if missing or marked or any(unknown.values()) or set(row['label']) & set('li'):
            review.append({**record, "review_needed": bool(missing or marked or any(unknown.values()) or 'l' in row['label']),
                           "note": "Independent pre-inference manifest; original labels unchanged. "
                                   "Human excluded lowercase l before inference due to visual ambiguity with L/I; "
                                   "lowercase i alone is informational."})
    return {"data": str(root), "split": "validation", "policy": AUDIT_POLICY,
            "available": len(images), "eligible_count": len(eligible), "excluded_count": len(excluded),
            "saved_labels": sum(bool(r['expected'].strip()) for r in records),
            "review_flagged": sum(r['label_status'] == 'review' for r in records),
            "missing_labels": sum(not r['expected'].strip() for r in records),
            "state_present": state_bytes is not None,
            "review_evidence": "The labeler has no separate affirmative human-review timestamp/status.",
            "labels_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest() if state_bytes is not None else None,
            "duplicate_sha256_groups_touching_validation": sorted({r['sha256'] for r in selected if sha_counts[r['sha256']] > 1}),
            "records": records, "eligible": eligible, "exclusions": excluded, "review_manifest": review}


def verify_preservation(audit, checkpoints):
    root = Path(audit["data"])
    directory = root / 'validation'
    if directory.is_symlink() or directory.is_junction() or directory.resolve() != directory:
        raise RuntimeError('Validation directory redirected during evaluation.')
    files = list(directory.iterdir())
    if any(path.is_symlink() or path.is_junction() for path in files):
        raise RuntimeError('Validation file redirected during evaluation.')
    if {path.name for path in files if path.is_file() and path.suffix.lower() == '.png'} != {r['filename'] for r in audit['records']}:
        raise RuntimeError('Validation image inventory changed.')
    if file_sha256(root / "labels.csv") != audit["labels_sha256"]:
        raise RuntimeError("Original labels changed.")
    state = root / "labeling_state.json"
    if (file_sha256(state) if state.exists() else None) != audit["state_sha256"]:
        raise RuntimeError("Original review state changed.")
    for row in audit["records"]:
        if file_sha256(root / "validation" / row["filename"]) != row["sha256"]:
            raise RuntimeError(f"Original PNG changed: {row['filename']}")
    for path, digest in checkpoints.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Checkpoint changed: {path}")


class RealValidationDataset(Dataset):
    def __init__(self, audit, charset):
        self.root = Path(audit["data"])
        self.records = audit["eligible"]
        self.charset = charset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {"filename": record["filename"], "label": record["expected"],
                "image": read_image_tensor(self.root / "validation" / record["filename"]),
                "target": torch.tensor(self.charset.encode(record["expected"], record["filename"]),
                                       dtype=torch.long)}


def paired_comparison(predictions_a, predictions_b):
    a, stats_a = analyze_predictions(predictions_a)
    b, stats_b = analyze_predictions(predictions_b)
    right = {r["filename"]: r for r in b}
    if {r['filename'] for r in a} != set(right):
        raise ValueError("Models must use exactly the same eligible subset.")
    concordance = {key: [] for key in ("both_correct", "both_incorrect", "b_only_correct", "a_only_correct")}
    distances = {key: [] for key in ("b_lower", "a_lower", "tie")}
    rows = []
    for left in a:
        r = right[left["filename"]]
        if left["expected"] != r["expected"]:
            raise ValueError("Paired references differ.")
        row = {"filename": left["filename"], "expected": left["expected"],
               "expected_length": left["expected_length"]}
        for key in ("predicted", "correct", "edit_distance", "predicted_length"):
            row[key + "_a"], row[key + "_b"] = left[key], r[key]
        group = ("both_correct" if left['correct'] and r['correct'] else
                 "a_only_correct" if left['correct'] else
                 "b_only_correct" if r['correct'] else "both_incorrect")
        concordance[group].append(row["filename"])
        delta = r["edit_distance"] - left["edit_distance"]
        distances["b_lower" if delta < 0 else "a_lower" if delta > 0 else "tie"].append(row["filename"])
        rows.append(row)
    return rows, {"a": stats_a, "b": stats_b,
                  "paired": {"concordance": concordance, "edit_distance": distances}}


def render_report(summary, rows):
    audit = summary["audit"]
    lines = ["# Transferência sintético → real", "", "## Metodologia", "",
             f"Dataset: `{audit['data']}/validation`; {audit['available']} imagens disponíveis, "
             f"{audit['eligible_count']} elegíveis e {audit['excluded_count']} excluídas.", "",
             f"Labels preenchidos: {audit['saved_labels']}; pendentes: {audit['missing_labels']}; "
             f"marcados para revisão: {audit['review_flagged']}. Estado de revisão presente: {audit['state_present']}.",
             "O rotulador não mantém confirmação positiva de revisão; label salvo sem marca pendente é o critério operacional. "
             "Isso não comprova ausência de ambiguidades humanas.", "",
             "Inclusão: label salvo, não vazio, sem marca de revisão e representável em ambos os charsets. "
             "Exclusão: revisão pendente, caractere fora do vocabulário, duplicata SHA-256 (todos os membros) "
             "ou label contendo l minúsculo, conforme decisão humana explícita anterior à inferência. "
             "Justificativa: o usuário identificou posteriormente que l é visualmente problemático, podendo ser "
             "indistinguível de L/I maiúsculos; o charset não representa L/I. Excluir a imagem inteira evita impor "
             "uma referência incerta, equivalência artificial ou penalização dependente da previsão. "
             "Qualquer label vazio impede toda a avaliação. A presença isolada de i minúsculo não determina exclusão. "
             "Não se convertem L/I para l, não se normaliza caixa e não se substituem desconhecidos.", "",
             "`eligibility.json`, `review-manifest.json` e `exclusions.csv` foram gravados antes da inferência. "
             "O manifesto independente preserva os labels originais e não contém previsões. "
             "Nenhuma inclusão/exclusão ou referência foi alterada após observar os modelos.", "",
             "Somente PNGs de validation foram lidos. O CSV compartilhado é lido para correspondência, nomes e "
             "duplicatas por hashes registrados; hashes de outros splits não foram recalculados e suas imagens não foram abertas.", ""]
    lines += ["| Arquivo excluído | Label original | Motivo |", "|---|---|---|"]
    references = {r['filename']: r['expected'] for r in audit['records']}
    for r in audit['exclusions']:
        lines.append(f"| {r['filename']} | `{references[r['filename']]}` | {r['reason']} |")
    if not audit['exclusions']:
        lines.append("| Nenhum | — | — |")
    lines.append("")
    for key in ('a', 'b'):
        m = summary['models'][key]
        lines += [f"Modelo {key.upper()}: treino sintético com {m['dataset_identity']['counts']['train']} imagens; "
                  f"`{m['path']}`, época {m['epoch']}; SHA-256 `{m['sha256']}`.", "",
                  f"Arquitetura armazenada: `{json.dumps(m['model_config'])}`; configuração de treino: "
                  f"`{json.dumps(m['config'], ensure_ascii=False)}`.", ""]
    lines += [f"Charset exato: `{summary['charset']}`; blank=0. Entrada inteira grayscale float32 "
              "[1,80,215], normalização [-1,1], sem resize, augmentation ou correções. "
              "Inferência exclusivamente CPU, eval(), requires_grad_(False) e torch.inference_mode(); "
              "forward, loss auxiliar e greedy CTC existentes. Nenhum otimizador é instanciado na avaliação.", "",
              f"Batch={summary['batch_size']}; threads={summary['threads']}; versões: "
              f"`{json.dumps(summary['versions'])}`. Checkpoints, CSV, estado de revisão e todos os PNGs de validação "
              "conferidos por SHA-256 antes/depois, inclusive após a galeria; state_dict conferido em memória.", "",
              "Exact match exige igualdade integral. CER global = soma das distâncias de Levenshtein / soma dos "
              "caracteres de referência, sem média dos CERs por imagem. Alinhamento mínimo de custo unitário; "
              "desempate: match, substituição, deleção, inserção. Alinhamentos alternativos podem mudar S/D/I, "
              "mas não a distância total. Excluídos não recebem acerto/erro e não entram nos denominadores.", "",
              "## Resultados", "", "| Métrica | Modelo A | Modelo B | Diferença B − A |", "|---|---:|---:|---:|"]
    labels = {'exact_match': 'Exact match', 'cer': 'CER', 'num_images': 'Imagens',
              'num_correct': 'Sequências corretas', 'num_incorrect': 'Sequências incorretas',
              'total_edit_distance': 'Distância total de Levenshtein',
              'total_reference_characters': 'Caracteres de referência',
              'substitutions': 'Substituições', 'insertions': 'Inserções', 'deletions': 'Deleções'}
    a, b = summary['a']['global'], summary['b']['global']
    for key, label in labels.items():
        x, y = a[key], b[key]
        values = (percent(x), percent(y), f"{(y-x)*100:+.4f} p.p.") if key in ('exact_match', 'cer') else (x, y, f"{y-x:+d}")
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines += ["", f"CER A = {a['total_edit_distance']}/{a['total_reference_characters']}; "
              f"CER B = {b['total_edit_distance']}/{b['total_reference_characters']}.", "",
              "| Comprimento | n | Corretas A | Exact match A | Corretas B | Exact match B |", "|---|---:|---:|---:|---:|---:|"]
    for length, left in summary['a']['by_length'].items():
        right = summary['b']['by_length'][length]
        lines.append(f"| {length} | {left['num_images']} | {left['num_correct']} | {percent(left['exact_match'])} | "
                     f"{right['num_correct']} | {percent(right['exact_match'])} |")
    c = summary['paired']['concordance']
    d = summary['paired']['edit_distance']
    lines += ["", "## Comparação pareada", "", "| | B acerta | B erra |", "|---|---:|---:|",
              f"| A acerta | {len(c['both_correct'])} | {len(c['a_only_correct'])} |",
              f"| A erra | {len(c['b_only_correct'])} | {len(c['both_incorrect'])} |", "",
              f"Distância menor em B: {len(d['b_lower'])}; menor em A: {len(d['a_lower'])}; empate: {len(d['tie'])}.", "",
              "Lista completa por imagem (a distância considera também melhorias parciais entre dois erros):", "",
              "| Arquivo | Referência | A | B | Correto A/B | Distância A/B |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['filename']} | `{r['expected']}` | `{r['predicted_a']}` | `{r['predicted_b']}` | "
                     f"{r['correct_a']}/{r['correct_b']} | {r['edit_distance_a']}/{r['edit_distance_b']} |")
    lines += ["", "Grupos e filenames completos também estão em `summary.json`. Não foi aplicado teste estatístico "
              "ou intervalo de confiança: a análise é descritiva de uma amostra pequena.", "",
              "## Análise qualitativa", ""]
    for key in ('a', 'b'):
        s = summary[key]
        pairs = ', '.join(f"{p['expected']}→{p['predicted']} ({p['count']})" for p in s['substitutions'][:10]) or 'nenhuma'
        lines += [f"Modelo {key.upper()}: substituições frequentes: {pairs}. "
                  f"Caracteres deletados: `{json.dumps(s['deletions'])}`; inseridos: `{json.dumps(s['insertions'])}`. "
                  f"Relação de comprimentos: `{json.dumps(s['length_comparison'])}`.", ""]
    lines += ["![Comparação de todas as imagens elegíveis](comparison-gallery.png)", "",
              "A galeria mostra os PNGs originais, sem transformação, com bordas para os quatro resultados pareados. "
              "Caracteres l/I/L visualmente indistinguíveis não podem ser resolvidos pela previsão; "
              "o modelo não pode emitir L ou I maiúsculos. Todos os labels com l foram excluídos por decisão humana "
              "anterior à inferência; o manifesto registra a justificativa. A galeria só inclui elegíveis. "
              "Labels não foram corrigidos pela galeria.", "",
              "Atribuir causalidade a fonte, ruído, "
              "distorção ou traços exige novos experimentos independentes. As métricas de confusão são contagens, "
              "não probabilidades normalizadas por exposição.", ""]
    if summary.get('qualitative_review'):
        lines += ["Revisão visual posterior à inferência, sem alteração de referências ou elegibilidade:", "",
                  *[paragraph + '\n' for paragraph in summary['qualitative_review']]]
    lines += ["## Limitações", "",
              "Amostra pequena, com incertezas humanas de rotulagem e sem registro independente de dupla revisão. "
              "A exclusão de todos os labels contendo l restringe a população avaliada: as métricas são condicionais "
              "às imagens sem esse caractere e não medem a capacidade de resolver a ambiguidade l/L/I. "
              "Diferenças de domínio e de comprimentos dificultam a comparação direta com os 300 sintéticos. "
              "As validações sintéticas já selecionaram os best.pt; real/validation não foi usada para ajustar "
              "hiperparâmetros ou arquitetura. Não há múltiplas seeds nem evidência causal isolada do efeito de "
              "cardinalidade: épocas, exposições por imagem e dinâmica do treino também diferem. Os percentuais "
              "não podem ser generalizados diretamente a todos os CAPTCHAs reais. Não se acessou real/test.", "",
              "## Conclusão", ""]
    improved = b['exact_match'] > a['exact_match'] and b['cer'] < a['cer']
    lines += [f"1. Neste subconjunto, B {'melhorou exact match e CER' if improved else 'não melhorou simultaneamente exact match e CER'}: "
              f"exact match {percent(a['exact_match'])}→{percent(b['exact_match'])}, CER {percent(a['cer'])}→{percent(b['cer'])}.",
              f"2. A mudança por imagem inclui {len(d['b_lower'])} melhorias de distância, {len(d['a_lower'])} pioras e "
              f"{len(d['tie'])} empates; {len(c['b_only_correct'])} novos acertos contra {len(c['a_only_correct'])} acertos perdidos. "
              "Essas contagens descrevem a consistência observada, sem afirmar significância."]
    domain_comparison = []
    for key in ('a', 'b'):
        real, synthetic = summary[key]['global'], summary['models'][key]['synthetic_metrics']
        domain_comparison.append(f"{key.upper()}: sintético exact match {percent(synthetic['exact_match'])}, CER {percent(synthetic['cer'])}; "
                     f"real {percent(real['exact_match'])}, {percent(real['cer'])}; diferenças real−sintético "
                     f"{(real['exact_match']-synthetic['exact_match'])*100:+.4f} p.p. e {(real['cer']-synthetic['cer'])*100:+.4f} p.p., respectivamente.")
    lines.append('3. ' + ' '.join(domain_comparison))
    lines += [f"4. Em B permanecem {b['num_incorrect']} sequências incorretas, com {b['substitutions']} substituições, "
              f"{b['deletions']} deleções e {b['insertions']} inserções."]
    gaps = all(summary[k]['global']['cer'] > summary['models'][k]['synthetic_metrics']['cer'] for k in ('a', 'b'))
    lines += ["5. " + ("O CER real superior ao sintético em ambos sugere um domain gap relevante nesta amostra; "
                        if gaps else "A comparação entre domínios é descritiva e não basta para estabelecer um domain gap relevante; ") +
              "as causas visuais e a magnitude populacional permanecem hipóteses. Aumento sintético e transferência "
              "não são equivalentes por definição.", ""]
    lines += ["## Reprodutibilidade e verificação", "",
              "Na raiz do repositório (use uma saída nova para repetir):", "", "```powershell",
              f".\\.venv-ml\\Scripts\\python.exe -m ml.compare_transfer --checkpoint-a \"{summary['models']['a']['path']}\" "
              f"--checkpoint-b \"{summary['models']['b']['path']}\" --data \"{audit['data']}\" "
              f"--output outputs/analysis/real-transfer-rerun --batch-size {summary['batch_size']} --threads {summary['threads']}",
              "```", "",
              "Não houve treinamento ou geração de CAPTCHAs no experimento A/B. A suíte histórica, quando executada, "
              "inclui treino e geração somente em fixtures temporários independentes, autorizados separadamente pelo usuário. "
              "Não houve acesso HTTP, aos Correios, à A2000 ou à Aegis. Sem commit ou push.", ""]
    if summary.get('execution_validation'):
        v = summary['execution_validation']
        lines += [f"Suíte completa: `{v['command']}`; {v['tests_run']} testes aprovados, {v['failures']} falhas, "
                  f"{v['errors']} erros, {v['skipped']} skips. Log: [test-results.txt](test-results.txt). "
                  f"git diff --check: {v['diff_check']}. CSV exportado conferido contra summary.json; "
                  "galeria renderizada e inspecionada visualmente. Hashes originais reconferidos após os testes.", ""]
    return '\n'.join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-a', type=Path, required=True)
    p.add_argument('--checkpoint-b', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('dataset/real'))
    p.add_argument('--output', type=Path, default=Path('outputs/analysis/real-transfer'))
    p.add_argument('--split', choices=('validation',), default='validation')
    p.add_argument('--batch-size', type=positive, default=16)
    p.add_argument('--threads', type=positive, default=6)
    p.add_argument('--audit-only', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        output = check_output(args.output, args.data, args.checkpoint_a)
        check_output(output, args.data, args.checkpoint_b)
        paths = {'a': args.checkpoint_a, 'b': args.checkpoint_b}
        hashes = {path: file_sha256(path) for path in paths.values()}
        checkpoints = {key: load_checkpoint(path) for key, path in paths.items()}
        charsets = {key: Charset.from_dict(c['charset']) for key, c in checkpoints.items()}
        if charsets['a'].to_dict() != charsets['b'].to_dict():
            raise ValueError('Checkpoint charsets differ; comparison refused.')
        for c in checkpoints.values():
            if c['model_config'].get('num_classes') != charsets['a'].num_classes:
                raise ValueError('Checkpoint classes incompatible with charset.')
        audit = audit_validation(args.data, charsets)
        output.mkdir(parents=True, exist_ok=True)
        json_write(output / 'eligibility.json', audit)
        json_write(output / 'review-manifest.json', audit['review_manifest'])
        export_csv(output / 'exclusions.csv', EXCLUSION_FIELDS, audit['exclusions'])
        models = {key: {'path': str(paths[key]), 'sha256': hashes[paths[key]],
                        'epoch': c['epoch'], 'model_config': c['model_config'], 'config': c['config'],
                        'dataset_identity': c['dataset_identity'], 'synthetic_metrics': c.get('metrics', {})}
                  for key, c in checkpoints.items()}
        json_write(output / 'checkpoints.json', models)
        if audit['missing_labels'] or not audit['eligible_count'] or args.audit_only:
            verify_preservation(audit, hashes)
            message = (f"Evaluation not executed: available={audit['available']}, eligible={audit['eligible_count']}, "
                       f"excluded={audit['excluded_count']}, missing_labels={audit['missing_labels']}. "
                       "Human labeling: .\\.venv-ml\\Scripts\\python.exe -m labeling.labeler --dataset "
                       f'"{args.data}" --split validation')
            atomic_write(output / 'REPORT.md', lambda stream: stream.write('# Auditoria antes da inferência\n\n' + message + '\n'))
            print(message)
            return 0 if args.audit_only else 2
        torch.set_num_threads(args.threads)
        predictions = {}
        for key, c in checkpoints.items():
            verify_preservation(audit, hashes)
            dataset = RealValidationDataset(audit, charsets[key])
            predictions[key] = frozen_inference(c, dataset, 'cpu', args.batch_size)['metrics']['predictions']
        rows, summary = paired_comparison(predictions['a'], predictions['b'])
        if len(rows) != audit['eligible_count'] or {r['filename'] for r in rows} != {r['filename'] for r in audit['eligible']}:
            raise RuntimeError('Predictions differ from pre-inference eligibility manifest.')
        summary.update(audit=audit, models=models, charset=charsets['a'].characters,
                       device='cpu', batch_size=args.batch_size, threads=args.threads, versions=versions())
        export_csv(output / 'predictions.csv', PREDICTION_FIELDS, rows, lowercase_booleans=True)
        generate_gallery(rows, {r['filename']: Path(audit['data']) / 'validation' / r['filename'] for r in rows},
                         output / 'comparison-gallery.png', paired=True)
        verify_preservation(audit, hashes)
        summary['preservation_verified'] = True
        json_write(output / 'summary.json', summary)
        atomic_write(output / 'REPORT.md', lambda stream: stream.write(render_report(summary, rows)))
        print(json.dumps({'output': str(output), 'audit': {k: audit[k] for k in
                         ('available', 'eligible_count', 'excluded_count', 'missing_labels')},
                          'a': summary['a']['global'], 'b': summary['b']['global'],
                          'paired': {k: {group: len(names) for group, names in groups.items()}
                                     for k, groups in summary['paired'].items()}}, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
