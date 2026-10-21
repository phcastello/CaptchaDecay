"""Experiment E: offline, CPU-only A/B/D comparison; real/test is reserved."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch
from torch.utils.data import Dataset

from labeling.labeler import LabelStore, parse_state
from .analysis_metrics import analyze_predictions
from .analysis_visuals import _font, _wrap
from .charset import Charset
from .compare_transfer import json_write
from .dataset import read_image_tensor
from .evaluate import evaluate_checkpoint, export_csv, file_sha256, frozen_inference
from .expanded_metrics import MODELS, compare_pair, summarize
from .train import atomic_write, load_checkpoint, positive


SPLITS = ('train', 'validation')
PREDICTION_FIELDS = ('filename', 'split', 'expected',
                     'predicted_a', 'predicted_b', 'predicted_d',
                     'correct_a', 'correct_b', 'correct_d',
                     'edit_distance_a', 'edit_distance_b', 'edit_distance_d',
                     'expected_length', 'predicted_length_a', 'predicted_length_b', 'predicted_length_d')
ELIGIBILITY_FIELDS = ('filename', 'split', 'label_original', 'label_status', 'eligible',
                      'status', 'exclusion_reason', 'pending_reason', 'unknown_characters', 'sha256')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plain_path(path):
    """Reject redirects before resolving or opening any image, including ancestors."""
    if '://' in str(path) or '..' in Path(path).parts:
        raise ValueError('Only direct local paths without parent traversal are permitted.')
    path = Path(path).absolute()
    for part in reversed((path, *path.parents)):
        if part.is_symlink() or part.is_junction():
            raise ValueError(f'Redirected path refused: {part}')
    return path


def image_path(root, split, filename):
    if split not in SPLITS:
        raise ValueError('real/test access forbidden; only train and validation are allowed.')
    if (not filename or Path(filename).name != filename or '/' in filename
            or '\\' in filename or ':' in filename or Path(filename).suffix.lower() != '.png'):
        raise ValueError('Expected a plain PNG filename.')
    return plain_path(Path(root) / split / filename)


def audit_real(data, charsets, human_exclusions=(), *, splits=SPLITS):
    """Enumerate/open only explicit allowed splits; no general real-data loader."""
    if not splits or len(set(splits)) != len(splits) or any(s not in SPLITS for s in splits):
        raise ValueError('Only train and validation are allowed; real/test access forbidden.')
    root = plain_path(data)
    labels_path = plain_path(root / 'labels.csv')
    labels_bytes = labels_path.read_bytes()
    reader = csv.DictReader(io.StringIO(labels_bytes.decode('utf-8-sig'), newline=''), strict=True)
    required = {'filename', 'split', 'label', 'sha256'}
    if not reader.fieldnames or not required.issubset(reader.fieldnames) or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError('Invalid real labels CSV header.')
    # Shared CSV metadata is filtered before any split path is constructed.
    selected = []
    for row in reader:
        if row.get('split') not in splits:
            continue
        if None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed allowed-split CSV row.')
        image_path(root, row['split'], row['filename'])
        selected.append(row)
    if len({(r['split'], r['filename'].casefold()) for r in selected}) != len(selected):
        raise ValueError('Duplicate allowed-split filename.')
    state_path = plain_path(root / 'labeling_state.json')
    state_bytes = state_path.read_bytes() if state_path.exists() else None
    state = parse_state(state_bytes)
    decisions = {}
    for decision in human_exclusions:
        if decision['split'] not in splits:
            raise ValueError('Human decisions must refer to a selected allowed split.')
        key = (decision['split'], decision['filename'])
        if key in decisions or not decision.get('reason') or not decision.get('authority'):
            raise ValueError('Duplicate or undocumented human exclusion.')
        decisions[key] = decision
    records = []
    for split in splits:
        directory = plain_path(root / split)
        entries = list(directory.iterdir())
        for path in entries:
            plain_path(path)
        images = {p.name for p in entries if p.is_file() and p.suffix.lower() == '.png'}
        split_rows = sorted((r for r in selected if r['split'] == split), key=lambda r: r['filename'])
        names = {r['filename'] for r in split_rows}
        if images != names or not images:
            raise ValueError(f'{split}: missing={sorted(names-images)}, unlisted={sorted(images-names)}, count={len(images)}')
        for row in split_rows:
            path = image_path(root, split, row['filename'])
            if path.stat().st_nlink != 1:
                raise ValueError(f'Hardlinked image refused without inspecting other splits: {path}')
            contents = path.read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            if digest != row['sha256']:
                raise ValueError(f'Image SHA-256 mismatch: {split}/{row["filename"]}')
            with Image.open(io.BytesIO(contents)) as image:
                if image.format != 'PNG' or image.size != (215, 80):
                    raise ValueError(f'Expected original 215 x 80 PNG: {path}')
                image.verify()
            with Image.open(io.BytesIO(contents)) as image:
                image.load()
            label = row['label']
            marked = LabelStore.identity(row) in state['review']
            unknown = {key: ''.join(sorted(set(label) - set(c.characters))) for key, c in charsets.items()}
            excluded, pending = [], []
            if 'l' in label:
                excluded.append('human_rule_contains_lowercase_l')
            decision = decisions.pop((split, row['filename']), None)
            if decision:
                if decision['label'] != label or decision['sha256'] != digest:
                    raise ValueError('Human exclusion no longer matches original label/image.')
                excluded.append(decision['reason'])
            if not label.strip():
                pending.append('missing_human_label')
            if marked:
                pending.append('flagged_for_human_review')
            if set(label) & set('LI'):
                pending.append('uppercase_L_or_I_requires_human_review')
            if any(unknown.values()):
                pending.append('incompatible_charset_requires_human_review')
            if any(c.isspace() or c in '?*' for c in label):
                pending.append('possibly_incomplete_or_uncertain_label')
            status = 'excluded' if excluded else 'pending' if pending else 'eligible'
            records.append({'filename': row['filename'], 'split': split, 'label_original': label,
                            'label_status': 'missing' if not label.strip() else 'review_flagged' if marked else 'saved_unflagged',
                            'eligible': status == 'eligible', 'status': status,
                            'exclusion_reason': ';'.join(excluded), 'pending_reason': ';'.join(pending),
                            'unknown_characters': json.dumps(unknown, sort_keys=True), 'sha256': digest})
    if decisions:
        raise ValueError('Human exclusions refer to absent images.')
    duplicate_hashes = Counter(r['sha256'] for r in records)
    return {'root': str(root), 'splits': list(splits), 'records': records,
            'labels_sha256': hashlib.sha256(labels_bytes).hexdigest(),
            'state_sha256': hashlib.sha256(state_bytes).hexdigest() if state_bytes is not None else None,
            'state_present': state_bytes is not None, 'manifest_sha256': canonical_hash(records),
            'pending_count': sum(r['status'] == 'pending' for r in records),
            'counts': {split: {'available': sum(r['split'] == split for r in records),
                              'labels_nonempty': sum(r['split'] == split and bool(r['label_original'].strip()) for r in records),
                              'eligible': sum(r['split'] == split and r['eligible'] for r in records),
                              'excluded': sum(r['split'] == split and r['status'] == 'excluded' for r in records),
                              'pending': sum(r['split'] == split and r['status'] == 'pending' for r in records)} for split in splits},
            'duplicates_within_allowed_splits': {h: n for h, n in duplicate_hashes.items() if n > 1},
            'human_exclusions': list(human_exclusions), 'test_accessed': False}


def verify_preservation(audit, checkpoints):
    root = plain_path(audit['root'])
    if file_sha256(plain_path(root / 'labels.csv')) != audit['labels_sha256']:
        raise RuntimeError('Original real labels changed.')
    state = plain_path(root / 'labeling_state.json')
    if (file_sha256(state) if state.exists() else None) != audit['state_sha256']:
        raise RuntimeError('Original labeling state changed.')
    for split in audit['splits']:
        files = list(plain_path(root / split).iterdir())
        for path in files:
            plain_path(path)
        if {p.name for p in files if p.is_file() and p.suffix.lower() == '.png'} != {
                r['filename'] for r in audit['records'] if r['split'] == split}:
            raise RuntimeError('Allowed image inventory changed.')
    for row in audit['records']:
        path = image_path(root, row['split'], row['filename'])
        if path.stat().st_nlink != 1 or file_sha256(path) != row['sha256']:
            raise RuntimeError('Original image changed or redirected.')
    for path, digest in checkpoints.items():
        if file_sha256(plain_path(path)) != digest:
            raise RuntimeError('Original checkpoint changed.')


class FrozenRealDataset(Dataset):
    def __init__(self, audit, charset):
        self.audit, self.charset = audit, charset
        self.records = [r for r in audit['records'] if r['eligible']]
        for row in self.records:
            image_path(audit['root'], row['split'], row['filename'])
            charset.encode(row['label_original'], row['filename'])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {'filename': row['split'] + '/' + row['filename'], 'label': row['label_original'],
                'image': read_image_tensor(image_path(self.audit['root'], row['split'], row['filename'])),
                'target': torch.tensor(self.charset.encode(row['label_original']), dtype=torch.long)}


def combine_predictions(audit, predictions):
    enriched = {key: {r['filename']: r for r in analyze_predictions(predictions[key])[0]} for key in MODELS}
    expected = {r['split'] + '/' + r['filename'] for r in audit['records'] if r['eligible']}
    if any(set(enriched[k]) != expected for k in MODELS):
        raise ValueError('Models must use exactly the frozen eligible images.')
    rows = []
    for record in audit['records']:
        if not record['eligible']:
            continue
        identity = record['split'] + '/' + record['filename']
        row = {'filename': record['filename'], 'split': record['split'],
               'expected': record['label_original'], 'expected_length': len(record['label_original'])}
        for key in MODELS:
            source = enriched[key][identity]
            if source['expected'] != record['label_original']:
                raise ValueError('Model references differ from the frozen human labels.')
            for field in ('predicted', 'correct', 'edit_distance', 'predicted_length'):
                row[field + '_' + key] = source[field]
        rows.append(row)
    return rows


def gallery_tags(row):
    tags = []
    if all(row['correct_' + k] for k in MODELS): tags.append('todos acertam')
    if not any(row['correct_' + k] for k in MODELS): tags.append('todos erram')
    if row['correct_b'] and not row['correct_a']: tags.append('B corrigiu A')
    if row['correct_d'] and not row['correct_b']: tags.append('D corrigiu B')
    if row['correct_b'] and not row['correct_d']: tags.append('D regrediu B')
    return tags


def generate_gallery(rows, audit, output):
    """All eligible images at native size, with three unchanged predictions."""
    font, title = _font(13), _font(20)
    width, gap, columns, height, top = 350, 12, 3, 340, 100
    canvas = Image.new('RGB', (columns * (width + gap) + gap,
                              top + ((len(rows) + columns - 1) // columns) * (height + gap)), '#eef2f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), 'Experimento E: train + validation (exploratorio)', font=title, fill='#172033')
    draw.text((gap, 38), 'Verde: todos acertam | Vermelho: todos erram | Azul: D corrige B | Laranja: D regride B', font=font, fill='#172033')
    draw.text((gap, 60), f'{len(rows)} imagens; referencias e elegibilidade congeladas antes da inferencia.', font=font, fill='#172033')
    draw.text((gap, 80), 'Roxo: outras transicoes; etiquetas identificam as correcoes de A para B.', font=font, fill='#172033')
    for index, row in enumerate(rows):
        x, y = gap + index % columns * (width + gap), top + index // columns * (height + gap)
        tags = gallery_tags(row)
        color = ('#b45309' if 'D regrediu B' in tags else '#2563eb' if 'D corrigiu B' in tags else
                 '#15803d' if 'todos acertam' in tags else '#b91c1c' if 'todos erram' in tags else '#7c3aed')
        draw.rectangle((x, y, x + width, y + height - gap), fill='white', outline=color, width=3)
        with Image.open(image_path(audit['root'], row['split'], row['filename'])) as image:
            canvas.paste(image.convert('RGB'), (x + (width - 215) // 2, y + 8))
        captions = [row['split'] + '/' + row['filename'], 'Ground truth: ' + row['expected']]
        captions += [k.upper() + ': ' + (row['predicted_' + k] or '<vazio>')
                     + (' | acerto' if row['correct_' + k] else ' | erro')
                     + f' | d={row["edit_distance_" + k]}' for k in MODELS]
        captions.append(' / '.join(tags))
        yy = y + 96
        for caption in captions:
            for line in _wrap(caption, draw, font, width - 20):
                draw.text((x + 10, yy), line, font=font, fill='#172033')
                yy += 19
    atomic_write(output, lambda stream: canvas.save(stream, format='PNG'), binary=True)


def load_models(protocol):
    checkpoints, metadata = {}, {}
    for key in MODELS:
        spec = protocol['models'][key]
        path = plain_path(spec['path'])
        if path.name != 'best.pt':
            raise ValueError('Experiment E requires best.pt; last.pt is forbidden.')
        digest = file_sha256(path)
        if digest != spec['sha256']:
            raise ValueError(f'Incorrect checkpoint SHA-256: {key}')
        checkpoint = load_checkpoint(path)  # map_location='cpu', including CUDA-trained files.
        if checkpoint['epoch'] != spec['epoch']:
            raise ValueError(f'Incorrect checkpoint epoch: {key}')
        if checkpoint['dataset_identity']['counts']['train'] != spec['train_images']:
            raise ValueError(f'Incorrect recorded training dataset: {key}')
        checkpoints[key] = checkpoint
        metadata[key] = {**spec, 'model_config': checkpoint['model_config'],
                         'charset': checkpoint['charset'], 'training_config': checkpoint['config'],
                         'dataset_identity': checkpoint['dataset_identity'],
                         'all_weight_tensors_loaded_on_cpu': all(t.device.type == 'cpu' for t in checkpoint['model_state_dict'].values())}
    first = checkpoints['a']
    for checkpoint in checkpoints.values():
        if checkpoint['charset'] != first['charset'] or checkpoint['model_config'] != first['model_config']:
            raise ValueError('Model architectures/charsets differ.')
        if checkpoint['model_config']['num_classes'] != Charset.from_dict(checkpoint['charset']).num_classes:
            raise ValueError('Model classes and charset differ.')
    return checkpoints, metadata


def synthetic_gate(protocol, output, batch_size, threads):
    root = plain_path(protocol['synthetic_data'])
    historical_path = plain_path(root / 'historical-validation/manifest.json')
    if file_sha256(historical_path) != protocol['historical_manifest_sha256']:
        raise ValueError('Historical synthetic manifest changed.')
    historical = json.loads(historical_path.read_bytes())
    historical_labels = plain_path(root / 'historical-validation/labels.csv')
    if file_sha256(historical_labels) != historical['labels_sha256']:
        raise ValueError('Historical synthetic labels changed.')
    if canonical_hash(historical['images']) != historical['image_records_sha256']:
        raise ValueError('Historical image records changed.')
    result = {}
    for key in MODELS:
        spec = protocol['models'][key]
        evaluation = evaluate_checkpoint(root, 'validation', spec['path'], 'cpu', batch_size, 0, threads)
        records = evaluation['dataset'].records
        actual = [{'filename': r.filename, 'label': r.label,
                   'sha256': evaluation['dataset'].manifest.image_hashes[('validation', r.filename)],
                   'bytes': r.path.stat().st_size} for r in records]
        if sorted(actual, key=lambda r: r['filename']) != sorted(historical['images'], key=lambda r: r['filename']):
            raise ValueError('Synthetic validation differs from the historical PNGs/labels.')
        predictions, stats = analyze_predictions(evaluation['metrics']['predictions'])
        measured = stats['global']
        matches = (measured['num_images'] == protocol['synthetic_num_images']
                   and measured['total_reference_characters'] == protocol['synthetic_reference_characters']
                   and measured['num_correct'] == spec['synthetic_correct']
                   and measured['total_edit_distance'] == spec['synthetic_edits'])
        result[key] = {'matches_historical': matches, 'metrics': stats,
                       'dataset_identity': evaluation['dataset_identity'],
                       'historical_manifest_sha256': protocol['historical_manifest_sha256'],
                       'historical_labels_sha256': historical['labels_sha256'],
                       'historical_image_records_sha256': historical['image_records_sha256'],
                       'preservation_verified': evaluation['preservation_verified'],
                       'parameter_count': evaluation['parameter_count']}
        export_csv(output / f'synthetic-regression-{key}.csv', tuple(predictions[0]), predictions, lowercase_booleans=True)
        if evaluation['parameter_count'] != protocol['parameter_count']:
            raise ValueError('Unexpected parameter count.')
    json_write(output / 'synthetic-regression.json', result)
    if not all(r['matches_historical'] for r in result.values()):
        raise ValueError('Synthetic regression failed; real inference blocked. See synthetic-regression.json.')
    return result


def previous_validation(scopes, expected):
    a = scopes['validation']['models']['a']['global']
    b = scopes['validation']['models']['b']['global']
    pair = scopes['validation']['paired']['A_vs_B']['counts']
    measured = {'num_images': a['num_images'], 'reference_characters': a['total_reference_characters'],
                'a_correct': a['num_correct'], 'a_edits': a['total_edit_distance'],
                'b_correct': b['num_correct'], 'b_edits': b['total_edit_distance'],
                **{k: pair[k] for k in ('both_correct', 'both_incorrect', 'first_only_correct', 'second_only_correct')}}
    return {'expected': expected, 'measured': measured, 'matches': measured == expected,
            'differences': {k: {'expected': expected.get(k), 'measured': measured.get(k)}
                            for k in expected.keys() | measured.keys() if expected.get(k) != measured.get(k)}}


def verify_exports(output, audit, rows):
    """Read the serialized deliverables back; never read reserved source images."""
    with (output / 'predictions.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != PREDICTION_FIELDS:
            raise RuntimeError('Unexpected prediction CSV columns.')
        exported = list(reader)
    expected = [{field: str(r[field]).lower() if isinstance(r[field], bool) else str(r[field])
                 for field in PREDICTION_FIELDS} for r in rows]
    if exported != expected:
        raise RuntimeError('Exported predictions differ from measured values.')
    with (output / 'eligibility.csv').open(encoding='utf-8', newline='') as stream:
        eligibility = list(csv.DictReader(stream))
    if len(eligibility) != len(audit['records']) or {
            (r['split'], r['filename']) for r in eligibility if r['eligible'] == 'true'} != {
            (r['split'], r['filename']) for r in rows}:
        raise RuntimeError('Serialized eligibility and predictions differ.')
    with (output / 'paired-comparison.csv').open(encoding='utf-8', newline='') as stream:
        paired = list(csv.DictReader(stream))
    calculated = [r for first, second in combinations(MODELS, 2)
                  for r in compare_pair(rows, first, second)[0]]
    expected_paired = [{key: str(value).lower() if isinstance(value, bool) else str(value)
                        for key, value in row.items()} for row in calculated]
    if paired != expected_paired:
        raise RuntimeError('Serialized paired comparisons differ.')
    gallery_files = [output / 'comparison-gallery.png',
                     *(output / f'comparison-gallery-{n + 1:02}.png' for n in range((len(rows) + 11) // 12))]
    for path in gallery_files:
        with Image.open(path) as image:
            if image.format != 'PNG':
                raise RuntimeError('Invalid gallery format.')
            image.verify()
    return {'prediction_rows': len(exported), 'eligibility_rows': len(eligibility),
            'paired_rows': len(paired), 'gallery_files': len(gallery_files), 'verified': True}


def run(protocol_path, output, *, batch_size=16, threads=6):
    protocol_path, output = plain_path(protocol_path), plain_path(output)
    protocol_hash = file_sha256(protocol_path)
    protocol = json.loads(protocol_path.read_bytes())
    data, synthetic_data = plain_path(protocol['real_data']), plain_path(protocol['synthetic_data'])
    protected = [data, synthetic_data, *(plain_path(s['path']).parent for s in protocol['models'].values())]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be separate from models and datasets.')
    if output.exists() and any(p.name not in ('preflight', 'incoming') or not p.is_dir() or p.is_symlink() for p in output.iterdir()):
        raise ValueError('Output contains existing results; choose a fresh output directory.')
    checkpoints, models = load_models(protocol)
    charsets = {k: Charset.from_dict(c['charset']) for k, c in checkpoints.items()}
    audit = audit_real(data, charsets, protocol.get('human_exclusions', ()))
    if audit['labels_sha256'] != protocol['real_labels_sha256']:
        raise ValueError('Original real labels differ from the experiment protocol.')
    output.mkdir(parents=True, exist_ok=True)
    export_csv(output / 'eligibility.csv', ELIGIBILITY_FIELDS, audit['records'], lowercase_booleans=True)
    json_write(output / 'eligibility.json', audit)
    if audit['pending_count'] or not any(r['eligible'] for r in audit['records']):
        atomic_write(output / 'REPORT.md', lambda f: f.write('# Experimento E pendente\n\nElegibilidade requer revisão humana; nenhuma inferência real executada.\n'))
        raise ValueError('Unresolved human eligibility; see eligibility.csv. No real inference executed.')
    frozen_manifest_hash = file_sha256(output / 'eligibility.csv')
    torch.set_num_threads(threads)
    synthetic = synthetic_gate(protocol, output, batch_size, threads)
    hashes = {Path(m['path']): m['sha256'] for m in models.values()}
    predictions = {}
    for key in MODELS:
        verify_preservation(audit, hashes)
        if file_sha256(output / 'eligibility.csv') != frozen_manifest_hash:
            raise RuntimeError('Frozen eligibility CSV changed.')
        inference = frozen_inference(checkpoints[key], FrozenRealDataset(audit, charsets[key]), 'cpu', batch_size, 0)
        predictions[key] = inference['metrics']['predictions']
    rows = combine_predictions(audit, predictions)
    scopes = summarize(rows)
    reproduction = previous_validation(scopes, protocol['previous_validation'])
    export_csv(output / 'predictions.csv', PREDICTION_FIELDS, rows, lowercase_booleans=True)
    paired_rows = [r for left, right in combinations(MODELS, 2) for r in compare_pair(rows, left, right)[0]]
    export_csv(output / 'paired-comparison.csv', tuple(paired_rows[0]), paired_rows, lowercase_booleans=True)
    generate_gallery(rows, audit, output / 'comparison-gallery.png')
    for start in range(0, len(rows), 12):
        generate_gallery(rows[start:start + 12], audit, output / f'comparison-gallery-{start // 12 + 1:02}.png')
    exported = verify_exports(output, audit, rows)
    verify_preservation(audit, hashes)
    if file_sha256(protocol_path) != protocol_hash or file_sha256(output / 'eligibility.csv') != frozen_manifest_hash:
        raise RuntimeError('Protocol or frozen eligibility changed.')
    source_files = ('model.py', 'evaluate.py', 'analyze.py', 'decode.py', 'dataset.py', 'metrics.py',
                    'analysis_metrics.py', 'compare_expanded.py', 'expanded_metrics.py', 'train.py')
    summary = {'schema_version': 1, 'experiment': 'E',
               'status': 'complete' if reproduction['matches'] else 'historical_real_divergence_requires_investigation',
               'models': models, 'dataset': audit, 'scopes': scopes,
               'synthetic_regression': synthetic, 'previous_real_validation': reproduction,
               'configuration': {'device': 'cpu', 'batch_size': batch_size, 'threads': threads, 'num_workers': 0,
                    'model_eval': True, 'inference_mode': True, 'state_dict_preservation_checked': True,
                    'decoder': 'ml.decode.greedy_decode', 'preprocessing': 'ml.dataset.read_image_tensor',
                    'normalization': 'grayscale float32 [-1,1], native 215x80, no resize',
                    'alignment_tie_break': 'exact match, substitution, deletion, insertion',
                    'cer': 'sum(edit_distance)/sum(reference_length)', 'training': False,
                    'eligibility_rule': 'exclude original human labels containing lowercase l before inference',
                    'human_exclusions': protocol.get('human_exclusions', []),
                    'statistics': 'Wilson 95%; exact two-sided McNemar; exploratory; no multiplicity adjustment'},
               'provenance': {'protocol': str(protocol_path), 'protocol_sha256': protocol_hash,
                    'eligibility_csv_sha256': frozen_manifest_hash,
                    'source_sha256': {str(Path('ml') / name): file_sha256(Path(__file__).parent / name) for name in source_files},
                    'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                        'os_release': Path('/etc/os-release').read_text() if Path('/etc/os-release').is_file() else platform.system(),
                        'torch': str(torch.__version__), 'numpy': np.__version__, 'pillow': PIL.__version__,
                        'mkldnn_enabled': torch.backends.mkldnn.enabled}},
               'tests': {'status': 'not_run_per_user_instruction',
                         'note': 'User requested leaving automated tests aside. Earlier preflight ran 32 existing tests; full suite and new tests were not executed.'},
               'preservation_verified': True, 'test_images_accessed': False,
               'export_verification': exported,
               'gallery': {'all_eligible_images': len(rows), 'page_size': 12, 'pages': (len(rows) + 11) // 12}}
    json_write(output / 'summary.json', summary)
    atomic_write(output / 'REPORT.md', lambda f: f.write(render_report(summary, rows)))
    atomic_write(output / 'test-results.txt', lambda f: f.write(
        'Experimento E: testes automatizados não executados nesta etapa, por instrução explícita do usuário.\n'
        'A suíte completa incluiria treino/geração em fixtures e foi deixada de lado.\n'
        '32 testes existentes passaram na pré-auditoria anterior; log preservado em preflight/.\n'
        'Os testes novos estão em tests/test_ml_compare_expanded.py; não foram executados.\n'
        'Verificações de avaliação executadas: hashes, épocas, regressão sintética A/B/D, integridade PNG,\n'
        'manifesto congelado antes da inferência real, métricas e decomposição determinística, preservação de pesos/dados.\n'
        'git diff --check: consultar registro da conferência final.\n'))
    return summary, rows


def render_report(summary, rows):
    """Scientific report generated entirely from the frozen measured results."""
    def percent(v): return f'{v * 100:.4f}%'
    def metric_table(scope):
        lines = ['| Modelo | n | Corretas | Incorretas | Exact match | Wilson 95% | CER | Edições/caracteres | S/D/I |',
                 '|---|---:|---:|---:|---:|---|---:|---|---|']
        for key in MODELS:
            g = summary['scopes'][scope]['models'][key]['global']
            ci = g['exact_match_wilson_95']
            lines.append(f'| {key.upper()} | {g["num_images"]} | {g["num_correct"]} | {g["num_incorrect"]} | {percent(g["exact_match"])} | {percent(ci["lower"])}–{percent(ci["upper"])} | {percent(g["cer"])} | {g["total_edit_distance"]}/{g["total_reference_characters"]} | {g["substitutions"]}/{g["deletions"]}/{g["insertions"]} |')
        return lines + ['']
    audit = summary['dataset']
    environment = summary['provenance']['environment']
    lines = ['# Experimento E — transferência ampliada dos modelos A, B e D', '', '## 12.1. Objetivo', '',
             'Avaliar a transferência de CRNNs treinadas com dados sintéticos para imagens reais, com ênfase nas mudanças pareadas B→D.', '',
             '## 12.2. Metodologia', '',
             'Arquitetura existente: CNN 1→32→64→128→256, média vertical, BiLSTM de duas camadas com hidden size 128 por direção, projeção para 31 classes CTC; 1.186.815 parâmetros em cada modelo.', '',
             'CPU para A/B/D; model.eval(), requires_grad_(False), torch.inference_mode(). O loader usa map_location="cpu" e weights_only=True, inclusive para D treinado em CUDA. Pesos e buffers de BatchNorm são comparados em memória antes/depois; nenhum otimizador é instanciado.', '',
             f'Batch={summary["configuration"]["batch_size"]}; threads={summary["configuration"]["threads"]}; workers=0. Sistema: {environment["platform"]}; Python {environment["python"].split()[0]}, PyTorch {environment["torch"]}, NumPy {environment["numpy"]}, Pillow {environment["pillow"]}. Ambiente completo em summary.json.', '',
             'Pré-processamento e inferência reutilizados de ml.dataset, ml.evaluate e ml.model: PNG original 215×80, grayscale float32, normalização [-1,1], sem resize, recorte, threshold ou augmentation. Decoder greedy CTC de ml.decode; nenhum ajuste com base em labels.', '',
             'Exact match exige igualdade literal. CER global = soma das distâncias de Levenshtein / soma dos comprimentos humanos, sem média simples dos CERs individuais. O alinhamento determinístico de ml.metrics/ml.analysis_metrics desempata por match, substituição, deleção e inserção; S+D+I é conferido contra a distância mínima.', '',
             '**Separação metodológica:** real/train é diagnóstico; suas características calibraram o gerador sintético v2, embora seus PNGs não tenham atualizado os pesos. real/validation é desenvolvimento já consultado. A união é exploratória, não um teste independente. real/test permaneceu completamente reservado, sem listagem de conteúdo, leitura de imagens ou inferência.', '',
             'Wilson 95% é apresentado para exact match (inclusive subgrupos pequenos, sujeitos a grande incerteza). McNemar exato bicaudal usa somente os pares discordantes, com probabilidade binomial 0,5. Estatísticas exploratórias, sem correção de múltiplas comparações; ausência de significância não demonstra equivalência.', '',
             '## 12.3. Integridade', '', '| Modelo | Treino sintético | Época interna | Checkpoint | SHA-256 |', '|---|---:|---:|---|---|']
    for key in MODELS:
        m = summary['models'][key]
        lines.append(f'| {key.upper()} | {m["train_images"]} | {m["epoch"]} | {m["path"]} | {m["sha256"]} |')
    lines += ['', f'Charset idêntico e literal: `{summary["models"]["a"]["charset"]["characters"]}`; blank=0.', '',
              f'CSV real original SHA-256: `{audit["labels_sha256"]}`. Manifesto canônico: `{audit["manifest_sha256"]}`. CSV de elegibilidade congelado: `{summary["provenance"]["eligibility_csv_sha256"]}`.', '',
              '| Split | Imagens | Labels preenchidos | Elegíveis | Excluídas | Pendentes |', '|---|---:|---:|---:|---:|---:|']
    for split, c in audit['counts'].items():
        lines.append(f'| {split} | {c["available"]} | {c["labels_nonempty"]} | {c["eligible"]} | {c["excluded"]} | {c["pending"]} |')
    lines += ['', 'As imagens permitidas foram conferidas contra SHA-256 do CSV, formato PNG, dimensões e decodificação completa. Correspondência CSV/PNG e nomes duplicados foram conferidos. Duplicatas de bytes entre splits permitidos: ' + json.dumps(audit['duplicates_within_allowed_splits']) + '.', '',
              'O CSV compartilhado foi filtrado para train/validation antes da construção de caminhos. Nenhum loader percorreu automaticamente todos os splits. Não se inspecionaram PNGs de test para duplicatas.', '',
              f'Estado de revisão presente: {audit["state_present"]}. Label salvo sem marca é o critério operacional; o rotulador não fornece confirmação positiva de dupla revisão.', '',
              'A elegibilidade foi gravada e congelada antes das previsões reais. A regra de l é literal e baseada somente no label original. Não houve conversão de caixa, equivalências, alteração de labels ou seleção por previsão. Exclusões individuais:', '',
              '| Split | Arquivo | Label original | Motivo |', '|---|---|---|---|']
    for r in audit['records']:
        if not r['eligible']:
            lines.append(f'| {r["split"]} | {r["filename"]} | `{r["label_original"]}` | {r["exclusion_reason"] or r["pending_reason"]} |')
    lines += ['', '**Exclusão adicional explicitamente autorizada pelo usuário:** a imagem train/f25eba4f1ecc9f210a7652ce51bdb5e82f847c11e2b85ddc8ecf3ea7d502f874.png, label original `6giL`, foi retirada por conter L maiúsculo fora do charset. Essa decisão foi tomada antes da inferência real e é separada das 20 exclusões por l minúsculo. O label foi preservado.', '',
              '### Regressão sintética em CPU', '', '| Modelo | Corretas | CER | Histórico reproduzido |', '|---|---:|---|---|']
    for key in MODELS:
        r = summary['synthetic_regression'][key]
        g = r['metrics']['global']
        lines.append(f'| {key.upper()} | {g["num_correct"]}/{g["num_images"]} | {g["total_edit_distance"]}/{g["total_reference_characters"]} | {r["matches_historical"]} |')
    lines += ['', 'Os 300 PNGs/labels atuais coincidem com o manifesto histórico preservado no dataset D. A/B usam essa mesma validação, embora os CSVs completos de seus datasets de treino tenham identidades distintas. O fingerprint da avaliação cobre o CSV completo e somente PNGs de validation, diferindo por escopo do fingerprint de treino. Nenhum dado foi regenerado.', '',
              '### Reprodução real anterior A/B', '',
              f'Reproduziu: **{summary["previous_real_validation"]["matches"]}**. Medido: `{json.dumps(summary["previous_real_validation"]["measured"])}`. Divergências: `{json.dumps(summary["previous_real_validation"]["differences"])}`.', '',
              '## 12.4. Resultados por split', '', '### real/train — diagnóstico', ''] + metric_table('train')
    lines += ['### real/validation — desenvolvimento', ''] + metric_table('validation')
    lines += ['## 12.5. Resultado agregado', '', '**União train + validation: análise exploratória de desenvolvimento; não é avaliação final independente.**', ''] + metric_table('aggregate')
    lines += ['## 12.6. Comparações pareadas', '',
              '| Escopo | Par | Ambos acertam | Ambos erram | Só primeiro | Só segundo | Menor distância primeiro | Menor distância segundo | Empate distância | Discordantes | McNemar p |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scope, content in summary['scopes'].items():
        for pair, p in content['paired'].items():
            c = p['counts']
            lines.append(f'| {scope} | {pair} | {c["both_correct"]} | {c["both_incorrect"]} | {c["first_only_correct"]} | {c["second_only_correct"]} | {c["first_lower_distance"]} | {c["second_lower_distance"]} | {c["equal_distance"]} | {p["mcnemar"]["discordant_pairs"]} | {p["mcnemar"]["p_value_two_sided"]:.6g} |')
    lines += ['', '### Mudanças individuais B→D', '',
              '| Split | Arquivo | Referência | B | D | Distância B→D | Mudança de acerto |', '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['predicted_b'] != r['predicted_d']:
            change = 'D corrigiu B' if r['correct_d'] and not r['correct_b'] else 'D regrediu B' if r['correct_b'] and not r['correct_d'] else 'ambos erram; ver distância'
            lines.append(f'| {r["split"]} | {r["filename"]} | `{r["expected"]}` | `{r["predicted_b"]}` | `{r["predicted_d"]}` | {r["edit_distance_b"]}→{r["edit_distance_d"]} | {change} |')
    lines += ['', 'Todos os pares A/B, B/D e A/D, inclusive empates, constam em paired-comparison.csv e summary.json.', '',
              '## 12.7. Análise de erros', '', '| Escopo | Modelo | Comprimento real | n | Corretas | Exact match | CER | Edições/caracteres |', '|---|---|---:|---:|---:|---:|---:|---|']
    for scope, content in summary['scopes'].items():
        for key, stats in content['models'].items():
            for length, g in stats['by_length'].items():
                lines.append(f'| {scope} | {key.upper()} | {length} | {g["num_images"]} | {g["num_correct"]} | {percent(g["exact_match"])} | {percent(g["cer"])} | {g["total_edit_distance"]}/{g["total_reference_characters"]} |')
    lines += ['', '### Frequência dos erros de comprimento', '',
              '| Escopo | Modelo | Mais curtas | Mais longas | Total com comprimento incorreto | Frequência | Mesmo comprimento, texto incorreto |',
              '|---|---|---:|---:|---:|---:|---:|']
    for scope, content in summary['scopes'].items():
        for key, stats in content['models'].items():
            relation = stats['length_comparison']
            wrong_length = relation['shorter'] + relation['longer']
            lines.append(f'| {scope} | {key.upper()} | {relation["shorter"]} | {relation["longer"]} | {wrong_length} | {percent(wrong_length / stats["global"]["num_images"])} | {relation["same_incorrect"]} |')
    lines += ['', 'A distribuição completa comprimento real→previsto está em summary.json. Mesmo comprimento não implica somente substituições: inserções/deleções podem se compensar.', '',
              '### Confusões de caracteres', '']
    for scope, content in summary['scopes'].items():
        for key, stats in content['models'].items():
            confusions = ', '.join(f'{r["expected"]}→{r["predicted"]} ({r["count"]})' for r in stats['substitutions']) or 'nenhuma'
            lines += [f'- {scope}, {key.upper()}: comprimentos `{json.dumps(stats["length_comparison"])}`; substituições: {confusions}; caracteres deletados `{json.dumps(stats["deletions"])}`; inseridos `{json.dumps(stats["insertions"])}`.']
    hard = [r for r in rows if not any(r['correct_' + k] for k in MODELS)]
    lines += ['', '### Imagens persistentemente difíceis', '', '| Split | Arquivo | Referência | A | B | D | Distâncias A/B/D |', '|---|---|---|---|---|---|---|']
    for r in hard:
        lines.append(f'| {r["split"]} | {r["filename"]} | `{r["expected"]}` | `{r["predicted_a"]}` | `{r["predicted_b"]}` | `{r["predicted_d"]}` | {r["edit_distance_a"]}/{r["edit_distance_b"]}/{r["edit_distance_d"]} |')
    lines += ['', 'Frequências de confusões são contagens absolutas do alinhamento, sem controle por exposição ou condições visuais; não demonstram dificuldade intrínseca ou causas. Inserções/deleções não são convertidas em substituições por posição.', '',
              'A galeria inclui todas as elegíveis em comparison-gallery.png e páginas numeradas de até 12 exemplos, com originais em resolução nativa, ground truth e previsões A/B/D. Etiquetas identificam acertos/erros compartilhados, correções A→B, correções B→D e regressões B→D. Nenhum label foi revisto a partir dessas previsões.', '',
              '## 12.8. Interpretação', '']
    b, d = (summary['scopes']['aggregate']['models'][k]['global'] for k in ('b', 'd'))
    p = summary['scopes']['aggregate']['paired']['B_vs_D']
    c = p['counts']
    train_pair = summary['scopes']['train']['paired']['B_vs_D']
    validation_pair = summary['scopes']['validation']['paired']['B_vs_D']
    train_c, validation_c = train_pair['counts'], validation_pair['counts']
    consistent = all(group['second_only_correct'] > group['first_only_correct']
                     for group in (train_c, validation_c))
    direction = ('Sim, nesta amostra, em exact match e CER.' if d['exact_match'] > b['exact_match'] and d['cer'] < b['cer'] else
                 'As direções das métricas precisam ser distinguidas; não há superioridade conjunta demonstrada.')
    lines += [f'1. **D supera B nas imagens reais?** {direction} Exact match B={b["num_correct"]}/{b["num_images"]}, D={d["num_correct"]}/{d["num_images"]}, diferença de {(d["exact_match"]-b["exact_match"])*100:+.4f} pontos percentuais; CER B={b["total_edit_distance"]}/{b["total_reference_characters"]}, D={d["total_edit_distance"]}/{d["total_reference_characters"]}.', '',
              f'2. **Consistência/concentração:** D corrige {c["second_only_correct"]} dos {b["num_incorrect"]} erros de B e regride em {c["first_only_correct"]} acertos; possui menor distância em {c["second_lower_distance"]} imagens, maior em {c["first_lower_distance"]} e empata em {c["equal_distance"]}. As correções se concentram em {train_c["second_only_correct"]} imagens train e {validation_c["second_only_correct"]} validation. A direção de exact match é favorável em ambos os splits: **{consistent}**. Os números absolutos são pequenos e não constituem replicações independentes.', '',
              f'3. **Mais dados sintéticos continuam trazendo benefícios?** {direction} Isso descreve o experimento completo: D também modificou amostragem, orçamento de treinamento e scheduler. Não se pode atribuir o ganho exclusivamente às 20.000 imagens nem extrapolar uma curva de escala a partir de três execuções.', '',
              f'4. **Erros compartilhados:** total de imagens erradas pelos três modelos: **{len(hard)}**; as previsões estão na tabela de casos persistentemente difíceis.', '',
              '5. **Domain gap:** a tabela abaixo compara os domínios descritivamente. A avaliação real exclui l e tem distribuição diferente; diferenças não isolam efeito de domínio, dificuldade ou comprimento. Não há evidência causal suficiente para explicar erros pela aparência das imagens.', '',
              '| Modelo | Exact match sintético | CER sintético | Exact match real agregado | CER real agregado |', '|---|---:|---:|---:|---:|']
    for key in MODELS:
        syn = summary['synthetic_regression'][key]['metrics']['global']
        real = summary['scopes']['aggregate']['models'][key]['global']
        lines.append(f'| {key.upper()} | {percent(syn["exact_match"])} | {percent(syn["cer"])} | {percent(real["exact_match"])} | {percent(real["cer"])} |')
    no_adverse_gap = all(summary['scopes']['aggregate']['models'][key]['global']['exact_match'] >= summary['synthetic_regression'][key]['metrics']['global']['exact_match']
                        and summary['scopes']['aggregate']['models'][key]['global']['cer'] <= summary['synthetic_regression'][key]['metrics']['global']['cer'] for key in MODELS)
    gap_observation = ('Não se observou piora real em exact match ou CER em nenhum dos três modelos; portanto, estes números não sustentam afirmar um domain gap adverso aqui.'
                       if no_adverse_gap else 'Existe piora em pelo menos uma métrica/modelo no real; essa diferença descritiva não isola o efeito do domínio.')
    nominal = ('O agregado cruza o limiar nominal de 5%' if p['mcnemar']['p_value_two_sided'] < .05
               else 'O agregado não cruza o limiar nominal de 5%')
    split_nominal = ('os splits separados não o cruzam' if all(q['mcnemar']['p_value_two_sided'] >= .05 for q in (train_pair, validation_pair))
                     else 'pelo menos um split também cruza esse limiar')
    lines += ['', gap_observation + ' Isso não demonstra ausência de domain gap na população: as referências reais excluem l, são poucas e participaram do desenvolvimento, enquanto o sintético inclui l. Os domínios não foram comparados sob distribuição controlada.', '',
              '6. **Limitações:** amostra pequena e de desenvolvimento, exclusões que restringem a população, labels sem comprovação de dupla revisão, subgrupos muito pequenos, alinhamentos mínimos eventualmente não únicos, múltiplas comparações exploratórias e alterações simultâneas em D. Wilson e McNemar tratam as imagens como observações independentes; possível correlação da coleta não foi estimada. Nenhum resultado estatístico foi usado para escolher checkpoints ou ajustar hiperparâmetros. O conjunto real/test continua reservado.', '',
              f'McNemar B/D agregado: {p["mcnemar"]["discordant_pairs"]} pares discordantes, p={p["mcnemar"]["p_value_two_sided"]:.6g}; train p={train_pair["mcnemar"]["p_value_two_sided"]:.6g}, validation p={validation_pair["mcnemar"]["p_value_two_sided"]:.6g}. {nominal} nesta execução, sem ajuste para múltiplas comparações; {split_nominal}. Os três escopos se sobrepõem e não são três testes independentes. Ausência de significância estatística não demonstra equivalência.', '',
              '## Reprodutibilidade e testes', '',
              'Protocolo versionável: experiments/real-transfer-expanded.json, incluindo hashes dos originais, épocas, regressão e decisão humana sobre 6giL. Hashes do protocolo, fontes e ambiente estão em summary.json. Repetir usando uma saída nova:', '',
              '```bash', '.venv-ml/bin/python -m ml.compare_expanded --protocol experiments/real-transfer-expanded.json --output outputs/analysis/real-transfer-expanded-rerun --batch-size 16 --threads 6', '```', '',
              '**Testes automatizados não executados nesta etapa, por solicitação explícita do usuário.** A suíte completa e os testes novos de tests/test_ml_compare_expanded.py foram deixados de lado. A pré-auditoria anterior executou 32 testes existentes com sucesso; esses não equivalem à suíte completa. O log anterior está preservado em preflight/. Verificações de integridade/regressão/metrificação foram executadas como parte da avaliação. Resultado de git diff --check consta em test-results.txt.', '',
              'Nenhum treinamento, fine-tuning, geração de CAPTCHA, download, requisição HTTP, alteração de modelos/dados originais, commit ou push. As únicas alterações de código são avaliação, estatísticas, protocolo, testes e documentação.', '',
              'Arquivos criados no projeto: `ml/compare_expanded.py`, `ml/expanded_metrics.py`, `experiments/real-transfer-expanded.json`, `tests/test_ml_compare_expanded.py` e `docs/experiment-e.md`. Alterações preexistentes em outros arquivos foram preservadas.', '',
              'Artefatos em `outputs/analysis/real-transfer-expanded/`: REPORT.md, summary.json, eligibility.csv, eligibility.json, predictions.csv, paired-comparison.csv, comparison-gallery.png, comparison-gallery-01.png a comparison-gallery-06.png, synthetic-regression.json, synthetic-regression-a.csv, synthetic-regression-b.csv, synthetic-regression-d.csv e test-results.txt. A pasta preflight preserva o estágio preparatório; incoming preserva os checkpoints recebidos.', '',
              '# RESULTADO DO EXPERIMENTO E', '',
              f'**{direction}** B reconhece {b["num_correct"]}/{b["num_images"]}; D reconhece {d["num_correct"]}/{d["num_images"]}. D corrige {c["second_only_correct"]} erros de B e perde {c["first_only_correct"]} acertos. CER passa de {b["total_edit_distance"]}/{b["total_reference_characters"]} para {d["total_edit_distance"]}/{d["total_reference_characters"]}. Resultado exploratório, limitado à amostra e às exclusões declaradas.', '']
    return '\n'.join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('experiments/real-transfer-expanded.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/analysis/real-transfer-expanded'))
    parser.add_argument('--batch-size', type=positive, default=16)
    parser.add_argument('--threads', type=positive, default=6)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        summary, _ = run(args.protocol, args.output, batch_size=args.batch_size, threads=args.threads)
        print(json.dumps({'status': summary['status'], 'output': str(args.output),
                          'counts': summary['dataset']['counts'],
                          'metrics': {k: v['global'] for k, v in summary['scopes']['aggregate']['models'].items()},
                          'B_vs_D': summary['scopes']['aggregate']['paired']['B_vs_D']}, indent=2))
        return 0 if summary['previous_real_validation']['matches'] else 2
    except (ValueError, RuntimeError, OSError, csv.Error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
