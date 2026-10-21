"""Tiny offline fixtures for Experiment E; no training or CAPTCHA generator.

These tests were added but deliberately not executed in the Experiment E run,
following the user's subsequent instruction to leave automated tests aside.
"""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

from ml.charset import Charset
from ml.compare_expanded import (audit_real, canonical_hash, combine_predictions,
                                 FrozenRealDataset, generate_gallery, image_path,
                                 load_models, run, verify_preservation)
from ml.evaluate import export_csv, file_sha256, frozen_inference
from ml.expanded_metrics import compare_pair, exact_mcnemar, summarize, wilson
from ml.model import CRNN
from ml.train import load_checkpoint


class StatisticsTests(unittest.TestCase):
    def rows(self):
        data = [('train', 'one.png', 'abcd', 'abcd', 'abcd', 'abc'),
                ('validation', 'two.png', 'abcdefgh', 'abc', 'abcdefgh', 'abcdefgh'),
                ('validation', 'three.png', 'xyz', 'x', 'x', 'xyz'),
                ('train', 'four.png', 'ab', '', '', '')]
        result = []
        from ml.metrics import levenshtein
        for split, name, expected, a, b, d in data:
            row = {'filename': name, 'split': split, 'expected': expected,
                   'expected_length': len(expected)}
            for key, predicted in zip(('a', 'b', 'd'), (a, b, d)):
                row.update({f'predicted_{key}': predicted, f'correct_{key}': expected == predicted,
                            f'edit_distance_{key}': levenshtein(expected, predicted),
                            f'predicted_length_{key}': len(predicted)})
            result.append(row)
        return result

    def test_wilson_known_value_boundaries_and_empty(self):
        interval = wilson(5, 10)
        self.assertAlmostEqual(interval['lower'], .236593090512564, places=12)
        self.assertAlmostEqual(interval['upper'], .763406909487436, places=12)
        self.assertAlmostEqual(wilson(0, 10)['lower'], 0.)
        self.assertAlmostEqual(wilson(10, 10)['upper'], 1.)
        self.assertIsNone(wilson(0, 0))
        for args in ((2, 1), (-1, 1), (1., 2), (True, 2)):
            with self.assertRaises(ValueError): wilson(*args)

    def test_exact_mcnemar_known_values_symmetry_and_no_discordance(self):
        self.assertEqual(exact_mcnemar(0, 0)['p_value_two_sided'], 1.)
        self.assertFalse(exact_mcnemar(0, 0)['informative'])
        self.assertEqual(exact_mcnemar(0, 3)['p_value_two_sided'], .25)
        self.assertEqual(exact_mcnemar(0, 6)['p_value_two_sided'], .03125)
        self.assertEqual(exact_mcnemar(1, 5)['p_value_two_sided'], .21875)
        self.assertEqual(exact_mcnemar(5, 1), {**exact_mcnemar(1, 5), 'first_only_correct': 5, 'second_only_correct': 1})
        with self.assertRaises(ValueError): exact_mcnemar(-1, 2)

    def test_paired_transitions_and_distance_changes(self):
        exported, paired = compare_pair(self.rows(), 'b', 'd')
        self.assertEqual(paired['counts']['both_correct'], 1)
        self.assertEqual(paired['counts']['both_incorrect'], 1)
        self.assertEqual(paired['counts']['first_only_correct'], 1)
        self.assertEqual(paired['counts']['second_only_correct'], 1)
        self.assertEqual(paired['mcnemar']['discordant_pairs'], 2)
        self.assertTrue(exported[0]['regression_exact'])
        self.assertTrue(exported[2]['improvement_exact'])
        self.assertEqual(exported[2]['delta_edit_distance_second_minus_first'], -2)

    def test_global_cer_split_metrics_alignment_and_reproducibility(self):
        rows = self.rows()
        summary = summarize(rows)
        self.assertEqual(summary, summarize(rows))
        for model in ('a', 'b', 'd'):
            total = summary['aggregate']['models'][model]['global']
            train = summary['train']['models'][model]['global']
            validation = summary['validation']['models'][model]['global']
            self.assertEqual(total['num_images'], 4)
            self.assertEqual(train['num_images'], 2)
            self.assertEqual(validation['num_images'], 2)
            self.assertEqual(total['total_edit_distance'], train['total_edit_distance'] + validation['total_edit_distance'])
            self.assertEqual(total['total_reference_characters'], 17)
            self.assertEqual(total['cer'], total['total_edit_distance'] / 17)
            self.assertEqual(total['total_edit_distance'], sum(total[k] for k in ('substitutions', 'deletions', 'insertions')))
        self.assertNotEqual(summary['aggregate']['models']['d']['global']['cer'], (1/4 + 0 + 0 + 1)/4)


class ExpandedEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'real'
        self.charset = Charset('abl')
        self.charsets = {k: self.charset for k in ('a', 'b', 'd')}
        self.rows = []
        for split, name, label, color in (('train', 'one.png', 'abba', 40),
                                          ('train', 'excluded.png', 'allb', 90),
                                          ('validation', 'two.png', 'baab', 150)):
            directory = self.data / split
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / name
            Image.new('L', (215, 80), color).save(path)
            self.rows.append({'filename': name, 'label': label, 'split': split,
                              'sha256': file_sha256(path), 'source': 'fixture', 'split_seed': '42'})
        # Only test metadata exists: reading/listing any test image must fail.
        self.rows.append({'filename': 'reserved.png', 'label': '', 'split': 'test',
                          'sha256': '0' * 64, 'source': 'fixture', 'split_seed': '42'})
        self.write_labels()
        model = CRNN(self.charset.num_classes, hidden_size=8, num_layers=1)
        self.protocol = {'models': {}}
        for key, epoch, count in (('a', 95, 2700), ('b', 28, 9000), ('d', 46, 20000)):
            path = self.root / key / 'best.pt'
            path.parent.mkdir()
            payload = dict(format_version=1, epoch=epoch, model_config=model.config,
                           model_state_dict=model.state_dict(), charset=self.charset.to_dict(),
                           optimizer_state_dict={}, config={}, rng_state={}, best_metrics={}, history=[],
                           dataset_identity={'counts': {'train': count, 'validation': 300}})
            # CPU tensors serialized with CUDA device tags: loader must remap them.
            with mock.patch('torch.serialization.location_tag', return_value='cuda:0'):
                torch.save(payload, path)
            self.protocol['models'][key] = {'path': str(path), 'epoch': epoch,
                                            'train_images': count, 'sha256': file_sha256(path)}

    def write_labels(self):
        with (self.data / 'labels.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_all_three_checkpoints_and_cuda_to_cpu_compatibility(self):
        checkpoints, metadata = load_models(self.protocol)
        self.assertEqual([checkpoints[k]['epoch'] for k in ('a', 'b', 'd')], [95, 28, 46])
        self.assertTrue(all(m['all_weight_tensors_loaded_on_cpu'] for m in metadata.values()))
        with mock.patch('ml.train.torch.load', wraps=torch.load) as loader:
            load_checkpoint(self.protocol['models']['d']['path'])
        self.assertEqual(loader.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})

    def test_wrong_hash_epoch_charset_and_last_checkpoint_refused(self):
        original_hash = self.protocol['models']['a']['sha256']
        self.protocol['models']['a']['epoch'] = 94
        with self.assertRaisesRegex(ValueError, 'epoch'): load_models(self.protocol)
        self.protocol['models']['a']['epoch'] = 95
        self.protocol['models']['a']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA-256'): load_models(self.protocol)
        self.protocol['models']['a']['sha256'] = original_hash
        b_path = self.protocol['models']['b']['path']
        changed = load_checkpoint(b_path)
        changed['charset'] = Charset('abi').to_dict()
        torch.save(changed, b_path)
        self.protocol['models']['b']['sha256'] = file_sha256(b_path)
        with self.assertRaisesRegex(ValueError, 'charsets'): load_models(self.protocol)
        self.protocol['models']['a']['path'] = str(self.root / 'last.pt')
        with self.assertRaisesRegex(ValueError, 'last.pt'): load_models(self.protocol)

    def test_allowed_splits_manifest_determinism_l_rule_and_no_test_access(self):
        original_open, original_iterdir = Path.open, Path.iterdir
        def safe_open(path, *args, **kwargs):
            if 'test' in path.parts: raise AssertionError('Reserved split opened')
            return original_open(path, *args, **kwargs)
        def safe_iterdir(path):
            if 'test' in path.parts: raise AssertionError('Reserved split listed')
            return original_iterdir(path)
        with mock.patch.object(Path, 'open', safe_open), mock.patch.object(Path, 'iterdir', safe_iterdir):
            audit = audit_real(self.data, self.charsets)
            self.assertEqual(audit, audit_real(self.data, self.charsets))
            dataset = FrozenRealDataset(audit, self.charset)
            self.assertEqual(len(dataset), 2)
            self.assertEqual({dataset[i]['filename'] for i in range(len(dataset))}, {'train/one.png', 'validation/two.png'})
        self.assertEqual(audit['counts']['train']['excluded'], 1)
        excluded = next(r for r in audit['records'] if not r['eligible'])
        self.assertEqual(excluded['label_original'], 'allb')
        self.assertEqual(excluded['exclusion_reason'], 'human_rule_contains_lowercase_l')
        self.rows.reverse()
        self.write_labels()
        reordered = audit_real(self.data, self.charsets)
        self.assertEqual(reordered['records'], audit['records'])
        self.assertEqual(reordered['manifest_sha256'], audit['manifest_sha256'])
        with mock.patch.object(Path, 'iterdir', side_effect=AssertionError('Too late')):
            with self.assertRaisesRegex(ValueError, 'test'): audit_real(self.data, self.charsets, splits=('test',))
        with self.assertRaisesRegex(ValueError, 'test'): image_path(self.data, 'test', 'reserved.png')

    def test_symlink_into_test_is_blocked_before_opening(self):
        path = self.data / 'train' / 'one.png'
        path.unlink()
        try: path.symlink_to(self.data / 'test' / 'reserved.png')
        except OSError: self.skipTest('Symlinks unavailable')
        with self.assertRaisesRegex(ValueError, 'Redirected'): audit_real(self.data, self.charsets)

    def test_missing_label_remains_pending_without_inference(self):
        self.rows[0]['label'] = ''
        self.write_labels()
        audit = audit_real(self.data, self.charsets)
        self.assertEqual(audit['pending_count'], 1)
        self.assertEqual(audit['records'][1]['status'], 'pending')
        self.protocol.update(real_data=str(self.data), synthetic_data=str(self.root/'synthetic'),
                             real_labels_sha256=file_sha256(self.data/'labels.csv'))
        protocol_path = self.root/'protocol.json'
        protocol_path.write_text(json.dumps(self.protocol))
        with mock.patch('ml.compare_expanded.frozen_inference', side_effect=AssertionError('Forbidden')) as infer:
            with self.assertRaisesRegex(ValueError, 'eligibility'): run(protocol_path, self.root/'analysis')
            infer.assert_not_called()

    def test_uppercase_incompatible_label_requires_explicit_human_decision(self):
        self.rows[0]['label'] = 'abIL'
        self.write_labels()
        audit = audit_real(self.data, self.charsets)
        record = next(r for r in audit['records'] if r['filename'] == 'one.png')
        self.assertEqual(record['label_original'], 'abIL')
        self.assertEqual(record['status'], 'pending')
        self.assertEqual(record['exclusion_reason'], '')
        decision = {'split': 'train', 'filename': 'one.png', 'label': 'abIL',
                    'sha256': self.rows[0]['sha256'], 'reason': 'human_excluded_uppercase', 'authority': 'explicit fixture decision'}
        audit = audit_real(self.data, self.charsets, [decision])
        self.assertEqual(audit['pending_count'], 0)
        self.assertEqual(next(r for r in audit['records'] if r['filename'] == 'one.png')['exclusion_reason'], 'human_excluded_uppercase')
        decision['label'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'no longer matches'): audit_real(self.data, self.charsets, [decision])

    def test_integrity_corruption_missing_and_unlisted_images(self):
        path = self.data/'train'/'one.png'
        path.write_bytes(b'corrupt')
        self.rows[0]['sha256'] = file_sha256(path)
        self.write_labels()
        with self.assertRaises(OSError): audit_real(self.data, self.charsets)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing='): audit_real(self.data, self.charsets)
        Image.new('L', (215, 80), 40).save(path)
        self.rows[0]['sha256'] = file_sha256(path)
        self.write_labels()
        Image.new('L', (215, 80), 10).save(self.data/'train'/'unlisted.png')
        with self.assertRaisesRegex(ValueError, 'unlisted='): audit_real(self.data, self.charsets)

    def test_exclusions_are_applied_before_forward_and_model_state_is_frozen(self):
        audit = audit_real(self.data, self.charsets)
        checkpoints, _ = load_models(self.protocol)
        before = {k: v.clone() for k, v in checkpoints['a']['model_state_dict'].items()}
        real_forward = CRNN.forward
        def forward(model, images):
            self.assertFalse(model.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(images.shape[0], 2)
            return real_forward(model, images)
        with mock.patch.object(CRNN, 'forward', forward):
            predictions = frozen_inference(checkpoints['a'], FrozenRealDataset(audit, self.charset), 'cpu', 16)['metrics']['predictions']
        self.assertEqual(len(predictions), 2)
        for key, value in before.items(): self.assertTrue(torch.equal(value, checkpoints['a']['model_state_dict'][key]))

    def test_csv_exports_pairs_gallery_and_original_preservation(self):
        audit = audit_real(self.data, self.charsets)
        predictions = [{'filename': r['split']+'/'+r['filename'], 'expected': r['label_original'],
                        'predicted': r['label_original']} for r in audit['records'] if r['eligible']]
        rows = combine_predictions(audit, dict.fromkeys(('a', 'b', 'd'), predictions))
        path = self.root/'predictions.csv'
        export_csv(path, tuple(rows[0]), rows, lowercase_booleans=True)
        with path.open(newline='') as stream: exported = list(csv.DictReader(stream))
        self.assertEqual(len(exported), 2)
        self.assertTrue(all(r['correct_d'] == 'true' for r in exported))
        pair_rows, _ = compare_pair(rows, 'b', 'd')
        export_csv(self.root/'paired.csv', tuple(pair_rows[0]), pair_rows, lowercase_booleans=True)
        gallery = self.root/'gallery.png'
        generate_gallery(rows, audit, gallery)
        with Image.open(gallery) as image:
            self.assertEqual(image.format, 'PNG')
            self.assertGreater(image.height, 80)
        verify_preservation(audit, {})
        predictions.pop()
        with self.assertRaisesRegex(ValueError, 'exactly'): combine_predictions(audit, dict.fromkeys(('a', 'b', 'd'), predictions))


if __name__ == '__main__':
    unittest.main()
