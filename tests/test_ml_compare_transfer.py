"""Frozen local fixtures: no training or CAPTCHA generator is needed here."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from PIL import Image

from labeling.labeler import COLUMNS, LabelStore
from ml.charset import Charset
from ml.analysis_visuals import generate_gallery
from ml.compare_transfer import (audit_validation, main, paired_comparison,
                                 RealValidationDataset, verify_preservation, PREDICTION_FIELDS)
from ml.dataset import read_image_tensor
from ml.evaluate import file_sha256
from ml.model import CRNN
from ml.train import load_checkpoint


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        (self.data / 'validation').mkdir(parents=True)
        self.output = self.root / 'analysis'
        self.charset = Charset('abcil')
        self.charsets = dict(a=self.charset, b=self.charset)
        self.rows = []
        for i, label in enumerate(('abba', 'abcab')):
            path = self.data / 'validation' / f'{i}.png'
            Image.new('RGB', (215, 80), (30 + 90 * i, 255, 0)).save(path)
            self.rows.append(dict(filename=path.name, label=label, split='validation',
                                  sha256=file_sha256(path), source='local', split_seed='42'))
        # Only metadata exists for other splits: any attempt to open their PNGs fails.
        self.rows += [dict(filename=split+'.png', label='', split=split,
                           sha256=hashlib.sha256(split.encode()).hexdigest(), source='local', split_seed='42')
                      for split in ('train', 'test')]
        self.write_csv()
        self.paths = {}
        model = CRNN(self.charset.num_classes, hidden_size=8, num_layers=1)
        for key, epoch, count in (('a', 95, 2700), ('b', 28, 9000)):
            directory = self.root / key
            directory.mkdir()
            path = directory / 'best.pt'
            torch.save(dict(format_version=1, epoch=epoch, model_state_dict=model.state_dict(),
                            optimizer_state_dict={}, model_config=model.config,
                            charset=self.charset.to_dict(), config={'seed': 42},
                            dataset_identity={'counts': {'train': count, 'validation': 300}},
                            rng_state={}, best_metrics={}, history=[],
                            metrics={'exact_match': .5, 'cer': .2}), path)
            self.paths[key] = path

    def write_csv(self):
        with (self.data / 'labels.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

    def args(self):
        return ['--checkpoint-a', str(self.paths['a']), '--checkpoint-b', str(self.paths['b']),
                '--data', str(self.data), '--output', str(self.output), '--threads', '2']

    def run_cli(self, extra=()):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(self.args() + list(extra))
        return code, out.getvalue() + err.getvalue()

    def test_audit_selects_only_validation_and_preserves_data(self):
        before = {p: p.read_bytes() for p in self.data.rglob('*') if p.is_file()}
        audit = audit_validation(self.data, self.charsets)
        self.assertEqual(audit['available'], 2)
        self.assertEqual(audit['eligible_count'], 2)
        self.assertEqual(audit['missing_labels'], 0)  # Empty train/test labels are irrelevant.
        self.assertFalse(audit['state_present'])
        dataset = RealValidationDataset(audit, self.charset)
        self.assertEqual(len(dataset), 2)
        self.assertTrue(torch.equal(dataset[0]['image'], read_image_tensor(self.data/'validation/0.png')))
        verify_preservation(audit, {})
        self.assertEqual(before, {p: p.read_bytes() for p in self.data.rglob('*') if p.is_file()})

    def test_empty_label_blocks_all_inference_and_never_exports_predictions(self):
        self.rows[0]['label'] = ' '
        self.write_csv()
        with mock.patch('ml.compare_transfer.frozen_inference', side_effect=AssertionError('inference forbidden')):
            code, message = self.run_cli()
        self.assertEqual(code, 2, message)
        self.assertIn('--split validation', message)
        self.assertFalse((self.output/'predictions.csv').exists())
        self.assertFalse((self.output/'comparison-gallery.png').exists())
        audit = json.loads((self.output/'eligibility.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['missing_labels'], 1)

    def test_review_and_l_ambiguity_excluded_before_predictions(self):
        self.rows[0]['label'] = 'all'
        self.write_csv()
        state = {'version': 1, 'review': {LabelStore.identity(self.rows[1]): True}}
        (self.data/'labeling_state.json').write_text(json.dumps(state), encoding='utf-8')
        before = (self.data/'labels.csv').read_bytes()
        audit = audit_validation(self.data, self.charsets)
        self.assertEqual(audit['eligible_count'], 0)
        self.assertEqual(audit['excluded_count'], 2)
        self.assertIn('human_declared_visual_ambiguity_lowercase_l', audit['records'][0]['reasons'])
        self.assertIn('pending_human_review', audit['records'][1]['reasons'])
        self.assertEqual((self.data/'labels.csv').read_bytes(), before)
        with mock.patch('ml.compare_transfer.frozen_inference', side_effect=AssertionError):
            self.assertEqual(self.run_cli()[0], 2)

    def test_unknown_L_I_are_not_normalized_and_i_is_not_excluded(self):
        self.rows[0]['label'] = 'aLI'
        self.rows[1]['label'] = 'aib'
        self.write_csv()
        audit = audit_validation(self.data, self.charsets)
        self.assertEqual(audit['eligible_count'], 1)
        self.assertEqual(audit['eligible'][0]['expected'], 'aib')
        self.assertEqual(audit['records'][0]['expected'], 'aLI')
        self.assertIn('out_of_vocabulary', audit['records'][0]['reasons'])
        self.assertEqual(json.loads(audit['records'][0]['charset_ambiguity'])['unknown']['a'], 'IL')

    def test_duplicate_content_excludes_all_members(self):
        first = self.data/'validation/0.png'
        (self.data/'validation/1.png').write_bytes(first.read_bytes())
        self.rows[1]['sha256'] = file_sha256(first)
        self.write_csv()
        audit = audit_validation(self.data, self.charsets)
        self.assertEqual(audit['excluded_count'], 2)
        self.assertTrue(all('duplicate_sha256_in_csv' in r['reasons'] for r in audit['records']))

    def test_duplicate_against_test_metadata_does_not_open_test(self):
        self.rows[-1]['sha256'] = self.rows[0]['sha256']
        self.write_csv()
        audit = audit_validation(self.data, self.charsets)
        self.assertEqual(audit['excluded_count'], 1)
        self.assertEqual(audit['eligible'][0]['filename'], '1.png')

    def test_missing_unlisted_corrupt_and_duplicate_names_refused(self):
        self.rows.append(dict(self.rows[0]))
        self.write_csv()
        with self.assertRaises(ValueError):
            audit_validation(self.data, self.charsets)
        self.rows.pop()
        self.write_csv()
        Image.new('RGB', (215,80)).save(self.data/'validation/extra.png')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            audit_validation(self.data, self.charsets)
        (self.data/'validation/extra.png').unlink()
        (self.data/'validation/0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            audit_validation(self.data, self.charsets)
        (self.data/'validation/0.png').unlink()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            audit_validation(self.data, self.charsets)

    def test_real_frozen_cpu_execution_export_gallery_and_preservation(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        forward = CRNN.forward
        modes = []
        def checked_forward(model, images):
            self.assertTrue((self.output/'eligibility.json').is_file())
            self.assertTrue((self.output/'exclusions.csv').is_file())
            modes.append((model.training, torch.is_inference_mode_enabled(), images.device.type,
                          any(p.requires_grad for p in model.parameters())))
            return forward(model, images)
        open_path = Path.open
        def checked_open(path, *args, **kwargs):
            if path.suffix.lower() == '.png' and path.parent.name in ('test', 'train'):
                raise AssertionError('non-validation image access')
            return open_path(path, *args, **kwargs)
        with mock.patch.object(CRNN, 'forward', checked_forward), \
             mock.patch.object(Path, 'open', checked_open), \
             mock.patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), \
             mock.patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('training forbidden')):
            code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        self.assertTrue(modes)
        self.assertTrue(all(m == (False, True, 'cpu', False) for m in modes))
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents, str(path))
        with (self.output/'predictions.csv').open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(PREDICTION_FIELDS))
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['expected'] for r in rows}, {'abba', 'abcab'})
        s = json.loads((self.output/'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(s['preservation_verified'])
        self.assertEqual(s['models']['a']['epoch'], 95)
        self.assertEqual(s['models']['b']['epoch'], 28)
        self.assertEqual(s['a']['global'], s['b']['global'])
        with Image.open(self.output/'comparison-gallery.png') as image:
            image.verify()
        self.assertEqual(self.run_cli()[0], 1)  # Existing artifacts are not overwritten.

    def test_incompatible_or_malformed_checkpoints_refused(self):
        c = load_checkpoint(self.paths['b'])
        c['charset'] = Charset('ab').to_dict()
        torch.save(c, self.paths['b'])
        code, message = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn('charsets differ', message)
        self.assertFalse(self.output.exists())
        torch.save({'format_version': 1}, self.paths['b'])
        self.assertEqual(self.run_cli()[0], 1)

    def test_split_and_source_output_guards(self):
        for split in ('train', 'test'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(self.args() + ['--split', split])
        for directory in (self.data, self.data/'nested', self.paths['a'].parent,
                          self.paths['b'].parent/'nested', self.root):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(self.args()+['--output',str(directory)]), 1)

    def test_preservation_guard_detects_changed_label_or_state_or_image(self):
        audit = audit_validation(self.data, self.charsets)
        path = self.data/'labels.csv'
        original = path.read_bytes()
        path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(RuntimeError, 'labels changed'):
            verify_preservation(audit, {})
        path.write_bytes(original)
        (self.data/'labeling_state.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'review state changed'):
            verify_preservation(audit, {})
        (self.data/'labeling_state.json').unlink()
        (self.data/'validation/0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'PNG changed'):
            verify_preservation(audit, {})

    def test_audit_only_never_infers(self):
        with mock.patch('ml.compare_transfer.frozen_inference', side_effect=AssertionError):
            code, message = self.run_cli(['--audit-only'])
        self.assertEqual(code, 0, message)
        self.assertFalse((self.output/'predictions.csv').exists())

    def test_one_exclusion_is_shared_and_predictions_cannot_change_eligibility(self):
        self.rows[0]['label'] = 'all'
        self.write_csv()
        calls = []
        def infer(checkpoint, dataset, *args):
            calls.append([r['filename'] for r in dataset.records])
            # Predicted l is retained as an error on an eligible reference.
            return {'metrics': {'predictions': [dict(filename='1.png',expected='abcab',predicted='l')]}}
        with mock.patch('ml.compare_transfer.frozen_inference', side_effect=infer):
            code, message = self.run_cli()
        self.assertEqual(code, 0, message)
        self.assertEqual(calls, [['1.png'], ['1.png']])
        with (self.output/'predictions.csv').open(encoding='utf-8',newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['predicted_a'], 'l')
        self.assertEqual(rows[0]['correct_a'], 'false')
        audit = json.loads((self.output/'eligibility.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['eligible_count'], 1)
        self.assertEqual(audit['excluded_count'], 1)

    def test_paired_gallery_includes_every_row_and_four_result_colors(self):
        rows = []
        outcomes = [(True,True),(False,False),(False,True),(True,False)]
        for i in range(25):
            a, b = outcomes[i%4]
            rows.append(dict(filename=f'{i:02}.png', expected='ab', predicted_a='ab' if a else 'ac',
                             predicted_b='ab' if b else 'ac', correct_a=a, correct_b=b,
                             edit_distance_a=int(not a), edit_distance_b=int(not b)))
        source = self.data/'validation/0.png'
        before = source.read_bytes()
        paths = {r['filename']: source for r in rows}
        gallery = self.root/'gallery.png'
        selected = generate_gallery(rows, paths, gallery, paired=True)
        self.assertEqual(len(selected), 25)
        self.assertEqual(source.read_bytes(), before)
        with Image.open(gallery) as image:
            for i, color in enumerate(((21,128,61),(185,28,28),(37,99,235),(180,83,9))):
                self.assertEqual(image.getpixel((12+i*322,90)), color)


class PairedTests(unittest.TestCase):
    def test_all_concordance_cells_partial_improvements_and_weighted_cer(self):
        expected = ['ab', 'abcdef', 'ab', 'ab', 'abc', 'abc']
        a = ['ab', 'xxxxxx', 'xx', 'ab', 'ab', 'abc']
        b = ['ab', 'xxxxxx', 'ab', 'xx', 'abc', 'ab']
        predictions = lambda p: [dict(filename=str(i), expected=e, predicted=v)
                                  for i, (e, v) in enumerate(zip(expected, p))]
        rows, s = paired_comparison(predictions(a), predictions(b))
        self.assertEqual({k:len(v) for k,v in s['paired']['concordance'].items()},
                         dict(both_correct=1, both_incorrect=1, b_only_correct=2, a_only_correct=2))
        self.assertEqual({k:len(v) for k,v in s['paired']['edit_distance'].items()},
                         dict(b_lower=2, a_lower=2, tie=2))
        self.assertEqual(s['a']['global']['total_reference_characters'], 18)
        self.assertEqual(s['a']['global']['total_edit_distance'], 9)
        self.assertEqual(s['a']['global']['cer'], .5)
        self.assertEqual(s['a']['global']['substitutions'], 8)
        self.assertEqual(s['a']['global']['deletions'], 1)
        self.assertEqual(s['a']['global']['insertions'], 0)
        self.assertEqual(len(rows), 6)

    def test_paired_rejects_different_subsets_and_references(self):
        a = [dict(filename='a',expected='ab',predicted='ab')]
        with self.assertRaisesRegex(ValueError, 'subset'):
            paired_comparison(a, [dict(filename='b',expected='ab',predicted='ab')])
        with self.assertRaisesRegex(ValueError, 'references'):
            paired_comparison(a, [dict(filename='a',expected='ac',predicted='ab')])

    def test_partial_edits_and_insertions_count_without_new_exact_match(self):
        a = [dict(filename='a',expected='abc',predicted='xxx'),
             dict(filename='b',expected='ab',predicted='abb')]
        b = [dict(filename='a',expected='abc',predicted='abx'),
             dict(filename='b',expected='ab',predicted='abb')]
        _, s = paired_comparison(a, b)
        self.assertEqual(s['paired']['edit_distance']['b_lower'], ['a'])
        self.assertEqual(s['paired']['concordance']['both_incorrect'], ['a','b'])
        self.assertEqual(s['b']['global']['insertions'], 1)
        self.assertEqual(s['b']['global']['cer'], 2/5)


if __name__ == '__main__':
    unittest.main()
