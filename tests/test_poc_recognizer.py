"""Unlabelled inference, equivalence to the existing evaluator, immutable BN."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

from ml.charset import Charset
from ml.dataset import read_image_tensor
from ml.decode import greedy_decode
from ml.evaluate import frozen_inference
from ml.model import CRNN
from poc.recognizer import Recognizer


class RecognizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'best.pt'
        charset = Charset('abl2')
        torch.set_num_threads(2)
        model = CRNN(charset.num_classes, hidden_size=8, num_layers=1)
        # A known, nonempty greedy path containing l; independent of image/label.
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(-10)
            model.classifier.bias[charset.char_to_index['l']] = 10
        self.checkpoint = dict(format_version=1, epoch=46, model_config=model.config,
            model_state_dict=model.state_dict(), charset=charset.to_dict(), optimizer_state_dict={},
            config={}, rng_state={}, best_metrics={}, history=[], dataset_identity={})
        torch.save(self.checkpoint, self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        buffer = io.BytesIO()
        Image.new('RGB', (215,80), (80,100,240)).save(buffer, format='PNG')
        self.png = buffer.getvalue()
        self.patcher = mock.patch('poc.recognizer.D_SHA256', self.digest)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_inference_reuses_original_preprocessing_decoder_without_labels(self):
        rec = Recognizer(self.path, threads=2)
        original = self.path.read_bytes()
        before = {k:v.clone() for k,v in rec.model.state_dict().items()}
        observed = []
        def inspect(model, inputs):
            observed.append((torch.is_inference_mode_enabled(), inputs[0].clone()))
        hook = rec.model.register_forward_pre_hook(inspect)
        self.addCleanup(hook.remove)
        result = rec.recognize(self.png)
        self.assertEqual(result['text'], 'l')
        self.assertEqual(result['checkpoint_epoch'], 46)
        self.assertEqual(result['checkpoint_sha256'], self.digest)
        self.assertEqual(result['image_sha256'], hashlib.sha256(self.png).hexdigest())
        self.assertTrue(observed[0][0])
        tensor = read_image_tensor(io.BytesIO(self.png)).unsqueeze(0)
        self.assertTrue(torch.equal(observed[0][1], tensor))
        self.assertEqual(tuple(tensor.shape), (1,1,80,215))
        self.assertEqual(tensor.dtype, torch.float32)
        with torch.inference_mode():
            expected = greedy_decode(rec.model(tensor), rec.charset)[0]
        self.assertEqual(result['text'], expected)
        self.assertFalse(rec.model.training)
        self.assertTrue(all(not p.requires_grad for p in rec.model.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in rec.model.state_dict().items()))
        rec.verify_preservation()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertGreater(result['inference_seconds'], 0)

    def test_same_prediction_as_previous_frozen_evaluation_path(self):
        # Labels exist ONLY in this equivalence fixture for the existing evaluator.
        # The new recognizer receives PNG bytes, never this dataset or its label.
        rec = Recognizer(self.path, threads=2)
        charset = rec.charset
        png = self.png
        class LabelledEquivalenceFixture:
            def __init__(self): self.charset = charset
            def __len__(self): return 1
            def __getitem__(self, i):
                return {'image':read_image_tensor(io.BytesIO(png)), 'target':torch.tensor(charset.encode('l')),
                        'label':'l', 'filename':'fixture.png'}
        old = frozen_inference(self.checkpoint, LabelledEquivalenceFixture(), batch_size=1)
        self.assertEqual(old['metrics']['predictions'][0]['predicted'], rec.recognize(png)['text'])

    def test_missing_hash_epoch_and_incompatible_model_fail_before_inference(self):
        with self.assertRaises(FileNotFoundError): Recognizer(self.path/'missing')
        with mock.patch('poc.recognizer.D_SHA256', '0'*64), self.assertRaisesRegex(ValueError, 'SHA-256'):
            Recognizer(self.path)
        for change, message in (({'epoch':45}, 'Época'), ({'model_config':{'num_classes':2}}, 'Classes')):
            with self.subTest(change=change), mock.patch('poc.recognizer.load_checkpoint',
                return_value={**self.checkpoint, **change}), self.assertRaisesRegex(ValueError, message):
                Recognizer(self.path)

    def test_invalid_image_empty_prediction_and_invalid_logits_fail(self):
        rec = Recognizer(self.path, threads=2)
        bad = io.BytesIO()
        Image.new('L', (214,80)).save(bad, format='PNG')
        for png in (b'not png', bad.getvalue()):
            with self.assertRaises(ValueError): rec.recognize(png)
        with mock.patch.object(rec.model, 'forward', return_value=torch.zeros(53,1,5)):
            with self.assertRaisesRegex(ValueError, 'vazia'): rec.recognize(self.png)
        with mock.patch.object(rec.model, 'forward', return_value=torch.full((53,1,5), float('nan'))):
            with self.assertRaisesRegex(RuntimeError, 'NaN/Inf'): rec.recognize(self.png)

    def test_preservation_detects_mutated_batchnorm_and_checkpoint(self):
        rec = Recognizer(self.path, threads=2)
        with torch.no_grad(): rec.model.cnn[1].running_mean.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'BatchNorm'): rec.verify_preservation()
        rec = Recognizer(self.path, threads=2)
        self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint'): rec.verify_preservation()


if __name__ == '__main__': unittest.main()
