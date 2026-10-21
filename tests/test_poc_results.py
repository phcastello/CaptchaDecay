import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from poc.correios import execute
from poc.results import Artifacts, classify_response, mask_code, outcome, validate_code
from poc.run import main, parse_args
from tests.poc_support import FIXTURE_CODE


class ResultTests(unittest.TestCase):
    def test_no_live_flag_never_instantiates_browser_or_model(self):
        with (mock.patch('poc.correios.async_playwright', side_effect=AssertionError('network forbidden')),
             mock.patch('poc.run.Recognizer', side_effect=AssertionError('model forbidden')),
             redirect_stdout(io.StringIO())):
            self.assertEqual(main([]), 0)
            self.assertEqual(main(['--inspect']), 0)

    def test_one_code_only_and_bounded_arguments(self):
        self.assertEqual(validate_code(FIXTURE_CODE), FIXTURE_CODE)
        self.assertEqual(mask_code(FIXTURE_CODE), 'ZZ*********ZZ')
        for code in ('', 'AA000000001ZZ,AA000000002ZZ', 'AA000000001ZZ AA000000002ZZ',
                     ['AA000000001ZZ'], 'https://example.invalid', '12345678901'):
            with self.assertRaises(ValueError): validate_code(code)
        for args in (['--max-attempts','4'], ['--timeout-ms','0'], ['--url','http://external.invalid'],
                     ['--live','--smoke-image','some.png']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): parse_args(args)

    def test_current_request_and_events_required_no_http_200_fallback(self):
        payload = {'codObjeto':FIXTURE_CODE, 'eventos':[
            {'descricao':'event', 'dtHrCriado':{'date':'2026-10-01 01:00:00'}}]}
        for correlated, data, expected in ((True,payload,'tracking_success'),
            (False,payload,'site_changed_or_unrecognized'), (True,{},'site_changed_or_unrecognized'),
            (True,{'erro':True,'mensagem':'session invalid'},'site_changed_or_unrecognized')):
            self.assertEqual(classify_response(200, data, correlated=correlated,
                                               tracking_code=FIXTURE_CODE)['status'], expected)

    def test_explicit_not_found_semantics_without_assuming_server_literal(self):
        # Fabricated error strings test semantics, not observed Correios messages.
        for message in ('Objeto não encontrado na fixture.', 'Objeto não foi localizado.'):
            r = classify_response(200, {'erro':True, 'mensagem':message}, correlated=True,
                                  tracking_code=FIXTURE_CODE)
            self.assertEqual(r['status'], 'object_not_found')
            self.assertIsNone(r['evidence']['captcha_accepted'])
        r = classify_response(200, {'erro':True, 'mensagem':'Não foi possível consultar o objeto.'},
                              correlated=True, tracking_code=FIXTURE_CODE)
        self.assertEqual(r['status'], 'site_changed_or_unrecognized')

    def test_unique_directories_and_no_sensitive_screenshot(self):
        with tempfile.TemporaryDirectory() as root:
            a = Artifacts(root, mode='local_fixture', code=FIXTURE_CODE)
            b = Artifacts(root, mode='local_fixture', code=FIXTURE_CODE)
            self.assertNotEqual(a.directory, b.directory)
            r = a.finish(outcome('network_error','test'), .1)
            raw = (a.directory/'result.json').read_text()
            self.assertNotIn(FIXTURE_CODE, raw)
            self.assertFalse(r['screenshot']['saved'])
            self.assertEqual(a.directory.stat().st_mode & 0o777, 0o700)

    def test_invalid_attempt_budget_rejected_without_browser(self):
        with tempfile.TemporaryDirectory() as root:
            artifacts = Artifacts(root, mode='local_fixture')
            for budget in (0,4,True):
                with (mock.patch('poc.correios.async_playwright', side_effect=AssertionError('network')),
                     self.assertRaises(ValueError)):
                    asyncio.run(execute(None, FIXTURE_CODE, artifacts, max_attempts=budget))

    def test_missing_checkpoint_or_code_fails_before_browser(self):
        with (tempfile.TemporaryDirectory() as root, mock.patch('poc.run.Artifacts') as a,
             mock.patch('poc.correios.async_playwright', side_effect=AssertionError('network')),
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            self.assertEqual(main(['--live','--checkpoint',str(Path(root)/'missing.pt'),
                                   '--tracking-code',FIXTURE_CODE]),1)
        with (mock.patch.dict('os.environ',{},clear=True), mock.patch('sys.stdin.isatty',return_value=False),
             redirect_stderr(io.StringIO())):
            self.assertEqual(main(['--live']),1)


if __name__ == '__main__': unittest.main()
