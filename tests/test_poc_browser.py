"""Real Chromium on loopback fixtures; recognizer simulated, never Correios."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from poc.correios import execute
from poc.results import Artifacts
from tests.poc_support import Fixture, FIXTURE_CODE, FIXED_ANSWER, SimulatedRecognizer


class BrowserTests(unittest.TestCase):
    def run_case(self, scenario='success', *, prediction=FIXED_ANSWER, max_attempts=1):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with Fixture(scenario) as fixture:
            artifacts = Artifacts(temp.name, mode='local_fixture', code=FIXTURE_CODE, max_attempts=max_attempts)
            recognizer = SimulatedRecognizer(prediction)
            result = asyncio.run(execute(recognizer, FIXTURE_CODE, artifacts, interface=fixture.interface,
                max_attempts=max_attempts, timeout_ms=1200, retry_delay=.01,
                context_setup=fixture.setup_context))
            self.assertTrue(result['browser_closed'])
            self.assertFalse(fixture.browser.is_connected())
            self.assertEqual(fixture.context.pages, [])
            self.assertTrue((artifacts.directory/'result.json').is_file())
            self.assertTrue((artifacts.directory/'attempts.csv').is_file())
            self.assertEqual(result['mode'], 'local_fixture')
            saved = json.loads((artifacts.directory/'result.json').read_text())
            self.assertEqual(saved['status'], result['status'])
            for path in artifacts.directory.iterdir():
                if path.suffix != '.png':
                    raw = path.read_text()
                    for secret in (FIXTURE_CODE, 'PRIVATE ADDRESS', 'PRIVATE TOKEN', 'PRIVATE PERSON',
                                   'private@example.invalid', 'fixture_session', '?objeto='):
                        self.assertNotIn(secret, raw)
            return result, fixture, recognizer

    def test_capture_original_response_no_extra_get_exact_prediction_session_and_success(self):
        r, f, rec = self.run_case()
        self.assertEqual(r['status'], 'tracking_success', r)
        self.assertEqual(r['submissions'], 1)
        self.assertEqual(r['queries_started'], 1)
        self.assertEqual(r['queries_completed'], 1)
        self.assertEqual(f.submissions[0]['query']['captcha'], [FIXED_ANSWER])
        self.assertEqual(f.submissions[0]['query']['objeto'], [FIXTURE_CODE])
        self.assertEqual(f.submissions[0]['image_gets'], 1)
        self.assertEqual(f.submissions[0]['session_images'], 1)
        self.assertEqual(len(rec.images), 1)
        self.assertEqual(r['attempts'][0]['capture_source'], 'browser_response_body')
        self.assertEqual(r['attempts'][0]['submitted_prediction'], FIXED_ANSWER)
        self.assertEqual(f.external_blocked, [])

    def test_http_200_errors_other_objects_old_results_and_empty_events_never_succeed(self):
        for scenario in ('error_200', 'other_object', 'stale', 'no_events'):
            with self.subTest(scenario=scenario):
                r, f, _ = self.run_case(scenario, max_attempts=3)
                self.assertEqual(r['status'], 'site_changed_or_unrecognized')
                self.assertFalse(r['success'])
                self.assertEqual(len(f.submissions), 1)

    def test_rejection_is_explicit_and_not_retried_by_default(self):
        r, f, _ = self.run_case('rejected')
        self.assertEqual(r['status'], 'captcha_rejected')
        self.assertEqual(r['attempt_count'], 1)
        self.assertEqual(len(f.submissions), 1)

    def test_not_found_is_separate_and_does_not_claim_captcha_acceptance(self):
        r, _, _ = self.run_case('not_found')
        self.assertEqual(r['status'], 'object_not_found')
        self.assertIsNone(r['evidence']['captcha_accepted'])
        self.assertFalse(r['success'])

    def test_image_changed_before_submission_cancels(self):
        for scenario in ('changed', 'changed_at_click'):
            with self.subTest(scenario=scenario):
                r, f, _ = self.run_case(scenario)
                self.assertEqual(r['status'], 'challenge_expired_or_changed', r)
                self.assertEqual(r['submissions'], 0)
                self.assertEqual(len(f.submissions), 0)

    def test_image_from_one_session_cannot_succeed_in_another(self):
        r, f, _ = self.run_case('wrong_session')
        self.assertEqual(r['status'], 'site_changed_or_unrecognized')
        self.assertFalse(r['success'])
        self.assertEqual(f.submissions[0]['session_images'], 0)
        self.assertNotEqual(r['status'], 'captcha_rejected')

    def test_429_and_access_denial_stop_even_with_three_attempt_budget(self):
        for scenario, expected, submissions in (
            ('rate','rate_limited',1), ('blocked','access_blocked',1),
            ('initial_rate','rate_limited',0), ('initial_blocked','access_blocked',0)):
            with self.subTest(scenario=scenario):
                r, f, _ = self.run_case(scenario, max_attempts=3)
                self.assertEqual(r['status'], expected, r)
                self.assertEqual(len(f.submissions), submissions)
                self.assertEqual(r['submissions'], submissions)
                if expected == 'rate_limited': self.assertGreater(r['evidence']['retry_after_seconds'], 0)

    def test_timeout_after_submission_has_unknown_state_and_no_resend(self):
        r, f, _ = self.run_case('timeout', max_attempts=3)
        self.assertEqual(r['status'], 'network_error', r)
        self.assertEqual(r['evidence']['submission_state'], 'unknown')
        self.assertEqual(len(f.submissions), 1)

    def test_missing_controls_absent_captcha_and_wrong_dimensions(self):
        for scenario, expected in (('missing_selector','site_changed_or_unrecognized'),
                                   ('no_captcha','challenge_not_present'),
                                   ('wrong_dimensions','inference_error')):
            with self.subTest(scenario=scenario):
                r, f, rec = self.run_case(scenario)
                self.assertEqual(r['status'], expected, r)
                self.assertEqual(len(f.submissions), 0)
                self.assertFalse(rec.images)

    def test_empty_prediction_has_no_submission(self):
        r, f, _ = self.run_case(prediction='')
        self.assertEqual(r['status'], 'inference_error')
        self.assertEqual(len(f.submissions), 0)

    def test_opt_in_retry_after_explicit_rejection_accounts_for_failure(self):
        r, f, _ = self.run_case('retry_success', max_attempts=2)
        self.assertTrue(r['success'], r)
        self.assertEqual([a['status'] for a in r['attempts']], ['captcha_rejected','tracking_success'])
        self.assertEqual(r['attempt_count'], 2)
        self.assertEqual(r['submissions'], 2)
        self.assertNotEqual(r['attempts'][0]['image_sha256'], r['attempts'][1]['image_sha256'])
        self.assertEqual(len({s['sid'] for s in f.submissions}), 1)

    def test_attempt_limit_is_obeyed(self):
        r, f, _ = self.run_case('rejected', max_attempts=3)
        self.assertEqual(r['attempt_count'], 3)
        self.assertEqual(len(f.submissions), 3)
        self.assertTrue(all(a['status'] == 'captcha_rejected' for a in r['attempts']))

    def test_external_network_is_aborted_by_test_infrastructure(self):
        r, f, _ = self.run_case('external')
        self.assertTrue(r['success'], r)
        self.assertEqual(f.external_blocked, ['external.invalid'])

    def test_explicit_inspection_does_not_fill_or_submit(self):
        with tempfile.TemporaryDirectory() as root, Fixture() as fixture:
            artifacts = Artifacts(root, mode='local_fixture_inspection')
            r = asyncio.run(execute(None, None, artifacts, inspect=True, interface=fixture.interface,
                                   context_setup=fixture.setup_context, timeout_ms=1200))
            self.assertFalse(r['success'])
            self.assertTrue(r['inspection']['selectors_present'])
            self.assertFalse(r['inspection']['result_contract_verified'])
            self.assertEqual(r['queries_started'], 0)
            self.assertEqual(len(fixture.submissions), 0)
            self.assertEqual(fixture.image_gets, 1)
            self.assertFalse(fixture.browser.is_connected())

    def test_inference_exception_still_closes_browser_and_writes_results(self):
        with tempfile.TemporaryDirectory() as root, Fixture() as fixture:
            artifacts = Artifacts(root, mode='local_fixture', code=FIXTURE_CODE)
            rec = SimulatedRecognizer()
            with mock.patch.object(rec, 'recognize', side_effect=RuntimeError('PRIVATE TOKEN')):
                r = asyncio.run(execute(rec, FIXTURE_CODE, artifacts, interface=fixture.interface,
                    context_setup=fixture.setup_context, timeout_ms=1200))
            self.assertEqual(r['status'], 'inference_error')
            self.assertEqual(r['submissions'], 0)
            self.assertEqual(r['attempts'][0]['status'], 'inference_error')
            self.assertTrue(r['browser_closed'])
            self.assertFalse(fixture.browser.is_connected())
            self.assertTrue((artifacts.directory/'attempts.csv').exists())
            self.assertNotIn('PRIVATE TOKEN', (artifacts.directory/'result.json').read_text())


if __name__ == '__main__':
    unittest.main()
