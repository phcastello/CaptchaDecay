import io
import struct
import tempfile
import unittest
import zlib
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import captcha_dataset as dataset


def response(data=b"page", content_type="text/html"):
    headers = Message()
    headers["Content-Type"] = content_type
    result = io.BytesIO(data)
    result.headers = headers
    result.geturl = lambda: "https://example.test/app/index.php"
    return result


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://example.test", code, "fixture", headers, io.BytesIO())


def png(number):
    def chunk(kind, data):
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data)))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack("!2I5B", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, number, 50, 70])))
            + chunk(b"IEND", b""))


class RetryTests(unittest.TestCase):
    @patch("captcha_dataset.time.sleep")
    def test_429_waits_retry_after_plus_margin_even_without_retries(self, sleep):
        error = http_error(429, "180")
        opener = Mock()
        opener.open.side_effect = [error, response()]
        self.assertEqual(dataset._fetch(opener, "url", timeout=30, retries=0, backoff=60)[3], b"page")
        sleep.assert_called_once_with(182.0)
        self.assertEqual(opener.open.call_count, 2)
        self.assertTrue(error.fp.closed)

    @patch("captcha_dataset.time.sleep")
    def test_429_without_valid_retry_after_aborts_immediately(self, sleep):
        for value in (None, "invalid", "-10", "1.5", ""):
            with self.subTest(retry_after=value):
                error = http_error(429, value)
                opener = Mock()
                opener.open.side_effect = error
                with self.assertRaisesRegex(dataset.RateLimitError, "HTTP 429.*Retry-After válido"):
                    dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)
                sleep.assert_not_called()
                self.assertEqual(opener.open.call_count, 1)
                self.assertTrue(error.fp.closed)

    @patch("captcha_dataset.time.sleep")
    def test_second_429_aborts_regardless_of_retry_after(self, sleep):
        for value in (None, "120", "invalid"):
            with self.subTest(retry_after=value):
                sleep.reset_mock()
                errors = [http_error(429, "90"), http_error(429, value)]
                opener = Mock()
                opener.open.side_effect = errors
                with self.assertRaisesRegex(dataset.RateLimitError, "HTTP 429 novamente.*interrompida"):
                    dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)
                sleep.assert_called_once_with(92.0)
                self.assertEqual(opener.open.call_count, 2)
                self.assertTrue(all(error.fp.closed for error in errors))

    @patch("captcha_dataset.time.sleep")
    def test_429_with_http_date(self, sleep):
        now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
        deadline = format_datetime(now + timedelta(seconds=120), usegmt=True)
        opener = Mock()
        opener.open.side_effect = [http_error(429, deadline), response()]
        with patch("captcha_dataset.datetime") as clock:
            clock.now.return_value = now
            dataset._fetch(opener, "url", timeout=30, retries=0, backoff=60)
        sleep.assert_called_once_with(122.0)

    @patch("captcha_dataset.time.sleep")
    def test_429_with_zero_retry_after_is_valid(self, sleep):
        opener = Mock()
        opener.open.side_effect = [http_error(429, "0"), response()]
        dataset._fetch(opener, "url", timeout=30, retries=0, backoff=60)
        sleep.assert_called_once_with(2.0)

    @patch("captcha_dataset.time.sleep")
    def test_temporary_http_errors_retry_then_succeed(self, sleep):
        for code in (408, 500, 502, 503, 504):
            with self.subTest(code=code):
                sleep.reset_mock()
                error = http_error(code)
                opener = Mock()
                opener.open.side_effect = [error, response()]
                self.assertEqual(dataset._fetch(opener, "url", timeout=30, retries=1, backoff=60)[3], b"page")
                sleep.assert_called_once_with(60)
                self.assertEqual(opener.open.call_count, 2)
                self.assertTrue(error.fp.closed)

    @patch("captcha_dataset.time.sleep")
    def test_temporary_errors_have_bounded_exponential_backoff(self, sleep):
        opener = Mock()
        opener.open.side_effect = [http_error(500) for _ in range(4)]
        with self.assertRaisesRegex(RuntimeError, "HTTP 500.*4 tentativa"):
            dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 120, 240])
        self.assertEqual(opener.open.call_count, 4)

    @patch("captcha_dataset.time.sleep")
    def test_connection_error_retries(self, sleep):
        opener = Mock()
        opener.open.side_effect = [URLError("temporary connection failure"), response()]
        dataset._fetch(opener, "url", timeout=30, retries=1, backoff=60)
        sleep.assert_called_once_with(60)

    @patch("captcha_dataset.time.sleep")
    def test_temporary_errors_respect_retry_after(self, sleep):
        now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
        deadline = format_datetime(now + timedelta(seconds=180), usegmt=True)
        for value, expected in (("180", 180), (deadline, 180), ("10", 60), ("invalid", 60)):
            with self.subTest(retry_after=value), patch("captcha_dataset.datetime") as clock:
                sleep.reset_mock()
                clock.now.return_value = now
                opener = Mock()
                opener.open.side_effect = [http_error(503, value), response()]
                dataset._fetch(opener, "url", timeout=30, retries=1, backoff=60)
                sleep.assert_called_once_with(expected)

    @patch("captcha_dataset.time.sleep")
    def test_429_does_not_consume_temporary_error_retries(self, sleep):
        opener = Mock()
        opener.open.side_effect = [http_error(500), http_error(429, "90"), http_error(503), response()]
        dataset._fetch(opener, "url", timeout=30, retries=2, backoff=60)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 92, 120])
        self.assertEqual(opener.open.call_count, 4)

    @patch("captcha_dataset.time.sleep")
    def test_permanent_http_error_is_propagated(self, sleep):
        for code in (400, 401, 403, 404):
            with self.subTest(code=code):
                opener = Mock()
                error = http_error(code)
                opener.open.side_effect = error
                with self.assertRaises(HTTPError) as raised:
                    dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)
                self.assertIs(raised.exception, error)
                sleep.assert_not_called()
                self.assertEqual(opener.open.call_count, 1)

    def test_retry_after_dates_and_invalid_values(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=180)
        wait = dataset._retry_after_seconds(format_datetime(future, usegmt=True))
        self.assertGreater(wait, 175)
        self.assertLessEqual(wait, 180)
        past = datetime.now(timezone.utc) - timedelta(seconds=180)
        self.assertEqual(dataset._retry_after_seconds(format_datetime(past)), 0)
        self.assertEqual(dataset._retry_after_seconds(" 120 "), 120)
        self.assertEqual(dataset._retry_after_seconds("0"), 0)
        for value in (None, "", " ", "invalid", "-10", "1.5", "inf", "9" * 400):
            with self.subTest(value=value):
                self.assertIsNone(dataset._retry_after_seconds(value))


class CollectionTests(unittest.TestCase):
    @patch("captcha_dataset.random.random", return_value=0.842381)
    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_valid_png_replaces_query_and_uses_identifiable_user_agent(self, build, sleep, random):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="../captcha.php?token=abc&amp;empty=#old">'),
            response(png(1), "image/png"),
        ]
        with tempfile.TemporaryDirectory() as folder, self.assertLogs(dataset.LOGGER, level="INFO") as logs:
            self.assertEqual(dataset.collect_captchas(1, folder), {"train": 1, "validation": 0, "test": 0})
            images = list(Path(folder).glob("*/*.png"))
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].read_bytes(), png(1))
        request = build.return_value.open.call_args_list[1].args[0]
        self.assertEqual(request.full_url, "https://example.test/captcha.php?0.842381")
        self.assertEqual(request.get_header("Referer"), "https://example.test/app/index.php")
        self.assertFalse(request.has_header("Cache-control"))
        self.assertEqual(build.return_value.addheaders, [("User-Agent", "CaptchaDecay-Academic-PoC/1.0")])
        random.assert_called_once_with()
        self.assertIn("Página carregada. CAPTCHA encontrado em /captcha.php.", "\n".join(logs.output))
        self.assertNotIn("token=abc", "\n".join(logs.output))
        sleep.assert_not_called()

    @patch("captcha_dataset.random.random", side_effect=[0.35976709834294707, 0.8273619234812, 0.12415191283])
    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_refreshes_reuse_one_session_and_load_page_only_once(self, build, sleep, random):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="../core/securimage/securimage_show.php?0.123456">'),
            *[response(png(number), "image/png") for number in range(3)],
        ]
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(sum(dataset.collect_captchas(3, folder).values()), 3)
        build.assert_called_once()
        processor = build.call_args.args[0]
        self.assertIsInstance(processor, dataset.HTTPCookieProcessor)
        self.assertIsInstance(processor.cookiejar, dataset.http.cookiejar.CookieJar)
        calls = build.return_value.open.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0].args[0], dataset.URL)
        self.assertEqual([call.args[0].full_url for call in calls[1:]], [
            "https://example.test/core/securimage/securimage_show.php?0.35976709834294707",
            "https://example.test/core/securimage/securimage_show.php?0.8273619234812",
            "https://example.test/core/securimage/securimage_show.php?0.12415191283",
        ])
        self.assertEqual(random.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 5])

    @patch("captcha_dataset.random.random", return_value=0.842381)
    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_image_retry_reuses_refresh_request_without_reloading_page(self, build, sleep, random):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="captcha.php">'),
            http_error(500), response(png(1), "image/png"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(sum(dataset.collect_captchas(1, folder).values()), 1)
        calls = build.return_value.open.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertIs(calls[1].args[0], calls[2].args[0])
        random.assert_called_once_with()
        build.assert_called_once()
        sleep.assert_called_once_with(60)

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_non_png_responses_are_rejected_without_saving(self, build, sleep):
        for data, content_type in ((b"<html>error</html>", "text/html"),
                                   (b"<html>error</html>", "image/png"),
                                   (png(1), "text/html"), (b"", "image/png")):
            with self.subTest(content_type=content_type, data=data), tempfile.TemporaryDirectory() as folder:
                build.return_value.open.side_effect = [
                    response(b'<img id="captcha_image" src="captcha.php">'),
                    response(data, content_type),
                ]
                with self.assertRaisesRegex(RuntimeError, "Resposta inesperada.*esperado PNG"):
                    dataset.collect_captchas(1, folder)
                self.assertEqual(list(Path(folder).glob("*/*.png")), [])
        sleep.assert_not_called()

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_collection_recovers_on_page_and_image_preserving_split(self, build, sleep):
        opener = build.return_value
        opener.open.side_effect = [
            http_error(429, "90"),
            response(b'<img id="captcha_image" src="../captcha.php">'),
            TimeoutError("timed out"),
            *[response(png(number), "image/png") for number in range(20)],
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = dataset.collect_captchas(20, folder, delay=0)
            self.assertEqual(result, {"train": 12, "validation": 4, "test": 4})
            for name, count in result.items():
                self.assertEqual(len(list((Path(folder) / name).glob("*.png"))), count)
        waits = [call.args[0] for call in sleep.call_args_list if call.args[0]]
        self.assertEqual(waits, [92, 60])

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_rate_limit_preserves_saved_image(self, build, sleep):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="../captcha.php">'),
            response(png(1), "image/png"),
            http_error(429),
        ]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(dataset.RateLimitError, "HTTP 429"):
                dataset.collect_captchas(20, folder, delay=0, retries=1)
            images = list(Path(folder).glob("*/*.png"))
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].read_bytes(), png(1))
        self.assertEqual(build.return_value.open.call_count, 3)

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_duplicates_are_ignored_with_delay_between_requests(self, build, sleep):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="captcha.php">'),
            response(png(1), "image/png"),
            response(png(1), "image/png"),
            response(png(2), "image/png"),
        ]
        with tempfile.TemporaryDirectory() as folder, self.assertLogs(dataset.LOGGER, level="INFO") as logs:
            self.assertEqual(sum(dataset.collect_captchas(2, folder, delay=5).values()), 2)
            self.assertEqual(len(list(Path(folder).glob("*/*.png"))), 2)
        self.assertIn("Imagem repetida ignorada.", "\n".join(logs.output))
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 5])

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_excessive_duplicates_abort_after_normal_refreshes(self, build, sleep):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="captcha.php">'),
            *[response(png(1), "image/png") for _ in range(6)],
        ]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "não foi possível obter imagens novas"):
                dataset.collect_captchas(2, folder, delay=0)
            self.assertEqual(len(list(Path(folder).glob("*/*.png"))), 1)
        requests = [call.args[0] for call in build.return_value.open.call_args_list[1:]]
        self.assertEqual(len(requests), 6)
        for request in requests:
            parts = dataset.urlsplit(request.full_url)
            self.assertEqual(parts.path, "/app/captcha.php")
            self.assertGreaterEqual(float(parts.query), 0)
            self.assertLess(float(parts.query), 1)
            self.assertNotIn("=", parts.query)
            self.assertFalse(request.has_header("Cache-control"))

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_existing_files_are_preserved_on_rate_limit_and_resumption(self, build, sleep):
        with tempfile.TemporaryDirectory() as folder:
            existing = Path(folder) / "validation" / "original.png"
            existing.parent.mkdir()
            existing.write_bytes(png(99))
            build.return_value.open.side_effect = [
                response(b'<img id="captcha_image" src="captcha.php">'),
                response(png(1), "image/png"), http_error(429),
            ]
            with self.assertRaises(dataset.RateLimitError):
                dataset.collect_captchas(2, folder, delay=0)
            before = {path: path.read_bytes() for path in Path(folder).glob("*/*.png")}
            self.assertEqual(len(before), 2)
            build.return_value.open.side_effect = [
                response(b'<img id="captcha_image" src="captcha.php">'),
                *[response(png(number), "image/png") for number in (99, 1, 2, 3)],
            ]
            self.assertEqual(sum(dataset.collect_captchas(2, folder, delay=0).values()), 2)
            self.assertEqual(len(list(Path(folder).glob("*/*.png"))), 4)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)


class CLITests(unittest.TestCase):
    @patch("sys.argv", ["captcha_dataset.py", "--count", "500"])
    @patch("captcha_dataset.collect_captchas")
    def test_rate_limit_exits_cleanly_with_informative_message(self, collect):
        collect.side_effect = dataset.RateLimitError("HTTP 429. Coleta interrompida; imagens preservadas.")
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as raised:
                dataset.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("HTTP 429. Coleta interrompida; imagens preservadas.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(collect.call_args.args[0], 500)


if __name__ == "__main__":
    unittest.main()
