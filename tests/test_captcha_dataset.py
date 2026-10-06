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
from urllib.error import HTTPError

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
    def test_429_waits_at_least_retry_after(self, sleep):
        error = http_error(429, "180")
        opener = Mock()
        opener.open.side_effect = [error, response()]
        self.assertEqual(dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)[3], b"page")
        sleep.assert_called_once_with(180.0)
        self.assertTrue(error.fp.closed)

    @patch("captcha_dataset.time.sleep")
    def test_persistent_429_has_bounded_exponential_waits(self, sleep):
        opener = Mock()
        opener.open.side_effect = [http_error(429) for _ in range(4)]
        with self.assertRaisesRegex(RuntimeError, "HTTP 429.*4 tentativa"):
            dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [60, 120, 240])
        self.assertEqual(opener.open.call_count, 4)

    @patch("captcha_dataset.time.sleep")
    def test_404_is_not_retried(self, sleep):
        opener = Mock()
        opener.open.side_effect = http_error(404)
        with self.assertRaises(HTTPError):
            dataset._fetch(opener, "url", timeout=30, retries=3, backoff=60)
        sleep.assert_not_called()
        self.assertEqual(opener.open.call_count, 1)

    def test_retry_after_dates_and_invalid_values(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=180)
        wait = dataset._retry_after_seconds(format_datetime(future, usegmt=True))
        self.assertGreater(wait, 175)
        self.assertLessEqual(wait, 180)
        past = datetime.now(timezone.utc) - timedelta(seconds=180)
        self.assertEqual(dataset._retry_after_seconds(format_datetime(past)), 0)
        for value in (None, "invalid", "-10", "1.5"):
            self.assertEqual(dataset._retry_after_seconds(value), 0)

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
            self.assertEqual(result, {"train": 17, "validation": 2, "test": 1})
            for name, count in result.items():
                self.assertEqual(len(list((Path(folder) / name).glob("*.png"))), count)
        waits = [call.args[0] for call in sleep.call_args_list if call.args[0]]
        self.assertEqual(waits, [90, 60])

    @patch("captcha_dataset.time.sleep")
    @patch("captcha_dataset.build_opener")
    def test_failed_retries_preserve_saved_image(self, build, sleep):
        build.return_value.open.side_effect = [
            response(b'<img id="captcha_image" src="../captcha.php">'),
            response(png(1), "image/png"),
            http_error(429), http_error(429),
        ]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                dataset.collect_captchas(20, folder, delay=0, retries=1)
            images = list(Path(folder).glob("*/*.png"))
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0].read_bytes(), png(1))


if __name__ == "__main__":
    unittest.main()
