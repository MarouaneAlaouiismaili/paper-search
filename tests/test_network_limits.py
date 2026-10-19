import unittest
from unittest.mock import Mock, patch

from paper_search.network import DNSResolutionError, http_get, http_post_json


class FakeResponse:
    status_code = 200
    headers = {"Content-Length": "20"}

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self.chunks

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def get(self, url, **kwargs):
        self.kwargs = kwargs
        return self.response

    def post(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        return self.response


class NetworkLimitTests(unittest.TestCase):
    def test_dns_failure_is_reported_without_repeating_the_same_request(self):
        from curl_cffi.requests.exceptions import DNSError

        session = FakeSession(None)
        session.get = Mock(side_effect=DNSError("Could not resolve host"))
        with patch("paper_search.network._get_session", return_value=session):
            with self.assertRaisesRegex(DNSResolutionError, "DNS lookup failed for example.org"):
                http_get("https://example.org/page", retries=4)
        session.get.assert_called_once()

    def test_streamed_response_cannot_exceed_byte_limit(self):
        response = FakeResponse([b"123456", b"789012"])
        response.headers = {"Content-Length": "10"}
        session = FakeSession(response)
        with patch("paper_search.network._get_session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "10-byte limit"):
                http_get("https://example.org/page", timeout=8, max_bytes=10, allow_redirects=False, retries=1)
        self.assertFalse(session.kwargs["allow_redirects"])
        self.assertTrue(session.kwargs["stream"])
        self.assertTrue(response.closed)

    def test_declared_oversize_is_rejected_before_reading(self):
        response = FakeResponse([])
        response.headers = {"Content-Length": "2048"}
        session = FakeSession(response)
        with patch("paper_search.network._get_session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "10-byte limit"):
                http_get("https://example.org/page", timeout=8, max_bytes=10, allow_redirects=False, retries=1)
        self.assertTrue(response.closed)

    def test_json_post_is_https_only_bounded_and_does_not_follow_redirects(self):
        response = FakeResponse([b'{"ok":true}'])
        response.headers = {"Content-Length": "11"}
        session = FakeSession(response)
        with patch("paper_search.network._get_session", return_value=session):
            status, body = http_post_json(
                "https://vision.googleapis.com/v1/images:annotate",
                {"requests": []},
                headers={"X-Goog-Api-Key": "test"},
            )
        self.assertEqual((status, body), (200, b'{"ok":true}'))
        self.assertFalse(session.kwargs["allow_redirects"])
        self.assertTrue(session.kwargs["stream"])
        self.assertEqual(session.kwargs["headers"]["X-Goog-Api-Key"], "test")
        self.assertTrue(response.closed)

        with self.assertRaisesRegex(RuntimeError, "Only HTTPS"):
            http_post_json("http://vision.googleapis.com/v1/images:annotate", {})

    def test_json_post_rejects_oversize_response_before_returning_it(self):
        response = FakeResponse([b"123456"])
        response.headers = {"Content-Length": "6"}
        session = FakeSession(response)
        with patch("paper_search.network._get_session", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "5-byte limit"):
                http_post_json("https://vision.googleapis.com/v1/images:annotate", {},
                               max_response_bytes=5)
        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
