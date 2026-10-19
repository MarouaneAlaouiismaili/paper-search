import base64
import csv
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from paper_search import cli
from paper_search.sources import google_vision
from paper_search.sources.google_vision import ImageSearchError, MAX_INLINE_IMAGE_BYTES, search_image


class GoogleVisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture_image = b"\x89PNG\r\n\x1a\nfixture image bytes"
        self.api_response = {
            "responses": [{
                "webDetection": {
                    "fullMatchingImages": [{"url": "https://images.example/full.jpg"}],
                    "partialMatchingImages": [{"url": "https://images.example/crop.jpg", "score": 0.8}],
                    "pagesWithMatchingImages": [{
                        "url": "https://example.org/page",
                        "pageTitle": "Image source page",
                        "partialMatchingImages": [{"url": "https://images.example/crop.jpg"}],
                    }],
                    "visuallySimilarImages": [{"url": "https://images.example/similar.jpg"}],
                    "bestGuessLabels": [{"label": "A sample image"}],
                    "webEntities": [{"entityId": "/m/example", "score": 0.9, "description": "Example"}],
                },
            }],
        }

    def _temp_image(self, content=None):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "sample.png"
        path.write_bytes(self.fixture_image if content is None else content)
        return str(path)

    @patch("paper_search.sources.google_vision.http_post_json")
    def test_uploads_local_image_to_web_detection_and_maps_results(self, post):
        post.return_value = (200, json.dumps(self.api_response).encode())
        path = self._temp_image()

        report = search_image(path, api_key="test-api-key", max_results=7)

        self.assertEqual(report["provider"], "Google Cloud Vision Web Detection")
        self.assertEqual(report["full_matches"][0]["url"], "https://images.example/full.jpg")
        self.assertEqual(report["partial_matches"][0]["score"], 0.8)
        self.assertEqual(report["pages_with_matches"][0]["title"], "Image source page")
        self.assertEqual(report["visually_similar_images"][0]["url"], "https://images.example/similar.jpg")
        self.assertTrue(report["has_full_match"])
        self.assertTrue(report["has_web_references"])
        url, payload = post.call_args.args[:2]
        self.assertEqual(url, google_vision.VISION_ANNOTATE_URL)
        self.assertEqual(payload["requests"][0]["features"], [{"type": "WEB_DETECTION", "maxResults": 7}])
        self.assertEqual(
            base64.b64decode(payload["requests"][0]["image"]["content"]),
            self.fixture_image,
        )
        self.assertEqual(post.call_args.kwargs["headers"], {"X-Goog-Api-Key": "test-api-key"})
        self.assertNotIn("test-api-key", url)

    def test_requires_credentials_and_valid_image_before_upload(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ImageSearchError, "GOOGLE_CLOUD_VISION_API_KEY"):
                search_image("missing.png")

        invalid = self._temp_image(b"not an image")
        with patch("paper_search.sources.google_vision.http_post_json") as post:
            with self.assertRaisesRegex(ImageSearchError, "Unsupported or invalid image"):
                search_image(invalid, api_key="test-api-key")
        post.assert_not_called()

    def test_rejects_oversize_image_before_upload(self):
        path = self._temp_image(b"\x89PNG\r\n\x1a\n" + b"x" * MAX_INLINE_IMAGE_BYTES)
        with patch("paper_search.sources.google_vision.http_post_json") as post:
            with self.assertRaisesRegex(ImageSearchError, "7 MiB"):
                search_image(path, api_key="test-api-key")
        post.assert_not_called()

    @patch("paper_search.sources.google_vision.http_post_json", return_value=(403, b'{"error":{"message":"API not enabled"}}'))
    def test_reports_provider_authentication_errors_cleanly(self, _post):
        with self.assertRaisesRegex(ImageSearchError, "HTTP 403: API not enabled"):
            search_image(self._temp_image(), api_key="test-api-key")

    @patch("paper_search.cli.search_image")
    def test_cli_accepts_image_without_text_query_and_writes_exports(self, search):
        search.return_value = {
            "provider": "Google Cloud Vision Web Detection",
            "searched_at": "2026-10-18T00:00:00+00:00",
            "full_matches": [{"url": "https://images.example/full.jpg"}],
            "partial_matches": [],
            "pages_with_matches": [{"url": "https://example.org/page", "title": "Page", "full_matches": [], "partial_matches": []}],
            "visually_similar_images": [],
            "best_guess_labels": [],
            "web_entities": [],
            "has_full_match": True,
            "has_web_references": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "input.png"
            image.write_bytes(self.fixture_image)
            report_path = Path(folder) / "matches.md"
            export_path = Path(folder) / "matches.csv"
            output = StringIO()
            with redirect_stdout(output), redirect_stderr(StringIO()):
                exit_code = cli.main([
                    "--image", str(image), "--image-max-results", "5",
                    "--report", str(report_path), "--export", str(export_path),
                ])
            with export_path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            markdown = report_path.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertIn("مطابقة كاملة: 1", output.getvalue())
        self.assertIn("No match does not prove", markdown)
        self.assertEqual(rows[0]["category"], "full_match")
        search.assert_called_once_with(str(image), max_results=5)

    def test_cli_requires_either_text_query_or_image_path(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main([])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
