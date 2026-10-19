import io
import json
import os
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from paper_search import cli
from paper_search.models import Paper
from paper_search.network import DNSResolutionError
from paper_search.query_language import evaluate_query
from paper_search.sources.google import search_google
from paper_search.sources.duckduckgo import search_duckduckgo


class GoogleSourceTests(unittest.TestCase):
    def test_duckduckgo_pushes_site_restriction_into_search_request(self):
        html = (b'<a class="result__a" href="https://example.org/page">Research result</a>'
                b'<a class="result__snippet">Research notes and findings.</a>')
        with patch("paper_search.sources.duckduckgo.http_get", return_value=html) as http_get, \
                patch("paper_search.sources.duckduckgo.polite_wait"):
            records = search_duckduckgo("research", max_results=1, pages=1, site="example.org")

        self.assertEqual(len(records), 1)
        request_query = parse_qs(urlparse(http_get.call_args.args[0]).query)["q"][0]
        self.assertEqual(request_query, "research site:example.org")

    def test_duckduckgo_tries_lite_host_after_primary_dns_failure(self):
        html = (b'<a class="result__a" href="https://example.org/page">A result</a>'
                b'<a class="result__snippet">A useful excerpt for testing.</a>')
        with patch("paper_search.sources.duckduckgo.http_get",
                   side_effect=[DNSResolutionError("DNS lookup failed for html.duckduckgo.com"), html]) as get, \
                patch("paper_search.sources.duckduckgo.polite_wait"), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            records = search_duckduckgo("testing", max_results=1)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].url, "https://example.org/page")
        self.assertIn("lite.duckduckgo.com/lite/", get.call_args.args[0])
        self.assertIn("تعذر حلّ مضيف", stderr.getvalue())

    def test_duckduckgo_reports_captcha_as_a_failed_request(self):
        blocked = b"<html>captcha unusual traffic</html>"
        with patch("paper_search.sources.duckduckgo.http_get", side_effect=[blocked, blocked]), \
                patch("paper_search.sources.duckduckgo.polite_wait"), \
                patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaisesRegex(RuntimeError, "anti-bot/CAPTCHA"):
                search_duckduckgo("research", max_results=1)

    def test_google_search_maps_results_and_uses_required_parameters(self):
        response = {"items": [{
            "title": "Research result",
            "link": "https://example.org/paper",
            "snippet": "A useful search excerpt.",
        }]}
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "GOOGLE_CSE_ID": "test-cx"}), \
                patch("paper_search.sources.google.polite_wait"), \
                patch("paper_search.sources.google.http_get", return_value=json.dumps(response).encode()) as http_get:
            records = search_google("research", max_results=5)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "Google")
        self.assertEqual(records[0].title, "Research result")
        params = parse_qs(urlparse(http_get.call_args.args[0]).query)
        self.assertEqual(params["key"], ["test-key"])
        self.assertEqual(params["cx"], ["test-cx"])
        self.assertEqual(params["q"], ["research"])
        self.assertEqual(params["num"], ["5"])

    def test_google_search_pushes_site_restriction_into_query(self):
        response = {"items": [{
            "title": "Research result",
            "link": "https://example.org/paper",
            "snippet": "A useful search excerpt.",
        }]}
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "GOOGLE_CSE_ID": "test-cx"}), \
                patch("paper_search.sources.google.polite_wait"), \
                patch("paper_search.sources.google.http_get", return_value=json.dumps(response).encode()) as http_get:
            records = search_google("research", max_results=1, site="example.org")

        self.assertEqual(len(records), 1)
        params = parse_qs(urlparse(http_get.call_args.args[0]).query)
        self.assertEqual(params["q"], ["research site:example.org"])

    def test_google_preserves_generated_exact_identifier_query(self):
        response = {"items": [{
            "title": "Reference result",
            "link": "https://example.org/record",
            "snippet": "A useful exact-match excerpt.",
        }]}
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "GOOGLE_CSE_ID": "test-cx"}), \
                patch("paper_search.sources.google.polite_wait"), \
                patch("paper_search.sources.google.http_get", return_value=json.dumps(response).encode()) as http_get:
            search_google("AB1234567", max_results=5)

        params = parse_qs(urlparse(http_get.call_args.args[0]).query)
        self.assertEqual(params["q"], ['"AB1234567"'])

    def test_google_search_paginates_in_at_most_ten_results_per_request(self):
        def page(start, count):
            return {"items": [{
                "title": f"Result {start + index}",
                "link": f"https://example.org/{start + index}",
                "snippet": "Excerpt",
            } for index in range(count)]}

        responses = [json.dumps(page(1, 10)).encode(), json.dumps(page(11, 5)).encode()]
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "GOOGLE_CSE_ID": "test-cx"}), \
                patch("paper_search.sources.google.polite_wait"), \
                patch("paper_search.sources.google.http_get", side_effect=responses) as http_get:
            records = search_google("research", max_results=15, pages=1)

        self.assertEqual(len(records), 15)
        requests = [parse_qs(urlparse(call.args[0]).query) for call in http_get.call_args_list]
        self.assertEqual([request["start"][0] for request in requests], ["1", "11"])
        self.assertEqual([request["num"][0] for request in requests], ["10", "5"])

    def test_google_search_requires_credentials_and_redacts_key_from_errors(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY.*GOOGLE_CSE_ID"):
                search_google("research")

        secret = "secret+api-key"
        leaking_url = "https://www.googleapis.com/customsearch/v1?key=secret%2Bapi-key&cx=test"
        with patch.dict(os.environ, {"GOOGLE_API_KEY": secret, "GOOGLE_CSE_ID": "test-cx"}), \
                patch("paper_search.sources.google.polite_wait"), \
                patch("paper_search.sources.google.http_get", side_effect=RuntimeError(f"HTTP 403 on {leaking_url}")):
            with self.assertRaises(RuntimeError) as raised:
                search_google("research", max_results=1)
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn("secret%2Bapi-key", str(raised.exception))

    def test_google_source_is_web_only_and_optional_in_default_modes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("google", cli._selected_source_keys("web", []))
            self.assertNotIn("google", cli._selected_source_keys("investigation", []))
            self.assertNotIn("google", cli._selected_source_keys("academic", []))
            self.assertIn("google", cli._selected_source_keys("web", ["google"]))

        record = Paper("Google", "Google result", "https://example.org")
        self.assertTrue(evaluate_query("source:google", record))
        self.assertFalse(evaluate_query("source:academic", record))

    def test_google_results_can_be_selected_for_bounded_page_enrichment(self):
        record = Paper("Google", "Public web result", "https://example.org/page", abstract="research")
        with patch("paper_search.cli.is_public_http_url", return_value=True), \
                patch("paper_search.cli.fetch_page_text", return_value="Research details from the page.") as fetch:
            cli.enrich_with_deep_content([record], "research", max_pages=1)
        fetch.assert_called_once_with(record.url, timeout=8, max_bytes=1_048_576)
        self.assertTrue(record.deep_context)


if __name__ == "__main__":
    unittest.main()
