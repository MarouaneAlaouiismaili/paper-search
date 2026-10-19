import json
import os
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from paper_search import cli
from paper_search.models import Paper
from paper_search.sources.brave import search_brave
from paper_search.query_language import evaluate_query


class BraveSourceTests(unittest.TestCase):
    def test_brave_requires_a_key_without_exposing_it(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "BRAVE_SEARCH_API_KEY"):
                search_brave("research")

        secret = "secret-brave-token"
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": secret}), \
                patch("paper_search.sources.brave.polite_wait"), \
                patch("paper_search.sources.brave.http_get", side_effect=RuntimeError(f"failure: {secret}")):
            with self.assertRaises(RuntimeError) as raised:
                search_brave("research", max_results=1)
        self.assertNotIn(secret, str(raised.exception))

    def test_brave_pushes_site_filter_and_maps_result_snippets(self):
        response = {"web": {"results": [{
            "title": "Research result",
            "url": "https://example.org/paper",
            "description": "A <strong>useful</strong> search excerpt.",
        }]}}
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-token"}), \
                patch("paper_search.sources.brave.polite_wait"), \
                patch("paper_search.sources.brave.http_get", return_value=json.dumps(response).encode()) as http_get:
            records = search_brave("research", max_results=5, pages=1, site="example.org")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "Brave Search")
        self.assertEqual(records[0].abstract, "A useful search excerpt.")
        self.assertTrue(evaluate_query("source:brave", records[0]))
        query = parse_qs(urlparse(http_get.call_args.args[0]).query)
        self.assertEqual(query["q"], ["research site:example.org"])
        self.assertEqual(query["count"], ["5"])
        self.assertEqual(query["offset"], ["0"])
        self.assertEqual(http_get.call_args.kwargs["headers"]["X-Subscription-Token"], "test-token")

    def test_brave_preserves_generated_exact_identifier_query(self):
        response = {"web": {"results": [{
            "title": "Reference result",
            "url": "https://example.org/record",
            "description": "A useful exact-match excerpt.",
        }]}}
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-token"}), \
                patch("paper_search.sources.brave.polite_wait"), \
                patch("paper_search.sources.brave.http_get", return_value=json.dumps(response).encode()) as http_get:
            search_brave("AB1234567", max_results=5)

        query = parse_qs(urlparse(http_get.call_args.args[0]).query)
        self.assertEqual(query["q"], ['"AB1234567"'])

    def test_brave_paginates_and_deduplicates_overlapping_pages(self):
        page_one = {"web": {"results": [
            {"title": "One", "url": "https://example.org/1", "description": "First result."},
            {"title": "Two", "url": "https://example.org/2", "description": "Second result."},
        ]}}
        page_two = {"web": {"results": [
            {"title": "Two", "url": "https://example.org/2", "description": "Second result."},
            {"title": "Three", "url": "https://example.org/3", "description": "Third result."},
        ]}}
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-token"}), \
                patch("paper_search.sources.brave.polite_wait"), \
                patch("paper_search.sources.brave.http_get", side_effect=[
                    json.dumps(page_one).encode(), json.dumps(page_two).encode()
                ]) as http_get:
            records = search_brave("research", max_results=2, pages=2)

        self.assertEqual([record.title for record in records], ["One", "Two", "Three"])
        requests = [parse_qs(urlparse(call.args[0]).query) for call in http_get.call_args_list]
        self.assertEqual([request["offset"][0] for request in requests], ["0", "1"])

    def test_brave_splits_large_result_counts_and_caps_at_api_limit(self):
        def response_for(url, **_kwargs):
            offset = int(parse_qs(urlparse(url).query)["offset"][0])
            return json.dumps({"web": {"results": [{
                "title": f"Result {offset * 20 + index}",
                "url": f"https://example.org/{offset * 20 + index}",
                "description": "A public search result.",
            } for index in range(20)]}}).encode()

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-token"}), \
                patch("paper_search.sources.brave.polite_wait"), \
                patch("paper_search.sources.brave.http_get", side_effect=response_for) as http_get:
            records = search_brave("research", max_results=100, pages=10)

        self.assertEqual(len(records), 200)
        self.assertEqual(http_get.call_count, 10)
        offsets = [parse_qs(urlparse(call.args[0]).query)["offset"][0] for call in http_get.call_args_list]
        self.assertEqual(offsets, [str(value) for value in range(10)])

    def test_brave_stops_when_provider_says_no_more_results(self):
        response = {"query": {"more_results_available": False}, "web": {"results": [
            {"title": "Only result", "url": "https://example.org/only", "description": "A result."},
            {"title": "Second result", "url": "https://example.org/two", "description": "Another result."},
        ]}}
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-token"}), \
                patch("paper_search.sources.brave.polite_wait"), \
                patch("paper_search.sources.brave.http_get", return_value=json.dumps(response).encode()) as http_get:
            records = search_brave("research", max_results=2, pages=4)

        self.assertEqual(len(records), 2)
        http_get.assert_called_once()

    def test_brave_is_automatically_used_only_when_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("brave", cli._selected_source_keys("web", []))
            self.assertNotIn("brave", cli._selected_source_keys("investigation", []))
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-token"}, clear=True):
            self.assertIn("brave", cli._selected_source_keys("web", []))
            self.assertIn("brave", cli._selected_source_keys("investigation", []))
            self.assertNotIn("brave", cli._selected_source_keys("academic", []))

        record = Paper("Brave Search", "Web result", "https://example.org")
        self.assertTrue(evaluate_query("source:bravesearch", record))
        self.assertIn("Brave Search", cli.PUBLIC_WEB_SOURCES)


if __name__ == "__main__":
    unittest.main()
