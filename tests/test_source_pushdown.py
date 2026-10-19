import json
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from paper_search.sources.crossref import search_crossref
from paper_search.sources.openalex import search_openalex
from paper_search.sources.pubmed import search_pubmed


class SourcePushdownTests(unittest.TestCase):
    @patch("paper_search.sources.openalex.http_get")
    def test_openalex_pushes_supported_restrictions_to_api(self, http_get):
        http_get.return_value = json.dumps({"results": []}).encode()
        search_openalex("language models", 10, 1, from_year=2020, to_year=2024,
                        author="Ada Lovelace", min_citations=25, open_access=True)
        url = http_get.call_args.args[0]
        params = parse_qs(urlparse(url).query)
        self.assertIn("from_publication_date:2020-01-01", params["filter"][0])
        self.assertIn("to_publication_date:2024-12-31", params["filter"][0])
        self.assertIn("raw_author_name.search:Ada Lovelace", params["filter"][0])
        self.assertIn("cited_by_count:>=25", params["filter"][0])
        self.assertIn("open_access.is_oa:true", params["filter"][0])

    @patch("paper_search.sources.crossref.http_get")
    def test_crossref_pushes_dates_and_author_to_api(self, http_get):
        http_get.return_value = json.dumps({"message": {"items": []}}).encode()
        search_crossref("climate policy", 10, 1, from_year=2021, to_year=2023, author="Ada Lovelace")
        params = parse_qs(urlparse(http_get.call_args.args[0]).query)
        self.assertEqual(params["filter"], ["from-pub-date:2021-01-01,until-pub-date:2023-12-31"])
        self.assertEqual(params["query.author"], ["Ada Lovelace"])

    @patch("paper_search.sources.pubmed.http_get")
    def test_pubmed_pushes_author_and_publication_year_range(self, http_get):
        http_get.return_value = json.dumps({"esearchresult": {"idlist": []}}).encode()
        search_pubmed("climate policy", 10, 1, from_year=2021, to_year=2023, author="Ada Lovelace")
        term = parse_qs(urlparse(http_get.call_args.args[0]).query)["term"][0]
        self.assertIn("Ada Lovelace[Author]", term)
        self.assertIn("2021:2023[Date - Publication]", term)


if __name__ == "__main__":
    unittest.main()
