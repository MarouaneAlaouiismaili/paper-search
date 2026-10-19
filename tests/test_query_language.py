import unittest

from paper_search.models import Paper
from paper_search.query_language import (
    QuerySyntaxError, evaluate_query, parse_query, source_query, source_query_variants,
)
from paper_search.utils import calculate_bm25_scores, filter_records


class QueryLanguageTests(unittest.TestCase):
    def setUp(self):
        self.paper = Paper(
            source="OpenAlex", title="LLM security for climate change", url="https://example.edu/paper/1",
            authors="Ada Lovelace", date="2023-04-01", abstract="A study of AI safety and security.",
            citations=120, open_access=True, language="en",
        )

    def test_boolean_precedence_parentheses_and_implicit_and(self):
        self.assertTrue(evaluate_query("LLM AND security NOT blockchain", self.paper))
        self.assertTrue(evaluate_query("(LLM OR alignment) AND security", self.paper))
        self.assertFalse(evaluate_query("LLM OR alignment AND blockchain", Paper("x", "alignment only", "")))
        self.assertFalse(evaluate_query("LLM AND (security OR blockchain)", Paper("x", "LLM", "")))

    def test_exact_phrases_and_fields(self):
        self.assertTrue(evaluate_query('title:"climate change" AND year:2020..2024', self.paper))
        self.assertTrue(evaluate_query("site:example.edu AND type:paper AND source:openalex", self.paper))
        self.assertTrue(evaluate_query("citations:>=100 AND open_access:true AND language:en", self.paper))
        self.assertFalse(evaluate_query('title:"change climate"', self.paper))
        self.assertEqual(source_query(parse_query('title:"climate change" AND year:2020..2024')), "climate change")
        self.assertEqual(source_query_variants(parse_query("LLM OR security")), ["LLM", "security"])
        self.assertEqual(source_query_variants(parse_query("(LLM OR security) AND privacy")),
                         ["LLM privacy", "security privacy"])
        self.assertEqual(source_query_variants(parse_query("alpha beta gamma")), [
            "alpha beta gamma", "alpha beta", "alpha gamma", "beta gamma",
        ])
        self.assertEqual(source_query_variants(parse_query("alpha beta")), [
            "alpha beta", "alpha", "beta",
        ])
        variants = source_query_variants(parse_query("alpha bravo charlie delta echo foxtrot"))
        self.assertEqual(len(variants), 8)
        self.assertTrue(all(any(word in variant.split() for variant in variants[1:])
                            for word in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")))
        self.assertEqual(source_query_variants(parse_query('"alpha beta gamma"')),
                         ["alpha beta gamma"])
        with self.assertRaisesRegex(QuerySyntaxError, "positive search term"):
            source_query_variants(parse_query("LLM OR NOT blockchain"))
        with self.assertRaisesRegex(QuerySyntaxError, "more than 8"):
            source_query_variants(parse_query("a OR b OR c OR d OR e OR f OR g OR h OR i"))

    def test_invalid_syntax_and_field_values_are_rejected(self):
        for query in ('"unclosed', "title:", "title:unclosed AND", "x OR", "(x AND y", "color:red", "year:2024..2020", "year:0999", "citations:many", "open_access:maybe", "type:alien", "source:unknown", "site:example.com/path", "language:english"):
            with self.subTest(query=query), self.assertRaises(QuerySyntaxError):
                parse_query(query)

    def test_source_field_supports_academic_alias(self):
        web = Paper("DuckDuckGo", "LLM page", "https://example.org")
        self.assertFalse(evaluate_query("source:academic", web))
        self.assertTrue(evaluate_query("source:all", web))

    def test_matching_normalizes_accents_arabic_diacritics_and_word_boundaries(self):
        accented = Paper("OpenAlex", "Café research on machine learning", "https://example.org",
                         authors="Researcher Example")
        arabic = Paper("OpenAlex", "مَرْوَان العلوي الإسماعيلي", "https://example.org")
        self.assertTrue(evaluate_query("cafe", accented))
        self.assertTrue(evaluate_query("author:researcher", accented))
        self.assertTrue(evaluate_query("مروان", arabic))
        self.assertFalse(evaluate_query("art", Paper("x", "partial result", "")))

    def test_search_identifiers_match_urls_and_dois(self):
        email_result = Paper(
            "DuckDuckGo", "Public profile", "https://example.org/contact?email=demo.searcher%40example.org",
        )
        code_result = Paper(
            "CrossRef", "Record page", "https://example.org/records/", doi="10.1000/DEMO7319X",
        )
        self.assertTrue(evaluate_query("demo.searcher@example.org", email_result))
        self.assertTrue(evaluate_query("DEMO7319X", code_result))
        self.assertFalse(evaluate_query("DEMO7319X", Paper("x", "DEMO7319X0 record", "")))

        matched, rejected = filter_records([email_result], query="demo.searcher@example.org")
        self.assertEqual(matched, [email_result])
        self.assertEqual(rejected, {})
        calculate_bm25_scores([email_result], "demo.searcher@example.org")
        self.assertGreater(email_result.bm25_score, 0)

    def test_compound_identifiers_match_common_separator_variants(self):
        record = Paper(
            "DuckDuckGo", "Record AB-123 4567", "https://example.org/records/AB-123-4567",
            abstract="A public record with reference AB/123.4567.",
        )
        self.assertTrue(evaluate_query("AB1234567", record))
        self.assertFalse(evaluate_query("AB12345670", record))

        matched, rejected = filter_records([record], query="AB1234567")
        self.assertEqual(matched, [record])
        self.assertEqual(rejected, {})
        calculate_bm25_scores([record], "AB1234567")
        self.assertGreater(record.bm25_score, 0)

    def test_identifier_queries_add_exact_and_formatted_web_searches(self):
        self.assertEqual(source_query_variants(parse_query("AB1234567")), [
            "AB1234567", "AB 1234567",
        ])
        self.assertEqual(source_query_variants(parse_query("profile@example.org")), [
            "profile@example.org", "profile",
        ])

    def test_unknown_optional_metadata_can_be_included_with_visible_uncertainty(self):
        unknown = Paper(source="CrossRef", title="LLM result", url="https://example.org/result")
        matched, rejected = filter_records(
            [unknown], query="LLM AND year:2020", min_citations=50, include_unknown=True,
        )
        self.assertEqual(matched, [unknown])
        self.assertEqual(rejected, {})
        self.assertEqual(set(unknown.filter_uncertainties), {"citations", "query:year"})

        exact_alternative = Paper(source="CrossRef", title="LLM result", url="https://example.org/result")
        matched, _ = filter_records(
            [exact_alternative], query="LLM OR year:2020", include_unknown=True,
        )
        self.assertEqual(matched, [exact_alternative])
        self.assertEqual(exact_alternative.filter_uncertainties, [])

    def test_unknown_site_and_source_remain_fail_closed(self):
        missing_url = Paper(source="OpenAlex", title="LLM result", url="")
        matched, rejected = filter_records(
            [missing_url], query="LLM AND site:example.org", include_unknown=True,
        )
        self.assertEqual(matched, [])
        self.assertEqual(rejected["query_unknown"], 1)

        missing_source = Paper(source="", title="LLM result", url="https://example.org")
        matched, rejected = filter_records(
            [missing_source], query="LLM AND source:openalex", include_unknown=True,
        )
        self.assertEqual(matched, [])
        self.assertEqual(rejected["query_unknown"], 1)

    def test_filters_fail_closed_on_missing_metadata_and_report_reasons(self):
        unknown = Paper(source="CrossRef", title="LLM result", url="https://example.org/result")
        matched, rejected = filter_records(
            [self.paper, unknown], query="LLM", from_year=2020, min_citations=50,
            open_access=True, language="en",
        )
        self.assertEqual(matched, [self.paper])
        self.assertEqual(rejected["citations_unknown"], 1)
        self.assertEqual(rejected["open_access_unknown"], 1)
        self.assertEqual(rejected["language_unknown"], 1)

    def test_negated_field_filter_does_not_match_when_metadata_is_unknown(self):
        unknown = Paper(source="CrossRef", title="Unrelated title", url="https://example.org/result")
        self.assertFalse(evaluate_query("NOT year:2020", unknown))
        matched, rejected = filter_records([unknown], query="NOT year:2020")
        self.assertEqual(matched, [])
        self.assertEqual(rejected["query_unknown"], 1)

    def test_cli_style_filters_are_repeatable_and_strict(self):
        second = Paper("DuckDuckGo", "LLM report", "https://sub.example.edu/report", kind="website")
        matched, _ = filter_records([self.paper, second], query="LLM", sites=["example.edu"], types=["paper"])
        self.assertEqual(matched, [self.paper])


if __name__ == "__main__":
    unittest.main()
