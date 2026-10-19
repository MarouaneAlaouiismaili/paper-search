import json
import tempfile
import unittest
from pathlib import Path

from paper_search.models import EvidenceRecord, Paper
from paper_search.reports import build_report, write_report
from paper_search.utils import calculate_bm25_scores, combine_relevance_scores, dedupe


class RecordsAndReportsTests(unittest.TestCase):
    def test_paper_alias_preserves_old_constructor_and_adds_provenance(self):
        record = Paper("CrossRef", "A title", "https://example.org", "Ada Lovelace", "2020")
        self.assertIsInstance(record, EvidenceRecord)
        self.assertEqual(record.authors, "Ada Lovelace")
        self.assertEqual(record.kind, "paper")
        self.assertIn("2020", record.field_evidence["date"][0]["value"])
        self.assertTrue(record.retrieved_at.endswith("+00:00"))

    def test_dedupe_is_confident_and_does_not_merge_people_by_name(self):
        same_doi_a = Paper("OpenAlex", "Research title", "https://doi.org/10.1000/xyz", authors="A Person", date="2020", doi="10.1000/xyz", abstract="Abstract")
        same_doi_b = Paper("CrossRef", "Different metadata title", "https://publisher.example/work", authors="A Person", date="2020", doi="https://doi.org/10.1000/XYZ")
        url_a = Paper("DuckDuckGo", "A person's profile", "https://www.example.org/profile?utm_source=search", kind="person")
        url_b = Paper("DuckDuckGo", "A person's profile", "https://example.org/another-profile", kind="person")
        merged = dedupe([same_doi_a, same_doi_b, url_a, url_b])
        self.assertEqual(len(merged), 3)
        doi_record = next(item for item in merged if item.doi)
        self.assertEqual(set(doi_record.sources), {"OpenAlex", "CrossRef"})
        self.assertEqual(len(doi_record.field_evidence["title"]), 2)
        self.assertEqual(len([item for item in merged if item.kind == "person"]), 2)

    def test_same_name_across_page_types_remains_separate_evidence(self):
        university = Paper("DuckDuckGo", "Researcher Example — University profile",
                           "https://example.edu/people/researcher", abstract="Professor profile", kind="person")
        research = Paper("OpenAlex", "Numerical methods by Researcher Example",
                         "https://doi.org/10.1000/research", doi="10.1000/research")
        social = Paper("DuckDuckGo", "Researcher Example on Instagram",
                       "https://instagram.com/researcher", abstract="23 followers", kind="social")
        merged = dedupe([university, research, social])
        self.assertEqual(len(merged), 3)
        self.assertEqual({record.kind for record in merged}, {"person", "paper", "social"})

    def test_url_dedupe_removes_known_trackers_but_keeps_semantic_parameters(self):
        tracked = Paper("DuckDuckGo", "Profile", "https://example.org/profile?utm_source=search&fbclid=abc")
        canonical = Paper("DuckDuckGo", "Profile", "https://www.example.org/profile")
        account_a = Paper("DuckDuckGo", "Account A", "https://example.org/profile?source=a")
        account_b = Paper("DuckDuckGo", "Account B", "https://example.org/profile?source=b")
        merged = dedupe([tracked, canonical, account_a, account_b])
        self.assertEqual(len(merged), 3)
        self.assertEqual(set(merged[0].sources), {"DuckDuckGo"})

    def test_relevance_ignores_citations_and_age_bonuses(self):
        low = Paper("x", "LLM security", "https://example.org/a", citations=0, date="1990")
        high = Paper("x", "LLM security", "https://example.org/b", citations=100000, date="2026")
        calculate_bm25_scores([low, high], "LLM security")
        self.assertEqual(low.bm25_score, high.bm25_score)

    def test_source_rank_fusion_and_conflicting_fields_are_reported(self):
        first = Paper("OpenAlex", "Same work", "https://doi.org/10.1000/x", doi="10.1000/x", citations=5, abstract="An abstract.")
        first.source_ranks["OpenAlex"] = 1
        second = Paper("CrossRef", "Same work", "https://crossref.example/work", doi="10.1000/x", citations=9)
        second.source_ranks["CrossRef"] = 2
        merged = dedupe([first, second])
        report = build_report(merged, "Same work")
        self.assertEqual({item["field"] for item in report["conflicts"]}, {"citations", "url"})
        calculate_bm25_scores(merged, "Same work")
        combine_relevance_scores(merged)
        self.assertGreater(merged[0].score, 0)

    def test_json_and_markdown_reports_are_written(self):
        record = Paper("OpenAlex", "LLM work", "https://example.org", abstract="Evidence.", citations=0, open_access=False)
        record.filter_uncertainties = ["year", "query:year"]
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "case.json"
            md_path = Path(folder) / "case.md"
            write_report(str(json_path), [record], "LLM", filters={"mode": "academic"})
            write_report(str(md_path), [record], "LLM")
            report = json.loads(json_path.read_text())
            self.assertEqual(report["result_count"], 1)
            self.assertEqual(report["missing_metadata_counts"]["citations"], 0)
            self.assertEqual(report["missing_metadata_counts"]["open_access"], 0)
            self.assertEqual(report["inferences"], [])
            self.assertEqual(report["filter_uncertainties"], {"query:year": 1, "year": 1})
            self.assertEqual(report["results"][0]["filter_uncertainties"], ["year", "query:year"])
            self.assertIn("Field provenance", md_path.read_text())
            self.assertIn("Filter metadata missing", md_path.read_text())


if __name__ == "__main__":
    unittest.main()
