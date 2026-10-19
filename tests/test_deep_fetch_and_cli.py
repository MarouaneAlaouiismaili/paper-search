import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from paper_search import cli
from paper_search.llm import build_evidence, select_informative_results, MAX_EVIDENCE_CHARS
from paper_search.models import Paper


class DeepFetchAndCliTests(unittest.TestCase):
    def test_public_url_gate_rejects_private_hosts(self):
        for url in ("http://127.0.0.1/", "http://localhost/", "http://10.0.0.5/", "file:///etc/passwd", "https://user:pass@example.com/"):
            with self.subTest(url=url):
                self.assertFalse(cli.is_public_http_url(url))

    @patch("paper_search.cli.http_get")
    @patch("paper_search.cli.is_public_http_url", return_value=True)
    def test_deep_fetch_enforces_size_timeout_and_disables_redirects(self, _public, http_get):
        http_get.return_value = b"<html><body>Public page</body></html>"
        self.assertEqual(cli.fetch_page_text("https://example.org", timeout=8, max_bytes=1024), "Public page")
        http_get.assert_called_once_with("https://example.org", timeout=8, max_bytes=1024,
                                         allow_redirects=False, retries=1)

    @patch("paper_search.cli.is_public_http_url", return_value=True)
    @patch("paper_search.cli.fetch_page_text")
    def test_deep_fetch_processes_requested_records_up_to_limit(self, fetch, _public):
        fetch.return_value = "Research fact about a target name."
        records = [Paper("DuckDuckGo", f"Result {i}", f"https://example.org/{i}", abstract="target") for i in range(30)]
        cli.enrich_with_deep_content(records, "target", max_pages=30)
        self.assertEqual(fetch.call_count, 25)
        self.assertTrue(all(record.deep_context for record in records[:25]))
        self.assertFalse(records[25].deep_context)

    def test_informative_selection_keeps_specific_social_metadata_not_boilerplate(self):
        useful = Paper("DuckDuckGo", "Person social profile", "https://instagram.com/person",
                       abstract="34 followers, 20 following, 3 posts")
        generic = Paper("DuckDuckGo", "Person on Facebook", "https://facebook.com/person",
                        abstract="Join Facebook to connect with friends and others you may know.")
        selected = select_informative_results([generic, useful])
        self.assertEqual(selected, [useful])
        evidence = build_evidence("person", selected)
        self.assertIn("Type: social", evidence)
        self.assertLessEqual(len(evidence), MAX_EVIDENCE_CHARS)

    def test_llm_evidence_is_bounded_and_preserves_field_conflicts(self):
        record = Paper("OpenAlex", "A research profile", "https://example.org/person",
                       date="2020", abstract="Useful excerpt.", deep_context=["fact alpha" * 50, "fact beta" * 50, "fact gamma" * 50])
        record.field_evidence["date"].append({"source": "CrossRef", "value": "2021"})
        evidence = build_evidence("research profile", [record])
        self.assertIn("Conflicting date", evidence)
        self.assertLessEqual(evidence.count("Page context"), 1)
        self.assertLessEqual(len(evidence), MAX_EVIDENCE_CHARS)
        self.assertNotIn(record.deep_context[0], evidence)

    @patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", lambda *args, **kwargs: [
        Paper("DuckDuckGo", "A public researcher profile", "https://example.edu/person",
              abstract="Researcher at Example University; publications in applied mathematics.", kind="person")
    ])})
    def test_cli_web_search_exports_a_reviewable_report(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            report = str(Path(folder) / "case.json")
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                cli.main(["researcher", "--mode", "web", "--no-deep", "--top-influential", "0", "--report", report])
            self.assertIn("A public researcher profile", output.getvalue())
            self.assertTrue(Path(report).exists())

    def test_cli_includes_unknown_metadata_by_default_and_can_exclude_it(self):
        result = Paper("DuckDuckGo", "Researcher profile", "https://example.org/person",
                       abstract="A profile for the requested researcher.", kind="person")

        def source(*args, **kwargs):
            return [result]

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", source)}):
            included = io.StringIO()
            with redirect_stdout(included), redirect_stderr(io.StringIO()):
                cli.main(["researcher", "--mode", "web", "--source", "duckduckgo", "--year", "2020",
                          "--no-deep", "--top-influential", "0"])
            excluded = io.StringIO()
            with redirect_stdout(excluded), redirect_stderr(io.StringIO()):
                cli.main(["researcher", "--mode", "web", "--source", "duckduckgo", "--year", "2020",
                          "--unknown-metadata", "exclude", "--no-deep", "--top-influential", "0"])

        self.assertIn("Researcher profile", included.getvalue())
        self.assertIn("بيانات فلتر غير متوفرة: year", included.getvalue())
        self.assertIn("ما بقات حتى نتيجة", excluded.getvalue())

    def test_cli_strict_keeps_exact_structured_query_matches(self):
        result = Paper("DuckDuckGo", "Research overview", "https://example.org/research",
                       date="2020", abstract="A useful research overview.", kind="paper")

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", lambda *args, **kwargs: [result])}), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            cli.main(["research AND year:2020", "--mode", "web", "--source", "duckduckgo", "--strict",
                      "--no-deep", "--top-influential", "0"])

        self.assertIn("Research overview", output.getvalue())

    def test_cli_keeps_healthy_sources_when_another_source_fails(self):
        import json
        import tempfile
        from pathlib import Path

        def fail(*args, **kwargs):
            raise RuntimeError("fixture source failure")

        def succeed(*args, **kwargs):
            return [Paper("DuckDuckGo", "Public researcher profile", "https://example.org/person",
                          abstract="Researcher profile for the requested name.", kind="person")]

        with tempfile.TemporaryDirectory() as folder:
            report_path = str(Path(folder) / "case.json")
            with patch.dict(cli.SOURCE_FUNCTIONS, {
                "openalex": ("OpenAlex", fail),
                "duckduckgo": ("DuckDuckGo", succeed),
            }), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = cli.main(["researcher", "--mode", "investigation", "--source", "openalex",
                                      "--source", "duckduckgo", "--no-deep", "--top-influential", "0",
                                      "--report", report_path])
            report = json.loads(Path(report_path).read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["result_count"], 1)
        self.assertEqual(report["source_failures"][0]["source"], "OpenAlex")

    def test_cli_returns_failure_when_all_sources_fail(self):
        def fail(*args, **kwargs):
            raise RuntimeError("fixture source failure")

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", fail)}), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            exit_code = cli.main(["researcher", "--mode", "web", "--source", "duckduckgo",
                                  "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertEqual(exit_code, 1)
        self.assertIn("تعذر إكمال البحث", output.getvalue())
        self.assertNotIn("ما بقات حتى نتيجة تطابق", output.getvalue())

    def test_cli_distinguishes_a_partial_source_failure_from_all_sources_failing(self):
        responses = iter(([], RuntimeError("one request failed"), []))

        def source(*args, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", source)}), \
                patch("paper_search.cli.polite_wait"), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            exit_code = cli.main(["AB1234567", "--mode", "web", "--source", "duckduckgo",
                                  "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertEqual(exit_code, 0)
        self.assertIn("البحث غير مكتمل", output.getvalue())
        self.assertNotIn("فشلت جميع المصادر المختارة", output.getvalue())

    def test_cli_preserves_requested_source_pagination(self):
        calls = []

        def source(query, num, pages, **kwargs):
            calls.append((query, num, pages, kwargs))
            return []

        with patch.dict(cli.SOURCE_FUNCTIONS, {"openalex": ("OpenAlex", source)}), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main(["research", "--mode", "academic", "--source", "openalex", "--num", "7",
                      "--pages", "3", "--top-influential", "0", "--no-deep"])
        self.assertEqual(calls[0][1:3], (7, 3))

    def test_cli_searches_each_requested_site_with_multiple_default_pages(self):
        calls = []

        def source(query, num, pages, *, site=None, **kwargs):
            calls.append((query, num, pages, site))
            return [Paper("DuckDuckGo", f"Research at {site}", f"https://{site}/research",
                          abstract="Research information.")]

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", source)}), \
                patch("paper_search.cli.polite_wait"), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main(["research", "--mode", "web", "--source", "duckduckgo",
                      "--site", "example.edu", "--site", "example.org",
                      "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertEqual([call[3] for call in calls], ["example.edu", "example.org"])
        self.assertTrue(all(call[1:3] == (20, 3) for call in calls))

    def test_cli_keeps_labeled_partial_matches_for_plain_text_queries(self):
        records = [
            Paper("DuckDuckGo", "Alpha gamma profile", "https://example.org/partial",
                  abstract="A profile about alpha and gamma."),
            Paper("DuckDuckGo", "Beta profile", "https://example.org/unrelated",
                  abstract="Beta is mentioned here."),
        ]
        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", lambda *args, **kwargs: records)}), \
                patch("paper_search.cli.polite_wait"), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            cli.main(["alpha beta gamma", "--mode", "web", "--source", "duckduckgo",
                      "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertIn("Alpha gamma profile", output.getvalue())
        self.assertIn("تطابق جزئي: 2/3", output.getvalue())
        self.assertNotIn("Beta profile", output.getvalue())

    def test_cli_keeps_a_labeled_one_of_two_term_match(self):
        record = Paper("DuckDuckGo", "Alpha profile", "https://example.org/profile",
                       abstract="A public profile for alpha.")
        calls = []

        def source(query, num, pages, **kwargs):
            calls.append(query)
            return [record] if query == "alpha" else []

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", source)}), \
                patch("paper_search.cli.polite_wait"), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            cli.main(["alpha beta", "--mode", "web", "--source", "duckduckgo",
                      "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertEqual(calls, ["alpha beta", "alpha", "beta"])
        self.assertIn("Alpha profile", output.getvalue())
        self.assertIn("تطابق جزئي: 1/2", output.getvalue())

    def test_cli_continues_other_query_variants_after_a_request_fails(self):
        record = Paper("DuckDuckGo", "Alpha profile", "https://example.org/profile",
                       abstract="A public profile for alpha.")
        calls = []

        def source(query, num, pages, **kwargs):
            calls.append(query)
            if query == "alpha beta":
                raise RuntimeError("temporary request failure")
            return [record] if query == "alpha" else []

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", source)}), \
                patch("paper_search.cli.polite_wait"), \
                redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors:
            exit_code = cli.main(["alpha beta", "--mode", "web", "--source", "duckduckgo",
                                  "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertEqual(calls, ["alpha beta", "alpha", "beta"])
        self.assertEqual(exit_code, 0)
        self.assertIn("Alpha profile", output.getvalue())
        self.assertIn("أخطاء جزئية", errors.getvalue())

    def test_cli_runs_multiple_queries_for_plain_text(self):
        calls = []

        def source(query, num, pages, *, site=None, **kwargs):
            calls.append(query)
            return []

        with patch.dict(cli.SOURCE_FUNCTIONS, {"duckduckgo": ("DuckDuckGo", source)}), \
                patch("paper_search.cli.polite_wait") as wait, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cli.main(["alpha beta gamma", "--mode", "web", "--source", "duckduckgo",
                      "--no-deep", "--no-summary", "--top-influential", "0"])

        self.assertEqual(calls, ["alpha beta gamma", "alpha beta",
                                 "alpha gamma", "beta gamma"])
        self.assertEqual(wait.call_count, len(calls) - 1)


if __name__ == "__main__":
    unittest.main()
