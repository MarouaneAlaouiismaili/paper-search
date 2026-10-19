import html
import re
import math
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime
from typing import List, Optional
from urllib.parse import unquote

from .models import Paper
from .query_language import (
    _EMAIL_TERM_RE, _identifier_pattern, evaluate_query, matches_query, normalized_text_contains,
    parse_query, positive_terms, unknown_query_fields,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!؟?])\s+")
_NORM_RE = re.compile(r"[^\w\s]", re.UNICODE)
TAG_RE = re.compile(r"<[^>]+>")

def _term_count(text: str, term: str) -> int:
    identifier_pattern = _identifier_pattern(term)
    if identifier_pattern:
        return sum(1 for _ in identifier_pattern.finditer(unquote(text or "")))
    return len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, flags=re.IGNORECASE | re.UNICODE))

def clean_html(raw: str) -> str:
    return html.unescape(TAG_RE.sub("", raw)).strip()

def _paper_year(paper: Paper) -> Optional[int]:
    d = paper.date or ""
    m = re.search(r"(19|20)\d{2}", d)
    return int(m.group(0)) if m else None

def parse_logical_query(query: str):
    """Compatibility helper exposing positive and excluded terms."""
    tree = parse_query(query)

    def collect(node, negated=False):
        if node is None:
            return [], []
        if node.kind == "NOT":
            return collect(node.children[0], not negated)
        if node.kind == "TERM":
            return ([], [node.value]) if negated else ([node.value], [])
        positive, negative = [], []
        for child in node.children:
            pos, neg = collect(child, negated)
            positive.extend(pos)
            negative.extend(neg)
        return positive, negative

    positive, negative = collect(tree)
    return [], positive, negative

def query_terms(query: str) -> List[str]:
    return positive_terms(parse_query(query), include_filter_values=True)


def web_query_text(query: str) -> str:
    """Keep generated exact-phrase searches intact for public web providers."""
    query = (query or "").strip()
    if len(query) >= 2 and query[0] == query[-1] and query[0] in {'"', "'"}:
        return query
    terms = query_terms(query)
    if len(terms) == 1 and (
            _EMAIL_TERM_RE.fullmatch(terms[0])
            or re.fullmatch(r"[A-Za-z0-9]+", terms[0]) and _identifier_pattern(terms[0])):
        return f'"{terms[0]}"'
    return " ".join(terms).strip()

def is_relevant(paper: Paper, query: str, require_all: bool = False, target_year: str = None, target_author: str = None) -> bool:
    # 1. Filter by Year
    if target_year:
        pyear = _paper_year(paper)
        if not pyear:
            return False
        if "-" in target_year:
            parts = target_year.split("-")
            try:
                min_y = int(parts[0]) if parts[0] else 0
                max_y = int(parts[1]) if parts[1] else 9999
                if not (min_y <= pyear <= max_y):
                    return False
            except ValueError:
                pass
        else:
            try:
                if pyear != int(target_year):
                    return False
            except ValueError:
                pass

    # 2. Filter by Author
    if target_author:
        if target_author.lower() not in (paper.authors or "").lower():
            return False

    if require_all:
        haystack = f"{paper.title} {paper.abstract} {paper.authors}".casefold()
        if any(not _term_count(haystack, term) for term in query_terms(query)):
            return False
    return evaluate_query(query, paper)

# BM25 Implementation
def calculate_bm25_scores(papers: List[Paper], query: str) -> List[Paper]:
    """
    Applies BM25 scoring algorithm to rank the papers.
    """
    terms = list(dict.fromkeys(query_terms(query)))
    if not terms or not papers:
        for p in papers:
            p.bm25_score = 0.0
            p.score = 0.0
        return papers

    k1 = 1.5
    b = 0.75

    df = {t: 0 for t in terms}
    doc_lengths = []

    for p in papers:
        text = f"{p.title} {p.abstract} {p.authors} {p.url} {unquote(p.url)} {p.doi} {' '.join(p.deep_context)}".lower()
        length = len(text.split())
        doc_lengths.append(length)
        for t in terms:
            if _term_count(text, t):
                df[t] += 1

    N = len(papers)
    avgdl = sum(doc_lengths) / max(1, N)

    idf = {}
    for t in terms:
        idf[t] = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))

    for i, p in enumerate(papers):
        text = f"{p.title} {p.abstract} {p.authors} {p.url} {unquote(p.url)} {p.doi} {' '.join(p.deep_context)}".lower()
        title_text = (p.title or "").lower()
        length = doc_lengths[i]

        score = 0.0
        for t in terms:
            tf = _term_count(text, t) + (_term_count(title_text, t) * 2)
            if tf > 0:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (length / max(1, avgdl)))
                score += idf[t] * (numerator / denominator)

        p.bm25_score = round(score, 6)
        p.score = p.bm25_score

    return papers

def _normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title or "").casefold()
    t = _NORM_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _normalize_doi(doi: str) -> str:
    doi = (doi or "").strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.rstrip(" .;,)")


def _canonical_url(url: str) -> str:
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
    parsed = urlsplit((url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
    tracking = {"fbclid", "gclid"}
    query = urlencode([(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                       if not k.casefold().startswith("utm_") and k.casefold() not in tracking])
    path = re.sub(r"/{2,}", "/", parsed.path or "/").rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), host + port, path, query, ""))

def dedupe(results: List[Paper]) -> List[Paper]:
    n = len(results)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    doi_to_idx = {}
    url_to_idx = {}
    openalex_to_idx = {}
    for i, r in enumerate(results):
        doi = _normalize_doi(r.doi or "")
        canonical_url = _canonical_url(r.url)
        if doi:
            if doi in doi_to_idx:
                union(doi_to_idx[doi], i)
            else:
                doi_to_idx[doi] = i
        if canonical_url:
            if canonical_url in url_to_idx:
                union(url_to_idx[canonical_url], i)
            else:
                url_to_idx[canonical_url] = i
        openalex_id = (r.openalex_id or "").strip().rstrip("/").rsplit("/", 1)[-1].casefold()
        if re.fullmatch(r"w\d+", openalex_id):
            if openalex_id in openalex_to_idx:
                union(openalex_to_idx[openalex_id], i)
            else:
                openalex_to_idx[openalex_id] = i

    # Similar titles are only merged when scholarly metadata independently agrees.
    paper_indexes = [i for i, record in enumerate(results)
                     if record.kind == "paper" and record.authors and _paper_year(record)]
    for offset, left_index in enumerate(paper_indexes):
        left = results[left_index]
        for right_index in paper_indexes[offset + 1:]:
            right = results[right_index]
            if _paper_year(left) != _paper_year(right):
                continue
            if _normalize_title(left.authors) != _normalize_title(right.authors):
                continue
            title_similarity = SequenceMatcher(None, _normalize_title(left.title), _normalize_title(right.title)).ratio()
            if title_similarity >= 0.98:
                union(left_index, right_index)

    groups = {}
    for i, r in enumerate(results):
        if not (r.doi or "").strip() and not _canonical_url(r.url) and not _normalize_title(r.title):
            continue
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        idxs_sorted = sorted(idxs)
        best_idx = max(idxs_sorted, key=lambda i: sum(bool(getattr(results[i], field)) for field in
                        ("authors", "date", "abstract", "citations", "doi")))
        base = results[best_idx]
        sources = []
        field_evidence = {}
        source_ranks = {}
        discovery_methods = []
        discovered_from = []
        for i in idxs_sorted:
            r = results[i]
            for source in (r.sources or [r.source]):
                if source not in sources:
                    sources.append(source)
            for field, entries in r.field_evidence.items():
                field_evidence.setdefault(field, []).extend(entries)
            for source, rank in r.source_ranks.items():
                source_ranks[source] = min(source_ranks.get(source, rank), rank)
            for method in r.discovery_methods:
                if method not in discovery_methods:
                    discovery_methods.append(method)
            for parent_title in r.discovered_from:
                if parent_title not in discovered_from:
                    discovered_from.append(parent_title)
            if r is not base:
                for field in ("citations", "doi", "abstract", "openalex_id", "referenced_works", "authors", "date", "url", "open_access", "language"):
                    if not getattr(base, field) and getattr(r, field):
                        setattr(base, field, getattr(r, field))
        base.sources = sources
        base.field_evidence = field_evidence
        base.source_ranks = source_ranks
        base.discovery_methods = discovery_methods
        base.discovered_from = discovered_from
        merged.append((min(idxs_sorted), base))

    merged.sort(key=lambda t: t[0])
    return [b for _, b in merged]

def sort_results(results: List[Paper], by: str = "relevance") -> List[Paper]:
    if by == "date":
        def date_key(r):
            d = r.date or ""
            try:
                return datetime.strptime(d[:10], "%Y-%m-%d")
            except Exception:
                year = _paper_year(r)
                return datetime(year, 1, 1) if year else datetime.min
        return sorted(results, key=date_key, reverse=True)
    if by == "citations":
        return sorted(results, key=lambda r: r.citations or 0, reverse=True)
    return sorted(
        results,
        key=lambda r: (r.score, r.match_count, len(r.sources)),
        reverse=True,
    )


def combine_relevance_scores(results: List[Paper]) -> List[Paper]:
    """Fuse BM25 and provider rankings with reciprocal rank fusion (RRF)."""
    lexical_ranked = sorted(
        (record for record in results if record.bm25_score > 0),
        key=lambda record: (-record.bm25_score, record.title.casefold(), record.url),
    )
    lexical_ranks = {id(record): rank for rank, record in enumerate(lexical_ranked, 1)}
    rrf_scores = {}
    for record in results:
        ranks = list(record.source_ranks.values())
        if id(record) in lexical_ranks:
            ranks.append(lexical_ranks[id(record)])
        rrf_scores[id(record)] = sum(1 / (60 + max(1, int(rank))) for rank in ranks)

    max_rrf = max(rrf_scores.values(), default=0.0)
    for record in results:
        record.score = round(rrf_scores[id(record)] / max_rrf, 6) if max_rrf else 0.0
    return results


def filter_records(records: List[Paper], *, query: str = "", year: str = None,
                   from_year: int = None, to_year: int = None, author: str = None,
                   sites: List[str] = None, types: List[str] = None, sources: List[str] = None,
                   min_citations: int = None, open_access: bool = False,
                   language: str = None, include_unknown: bool = False):
    """Filter records; optionally keep missing non-scope metadata with uncertainty labels."""
    tree = parse_query(query)
    kept, rejected = [], {}
    year_range = None
    if year:
        match = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", year)
        if not match:
            raise ValueError("--year expects YYYY or YYYY-YYYY")
        year_range = (int(match.group(1)), int(match.group(2) or match.group(1)))
        if year_range[0] > year_range[1]:
            raise ValueError("--year start must not exceed end")
    if from_year is not None and to_year is not None and from_year > to_year:
        raise ValueError("--from-year must not exceed --to-year")

    for record in records:
        reasons = []
        pyear = _paper_year(record)
        if year_range:
            if pyear is None:
                reasons.append("year_unknown")
            elif not year_range[0] <= pyear <= year_range[1]:
                reasons.append("year")
        if from_year is not None:
            if pyear is None:
                reasons.append("from-year_unknown")
            elif pyear < from_year:
                reasons.append("from-year")
        if to_year is not None:
            if pyear is None:
                reasons.append("to-year_unknown")
            elif pyear > to_year:
                reasons.append("to-year")
        if author:
            if not record.authors:
                reasons.append("author_unknown")
            elif not normalized_text_contains(record.authors, author):
                reasons.append("author")
        if sites:
            from urllib.parse import urlparse
            from .query_language import _site_matches
            if not urlparse(record.url or "").hostname:
                reasons.append("site_unknown")
            elif not any(_site_matches(record, site) for site in sites):
                reasons.append("site")
        if types:
            if not record.kind or record.kind.casefold() == "unknown":
                reasons.append("type_unknown")
            elif record.kind.casefold() not in {value.casefold() for value in types}:
                reasons.append("type")
        record_sources = record.sources or [record.source]
        normalize_source = lambda value: re.sub(r"[^a-z0-9]", "", value.casefold())
        if sources and not any(normalize_source(source) == normalize_source(selected)
                               for source in record_sources for selected in sources):
            reasons.append("source_unknown" if not any(record_sources) else "source")
        if min_citations is not None and (record.citations is None or record.citations < min_citations):
            reasons.append("citations_unknown" if record.citations is None else "citations")
        if open_access and record.open_access is not True:
            reasons.append("open_access_unknown" if record.open_access is None else "open_access")
        if language and (not record.language or record.language.casefold() != language.casefold()):
            reasons.append("language_unknown" if not record.language else "language")
        query_match = matches_query(record, tree, include_unknown_fields=include_unknown)
        query_used_unknown = (
            include_unknown and query_match is True and matches_query(record, tree) is not True
        )
        if query_match is None:
            reasons.append("query_unknown")
        elif not query_match:
            reasons.append("query")

        unknown_labels = {
            "year_unknown": "year", "from-year_unknown": "from_year", "to-year_unknown": "to_year",
            "author_unknown": "author", "type_unknown": "type", "citations_unknown": "citations",
            "open_access_unknown": "open_access", "language_unknown": "language",
        }
        uncertainties = []
        if include_unknown:
            uncertainties.extend(unknown_labels[reason] for reason in reasons if reason in unknown_labels)
            if query_used_unknown:
                uncertainties.extend(f"query:{field}" for field in unknown_query_fields(record, tree)
                                     if field not in {"site", "source"})
            reasons = [reason for reason in reasons if reason not in unknown_labels]
        if reasons:
            for reason in reasons:
                rejected[reason] = rejected.get(reason, 0) + 1
        else:
            record.filter_uncertainties = list(dict.fromkeys(uncertainties))
            kept.append(record)
    return kept, rejected

def get_top_influential(results: List[Paper], n: int = 5) -> List[Paper]:
    cited = [r for r in results if isinstance(r.citations, (int, float)) and r.citations > 0]
    return sorted(cited, key=lambda r: r.citations or 0, reverse=True)[:n]

def build_citation_links(results: List[Paper]) -> List[Paper]:
    for r in results:
        r.cites_in_results.clear()
        r.cited_by_in_results.clear()
    id_to_result = {r.openalex_id: r for r in results if r.openalex_id}
    for r in results:
        refs = r.referenced_works or []
        for ref_id in refs:
            target = id_to_result.get(ref_id)
            if target and target is not r:
                r.cites_in_results.append(target.title)
                target.cited_by_in_results.append(r.title)
    return results

def summarize_abstract(abstract: str, query: str, max_sentences: int = 2) -> str:
    if not abstract:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(abstract) if s.strip()]
    if len(sentences) <= max_sentences:
        return abstract

    terms = query_terms(query)
    scored = []
    for i, s in enumerate(sentences):
        low = s.lower()
        term_score = sum(low.count(t) for t in terms)
        position_bonus = 1.0 if i == 0 else 0.0
        length_penalty = 0.0 if 40 <= len(s) <= 300 else 0.5
        scored.append((term_score + position_bonus - length_penalty, i, s))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sorted_by_position = sorted(top, key=lambda x: x[1])
    return " ".join(s for _, _, s in top_sorted_by_position)

def extract_matching_context(text: str, query: str, context_chars: int = 180, max_matches: int = 3) -> List[str]:
    terms = query_terms(query)
    if not terms or not text:
        return []
    patterns = [(_identifier_pattern(term).pattern if _identifier_pattern(term) else re.escape(term))
                for term in terms]
    pattern = re.compile("|".join(patterns), re.IGNORECASE)

    matches = []
    for m in pattern.finditer(text):
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        snippet = text[start:end].strip()
        if snippet not in matches:
            matches.append(snippet)
        if len(matches) >= max_matches:
            break
    return matches
