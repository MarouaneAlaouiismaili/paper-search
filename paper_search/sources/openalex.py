import urllib.parse
import json
import re
from typing import List

from ..network import http_get, polite_wait
from ..models import Paper
from ..utils import query_terms

OPENALEX_BASE = "https://api.openalex.org/works"
WORK_FIELDS = (
    "id,title,abstract_inverted_index,publication_date,authorships,"
    "primary_location,cited_by_count,referenced_works,doi,open_access"
)


def _rebuild_abstract(inv_index):
    if not inv_index:
        return ""
    positions = {}
    for word, indexes in inv_index.items():
        for index in indexes:
            positions[index] = word
    return " ".join(positions[index] for index in sorted(positions))


def _work_to_paper(work, *, discovery_method="", discovered_from=None):
    authors = [entry.get("author", {}).get("display_name", "")
               for entry in work.get("authorships", [])]
    abstract = _rebuild_abstract(work.get("abstract_inverted_index"))
    work_id = work.get("id") or ""
    landing_page = (work.get("primary_location") or {}).get("landing_page_url")
    record = Paper(
        source="OpenAlex",
        title=work.get("title") or "",
        authors=", ".join(name for name in authors if name),
        date=work.get("publication_date") or "",
        url=landing_page or work_id,
        abstract=abstract[:400] + ("..." if len(abstract) > 400 else ""),
        citations=work.get("cited_by_count"),
        open_access=(work.get("open_access") or {}).get("is_oa"),
        openalex_id=work_id,
        referenced_works=work.get("referenced_works") or [],
        doi=work.get("doi"),
    )
    if discovery_method:
        record.discovery_methods.append(discovery_method)
    if discovered_from:
        record.discovered_from.extend(dict.fromkeys(discovered_from))
    return record

def search_openalex(query: str, max_results: int = 15, pages: int = 1, *,
                    from_year: int = None, to_year: int = None,
                    author: str = None, min_citations: int = None,
                    open_access: bool = False) -> List[Paper]:
    api_query = " ".join(query_terms(query))
    base = OPENALEX_BASE

    results = []
    for page in range(1, pages + 1):
        params = {
            "search": api_query,
            "per_page": max_results,
            "page": page,
            "sort": "relevance_score:desc",
            "select": WORK_FIELDS,
        }
        filters = []
        if from_year is not None:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year is not None:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if author:
            filters.append("raw_author_name.search:" + author)
        if min_citations is not None:
            filters.append(f"cited_by_count:>={min_citations}")
        if open_access:
            filters.append("open_access.is_oa:true")
        if filters:
            params["filter"] = ",".join(filters)
        url = base + "?" + urllib.parse.urlencode(params)
        raw = http_get(url, headers={"User-Agent": "paper_search.py (mailto:research@example.com)"})
        data = json.loads(raw)

        items = data.get("results", [])
        if not items:
            break

        for w in items:
            results.append(_work_to_paper(w))
    return results


def expand_openalex_graph(seed_records: List[Paper], limit: int = 12):
    """Fetch one bounded citation-graph hop: references and works citing top seeds.

    Returns ``(records, errors)`` so an unavailable graph endpoint never blocks
    the normal search results. At most three API calls are made.
    """
    limit = min(max(0, limit), 20)
    if not limit:
        return [], []

    seeds = []
    for record in seed_records:
        raw_id = record.openalex_id or ""
        work_id = raw_id.rsplit("/", 1)[-1]
        if re.fullmatch(r"W\d+", work_id) and work_id not in {item[0] for item in seeds}:
            seeds.append((work_id, record.title))
        if len(seeds) >= 4:
            break
    if not seeds:
        return [], []

    errors = []
    expanded = []
    headers = {"User-Agent": "paper_search.py (mailto:research@example.com)"}

    def fetch(params, method, parents):
        try:
            polite_wait()
            url = OPENALEX_BASE + "?" + urllib.parse.urlencode(params)
            data = json.loads(http_get(url, headers=headers, timeout=8, retries=1))
            for work in data.get("results", []):
                record = _work_to_paper(work, discovery_method=method, discovered_from=parents)
                expanded.append(record)
                record.source_ranks["OpenAlex citation graph"] = len(expanded)
                if len(expanded) >= limit:
                    break
        except Exception as exc:
            errors.append(f"{method}: {exc}")

    reference_parents = {}
    for work_id, title in seeds:
        seed = next((record for record in seed_records
                     if (record.openalex_id or "").rsplit("/", 1)[-1] == work_id), None)
        for reference in (seed.referenced_works if seed else []) or []:
            ref_id = (reference or "").rsplit("/", 1)[-1]
            if re.fullmatch(r"W\d+", ref_id):
                reference_parents.setdefault(ref_id, []).append(title)

    if reference_parents and len(expanded) < limit:
        reference_ids = list(reference_parents)[:limit]
        fetch({
            "filter": "openalex:" + "|".join(reference_ids),
            "per_page": min(limit, len(reference_ids)),
            "select": WORK_FIELDS,
        }, "مراجع OpenAlex", list(dict.fromkeys(
            title for ref_id in reference_ids for title in reference_parents[ref_id]
        )))
        for record in expanded:
            work_id = (record.openalex_id or "").rsplit("/", 1)[-1]
            record.discovered_from = list(dict.fromkeys(reference_parents.get(work_id, record.discovered_from)))

    for work_id, title in seeds[:2]:
        if len(expanded) >= limit:
            break
        fetch({
            "filter": f"cites:{work_id}",
            "per_page": min(6, limit - len(expanded)),
            "select": WORK_FIELDS,
        }, "عمل يستشهد بسجل OpenAlex", [title])

    return expanded[:limit], errors
