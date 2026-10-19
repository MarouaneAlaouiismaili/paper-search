import urllib.parse
import json
from typing import List

from ..network import http_get
from ..utils import clean_html, query_terms
from ..models import Paper

def search_crossref(query: str, max_results: int = 15, pages: int = 1, *,
                    from_year: int = None, to_year: int = None,
                    author: str = None) -> List[Paper]:
    api_query = " ".join(query_terms(query))
    base = "https://api.crossref.org/works"

    results = []
    for page in range(pages):
        params = {
            "query": api_query,
            "rows": max_results,
            "offset": page * max_results,
            "select": "title,author,published-print,published-online,URL,abstract,"
                       "is-referenced-by-count,DOI",
        }
        filters = []
        if from_year is not None:
            filters.append(f"from-pub-date:{from_year}-01-01")
        if to_year is not None:
            filters.append(f"until-pub-date:{to_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if author:
            params["query.author"] = author
        url = base + "?" + urllib.parse.urlencode(params)
        raw = http_get(url, headers={"User-Agent": "paper_search.py (mailto:research@example.com)"})
        data = json.loads(raw)

        items = data.get("message", {}).get("items", [])
        if not items:
            break

        for item in items:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else ""
            authors_list = []
            for a in item.get("author", []) or []:
                name = " ".join(filter(None, [a.get("given"), a.get("family")]))
                if name:
                    authors_list.append(name)
            date_parts = (item.get("published-print") or item.get("published-online") or {}).get("date-parts")
            date = "-".join(str(p) for p in date_parts[0]) if date_parts and date_parts[0] else ""
            abstract = clean_html(item.get("abstract", "") or "")
            results.append(Paper(
                source="CrossRef",
                title=title,
                authors=", ".join(authors_list),
                date=date,
                url=item.get("URL", ""),
                abstract=abstract[:400] + ("..." if len(abstract) > 400 else ""),
                citations=item.get("is-referenced-by-count"),
                doi=item.get("DOI"),
            ))
    return results
