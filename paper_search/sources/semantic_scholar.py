import urllib.parse
import json
from typing import List

from ..network import http_get
from ..models import Paper
from ..utils import query_terms

def search_semantic_scholar(query: str, max_results: int = 15, pages: int = 1) -> List[Paper]:
    api_query = " ".join(query_terms(query))
    base = "https://api.semanticscholar.org/graph/v1/paper/search"

    results = []
    for page in range(pages):
        params = {
            "query": api_query,
            "limit": max_results,
            "offset": page * max_results,
            "fields": "title,abstract,year,authors,url,publicationDate,citationCount",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        raw = http_get(url)
        data = json.loads(raw)

        items = data.get("data", [])
        if not items:
            break

        for p in items:
            authors_list = [a.get("name", "") for a in p.get("authors", [])]
            abstract = p.get("abstract") or ""
            results.append(Paper(
                source="Semantic Scholar",
                title=p.get("title", ""),
                authors=", ".join(authors_list),
                date=p.get("publicationDate") or str(p.get("year", "")),
                url=p.get("url", ""),
                abstract=abstract[:400] + ("..." if len(abstract) > 400 else ""),
                citations=p.get("citationCount"),
            ))
    return results
