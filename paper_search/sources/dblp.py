import urllib.parse
import json
from typing import List

from ..network import http_get
from ..models import Paper
from ..utils import query_terms

def search_dblp(query: str, max_results: int = 15, pages: int = 1) -> List[Paper]:
    api_query = " ".join(query_terms(query))
    base = "https://dblp.org/search/publ/api"

    results = []
    for page in range(pages):
        params = {
            "q": api_query,
            "h": max_results,
            "f": page * max_results,
            "format": "json"
        }
        url = base + "?" + urllib.parse.urlencode(params)
        try:
            raw = http_get(url)
            data = json.loads(raw)
        except Exception:
            break

        items = data.get("result", {}).get("hits", {}).get("hit", [])
        if not items:
            break

        for item in items:
            info = item.get("info", {})
            title = info.get("title", "")

            authors_data = info.get("authors", {}).get("author", [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]
            elif isinstance(authors_data, str):
                authors_data = [{"text": authors_data}]

            authors_list = [a.get("text", "") for a in authors_data if isinstance(a, dict)]

            year = info.get("year", "")
            url = info.get("ee", "") or info.get("url", "")
            doi = info.get("doi", "")

            results.append(Paper(
                source="DBLP",
                title=title,
                authors=", ".join(authors_list),
                date=year,
                url=url,
                abstract="", # DBLP usually doesn't provide abstracts in search
                doi=doi,
            ))
    return results
