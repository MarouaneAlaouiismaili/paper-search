import urllib.parse
import xml.etree.ElementTree as ET
from typing import List

from ..network import http_get
from ..models import Paper
from ..utils import query_terms

ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

def search_arxiv(query: str, max_results: int = 15, pages: int = 1) -> List[Paper]:
    # Use only MUST and SHOULD terms for the API query
    api_query = " ".join(query_terms(query))
    base = "http://export.arxiv.org/api/query"

    results = []
    for page in range(pages):
        params = {
            "search_query": f"all:{api_query}",
            "start": page * max_results,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        raw = http_get(url)
        root = ET.fromstring(raw)

        entries = root.findall("atom:entry", ARXIV_NS)
        if not entries:
            break

        for entry in entries:
            title = entry.findtext("atom:title", default="", namespaces=ARXIV_NS).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", default="", namespaces=ARXIV_NS).strip().replace("\n", " ")
            link = entry.findtext("atom:id", default="", namespaces=ARXIV_NS).strip()
            published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)[:10]
            authors_list = [a.findtext("atom:name", default="", namespaces=ARXIV_NS)
                       for a in entry.findall("atom:author", ARXIV_NS)]

            results.append(Paper(
                source="arXiv",
                title=title,
                authors=", ".join(authors_list),
                date=published,
                url=link,
                abstract=summary[:400] + ("..." if len(summary) > 400 else ""),
            ))
    return results
