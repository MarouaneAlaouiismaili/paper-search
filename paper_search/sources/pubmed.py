import urllib.parse
import json
import xml.etree.ElementTree as ET
from typing import List

from ..network import http_get, polite_wait
from ..models import Paper
from ..utils import query_terms

def _fetch_pubmed_abstracts(ids: List[str]) -> dict:
    if not ids:
        return {}
    efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"}
    url = efetch + "?" + urllib.parse.urlencode(params)
    try:
        raw = http_get(url)
    except Exception:
        return {}

    abstracts = {}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}

    for article in root.findall(".//PubmedArticle"):
        pid_el = article.find(".//PMID")
        if pid_el is None or not pid_el.text:
            continue
        pid = pid_el.text.strip()
        parts = [
            (node.text or "").strip()
            for node in article.findall(".//Abstract/AbstractText")
            if node.text
        ]
        if parts:
            abstracts[pid] = " ".join(parts)
    return abstracts

def search_pubmed(query: str, max_results: int = 15, pages: int = 1, *,
                  from_year: int = None, to_year: int = None,
                  author: str = None) -> List[Paper]:
    api_query = " ".join(query_terms(query))
    if author:
        api_query = f"({api_query}) AND ({author}[Author])"
    if from_year is not None or to_year is not None:
        start = from_year or 1
        end = to_year or 3000
        api_query = f"({api_query}) AND ({start}:{end}[Date - Publication])"
    esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    results = []
    for page in range(pages):
        params = {
            "db": "pubmed",
            "term": api_query,
            "retmax": max_results,
            "retstart": page * max_results,
            "retmode": "json"
        }
        url = esearch + "?" + urllib.parse.urlencode(params)
        raw = http_get(url)
        ids = json.loads(raw).get("esearchresult", {}).get("idlist", [])
        if not ids:
            break

        polite_wait(0.4)
        esummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params2 = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        url2 = esummary + "?" + urllib.parse.urlencode(params2)
        raw2 = http_get(url2)
        summary = json.loads(raw2).get("result", {})

        polite_wait(0.4)
        abstracts = _fetch_pubmed_abstracts(ids)

        for pid in summary.get("uids", []):
            item = summary.get(pid, {})
            authors_list = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            abstract = abstracts.get(pid, "")
            results.append(Paper(
                source="PubMed",
                title=item.get("title", ""),
                authors=", ".join(authors_list),
                date=item.get("pubdate", ""),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                abstract=abstract[:400] + ("..." if len(abstract) > 400 else ""),
                citations=None,
            ))
    return results
