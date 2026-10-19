from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse


def infer_record_kind(source: str, title: str, url: str, abstract: str) -> str:
    source_key = (source or "").casefold()
    if source_key in {"arxiv", "semantic scholar", "openalex", "crossref", "pubmed", "dblp"}:
        return "paper"
    host = (urlparse(url or "").hostname or "").casefold().removeprefix("www.")
    text = f"{title} {abstract}".casefold()
    if any(domain == host or host.endswith("." + domain) for domain in
           ("instagram.com", "facebook.com", "youtube.com", "tiktok.com", "x.com", "twitter.com")):
        return "social"
    if any(word in text for word in ("research works", "publications", "journal article", "doi.org")):
        return "paper"
    if any(word in text for word in ("profile", "biography", "professor", "researcher", "professional named", "chercheur")):
        return "person"
    if any(word in text for word in ("university", "université", "company", "organization", "association", "organisation")):
        return "organization"
    if any(word in text for word in ("news", "breaking", "press release", "journalism")):
        return "news"
    return "website"

@dataclass
class EvidenceRecord:
    source: str
    title: str
    url: str
    authors: str = ""
    date: str = ""
    abstract: str = ""
    citations: Optional[int] = None
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    referenced_works: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    deep_context: List[str] = field(default_factory=list)
    match_count: int = 0
    query_match: str = ""
    query_terms_matched: int = 0
    query_terms_total: int = 0
    filter_uncertainties: List[str] = field(default_factory=list)
    cites_in_results: List[str] = field(default_factory=list)
    cited_by_in_results: List[str] = field(default_factory=list)
    score: float = 0.0
    smart_summary: str = ""
    kind: str = "unknown"
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    open_access: Optional[bool] = None
    language: Optional[str] = None
    source_ranks: Dict[str, int] = field(default_factory=dict)
    field_evidence: Dict[str, List[dict]] = field(default_factory=dict)
    bm25_score: float = 0.0
    discovery_methods: List[str] = field(default_factory=list)
    discovered_from: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.kind == "unknown":
            self.kind = infer_record_kind(self.source, self.title, self.url, self.abstract)
        for name in ("title", "authors", "date", "url", "abstract", "citations", "doi", "open_access", "language"):
            value = getattr(self, name)
            if value is not None and value != "":
                self.field_evidence.setdefault(name, []).append({"source": self.source, "value": value})

    def to_dict(self) -> dict:
        """تحويل الكائن إلى قاموس (لتسهيل التصدير لاحقاً)"""
        d = {
            "source": self.source,
            "kind": self.kind,
            "title": self.title,
            "authors": self.authors,
            "date": self.date,
            "url": self.url,
            "abstract": self.abstract,
            "citations": self.citations,
            "doi": self.doi,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "retrieved_at": self.retrieved_at,
            "open_access": self.open_access,
            "language": self.language,
            "openalex_id": self.openalex_id,
            "referenced_works": self.referenced_works,
            "discovery_methods": list(self.discovery_methods),
            "discovered_from": list(self.discovered_from),
            "match_count": self.match_count,
            "query_match": self.query_match,
            "query_terms_matched": self.query_terms_matched,
            "query_terms_total": self.query_terms_total,
            "filter_uncertainties": list(self.filter_uncertainties),
            "smart_summary": self.smart_summary,
            "sources_count": len(self.sources) if self.sources else 1,
            "sources": " | ".join(self.sources) if self.sources else self.source,
        }
        if self.deep_context:
            d["deep_context"] = " | ".join(self.deep_context)
            d["deep_contexts"] = list(self.deep_context)
        if self.cites_in_results:
            d["cites_in_results"] = " | ".join(self.cites_in_results)
            d["cites_in_results_list"] = list(self.cites_in_results)
        if self.cited_by_in_results:
            d["cited_by_in_results"] = " | ".join(self.cited_by_in_results)
            d["cited_by_in_results_list"] = list(self.cited_by_in_results)
        d["source_names"] = list(self.sources) if self.sources else [self.source]
        if self.field_evidence:
            d["field_evidence"] = self.field_evidence
        if self.source_ranks:
            d["source_ranks"] = self.source_ranks

        return d


# Backwards-compatible name used by existing source adapters.
Paper = EvidenceRecord
