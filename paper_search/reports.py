"""Evidence report serialization with field-level provenance and conflict notes."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .models import EvidenceRecord


def _conflicts(records: List[EvidenceRecord]) -> List[dict]:
    conflicts = []
    for record in records:
        for field_name, evidence in record.field_evidence.items():
            values = []
            for entry in evidence:
                value = entry.get("value")
                if value is None or value == "":
                    continue
                normalized = str(value).strip().casefold()
                if normalized not in [item[0] for item in values]:
                    values.append((normalized, {"value": value, "sources": [entry.get("source", "unknown")] }))
                else:
                    for key, item in values:
                        if key == normalized and entry.get("source") not in item["sources"]:
                            item["sources"].append(entry.get("source", "unknown"))
            if len(values) > 1:
                conflicts.append({
                    "record": record.title,
                    "url": record.url,
                    "field": field_name,
                    "values": [item for _, item in values],
                })
    return conflicts


def build_report(records: List[EvidenceRecord], query: str, *, filters: Dict = None,
                 filter_rejections: Dict = None, failed_sources: List = None) -> dict:
    missing = {}
    for field_name in ("date", "authors", "citations", "doi", "open_access", "language", "abstract"):
        missing[field_name] = sum(getattr(record, field_name) is None or getattr(record, field_name) == ""
                                  for record in records)
    uncertainty_counts = Counter(
        label for record in records for label in getattr(record, "filter_uncertainties", [])
    )
    return {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "filters": filters or {},
        "source_failures": [{"source": name, "error": error} for name, error in (failed_sources or [])],
        "filter_rejections": filter_rejections or {},
        "filter_uncertainties": dict(sorted(uncertainty_counts.items())),
        "result_count": len(records),
        "inferences": [],
        "inference_policy": "This report preserves source-reported records and does not infer identity or merge people by name.",
        "missing_metadata_counts": missing,
        "conflicts": _conflicts(records),
        "results": [record.to_dict() for record in records],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Evidence report",
        "",
        "> Source-reported records are evidence for review, not independent verification. No identity conclusions are inferred.",
        "",
        f"- Query: {report['query']}",
        f"- Generated: {report['generated_at']}",
        f"- Results: {report['result_count']}",
        "",
        "## Search and filtering",
        "",
        f"- Filters: `{json.dumps(report['filters'], ensure_ascii=False, sort_keys=True)}`",
    ]
    if report["filter_rejections"]:
        lines.append(f"- Excluded counts: `{json.dumps(report['filter_rejections'], ensure_ascii=False, sort_keys=True)}`")
    if report.get("filter_uncertainties"):
        lines.append(f"- Included results with unknown filter metadata: `{json.dumps(report['filter_uncertainties'], ensure_ascii=False, sort_keys=True)}`")
    if report["source_failures"]:
        lines.extend(["", "## Source failures", ""])
        for failure in report["source_failures"]:
            lines.append(f"- **{failure['source']}**: {failure['error']}")
    lines.extend(["", "## Source-reported evidence", ""])
    if not report["results"]:
        lines.append("No records matched the filters.")
    for index, result in enumerate(report["results"], 1):
        lines.extend([
            f"### [{index}] {result['title']}",
            "",
            f"- Type: {result['kind']}",
            f"- Source(s): {result['sources']}",
            f"- Published: {result['date'] or 'unknown'}; collected: {result['retrieved_at']}",
            f"- URL: {result['url']}",
        ])
        if result.get("filter_uncertainties"):
            lines.append(f"- ⚠️ Filter metadata missing: {', '.join(result['filter_uncertainties'])}")
        if result.get("discovery_methods"):
            lines.append(f"- Discovery path: {', '.join(result['discovery_methods'])}")
        if result.get("discovered_from"):
            lines.append(f"- Linked from: {'; '.join(result['discovered_from'])}")
        for field_name, label in (("authors", "Authors"), ("doi", "DOI"), ("citations", "Citations"),
                                  ("open_access", "Open access"), ("language", "Language")):
            if result.get(field_name) not in (None, ""):
                lines.append(f"- {label}: {result[field_name]}")
        if result.get("abstract"):
            lines.extend(["", "**Reported abstract / search snippet:**", "", result["abstract"]])
        for context in result.get("deep_contexts") or ([result.get("deep_context")] if result.get("deep_context") else []):
            lines.extend(["", f"> {context}"])
        lines.extend(["", "**Field provenance:**", ""])
        for field_name, evidence in result.get("field_evidence", {}).items():
            for entry in evidence:
                lines.append(f"- `{field_name}` from {entry['source']}: {entry['value']}")
        lines.append("")
    lines.extend(["## Inferences", "", "No independent identity or factual conclusions are generated by this report.", "",
                  "## Conflicts", ""])
    if not report["conflicts"]:
        lines.append("No conflicting field values were detected among merged records.")
    else:
        for conflict in report["conflicts"]:
            values = "; ".join(f"{item['value']} ({', '.join(item['sources'])})" for item in conflict["values"])
            lines.append(f"- **{conflict['record']} — {conflict['field']}**: {values}")
    lines.extend(["", "## Missing metadata counts", ""])
    for key, value in report["missing_metadata_counts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(path: str, records: List[EvidenceRecord], query: str, *, filters: Dict = None,
                 filter_rejections: Dict = None, failed_sources: List = None) -> dict:
    report = build_report(records, query, filters=filters,
                          filter_rejections=filter_rejections, failed_sources=failed_sources)
    target = Path(path)
    if target.suffix.casefold() == ".json":
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    elif target.suffix.casefold() == ".md":
        target.write_text(render_markdown(report), encoding="utf-8")
    else:
        raise ValueError("Report path must end with .md or .json")
    return report
