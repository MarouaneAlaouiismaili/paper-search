import argparse
import concurrent.futures
import csv
import json
import re
import sys
import ipaddress
import socket
from html.parser import HTMLParser
from datetime import datetime, timezone
from typing import List
from urllib.parse import unquote, urlparse

from .config import global_config
from .models import Paper
from .llm import (
    DEFAULT_MODEL, MAX_EVIDENCE_PAPERS, build_evidence, select_informative_results, stream_answer
)
from .network import http_get, polite_wait
from .utils import (
    dedupe, sort_results, filter_records, combine_relevance_scores,
    get_top_influential, build_citation_links, calculate_bm25_scores,
    summarize_abstract, extract_matching_context, query_terms
)
from .sources import (
    search_arxiv, search_semantic_scholar, search_openalex,
    search_crossref, search_pubmed, search_duckduckgo, search_dblp,
    search_google, google_search_configured, search_brave, brave_search_configured,
)
from .sources.openalex import expand_openalex_graph
from .sources.google_vision import ImageSearchError, MAX_IMAGE_RESULTS, search_image
from .query_language import (
    QuerySyntaxError, normalize_search_text, parse_query, source_query, source_query_variants,
)
from .reports import write_report

SOURCE_FUNCTIONS = {
    "arxiv": ("arXiv", search_arxiv),
    "semanticscholar": ("Semantic Scholar", search_semantic_scholar),
    "openalex": ("OpenAlex", search_openalex),
    "crossref": ("CrossRef", search_crossref),
    "pubmed": ("PubMed", search_pubmed),
    "dblp": ("DBLP", search_dblp),
    "duckduckgo": ("DuckDuckGo", search_duckduckgo),
    "google": ("Google", search_google),
    "brave": ("Brave Search", search_brave),
}
ACADEMIC_SOURCE_KEYS = tuple(key for key in SOURCE_FUNCTIONS if key not in {"duckduckgo", "google", "brave"})
RECORD_TYPES = ("paper", "person", "organization", "website", "news", "social", "other")
PUBLIC_WEB_SOURCES = {"DuckDuckGo", "Google", "Brave Search"}
MAX_RESULTS_PER_PAGE = 100
MAX_SEARCH_PAGES = 10
MAX_DEEP_PAGES = 25
MAX_PLANNED_SEARCH_REQUESTS = 200


class _ReadableTextParser(HTMLParser):
    """Extract article-like text while dropping common page chrome."""

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre", "dd"}
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    NOISE_TAGS = {"nav", "footer", "aside", "form", "script", "style", "noscript", "svg", "iframe", "button"}
    NOISE_MARKERS = (
        "navigation", "navbar", "sidebar", "cookie", "consent", "breadcrumb", "pagination",
        "recommend", "related", "share", "social-links", "advert", "promo", "modal", "overlay",
    )

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.main_seen = False
        self.blocks = []
        self.fallback = []
        self.current = []
        self.current_link_chars = 0
        self.current_main = False

    def _flush(self):
        text = re.sub(r"\s+", " ", " ".join(self.current)).strip()
        if text and (len(text) >= 35 or text.endswith((".", "!", "?", ":"))):
            link_ratio = self.current_link_chars / max(1, len(text))
            if link_ratio < 0.55 or len(text) >= 500:
                self.blocks.append((self.current_main, text))
        self.current = []
        self.current_link_chars = 0
        self.current_main = False

    def handle_starttag(self, tag, attrs):
        attrs = {key.casefold(): value or "" for key, value in attrs}
        parent_excluded = self.stack[-1][1] if self.stack else False
        parent_main = self.stack[-1][2] if self.stack else False
        label = f"{attrs.get('id', '')} {attrs.get('class', '')}".casefold()
        role = attrs.get("role", "").casefold()
        hidden = "hidden" in attrs or attrs.get("aria-hidden", "").casefold() == "true"
        hidden = hidden or bool(re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", attrs.get("style", ""), re.I))
        noisy_label = any(marker in label for marker in self.NOISE_MARKERS)
        excluded = parent_excluded or tag in self.NOISE_TAGS or hidden or noisy_label
        if tag == "header" and not parent_main:
            excluded = True
        main = parent_main or tag in {"main", "article"} or role == "main"
        if tag in {"main", "article"} or role == "main":
            self.main_seen = True
        if tag in self.BLOCK_TAGS:
            self._flush()
            self.current_main = main and not excluded
        if tag == "br":
            self._flush()
        if tag not in self.VOID_TAGS:
            self.stack.append((tag, excluded, main))

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS or tag == "br":
            self._flush()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not data or (self.stack and self.stack[-1][1]):
            return
        self.fallback.append(data)
        if self.current or any(item[0] in self.BLOCK_TAGS for item in self.stack):
            self.current.append(data)
            if any(item[0] == "a" for item in self.stack):
                self.current_link_chars += len(data)

    def text(self, html_text: str) -> str:
        self.feed(html_text)
        self.close()
        self._flush()
        main_blocks = [text for is_main, text in self.blocks if is_main]
        chosen = main_blocks if main_blocks else [text for _, text in self.blocks]
        if chosen:
            seen = set()
            chosen = [text for text in chosen if not (text.casefold() in seen or seen.add(text.casefold()))]
            result = " ".join(chosen)
        else:
            result = " ".join(self.fallback)
        return re.sub(r"\s+", " ", result).strip()[:100_000]


def _selected_source_keys(mode: str, source_filters: List[str]) -> List[str]:
    available = [key for key in SOURCE_FUNCTIONS
                 if (key != "google" or google_search_configured())
                 and (key != "brave" or brave_search_configured())]
    if source_filters:
        selected = []
        for value in source_filters:
            keys = available if value == "all" else ACADEMIC_SOURCE_KEYS if value == "academic" else (value,)
            for key in keys:
                if key not in selected:
                    selected.append(key)
        return selected
    if mode == "academic":
        return list(ACADEMIC_SOURCE_KEYS)
    if mode == "web":
        return (["google"] if google_search_configured() else []) + (["brave"] if brave_search_configured() else []) + ["duckduckgo"]
    return available


def _positive_field_values(node, field: str, negated: bool = False) -> List[str]:
    if node is None:
        return []
    if node.kind == "NOT":
        return _positive_field_values(node.children[0], field, not negated)
    if node.kind == "TERM":
        return [node.value] if not negated and node.field == field else []
    values = []
    for child in node.children:
        values.extend(_positive_field_values(child, field, negated))
    return values


def _plain_text_terms(node) -> List[str]:
    """Return terms eligible for recall-oriented fallback; leave structured queries strict."""
    if node is None:
        return []
    if node.kind == "TERM":
        if node.field is not None or not re.fullmatch(r"[\w]+", node.value, flags=re.UNICODE):
            return []
        return [node.value.casefold()]
    if node.kind != "AND":
        return []
    terms = []
    for child in node.children:
        child_terms = _plain_text_terms(child)
        if not child_terms:
            return []
        terms.extend(child_terms)
    return list(dict.fromkeys(terms))


def _partial_text_matches(records, exact_results, terms):
    if not terms:
        return []
    exact_ids = {id(record) for record in exact_results}
    minimum_matches = 1 if len(terms) == 2 else (2 * len(terms) + 2) // 3
    normalized_terms = [normalize_search_text(term) for term in terms]
    partial = []
    for record in records:
        if id(record) in exact_ids:
            continue
        text = normalize_search_text(" ".join((
            record.title or "", record.abstract or "", record.authors or "", record.url or "",
            unquote(record.url or ""), record.doi or "", " ".join(record.deep_context or []),
        )))
        matched = sum(bool(term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.UNICODE))
                      for term in normalized_terms)
        if matched >= minimum_matches:
            record.query_match = "partial"
            record.query_terms_matched = matched
            record.query_terms_total = len(terms)
            partial.append(record)
    return partial

def is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return False
    host = parsed.hostname.casefold().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
        return False
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))]
        except (OSError, ValueError):
            return False
    return bool(addresses) and all(address.is_global for address in addresses)


def fetch_page_text(url: str, timeout: int = 8, max_bytes: int = 1_048_576) -> str:
    try:
        if not is_public_http_url(url):
            return ""
        raw = http_get(url, timeout=timeout, max_bytes=max_bytes, allow_redirects=False, retries=1)
        if raw.startswith(b"%PDF-") or b"\x00" in raw[:4096]:
            return ""
        text = raw.decode("utf-8", errors="ignore")
        return _ReadableTextParser().text(text)
    except Exception:
        return ""

def enrich_with_deep_content(results: List[Paper], query: str, max_pages: int = 10):
    max_pages = min(max(0, max_pages), MAX_DEEP_PAGES)
    ddg_targets = [r for r in results if any(source in PUBLIC_WEB_SOURCES
                   for source in (r.sources or [r.source])) and is_public_http_url(r.url)][:max_pages]
    if not ddg_targets:
        return results

    def _process_item(r: Paper):
        text = fetch_page_text(r.url, timeout=8, max_bytes=1_048_576)
        if text:
            contexts = extract_matching_context(text, query)
            if contexts:
                r.deep_context = contexts
                r.match_count = sum(len(re.findall(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text, flags=re.IGNORECASE))
                                    for t in query_terms(query))
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(_process_item, ddg_targets))

    return results

def export_results(results: List[Paper], path: str):
    flat = [r.to_dict() for r in results]
    if path.casefold().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)
    elif path.casefold().endswith(".csv"):
        if not flat:
            fieldnames = list(Paper("", "", "").to_dict().keys())
        else:
            fieldnames = list(dict.fromkeys(key for row in flat for key in row))
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            rows = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                     for key, value in row.items()} for row in flat]
            writer.writerows(rows)
    else:
        raise ValueError("امتداد الملف يجب أن يكون .csv أو .json")


def _image_match_rows(report: dict) -> List[dict]:
    rows = []
    for category, key in (
        ("full_match", "full_matches"),
        ("partial_match", "partial_matches"),
        ("visually_similar", "visually_similar_images"),
    ):
        for item in report.get(key, []):
            rows.append({
                "category": category,
                "url": item.get("url", ""),
                "page_url": "",
                "page_title": "",
                "score": item.get("score", ""),
            })
    for page in report.get("pages_with_matches", []):
        page_url = page.get("url", "")
        page_title = page.get("title", "")
        page_matches = [
            ("full_match_on_page", item) for item in page.get("full_matches", [])
        ] + [
            ("partial_match_on_page", item) for item in page.get("partial_matches", [])
        ]
        if not page_matches:
            page_matches = [("page_with_matching_image", {"url": page_url})]
        for category, item in page_matches:
            rows.append({
                "category": category,
                "url": item.get("url", page_url),
                "page_url": page_url,
                "page_title": page_title,
                "score": item.get("score", ""),
            })
    return rows


def export_image_results(report: dict, path: str):
    if path.casefold().endswith(".json"):
        with open(path, "w", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
        return
    if path.casefold().endswith(".csv"):
        fields = ("category", "url", "page_url", "page_title", "score")
        with open(path, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(_image_match_rows(report))
        return
    raise ValueError("Image export path must end with .csv or .json")


def write_image_report(report: dict, path: str):
    if path.casefold().endswith(".json"):
        export_image_results(report, path)
        return
    lines = [
        "# Reverse image search",
        "",
        f"- Provider: {report['provider']}",
        f"- Searched at: {report['searched_at']}",
        f"- Full matches: {len(report['full_matches'])}",
        f"- Partial matches: {len(report['partial_matches'])}",
        f"- Pages with matching images: {len(report['pages_with_matches'])}",
        f"- Visually similar images: {len(report['visually_similar_images'])}",
        "",
        "> Results reflect the provider's indexed web references. No match does not prove that the image has never appeared online.",
    ]
    for heading, key in (
        ("Full image matches", "full_matches"),
        ("Partial image matches", "partial_matches"),
        ("Pages with matching images", "pages_with_matches"),
        ("Visually similar images", "visually_similar_images"),
    ):
        lines.extend(["", f"## {heading}", ""])
        items = report.get(key, [])
        if not items:
            lines.append("No results.")
            continue
        for item in items:
            if key == "pages_with_matches":
                lines.append(f"- [{item.get('title') or item['url']}]({item['url']})")
                for image in item.get("full_matches", []) + item.get("partial_matches", []):
                    lines.append(f"  - Matching image: {image['url']}")
            else:
                suffix = f" (score: {item['score']})" if "score" in item else ""
                lines.append(f"- {item['url']}{suffix}")
    lines.extend(["", "## Best-guess labels", ""])
    lines.extend(f"- {item['label']}" for item in report.get("best_guess_labels", []))
    if not report.get("best_guess_labels"):
        lines.append("No labels returned.")
    with open(path, "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def print_image_results(report: dict):
    full = report["full_matches"]
    partial = report["partial_matches"]
    pages = report["pages_with_matches"]
    similar = report["visually_similar_images"]
    if full:
        print(f"✅ وجد Google Vision {len(full)} صورة مطابقة كاملة.")
    elif partial or pages:
        print("لم تظهر مطابقة كاملة، لكن ظهرت نتائج جزئية أو صفحات تحتوي صورًا مطابقة.")
    elif similar:
        print("لم تظهر مطابقة مباشرة، لكن ظهرت صور مشابهة بصريًا.")
    else:
        print("لم يعثر فهرس Google Vision على صور مطابقة.")
    print(f"   مطابقة كاملة: {len(full)} | جزئية: {len(partial)} | صفحات: {len(pages)} | مشابهة بصريًا: {len(similar)}")
    for heading, key in (
        ("صور مطابقة كاملة", "full_matches"),
        ("صور مطابقة جزئيًا", "partial_matches"),
        ("صفحات تحتوي الصور", "pages_with_matches"),
        ("صور مشابهة بصريًا", "visually_similar_images"),
    ):
        items = report.get(key, [])
        if not items:
            continue
        print(f"\n{heading}:")
        for item in items:
            label = item.get("title") or item.get("url", "")
            print(f"  - {label}")
            if item.get("title"):
                print(f"    {item['url']}")
    if report.get("best_guess_labels"):
        print("\nتخمينات وصف الصورة:")
        print("  " + ", ".join(item["label"] for item in report["best_guess_labels"]))
    print("\nتنبيه: النتائج تعتمد على صفحات الويب التي فهرسها مزود البحث، ولا يثبت غياب النتيجة أن الصورة لم تُنشر قط.")

def print_top_influential(top_papers: List[Paper]):
    if not top_papers:
        return
    print("\n" + "★" * 80)
    print("🏆 أكثر الأبحاث تأثيرًا فهاد النتائج (بعدد الاستشهادات):")
    for i, r in enumerate(top_papers, 1):
        sources = r.sources or [r.source]
        label = " + ".join(sources)
        print(f"   {i}. {r.title}  —  {r.citations} استشهاد  ({label})")
    print("★" * 80)

def print_results(results: List[Paper], query: str):
    print(f"\n🔎 النتائج المرتبطة بـ: \"{query}\"  —  ({len(results)} نتيجة)\n")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.title}")
        sources = r.sources or [r.source]
        source_label = " + ".join(sources) if len(sources) > 1 else sources[0]
        line = f"    🧭 النوع: {r.kind}   📚 المصدر: {source_label}   📅 التاريخ: {r.date or 'غير معروف'}"
        if len(sources) > 1:
            line += f"   🔎 ظهر في {len(sources)} مصادر"
        if isinstance(r.citations, (int, float)):
            line += f"   📈 استشهادات: {int(r.citations)}"
        line += f"   📊 نقطة الصلة المدمجة: {r.score}"
        print(line)
        if r.authors:
            print(f"    ✍️  الكتّاب: {r.authors}")
        print(f"    🔗 {r.url}")
        if r.doi:
            print(f"    🆔 DOI: {r.doi}")
        if r.match_count:
            print(f"    🎯 عدد مرات ظهور الكلمة فالصفحة: {r.match_count}")
        if r.query_match == "partial":
            print(f"    🔎 تطابق جزئي: {r.query_terms_matched}/{r.query_terms_total} من كلمات البحث؛ راجع المصدر قبل اعتماده.")
        if r.filter_uncertainties:
            print(f"    ⚠️ بيانات فلتر غير متوفرة: {', '.join(r.filter_uncertainties)}؛ النتيجة تحتاج مراجعة.")
        if r.smart_summary:
            print(f"    ✨ الخلاصة الذكية: {r.smart_summary}")
        elif r.abstract:
            print(f"    📝 {r.abstract}")
        if r.cites_in_results:
            print(f"    ➡️  كيقتبس (داخل النتائج) من: {', '.join(r.cites_in_results[:3])}")
        if r.cited_by_in_results:
            print(f"    ⬅️  مقتبَس منه (داخل النتائج) من طرف: {', '.join(r.cited_by_in_results[:3])}")
        if r.deep_context:
            print(f"    📖 سياق مباشر من الصفحة:")
            for ctx in r.deep_context:
                print(f"       ...{ctx}...")
        if r.discovery_methods:
            print(f"    🧭 طريقة الاكتشاف: {', '.join(r.discovery_methods)}")
        if r.discovered_from:
            print(f"    🔗 مرتبط بسجل سابق: {'; '.join(r.discovered_from[:3])}")
    print("\n" + "=" * 80)

def main(argv=None):
    parser = argparse.ArgumentParser(description="بحث وتحقيق رقمي في المصادر العامة مع فلاتر وأدلة قابلة للمراجعة.")
    parser.add_argument("query", nargs="?", help='استعلام يدعم الأقواس وAND/OR/NOT والحقول، مثال: ("climate change" OR warming) AND year:2020..2025')
    parser.add_argument("--image", help="مسار صورة محلية للبحث العكسي عن الصور على الويب")
    parser.add_argument("--image-max-results", type=int, default=10,
                        help=f"أقصى النتائج لكل فئة في البحث العكسي (1-{MAX_IMAGE_RESULTS}; الافتراضي 10)")
    parser.add_argument("-n", "--num", type=int, default=20,
                        help=f"عدد النتائج لكل صفحة وموقع (1-{MAX_RESULTS_PER_PAGE}; الافتراضي: 20)")
    parser.add_argument("--pages", type=int, default=3,
                        help=f"عدد صفحات النتائج لكل بحث ومصدر (1-{MAX_SEARCH_PAGES}; الافتراضي: 3)")
    parser.add_argument("--mode", choices=("investigation", "academic", "web"), default="investigation",
                        help="نمط البحث: التحقيق (الكل)، أكاديمي، أو ويب عام")
    parser.add_argument("--source", action="append", choices=(*SOURCE_FUNCTIONS.keys(), "academic", "all"),
                        help="حصر المصادر؛ يمكن تكراره. يدعم academic وall")
    parser.add_argument("--type", dest="types", action="append", choices=RECORD_TYPES,
                        help="نوع السجل المطلوب؛ يمكن تكراره")
    parser.add_argument("--year", help="سنة واحدة أو نطاق مثل 2020-2024")
    parser.add_argument("--from-year", type=int, help="أول سنة مشمولة")
    parser.add_argument("--to-year", type=int, help="آخر سنة مشمولة")
    parser.add_argument("--author", help="فلتر اسم المؤلف/الشخص")
    parser.add_argument("--site", action="append", help="إرسال بحث ويب مستقل لكل نطاق محدد؛ يمكن تكراره")
    parser.add_argument("--language", help="رمز اللغة المسجل مثل en أو fr")
    parser.add_argument("--open-access", action="store_true", help="تفضيل الأبحاث المفتوحة؛ البيانات المجهولة تُضمّن مع وسم للمراجعة")
    parser.add_argument("--min-citations", type=int, help="أقل عدد للاستشهادات؛ المجهول يُضمّن افتراضيًا مع وسم للمراجعة")
    parser.add_argument("--sort", choices=("relevance", "date", "citations"), default="relevance",
                        help="ترتيب الصلة أو تاريخ النشر أو الاستشهادات")
    parser.add_argument("--export", help="تصدير جدولي .csv أو سجلات .json")
    parser.add_argument("--report", help="تقرير أدلة قابل للمراجعة بصيغة .md أو .json")
    parser.add_argument("--unknown-metadata", choices=("include", "exclude"), default="include",
                        help="سياسة البيانات الناقصة في فلاتر البيانات الوصفية (الافتراضي include مع تنبيه؛ site/source دائمًا صارمان)")
    parser.add_argument("--strict", action="store_true", help="تعطيل إضافة النتائج ذات التطابق النصي الجزئي")
    parser.add_argument("--top-influential", type=int, default=5, help="عدد الأبحاث الأعلى استشهادًا؛ 0 لتعطيله")
    parser.add_argument("--no-summary", action="store_true", help="تعطيل الملخص المحلي البسيط")
    parser.add_argument("--no-deep", action="store_true", help="عدم قراءة صفحات الويب العامة المختارة")
    parser.add_argument("--deep-pages", type=int, default=15,
                        help=f"عدد صفحات الويب العامة التي تقرأ بعمق (0-{MAX_DEEP_PAGES}; الافتراضي: 15)")
    parser.add_argument("--citation-expansion-limit", type=int, default=12,
                        help="حد أقصى للأعمال المكتشفة عبر خطوة استشهاد واحدة في OpenAlex (0-20)")
    parser.add_argument("--debug", action="store_true", help="حفظ HTML خام من DuckDuckGo في /tmp")
    parser.add_argument("--delay", type=float, default=1.0, help="الحد الأدنى بين طلبات المصادر")
    parser.add_argument("--llm", action="store_true", help="تلخيص الأدلة المختارة باستخدام خادم LLM اختياري")
    parser.add_argument("--llm-question", help="السؤال الموجّه للنموذج")
    parser.add_argument("--llm-url", help="رابط Chat Completions المحلي")
    parser.add_argument("--llm-model", help=f"اسم النموذج (الافتراضي: {DEFAULT_MODEL})")
    parser.add_argument("--llm-max-tokens", type=int, default=400, help="حد مخرجات النموذج (1-400)")
    parser.add_argument("--llm-thinking", action="store_true", help="تفعيل التفكير الداخلي للنموذج")
    args = parser.parse_args(argv)

    if args.num < 1 or args.num > MAX_RESULTS_PER_PAGE:
        parser.error(f"--num must be between 1 and {MAX_RESULTS_PER_PAGE}")
    if args.pages < 1 or args.pages > MAX_SEARCH_PAGES:
        parser.error(f"--pages must be between 1 and {MAX_SEARCH_PAGES}")
    if len(args.site or []) > 10:
        parser.error("use at most 10 --site filters in one search")
    for site in args.site or []:
        try:
            parse_query(f"site:{site}")
        except QuerySyntaxError as exc:
            parser.error(f"invalid --site value {site!r}: {exc}")
    if any(year is not None and not 1000 <= year <= 9999 for year in (args.from_year, args.to_year)):
        parser.error("year bounds must be between 1000 and 9999")
    if args.language and not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", args.language):
        parser.error("--language expects a language code such as en, fr, or zh-Hans")
    if args.deep_pages < 0 or args.deep_pages > MAX_DEEP_PAGES:
        parser.error(f"--deep-pages must be between 0 and {MAX_DEEP_PAGES}")
    if args.citation_expansion_limit < 0 or args.citation_expansion_limit > 20:
        parser.error("--citation-expansion-limit must be between 0 and 20")
    if args.min_citations is not None and args.min_citations < 0:
        parser.error("--min-citations cannot be negative")
    if args.top_influential < 0:
        parser.error("--top-influential cannot be negative")
    if not 1 <= args.llm_max_tokens <= 400:
        parser.error("--llm-max-tokens must be between 1 and 400")
    if args.llm_question and not args.llm:
        parser.error("--llm-question requires --llm")
    if args.report and not args.report.casefold().endswith((".md", ".json")):
        parser.error("--report path must end with .md or .json")
    if args.export and not args.export.casefold().endswith((".csv", ".json")):
        parser.error("--export path must end with .csv or .json")
    if args.image:
        if args.query:
            parser.error("provide either a text query or --image, not both")
        unsupported = []
        if args.mode != "investigation":
            unsupported.append("--mode")
        if args.source:
            unsupported.append("--source")
        if args.types:
            unsupported.append("--type")
        if any((args.year, args.from_year, args.to_year, args.author, args.site, args.language,
                args.open_access, args.min_citations is not None)):
            unsupported.append("text filters")
        if args.llm:
            unsupported.append("--llm")
        if unsupported:
            parser.error(f"--image cannot be combined with {', '.join(unsupported)}")
        if not 1 <= args.image_max_results <= MAX_IMAGE_RESULTS:
            parser.error(f"--image-max-results must be between 1 and {MAX_IMAGE_RESULTS}")
        if args.report and args.export and args.report == args.export:
            parser.error("--report and --export must use different paths")
        try:
            image_report = search_image(args.image, max_results=args.image_max_results)
        except (ImageSearchError, RuntimeError) as exc:
            print(f"❌ تعذّر البحث عن الصورة: {exc}", file=sys.stderr)
            return 1
        print_image_results(image_report)
        try:
            if args.export:
                export_image_results(image_report, args.export)
                print(f"\n✅ صُدّرت نتائج الصور إلى: {args.export}")
            if args.report:
                write_image_report(image_report, args.report)
                print(f"\n✅ حُفظ تقرير الصور في: {args.report}")
        except OSError as exc:
            print(f"❌ تعذّر حفظ ملف النتائج: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.query:
        parser.error("provide a text query or use --image PATH")
    try:
        query_tree = parse_query(args.query)
    except QuerySyntaxError as exc:
        parser.error(str(exc))
    upstream_query = source_query(query_tree)
    if not upstream_query:
        parser.error("include at least one searchable term in addition to field filters")

    global_config.request_delay = max(0.0, args.delay)
    source_keys = _selected_source_keys(args.mode, args.source or [])
    try:
        query_variants = source_query_variants(query_tree)
    except QuerySyntaxError as exc:
        parser.error(str(exc))
    if not query_variants:
        parser.error("query must contain a positive text term; a filter alone cannot retrieve records")
    web_sites = list(dict.fromkeys((args.site or []) + _positive_field_values(query_tree, "site")))
    web_site_multiplier = max(1, len(web_sites))
    per_variant_budget = max(1, (args.num + len(query_variants) - 1) // len(query_variants))
    result_target_per_variant = per_variant_budget * args.pages

    def planned_requests_for(key):
        if key == "google":
            return (min(result_target_per_variant, 100) + 9) // 10
        if key == "brave":
            target = min(result_target_per_variant, 20 * 10)
            return (target + min(per_variant_budget, 20) - 1) // min(per_variant_budget, 20)
        if key == "duckduckgo":
            return min((result_target_per_variant + 29) // 30, 7)
        return args.pages

    planned_page_requests = len(query_variants) * sum(
        planned_requests_for(key) * (web_site_multiplier if key in {"duckduckgo", "google", "brave"} else 1)
        for key in source_keys
    )
    if planned_page_requests > MAX_PLANNED_SEARCH_REQUESTS:
        parser.error(
            f"search plan needs {planned_page_requests} work units; "
            f"reduce pages, sites, or OR branches (maximum {MAX_PLANNED_SEARCH_REQUESTS})"
        )
    explicit_source_names = None
    if args.source:
        explicit_source_names = [SOURCE_FUNCTIONS[key][0] for key in source_keys]
    jobs = []
    provider_from_year, provider_to_year = args.from_year, args.to_year
    if args.year:
        try:
            range_match = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", args.year)
            if not range_match:
                raise ValueError("--year expects YYYY or YYYY-YYYY")
            year_start = int(range_match.group(1))
            year_end = int(range_match.group(2) or range_match.group(1))
            if year_start > year_end:
                raise ValueError("--year start must not exceed end")
            provider_from_year = max(provider_from_year or year_start, year_start)
            provider_to_year = min(provider_to_year or year_end, year_end)
        except ValueError as exc:
            parser.error(str(exc))
    if provider_from_year is not None and provider_to_year is not None and provider_from_year > provider_to_year:
        parser.error("year filters do not overlap")
    for key in source_keys:
        name, function = SOURCE_FUNCTIONS[key]
        def source_job(function=function, key=key, name=name, per_variant_budget=per_variant_budget):
            combined = []
            request_errors = []
            successful_requests = 0
            request_number = 0
            for variant in query_variants:
                site_targets = web_sites if key in {"duckduckgo", "google", "brave"} and web_sites else [None]
                for site in site_targets:
                    if request_number:
                        polite_wait()
                    request_number += 1
                    try:
                        if key in {"duckduckgo", "google", "brave"}:
                            kwargs = {"site": site}
                            if key == "duckduckgo":
                                kwargs["debug"] = args.debug
                            results = function(variant, per_variant_budget, args.pages, **kwargs)
                        elif key in {"openalex", "crossref", "pubmed"}:
                            kwargs = {"from_year": provider_from_year, "to_year": provider_to_year,
                                      "author": args.author}
                            if key == "openalex":
                                kwargs.update(min_citations=args.min_citations, open_access=args.open_access)
                            results = function(variant, per_variant_budget, args.pages, **kwargs)
                        else:
                            results = function(variant, per_variant_budget, args.pages)
                    except Exception as exc:
                        if len(request_errors) < 3:
                            request_errors.append(str(exc))
                        continue
                    successful_requests += 1
                    for rank, record in enumerate(results, 1):
                        prior_rank = record.source_ranks.get(name)
                        record.source_ranks[name] = min(prior_rank, rank) if prior_rank else rank
                    combined.extend(results)
            return combined, request_errors, successful_requests
        fn = source_job
        jobs.append((name, fn))

    def run_job(job):
        name, function = job
        try:
            records, request_errors, successful_requests = function()
            for rank, record in enumerate(records, 1):
                record.source_ranks.setdefault(record.source, rank)
            return name, records, "; ".join(request_errors) if request_errors else None, successful_requests
        except Exception as exc:
            return name, [], str(exc), 0

    print(f"⏳ كنبحث فـ {len(jobs)} مصدر؛ {len(query_variants)} صيغ استعلام؛ {args.pages} صفحات لكل صيغة"
          f"{f' عبر {len(web_sites)} مواقع' if web_sites else ''} ({args.mode}): "
          f"{', '.join(name for name, _ in jobs)} ...")
    all_results, failed_sources = [], []
    any_source_request_succeeded = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as executor:
        for name, records, error, successful_requests in executor.map(run_job, jobs):
            any_source_request_succeeded = any_source_request_succeeded or successful_requests > 0
            if error:
                failed_sources.append((name, error))
                status = "واجه أخطاء جزئية" if records else "ما جاوبش دابا"
                print(f"⚠️  {name} {status}: {error}", file=sys.stderr)
            if records or not error:
                print(f"   ✅ {name}: {len(records)} نتيجة")
            all_results.extend(records)
    source_search_failed = bool(jobs) and not any_source_request_succeeded and not all_results

    all_results = dedupe(all_results)
    filter_args = dict(year=args.year, from_year=args.from_year, to_year=args.to_year,
                       author=args.author, sites=args.site, types=args.types,
                       sources=explicit_source_names, min_citations=args.min_citations,
                       open_access=args.open_access, language=args.language,
                       include_unknown=args.unknown_metadata == "include")
    try:
        candidates, initial_rejections = filter_records(all_results, query="", **filter_args)
    except ValueError as exc:
        parser.error(str(exc))

    if candidates and args.citation_expansion_limit and "openalex" in source_keys:
        seed_candidates, _ = filter_records(candidates, query=args.query, **filter_args)
        calculate_bm25_scores(seed_candidates, upstream_query)
        combine_relevance_scores(seed_candidates)
        seed_candidates = [record for record in sort_results(seed_candidates, "relevance")
                           if record.openalex_id][:4]
        if seed_candidates:
            print("⏳ كنوسع البحث خطوة واحدة عبر مراجع OpenAlex والأعمال التي تستشهد بالنتائج الأقوى ...")
            graph_records, graph_errors = expand_openalex_graph(
                seed_candidates, limit=args.citation_expansion_limit
            )
            for error in graph_errors:
                failed_sources.append(("OpenAlex citation graph", error))
                print(f"⚠️  تعذر توسيع جزء من شبكة الاستشهادات: {error}", file=sys.stderr)
            if graph_records:
                unique_count_before = len(all_results)
                all_results = dedupe(all_results + graph_records)
                new_unique_count = max(0, len(all_results) - unique_count_before)
                candidates, initial_rejections = filter_records(all_results, query="", **filter_args)
                print(f"   ✅ شبكة الاستشهادات: استُرجع {len(graph_records)} سجلًا، منها {new_unique_count} جديدة بعد إزالة التكرار")

    if candidates and not args.no_deep and args.deep_pages:
        calculate_bm25_scores(candidates, upstream_query)
        combine_relevance_scores(candidates)
        candidates = sort_results(candidates, "relevance")
        web_candidates = [r for r in candidates if any(source in PUBLIC_WEB_SOURCES
                          for source in (r.sources or [r.source]))]
        if web_candidates:
            print(f"⏳ كنقرا بعمق فـ {min(len(web_candidates), args.deep_pages, 10)} صفحة عامة بحد 1MB و8 ثوانٍ لكل صفحة ...")
            enrich_with_deep_content(web_candidates, args.query, max_pages=args.deep_pages)

    try:
        results, query_rejections = filter_records(candidates, query=args.query, **filter_args)
    except ValueError as exc:
        parser.error(str(exc))
    filter_rejections = dict(initial_rejections)
    for reason, count in query_rejections.items():
        filter_rejections[reason] = max(filter_rejections.get(reason, 0), count)
    for record in results:
        record.query_match = "exact"
        record.query_terms_matched = 0
        record.query_terms_total = 0
    partial_matches = []
    if not args.strict:
        partial_matches = _partial_text_matches(candidates, results, _plain_text_terms(query_tree))
        if partial_matches:
            filter_rejections["query"] = max(0, filter_rejections.get("query", 0) - len(partial_matches))
            results.extend(partial_matches)
            print(f"ℹ️  أضفنا {len(partial_matches)} نتيجة بتطابق جزئي؛ وضحناها في القائمة للمراجعة.")
    build_citation_links(results)
    calculate_bm25_scores(results, upstream_query)
    combine_relevance_scores(results)
    if not args.no_summary:
        for record in results:
            if record.abstract:
                record.smart_summary = summarize_abstract(record.abstract, args.query)
    results = sort_results(results, args.sort)
    results = sorted(results, key=lambda record: record.query_match == "partial")
    site_multiplier = web_site_multiplier if any(key in {"duckduckgo", "google", "brave"} for key in source_keys) else 1
    results = results[: (args.num * args.pages) * max(1, len(jobs)) * site_multiplier]

    if filter_rejections:
        details = ", ".join(f"{name}: {count}" for name, count in sorted(filter_rejections.items()))
        print(f"ℹ️  نتائج مستبعدة حسب الفلاتر ({details}).")
    uncertain_records = sum(bool(record.filter_uncertainties) for record in results)
    if uncertain_records:
        print(f"⚠️  {uncertain_records} نتيجة تتضمن بيانات ناقصة في فلاتر البيانات الوصفية؛ افحص التنبيه بجانب كل نتيجة.")
    if results:
        if args.top_influential:
            print_top_influential(get_top_influential(results, args.top_influential))
        print_results(results, args.query)
    else:
        if source_search_failed:
            print("❌ تعذر إكمال البحث: فشلت جميع المصادر المختارة. هذه ليست نتيجة تؤكد عدم وجود تطابق؛ راجع أخطاء المصادر واتصال الشبكة أو بيانات اعتماد API.")
        elif not all_results and failed_sources:
            print("⚠️ البحث غير مكتمل: لم تُرجع المصادر السليمة سجلات، وبعض المصادر فشلت. لا يمكن اعتبار ذلك تأكيدًا لعدم وجود تطابق.")
        elif not all_results:
            print("ℹ️ لم تُرجع المصادر المختارة سجلات لهذا الاستعلام. هذا لا يثبت عدم وجود معلومات على الإنترنت.")
        else:
            print(f"⚠️ ما بقات حتى نتيجة تطابق الاستعلام والفلاتر: {args.query}")

    if args.export:
        export_results(results, args.export)
        print(f"\n✅ تصدرت النتائج إلى: {args.export}")
    if args.report:
        report_filters = {"mode": args.mode, "sources": args.source, "types": args.types,
                          "year": args.year, "from_year": args.from_year, "to_year": args.to_year,
                          "author": args.author, "sites": args.site, "language": args.language,
                          "open_access": args.open_access, "min_citations": args.min_citations,
                          "unknown_metadata": args.unknown_metadata,
                          "sort": args.sort,
                          "deep_pages": 0 if args.no_deep else args.deep_pages,
                          "citation_expansion_limit": args.citation_expansion_limit}
        write_report(args.report, results, args.query, filters=report_filters,
                     filter_rejections=filter_rejections, failed_sources=failed_sources)
        print(f"\n✅ تصدر تقرير الأدلة إلى: {args.report}")

    if args.llm:
        selected_results = select_informative_results(results)
        if not selected_results:
            print("\n🤖 ما أرسلتش النتائج للنموذج: المقتطفات ما فيهاش معلومات محددة كافية باش نلخصها.")
            return 1 if source_search_failed else 0
        evidence = build_evidence(args.query, selected_results)
        print(f"\n\n🤖 كنرسل للنموذج {len(selected_results)} سجلات مختارة ({len(evidence)} حرفًا كحد أقصى)؛ سقف الرد {args.llm_max_tokens} token.")
        print("📎 السجلات المرسلة:")
        for index, record in enumerate(selected_results, 1):
            print(f"   [{index}] [{record.kind}] {record.title} — {record.url}")
        try:
            stream_answer(args.query, selected_results, question=args.llm_question, url=args.llm_url,
                          model=args.llm_model, max_tokens=args.llm_max_tokens, thinking=args.llm_thinking,
                          on_text=lambda chunk: print(chunk, end="", flush=True))
            print()
        except (RuntimeError, ValueError) as exc:
            print(f"\n❌ تعذر الحصول على إجابة من LLM: {exc}", file=sys.stderr)

    return 1 if source_search_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
