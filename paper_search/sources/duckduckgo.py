import urllib.parse
import re
import sys
from typing import List, Tuple

from ..network import DNSResolutionError, http_get, polite_wait
from ..utils import clean_html, web_query_text
from ..models import Paper

ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.DOTALL | re.IGNORECASE)
ATTR_HREF_RE = re.compile(r'href\s*=\s*"([^"]*)"', re.IGNORECASE)
ATTR_CLASS_RE = re.compile(r'class\s*=\s*"([^"]*)"', re.IGNORECASE)
BLOCK_MARKERS = ["anomaly", "unusual traffic", "captcha", "verify you are human"]

def _resolve_ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    params = urllib.parse.parse_qs(parsed.query)
    if "duckduckgo.com" in parsed.netloc and "uddg" in params:
        return urllib.parse.unquote(params["uddg"][0])
    return href

def _parse_ddg_html(raw_html: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    titles, snippets = [], []
    for attrs, inner in ANCHOR_RE.findall(raw_html):
        class_m = ATTR_CLASS_RE.search(attrs)
        classes = class_m.group(1).split() if class_m else []
        href_m = ATTR_HREF_RE.search(attrs)
        href = href_m.group(1) if href_m else ""

        if "result__a" in classes and href:
            titles.append((href, inner))
        elif "result__snippet" in classes:
            snippets.append(inner)
    return titles, snippets

def search_duckduckgo(query: str, max_results: int = 15, pages: int = 1, debug: bool = False,
                      site: str = None) -> List[Paper]:
    api_query = web_query_text(query)
    if site:
        api_query = f"{api_query} site:{site}".strip()
    headers = {
        "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        "Referer": "https://duckduckgo.com/",
    }

    results = []
    seen_urls = set()
    page_num = 0
    base_host = "https://html.duckduckgo.com/html/"
    switched_to_lite = False

    target_results = max_results * pages

    offset = 0
    while len(results) < target_results and offset < 200:
        params = {"q": api_query}
        if offset:
            params["s"] = offset
        url = base_host + "?" + urllib.parse.urlencode(params)

        try:
            raw = http_get(url, headers=headers).decode("utf-8", errors="ignore")
        except DNSResolutionError:
            if not switched_to_lite:
                base_host = "https://lite.duckduckgo.com/lite/"
                switched_to_lite = True
                print("    ↪️ تعذر حلّ مضيف DuckDuckGo الرئيسي؛ نجرب المضيف البديل ...", file=sys.stderr)
                continue
            raise
        except RuntimeError:
            if not switched_to_lite:
                base_host = "https://lite.duckduckgo.com/lite/"
                switched_to_lite = True
                print("    ↪️ تعذّر طلب DuckDuckGo العادي؛ نجرب المضيف البديل ...", file=sys.stderr)
                continue
            raise
        page_num += 1

        if debug:
            dbg_path = f"/tmp/ddg_debug_page{page_num}.html"
            with open(dbg_path, "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"    🐛 [debug] تصفحت {len(raw)} حرف، خزنتهم فـ {dbg_path}", file=sys.stderr)

        low = raw.lower()
        if any(marker in low for marker in BLOCK_MARKERS):
            if not switched_to_lite:
                base_host = "https://lite.duckduckgo.com/lite/"
                switched_to_lite = True
                print("    ↪️  بدلت لنسخة lite.duckduckgo.com بعد صفحة حماية ...", file=sys.stderr)
                continue
            print("    ⚠️  DuckDuckGo واجهتنا بصفحة حماية (anti-bot/CAPTCHA)، ما قدرناش نجيبو نتائج.", file=sys.stderr)
            if not results:
                raise RuntimeError("DuckDuckGo blocked this search with an anti-bot/CAPTCHA page")
            break

        titles_links, snippets = _parse_ddg_html(raw)

        if not titles_links:
            break

        for i, (href, title_html) in enumerate(titles_links):
            title = clean_html(title_html)
            link = _resolve_ddg_url(href)
            snippet = clean_html(snippets[i]) if i < len(snippets) else ""
            if not title or not link or link in seen_urls:
                continue
            seen_urls.add(link)
            results.append(Paper(
                source="DuckDuckGo",
                title=title,
                authors="",
                date="",
                url=link,
                abstract=snippet[:500] + ("..." if len(snippet) > 500 else ""),
            ))
            if len(results) >= target_results:
                break

        offset += 30
        polite_wait()

    return results
