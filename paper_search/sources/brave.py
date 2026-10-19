"""Brave Search connector using the official Web Search API."""

import json
import os
import urllib.parse
from typing import List

from ..models import Paper
from ..network import http_get, polite_wait
from ..utils import clean_html, web_query_text

API_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_COUNT = 20
MAX_OFFSET = 9
MAX_RESULTS_PER_QUERY = MAX_COUNT * (MAX_OFFSET + 1)


def brave_search_configured() -> bool:
    return bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip())


def _redact_error(error: Exception, token: str) -> str:
    message = str(error)
    for secret in {token, urllib.parse.quote(token, safe=""), urllib.parse.quote_plus(token)}:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def search_brave(query: str, max_results: int = 15, pages: int = 1, *, site: str = None) -> List[Paper]:
    """Fetch bounded, paginated public web results from Brave Search.

    Brave's API requires a user-provided key; costs and quotas are controlled by
    the account that owns that key. The token is read from the environment and
    never written to the query URL or CLI output.
    """
    token = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not token:
        raise RuntimeError("Brave Search needs BRAVE_SEARCH_API_KEY; see the README for setup and pricing.")

    api_query = web_query_text(query)
    if site:
        api_query = f"{api_query} site:{site}".strip()
    if not api_query:
        return []

    if max_results <= 0 or pages <= 0:
        return []
    result_limit = min(max_results * pages, MAX_RESULTS_PER_QUERY)
    count = min(max_results, MAX_COUNT)
    page_count = min((result_limit + count - 1) // count, MAX_OFFSET + 1)
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": token,
    }
    records = []
    seen_urls = set()

    for offset in range(page_count):
        params = {"q": api_query, "count": count, "offset": offset, "text_decorations": "false"}
        url = API_URL + "?" + urllib.parse.urlencode(params)
        try:
            polite_wait()
            payload = json.loads(http_get(url, headers=headers, timeout=15, retries=2))
        except Exception as exc:
            raise RuntimeError(f"Brave Search request failed: {_redact_error(exc, token)}") from None

        page_results = (payload.get("web") or {}).get("results", [])
        for item in page_results:
            title = (item.get("title") or "").strip()
            link = (item.get("url") or "").strip()
            if not title or not link or link in seen_urls:
                continue
            seen_urls.add(link)
            description = clean_html(item.get("description") or "")
            records.append(Paper(
                source="Brave Search",
                title=title,
                authors="",
                date="",
                url=link,
                abstract=description[:500] + ("..." if len(description) > 500 else ""),
            ))

        more_results = (payload.get("query") or {}).get("more_results_available")
        if len(page_results) < count or more_results is False:
            break

    return records[:result_limit]
