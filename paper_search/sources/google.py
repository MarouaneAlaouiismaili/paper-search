"""Google Programmable Search Engine connector using the official JSON API."""

import json
import os
import urllib.parse
from typing import List

from ..models import Paper
from ..network import http_get, polite_wait
from ..utils import web_query_text

API_URL = "https://www.googleapis.com/customsearch/v1"


def google_search_configured() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY", "").strip()
                and os.environ.get("GOOGLE_CSE_ID", "").strip())


def _redact_error(error: Exception, api_key: str) -> str:
    message = str(error)
    for secret in {api_key, urllib.parse.quote(api_key, safe=""), urllib.parse.quote_plus(api_key)}:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def search_google(query: str, max_results: int = 15, pages: int = 1, *, site: str = None) -> List[Paper]:
    """Return bounded web results from Google's Custom Search JSON API.

    Google currently limits this API to existing customers. Credentials are read
    from the environment and never printed, including when a request fails.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    engine_id = os.environ.get("GOOGLE_CSE_ID", "").strip()
    if not api_key or not engine_id:
        raise RuntimeError(
            "Google Search needs GOOGLE_API_KEY and GOOGLE_CSE_ID for an existing "
            "Custom Search JSON API account. Google stopped accepting new API customers."
        )

    api_query = web_query_text(query)
    if site:
        api_query = f"{api_query} site:{site}".strip()
    if not api_query:
        return []

    result_limit = min(max(0, max_results * pages), 100)
    results = []
    seen_urls = set()

    for start in range(1, result_limit + 1, 10):
        count = min(10, result_limit - start + 1)
        params = {
            "key": api_key,
            "cx": engine_id,
            "q": api_query,
            "start": start,
            "num": count,
        }
        try:
            polite_wait()
            url = API_URL + "?" + urllib.parse.urlencode(params)
            payload = json.loads(http_get(url, timeout=15, retries=1))
        except Exception as exc:
            # The API key is part of Google's required query parameters. Redact it
            # before an exception can reach the CLI's source-failure message.
            raise RuntimeError(f"Google Custom Search request failed: {_redact_error(exc, api_key)}") from None

        api_error = payload.get("error")
        if api_error:
            message = api_error.get("message") or "Google API returned an error"
            raise RuntimeError(_redact_error(RuntimeError(str(message)), api_key))

        items = payload.get("items", [])
        for item in items:
            link = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            if not link or not title or link in seen_urls:
                continue
            seen_urls.add(link)
            snippet = " ".join((item.get("snippet") or "").split())
            results.append(Paper(
                source="Google",
                title=title,
                authors="",
                date="",
                url=link,
                abstract=snippet[:500] + ("..." if len(snippet) > 500 else ""),
            ))

        if len(items) < count:
            break

    return results[:result_limit]
