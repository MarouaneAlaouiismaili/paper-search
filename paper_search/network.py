import random
import threading
import time
import sys
from urllib.parse import urlsplit

from .config import global_config

try:
    from curl_cffi import requests
except ImportError:
    print("❌ مكتبة 'curl_cffi' غير مثبتة. يرجى تثبيتها بالمرور بالأمر التالي:", file=sys.stderr)
    print("    pip install curl_cffi\n", file=sys.stderr)
    sys.exit(1)

DEFAULT_UA = (
    "paper-search/1.0 (+public research; contact details may be configured by the user)"
)

_session_lock = threading.Lock()
_session = None


class DNSResolutionError(RuntimeError):
    """A DNS lookup failed; retrying the same request is unlikely to help."""


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
    return _session

def polite_wait(base=None):
    """كينتظر مدة (base + تذبذب عشوائي بسيط) بين كل طلب وطلب."""
    base = global_config.request_delay if base is None else base
    if base <= 0:
        return
    jitter = random.uniform(0, base * 0.3)
    time.sleep(base + jitter)

def http_get(url, headers=None, retries=4, timeout=20, max_bytes=None, allow_redirects=True):
    """Fetch a public HTTP resource, optionally enforcing a strict response-size cap."""
    hdrs = {"User-Agent": DEFAULT_UA}
    if headers:
        hdrs.update(headers)

    last_err = None
    session = _get_session()
    for attempt in range(retries):
        resp = None
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise RuntimeError("Only public HTTP(S) URLs without embedded credentials are allowed")
            resp = session.get(
                url,
                headers=hdrs,
                timeout=timeout,
                allow_redirects=allow_redirects,
                stream=max_bytes is not None,
            )

            if resp.status_code == 429:
                if max_bytes is not None:
                    raise RuntimeError("HTTP 429: bounded page fetch was rate limited")
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 3 * (attempt + 1)
                except ValueError:
                    wait = 3 * (attempt + 1)
                time.sleep(min(wait, 30))
                continue

            if resp.status_code in (403, 404):
                raise RuntimeError(f"HTTP {resp.status_code} على {url}")
            if not allow_redirects and 300 <= resp.status_code < 400:
                raise RuntimeError(f"Redirect not followed for bounded public-page fetch: HTTP {resp.status_code}")

            if resp.status_code >= 500:
                last_err = RuntimeError(f"HTTP {resp.status_code} (خطأ سيرفر)")
                if max_bytes is None:
                    time.sleep(min(1.5 * (attempt + 1), 5))
                continue

            resp.raise_for_status()
            if max_bytes is None:
                return resp.content
            length = resp.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise RuntimeError(f"Response exceeds the {max_bytes}-byte limit")
            chunks = bytearray()
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                if len(chunks) + len(chunk) > max_bytes:
                    raise RuntimeError(f"Response exceeds the {max_bytes}-byte limit")
                chunks.extend(chunk)
            return bytes(chunks)
        except RuntimeError:
            raise
        except getattr(requests.exceptions, "DNSError", ()) as e:
            hostname = urlsplit(url).hostname or "unknown host"
            raise DNSResolutionError(f"DNS lookup failed for {hostname}: {e}") from e
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
        finally:
            if resp is not None:
                resp.close()

    raise RuntimeError(f"فشل الطلب بعد {retries} محاولات: {last_err}")


def http_post_json(url, payload, headers=None, timeout=30, max_response_bytes=1_000_000):
    """Send one bounded JSON POST without redirects or automatic retries."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("Only HTTPS URLs without embedded credentials are allowed for JSON POST requests")

    hdrs = {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
    }
    if headers:
        hdrs.update(headers)

    response = None
    try:
        response = _get_session().post(
            url, json=payload, headers=hdrs, timeout=timeout,
            allow_redirects=False, stream=True,
        )
        if 300 <= response.status_code < 400:
            raise RuntimeError(f"Refused HTTP redirect from {parsed.hostname}")

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_response_bytes:
            raise RuntimeError(f"JSON response exceeds the {max_response_bytes}-byte limit")

        body = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            if len(body) + len(chunk) > max_response_bytes:
                raise RuntimeError(f"JSON response exceeds the {max_response_bytes}-byte limit")
            body.extend(chunk)
        return response.status_code, bytes(body)
    except RuntimeError:
        raise
    except getattr(requests.exceptions, "DNSError", ()) as exc:
        raise DNSResolutionError(f"DNS lookup failed for {parsed.hostname}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"HTTPS JSON POST to {parsed.hostname} failed ({type(exc).__name__})") from exc
    finally:
        if response is not None:
            response.close()
