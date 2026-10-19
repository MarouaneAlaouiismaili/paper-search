"""Reverse image search using Google Cloud Vision Web Detection."""

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from ..network import http_post_json

VISION_ANNOTATE_URL = "https://vision.googleapis.com/v1/images:annotate"
MAX_INLINE_IMAGE_BYTES = 7 * 1024 * 1024
MAX_IMAGE_RESULTS = 50
MAX_RESPONSE_BYTES = 1_000_000


class ImageSearchError(ValueError):
    """Raised when an image cannot be searched or the provider rejects it."""


def _validate_image(data: bytes):
    if data.startswith(b"\xff\xd8\xff"):
        return
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return
    if data.startswith((b"GIF87a", b"GIF89a")):
        return
    if data.startswith(b"BM"):
        return
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return
    if data.startswith(b"\x00\x00\x01\x00"):
        return
    raise ImageSearchError("Unsupported or invalid image. Use JPEG, PNG, GIF, BMP, WebP, TIFF, or ICO.")


def _read_image(path: str) -> bytes:
    try:
        image_path = Path(path).expanduser()
        if not image_path.is_file():
            raise ImageSearchError("Image path is not a readable file.")
        with image_path.open("rb") as image_file:
            content = image_file.read(MAX_INLINE_IMAGE_BYTES + 1)
    except ImageSearchError:
        raise
    except OSError as exc:
        raise ImageSearchError("Could not read the image file.") from exc

    if len(content) > MAX_INLINE_IMAGE_BYTES:
        raise ImageSearchError(
            f"Image exceeds the {MAX_INLINE_IMAGE_BYTES // (1024 * 1024)} MiB inline upload limit."
        )
    _validate_image(content)
    return content


def _image_items(items: object) -> List[dict]:
    if not isinstance(items, list):
        return []
    output = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str) or not item.get("url"):
            continue
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        entry = {"url": url}
        if isinstance(item.get("score"), (int, float)):
            entry["score"] = item["score"]
        output.append(entry)
    return output


def _page_items(items: object) -> List[dict]:
    if not isinstance(items, list):
        return []
    output = []
    seen = set()
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("url"), str)
                or not item.get("url") or item["url"] in seen):
            continue
        seen.add(item["url"])
        page = {"url": item["url"]}
        if item.get("pageTitle"):
            page["title"] = item["pageTitle"]
        page["full_matches"] = _image_items(item.get("fullMatchingImages"))
        page["partial_matches"] = _image_items(item.get("partialMatchingImages"))
        output.append(page)
    return output


def _entities(items: object) -> List[dict]:
    if not isinstance(items, list):
        return []
    output = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entity = {}
        for source_key, output_key in (
            ("description", "description"), ("entityId", "entity_id"), ("score", "score"),
        ):
            value = item.get(source_key)
            if isinstance(value, (str, int, float)) and value != "":
                entity[output_key] = value
        if entity:
            output.append(entity)
    return output


def search_image(path: str, *, api_key: str = None, max_results: int = 10) -> dict:
    """Find web pages and images that fully or partially match a local image.

    The image is uploaded once to Google Cloud Vision. No local path or image
    bytes are included in the returned report.
    """
    key = (api_key or os.environ.get("GOOGLE_CLOUD_VISION_API_KEY", "")).strip()
    if not key:
        raise ImageSearchError(
            "Set GOOGLE_CLOUD_VISION_API_KEY after enabling the Cloud Vision API for your Google Cloud project."
        )
    if not isinstance(max_results, int) or not 1 <= max_results <= MAX_IMAGE_RESULTS:
        raise ImageSearchError(f"max_results must be between 1 and {MAX_IMAGE_RESULTS}.")

    content = _read_image(path)
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(content).decode("ascii")},
            "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
        }],
    }
    status, response_body = http_post_json(
        VISION_ANNOTATE_URL,
        payload,
        headers={"X-Goog-Api-Key": key},
        timeout=30,
        max_response_bytes=MAX_RESPONSE_BYTES,
    )
    try:
        response = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImageSearchError("Google Vision returned an invalid JSON response.") from exc

    top_level_error = response.get("error") if isinstance(response, dict) else None
    if status < 200 or status >= 300:
        message = top_level_error.get("message") if isinstance(top_level_error, dict) else None
        suffix = f": {message}" if message else ""
        raise ImageSearchError(f"Google Vision returned HTTP {status}{suffix}")
    if top_level_error:
        message = top_level_error.get("message", "unknown service error") if isinstance(top_level_error, dict) else str(top_level_error)
        raise ImageSearchError(f"Google Vision could not analyze this image: {message}")

    responses = response.get("responses") if isinstance(response, dict) else None
    if not isinstance(responses, list) or not responses or not isinstance(responses[0], dict):
        raise ImageSearchError("Google Vision returned no image annotation response.")
    first = responses[0]
    if first.get("error"):
        message = first["error"].get("message", "image annotation failed") if isinstance(first["error"], dict) else str(first["error"])
        raise ImageSearchError(f"Google Vision could not analyze this image: {message}")

    web = first.get("webDetection") if isinstance(first.get("webDetection"), dict) else {}
    full_matches = _image_items(web.get("fullMatchingImages"))
    partial_matches = _image_items(web.get("partialMatchingImages"))
    pages = _page_items(web.get("pagesWithMatchingImages"))
    similar = _image_items(web.get("visuallySimilarImages"))
    labels = [
        {"label": item["label"]}
        for item in web.get("bestGuessLabels", [])
        if isinstance(item, dict) and item.get("label")
    ] if isinstance(web.get("bestGuessLabels"), list) else []

    return {
        "provider": "Google Cloud Vision Web Detection",
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "full_matches": full_matches,
        "partial_matches": partial_matches,
        "pages_with_matches": pages,
        "visually_similar_images": similar,
        "best_guess_labels": labels,
        "web_entities": _entities(web.get("webEntities")),
        "has_full_match": bool(full_matches),
        "has_web_references": bool(full_matches or partial_matches or pages or similar),
    }
