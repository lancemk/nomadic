#!/usr/bin/env python3
"""Client helpers for online metadata search."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_METADATA_BASE_URL = "http://localhost:8888"
DEFAULT_METADATA_TIMEOUT_SECONDS = 10.0
DEFAULT_RESULT_PREVIEW_LIMIT = 5
DEFAULT_SNIPPET_CHARS = 240
DEFAULT_RESPONSE_PREVIEW_CHARS = 4_000


def resolve_metadata_base_url(base_url: Optional[str] = None) -> str:
    """Resolve the configured metadata base URL.

    Preference order:
    1. explicit argument
    2. METADATA_BASE_URL environment variable
    3. local default
    """
    candidate = base_url if base_url is not None else os.getenv("METADATA_BASE_URL")
    candidate = (candidate or DEFAULT_METADATA_BASE_URL).strip()
    if not candidate:
        candidate = DEFAULT_METADATA_BASE_URL
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def build_metadata_search_endpoint(base_url: Optional[str] = None) -> str:
    return f"{resolve_metadata_base_url(base_url)}/search"


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _coerce_limit(limit: Any) -> Optional[int]:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    return limit


def _coerce_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must be a non-empty string")
    return normalized


def _response_text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _response_preview(text: str) -> str:
    return _truncate(text, DEFAULT_RESPONSE_PREVIEW_CHARS)


def _pick_first_string(mapping: dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _pick_first_number(mapping: dict[str, Any], keys: tuple[str, ...]) -> Optional[float | int]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _normalize_result_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        text = str(item)
        return {
            "title": None,
            "snippet": _truncate(text, DEFAULT_SNIPPET_CHARS),
            "url": None,
            "id": None,
            "score": None,
            "metadata": None,
            "raw": item,
        }

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = None

    return {
        "title": _pick_first_string(item, ("title", "name", "label", "headline")),
        "snippet": _truncate(
            _pick_first_string(item, ("snippet", "summary", "excerpt", "description", "text", "content"))
            or "",
            DEFAULT_SNIPPET_CHARS,
        )
        or None,
        "url": _pick_first_string(item, ("url", "link", "href", "source", "web_url")),
        "id": _pick_first_string(item, ("id", "uuid", "key", "slug")),
        "score": _pick_first_number(item, ("score", "rank", "relevance")),
        "metadata": metadata,
        "raw": item,
    }


def _extract_result_items(response_data: Any) -> tuple[list[Any], str]:
    if isinstance(response_data, list):
        return response_data, "list"

    if isinstance(response_data, dict):
        for key in ("results", "items", "documents", "matches", "hits", "data"):
            value = response_data.get(key)
            if isinstance(value, list):
                return value, key
        for key in ("result", "item", "document", "match", "hit", "entry"):
            value = response_data.get(key)
            if isinstance(value, dict):
                return [value], key
        return [], "dict"

    if response_data is None:
        return [], "none"

    return [response_data], type(response_data).__name__


def _build_success_payload(
    *,
    query: str,
    limit: Optional[int],
    base_url: str,
    endpoint: str,
    http_status: int,
    content_type: Optional[str],
    response_text: str,
    response_data: Any,
) -> dict[str, Any]:
    raw_items, source_shape = _extract_result_items(response_data)
    results = [_normalize_result_item(item) for item in raw_items]
    count = len(results)
    summary = (
        f'Found {count} metadata result(s) for "{query}" from {endpoint}.'
        if count
        else f'No metadata results for "{query}" from {endpoint}.'
    )

    return {
        "ok": True,
        "status": "ok",
        "error": None,
        "error_type": None,
        "query": query,
        "limit": limit,
        "base_url": base_url,
        "endpoint": endpoint,
        "http_status": http_status,
        "content_type": content_type,
        "result_count": count,
        "summary": summary,
        "results": results,
        "raw_response": response_data,
        "raw_text": response_text,
        "response_shape": source_shape,
    }


def _build_error_payload(
    *,
    query: Optional[str],
    limit: Optional[int],
    base_url: str,
    endpoint: str,
    http_status: Optional[int],
    content_type: Optional[str],
    error_type: str,
    error: str,
    response_text: Optional[str] = None,
    response_data: Any = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "error",
        "error": error,
        "error_type": error_type,
        "query": query,
        "limit": limit,
        "base_url": base_url,
        "endpoint": endpoint,
        "http_status": http_status,
        "content_type": content_type,
        "result_count": 0,
        "summary": error,
        "results": [],
        "raw_response": response_data,
        "raw_text": response_text,
        "response_shape": None,
    }


def search_metadata(
    query: Any,
    *,
    limit: Any = None,
    base_url: Optional[str] = None,
    timeout_seconds: float = DEFAULT_METADATA_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Search the configured metadata service and return a normalized payload."""
    endpoint = ""
    resolved_base_url = DEFAULT_METADATA_BASE_URL
    normalized_query: Optional[str] = None
    normalized_limit: Optional[int] = None

    try:
        resolved_base_url = resolve_metadata_base_url(base_url)
        endpoint = build_metadata_search_endpoint(resolved_base_url)
        normalized_query = _coerce_query(query)
        normalized_limit = _coerce_limit(limit)
    except Exception as exc:
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=None,
            content_type=None,
            error_type="invalid_argument",
            error=str(exc),
        )

    request_body: dict[str, Any] = {"query": normalized_query}
    if normalized_limit is not None:
        request_body["limit"] = normalized_limit

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            http_status = getattr(response, "status", response.getcode())
            content_type = response.headers.get("content-type")
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        try:
            response_body = exc.read() if hasattr(exc, "read") else b""
        finally:
            try:
                exc.close()
            except Exception:
                pass
        response_text = _response_text(response_body)
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=exc.code,
            content_type=exc.headers.get("content-type") if getattr(exc, "headers", None) else None,
            error_type="http_error",
            error=f"metadata service returned HTTP {exc.code}",
            response_text=_response_preview(response_text),
            response_data=_safe_json_loads(response_text),
        )
    except urllib.error.URLError as exc:
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=None,
            content_type=None,
            error_type="connection_error",
            error=f"could not reach metadata service at {resolved_base_url}: {exc.reason}",
        )
    except TimeoutError:
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=None,
            content_type=None,
            error_type="timeout",
            error=f"metadata search timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=None,
            content_type=None,
            error_type="request_error",
            error=str(exc),
        )

    response_text = _response_text(response_body)
    if not response_text.strip():
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=http_status,
            content_type=content_type,
            error_type="invalid_json",
            error="metadata service returned an empty response body",
        )
    try:
        response_data = json.loads(response_text) if response_text.strip() else None
    except json.JSONDecodeError as exc:
        return _build_error_payload(
            query=normalized_query,
            limit=normalized_limit,
            base_url=resolved_base_url,
            endpoint=endpoint,
            http_status=http_status,
            content_type=content_type,
            error_type="invalid_json",
            error=f"metadata service returned non-JSON payload: {exc.msg}",
            response_text=_response_preview(response_text),
        )

    payload = _build_success_payload(
        query=normalized_query,
        limit=normalized_limit,
        base_url=resolved_base_url,
        endpoint=endpoint,
        http_status=http_status,
        content_type=content_type,
        response_text=response_text,
        response_data=response_data,
    )
    return payload


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        return None


def render_metadata_search_result(payload: dict[str, Any]) -> str:
    """Render the normalized payload as a human-readable tool result."""
    lines: list[str] = [
        "Tool: metadata_search",
        f"Endpoint: {payload.get('endpoint') or '[unknown]'}",
        f"Base URL: {payload.get('base_url') or '[unknown]'}",
        f"Query: {payload.get('query') or '[none]'}",
        f"Limit: {payload.get('limit') if payload.get('limit') is not None else '[none]'}",
        f"Status: {payload.get('status')}",
    ]

    if payload.get("ok"):
        lines.append(f"Result Count: {payload.get('result_count', 0)}")
        lines.append("")
        lines.append(str(payload.get("summary") or ""))
        results = payload.get("results") or []
        if results:
            lines.append("")
            lines.append("Top Results:")
            for idx, item in enumerate(results[:DEFAULT_RESULT_PREVIEW_LIMIT], start=1):
                title = item.get("title") or item.get("id") or "[untitled]"
                snippet = item.get("snippet") or ""
                url = item.get("url")
                result_line = f"{idx}. {title}"
                if url:
                    result_line += f" — {url}"
                lines.append(result_line)
                if snippet:
                    lines.append(f"   {snippet}")
        else:
            lines.append("")
            lines.append("No results returned.")
    else:
        lines.append(f"Error Type: {payload.get('error_type') or '[unknown]'}")
        lines.append(f"Error: {payload.get('error') or '[unknown]'}")
        response_text = payload.get("raw_text")
        if response_text:
            lines.append("")
            lines.append("Response Preview:")
            lines.append(_truncate(str(response_text), DEFAULT_RESPONSE_PREVIEW_CHARS))

    return "\n".join(lines)
