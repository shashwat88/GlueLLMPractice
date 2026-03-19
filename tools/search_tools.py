"""Wikipedia-based research tools for tool-using agents.

These tools provide:
- search for relevant Wikipedia pages
- fetch a page summary for synthesis

All HTTP calls are made via `httpx.get` so that unit tests can mock them.
"""

from __future__ import annotations

import html
import os
from typing import Any, TypedDict
from urllib.parse import quote

import httpx

WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"

DEFAULT_TIMEOUT_SECS = 20.0
DEFAULT_WIKIPEDIA_USER_AGENT = "GlueLLMPractice/0.1 (https://github.com/; contact: unknown)"


def _wiki_headers() -> dict[str, str]:
    """Return HTTP headers suitable for Wikipedia API access.

    Wikipedia APIs may block requests missing a descriptive User-Agent.

    Returns:
        Headers including a User-Agent string.
    """
    ua = os.environ.get("WIKIPEDIA_USER_AGENT") or DEFAULT_WIKIPEDIA_USER_AGENT
    return {"User-Agent": ua, "Accept": "application/json"}


class PageSearchResult(TypedDict):
    """A single search result entry."""

    query: str
    title: str
    url: str


class PageSummary(TypedDict):
    """Summary fetched from Wikipedia REST summary endpoint."""

    title: str
    url: str
    summary: str


def _http_get_json(url: str, *, params: dict[str, str]) -> dict[str, Any]:
    """Fetch JSON from Wikipedia and return parsed data."""
    resp = httpx.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECS, headers=_wiki_headers())
    resp.raise_for_status()
    return resp.json()


def _http_get_json_rest(url: str) -> dict[str, Any]:
    """Fetch JSON from a REST endpoint and return parsed data."""
    resp = httpx.get(url, timeout=DEFAULT_TIMEOUT_SECS, headers=_wiki_headers())
    resp.raise_for_status()
    return resp.json()


def search_wikipedia(query: str, *, limit: int = 5) -> list[PageSearchResult]:
    """Search Wikipedia pages via the MediaWiki search API.

    Args:
        query: Search query.
        limit: Maximum number of pages to return.

    Returns:
        A list of candidate pages (may be empty).
    """
    q = query.strip()
    if not q:
        return []

    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json",
    }
    data = _http_get_json(WIKI_SEARCH_URL, params=params)
    query_info = data.get("query") or {}
    search_list = (query_info.get("search") if isinstance(query_info, dict) else None) or []

    results: list[PageSearchResult] = []
    for item in search_list[:limit]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        if not title:
            continue
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append({"query": q, "title": title, "url": url})
    return results


def get_page_summary(title: str, *, max_chars: int = 2000) -> PageSummary | None:
    """Fetch a Wikipedia page summary by title.

    Args:
        title: Wikipedia page title.
        max_chars: Max number of characters to return.

    Returns:
        A `PageSummary` entry or None if the page is not available.
    """
    t = title.strip()
    if not t:
        return None

    url = f"{WIKI_REST_SUMMARY_BASE}/{quote(t, safe='')}"
    data = _http_get_json_rest(url)
    # REST v1 includes `title`, `url`, and `extract` (or error payloads).
    extract = data.get("extract") or ""
    if not extract or not isinstance(extract, str):
        return None

    summary = html.unescape(extract).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars] + " ..."

    title_out = str(data.get("title") or t)
    url_out = url
    content_urls = data.get("content_urls")
    if isinstance(content_urls, dict):
        desktop = content_urls.get("desktop")
        if isinstance(desktop, dict):
            page_url = desktop.get("page")
            if isinstance(page_url, str) and page_url:
                url_out = page_url
    return {"title": title_out, "url": url_out, "summary": summary}


def search_and_summarize(query: str, *, limit: int = 3) -> list[PageSummary]:
    """Search Wikipedia and return summaries for the top results.

    Args:
        query: Search query.
        limit: Number of pages to summarize.

    Returns:
        A list of page summaries (may be empty).
    """
    pages = search_wikipedia(query, limit=limit)
    summaries: list[PageSummary] = []
    for p in pages:
        title = p["title"]
        s = get_page_summary(title)
        if s is not None:
            summaries.append(s)
    return summaries
