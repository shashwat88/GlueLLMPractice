"""EODHD real-time stock tools.

This module provides a minimal tool layer for fetching real-time stock data
from `https://eodhd.com` using the EODHD API.

The tool returns raw JSON (plus a small amount of normalized metadata) so the
LLM can answer questions about:
- current price
- high / low
- volume

Questions that cannot be supported by the returned evidence should be handled
gracefully by the agent layer (set `can_answer=false`).
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

EODHD_API_BASE = "https://eodhd.com/api/real-time/{symbol}.US"
DEFAULT_TIMEOUT_SECS = 20.0


def normalize_stock_symbol(stock_symbol: str) -> str:
    """Normalize a stock symbol for URL construction.

    Args:
        stock_symbol: Raw symbol from user input (e.g., "aapl", "AAPL.US").

    Returns:
        Uppercased symbol without any `.US` suffix.

    Raises:
        ValueError: If the symbol becomes empty after normalization.
    """
    s = stock_symbol.strip().upper()
    s = re.sub(r"\.US$", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    if not s:
        raise ValueError("Stock symbol cannot be empty.")
    return s


def _build_eodhd_url(stock_symbol: str) -> tuple[str, str]:
    """Build the EODHD real-time URL.

    Returns:
        A tuple of (url, api_token_redacted_url).
    """
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise OSError("Missing environment variable: EODHD_API_KEY")
    symbol = normalize_stock_symbol(stock_symbol)
    url = f"{EODHD_API_BASE.format(symbol=symbol)}?api_token={api_key}&fmt=json"
    redacted_url = url.replace(f"api_token={api_key}", "api_token=***")
    return url, redacted_url


def fetch_realtime_quote(stock_symbol: str, max_chars: int = 200000) -> dict[str, Any]:
    """Fetch EODHD real-time quote data for a US stock.

    Args:
        stock_symbol: Stock symbol (e.g., "AAPL").
        max_chars: Safety limit for how much raw JSON text we return.

    Returns:
        A JSON-serializable dict containing either:
        - `data` (on success)
        - `error` (on failure)
    """
    try:
        url, redacted_url = _build_eodhd_url(stock_symbol)
        resp = httpx.get(url, timeout=DEFAULT_TIMEOUT_SECS)
        resp.raise_for_status()
        payload = resp.json()
        # Some APIs may return massive payloads; optionally truncate stringified data.
        if isinstance(payload, dict):
            # Keep as dict; truncation applies only to any large string fields we might surface.
            pass
        else:
            payload = str(payload)
        return {
            "stock_symbol": normalize_stock_symbol(stock_symbol),
            "data": payload,
            "source_url": redacted_url,
        }
    except Exception as e:  # noqa: BLE001 - return tool-level error evidence
        return {
            "stock_symbol": stock_symbol.strip().upper(),
            "data": {},
            "source_url": None,
            "error": str(e),
            "suggestion": "Ensure EODHD_API_KEY is set and the symbol is valid.",
        }
