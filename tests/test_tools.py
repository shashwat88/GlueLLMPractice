"""Unit tests for tool modules.

These tests mock all external HTTP calls to keep them deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from core.loop import normalize_rps_move, rps_winner
from tools.eodhd_tools import fetch_realtime_quote, normalize_stock_symbol
from tools.edgar_tools import fetch_filing_text, list_recent_filings, search_company
from tools.filesystem_tools import count_by_extension, largest_file, scan_directory
from tools.search_tools import get_page_summary, search_wikipedia


def test_normalize_rps_move() -> None:
    """`normalize_rps_move` should accept valid moves and normalize case."""
    assert normalize_rps_move("Rock") == "rock"
    assert normalize_rps_move(" scissors ") == "scissors"


def test_rps_winner_basic() -> None:
    """`rps_winner` should return correct winner labels."""
    assert rps_winner("rock", "scissors") == "a"
    assert rps_winner("paper", "rock") == "a"
    assert rps_winner("scissors", "paper") == "a"
    assert rps_winner("rock", "rock") == "draw"
    assert rps_winner("rock", "paper") == "b"


def test_scan_directory_and_helpers(tmp_path: Path) -> None:
    """Directory scanning should count files and compute depth and largest file."""
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)

    p1 = root / "a" / "x.py"
    p1.write_text("print('x')", encoding="utf-8")
    p2 = root / "a" / "b" / "y.txt"
    p2.write_text("hello world", encoding="utf-8")
    p3 = root / "z.py"
    p3.write_text("print('z')\n" + "a" * 50, encoding="utf-8")

    report = scan_directory(str(root))
    assert report.total_files == 3
    assert report.total_dirs >= 2  # a/ and a/b/
    assert report.max_depth >= 2
    assert count_by_extension(report, ".py") == 2

    largest = largest_file(report)
    assert largest is not None
    assert largest.extension == "py"
    assert "z.py" in largest.relative_path


def test_edgar_tools_search_company_mocked() -> None:
    """`search_company` should return CIK matches given mocked SEC tickers."""

    company_tickers = [
        {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc"},
        {"cik_str": "789019", "ticker": "MSFT", "title": "Microsoft Corporation"},
    ]

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float = 0) -> Any:
        resp = Mock()
        if "company_tickers.json" in url:
            resp.json.return_value = company_tickers
            resp.text = ""
            resp.raise_for_status.return_value = None
            return resp
        raise AssertionError("Unexpected URL in test.")

    with patch("tools.edgar_tools.httpx.get", side_effect=fake_get):
        results = search_company("AAPL", limit=1)
        assert len(results) == 1
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["cik"] == "0000320193"


def test_edgar_tools_list_recent_filings_mocked() -> None:
    """`list_recent_filings` should parse recent filing arrays from submissions JSON."""
    company_tickers = [{"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc"}]
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-24-000001"],
                "filingDate": ["2024-01-01"],
                "form": ["10-K"],
                "primaryDocument": ["d10k.htm"],
            }
        }
    }

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float = 0, params: Any | None = None) -> Any:
        resp = Mock()
        if "company_tickers.json" in url:
            resp.json.return_value = company_tickers
            resp.raise_for_status.return_value = None
            return resp
        if "submissions/CIK" in url:
            resp.json.return_value = submissions
            resp.raise_for_status.return_value = None
            return resp
        raise AssertionError(f"Unexpected URL in test: {url}")

    with patch("tools.edgar_tools.httpx.get", side_effect=fake_get):
        filings = list_recent_filings("AAPL", limit=1)
        assert len(filings) == 1
        f = filings[0]
        assert f["form"] == "10-K"
        assert f["filing_date"] == "2024-01-01"
        assert f["primary_document"] == "d10k.htm"


def test_edgar_tools_fetch_filing_text_mocked() -> None:
    """`fetch_filing_text` should extract HTML and truncate."""
    long_html = "<html><body><p>" + ("hello " * 1000) + "</p></body></html>"

    def fake_get(url: str, headers: dict[str, str] | None = None, timeout: float = 0) -> Any:
        resp = Mock()
        assert "Archives/edgar/data" in url
        resp.text = long_html
        resp.raise_for_status.return_value = None
        return resp

    with patch("tools.edgar_tools.httpx.get", side_effect=fake_get):
        result = fetch_filing_text(
            cik="320193",
            accession_number="0000320193-24-000001",
            primary_document="d10k.htm",
            max_chars=50,
        )
        assert result["cik"] == "0000320193"
        assert "d10k.htm" in result["primary_document"]
        assert len(result["text"]) <= 50 + len(" ...") + 1
        assert result["url"].startswith("https://www.sec.gov/Archives/edgar/data/")


def test_search_tools_search_wikipedia_mocked() -> None:
    """`search_wikipedia` should turn MediaWiki search items into title/url entries."""

    mw_response = {
        "query": {
            "search": [
                {"title": "OpenAI"},
                {"title": "Artificial intelligence"},
            ]
        }
    }

    def fake_get(url: str, params: dict[str, str] | None = None, timeout: float = 0) -> Any:
        resp = Mock()
        assert "w/api.php" in url
        resp.raise_for_status.return_value = None
        resp.json.return_value = mw_response
        return resp

    with patch("tools.search_tools.httpx.get", side_effect=fake_get):
        results = search_wikipedia("OpenAI", limit=1)
        assert len(results) == 1
        assert results[0]["title"] == "OpenAI"
        assert results[0]["url"].startswith("https://en.wikipedia.org/wiki/")


def test_search_tools_get_page_summary_mocked() -> None:
    """`get_page_summary` should return None when `extract` is missing or empty."""

    def fake_get(url: str, timeout: float = 0) -> Any:
        resp = Mock()
        resp.raise_for_status.return_value = None
        if "missing" in url:
            resp.json.return_value = {"title": "Missing", "extract": ""}
        else:
            resp.json.return_value = {
                "title": "Example",
                "extract": "A short summary of Example.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Example"}},
            }
        return resp

    with patch("tools.search_tools.httpx.get", side_effect=fake_get):
        s = get_page_summary("Example", max_chars=200)
        assert s is not None
        assert s["title"] == "Example"
        assert "Example" in s["summary"]
        assert get_page_summary("missing", max_chars=200) is None


def test_normalize_stock_symbol() -> None:
    """Stock symbol normalization should strip `.US` and non-alphanumerics."""
    assert normalize_stock_symbol("aapl") == "AAPL"
    assert normalize_stock_symbol("msft.us") == "MSFT"
    assert normalize_stock_symbol("  brk.b  ") == "BRKB"


def test_fetch_realtime_quote_success_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fetch_realtime_quote` should return evidence when HTTP succeeds."""
    monkeypatch.setenv("EODHD_API_KEY", "test-token")

    fake_json = {"symbol": "AAPL", "price": 123.45, "volume": 1000, "high": 130.0, "low": 120.0}

    def fake_get(url: str, timeout: float = 0) -> Any:
        assert "api/real-time/AAPL.US" in url
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = fake_json
        return resp

    with patch("tools.eodhd_tools.httpx.get", side_effect=fake_get):
        result = fetch_realtime_quote("AAPL", max_chars=100000)
        assert result["stock_symbol"] == "AAPL"
        assert "error" not in result or not result["error"]
        assert result["data"] == fake_json
        assert result["source_url"] is not None


def test_fetch_realtime_quote_missing_api_key() -> None:
    """When EODHD_API_KEY is missing, tool should return an error dict."""
    # Ensure env var is not set for this test.
    with patch.dict("os.environ", {}, clear=True):
        result = fetch_realtime_quote("AAPL", max_chars=100000)
        assert result["data"] == {}
        assert "Missing environment variable" in result["error"]

