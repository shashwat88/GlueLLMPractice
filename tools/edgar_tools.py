"""SEC EDGAR tool functions for company filing research.

These tools are designed to be consumed by GlueLLM's tool-calling loop. They:
1) map company name / ticker to a CIK
2) list recent filings for the CIK with metadata (form, date, accession, doc)
3) fetch a filing's primary document text for summarization / citation

All HTTP calls are done with `httpx.get` so unit tests can mock them easily.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any, TypedDict

import httpx

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_nodashes}/{primary_document}"
)

DEFAULT_TIMEOUT_SECS = 20.0


class CompanyMatch(TypedDict):
    """CIK match for a company name or ticker query."""

    query: str
    cik: str
    ticker: str
    title: str


class FilingMeta(TypedDict):
    """Metadata for a recent filing."""

    cik: str
    cik_int: str
    form: str
    filing_date: str
    accession_number: str
    primary_document: str
    filing_url: str


class FilingTextResult(TypedDict):
    """Fetched filing text returned to the LLM."""

    cik: str
    accession_number: str
    primary_document: str
    url: str
    text: str


def _sec_headers() -> dict[str, str]:
    """Build SEC request headers.

    SEC asks clients to identify themselves with a descriptive User-Agent.
    """
    user_agent = os.getenv(
        "GLUELLM_EDGAR_USER_AGENT", "gluellm-interview (contact: dev@example.com)"
    )
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip"}


def _normalize_cik(cik: str) -> str:
    """Normalize a CIK into SEC's 10-digit string format."""
    digits = re.sub(r"\D", "", cik)
    return digits.zfill(10)


def _cik_to_int_str(cik: str) -> str:
    """Convert a CIK string to its integer representation string for URLs."""
    norm = _normalize_cik(cik)
    return str(int(norm))


def _extract_text_from_html(raw_html: str) -> str:
    """Extract approximate plain text from an HTML document.

    Note: SEC documents can be complex; this is a lightweight extraction
    appropriate for interview/demo purposes.
    """
    cleaned = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style[^>]*>.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _http_get_json(url: str) -> Any:
    """Fetch JSON from SEC with default headers."""
    resp = httpx.get(url, headers=_sec_headers(), timeout=DEFAULT_TIMEOUT_SECS)
    resp.raise_for_status()
    return resp.json()


def _http_get_text(url: str) -> str:
    """Fetch text from SEC with default headers."""
    resp = httpx.get(url, headers=_sec_headers(), timeout=DEFAULT_TIMEOUT_SECS)
    resp.raise_for_status()
    return resp.text


def search_company(query: str, limit: int = 5) -> list[CompanyMatch]:
    """Search SEC company tickers by company name or ticker.

    Args:
        query: Company name or ticker symbol.
        limit: Maximum number of results.

    Returns:
        A list of candidate `CompanyMatch` entries (may be empty).
    """
    query_clean = query.strip()
    if not query_clean:
        return []

    tickers = _http_get_json(SEC_COMPANY_TICKERS_URL)
    q = query_clean.lower()
    q_upper = query_clean.upper()

    def score(item: dict[str, str]) -> int:
        ticker = (item.get("ticker") or "").upper()
        title = (item.get("title") or "").lower()
        if ticker == q_upper:
            return 0
        if q in title:
            return 1
        if q in ticker.lower():
            return 2
        return 3

    items: list[dict[str, str]] = list(tickers)[:]
    items.sort(key=score)
    results: list[CompanyMatch] = []
    for item in items:
        if score(item) >= 3:
            break
        results.append(
            CompanyMatch(
                query=query_clean,
                cik=_normalize_cik(str(item["cik_str"])),
                ticker=item.get("ticker", ""),
                title=item.get("title", ""),
            )
        )
        if len(results) >= limit:
            break
    return results


def list_recent_filings(company_query: str, limit: int = 5) -> list[FilingMeta]:
    """List recent filings for a company (by name or ticker).

    Args:
        company_query: Company name or ticker symbol.
        limit: Maximum filings to return across matching companies.

    Returns:
        A list of filing metadata entries (may be empty).
    """
    candidates = search_company(company_query, limit=5)
    filings: list[FilingMeta] = []

    for c in candidates:
        cik = c["cik"]
        submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
        submissions = _http_get_json(submissions_url)
        recent = submissions.get("filings", {}).get("recent", {}) or {}

        accession_list = recent.get("accessionNumber", []) or []
        filing_date_list = recent.get("filingDate", []) or []
        form_list = recent.get("form", []) or []
        primary_doc_list = recent.get("primaryDocument", []) or []

        for accession, filing_date, form, primary_doc in zip(
            accession_list, filing_date_list, form_list, primary_doc_list, strict=False
        ):
            if len(filings) >= limit:
                return filings
            accession_no_nodashes = re.sub(r"-", "", str(accession))
            url = SEC_ARCHIVES_BASE.format(
                cik_int=_cik_to_int_str(cik),
                accession_no_nodashes=accession_no_nodashes,
                primary_document=str(primary_doc),
            )
            filings.append(
                FilingMeta(
                    cik=cik,
                    cik_int=_cik_to_int_str(cik),
                    form=str(form),
                    filing_date=str(filing_date),
                    accession_number=str(accession),
                    primary_document=str(primary_doc),
                    filing_url=url,
                )
            )
    return filings


def fetch_filing_text(
    cik: str,
    accession_number: str,
    primary_document: str,
    max_chars: int = 8000,
) -> FilingTextResult:
    """Fetch a filing's primary document text.

    Args:
        cik: CIK string (may be non-normalized; will be normalized).
        accession_number: Full accession number from filings metadata.
        primary_document: Primary document filename/path from EDGAR metadata.
        max_chars: Max characters to return for prompt-size control.

    Returns:
        A `FilingTextResult` including URL and extracted text.
    """
    norm_cik = _normalize_cik(cik)
    cik_int = _cik_to_int_str(norm_cik)
    accession_no_nodashes = re.sub(r"-", "", accession_number)
    url = SEC_ARCHIVES_BASE.format(
        cik_int=cik_int,
        accession_no_nodashes=accession_no_nodashes,
        primary_document=primary_document,
    )
    raw_text = _http_get_text(url)
    text = _extract_text_from_html(raw_text)
    if len(text) > max_chars:
        text = text[:max_chars] + " ..."
    return FilingTextResult(
        cik=norm_cik,
        accession_number=accession_number,
        primary_document=primary_document,
        url=url,
        text=text,
    )
