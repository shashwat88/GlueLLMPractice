"""Deterministic scorers for eval datasets (no LLM-as-judge)."""

from __future__ import annotations

import re

from core.loop import normalize_rps_move


def score_rps_output(text: str) -> bool:
    """Return True if `text` normalizes to a valid rock/paper/scissors move."""
    try:
        normalize_rps_move(text)
    except ValueError:
        return False
    return True


def parse_poem_critic_score(feedback: str) -> int | None:
    """Parse critic score from free-form feedback (delegates to poem_loop)."""
    from agents.poem_loop import _parse_score

    return _parse_score(feedback)


def score_poem_feedback(feedback: str, *, min_score: int) -> bool:
    """True if parsed score exists and meets minimum."""
    score = parse_poem_critic_score(feedback)
    if score is None:
        return False
    return score >= min_score


def basic_research_answer_has_substrings(answer: str, substrings: list[str]) -> bool:
    """Case-insensitive substring checks for smoke evals."""
    lower = answer.lower()
    return all(s.lower() in lower for s in substrings)


def sec_answer_has_citation_markers(answer: str) -> bool:
    """Heuristic: SEC-style eval rows often mention form types or accession-like tokens."""
    if not answer.strip():
        return False
    # At least one digit (years, accession fragments) or common filing tokens.
    if re.search(r"\b(10-K|10-Q|8-K|DEF\s*14A)\b", answer, re.IGNORECASE):
        return True
    if re.search(r"\b20\d{2}\b", answer):
        return True
    return len(answer) > 80


def eodhd_answer_mentions_symbol(answer: str, symbol: str) -> bool:
    """Smoke check that the answer references the ticker."""
    return symbol.upper() in answer.upper()
