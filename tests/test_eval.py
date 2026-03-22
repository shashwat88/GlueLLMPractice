"""Tests for eval harness scorers and helpers (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.recording import enable_session_recording, eval_logs_dir
from eval.scoring import (
    basic_research_answer_has_substrings,
    eodhd_answer_mentions_symbol,
    parse_poem_critic_score,
    score_poem_feedback,
    score_rps_output,
    sec_answer_has_citation_markers,
)


def test_score_rps_output_accepts_valid_moves() -> None:
    assert score_rps_output("rock") is True
    assert score_rps_output(" Paper\n") is True
    assert score_rps_output("SCISSORS") is True


def test_score_rps_output_rejects_invalid() -> None:
    assert score_rps_output("rocky") is False
    assert score_rps_output("") is False


def test_parse_poem_critic_score() -> None:
    assert (
        parse_poem_critic_score(
            '{"score": 7, "strengths": [], "suggestions": [], "revised_poem_guidance": "x"}'
        )
        == 7
    )
    assert parse_poem_critic_score("score: 4") == 4


def test_score_poem_feedback() -> None:
    assert (
        score_poem_feedback(
            '{"score": 8, "strengths": [], "suggestions": [], "revised_poem_guidance": "x"}',
            min_score=8,
        )
        is True
    )
    assert score_poem_feedback("score: 2", min_score=8) is False


def test_basic_research_substrings() -> None:
    assert basic_research_answer_has_substrings("Hello Python world", ["python"]) is True
    assert basic_research_answer_has_substrings("Hello", ["missing"]) is False


def test_sec_answer_heuristic() -> None:
    assert sec_answer_has_citation_markers("Discussed in the 10-K for 2023.") is True
    assert sec_answer_has_citation_markers("Nope") is False


def test_eodhd_symbol() -> None:
    assert eodhd_answer_mentions_symbol("AAPL price", "AAPL") is True
    assert eodhd_answer_mentions_symbol("price up", "AAPL") is False


def test_eval_logs_dir_creates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    d = eval_logs_dir()
    assert d == tmp_path / "logs"
    assert d.is_dir()


def test_enable_session_recording_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from gluellm.eval import get_global_eval_store

    p = enable_session_recording("test_agent", project_root=tmp_path)
    assert p.name.startswith("eval_test_agent_")
    assert p.suffix == ".jsonl"
    store = get_global_eval_store()
    assert store is not None
