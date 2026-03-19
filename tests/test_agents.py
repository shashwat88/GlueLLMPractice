"""Unit tests for agent loop modules and interactive agents.

These tests stub out LLM behavior so no external API calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from core.loop import LoopLimitError, run_refinement_loop


class CritiqueModel(BaseModel):
    """Simple critique model for refinement-loop tests."""

    score: int = Field(ge=0, le=10)


@pytest.mark.asyncio
async def test_run_refinement_loop_accepts_on_first_iteration() -> None:
    """If the critic already meets the threshold, writer_fn should not run."""
    critic_calls: int = 0
    writer_calls: int = 0

    async def critic_fn(draft: str) -> CritiqueModel:
        nonlocal critic_calls
        critic_calls += 1
        return CritiqueModel(score=8)

    async def writer_fn(i: int, draft: str, critique: CritiqueModel) -> str:
        nonlocal writer_calls
        writer_calls += 1
        return draft + " (revised)"

    def accept_if(c: CritiqueModel) -> bool:
        return c.score >= 8

    result = await run_refinement_loop(
        max_iters=3,
        start_draft="start",
        critic_fn=critic_fn,
        writer_fn=writer_fn,
        accept_if=accept_if,
    )
    assert result.accepted_iteration == 1
    assert result.accepted_draft == "start"
    assert critic_calls == 1
    assert writer_calls == 0


@pytest.mark.asyncio
async def test_run_refinement_loop_runs_until_acceptance() -> None:
    """Writer_fn should run for iterations where acceptance is not met."""
    scores = [3, 6, 9]
    critic_idx = 0
    writer_calls: list[int] = []

    async def critic_fn(draft: str) -> CritiqueModel:
        nonlocal critic_idx
        score = scores[critic_idx]
        critic_idx += 1
        return CritiqueModel(score=score)

    async def writer_fn(i: int, draft: str, critique: CritiqueModel) -> str:
        writer_calls.append(i)
        return draft + f" (it{i})"

    def accept_if(c: CritiqueModel) -> bool:
        return c.score >= 8

    result = await run_refinement_loop(
        max_iters=5,
        start_draft="start",
        critic_fn=critic_fn,
        writer_fn=writer_fn,
        accept_if=accept_if,
    )
    assert result.accepted_iteration == 3
    assert writer_calls == [1, 2]


@pytest.mark.asyncio
async def test_run_refinement_loop_raises_on_limit() -> None:
    """The refinement loop should raise LoopLimitError when it never converges."""

    async def critic_fn(draft: str) -> CritiqueModel:
        return CritiqueModel(score=1)

    async def writer_fn(i: int, draft: str, critique: CritiqueModel) -> str:
        return draft + f" (it{i})"

    def accept_if(c: CritiqueModel) -> bool:
        return c.score >= 8

    with pytest.raises(LoopLimitError):
        await run_refinement_loop(
            max_iters=2,
            start_draft="start",
            critic_fn=critic_fn,
            writer_fn=writer_fn,
            accept_if=accept_if,
        )


class _FakeStructuredResult:
    """Minimal object shaped like GlueLLM structured output results."""

    def __init__(self, structured_output: BaseModel):
        """Store the provided structured output."""
        self.structured_output = structured_output


class _FakeGlueLLM:
    """Fake GlueLLM client for agent interactive-loop tests."""

    def __init__(self, *, structured_outputs: list[BaseModel]):
        """Create a fake client that returns scripted structured outputs."""
        self._structured_outputs = structured_outputs
        self._idx = 0

    async def structured_complete(
        self, question: str, response_format: type[BaseModel], **kwargs: Any
    ) -> _FakeStructuredResult:
        """Return the next scripted structured output.

        Args:
            question: The input question (ignored in this fake).
            response_format: Expected response model type (ignored in this fake).
            **kwargs: Additional arguments (ignored).

        Returns:
            A `_FakeStructuredResult` containing the scripted output.
        """
        if self._idx >= len(self._structured_outputs):
            raise AssertionError("FakeGlueLLM ran out of scripted outputs.")
        out = self._structured_outputs[self._idx]
        self._idx += 1
        return _FakeStructuredResult(out)


@pytest.mark.asyncio
async def test_directory_crawler_interactive_prints_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Directory crawler should print an answer or a graceful cannot-answer message."""
    from agents.directory_crawler import DirectoryAnswer, directory_crawler_interactive
    from tools.filesystem_tools import DirectoryReport, FileInfo

    # Scripted outputs for two turns (initial + follow-up).
    fake_outputs = [
        DirectoryAnswer(
            can_answer=True, answer="There are 2 Python files.", cannot_answer_reason=None
        ),
        DirectoryAnswer(
            can_answer=False,
            answer="",
            cannot_answer_reason="Not enough information in the report.",
        ),
    ]

    idx = 0

    async def fake_run_reflection_workflow_parsed(*args: Any, **kwargs: Any) -> DirectoryAnswer:
        nonlocal idx
        _ = args, kwargs
        out = fake_outputs[idx]
        idx += 1
        return out

    # Patch scan_directory to return a minimal report.
    report = DirectoryReport(
        root="/fake",
        max_depth=2,
        total_files=2,
        total_dirs=1,
        files=[
            FileInfo(relative_path="a/x.py", extension="py", size_bytes=10, depth=2),
            FileInfo(relative_path="y.py", extension="py", size_bytes=20, depth=1),
        ],
    )
    scan_mock = Mock(return_value=report)
    monkeypatch.setattr("agents.directory_crawler.scan_directory", scan_mock)

    # Provide interactive stdin inputs:
    # root, question, follow-up, quit
    inputs = iter(["/fake", "How many Python files?", "What is the largest file?", "quit"])

    def fake_input(prompt: str = "") -> str:  # noqa: ARG001 - prompt is unused in the test
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        "agents.directory_crawler.run_reflection_workflow_parsed",
        fake_run_reflection_workflow_parsed,
    )

    await directory_crawler_interactive(model="fake:model", max_tool_iterations=1)
    captured = capsys.readouterr().out
    assert "There are 2 Python files." in captured
    assert "I cannot answer that question from the scanned directory evidence." in captured
    assert scan_mock.call_count == 1


@pytest.mark.asyncio
async def test_basic_research_interactive_quit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Basic research loop should exit when the user types quit."""
    from agents.basic_research import ResearchResponse, ResearchSource, basic_research_interactive

    fake_outputs = [
        ResearchResponse(
            can_answer=True,
            query="What is AI?",
            summary="AI stands for artificial intelligence.",
            key_points=["It is used for smart tasks."],
            sources=[
                ResearchSource(title="Artificial intelligence", url="https://example.com/ai"),
            ],
            cannot_answer_reason=None,
        ),
    ]

    idx = 0

    async def fake_run_reflection_workflow_parsed(*args: Any, **kwargs: Any) -> ResearchResponse:
        nonlocal idx
        _ = args, kwargs
        out = fake_outputs[idx]
        idx += 1
        return out

    inputs = iter(["What is AI?", "quit"])

    def fake_input(prompt: str = "") -> str:  # noqa: ARG001
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        "agents.basic_research.run_reflection_workflow_parsed", fake_run_reflection_workflow_parsed
    )

    await basic_research_interactive(model="fake:model", max_tool_iterations=1)
    captured = capsys.readouterr().out
    assert "AI stands for artificial intelligence." in captured


@pytest.mark.asyncio
async def test_sec_research_interactive_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SEC research loop should print graceful fallback when can_answer is false."""
    from agents.sec_research import SecAnswer, sec_research_interactive

    fake_outputs = [
        SecAnswer(
            can_answer=False, answer="", citations=[], cannot_answer_reason="No matching filings."
        ),
    ]

    idx = 0

    async def fake_run_reflection_workflow_parsed(*args: Any, **kwargs: Any) -> SecAnswer:
        nonlocal idx
        _ = args, kwargs
        out = fake_outputs[idx]
        idx += 1
        return out

    inputs = iter(["Some SEC question", "quit"])

    def fake_input(prompt: str = "") -> str:  # noqa: ARG001
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        "agents.sec_research.run_reflection_workflow_parsed", fake_run_reflection_workflow_parsed
    )

    await sec_research_interactive(model="fake:model", max_tool_iterations=1)
    captured = capsys.readouterr().out
    assert "I cannot answer that question based on the retrieved SEC evidence." in captured
    assert "No matching filings." in captured


@pytest.mark.asyncio
async def test_eodhd_stock_agent_interactive_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stock agent should respond gracefully when can_answer=false."""
    from agents.eodhd_stock_agent import StockAnswer, eodhd_stock_agent_interactive

    fake_outputs = [
        StockAnswer(
            can_answer=False,
            stock_symbol="AAPL",
            answer="",
            cannot_answer_reason="Question is unrelated to real-time quote facts.",
        ),
    ]

    idx = 0

    async def fake_run_reflection_workflow_parsed(*args: Any, **kwargs: Any) -> StockAnswer:
        nonlocal idx
        _ = args, kwargs
        out = fake_outputs[idx]
        idx += 1
        return out

    # symbol, question, quit
    inputs = iter(["AAPL", "What is the weather?", "quit"])

    def fake_input(prompt: str = "") -> str:  # noqa: ARG001
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        "agents.eodhd_stock_agent.run_reflection_workflow_parsed",
        fake_run_reflection_workflow_parsed,
    )

    await eodhd_stock_agent_interactive(model="fake:model", max_tool_iterations=1)
    captured = capsys.readouterr().out
    assert (
        "I cannot answer that question based on the available real-time stock evidence." in captured
    )
    assert "Question is unrelated to real-time quote facts." in captured


@pytest.mark.asyncio
async def test_poem_workflow_iteration_logging(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Poem workflow should log iterations and return final output when workflow is stubbed."""
    from agents.poem_loop import run_poem_workflow

    class _FakeWorkflowResult:
        """Fake WorkflowResult matching the fields used by `run_poem_workflow`."""

        def __init__(self) -> None:
            self.final_output = "final poem"
            self.iterations = 2
            self.agent_interactions = [
                {"iteration": 1, "agent": "producer", "output": "poem1"},
                {
                    "iteration": 1,
                    "agent": "critic_poetry_quality",
                    "output": json.dumps(
                        {
                            "score": 7,
                            "strengths": ["imagery"],
                            "suggestions": ["clarity"],
                            "revised_poem_guidance": "Make it clearer.",
                        }
                    ),
                },
                {"iteration": 2, "agent": "producer", "output": "poem2"},
                {
                    "iteration": 2,
                    "agent": "critic_poetry_quality",
                    "output": json.dumps(
                        {
                            "score": 9,
                            "strengths": ["coherence"],
                            "suggestions": [],
                            "revised_poem_guidance": "Looks good.",
                        }
                    ),
                },
            ]

    class _FakeWorkflow:
        """Fake IterativeRefinementWorkflow that returns a scripted result."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - test stub
            _ = args, kwargs

        async def execute(self, initial_input: str) -> _FakeWorkflowResult:
            _ = initial_input
            return _FakeWorkflowResult()

    # Patch the workflow class used in the poem_loop module.
    monkeypatch.setattr("agents.poem_loop.IterativeRefinementWorkflow", _FakeWorkflow)

    poem, iters = await run_poem_workflow(
        topic="winter", threshold=8, max_iters=3, model="fake:model"
    )
    assert poem == "final poem"
    assert iters == 2

    out = capsys.readouterr().out
    assert "Iteration: 1" in out
    assert "Parsed score: 7/10" in out
    assert "Iteration: 2" in out
    assert "Parsed score: 9/10" in out
