"""Validation tests for non-interactive agent entrypoints."""

from __future__ import annotations

import pytest

from agents.basic_research import _answer_with_workflow as run_basic
from agents.directory_crawler import _answer_with_workflow as run_directory
from agents.eodhd_stock_agent import _answer_with_workflow as run_stock
from agents.poem_loop import run_poem_workflow
from agents.rock_paper_scissors import run as run_rps
from agents.sec_research import _answer_with_workflow as run_sec
from tools.filesystem_tools import DirectoryReport


@pytest.mark.asyncio
async def test_rps_rejects_negative_rounds() -> None:
    with pytest.raises(ValueError):
        await run_rps(-1)


@pytest.mark.asyncio
async def test_poem_loop_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        await run_poem_workflow(topic="", threshold=8, max_iters=3, model="openai:gpt-4o-mini")
    with pytest.raises(ValueError):
        await run_poem_workflow(topic="x", threshold=11, max_iters=3, model="openai:gpt-4o-mini")
    with pytest.raises(ValueError):
        await run_poem_workflow(topic="x", threshold=8, max_iters=0, model="openai:gpt-4o-mini")


@pytest.mark.asyncio
async def test_sec_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        await run_sec(model="", max_tool_iterations=4, question="q", history=[])
    with pytest.raises(ValueError):
        await run_sec(model="openai:gpt-4o-mini", max_tool_iterations=0, question="q", history=[])
    with pytest.raises(ValueError):
        await run_sec(model="openai:gpt-4o-mini", max_tool_iterations=4, question=" ", history=[])


@pytest.mark.asyncio
async def test_basic_research_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        await run_basic(model="", max_tool_iterations=4, question="q", history=[])
    with pytest.raises(ValueError):
        await run_basic(model="openai:gpt-4o-mini", max_tool_iterations=0, question="q", history=[])
    with pytest.raises(ValueError):
        await run_basic(model="openai:gpt-4o-mini", max_tool_iterations=4, question=" ", history=[])


@pytest.mark.asyncio
async def test_directory_rejects_invalid_inputs() -> None:
    report = DirectoryReport(
        root="/tmp", max_depth=0, total_files=0, total_dirs=0, files=[], dirs=[]
    )
    with pytest.raises(ValueError):
        await run_directory(
            model="", max_tool_iterations=4, report=report, question="q", history=[]
        )
    with pytest.raises(ValueError):
        await run_directory(
            model="openai:gpt-4o-mini",
            max_tool_iterations=0,
            report=report,
            question="q",
            history=[],
        )
    with pytest.raises(ValueError):
        await run_directory(
            model="openai:gpt-4o-mini",
            max_tool_iterations=4,
            report=report,
            question=" ",
            history=[],
        )


@pytest.mark.asyncio
async def test_stock_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        await run_stock(
            model="", max_tool_iterations=4, stock_symbol="AAPL", question="q", history=[]
        )
    with pytest.raises(ValueError):
        await run_stock(
            model="openai:gpt-4o-mini",
            max_tool_iterations=0,
            stock_symbol="AAPL",
            question="q",
            history=[],
        )
    with pytest.raises(ValueError):
        await run_stock(
            model="openai:gpt-4o-mini",
            max_tool_iterations=4,
            stock_symbol="",
            question="q",
            history=[],
        )
    with pytest.raises(ValueError):
        await run_stock(
            model="openai:gpt-4o-mini",
            max_tool_iterations=4,
            stock_symbol="AAPL",
            question=" ",
            history=[],
        )
