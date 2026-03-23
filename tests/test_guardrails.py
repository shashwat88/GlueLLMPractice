"""Tests for project guardrails config and GlueLLM wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt

from agents.rock_paper_scissors import _AgentTextExecutor as RPSTextExecutor
from core.base_agent import BaseAgent
from core.guardrails import default_guardrails_config
from core.workflow_wrappers import _AgentTextExecutor as WorkflowTextExecutor
from eval.run_eval_recording import run_eval_recording


def test_default_guardrails_config_uses_sdk_types() -> None:
    cfg = default_guardrails_config()
    assert cfg.enabled is True
    assert cfg.max_length is not None
    assert cfg.pii is not None
    assert cfg.max_output_guardrail_retries >= 1


def test_default_guardrails_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_GUARDRAILS_ENABLED", "false")
    monkeypatch.setenv("PROJECT_GUARDRAILS_MAX_INPUT_LENGTH", "321")
    monkeypatch.setenv("PROJECT_GUARDRAILS_MAX_OUTPUT_LENGTH", "123")
    monkeypatch.setenv("PROJECT_GUARDRAILS_BLOCKLIST_PATTERNS", "foo,bar")
    cfg = default_guardrails_config()
    assert cfg.enabled is False
    assert cfg.max_length is not None
    assert cfg.max_length.max_input_length == 321
    assert cfg.max_length.max_output_length == 123
    assert cfg.blocklist is not None
    assert cfg.blocklist.patterns == ["foo", "bar"]


@pytest.mark.asyncio
async def test_base_agent_passes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGlueLLM:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

        async def complete(self, user_message: str, **kwargs: Any) -> Any:
            _ = user_message, kwargs
            return SimpleNamespace(final_response="ok")

    monkeypatch.setattr("core.base_agent.GlueLLM", FakeGlueLLM)

    agent: BaseAgent[Any] = BaseAgent(name="A", system_prompt="S")
    out = await agent.complete_text("hello")
    assert out == "ok"
    assert captured.get("guardrails") is not None


@pytest.mark.asyncio
async def test_workflow_executor_passes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGlueLLM:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

        async def complete(self, query: str, **kwargs: Any) -> Any:
            _ = query, kwargs
            return SimpleNamespace(final_response="ok")

    monkeypatch.setattr("core.workflow_wrappers.GlueLLM", FakeGlueLLM)

    ex = WorkflowTextExecutor(
        Agent(name="A", description="d", system_prompt=SystemPrompt(content="s"), tools=[])
    )
    out = await ex.execute("hello")
    assert out == "ok"
    assert captured.get("guardrails") is not None


@pytest.mark.asyncio
async def test_rps_executor_passes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGlueLLM:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

        async def complete(self, query: str, **kwargs: Any) -> Any:
            _ = query, kwargs
            return SimpleNamespace(final_response="rock")

    monkeypatch.setattr("agents.rock_paper_scissors.GlueLLM", FakeGlueLLM)

    ex = RPSTextExecutor(
        Agent(name="A", description="d", system_prompt=SystemPrompt(content="s"), tools=[])
    )
    out = await ex.execute("round")
    assert out == "rock"
    assert captured.get("guardrails") is not None


@pytest.mark.asyncio
async def test_eval_runner_passes_guardrails(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, Any] = {}

    class FakeStore:
        def __init__(self, path: str):
            self.path = path

        async def close(self) -> None:
            return None

    class FakeGlueLLM:
        def __init__(self, **kwargs: Any):
            captured.update(kwargs)

        async def complete(self, prompt: str) -> Any:
            _ = prompt
            return SimpleNamespace(final_response="ok")

    monkeypatch.setattr("eval.run_eval_recording.JSONLFileStore", FakeStore)
    monkeypatch.setattr("eval.run_eval_recording.GlueLLM", FakeGlueLLM)

    await run_eval_recording(
        output_path=tmp_path / "records.jsonl",
        model="openai:gpt-4o-mini",
        prompts=["p1"],
    )
    assert captured.get("guardrails") is not None
