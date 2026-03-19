"""Reusable helpers for GlueLLM workflow outputs.

GlueLLM workflows generally return `final_output` as a string. For this
project we often ask agents to output strict JSON that can be parsed into a
Pydantic model.

This module provides small utilities to:
- extract the first JSON object from a string
- run a ReflectionWorkflow (generator + reflector)
- parse its final output into a structured Pydantic model
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TypeVar

from gluellm.api import GlueLLM
from gluellm.models.agent import Agent
from gluellm.models.workflow import ReflectionConfig
from gluellm.workflows.reflection import ReflectionWorkflow
from pydantic import BaseModel

from core.logging_config import log_llm_request_response

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("gluellm_practice")


class _AgentTextExecutor:
    """Executor adapter that returns plain text for workflows.

    GlueLLM's workflow implementations expect executors to return strings.
    Some GlueLLM executor implementations return `ExecutionResult` objects
    instead. This adapter calls `GlueLLM.complete()` and returns the final
    response text so workflows always receive a `str`.
    """

    def __init__(self, agent: Agent):
        """Store the agent configuration used for execution."""
        self._agent = agent

    async def execute(self, query: str) -> str:
        """Execute query with the agent config and return plain text."""
        client = GlueLLM(
            model=self._agent.model,
            system_prompt=self._agent.system_prompt.content if self._agent.system_prompt else None,
            tools=self._agent.tools,
            max_tool_iterations=self._agent.max_tool_iterations,
            max_tokens=self._agent.max_tokens,
        )
        result = await client.complete(query)
        try:
            response_text = str(result.final_response)
        except AttributeError:  # pragma: no cover
            response_text = str(result.final_result)
        log_llm_request_response(query, response_text)
        return response_text


def extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object from text.

    This is intentionally conservative: it starts at the first `{` and returns
    only if it can find a matching closing `}` while tracking string/escape
    state.

    Args:
        text: Raw string possibly containing JSON.

    Returns:
        The JSON substring (including surrounding braces), or None.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


async def run_reflection_workflow_parsed(
    *,
    initial_input: str,
    response_model: type[T],
    generator_agent: Agent,
    reflector_agent: Agent,
    max_reflections: int = 2,
    on_parse_error: Callable[[str, Exception], T] | None = None,
) -> T:
    """Run a ReflectionWorkflow and parse its final JSON into a model.

    Args:
        initial_input: Workflow initial input (passed to the generator at reflection 0).
        response_model: Pydantic model to validate output against.
        generator_agent: Agent used as the generator in the workflow.
        reflector_agent: Agent used as the reflector in the workflow.
        max_reflections: Maximum number of reflection iterations (2 means reflector runs once).
        on_parse_error: Optional callback to convert parse/validation errors into a structured fallback model.

    Returns:
        Parsed and validated `response_model` instance.

    Raises:
        ValueError: If JSON extraction or parsing fails and no `on_parse_error` is provided.
    """
    logger.info(
        "workflow.start type=reflection response_model=%s generator=%s reflector=%s max_reflections=%s input_len=%s",
        response_model.__name__,
        generator_agent.name,
        reflector_agent.name,
        max_reflections,
        len(initial_input),
    )
    workflow = ReflectionWorkflow(
        generator=_AgentTextExecutor(generator_agent),
        reflector=_AgentTextExecutor(reflector_agent),
        config=ReflectionConfig(max_reflections=max_reflections),
    )
    result = await workflow.execute(initial_input)
    raw = result.final_output

    try:
        json_text = extract_first_json_object(raw)
        if json_text is None:
            raise ValueError("No JSON object found in workflow output.")
        obj = json.loads(json_text)
        parsed = response_model.model_validate(obj)
        logger.info(
            "workflow.parsed response_model=%s parsed_type=%s json_len=%s",
            response_model.__name__,
            type(parsed).__name__,
            len(json_text),
        )
        return parsed
    except Exception as e:
        logger.warning(
            "workflow.parse_failed response_model=%s error=%s raw_prefix=%r",
            response_model.__name__,
            type(e).__name__,
            raw[:200],
        )
        if on_parse_error is not None:
            logger.info(
                "workflow.fallback_used response_model=%s fallback=%s",
                response_model.__name__,
                getattr(on_parse_error, "__name__", "on_parse_error"),
            )
            return on_parse_error(raw, e)
        raise ValueError(
            f"Failed to parse workflow output into {response_model.__name__}: {e}"
        ) from e
