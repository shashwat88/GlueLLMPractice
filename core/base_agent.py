"""Base abstractions for agents built on top of GlueLLM.

This module centralizes the common GlueLLM wiring so that individual agents can
focus on their orchestration logic (loops, prompts, tool choice) rather than
repeating setup code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from gluellm.api import GlueLLM
from pydantic import BaseModel

from core.guardrails import default_guardrails_config
from core.logging_config import log_llm_request_response

TStructured = TypeVar("TStructured", bound=BaseModel)

Tool = Callable[..., Any]


@dataclass
class BaseAgent(Generic[TStructured]):
    """A thin wrapper around a `gluellm.api.GlueLLM` client.

    Attributes:
        name: Human readable agent name (used in logs).
        system_prompt: System prompt passed to the underlying GlueLLM client.
        tools: Optional tool functions exposed to the LLM tool-calling loop.
        model: Provider:model string used by GlueLLM.
        max_tool_iterations: Safety cap for tool execution rounds.
        client: Optional pre-constructed GlueLLM client (useful for tests).
    """

    name: str
    system_prompt: str
    tools: list[Tool] = field(default_factory=list)
    model: str = "openai:gpt-4o-mini"
    max_tool_iterations: int = 8
    client: GlueLLM | None = None

    def _ensure_client(self) -> GlueLLM:
        """Create a GlueLLM client if one was not injected."""
        if self.client is not None:
            return self.client
        self.client = GlueLLM(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            max_tool_iterations=self.max_tool_iterations,
            guardrails=default_guardrails_config(),
        )
        return self.client

    async def complete_text(self, user_message: str, **kwargs: Any) -> str:
        """Run a plain completion and return the final response text.

        Args:
            user_message: The user message to send to the LLM.
            **kwargs: Extra keyword arguments forwarded to `GlueLLM.complete`.

        Returns:
            The model's final response text.
        """
        client = self._ensure_client()
        result = await client.complete(user_message, **kwargs)
        response_text = result.final_response
        log_llm_request_response(user_message, str(response_text))
        return response_text

    async def complete_structured(
        self, user_message: str, response_format: type[TStructured], **kwargs: Any
    ) -> TStructured:
        """Run a structured completion and return the parsed Pydantic model.

        Args:
            user_message: The user message to send to the LLM.
            response_format: Pydantic model type to parse the response into.
            **kwargs: Extra keyword arguments forwarded to `GlueLLM.structured_complete`.

        Returns:
            A validated instance of `response_format`.
        """
        client = self._ensure_client()
        result = await client.structured_complete(
            user_message,
            response_format=response_format,
            **kwargs,
        )
        structured = result.structured_output
        log_llm_request_response(user_message, str(structured))
        return structured
