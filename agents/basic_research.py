"""Basic research agent (Wikipedia tools + interactive follow-ups).

This agent:
- accepts a runtime user query
- uses an external tool (Wikipedia search + page summaries)
- synthesizes findings into a structured summary
- supports follow-up questions until the user types `quit`

If no relevant sources are available from the tool results, it responds
gracefully that it cannot answer.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from pydantic import BaseModel, Field

from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt

from tools.search_tools import search_and_summarize
from core.workflow_wrappers import run_reflection_workflow_parsed


class ResearchSource(BaseModel):
    """A source entry used for research citations."""

    title: str
    url: str


class ResearchResponse(BaseModel):
    """Structured response returned by the research agent."""

    can_answer: bool = Field(description="Whether there are sufficient sources to answer.")
    query: str = Field(description="The research query being answered.")
    summary: str = Field(description="Concise synthesized summary (only if can_answer is true).")
    key_points: list[str] = Field(description="Bulleted key points (only if can_answer is true).")
    sources: list[ResearchSource] = Field(description="Source list with title and URL.")
    cannot_answer_reason: str | None = Field(description="Reason if can_answer is false.")


def _is_quit(text: str) -> bool:
    """Return True if user typed `quit`."""
    return text.strip().lower() == "quit"


def _build_conversation_history(history: list[tuple[str, str]]) -> str:
    """Render prior Q/A into a compact text block for the workflow prompt."""
    if not history:
        return "No prior conversation."
    parts: list[str] = []
    for i, (q, a) in enumerate(history, 1):
        parts.append(f"[Turn {i}] User: {q}\nAssistant: {a}")
    return "\n\n".join(parts)


def _build_generator_agent(*, model: str, max_tool_iterations: int) -> Agent:
    """Create the generator agent that produces strict ResearchResponse JSON."""
    system_prompt = (
        "You are a research assistant. Use the provided Wikipedia tool(s) to gather evidence. "
        "Synthesize results into a concise, structured answer. "
        "If the tools return no relevant sources, set can_answer=false and explain cannot_answer_reason. "
        "Do not guess or fabricate sources. "
        "You MUST output ONLY valid JSON matching the ResearchResponse schema:\n"
        "{\n"
        '  "can_answer": boolean,\n'
        '  "query": string,\n'
        '  "summary": string,\n'
        '  "key_points": string[],\n'
        '  "sources": {"title": string, "url": string}[],\n'
        '  "cannot_answer_reason": string|null\n'
        "}\n"
        "Rules:\n"
        "- If can_answer=false: summary must be \"\", key_points must be [], sources must be [], and cannot_answer_reason must be non-empty.\n"
        "- If can_answer=true: summary must be non-empty, key_points must be non-empty, and sources must be non-empty.\n"
        "- Set `query` to the latest user question."
    )
    return Agent(
        name="ResearchGenerator",
        description="Wikipedia evidence generator that outputs ResearchResponse JSON.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[search_and_summarize],
        max_tool_iterations=max_tool_iterations,
        model=model,
    )


def _build_reflector_agent(*, model: str) -> Agent:
    """Create the reflector agent that validates evidence constraints."""
    system_prompt = (
        "You are a compliance reflector for basic research. "
        "Validate that the generator's JSON obeys the can_answer/source rules. "
        "Return ONLY valid JSON matching the same ResearchResponse schema. "
        "If sources are missing when can_answer=true, set can_answer=false with a clear cannot_answer_reason."
    )
    return Agent(
        name="ResearchReflector",
        description="Reflector that validates ResearchResponse JSON and evidence presence.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[],
        max_tool_iterations=1,
        model=model,
    )


def _extract_query_for_response(user_text: str) -> str:
    """Derive the query label for the response model."""
    t = user_text.strip()
    return t


async def _answer_with_workflow(
    *,
    model: str,
    max_tool_iterations: int,
    question: str,
    history: list[tuple[str, str]],
) -> ResearchResponse:
    """Answer one research question using ReflectionWorkflow and parse strict JSON."""
    generator = _build_generator_agent(model=model, max_tool_iterations=max_tool_iterations)
    reflector = _build_reflector_agent(model=model)
    history_text = _build_conversation_history(history)
    initial_input = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Current user question:\n{question}\n\n"
        "Return the JSON object only."
    )

    def _on_parse_error(raw: str, e: Exception) -> ResearchResponse:
        _ = raw
        return ResearchResponse(
            can_answer=False,
            query=question,
            summary="",
            key_points=[],
            sources=[],
            cannot_answer_reason=f"Workflow output could not be parsed as valid Research JSON: {type(e).__name__}",
        )

    parsed = await run_reflection_workflow_parsed(
        initial_input=initial_input,
        response_model=ResearchResponse,
        generator_agent=generator,
        reflector_agent=reflector,
        max_reflections=2,
        on_parse_error=_on_parse_error,
    )

    if parsed.can_answer and not parsed.sources:
        return ResearchResponse(
            can_answer=False,
            query=question,
            summary="",
            key_points=[],
            sources=[],
            cannot_answer_reason="Answer claimed can_answer=true but no sources were provided.",
        )
    return parsed


async def basic_research_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the interactive basic research loop."""
    history: list[tuple[str, str]] = []
    query = input("Research query (or type 'quit' to exit): ").strip()
    if _is_quit(query):
        return
    while not query:
        query = input("Query cannot be empty. Research query (or 'quit'): ").strip()
        if _is_quit(query):
            return

    while True:
        question = query
        response = await _answer_with_workflow(
            model=model,
            max_tool_iterations=max_tool_iterations,
            question=question,
            history=history,
        )

        if response.can_answer:
            print("\nSummary\n" + "-" * 40)
            print(response.summary)
            print("\nKey points\n" + "-" * 40)
            for kp in response.key_points:
                print(f"- {kp}")
            if response.sources:
                print("\nSources\n" + "-" * 40)
                for s in response.sources:
                    print(f"- {s.title}: {s.url}")
        else:
            reason = response.cannot_answer_reason or "No relevant sources found."
            print("\nI cannot answer that question based on the available Wikipedia evidence.")
            print("Reason: " + reason)

        assistant_text = response.summary if response.can_answer else f"Cannot answer: {reason}"
        history.append((question, assistant_text))

        follow_up = input("\nFollow-up (or type 'quit' to exit): ").strip()
        if _is_quit(follow_up):
            return
        query = follow_up


async def main() -> None:
    """CLI entrypoint for the basic research agent."""
    parser = argparse.ArgumentParser(description="Basic research agent with Wikipedia tools.")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string.")
    parser.add_argument("--max-iters", type=int, default=6, help="Max tool execution iterations.")
    args = parser.parse_args()

    await basic_research_interactive(model=args.model, max_tool_iterations=args.max_iters)


if __name__ == "__main__":
    asyncio.run(main())

