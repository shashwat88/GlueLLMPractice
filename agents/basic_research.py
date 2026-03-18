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

from gluellm.api import GlueLLM

from tools.search_tools import search_and_summarize


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


def _make_client(*, model: str, max_tool_iterations: int) -> GlueLLM:
    """Create the GlueLLM client with Wikipedia tool access."""
    system_prompt = (
        "You are a research assistant. Use the provided Wikipedia tools to gather evidence. "
        "Synthesize results into a concise, structured answer. "
        "If the tools return no relevant sources, set can_answer=false and explain cannot_answer_reason. "
        "Do not guess or fabricate citations. "
        "In the output model, set `query` to the latest user question."
    )
    return GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=[search_and_summarize],
        max_tool_iterations=max_tool_iterations,
    )


def _extract_query_for_response(user_text: str) -> str:
    """Derive the query label for the response model."""
    t = user_text.strip()
    return t


async def _run_turn(client: GlueLLM, question: str) -> ResearchResponse:
    """Run one structured research turn."""
    response = await client.structured_complete(
        question,
        response_format=ResearchResponse,
    )
    return response.structured_output


async def basic_research_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the interactive basic research loop."""
    client = _make_client(model=model, max_tool_iterations=max_tool_iterations)

    query = input("Research query (or type 'quit' to exit): ").strip()
    if _is_quit(query):
        return
    while not query:
        query = input("Query cannot be empty. Research query (or 'quit'): ").strip()
        if _is_quit(query):
            return

    while True:
        question = query
        response = await _run_turn(client, question)

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

