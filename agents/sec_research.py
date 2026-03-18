"""SEC EDGAR research agent (tool-using with interactive Q&A).

This agent:
- accepts a user-provided natural language question at runtime
- uses the EDGAR tools to retrieve filing metadata and primary document text
- responds with a grounded answer and citations
- supports a follow-up Q&A loop until the user types `quit`

If evidence is insufficient to answer, the agent responds gracefully and does
not fabricate citations.
"""

from __future__ import annotations

import argparse
import asyncio
from pydantic import BaseModel, Field

from gluellm.api import GlueLLM

from tools.edgar_tools import fetch_filing_text, list_recent_filings, search_company


class FilingCitation(BaseModel):
    """Citation metadata used in SEC research answers."""

    form: str = Field(description="SEC form type")
    filing_date: str = Field(description="Filing date (YYYY-MM-DD)")
    accession_number: str = Field(description="SEC accession number")
    primary_document: str = Field(description="Primary document filename")
    filing_url: str | None = Field(
        default=None,
        description="Direct EDGAR archives URL for the primary document (when available).",
    )


class SecAnswer(BaseModel):
    """Structured answer model for SEC research."""

    can_answer: bool = Field(description="Whether the provided evidence is sufficient to answer.")
    answer: str = Field(description="The grounded answer text (only if can_answer is true).")
    citations: list[FilingCitation] = Field(
        description="List of filing citations used to answer the question (empty if can_answer is false)."
    )
    cannot_answer_reason: str | None = Field(description="Reason for inability to answer (if can_answer is false).")


def _is_quit(text: str) -> bool:
    """Return True if the user wants to exit the interactive loop."""
    return text.strip().lower() == "quit"


def _make_client(*, model: str, max_tool_iterations: int) -> GlueLLM:
    """Create a GlueLLM client configured with SEC tools."""
    return GlueLLM(
        model=model,
        system_prompt=(
            "You are a diligent SEC filings research assistant. "
            "You must use the provided tools to gather evidence. "
            "When you answer, include citations with form, filing date, accession number, and the primary document URL. "
            "If the tools return no relevant filings or insufficient details, set can_answer=false "
            "and explain cannot_answer_reason. Do not guess."
        ),
        tools=[search_company, list_recent_filings, fetch_filing_text],
        max_tool_iterations=max_tool_iterations,
    )


def _format_citations(citations: list[FilingCitation]) -> str:
    """Format citations for printing."""
    if not citations:
        return ""
    lines: list[str] = []
    for c in citations:
        url_part = c.filing_url or "URL unavailable"
        lines.append(f"- {c.form} | {c.filing_date} | {c.accession_number} | {url_part}")
    return "\n".join(lines)


async def _run_turn(client: GlueLLM, question: str) -> SecAnswer:
    """Run one question/follow-up turn and return structured answer."""
    result = await client.structured_complete(question, response_format=SecAnswer)
    return result.structured_output


async def sec_research_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the interactive SEC EDGAR research loop."""
    client = _make_client(model=model, max_tool_iterations=max_tool_iterations)

    question = input("Enter your SEC research question (or type 'quit' to exit): ").strip()
    while question and _is_quit(question):
        return
    while not question:
        question = input("Question cannot be empty. Enter your SEC research question (or 'quit'): ").strip()
        if _is_quit(question):
            return

    while True:
        answer = await _run_turn(client, question)
        # Evidence guard: avoid presenting an answer without citations.
        if answer.can_answer and not answer.citations:
            answer = SecAnswer(
                can_answer=False,
                answer="",
                citations=[],
                cannot_answer_reason="The model did not provide any filing citations/evidence for this question.",
            )

        if answer.can_answer:
            print("\nAnswer\n" + "-" * 40)
            print(answer.answer)
            print("\nCitations\n" + "-" * 40)
            print(_format_citations(answer.citations))
        else:
            reason = answer.cannot_answer_reason or "Insufficient evidence from retrieved filings."
            print("\nI cannot answer that question based on the retrieved SEC evidence.")
            print("Reason: " + reason)

        follow_up = input("\nFollow-up (or type 'quit' to exit): ").strip()
        if _is_quit(follow_up):
            return
        question = follow_up


async def main() -> None:
    """CLI entrypoint for SEC EDGAR research."""
    parser = argparse.ArgumentParser(description="SEC EDGAR research agent with interactive follow-ups.")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string.")
    parser.add_argument("--max-iters", type=int, default=6, help="Max tool execution iterations.")
    args = parser.parse_args()

    await sec_research_interactive(model=args.model, max_tool_iterations=args.max_iters)


if __name__ == "__main__":
    asyncio.run(main())

