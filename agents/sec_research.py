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

from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt

from tools.edgar_tools import fetch_filing_text, list_recent_filings, search_company
from core.workflow_wrappers import run_reflection_workflow_parsed


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

def _build_conversation_history(history: list[tuple[str, str]]) -> str:
    """Render prior Q/A into a compact text block for the workflow prompt."""
    if not history:
        return "No prior conversation."
    parts: list[str] = []
    for i, (q, a) in enumerate(history, 1):
        parts.append(f"[Turn {i}] User: {q}\nAssistant: {a}")
    return "\n\n".join(parts)


def _build_generator_agent(*, model: str, max_tool_iterations: int) -> Agent:
    """Create the generator agent that outputs strict JSON evidence + citations."""
    system_prompt = (
        "You are a diligent SEC filings research assistant. "
        "Use the provided tools to gather evidence from SEC EDGAR. "
        "You MUST NOT fabricate filings, dates, accession numbers, or URLs. "
        "When you answer, you must output ONLY valid JSON matching the schema:\n"
        "{\n"
        '  "can_answer": boolean,\n'
        '  "answer": string, \n'
        '  "citations": [\n'
        "    {\n"
        '      "form": string,\n'
        '      "filing_date": string,\n'
        '      "accession_number": string,\n'
        '      "primary_document": string,\n'
        '      "filing_url": string|null\n'
        "    }\n"
        "  ],\n"
        '  "cannot_answer_reason": string|null\n'
        "}\n"
        "Rules:\n"
        "- If can_answer is false: answer must be \"\", citations must be [], and cannot_answer_reason must be a helpful explanation.\n"
        "- If can_answer is true: citations must be a non-empty list.\n"
    )
    return Agent(
        name="SECGenerator",
        description="Generator that retrieves SEC evidence and returns strict JSON.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[search_company, list_recent_filings, fetch_filing_text],
        max_tool_iterations=max_tool_iterations,
        model=model,
    )


def _build_reflector_agent(*, model: str) -> Agent:
    """Create the reflector agent that validates evidence/citations consistency."""
    system_prompt = (
        "You are a compliance reflector for SEC research. "
        "You receive the generator's output and are asked to validate it. "
        "Return ONLY valid JSON matching the same schema. "
        "If citations are missing or inconsistent with can_answer, set can_answer=false "
        "with a clear cannot_answer_reason. Never fabricate evidence."
    )
    return Agent(
        name="SECReflector",
        description="Reflector that validates JSON and evidence consistency.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[],
        max_tool_iterations=1,
        model=model,
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


async def _answer_with_workflow(
    *,
    model: str,
    max_tool_iterations: int,
    question: str,
    history: list[tuple[str, str]],
) -> SecAnswer:
    """Answer a single question using ReflectionWorkflow and parse strict JSON."""
    generator = _build_generator_agent(model=model, max_tool_iterations=max_tool_iterations)
    reflector = _build_reflector_agent(model=model)
    history_text = _build_conversation_history(history)
    initial_input = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Current user question:\n{question}\n\n"
        "Return the JSON object only."
    )

    def _on_parse_error(raw: str, e: Exception) -> SecAnswer:
        _ = raw
        return SecAnswer(
            can_answer=False,
            answer="",
            citations=[],
            cannot_answer_reason=f"Workflow output could not be parsed as valid SEC JSON: {type(e).__name__}",
        )

    parsed = await run_reflection_workflow_parsed(
        initial_input=initial_input,
        response_model=SecAnswer,
        generator_agent=generator,
        reflector_agent=reflector,
        max_reflections=2,
        on_parse_error=_on_parse_error,
    )

    # Evidence guard: avoid presenting an answer without citations.
    if parsed.can_answer and not parsed.citations:
        return SecAnswer(
            can_answer=False,
            answer="",
            citations=[],
            cannot_answer_reason="Answer claimed can_answer=true but no citations were provided.",
        )
    return parsed


async def sec_research_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the interactive SEC EDGAR research loop."""
    history: list[tuple[str, str]] = []

    question = input("Enter your SEC research question (or type 'quit' to exit): ").strip()
    while question and _is_quit(question):
        return
    while not question:
        question = input("Question cannot be empty. Enter your SEC research question (or 'quit'): ").strip()
        if _is_quit(question):
            return

    while True:
        answer = await _answer_with_workflow(
            model=model,
            max_tool_iterations=max_tool_iterations,
            question=question,
            history=history,
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

        assistant_text = answer.answer if answer.can_answer else f"Cannot answer: {reason}"
        history.append((question, assistant_text))

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

