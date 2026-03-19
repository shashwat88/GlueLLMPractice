"""Interactive EODHD real-time stock research agent.

This agent provides a CLI loop:
- ask the user for a stock symbol
- ask a natural-language question about that symbol
- answer the question using EODHD real-time quote data (via a tool call)
- accept follow-up questions until the user types `quit`

If a question is not relevant to real-time stock data, or if the fetched
evidence is insufficient, the agent responds gracefully with
`can_answer=false`.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from pydantic import BaseModel, Field

from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt 
from core.workflow_wrappers import run_reflection_workflow_parsed

from tools.eodhd_tools import fetch_realtime_quote, normalize_stock_symbol


class StockAnswer(BaseModel):
    """Structured answer produced by the stock agent."""

    can_answer: bool = Field(description="Whether the question can be answered from fetched quote evidence.")
    stock_symbol: str = Field(description="The stock symbol this answer refers to.")
    answer: str = Field(description="Answer text (only if can_answer=true).")
    cannot_answer_reason: str | None = Field(
        default=None,
        description="Reason for inability to answer (only if can_answer=false).",
    )


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


def _build_generator_agent(
    *,
    model: str,
    stock_symbol: str,
    max_tool_iterations: int,
) -> Agent:
    """Create generator agent that outputs strict StockAnswer JSON with evidence."""

    def get_realtime_quote_tool() -> dict[str, Any]:
        """Fetch quote evidence for the configured stock symbol."""
        return fetch_realtime_quote(stock_symbol=stock_symbol)

    system_prompt = (
        "You are a real-time stock research assistant for a fixed US stock symbol. "
        "You can answer questions ONLY about real-time quote facts from EODHD evidence "
        "(price, open/high/low, volume, and other quote fields present in the fetched evidence). "
        "You MUST call the provided tool to fetch evidence when needed. "
        "If the user asks an unrelated question or cannot be answered from the evidence fields, "
        "set can_answer=false and explain cannot_answer_reason. "
        "Do not guess values or fabricate numbers. "
        "Output ONLY valid JSON matching the StockAnswer schema:\n"
        "{\n"
        '  "can_answer": boolean,\n'
        '  "stock_symbol": string,\n'
        '  "answer": string,\n'
        '  "cannot_answer_reason": string|null\n'
        "}\n"
        f"\nFixed stock_symbol for this session: {stock_symbol}.\n"
        "Rules:\n"
        "- If can_answer=false: answer must be \"\" and cannot_answer_reason must be non-empty.\n"
        "- If can_answer=true: answer must be non-empty."
    )
    return Agent(
        name="StockGenerator",
        description="Generator that fetches quote evidence and returns StockAnswer JSON.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[get_realtime_quote_tool],
        max_tool_iterations=max_tool_iterations,
        model=model,
    )


def _build_reflector_agent(*, model: str) -> Agent:
    """Create reflector agent that validates stock answer relevance."""
    system_prompt = (
        "You are a compliance reflector for stock quote Q&A. "
        "Validate the generator's JSON for StockAnswer. "
        "If the question is unrelated to real-time quote facts, set can_answer=false. "
        "Return ONLY valid JSON matching the same StockAnswer schema."
    )
    return Agent(
        name="StockReflector",
        description="Reflector that validates StockAnswer JSON.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[],
        max_tool_iterations=1,
        model=model,
    )


async def _answer_with_workflow(
    *,
    model: str,
    max_tool_iterations: int,
    stock_symbol: str,
    question: str,
    history: list[tuple[str, str]],
) -> StockAnswer:
    """Answer one stock question using ReflectionWorkflow and parse strict JSON."""
    generator = _build_generator_agent(model=model, stock_symbol=stock_symbol, max_tool_iterations=max_tool_iterations)
    reflector = _build_reflector_agent(model=model)
    history_text = _build_conversation_history(history)
    initial_input = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Current user question:\n{question}\n\n"
        "Return the JSON object only."
    )

    def _on_parse_error(raw: str, e: Exception) -> StockAnswer:
        _ = raw
        return StockAnswer(
            can_answer=False,
            stock_symbol=stock_symbol,
            answer="",
            cannot_answer_reason=f"Workflow output could not be parsed as valid Stock JSON: {type(e).__name__}",
        )

    parsed = await run_reflection_workflow_parsed(
        initial_input=initial_input,
        response_model=StockAnswer,
        generator_agent=generator,
        reflector_agent=reflector,
        max_reflections=2,
        on_parse_error=_on_parse_error,
    )

    # Evidence guard: ensure symbol consistency and non-empty answer when can_answer=true.
    if parsed.can_answer:
        if parsed.stock_symbol.strip().upper() != stock_symbol.strip().upper() or not parsed.answer.strip():
            return StockAnswer(
                can_answer=False,
                stock_symbol=stock_symbol,
                answer="",
                cannot_answer_reason="The answer was not consistent with the selected session stock symbol.",
            )
    return parsed


async def eodhd_stock_agent_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the interactive stock Q&A loop."""
    symbol_input = input("Enter a stock symbol (e.g., AAPL): ").strip()
    if _is_quit(symbol_input):
        return
    try:
        stock_symbol = normalize_stock_symbol(symbol_input)
    except ValueError as e:
        print(f"Invalid stock symbol: {e}")
        return

    question = input("Stock question (or type 'quit' to exit): ").strip()
    if _is_quit(question):
        return
    while not question:
        question = input("Question cannot be empty. Stock question (or 'quit'): ").strip()
        if _is_quit(question):
            return

    history: list[tuple[str, str]] = []
    while True:
        answer = await _answer_with_workflow(
            model=model,
            max_tool_iterations=max_tool_iterations,
            stock_symbol=stock_symbol,
            question=question,
            history=history,
        )
        if answer.can_answer:
            print("\nAnswer\n" + "-" * 40)
            print(answer.answer)
        else:
            reason = answer.cannot_answer_reason or "Insufficient evidence."
            print("\nI cannot answer that question based on the available real-time stock evidence.")
            print("Reason: " + reason)

        assistant_text = answer.answer if answer.can_answer else f"Cannot answer: {reason}"
        history.append((question, assistant_text))

        follow_up = input("\nFollow-up (or type 'quit' to exit): ").strip()
        if _is_quit(follow_up):
            return
        question = follow_up


async def main() -> None:
    """CLI entrypoint for the EODHD stock research agent."""
    parser = argparse.ArgumentParser(description="EODHD real-time stock research agent.")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string.")
    parser.add_argument("--max-iters", type=int, default=6, help="Max tool execution iterations.")
    args = parser.parse_args()

    await eodhd_stock_agent_interactive(model=args.model, max_tool_iterations=args.max_iters)


if __name__ == "__main__":
    asyncio.run(main())

