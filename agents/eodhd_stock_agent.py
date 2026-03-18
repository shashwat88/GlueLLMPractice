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

from gluellm.api import GlueLLM

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


def _make_client(*, model: str, stock_symbol: str, max_tool_iterations: int) -> GlueLLM:
    """Create a GlueLLM client configured with the quote tool."""

    def get_realtime_quote_tool() -> dict[str, Any]:
        """Fetch quote evidence for the configured stock symbol."""
        return fetch_realtime_quote(stock_symbol=stock_symbol)

    system_prompt = (
        "You are a real-time stock research assistant for a fixed US stock symbol. "
        "You can answer questions ONLY about real-time quote facts from EODHD: "
        "current/last price, open/high/low, volume, and other quote fields present in the fetched evidence. "
        "You MUST call the tool to fetch evidence when needed. "
        "If the user asks something unrelated to stock quote facts (or cannot be answered with the evidence fields), "
        "set can_answer=false and explain cannot_answer_reason. "
        "Do not guess values or fabricate numbers."
        f"\n\nFixed stock_symbol for this session: {stock_symbol}."
    )

    return GlueLLM(
        model=model,
        system_prompt=system_prompt,
        tools=[get_realtime_quote_tool],
        max_tool_iterations=max_tool_iterations,
    )


async def _run_turn(client: GlueLLM, question: str, stock_symbol: str) -> StockAnswer:
    """Run one structured Q&A turn for the given question."""
    response = await client.structured_complete(
        question,
        response_format=StockAnswer,
    )
    answer = response.structured_output
    # Ensure the response is consistent with the session symbol.
    if answer.stock_symbol.strip().upper() != stock_symbol.strip().upper():
        return StockAnswer(
            can_answer=False,
            stock_symbol=stock_symbol,
            answer="",
            cannot_answer_reason="The question appears to reference a different symbol than the one selected for this session.",
        )
    return answer


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

    client = _make_client(model=model, stock_symbol=stock_symbol, max_tool_iterations=max_tool_iterations)

    question = input("Stock question (or type 'quit' to exit): ").strip()
    if _is_quit(question):
        return
    while not question:
        question = input("Question cannot be empty. Stock question (or 'quit'): ").strip()
        if _is_quit(question):
            return

    while True:
        answer = await _run_turn(client, question, stock_symbol=stock_symbol)
        if answer.can_answer:
            print("\nAnswer\n" + "-" * 40)
            print(answer.answer)
        else:
            reason = answer.cannot_answer_reason or "Insufficient evidence."
            print("\nI cannot answer that question based on the available real-time stock evidence.")
            print("Reason: " + reason)

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

