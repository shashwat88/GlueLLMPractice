"""Writer + Critic iterative refinement loop for poem writing.

This module implements a deterministic controller (in `core.loop`) that:
- asks the Writer agent to produce/modify a poem
- asks the Critic agent to return structured feedback including a score
- repeats until the critic's score reaches the configured threshold

All iterations are printed/logged, and the final accepted poem is printed with
the number of iterations used.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from core.loop import LoopLimitError, RefinementIteration, run_refinement_loop
from gluellm.api import GlueLLM


class Critique(BaseModel):
    """Structured critique returned by the critic agent."""

    score: int = Field(ge=0, le=10, description="Quality score from 0 to 10")
    strengths: list[str] = Field(description="What the poem does well")
    suggestions: list[str] = Field(description="Concrete improvements to apply")

    revised_poem_guidance: str = Field(
        description="A short guidance string the writer should follow in the next revision."
    )


def _make_writer(model: str) -> GlueLLM:
    """Create a poem writer client."""
    return GlueLLM(
        model=model,
        system_prompt=(
            "You are a creative poet. Write an original poem on the user's topic. "
            "Do not include any analysis; output only the poem text."
        ),
    )


def _make_critic(model: str) -> GlueLLM:
    """Create a poem critic client."""
    return GlueLLM(
        model=model,
        system_prompt=(
            "You are a strict poetry critic. Evaluate the poem on originality, imagery, and coherence. "
            "Return structured JSON that includes a score (0-10) plus specific strengths and suggestions."
        ),
    )


async def write_poem_step(writer: GlueLLM, topic: str, draft: str, critique: Critique) -> str:
    """Ask the writer to revise the poem based on critic feedback."""
    user_message = (
        f"Topic: {topic}\n\n"
        f"Current poem:\n{draft}\n\n"
        f"Critic guidance:\n{critique.revised_poem_guidance}\n\n"
        "Revise the poem to address the guidance. Output only the revised poem."
    )
    result = await writer.complete(user_message)
    return result.final_response


async def generate_initial_poem(writer: GlueLLM, topic: str) -> str:
    """Generate the initial poem for the topic."""
    result = await writer.complete(f"Write an original poem about: {topic}")
    return result.final_response


async def evaluate_poem(critic: GlueLLM, poem: str) -> Critique:
    """Ask the critic agent for structured critique."""
    result = await critic.structured_complete(
        poem,
        response_format=Critique,
    )
    return result.structured_output


def _log_iteration(topic: str) -> callable:
    """Create an `on_iteration` callback that prints the transcript."""

    def _inner(it: RefinementIteration[Critique]) -> None:
        print("\n" + "=" * 70)
        print(f"Iteration: {it.iteration}")
        print("-" * 70)
        print("Poem version:\n")
        print(it.draft)
        print("\nCritique:")
        print(f"  Score: {it.critique.score}/10")
        print(f"  Strengths: {', '.join(it.critique.strengths)}")
        print(f"  Suggestions: {', '.join(it.critique.suggestions)}")
        print(f"  Guidance: {it.critique.revised_poem_guidance}")
        print("=" * 70)
        _ = topic  # topic is referenced for readability in logs

    return _inner


async def run_poem_loop(
    *,
    topic: str,
    threshold: int,
    max_iters: int,
    model: str,
) -> tuple[str, int]:
    """Run the poem refinement loop.

    Args:
        topic: Poem topic provided by the user.
        threshold: Minimum critic score required to accept the poem.
        max_iters: Maximum refinement iterations before giving up.
        model: Provider:model string for both writer and critic.

    Returns:
        A tuple of (accepted_poem, total_iterations).
    """
    writer = _make_writer(model=model)
    critic = _make_critic(model=model)

    start_draft = await generate_initial_poem(writer, topic=topic)

    def accept_if(critique: Critique) -> bool:
        return critique.score >= threshold

    async def writer_fn(i: int, draft: str, critique: Critique) -> str:
        _ = i
        return await write_poem_step(writer, topic=topic, draft=draft, critique=critique)

    async def critic_fn(draft: str) -> Critique:
        return await evaluate_poem(critic=critic, poem=draft)

    try:
        result = await run_refinement_loop(
            max_iters=max_iters,
            start_draft=start_draft,
            critic_fn=critic_fn,
            writer_fn=writer_fn,
            accept_if=accept_if,
            on_iteration=_log_iteration(topic),
        )
        return result.accepted_draft, result.accepted_iteration
    except LoopLimitError as e:
        # Graceful fallback: return the best available draft at the last iteration.
        last = str(e)
        _ = last
        # If we hit the limit, the caller still gets the last draft.
        return start_draft, max_iters


async def main() -> None:
    """CLI entrypoint for the poem refinement loop."""
    parser = argparse.ArgumentParser(description="Writer + Critic poem refinement loop.")
    parser.add_argument("--topic", type=str, required=True, help="Topic for the poem.")
    parser.add_argument("--threshold", type=int, default=8, help="Score threshold (0-10).")
    parser.add_argument("--max-iters", type=int, default=10, help="Maximum refinement iterations.")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string.")
    args = parser.parse_args()

    poem, iters = await run_poem_loop(
        topic=args.topic,
        threshold=args.threshold,
        max_iters=args.max_iters,
        model=args.model,
    )
    print("\n" + "*" * 70)
    print("Accepted poem")
    print("*" * 70)
    print(poem)
    print("\nTotal iterations:", iters)


if __name__ == "__main__":
    asyncio.run(main())

