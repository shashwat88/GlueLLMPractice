"""Poem refinement loop using GlueLLM's iterative refinement workflow.

This module uses GlueLLM's `IterativeRefinementWorkflow` with:
- a producer agent ("writer") that generates the poem
- a critic agent that evaluates the poem and returns JSON feedback including a score

The loop stops when the critic's score reaches the configured `--threshold`
(default: >= 8), and logs each iteration's poem + critique + parsed score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re

from gluellm.executors import AgentExecutor
from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt
from gluellm.models.workflow import CriticConfig, IterativeConfig
from gluellm.workflows.iterative import IterativeRefinementWorkflow
from pydantic import BaseModel, Field

from core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


class Critique(BaseModel):
    """Structured critique expected from the critic agent."""

    score: int = Field(ge=0, le=10, description="Quality score from 0 to 10")
    strengths: list[str] = Field(default_factory=list, description="What the poem does well")
    suggestions: list[str] = Field(
        default_factory=list, description="Concrete improvements to apply"
    )
    revised_poem_guidance: str = Field(description="Guidance for the next revision")


def _parse_score(feedback: str) -> int | None:
    """Parse the critic's score from a free-form feedback string.

    The workflow critic prompt asks for JSON, but we keep parsing resilient by
    looking for either:
    - a JSON `"score": <int>` field
    - a plain `score: <int>` / `Score: <int>` pattern
    """
    if not feedback:
        return None

    # Try JSON first.
    try:
        obj = json.loads(feedback)
        if isinstance(obj, dict):
            parsed = Critique.model_validate(obj)
            return parsed.score
    except Exception:
        # Fall back to regex parsing below.
        pass

    match = re.search(r"\"?score\"?\s*:\s*(\d{1,2})", feedback, flags=re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 10:
            return val
    return None


def _build_writer_agent(*, model: str) -> Agent:
    """Create the poem writer agent."""
    return Agent(
        name="Writer",
        description="Creative poet who writes an original poem and revises it when given feedback.",
        system_prompt=SystemPrompt(
            content=(
                "You are a creative poet.\n"
                "Write an original poem about the given topic.\n"
                "When given previous attempts and critic feedback, revise the poem accordingly.\n"
                "Output only the poem text (no JSON, no commentary)."
            )
        ),
        tools=[],
        max_tool_iterations=5,
        model=model,
    )


def _build_critic_agent(*, model: str) -> Agent:
    """Create the poem critic agent."""
    return Agent(
        name="Critic",
        description="Strict poetry critic that evaluates poems and returns JSON with a score.",
        system_prompt=SystemPrompt(
            content=(
                "You are a strict poetry critic.\n"
                "Evaluate the poem for originality, imagery, coherence, and overall quality.\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\n"
                '  "score": integer 0-10,\n'
                '  "strengths": string array,\n'
                '  "suggestions": string array,\n'
                '  "revised_poem_guidance": string\n'
                "}\n"
                "Do not wrap the JSON in backticks."
            )
        ),
        tools=[],
        max_tool_iterations=5,
        model=model,
    )


def _format_iteration_log(
    *, iteration: int, poem: str | None, critic_feedback: str | None, score: int | None
) -> None:
    """Print one iteration transcript."""
    print("\n" + "=" * 70)
    print(f"Iteration: {iteration}")
    print("-" * 70)
    if poem is not None:
        print("Poem version:\n")
        print(poem)
    if critic_feedback is not None:
        print("\nCritique:")
        if score is not None:
            print(f"  Parsed score: {score}/10")
        else:
            print("  Parsed score: unavailable")
        print(critic_feedback)
    print("=" * 70)


async def run_poem_workflow(
    *,
    topic: str,
    threshold: int,
    max_iters: int,
    model: str,
) -> tuple[str, int]:
    """Run poem generation + critique refinement using GlueLLM workflow."""
    logger.info(
        "Starting poem workflow: topic=%r threshold=%s max_iters=%s model=%s",
        topic,
        threshold,
        max_iters,
        model,
    )
    writer_agent = _build_writer_agent(model=model)
    critic_agent = _build_critic_agent(model=model)

    producer = AgentExecutor(agent=writer_agent)
    critic_executor = AgentExecutor(agent=critic_agent)

    def quality_evaluator(content: str, critique: dict[str, str]) -> float:
        """Convert critic output into a normalized [0.0, 1.0] score."""
        _ = content
        if not critique:
            return 0.0
        # `IterativeRefinementWorkflow` passes a dict specialty -> critic_output.
        feedback_str = next(iter(critique.values()), "")
        score = _parse_score(feedback_str)
        if score is None:
            return 0.0
        return score / 10.0

    workflow = IterativeRefinementWorkflow(
        producer=producer,
        critics=[
            CriticConfig(
                executor=critic_executor,
                specialty="poetry_quality",
                goal="Provide constructive JSON feedback and an integer score from 0 to 10.",
            )
        ],
        config=IterativeConfig(
            max_iterations=max_iters,
            min_quality_score=threshold / 10.0,
            quality_evaluator=quality_evaluator,
        ),
    )

    initial_input = f"Topic: {topic}\nWrite an original poem about the topic."
    result = await workflow.execute(initial_input)

    # Per-iteration logging.
    poem_by_iter: dict[int, str] = {}
    feedback_by_iter: dict[int, str] = {}
    for interaction in result.agent_interactions:
        it = int(interaction.get("iteration") or 0)
        agent_name = str(interaction.get("agent") or "")
        output = str(interaction.get("output") or "")
        if agent_name == "producer":
            poem_by_iter[it] = output
        elif agent_name.startswith("critic_"):
            feedback_by_iter[it] = output

    for i in range(1, result.iterations + 1):
        poem = poem_by_iter.get(i)
        feedback = feedback_by_iter.get(i)
        score = _parse_score(feedback) if feedback is not None else None
        _format_iteration_log(iteration=i, poem=poem, critic_feedback=feedback, score=score)
        logger.info("Poem iteration=%s score=%s", i, score)

    return result.final_output, result.iterations


async def main() -> None:
    """CLI entrypoint for the poem refinement workflow."""
    setup_logging()
    logger.info("CLI start: poem_loop")
    parser = argparse.ArgumentParser(
        description="Writer + Critic poem refinement loop (GlueLLM workflow)."
    )
    parser.add_argument("--topic", type=str, required=True, help="Topic for the poem.")
    parser.add_argument("--threshold", type=int, default=8, help="Score threshold (0-10).")
    parser.add_argument("--max-iters", type=int, default=10, help="Maximum refinement iterations.")
    parser.add_argument(
        "--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string."
    )
    args = parser.parse_args()

    poem, iters = await run_poem_workflow(
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
