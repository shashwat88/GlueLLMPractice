"""Deterministic loop controllers shared across agents.

The goal of this module is to keep multi-round orchestration logic small,
testable, and reusable. Individual agents provide the "async decision makers"
(LLM calls or stubbed responses) while this module handles:
- iteration counting / stopping conditions
- validation / normalization
- score updates (for deterministic games like rock-paper-scissors)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

RPSMove = Literal["rock", "paper", "scissors"]


class LoopLimitError(RuntimeError):
    """Raised when a refinement loop hits its maximum number of iterations."""


def normalize_rps_move(move: str) -> RPSMove:
    """Normalize a raw move into the `RPSMove` domain.

    Args:
        move: Raw model output.

    Returns:
        A normalized `RPSMove`.

    Raises:
        ValueError: If the move is not recognized.
    """
    normalized = move.strip().lower()
    if normalized in {"rock", "paper", "scissors"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Invalid move: {move!r}")


def rps_winner(a: RPSMove, b: RPSMove) -> Literal["a", "b", "draw"]:
    """Return the winner label for a rock-paper-scissors match.

    Args:
        a: Player A move.
        b: Player B move.

    Returns:
        "a" if A wins, "b" if B wins, otherwise "draw".
    """
    if a == b:
        return "draw"
    if (
        (a == "rock" and b == "scissors")
        or (a == "paper" and b == "rock")
        or (a == "scissors" and b == "paper")
    ):
        return "a"
    return "b"


@dataclass(frozen=True)
class RPSRoundHistory:
    """One completed round of rock-paper-scissors."""

    round_index: int
    move_a: RPSMove
    move_b: RPSMove
    winner: Literal["a", "b", "draw"]


@dataclass(frozen=True)
class RPSGameResult:
    """Final result for a rock-paper-scissors game."""

    rounds: int
    score_a: int
    score_b: int
    draws: int
    history: list[RPSRoundHistory]


async def run_rps_game(
    *,
    rounds: int,
    choose_a: Callable[[int], Awaitable[str]],
    choose_b: Callable[[int], Awaitable[str]],
    max_retries_per_move: int = 2,
    on_round: Callable[[RPSRoundHistory], None] | None = None,
) -> RPSGameResult:
    """Run a multi-round deterministic rock-paper-scissors game.

    Args:
        rounds: Number of rounds to play.
        choose_a: Async function that returns Player A's raw move string.
        choose_b: Async function that returns Player B's raw move string.
        max_retries_per_move: How many times to re-try move validation on invalid output.
        on_round: Optional callback invoked after each successful round.

    Returns:
        The final `RPSGameResult`.

    Raises:
        ValueError: If a move cannot be normalized after retries.
    """
    score_a = 0
    score_b = 0
    draws = 0
    history: list[RPSRoundHistory] = []

    for i in range(1, rounds + 1):
        raw_a = None
        last_err: Exception | None = None
        for _ in range(max_retries_per_move + 1):
            try:
                raw_a = await choose_a(i)
                move_a = normalize_rps_move(raw_a)
                break
            except Exception as e:  # noqa: BLE001 - we re-raise after retries
                last_err = e
        else:  # pragma: no cover
            raise ValueError(f"Failed to obtain a valid move for A: {last_err}") from last_err

        raw_b = None
        last_err = None
        for _ in range(max_retries_per_move + 1):
            try:
                raw_b = await choose_b(i)
                move_b = normalize_rps_move(raw_b)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        else:  # pragma: no cover
            raise ValueError(f"Failed to obtain a valid move for B: {last_err}") from last_err

        winner = rps_winner(move_a, move_b)
        if winner == "a":
            score_a += 1
        elif winner == "b":
            score_b += 1
        else:
            draws += 1

        round_history = RPSRoundHistory(round_index=i, move_a=move_a, move_b=move_b, winner=winner)
        history.append(round_history)
        if on_round is not None:
            on_round(round_history)

    return RPSGameResult(
        rounds=rounds, score_a=score_a, score_b=score_b, draws=draws, history=history
    )


CritiqueT = TypeVar("CritiqueT", bound=BaseModel)


@dataclass(frozen=True)
class RefinementIteration(Generic[CritiqueT]):
    """State for a single refinement iteration."""

    iteration: int
    draft: str
    critique: CritiqueT


@dataclass(frozen=True)
class RefinementResult(Generic[CritiqueT]):
    """Final state for a refinement loop."""

    accepted_draft: str
    accepted_iteration: int
    final_critique: CritiqueT
    iterations: list[RefinementIteration[CritiqueT]]


async def run_refinement_loop(
    *,
    max_iters: int,
    start_draft: str,
    critic_fn: Callable[[str], Awaitable[CritiqueT]],
    writer_fn: Callable[[int, str, CritiqueT], Awaitable[str]],
    accept_if: Callable[[CritiqueT], bool],
    on_iteration: Callable[[RefinementIteration[CritiqueT]], None] | None = None,
) -> RefinementResult[CritiqueT]:
    """Run an iterative refinement loop until the critique passes a threshold.

    Args:
        max_iters: Maximum number of iterations before failing.
        start_draft: Initial draft text (e.g., poem produced by writer in iteration 0).
        critic_fn: Async function that evaluates a draft and returns structured critique.
        writer_fn: Async function that produces a revised draft given iteration, current draft, and critique.
        accept_if: Predicate over the critique that determines acceptance.
        on_iteration: Optional callback invoked after each critic evaluation.

    Returns:
        The accepted draft and the full iteration history.

    Raises:
        LoopLimitError: If `max_iters` is reached without acceptance.
    """
    iterations: list[RefinementIteration[CritiqueT]] = []
    current_draft = start_draft

    for i in range(1, max_iters + 1):
        critique = await critic_fn(current_draft)
        iteration = RefinementIteration(iteration=i, draft=current_draft, critique=critique)
        iterations.append(iteration)
        if on_iteration is not None:
            on_iteration(iteration)

        if accept_if(critique):
            return RefinementResult(
                accepted_draft=current_draft,
                accepted_iteration=i,
                final_critique=critique,
                iterations=iterations,
            )

        current_draft = await writer_fn(i, current_draft, critique)

    raise LoopLimitError(f"Refinement loop did not converge within {max_iters} iterations.")
