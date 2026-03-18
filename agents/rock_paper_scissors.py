"""Rock Paper Scissors multi-agent loop (two LLM players + deterministic referee).

This script runs a configurable number of rounds. Each round:
- Player A chooses a move (rock/paper/scissors)
- Player B chooses a move (rock/paper/scissors)
- The controller computes the winner and updates the score

The move choice is validated using structured output (Pydantic), so invalid
model outputs are retried by the controller.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal

from pydantic import BaseModel

from core.base_agent import BaseAgent
from core.loop import RPSGameResult, run_rps_game


class RPSMove(BaseModel):
    """Validated rock-paper-scissors move."""

    move: Literal["rock", "paper", "scissors"]


def _build_player_agent(*, name: str, system_prompt: str) -> BaseAgent[RPSMove]:
    """Create an LLM-backed RPS player agent."""
    return BaseAgent[RPSMove](name=name, system_prompt=system_prompt)


async def _choose_move(player: BaseAgent[RPSMove], round_index: int) -> str:
    """Ask a player agent to output a single RPS move."""
    user_message = f"Round {round_index}. Output exactly one move: rock, paper, or scissors."
    structured = await player.complete_structured(user_message, response_format=RPSMove)
    return structured.move


def _print_final(result: RPSGameResult) -> None:
    """Print final scoreboard."""
    print("\nFinal score")
    print("-" * 40)
    print(f"Rounds: {result.rounds}")
    print(f"Player A: {result.score_a}")
    print(f"Player B: {result.score_b}")
    print(f"Draws:    {result.draws}")
    print("-" * 40)
    if result.score_a > result.score_b:
        print("Winner: Player A")
    elif result.score_b > result.score_a:
        print("Winner: Player B")
    else:
        print("Winner: Draw")


async def run(rounds: int) -> RPSGameResult:
    """Run a full RPS game and return the result."""
    player_a = _build_player_agent(
        name="PlayerA",
        system_prompt=(
            "You are Player A in a rock-paper-scissors game. "
            "When asked, output your move using structured output."
        ),
    )
    player_b = _build_player_agent(
        name="PlayerB",
        system_prompt=(
            "You are Player B in a rock-paper-scissors game. "
            "When asked, output your move using structured output."
        ),
    )

    def on_round(round_history) -> None:
        """Print the result of one completed round."""
        winner_label = round_history.winner
        if winner_label == "a":
            outcome = "Player A wins"
        elif winner_label == "b":
            outcome = "Player B wins"
        else:
            outcome = "Draw"
        print(
            f"Round {round_history.round_index}: "
            f"A={round_history.move_a} | B={round_history.move_b} -> {outcome}"
        )

    result = await run_rps_game(
        rounds=rounds,
        choose_a=lambda idx: _choose_move(player_a, idx),
        choose_b=lambda idx: _choose_move(player_b, idx),
        on_round=on_round,
    )
    _print_final(result)
    return result


async def main() -> None:
    """CLI entrypoint for running the RPS loop."""
    parser = argparse.ArgumentParser(description="Rock Paper Scissors multi-agent loop.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to play.")
    args = parser.parse_args()
    await run(args.rounds)


if __name__ == "__main__":
    asyncio.run(main())

