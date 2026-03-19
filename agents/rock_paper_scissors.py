"""Rock Paper Scissors multi-agent loop using GlueLLM workflow.

Two agents (Player A and Player B) take turns choosing moves. A deterministic
referee computes the winner each round based on those moves.

The agents are orchestrated by GlueLLM's `RoundRobinWorkflow`. For each round
we print:
- the prompt (input) sent to each agent
- the raw move output from each agent
- the outcome (A wins / B wins / draw)
"""

from __future__ import annotations

import argparse
import asyncio

from gluellm.api import GlueLLM
from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt
from gluellm.models.workflow import RoundRobinConfig
from gluellm.workflows.round_robin import RoundRobinWorkflow

from core.loop import RPSGameResult, RPSRoundHistory, normalize_rps_move, rps_winner


class _AgentTextExecutor:
    """Executor adapter that returns plain text for workflows.

    The GlueLLM round-robin workflow expects `executor.execute()` to return a
    string contribution. Some GlueLLM executor implementations return an
    `ExecutionResult` object instead, which breaks prompt history formatting.
    """

    def __init__(self, agent: Agent):
        """Store the agent configuration used for execution."""
        self._agent = agent

    async def execute(self, query: str) -> str:
        """Execute query and return the final response text."""
        client = GlueLLM(
            model=self._agent.model,
            system_prompt=self._agent.system_prompt.content if self._agent.system_prompt else None,
            tools=self._agent.tools,
            max_tool_iterations=self._agent.max_tool_iterations,
            max_tokens=self._agent.max_tokens,
        )
        result = await client.complete(query)
        if hasattr(result, "final_response"):
            return str(getattr(result, "final_response"))
        if hasattr(result, "final_result"):  # pragma: no cover
            return str(getattr(result, "final_result"))
        raise AttributeError("ExecutionResult missing final response field.")


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
    player_a_agent = Agent(
        name="PlayerA",
        description="Rock-paper-scissors player A.",
        system_prompt=SystemPrompt(
            content=(
                "You are Player A. Output exactly one word: rock, paper, or scissors. "
                "Output only the word and nothing else."
            )
        ),
        tools=[],
        max_tool_iterations=1,
    )
    player_b_agent = Agent(
        name="PlayerB",
        description="Rock-paper-scissors player B.",
        system_prompt=SystemPrompt(
            content=(
                "You are Player B. Output exactly one word: rock, paper, or scissors. "
                "Output only the word and nothing else."
            )
        ),
        tools=[],
        max_tool_iterations=1,
    )
    player_a_executor = _AgentTextExecutor(player_a_agent)
    player_b_executor = _AgentTextExecutor(player_b_agent)

    score_a = 0
    score_b = 0
    draws = 0
    history: list[RPSRoundHistory] = []

    for round_idx in range(1, rounds + 1):
        move_a = None
        move_b = None
        a_raw = ""
        b_raw = ""
        winner = "draw"

        for attempt in range(1, 4):
            workflow = RoundRobinWorkflow(
                agents=[("PlayerA", player_a_executor), ("PlayerB", player_b_executor)],
                config=RoundRobinConfig(max_rounds=1, contribution_style="extend", final_synthesis=False),
            )
            initial_input = (
                f"Round {round_idx} of Rock Paper Scissors. "
                "Each agent must output exactly one word: rock, paper, or scissors. "
                "Output only the word and nothing else."
            )
            result = await workflow.execute(initial_input)

            a_int = next((x for x in result.agent_interactions if x.get("agent") == "PlayerA"), None)
            b_int = next((x for x in result.agent_interactions if x.get("agent") == "PlayerB"), None)
            if not a_int or not b_int:
                continue

            a_prompt = str(a_int.get("input") or "")
            b_prompt = str(b_int.get("input") or "")
            a_raw = str(a_int.get("output") or "")
            b_raw = str(b_int.get("output") or "")

            try:
                move_a = normalize_rps_move(a_raw)
                move_b = normalize_rps_move(b_raw)
            except Exception:
                if attempt >= 3:
                    raise
                continue

            winner = rps_winner(move_a, move_b)
            if winner == "a":
                score_a += 1
            elif winner == "b":
                score_b += 1
            else:
                draws += 1

            history.append(RPSRoundHistory(round_index=round_idx, move_a=move_a, move_b=move_b, winner=winner))

            if winner == "a":
                outcome = "Player A wins"
            elif winner == "b":
                outcome = "Player B wins"
            else:
                outcome = "Draw"

            print(f"Round {round_idx}: A={move_a} | B={move_b} -> {outcome}")
            break

        # If we failed to parse after retries, `winner` computation won't have happened.
        if move_a is None or move_b is None:  # pragma: no cover
            raise RuntimeError("Failed to obtain valid moves from both players.")

    final = RPSGameResult(rounds=rounds, score_a=score_a, score_b=score_b, draws=draws, history=history)
    _print_final(final)
    return final


async def main() -> None:
    """CLI entrypoint for running the RPS loop."""
    parser = argparse.ArgumentParser(description="Rock Paper Scissors multi-agent loop.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to play.")
    args = parser.parse_args()
    await run(args.rounds)


if __name__ == "__main__":
    asyncio.run(main())

