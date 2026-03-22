"""Eval: single-turn rock/paper/scissors move validity (GlueLLM + EvalRecord JSONL)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from gluellm.api import GlueLLM

from core.logging_config import setup_logging
from eval.recording import enable_session_recording
from eval.scoring import score_rps_output

RPS_SYSTEM_PROMPT = (
    "You are Player A. Output exactly one word: rock, paper, or scissors. "
    "Output only the word and nothing else."
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


async def run_eval(*, dataset: Path, model: str) -> tuple[int, int, list[tuple[str, bool, str]]]:
    """Run dataset rows; return (passed, total, detail rows)."""
    rows = _load_jsonl(dataset)
    client = GlueLLM(
        model=model,
        system_prompt=RPS_SYSTEM_PROMPT,
        tools=[],
        max_tool_iterations=1,
    )
    passed = 0
    details: list[tuple[str, bool, str]] = []
    for row in rows:
        rid = row.get("id", "?")
        prompt = row["prompt"]
        result = await client.complete(prompt)
        try:
            text = str(result.final_response)
        except AttributeError:
            text = str(result.final_result)
        ok = score_rps_output(text)
        if ok:
            passed += 1
        details.append((str(rid), ok, text.strip()[:200]))
    return passed, len(rows), details


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="RPS move validity eval (GlueLLM eval recording).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root() / "eval/data/rock_paper_scissors/move_validity.jsonl",
        help="JSONL with id + prompt per line.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai:gpt-4o-mini",
        help="GlueLLM model string.",
    )
    args = parser.parse_args()

    out_path = enable_session_recording("rock_paper_scissors", project_root=_project_root())
    print(f"GlueLLM eval recording -> {out_path}")

    passed, total, details = asyncio.run(run_eval(dataset=args.dataset, model=args.model))
    pct = (passed / total) if total else 0.0
    print(f"\nResults: {passed}/{total} valid moves ({pct:.1%})")
    for rid, ok, snippet in details:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {rid}: {snippet!r}")


if __name__ == "__main__":
    main()
