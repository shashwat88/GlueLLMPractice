"""SDK-native GlueLLM eval recording demo runner.

This follows GlueLLM's official eval recording example pattern:
- use `gluellm.eval.JSONLFileStore` directly
- pass store via `GlueLLM(eval_store=...)`
- run normal `complete()` calls
- close store and inspect recorded JSONL rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from gluellm import GlueLLM
from gluellm.eval import JSONLFileStore


def _default_output_path() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "eval_records.jsonl"


async def run_eval_recording(*, output_path: Path, model: str, prompts: list[str]) -> None:
    """Run prompts through GlueLLM and record eval traces to JSONL."""
    store = JSONLFileStore(str(output_path))
    client = GlueLLM(model=model, eval_store=store)

    try:
        for prompt in prompts:
            print(f"Prompt: {prompt}")
            try:
                result = await client.complete(prompt)
                print(f"Response: {result.final_response}\n")
            except Exception as exc:
                # SDK may still emit EvalRecord rows for failures.
                print(f"Request failed: {type(exc).__name__}: {exc}\n")
    finally:
        await store.close()


def _print_record_preview(path: Path, limit: int = 3) -> None:
    """Print a compact preview of recorded EvalRecord JSON rows."""
    if not path.exists():
        print(f"No eval records found at: {path}")
        return

    print(f"Recorded file: {path}")
    printed = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            print(
                f"- id={rec.get('id')} latency_ms={rec.get('latency_ms')} "
                f"cost={rec.get('estimated_cost_usd')} success={rec.get('success')}"
            )
            printed += 1
            if printed >= limit:
                break
    if printed == 0:
        print("Eval file exists but has no records.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GlueLLM eval recording demo.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="JSONL output path for recorded EvalRecord rows.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai:gpt-4o-mini",
        help="GlueLLM model string.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to evaluate (repeatable). Defaults to two sample prompts.",
    )
    args = parser.parse_args()

    prompts = args.prompts or [
        "What is 2 + 2? Answer briefly.",
        "In one sentence, define a deterministic algorithm.",
    ]

    asyncio.run(run_eval_recording(output_path=args.output, model=args.model, prompts=prompts))
    _print_record_preview(args.output)


if __name__ == "__main__":
    main()
