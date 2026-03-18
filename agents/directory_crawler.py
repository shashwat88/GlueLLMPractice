"""Directory crawler agent with cached filesystem report.

This agent:
1) prompts the user for a root directory path at runtime
2) scans the directory exactly once into an in-memory report
3) prompts the user for a natural language question
4) answers follow-up questions until the user types `quit`

If the question cannot be answered from the scanned evidence, the agent
responds gracefully without hallucinating.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from pydantic import BaseModel, Field

from gluellm.api import GlueLLM
from tools.filesystem_tools import DirectoryReport, FileInfo, count_by_extension, largest_file, scan_directory


class DirectoryAnswer(BaseModel):
    """Structured directory answer returned to the user."""

    can_answer: bool = Field(description="Whether the scanned report contains enough evidence to answer.")
    answer: str = Field(description="Answer text (only if can_answer is true).")
    cannot_answer_reason: str | None = Field(description="Why the agent cannot answer from the report.")


def _is_quit(text: str) -> bool:
    """Return True if the user typed 'quit'."""
    return text.strip().lower() == "quit"


def _make_client(*, model: str, report: DirectoryReport, max_tool_iterations: int) -> GlueLLM:
    """Create a GlueLLM client with tools backed by the provided report."""

    def count_extension(extension: str) -> int:
        """Count files in the report matching an extension."""
        return count_by_extension(report, extension)

    def most_common_extension_tool() -> dict[str, Any]:
        """Return the most frequent extension and its count."""
        if not report.files:
            return {"extension": "", "count": 0}
        counts: dict[str, int] = {}
        for f in report.files:
            counts[f.extension] = counts.get(f.extension, 0) + 1
        # Sort by count desc, then extension for stable output.
        best_ext = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return {"extension": best_ext[0], "count": best_ext[1]}

    def has_extension_tool(extension: str) -> dict[str, Any]:
        """Return whether the report contains files with the given extension."""
        ext_count = count_by_extension(report, extension)
        return {"extension": extension.strip().lstrip(".").lower(), "exists": ext_count > 0, "count": ext_count}

    def list_files_by_extension_tool(extension: str, limit: int = 20) -> list[dict[str, Any]]:
        """List (up to `limit`) file relative paths that match `extension`."""
        ext = extension.strip().lstrip(".").lower()
        matches = [f for f in report.files if f.extension == ext]
        matches.sort(key=lambda x: x.relative_path)
        out: list[dict[str, Any]] = []
        for f in matches[:limit]:
            out.append({"relative_path": f.relative_path, "size_bytes": f.size_bytes, "depth": f.depth})
        return out

    def largest_file_tool() -> dict[str, Any]:
        """Return the largest file in the report as a JSON-serializable dict."""
        info = largest_file(report)
        if info is None:
            return {}
        return info.model_dump()

    def max_depth_tool() -> int:
        """Return the maximum nesting depth encountered during the scan."""
        return report.max_depth

    system_prompt = (
        "You are a filesystem exploration assistant. "
        "You must answer ONLY using the scanned directory report and the provided tools. "
        "Never read or infer file contents. "
        "If the user asks for file contents (or anything that would require reading file text), "
        "set can_answer=false and explain why. "
        "For file-count and file-type questions, prefer calling tools like "
        "`most_common_extension_tool`, `count_extension`, and `has_extension_tool` rather than guessing. "
        "If the scanned report does not include enough information, set can_answer=false and explain why. "
        "When asked for counts or sizes, prefer calling the tools instead of guessing."
    )

    return GlueLLM(
        model=model,
        system_prompt=system_prompt
        + f"\n\nScanned report summary: total_files={report.total_files}, total_dirs={report.total_dirs}, max_depth={report.max_depth}.",
        tools=[
            count_extension,
            most_common_extension_tool,
            has_extension_tool,
            list_files_by_extension_tool,
            largest_file_tool,
            max_depth_tool,
        ],
        max_tool_iterations=max_tool_iterations,
    )


def _prompt_root_dir() -> str:
    """Prompt the user for a root directory and return the validated path."""
    return input("Enter root directory path: ").strip()


async def _answer_turn(client: GlueLLM, question: str) -> DirectoryAnswer:
    """Run one question turn with structured output."""
    result = await client.structured_complete(question, response_format=DirectoryAnswer)
    return result.structured_output


async def directory_crawler_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the directory crawler agent in interactive mode."""
    root = _prompt_root_dir()
    try:
        report = scan_directory(root)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Cannot scan directory: {e}")
        return

    client = _make_client(model=model, report=report, max_tool_iterations=max_tool_iterations)

    question = input("Question (or type 'quit' to exit): ").strip()
    while question and _is_quit(question):
        return
    while True:
        if not question:
            question = input("Question cannot be empty. Enter question (or 'quit'): ").strip()
            if _is_quit(question):
                return
            continue

        answer = await _answer_turn(client, question)
        if answer.can_answer:
            print("\nAnswer\n" + "-" * 40)
            print(answer.answer)
        else:
            reason = answer.cannot_answer_reason or "Insufficient evidence in the scanned report."
            print("\nI cannot answer that question from the scanned directory evidence.")
            print("Reason: " + reason)

        question = input("\nFollow-up (or type 'quit' to exit): ").strip()
        if _is_quit(question):
            return


async def main() -> None:
    """CLI entrypoint for the directory crawler agent."""
    parser = argparse.ArgumentParser(description="Directory crawler agent with cached scans.")
    parser.add_argument("--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string.")
    parser.add_argument("--max-iters", type=int, default=6, help="Max tool execution iterations.")
    args = parser.parse_args()
    await directory_crawler_interactive(model=args.model, max_tool_iterations=args.max_iters)


if __name__ == "__main__":
    asyncio.run(main())

