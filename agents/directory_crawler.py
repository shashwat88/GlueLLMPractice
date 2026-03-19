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
import uuid
from typing import Any

from gluellm.models.agent import Agent
from gluellm.models.prompt import SystemPrompt
from pydantic import BaseModel, Field

from core.logging_config import get_logger, setup_logging
from core.workflow_wrappers import run_reflection_workflow_parsed
from tools.filesystem_tools import (
    DirectoryReport,
    FileInfo,
    count_by_extension,
    largest_file,
    scan_directory,
)

logger = get_logger(__name__)


class DirectoryAnswer(BaseModel):
    """Structured directory answer returned to the user."""

    can_answer: bool = Field(
        description="Whether the scanned report contains enough evidence to answer."
    )
    answer: str = Field(description="Answer text (only if can_answer is true).")
    cannot_answer_reason: str | None = Field(
        description="Why the agent cannot answer from the report."
    )


def _is_quit(text: str) -> bool:
    """Return True if the user typed 'quit'."""
    return text.strip().lower() == "quit"


def _build_directory_tools(report: DirectoryReport) -> list[Any]:
    """Build tool functions backed by the cached directory scan."""

    def _normalize_ext(extension: str) -> str:
        """Normalize extension input (strip, lowercase, drop leading dot)."""
        ext = extension.strip().lower()
        if ext.startswith("."):
            ext = ext[1:]
        return ext

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
        return {
            "extension": extension.strip().lstrip(".").lower(),
            "exists": ext_count > 0,
            "count": ext_count,
        }

    def list_files_by_extension_tool(extension: str, limit: int = 20) -> list[dict[str, Any]]:
        """List (up to `limit`) file relative paths that match `extension`."""
        ext = _normalize_ext(extension)
        matches = [f for f in report.files if f.extension == ext]
        matches.sort(key=lambda x: x.relative_path)
        out: list[dict[str, Any]] = []
        for f in matches[:limit]:
            out.append(
                {"relative_path": f.relative_path, "size_bytes": f.size_bytes, "depth": f.depth}
            )
        return out

    def largest_file_tool() -> dict[str, Any]:
        """Return the largest file in the report as a JSON-serializable dict."""
        info = largest_file(report)
        if info is None:
            return {}
        return info.model_dump()

    def smallest_file_tool() -> dict[str, Any]:
        """Return the smallest file in the report as a JSON-serializable dict."""
        if not report.files:
            return {}
        info = min(report.files, key=lambda f: f.size_bytes)
        return info.model_dump()

    def total_size_tool() -> dict[str, Any]:
        """Return total size (bytes) across all scanned files."""
        total = sum(f.size_bytes for f in report.files)
        return {"total_size_bytes": total, "total_files": len(report.files)}

    def extension_size_stats_tool(extension: str) -> dict[str, Any]:
        """Return aggregate size stats for one file extension."""
        ext = _normalize_ext(extension)
        files = [f for f in report.files if f.extension == ext]
        if not files:
            return {"extension": ext, "count": 0}
        sizes = [f.size_bytes for f in files]
        total = sum(sizes)
        return {
            "extension": ext,
            "count": len(files),
            "total_size_bytes": total,
            "min_size_bytes": min(sizes),
            "max_size_bytes": max(sizes),
            "avg_size_bytes": total / len(files),
        }

    def extension_with_smallest_avg_size_tool(min_count: int = 1) -> dict[str, Any]:
        """Return the extension with the smallest average file size.

        Args:
            min_count: Only consider extensions with at least this many files.
        """
        buckets: dict[str, list[int]] = {}
        for f in report.files:
            buckets.setdefault(f.extension, []).append(f.size_bytes)
        candidates: list[tuple[str, float, int]] = []
        for ext, sizes in buckets.items():
            if len(sizes) < max(1, int(min_count)):
                continue
            candidates.append((ext, sum(sizes) / len(sizes), len(sizes)))
        if not candidates:
            return {}
        ext, avg, count = sorted(candidates, key=lambda t: (t[1], t[0]))[0]
        return {"extension": ext, "avg_size_bytes": avg, "count": count}

    def extension_with_largest_avg_size_tool(min_count: int = 1) -> dict[str, Any]:
        """Return the extension with the largest average file size.

        Args:
            min_count: Only consider extensions with at least this many files.
        """
        buckets: dict[str, list[int]] = {}
        for f in report.files:
            buckets.setdefault(f.extension, []).append(f.size_bytes)
        candidates: list[tuple[str, float, int]] = []
        for ext, sizes in buckets.items():
            if len(sizes) < max(1, int(min_count)):
                continue
            candidates.append((ext, sum(sizes) / len(sizes), len(sizes)))
        if not candidates:
            return {}
        ext, avg, count = sorted(candidates, key=lambda t: (-t[1], t[0]))[0]
        return {"extension": ext, "avg_size_bytes": avg, "count": count}

    def files_sorted_by_size_tool(order: str = "desc", limit: int = 20) -> list[dict[str, Any]]:
        """Return files sorted by size (desc/asc) with basic metadata."""
        direction = order.strip().lower()
        reverse = direction != "asc"
        files = sorted(report.files, key=lambda f: (f.size_bytes, f.relative_path), reverse=reverse)
        out: list[dict[str, Any]] = []
        for f in files[: max(1, int(limit))]:
            out.append(
                {
                    "relative_path": f.relative_path,
                    "extension": f.extension,
                    "size_bytes": f.size_bytes,
                    "depth": f.depth,
                }
            )
        return out

    def find_files_tool(
        name_contains: str,
        extension: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find files by substring match in relative path (case-insensitive)."""
        needle = name_contains.strip().lower()
        if not needle:
            return []
        ext = _normalize_ext(extension) if extension is not None else None
        matches: list[FileInfo] = []
        for f in report.files:
            if needle in f.relative_path.lower() and (ext is None or f.extension == ext):
                matches.append(f)
        matches.sort(key=lambda x: x.relative_path)
        out: list[dict[str, Any]] = []
        for f in matches[: max(1, int(limit))]:
            out.append(
                {
                    "relative_path": f.relative_path,
                    "extension": f.extension,
                    "size_bytes": f.size_bytes,
                    "depth": f.depth,
                }
            )
        return out

    def get_file_info_tool(relative_path: str) -> dict[str, Any]:
        """Return metadata for an exact relative path match."""
        p = relative_path.strip()
        if not p:
            return {}
        for f in report.files:
            if f.relative_path == p:
                return f.model_dump()
        return {}

    def list_directories_tool(
        limit: int = 50, name_contains: str | None = None
    ) -> list[dict[str, Any]]:
        """List subdirectories discovered during scan."""
        needle = (name_contains or "").strip().lower()
        dirs = report.dirs
        if needle:
            dirs = [d for d in dirs if needle in d.relative_path.lower()]
        dirs = sorted(dirs, key=lambda d: (d.depth, d.relative_path))
        out: list[dict[str, Any]] = []
        for d in dirs[: max(1, int(limit))]:
            out.append({"relative_path": d.relative_path, "depth": d.depth})
        return out

    def max_depth_tool() -> int:
        """Return the maximum nesting depth encountered during the scan."""
        return report.max_depth

    return [
        count_extension,
        most_common_extension_tool,
        has_extension_tool,
        list_files_by_extension_tool,
        largest_file_tool,
        smallest_file_tool,
        total_size_tool,
        extension_size_stats_tool,
        extension_with_smallest_avg_size_tool,
        extension_with_largest_avg_size_tool,
        files_sorted_by_size_tool,
        find_files_tool,
        get_file_info_tool,
        list_directories_tool,
        max_depth_tool,
    ]


def _build_conversation_history(history: list[tuple[str, str]]) -> str:
    """Render prior Q/A into a compact text block for the workflow prompt."""
    if not history:
        return "No prior conversation."
    parts: list[str] = []
    for i, (q, a) in enumerate(history, 1):
        parts.append(f"[Turn {i}] User: {q}\nAssistant: {a}")
    return "\n\n".join(parts)


def _build_generator_agent(*, model: str, max_tool_iterations: int, tools: list[Any]) -> Agent:
    """Create the generator agent that outputs strict JSON answers."""
    system_prompt = (
        "You are a filesystem exploration assistant. "
        "You must answer ONLY using the scanned directory report information exposed via tools. "
        "Never read or infer file contents. "
        "If the user asks for file contents (or anything that would require reading file text), "
        "set can_answer=false and explain why. "
        "You can answer questions about file types, sizes, paths/locations, names, depth, and directory listings. "
        "Prefer calling tools rather than guessing. "
        "Output ONLY valid JSON matching the schema:\n"
        "{\n"
        '  "can_answer": boolean,\n'
        '  "answer": string,\n'
        '  "cannot_answer_reason": string|null\n'
        "}\n"
        "Rules:\n"
        '- If can_answer=false: answer must be "".\n'
        "- If can_answer=true: answer must be derived from tools and be non-empty.\n"
    )
    return Agent(
        name="DirectoryGenerator",
        description="Generates strict JSON answers using filesystem metadata tools.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=tools,
        max_tool_iterations=max_tool_iterations,
        model=model,
    )


def _build_reflector_agent(*, model: str) -> Agent:
    """Create the reflector agent that validates tool-derived constraints."""
    system_prompt = (
        "You are a compliance reflector for directory crawling. "
        "Validate the generator's JSON so it does not rely on reading file contents. "
        "If the generator attempted to infer file text, set can_answer=false and provide cannot_answer_reason. "
        "Return ONLY valid JSON matching the same schema."
    )
    return Agent(
        name="DirectoryReflector",
        description="Reflector that validates JSON evidence constraints.",
        system_prompt=SystemPrompt(content=system_prompt),
        tools=[],
        max_tool_iterations=1,
        model=model,
    )


def _prompt_root_dir() -> str:
    """Prompt the user for a root directory and return the validated path."""
    return input("Enter root directory path: ").strip()


async def _answer_with_workflow(
    *,
    model: str,
    max_tool_iterations: int,
    report: DirectoryReport,
    question: str,
    history: list[tuple[str, str]],
) -> DirectoryAnswer:
    """Answer one directory question using ReflectionWorkflow and parse strict JSON."""
    turn_id = uuid.uuid4().hex[:8]
    logger.info(
        "dir.turn_start turn_id=%s question_len=%s history_turns=%s",
        turn_id,
        len(question),
        len(history),
    )
    tools = _build_directory_tools(report)
    generator = _build_generator_agent(
        model=model, max_tool_iterations=max_tool_iterations, tools=tools
    )
    reflector = _build_reflector_agent(model=model)

    history_text = _build_conversation_history(history)
    initial_input = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Current user question:\n{question}\n\n"
        "Return the JSON object only."
    )

    def _on_parse_error(raw: str, e: Exception) -> DirectoryAnswer:
        _ = raw
        return DirectoryAnswer(
            can_answer=False,
            answer="",
            cannot_answer_reason=f"Workflow output could not be parsed as valid directory JSON: {type(e).__name__}",
        )

    parsed = await run_reflection_workflow_parsed(
        initial_input=initial_input,
        response_model=DirectoryAnswer,
        generator_agent=generator,
        reflector_agent=reflector,
        max_reflections=2,
        on_parse_error=_on_parse_error,
    )
    logger.info(
        "dir.turn_parsed turn_id=%s can_answer=%s answer_len=%s",
        turn_id,
        parsed.can_answer,
        len(parsed.answer),
    )

    if parsed.can_answer and not parsed.answer:
        logger.info(
            "dir.evidence_guard_flip turn_id=%s reason=%s",
            turn_id,
            "can_answer_true_but_answer_empty",
        )
        return DirectoryAnswer(
            can_answer=False,
            answer="",
            cannot_answer_reason="Answer claimed can_answer=true but no evidence-derived answer was provided.",
        )
    return parsed


async def directory_crawler_interactive(*, model: str, max_tool_iterations: int) -> None:
    """Run the directory crawler agent in interactive mode."""
    root = _prompt_root_dir()
    try:
        report = scan_directory(root)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Cannot scan directory: {e}")
        logger.warning("Directory scan failed: root=%s error=%s", root, str(e))
        return

    logger.info(
        "Directory scan complete: root=%s total_files=%s total_dirs=%s max_depth=%s",
        report.root,
        report.total_files,
        report.total_dirs,
        report.max_depth,
    )
    history: list[tuple[str, str]] = []

    question = input("Question (or type 'quit' to exit): ").strip()
    while question and _is_quit(question):
        logger.info("cli.quit agent=directory_crawler stage=initial_question")
        return
    while True:
        if not question:
            question = input("Question cannot be empty. Enter question (or 'quit'): ").strip()
            if _is_quit(question):
                return
            continue

        answer = await _answer_with_workflow(
            model=model,
            max_tool_iterations=max_tool_iterations,
            report=report,
            question=question,
            history=history,
        )
        logger.info(
            "Directory Q&A: can_answer=%s question=%r",
            answer.can_answer,
            question,
        )
        if answer.can_answer:
            print("\nAnswer\n" + "-" * 40)
            print(answer.answer)
        else:
            reason = answer.cannot_answer_reason or "Insufficient evidence in the scanned report."
            print("\nI cannot answer that question from the scanned directory evidence.")
            print("Reason: " + reason)
            logger.info("Directory cannot_answer_reason=%r", reason)

        assistant_text = answer.answer if answer.can_answer else f"Cannot answer: {reason}"
        history.append((question, assistant_text))

        question = input("\nFollow-up (or type 'quit' to exit): ").strip()
        if _is_quit(question):
            logger.info("cli.quit agent=directory_crawler stage=follow_up")
            return


async def main() -> None:
    """CLI entrypoint for the directory crawler agent."""
    setup_logging()
    logger.info("CLI start: directory_crawler")
    parser = argparse.ArgumentParser(description="Directory crawler agent with cached scans.")
    parser.add_argument(
        "--model", type=str, default="openai:gpt-4o-mini", help="GlueLLM model string."
    )
    parser.add_argument("--max-iters", type=int, default=6, help="Max tool execution iterations.")
    args = parser.parse_args()
    await directory_crawler_interactive(model=args.model, max_tool_iterations=args.max_iters)


if __name__ == "__main__":
    asyncio.run(main())
