"""Shared GlueLLM eval recording setup (JSONL `EvalRecord` traces)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gluellm.eval import enable_file_recording


def eval_logs_dir(project_root: Path | None = None) -> Path:
    """Return `logs/` under the project root, creating it if needed."""
    root = project_root if project_root is not None else Path.cwd()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def enable_session_recording(
    agent_slug: str,
    *,
    project_root: Path | None = None,
) -> Path:
    """Enable global JSONL recording for this process; return the output file path.

    Uses `gluellm.eval.enable_file_recording`, which writes one `EvalRecord` JSON object
    per line for each GlueLLM interaction.
    """
    logs = eval_logs_dir(project_root)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_slug = agent_slug.replace("/", "_").replace(" ", "_")
    path = logs / f"eval_{safe_slug}_{ts}.jsonl"
    enable_file_recording(str(path))
    return path
