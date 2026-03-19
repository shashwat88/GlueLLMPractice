"""Filesystem scanning utilities for directory crawling.

These functions are deterministic and intended for unit testing. They scan a
local directory tree and return a structured report with:
- file and directory counts
- file extensions/types
- file sizes
- nesting depth information
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class FileInfo(BaseModel):
    """Information about a file found during a directory scan."""

    relative_path: str = Field(description="Path relative to scan root")
    extension: str = Field(
        description="Lowercased file extension (without leading dot); empty for no extension"
    )
    size_bytes: int = Field(description="File size in bytes")
    depth: int = Field(description="Nesting depth of the file relative to root")


class DirInfo(BaseModel):
    """Information about a subdirectory found during a directory scan."""

    relative_path: str = Field(description="Path relative to scan root")
    depth: int = Field(description="Nesting depth of the directory relative to root")


class DirectoryReport(BaseModel):
    """Aggregated results from a directory scan."""

    root: str = Field(description="Root directory path provided by the user")
    max_depth: int = Field(description="Maximum nesting depth encountered")
    total_files: int = Field(description="Total number of files scanned")
    total_dirs: int = Field(
        description="Total number of directories scanned (excluding root itself)"
    )
    files: list[FileInfo] = Field(description="Flat list of files with extension and size metadata")
    dirs: list[DirInfo] = Field(
        default_factory=list, description="Flat list of directories with depth metadata"
    )


def _extension_for_path(p: Path) -> str:
    """Return the lowercased file extension without the leading dot."""
    suffix = p.suffix
    if not suffix:
        return ""
    return suffix[1:].lower()


def scan_directory(root: str, *, max_files: int = 100000) -> DirectoryReport:
    """Recursively scan a directory and build a structured report.

    Args:
        root: Root directory to scan.
        max_files: Hard cap on the number of files collected (to avoid runaway scans).

    Returns:
        A `DirectoryReport` with counts, max depth, and per-file metadata.

    Raises:
        FileNotFoundError: If `root` does not exist.
        NotADirectoryError: If `root` is not a directory.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(str(root_path))
    if not root_path.is_dir():
        raise NotADirectoryError(str(root_path))

    files: list[FileInfo] = []
    dirs: list[DirInfo] = []
    max_depth = 0
    total_dirs = 0

    for dirpath, _dirnames, filenames in os.walk(root_path, topdown=True, followlinks=False):
        # Count directories (excluding root itself).
        rel_dir = Path(dirpath).relative_to(root_path)
        if rel_dir.parts:
            total_dirs += 1
            dirs.append(DirInfo(relative_path=str(rel_dir), depth=len(rel_dir.parts)))

        depth = len(rel_dir.parts)
        max_depth = max(max_depth, depth)

        for fname in filenames:
            if len(files) >= max_files:
                break
            file_path = Path(dirpath) / fname
            stat = file_path.stat()
            rel_path = str(file_path.relative_to(root_path))
            files.append(
                FileInfo(
                    relative_path=rel_path,
                    extension=_extension_for_path(file_path),
                    size_bytes=int(stat.st_size),
                    depth=depth + 1,
                )
            )

        if len(files) >= max_files:
            break

    return DirectoryReport(
        root=str(root_path),
        max_depth=max_depth,
        total_files=len(files),
        total_dirs=total_dirs,
        files=files,
        dirs=dirs,
    )


def count_by_extension(report: DirectoryReport, extension: str) -> int:
    """Count files matching a specific extension.

    Args:
        report: Directory report produced by `scan_directory`.
        extension: Extension to count (with or without leading dot).

    Returns:
        Number of matching files.
    """
    ext = extension.strip().lower()
    if ext.startswith("."):
        ext = ext[1:]
    return sum(1 for f in report.files if f.extension == ext)


def largest_file(report: DirectoryReport) -> FileInfo | None:
    """Return the largest file (by size) in the report."""
    if not report.files:
        return None
    return max(report.files, key=lambda f: f.size_bytes)
