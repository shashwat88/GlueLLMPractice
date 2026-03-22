"""Offline eval: scan fixture trees and verify counts (uses tools.filesystem_tools)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from core.logging_config import setup_logging
from tools.filesystem_tools import scan_directory


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Directory scan fixture eval (offline).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root() / "eval/data/directory_crawler/cases.jsonl",
    )
    args = parser.parse_args()

    root = _project_root()
    rows = _load_jsonl(args.dataset)
    passed = 0
    for row in rows:
        rid = row.get("id", "?")
        rel = str(row["root"])
        scan_root = (root / rel).resolve()
        report = scan_directory(str(scan_root))
        min_files = int(cast(int | float | str, row["expect_min_files"]))
        min_depth = int(cast(int | float | str, row["expect_min_depth"]))
        ok = report.total_files >= min_files and report.max_depth >= min_depth
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {rid} files={report.total_files} max_depth={report.max_depth} "
            f"(need files>={min_files}, depth>={min_depth})"
        )
    print(f"\nResults: {passed}/{len(rows)} rows passed")


if __name__ == "__main__":
    main()
