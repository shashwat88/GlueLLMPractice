"""Offline eval: heuristic checks on sample SEC-style answers (no LLM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.logging_config import setup_logging
from eval.scoring import sec_answer_has_citation_markers


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
    parser = argparse.ArgumentParser(description="SEC research answer heuristic eval (offline).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root() / "eval/data/sec_research/queries.jsonl",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    passed = 0
    for row in rows:
        rid = row.get("id", "?")
        answer = str(row["sample_answer"])
        ok = sec_answer_has_citation_markers(answer)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {rid}")
    print(f"\nResults: {passed}/{len(rows)} rows passed (heuristic: filing markers or year)")


if __name__ == "__main__":
    main()
