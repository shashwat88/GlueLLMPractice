"""Offline eval: parse critic feedback scores (same parser as agents.poem_loop)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from core.logging_config import setup_logging
from eval.scoring import score_poem_feedback


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
    parser = argparse.ArgumentParser(description="Poem critic feedback parsing eval (offline).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root() / "eval/data/poem_loop/topics.jsonl",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    passed = 0
    for row in rows:
        rid = row.get("id", "?")
        feedback = str(row["sample_critic_feedback"])
        min_score = int(cast(int | float | str, row["min_score"]))
        ok = score_poem_feedback(feedback, min_score=min_score)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {rid} (min_score={min_score})")
    print(f"\nResults: {passed}/{len(rows)} rows passed")


if __name__ == "__main__":
    main()
