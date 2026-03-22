"""Offline eval: verify sample answers mention the ticker symbol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.logging_config import setup_logging
from eval.scoring import eodhd_answer_mentions_symbol


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
    parser = argparse.ArgumentParser(description="EODHD-style ticker mention eval (offline).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root() / "eval/data/eodhd_stock_agent/cases.jsonl",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.dataset)
    passed = 0
    for row in rows:
        rid = row.get("id", "?")
        symbol = str(row["symbol"])
        answer = str(row["sample_answer"])
        ok = eodhd_answer_mentions_symbol(answer, symbol)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {rid} symbol={symbol}")
    print(f"\nResults: {passed}/{len(rows)} rows passed")


if __name__ == "__main__":
    main()
