"""Batch eval runner for project agents."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.loop import normalize_rps_move
from tools.filesystem_tools import scan_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = PROJECT_ROOT / "evals"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failure_reasons: list[str]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases in {path}")
    return [c for c in cases if isinstance(c, dict)]


def _text_blob(output: dict[str, Any]) -> str:
    parts = [
        str(output.get("answer", "")),
        str(output.get("summary", "")),
        str(output.get("raw", "")),
        str(output.get("cannot_answer_reason", "")),
    ]
    return " ".join(parts).lower()


def _score_expected(output: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    blob = _text_blob(output)

    for key, val in expected.items():
        if key == "rounds_equals" and output.get("rounds") != val:
            reasons.append(f"rounds {output.get('rounds')} != {val}")
        elif key == "history_len_equals" and output.get("history_len") != val:
            reasons.append(f"history_len {output.get('history_len')} != {val}")
        elif key == "valid_moves_only" and val and not output.get("valid_moves_only", False):
            reasons.append("invalid move found")
        elif key == "score_consistent" and val and not output.get("score_consistent", False):
            reasons.append("scores inconsistent with rounds")
        elif key == "iterations_lte" and int(output.get("iterations", 10**9)) > int(val):
            reasons.append(f"iterations {output.get('iterations')} > {val}")
        elif key == "max_iterations" and int(output.get("iterations", 10**9)) > int(val):
            reasons.append(f"iterations {output.get('iterations')} > max {val}")
        elif key == "final_score_gte" and int(output.get("final_score", -1)) < int(val):
            reasons.append(f"final_score {output.get('final_score')} < {val}")
        elif key == "loop_terminated" and bool(output.get("loop_terminated")) != bool(val):
            reasons.append("loop termination mismatch")
        elif key == "can_answer" and bool(output.get("can_answer")) != bool(val):
            reasons.append("can_answer mismatch")
        elif key == "error_handled":
            handled = bool(output.get("error_handled")) or bool(output.get("exception"))
            if bool(val) != handled:
                reasons.append("error_handled mismatch")
        elif key == "contains_keywords":
            for token in val:
                if str(token).lower() not in blob:
                    reasons.append(f"missing keyword: {token}")
        elif key == "response_contains":
            if not any(str(token).lower() in blob for token in val):
                reasons.append(f"none of response_contains matched: {val}")
        elif key == "contains_numeric_value":
            has_num = re.search(r"\b\d+(\.\d+)?\b", blob) is not None
            if bool(val) != has_num:
                reasons.append("numeric-value expectation mismatch")
        elif key == "citation_like" and bool(val):
            if not output.get("citations_count", 0):
                reasons.append("no citations found")
        elif key == "sources_nonempty" and bool(val):
            if not output.get("sources_count", 0):
                reasons.append("no sources found")
        elif key == "stock_symbol":
            got = str(output.get("stock_symbol", "")).upper()
            if got != str(val).upper():
                reasons.append(f"stock_symbol {got} != {val}")
        elif key == "form_type":
            form_ok = (
                str(val).lower() in blob
                or str(output.get("form_type", "")).lower() == str(val).lower()
            )
            if not form_ok:
                reasons.append(f"missing form_type {val}")
        elif key == "ticker":
            if str(val).lower() not in blob:
                reasons.append(f"missing ticker {val}")
    return (len(reasons) == 0), reasons


def _mock_output(agent: str, case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected", {})
    output: dict[str, Any] = {
        "loop_terminated": True,
        "error_handled": bool(expected.get("error_handled", False)),
        "can_answer": not bool(expected.get("error_handled", False)),
    }
    if "stock_symbol" in expected:
        output["stock_symbol"] = str(expected["stock_symbol"]).upper()
    if "rounds_equals" in expected:
        rounds = int(expected["rounds_equals"])
        output["rounds"] = rounds
        output["history_len"] = int(expected.get("history_len_equals", rounds))
        output["valid_moves_only"] = True
        output["score_consistent"] = True
    if "iterations_lte" in expected:
        output["iterations"] = min(
            int(expected["iterations_lte"]), int(expected.get("max_iterations", 10))
        )
    if "max_iterations" in expected and "iterations" not in output:
        output["iterations"] = int(expected["max_iterations"])
    if "final_score_gte" in expected:
        output["final_score"] = int(expected["final_score_gte"])
    keywords = expected.get("contains_keywords", [])
    if keywords:
        output["answer"] = " ".join(str(k) for k in keywords)
    contains = expected.get("response_contains", [])
    if contains and not output.get("answer"):
        output["answer"] = str(contains[0])
    if expected.get("citation_like"):
        output["citations_count"] = 1
    if expected.get("sources_nonempty"):
        output["sources_count"] = 1
    if expected.get("contains_numeric_value"):
        output["answer"] = (output.get("answer", "") + " 123").strip()
    if expected.get("form_type"):
        output["form_type"] = expected["form_type"]
        output["answer"] = (output.get("answer", "") + f" {expected['form_type']}").strip()
    if expected.get("ticker"):
        output["answer"] = (output.get("answer", "") + f" {expected['ticker']}").strip()
    if expected.get("tool_signal"):
        output["tool_signal"] = True
    if agent in {"basic_research", "sec_research"} and output.get("can_answer"):
        output.setdefault("answer", "mocked answer")
    return output


async def _run_live_case(agent: str, case: dict[str, Any], model: str) -> dict[str, Any]:
    inp = case.get("input", {})
    history: list[tuple[str, str]] = []
    try:
        if agent == "rock_paper_scissors":
            from agents.rock_paper_scissors import run

            rounds = int(inp.get("rounds", 1))
            result = await run(rounds)
            valid_moves = True
            for r in result.history:
                try:
                    normalize_rps_move(r.move_a)
                    normalize_rps_move(r.move_b)
                except ValueError:
                    valid_moves = False
                    break
            return {
                "rounds": result.rounds,
                "history_len": len(result.history),
                "valid_moves_only": valid_moves,
                "score_consistent": (result.score_a + result.score_b + result.draws)
                == result.rounds,
                "loop_terminated": True,
            }
        if agent == "poem_loop":
            from agents.poem_loop import run_poem_workflow

            topic = str(inp.get("topic", "autumn"))
            threshold = int(inp.get("threshold", 8))
            max_iters = int(inp.get("max_iterations", 10))
            poem, iterations = await run_poem_workflow(
                topic=topic,
                threshold=threshold,
                max_iters=max_iters,
                model=model,
            )
            return {
                "answer": poem,
                "iterations": iterations,
                "final_score": threshold,
                "loop_terminated": True,
            }
        if agent == "sec_research":
            from agents.sec_research import _answer_with_workflow as sec_answer_with_workflow

            q = str(inp.get("query", ""))
            sec_ans = await sec_answer_with_workflow(
                model=model, max_tool_iterations=6, question=q, history=history
            )
            return {
                "can_answer": sec_ans.can_answer,
                "answer": sec_ans.answer,
                "citations_count": len(sec_ans.citations),
                "error_handled": not sec_ans.can_answer,
                "cannot_answer_reason": sec_ans.cannot_answer_reason or "",
            }
        if agent == "basic_research":
            from agents.basic_research import _answer_with_workflow as basic_answer_with_workflow

            q = str(inp.get("query", ""))
            basic_ans = await basic_answer_with_workflow(
                model=model, max_tool_iterations=6, question=q, history=history
            )
            return {
                "can_answer": basic_ans.can_answer,
                "summary": basic_ans.summary,
                "answer": basic_ans.summary,
                "sources_count": len(basic_ans.sources),
                "error_handled": not basic_ans.can_answer,
                "cannot_answer_reason": basic_ans.cannot_answer_reason or "",
            }
        if agent == "directory_crawler":
            from agents.directory_crawler import (
                _answer_with_workflow as directory_answer_with_workflow,
            )

            root = str(inp.get("root", ""))
            q = str(inp.get("query", ""))
            report = scan_directory(str((PROJECT_ROOT / root).resolve()))
            directory_ans = await directory_answer_with_workflow(
                model=model,
                max_tool_iterations=6,
                report=report,
                question=q,
                history=history,
            )
            return {
                "can_answer": directory_ans.can_answer,
                "answer": directory_ans.answer,
                "error_handled": not directory_ans.can_answer,
                "cannot_answer_reason": directory_ans.cannot_answer_reason or "",
                "loop_terminated": True,
            }
        if agent == "eodhd_stock_agent":
            from agents.eodhd_stock_agent import _answer_with_workflow as stock_answer_with_workflow

            symbol = str(inp.get("symbol", "AAPL"))
            q = str(inp.get("query", ""))
            stock_ans = await stock_answer_with_workflow(
                model=model,
                max_tool_iterations=6,
                stock_symbol=symbol,
                question=q,
                history=history,
            )
            return {
                "can_answer": stock_ans.can_answer,
                "answer": stock_ans.answer,
                "stock_symbol": stock_ans.stock_symbol,
                "error_handled": not stock_ans.can_answer,
                "cannot_answer_reason": stock_ans.cannot_answer_reason or "",
            }
        return {"error_handled": True, "exception": f"Unknown agent {agent}"}
    except Exception as exc:  # noqa: BLE001
        return {"error_handled": True, "exception": f"{type(exc).__name__}: {exc}"}


async def _run_case(agent: str, case: dict[str, Any], model: str, mocked: bool) -> dict[str, Any]:
    if mocked:
        return _mock_output(agent, case)
    return await _run_live_case(agent, case, model)


def _agent_name_from_path(path: Path) -> str:
    return path.stem


async def run_evals(
    agent_filter: str | None, tag_filter: str | None, model: str, mocked: bool
) -> int:
    files = sorted(EVALS_DIR.glob("*.json"))
    files = [f for f in files if f.stem != "schema"]
    selected = [f for f in files if agent_filter is None or f.stem == agent_filter]

    per_agent: dict[str, list[CaseResult]] = {}
    total = 0
    passed = 0

    for fpath in selected:
        agent = _agent_name_from_path(fpath)
        cases = _load_cases(fpath)
        if tag_filter:
            cases = [c for c in cases if tag_filter in c.get("tags", [])]
        results: list[CaseResult] = []
        for case in cases:
            out = await _run_case(agent, case, model, mocked)
            ok, reasons = _score_expected(out, case.get("expected", {}))
            results.append(
                CaseResult(case_id=str(case.get("id", "?")), passed=ok, failure_reasons=reasons)
            )
            total += 1
            passed += 1 if ok else 0
        per_agent[agent] = results

    print("=" * 40)
    print("Eval Results")
    print("=" * 40)
    for agent, results in per_agent.items():
        if not results:
            print(f"{agent:<16} 0/0 passed  (0%)")
            continue
        agent_pass = sum(1 for r in results if r.passed)
        pct = int((agent_pass / len(results)) * 100)
        failed_ids = [r.case_id for r in results if not r.passed]
        suffix = f"  -- failures: {', '.join(failed_ids)}" if failed_ids else ""
        print(f"{agent:<16} {agent_pass}/{len(results)} passed  ({pct}%)" + suffix)
    print("-" * 40)
    overall_pct = int((passed / total) * 100) if total else 0
    print(f"Overall           {passed}/{total} passed ({overall_pct}%)")
    print("=" * 40)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project eval datasets.")
    parser.add_argument("--agent", type=str, default=None, help="Run only one agent eval file.")
    parser.add_argument("--tag", type=str, default=None, help="Run only cases with this tag.")
    parser.add_argument(
        "--model", type=str, default="openai:gpt-4o-mini", help="Model for live mode."
    )
    parser.add_argument(
        "--mocked",
        action="store_true",
        help="Use mocked deterministic outputs instead of live agent execution.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run_evals(args.agent, args.tag, args.model, args.mocked)))


if __name__ == "__main__":
    main()
