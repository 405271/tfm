from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


DEFAULT_CASES_PATH = Path("app/data/eval_memory_cases.json")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", _normalize(text)) if token}


def _jaccard(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _contains_all(reply: str, expected_parts: list[str]) -> bool:
    normalized_reply = _normalize(reply)
    return all(_normalize(part) in normalized_reply for part in expected_parts)


@dataclass
class TurnResult:
    index: int
    message: str
    reply: str
    escalated: bool
    sources_count: int
    latency_ms: float
    passed: bool
    failure_reasons: list[str]


@dataclass
class CaseResult:
    case_id: str
    variant: str
    passed: bool
    turns: list[TurnResult]
    exact_repeat_count: int
    high_similarity_repeat_count: int
    avg_similarity: float
    turns_to_resolution: int | None


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("The cases file must contain a JSON array.")
    return payload


def _run_case(
    client: httpx.Client,
    api_url: str,
    variant: str,
    case: dict[str, Any],
) -> CaseResult:
    history: list[dict[str, str]] = []
    session_id = str(case.get("session_id") or uuid4())
    turns: list[TurnResult] = []
    replies: list[str] = []
    turns_to_resolution: int | None = None

    for index, raw_turn in enumerate(case["turns"], start=1):
        if isinstance(raw_turn, str):
            turn = {"message": raw_turn}
        else:
            turn = dict(raw_turn)

        message = str(turn["message"])
        payload = {
            "message": message,
            "session_id": session_id,
            "history": history,
        }

        response = client.post(f"{api_url.rstrip('/')}/chat", json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        reply = str(data["reply"])
        escalated = bool(data.get("escalated"))
        sources = data.get("sources") or []
        latency_ms = response.elapsed.total_seconds() * 1000
        failure_reasons: list[str] = []

        expected_parts = turn.get("expect_reply_contains") or []
        if expected_parts and not _contains_all(reply, list(expected_parts)):
            failure_reasons.append(
                f"reply missing expected parts: {', '.join(expected_parts)}"
            )

        forbidden_parts = turn.get("expect_reply_not_contains") or []
        normalized_reply = _normalize(reply)
        for part in forbidden_parts:
            if _normalize(str(part)) in normalized_reply:
                failure_reasons.append(f"reply contains forbidden part: {part}")

        if "expect_escalated" in turn and escalated != bool(turn["expect_escalated"]):
            failure_reasons.append(
                f"expected escalated={bool(turn['expect_escalated'])}, got {escalated}"
            )

        if "min_sources" in turn and len(sources) < int(turn["min_sources"]):
            failure_reasons.append(
                f"expected at least {int(turn['min_sources'])} sources, got {len(sources)}"
            )

        if "max_sources" in turn and len(sources) > int(turn["max_sources"]):
            failure_reasons.append(
                f"expected at most {int(turn['max_sources'])} sources, got {len(sources)}"
            )

        if turn.get("marks_resolution") and turns_to_resolution is None:
            turns_to_resolution = index

        turn_result = TurnResult(
            index=index,
            message=message,
            reply=reply,
            escalated=escalated,
            sources_count=len(sources),
            latency_ms=latency_ms,
            passed=not failure_reasons,
            failure_reasons=failure_reasons,
        )
        turns.append(turn_result)
        replies.append(reply)
        session_id = str(data["session_id"])
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})

    exact_repeat_count = 0
    high_similarity_repeat_count = 0
    similarities: list[float] = []
    for previous, current in zip(replies, replies[1:]):
        similarity = _jaccard(previous, current)
        similarities.append(similarity)
        if _normalize(previous) == _normalize(current):
            exact_repeat_count += 1
        if similarity >= 0.8:
            high_similarity_repeat_count += 1

    return CaseResult(
        case_id=str(case["id"]),
        variant=variant,
        passed=all(turn.passed for turn in turns),
        turns=turns,
        exact_repeat_count=exact_repeat_count,
        high_similarity_repeat_count=high_similarity_repeat_count,
        avg_similarity=statistics.mean(similarities) if similarities else 0.0,
        turns_to_resolution=turns_to_resolution,
    )


def _aggregate(case_results: list[CaseResult]) -> dict[str, Any]:
    total_turns = sum(len(case.turns) for case in case_results)
    passed_turns = sum(turn.passed for case in case_results for turn in case.turns)
    passed_cases = sum(case.passed for case in case_results)
    latencies = [turn.latency_ms for case in case_results for turn in case.turns]
    resolution_turns = [
        case.turns_to_resolution
        for case in case_results
        if case.turns_to_resolution is not None
    ]

    return {
        "cases": len(case_results),
        "cases_passed": passed_cases,
        "case_pass_rate": (passed_cases / len(case_results)) if case_results else 0.0,
        "turns": total_turns,
        "turns_passed": passed_turns,
        "turn_pass_rate": (passed_turns / total_turns) if total_turns else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "avg_reply_similarity": statistics.mean(
            [case.avg_similarity for case in case_results]
        )
        if case_results
        else 0.0,
        "exact_repeat_count": sum(case.exact_repeat_count for case in case_results),
        "high_similarity_repeat_count": sum(
            case.high_similarity_repeat_count for case in case_results
        ),
        "avg_turns_to_resolution": statistics.mean(resolution_turns)
        if resolution_turns
        else None,
    }


def _write_csv(path: Path, case_results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "case_id",
                "passed",
                "turn_index",
                "turn_passed",
                "latency_ms",
                "escalated",
                "sources_count",
                "failure_reasons",
                "message",
                "reply",
            ]
        )
        for case in case_results:
            for turn in case.turns:
                writer.writerow(
                    [
                        case.variant,
                        case.case_id,
                        case.passed,
                        turn.index,
                        turn.passed,
                        round(turn.latency_ms, 2),
                        turn.escalated,
                        turn.sources_count,
                        " | ".join(turn.failure_reasons),
                        turn.message,
                        turn.reply,
                    ]
                )


def _evaluate_variant(
    cases: list[dict[str, Any]],
    api_url: str,
    variant: str,
) -> tuple[dict[str, Any], list[CaseResult]]:
    with httpx.Client() as client:
        health = client.get(f"{api_url.rstrip('/')}/health", timeout=30.0)
        health.raise_for_status()
        case_results = [_run_case(client, api_url, variant, case) for case in cases]

    summary = {
        "variant": variant,
        "api_url": api_url,
        "health": health.json(),
        "metrics": _aggregate(case_results),
        "cases": [
            {
                "case_id": case.case_id,
                "passed": case.passed,
                "exact_repeat_count": case.exact_repeat_count,
                "high_similarity_repeat_count": case.high_similarity_repeat_count,
                "avg_similarity": case.avg_similarity,
                "turns_to_resolution": case.turns_to_resolution,
                "turns": [
                    {
                        "index": turn.index,
                        "message": turn.message,
                        "reply": turn.reply,
                        "passed": turn.passed,
                        "latency_ms": turn.latency_ms,
                        "escalated": turn.escalated,
                        "sources_count": turn.sources_count,
                        "failure_reasons": turn.failure_reasons,
                    }
                    for turn in case.turns
                ],
            }
            for case in case_results
        ],
    }
    return summary, case_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chat behavior with and without memory."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the JSON file with evaluation cases.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Evaluate a single running backend at this URL.",
    )
    parser.add_argument(
        "--variant",
        default="current",
        help="Label for the single-backend evaluation.",
    )
    parser.add_argument(
        "--baseline-url",
        default=None,
        help="URL for the baseline backend, usually memory off.",
    )
    parser.add_argument(
        "--candidate-url",
        default=None,
        help="URL for the candidate backend, usually memory on.",
    )
    parser.add_argument(
        "--baseline-label",
        default="without_memory",
        help="Label for the baseline backend.",
    )
    parser.add_argument(
        "--candidate-label",
        default="with_memory",
        help="Label for the candidate backend.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/eval_memory/results.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/eval_memory/results.csv"),
        help="Where to write the CSV report.",
    )
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    summaries: list[dict[str, Any]] = []
    all_case_results: list[CaseResult] = []

    if args.api_url:
        summary, case_results = _evaluate_variant(cases, args.api_url, args.variant)
        summaries.append(summary)
        all_case_results.extend(case_results)
    elif args.baseline_url and args.candidate_url:
        baseline_summary, baseline_results = _evaluate_variant(
            cases, args.baseline_url, args.baseline_label
        )
        candidate_summary, candidate_results = _evaluate_variant(
            cases, args.candidate_url, args.candidate_label
        )
        summaries.extend([baseline_summary, candidate_summary])
        all_case_results.extend(baseline_results)
        all_case_results.extend(candidate_results)
    else:
        raise SystemExit(
            "Use --api-url for one backend or --baseline-url and --candidate-url for a comparison."
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.output_csv, all_case_results)

    for summary in summaries:
        metrics = summary["metrics"]
        print(
            f"[{summary['variant']}] "
            f"cases_pass_rate={metrics['case_pass_rate']:.2%} "
            f"turn_pass_rate={metrics['turn_pass_rate']:.2%} "
            f"avg_latency_ms={metrics['avg_latency_ms']:.1f} "
            f"exact_repeats={metrics['exact_repeat_count']} "
            f"high_similarity_repeats={metrics['high_similarity_repeat_count']}"
        )
    print(f"JSON report: {args.output_json}")
    print(f"CSV report: {args.output_csv}")


if __name__ == "__main__":
    main()
