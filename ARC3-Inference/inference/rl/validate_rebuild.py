"""Check context_rebuild against runs that saved full request logs.

For every logged request, the rebuilt request with the same analysis step and
request index is compared with what was actually sent:
- window: the retained history, as message signatures, equal or off by n;
- content: per message, whether the rebuilt text equals the logged text;
- size: rebuilt vs logged text chars, and the token estimate vs the server's
  prompt_tokens.

    python -m inference.rl.validate_rebuild runs/kaggle/<owner>/<notebook>/vNN [...]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from inference.rl.context_rebuild import estimate_tokens, message_chars, rebuild

NOTICE = "Earlier conversation has been trimmed."
REQUESTS = re.compile(r"(.+)_requests\.jsonl$")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def user_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return norm(content)
    return norm("\n".join(p.get("text") or "" for p in content or [] if p.get("type") != "image_url"))


def reasoning(message: dict[str, Any]) -> str:
    return message.get("reasoning") or message.get("reasoning_content") or ""


def arguments(message: dict[str, Any]) -> list[Any]:
    """Parsed tool-call arguments; string values right-stripped, since the
    transcript strips the trailing newline of code rebuilt from its markup."""
    out = []
    for call in message.get("tool_calls") or []:
        raw = call["function"].get("arguments") or "{}"
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, dict):
            value = {k: v.rstrip() if isinstance(v, str) else v for k, v in value.items()}
        out.append(value)
    return out


def tool_fields(message: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(message.get("content") or "")
    except json.JSONDecodeError:
        return {"stdout": message.get("content")}
    return payload if isinstance(payload, dict) else {"stdout": payload}


def signatures(messages: list[dict[str, Any]]) -> list[tuple]:
    """Role plus a content fingerprint, robust to the approximations the rebuild makes."""
    out, last = [], None
    for m in messages:
        role = m["role"]
        if role == "assistant":
            last = ("a", norm(reasoning(m))[:200], json.dumps(arguments(m), sort_keys=True)[:200])
            out.append(last)
        elif role == "tool":
            out.append(("t", last))
        else:
            out.append(("u", user_text(m)[:200]))
    return out


def as_rebuilt(message: dict[str, Any]) -> dict[str, Any]:
    out = dict(message)
    if "reasoning_content" in out:
        out["reasoning"] = out.pop("reasoning_content")
    return out


def window(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """History messages after the system prompt, and the count of trimmed-history notes dropped."""
    body = [m for m in messages if m["role"] != "system"]
    notes = [m for m in body if m["role"] == "user" and user_text(m).startswith(NOTICE)]
    return [m for m in body if not (m["role"] == "user" and user_text(m).startswith(NOTICE))], len(notes)


def offset(rebuilt: list[tuple], logged: list[tuple]) -> int | None:
    """Messages the rebuild kept beyond the logged window (negative: dropped), if they differ only at the start."""
    if rebuilt == logged:
        return 0
    short, long_ = sorted((rebuilt, logged), key=len)
    if long_[len(long_) - len(short):] == short:
        return len(rebuilt) - len(logged)
    return None


def transcript_for(run: Path, stem: str) -> str | None:
    path = run / "transcripts" / f"{stem}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    events = run / "artifacts" / f"{stem}_events.jsonl"
    if not events.exists():
        return None
    parts = []
    for line in events.open(encoding="utf-8"):
        event = json.loads(line)
        if event.get("type") == "analysis" and event.get("transcript"):
            parts.append(event["transcript"].rstrip("\n") + "\n")
    return "\n".join(parts)


def compare_game(run: Path, log: Path, stats: dict[str, Any]) -> None:
    stem = REQUESTS.fullmatch(log.name)[1]
    transcript = transcript_for(run, stem)
    if transcript is None:
        stats["games_without_transcript"] += 1
        return
    game = rebuild(transcript)
    # Only served requests reach the transcript, and only they get a response
    # record (whose messages are the request's), so the two pair up in order.
    # (step, index) alone is ambiguous: a turn retried after a timeout repeats it.
    logged = []
    for line in log.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("event") == "response":
            logged.append(((record.get("analysis_step"), record.get("request_index_within_turn")),
                           record["messages"], (record.get("usage") or {}).get("prompt_tokens")))
    stats["games"] += 1
    stats["unmatched"] += abs(len(logged) - len(game.requests))
    for (key, messages, tokens), request in zip(logged, game.requests):
        if key != (request.step, request.index):
            stats["misaligned"] += 1
            continue
        stats["requests"] += 1
        stats["uncertain"] += request.uncertain
        logged_window, notes = window(messages)
        rebuilt_window, _ = window(request.messages)
        stats["notes"] += notes
        shift = offset(signatures(rebuilt_window), signatures(logged_window))
        stats["offset"][shift if shift is None or abs(shift) <= 3 else ("<-3" if shift < 0 else ">3")] += 1
        stats["certain_exact"] += shift == 0 and not request.uncertain
        if shift is not None:
            # content of the messages both windows share
            n = min(len(rebuilt_window), len(logged_window))
            for a, b in zip(rebuilt_window[len(rebuilt_window) - n:], logged_window[len(logged_window) - n:]):
                role = b["role"]
                if role == "user":
                    same = user_text(a) == user_text(b) and message_chars(a)[1] == message_chars(as_rebuilt(b))[1]
                elif role == "assistant":
                    same = norm(reasoning(a)) == norm(reasoning(b)) and arguments(a) == arguments(b)
                else:
                    fa, fb = tool_fields(a), tool_fields(b)
                    same = all(norm(str(fa.get(k) or "")) == norm(str(fb.get(k) or "")) for k in ("stdout", "error"))
                    stats["retention_chars"] += len(fb.get("function_retention") or "")
                stats["content"][role][same] += 1
        rebuilt_chars = sum(message_chars(m)[0] for m in request.messages)
        logged_chars = sum(message_chars(as_rebuilt(m))[0] for m in messages)
        if logged_chars:
            stats["char_ratio"].append(rebuilt_chars / logged_chars)
        if tokens:
            stats["token_error"].append(estimate_tokens(request.messages) / tokens - 1)


def summary(stats: dict[str, Any]) -> dict[str, Any]:
    total = sum(stats["offset"].values()) or 1
    pct = lambda n: round(100 * n / total, 1)
    content = {role: f"{c[True]}/{c[True] + c[False]} ({round(100 * c[True] / max(1, c[True] + c[False]), 1)}%)"
               for role, c in stats["content"].items()}
    ratio, error = stats["char_ratio"], stats["token_error"]
    return dict(
        games=stats["games"], requests=stats["requests"], count_mismatch=stats["unmatched"], misaligned=stats["misaligned"],
        window_exact_pct=pct(stats["offset"][0]),
        window_within_1_pct=pct(sum(v for k, v in stats["offset"].items() if isinstance(k, int) and abs(k) <= 1)),
        uncertain_pct=pct(stats["uncertain"]),
        window_exact_among_certain_pct=round(100 * stats["certain_exact"] / max(1, total - stats["uncertain"]), 1),
        window_offsets={str(k): v for k, v in sorted(stats["offset"].items(), key=lambda kv: str(kv[0]))},
        content_exact=content,
        trimmed_notes_not_rebuilt=stats["notes"],
        lost_function_retention_chars=stats["retention_chars"],
        char_ratio_median=round(statistics.median(ratio), 4) if ratio else None,
        char_ratio_p10_p90=[round(x, 3) for x in statistics.quantiles(ratio, n=10)[::8]] if len(ratio) > 10 else None,
        token_estimate_abs_error_median_pct=round(100 * statistics.median(abs(e) for e in error), 1) if error else None,
    )


def new_stats() -> dict[str, Any]:
    return dict(games=0, games_without_transcript=0, requests=0, unmatched=0, misaligned=0, uncertain=0, certain_exact=0, notes=0, retention_chars=0,
                offset=Counter(), content=defaultdict(Counter), char_ratio=[], token_error=[])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="write the per-run and overall summary here")
    args = parser.parse_args(argv)
    overall, report = new_stats(), {}
    for run in args.runs:
        stats = new_stats()
        for log in sorted(run.glob("*_requests.jsonl")) + sorted(run.glob("artifacts/*_requests.jsonl")):
            compare_game(run, log, stats)
        for key in ("games", "games_without_transcript", "requests", "unmatched", "misaligned", "uncertain", "certain_exact", "notes", "retention_chars"):
            overall[key] += stats[key]
        overall["offset"].update(stats["offset"])
        for role, c in stats["content"].items():
            overall["content"][role].update(c)
        overall["char_ratio"] += stats["char_ratio"]
        overall["token_error"] += stats["token_error"]
        report[str(run)] = summary(stats)
        print(run, json.dumps(report[str(run)]), flush=True)
    report["overall"] = summary(overall)
    print("overall", json.dumps(report["overall"], indent=1))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
