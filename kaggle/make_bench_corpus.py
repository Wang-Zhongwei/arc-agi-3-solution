"""Build the serving benchmark's prompt corpus from a run's harness transcripts.

The benchmark (serving/kaggle/bench.py) replays real harness conversations so
the model writes what it writes in a game: grid reasoning followed by a
`python` tool call. Speculative acceptance depends on that content, so a word
salad prompt would measure a different accept length.

Each transcript (<game>_p0.txt, written by the tool agent) is turned into the
OpenAI message list the harness sent: the shared system prompt, then user
turns, assistant turns (reasoning_content + tool_calls) and tool results. Only
the first ~max-chars of each game is kept, cut where the model speaks next.

Usage (from the repo root):
    python kaggle/make_bench_corpus.py <transcripts_dir> [--games 16] [--max-chars 55000]
writes kaggle/bench_corpus.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "bench_corpus.json.gz"
STEP = re.compile(r"^--- analysis_step=\d+ .*---$")
HEADER = re.compile(
    r"^\[(SYSTEM PROMPT|USER PROMPT|ANALYZER STATUS|MODEL RESPONSE META|THINKING|ASSISTANT"
    r"|TOOL CALL: \w+|TOOL RESULT: \w+)\]$"
)
IMAGE_MARKER = "\n\n[image attached]"


def sections(text: str) -> list[tuple[str, str]]:
    """Split a transcript into (header, body) pairs, in order."""
    out: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        if STEP.match(line):
            out.append(("STEP", []))
        elif (m := HEADER.match(line)):
            out.append((m[1], []))
        elif out:
            out[-1][1].append(line)
    return [(kind, "\n".join(body).strip("\n")) for kind, body in out]


def _raw_tool_calls(meta: str) -> list[dict]:
    _, sep, rest = meta.partition("raw_tool_calls:")
    if not sep:
        return []
    try:
        calls = json.loads(rest.strip())
    except json.JSONDecodeError:
        return []
    return [
        {"id": c["id"], "type": "function",
         "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
        for c in calls if isinstance(c, dict) and c.get("id") and c.get("function")
    ]


def conversation(text: str) -> tuple[str, list[dict]]:
    """(system prompt, messages after it) for one transcript."""
    system, messages = "", []
    assistant: dict | None = None
    pending: list[str] = []  # tool_call ids still waiting for a result

    def close_assistant() -> None:
        nonlocal assistant
        if assistant is None:
            return
        answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
        calls = [c for c in assistant.get("tool_calls", []) if c["id"] in answered]
        if calls:
            assistant["tool_calls"] = calls
        else:
            assistant.pop("tool_calls", None)
        assistant = None

    for kind, body in sections(text):
        if kind == "SYSTEM PROMPT":
            system = system or body
        elif kind == "USER PROMPT":
            close_assistant()
            messages.append({"role": "user", "content": body.replace(IMAGE_MARKER, "")})
        elif kind == "MODEL RESPONSE META":
            close_assistant()
            assistant = {"role": "assistant", "content": "", "reasoning_content": ""}
            calls = _raw_tool_calls(body)
            if calls:
                assistant["tool_calls"] = calls
            pending = [c["id"] for c in calls]
            messages.append(assistant)
        elif kind == "THINKING" and assistant is not None:
            assistant["reasoning_content"] = body
        elif kind == "ASSISTANT" and assistant is not None:
            assistant["content"] = body
        elif kind.startswith("TOOL RESULT") and pending:
            messages.append({"role": "tool", "tool_call_id": pending.pop(0), "content": body})
    close_assistant()
    # Tool results are inserted after their assistant message; a result whose
    # assistant lost its call (above) is dropped so every tool message is answered.
    ids = {c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])}
    messages = [m for m in messages if m["role"] != "tool" or m["tool_call_id"] in ids]
    return system, messages


def truncate(messages: list[dict], max_chars: int) -> list[dict]:
    """The longest prefix under max_chars that ends where the model speaks next."""
    total, cut = 0, 0
    for i, m in enumerate(messages):
        total += len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        total += sum(len(c["function"]["arguments"]) for c in m.get("tool_calls", []))
        if total > max_chars:
            break
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if m["role"] in ("user", "tool") and (nxt is None or nxt["role"] == "assistant"):
            cut = i + 1
    return messages[:cut]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcripts", type=Path)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=55_000)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    systems, games = [], []
    for path in sorted(args.transcripts.glob("*_p0.txt")):
        system, messages = conversation(path.read_text(errors="replace"))
        messages = truncate(messages, args.max_chars)
        if not system or len(messages) < 4:
            continue
        systems.append(system)
        games.append({"name": path.name.split("-")[0], "messages": messages})
    if not games:
        raise SystemExit(f"no usable transcripts under {args.transcripts}")
    # Every stream shares the most common system prompt, as the harness does.
    system = max(set(systems), key=systems.count)
    games = games[: args.games]
    corpus = {"version": 1, "source": args.transcripts.parent.name, "system": system, "games": games}
    raw = json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode("utf-8")
    args.out.write_bytes(gzip.compress(raw, 9, mtime=0))
    turns = [sum(m["role"] != "assistant" for m in g["messages"]) for g in games]
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes gz, {len(raw)} raw): "
          f"{len(games)} games, system {len(system)} chars, model turns per game {turns}, "
          f"{systems.count(system)}/{len(systems)} games share the system prompt")


if __name__ == "__main__":
    main()
