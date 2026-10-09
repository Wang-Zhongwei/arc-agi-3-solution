"""Rough generated-token statistics per reply and per turn for one Kaggle run.

The transcripts record each reply's characters (reasoning, content, tool-call
arguments) but not its tokens. Like the harness's own estimate, a reply's
tokens are its characters divided by the game's characters-per-token divisor:
the latest `text_token_calibration` printed in that game's transcript, or the
seed of 3 before the first one. The estimated total is checked against the
exact generated-token total in benchmark.json.

    python docs/tools/scripts/turn_token_stats.py RUN_DIR
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

TURN_RE = re.compile(r"^--- analysis_step=\d+ \| ", re.M)
META_RE = re.compile(r"^\[MODEL RESPONSE META\]\n", re.M)
CALIBRATION_RE = re.compile(r"text_token_calibration: ([0-9.]+) chars/token")
SEED_CHARS_PER_TOKEN = 3.0
YIELD_TOKENS = 2048


def reply_chars(meta: str) -> int:
    """Generated characters of one reply: reasoning + content + tool-call arguments."""
    reasoning = int(re.search(r"^reasoning_chars: (\d+)", meta, re.M).group(1))
    content = int(re.search(r"^content_chars: (\d+)", meta, re.M).group(1))
    raw = re.search(r"^raw_tool_calls:\n", meta, re.M)
    args = 0
    if raw and meta[raw.end():].startswith("["):
        calls, _ = json.JSONDecoder().raw_decode(meta[raw.end():])
        args = sum(len(c.get("function", {}).get("arguments") or "") for c in calls)
    return reasoning + content + args


def outcome(block: str) -> str:
    if "message: Step executed." in block:
        return "executed"
    m = re.search(r"message: Yielded control to solver: (\w+)\.", block)
    return m.group(1) if m else "other"


def game_turns(transcript: Path) -> list[dict]:
    text = transcript.read_text()
    heads = [m.start() for m in TURN_RE.finditer(text)] + [len(text)]
    divisor, turns = SEED_CHARS_PER_TOKEN, []
    for start, end in zip(heads, heads[1:]):
        block = text[start:end]
        tokens = []
        # The divisor in force for a reply is the last calibration printed before it.
        cursor = 0
        for m in META_RE.finditer(block):
            for c in CALIBRATION_RE.finditer(block, cursor, m.start()):
                divisor = float(c.group(1))
            cursor = m.start()
            tokens.append(reply_chars(block[m.end():]) / divisor)
        for c in CALIBRATION_RE.finditer(block, cursor):
            divisor = float(c.group(1))
        turns.append({"outcome": outcome(block), "replies": tokens})
    return turns


def describe(values: list[float]) -> str:
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return (f"n={len(v)} mean={statistics.mean(v):,.0f} median={statistics.median(v):,.0f} "
            f"p10={q(.10):,.0f} p90={q(.90):,.0f} p99={q(.99):,.0f} max={v[-1]:,.0f}")


def main(run_dir: Path) -> None:
    turns = [t for p in sorted((run_dir / "transcripts").glob("*_p0.txt")) for t in game_turns(p)]
    replies = [r for t in turns for r in t["replies"]]
    bench = json.loads((run_dir / "benchmark.json").read_text())
    exact = sum(int(g["solver_note"].removeprefix("tokens=")) for g in bench["game_runs"])
    print(f"estimated generated tokens {sum(replies):,.0f}; exact {exact:,} "
          f"(estimate / exact = {sum(replies) / exact:.2f})")

    counts: dict[str, int] = {}
    for t in turns:
        counts[t["outcome"]] = counts.get(t["outcome"], 0) + 1
    print(f"turns {len(turns)} {counts}; replies {len(replies)}")
    print("tokens per reply:", describe(replies))
    for cut in (2048, 4096, 8192):
        print(f"  replies >= {cut}: {sum(r >= cut for r in replies) / len(replies):.1%}")

    groups = {"all turns": turns,
              "acting turns": [t for t in turns if t["outcome"] == "executed"],
              "yielded turns (turn_token_budget)": [t for t in turns if t["outcome"] == "turn_token_budget"]}
    for name, group in groups.items():
        per_turn = [sum(t["replies"]) for t in group]
        print(f"\n{name}: tokens per turn", describe(per_turn))
        print("  replies per turn", describe([len(t["replies"]) for t in group]))
        print(f"  share of all tokens {sum(per_turn) / sum(replies):.1%}")
        for cut in (2048, 4096, 8192):
            print(f"  turns >= {cut}: {sum(x >= cut for x in per_turn) / len(per_turn):.1%}")
    yielded = groups["yielded turns (turn_token_budget)"]
    before = [sum(t["replies"][:-1]) for t in yielded if t["replies"]]
    last = [t["replies"][-1] for t in yielded if t["replies"]]
    print("\nyielded turns: tokens before the reply that crossed 2,048", describe(before))
    print("yielded turns: tokens in the reply that crossed 2,048", describe(last))
    print(f"yielded turns crossed by a single reply: {sum(b == 0 for b in before) / len(before):.1%}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
