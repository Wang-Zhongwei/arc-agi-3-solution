"""Extract per-turn thinking of each game on its own generated-token axis.

For every turn (transcript header `analysis_step=i | action=a`), records the
reasoning tokens of the turn's last model call - the call that emitted its
actions, which is what ARC3_PRIORITY_THINKING reads - at the game's cumulative
generated-token count once the turn's actions are done. Token counts per
action come from benchmark.json `history[].generated_tokens`; reasoning tokens
are transcript `reasoning_chars` / CHARS_PER_TOKEN.

Writes data/thinking.json  {run: {game: [[tokens_after_turn, reasoning_tokens], ...]}}

Usage (from the main checkout):
  python docs/allocation/scripts/extract_thinking.py --runs-root ARC3-Inference/runs
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from itertools import accumulate
from pathlib import Path

from extract_trajectories import RUNS

DATA = Path(__file__).resolve().parents[1] / "data"
CHARS_PER_TOKEN = 4.0
HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|")
REASONING = re.compile(r"^reasoning_chars: (\d+)")


def turns(transcript: Path) -> list[tuple[int, int, int]]:
    """(step, action number at turn start, reasoning chars of the turn's last call); resumed
    headers of the same step are merged."""
    out: list[list[int]] = []
    for line in transcript.open(errors="ignore"):
        m = HEADER.match(line)
        if m:
            step, action = int(m.group(1)), int(m.group(2))
            if not out or out[-1][0] != step:
                out.append([step, action, -1])
            continue
        m = REASONING.match(line)
        if m and out:
            out[-1][2] = int(m.group(1))
    return [tuple(t) for t in out if t[2] >= 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("ARC3-Inference/runs"))
    args = ap.parse_args()
    result: dict = {}
    gaps, last_call = [], []
    for run in RUNS:
        folder = args.runs_root / "kaggle" / run
        bench = json.loads((folder / "benchmark.json").read_text())
        games = {}
        for gr in bench["game_runs"]:
            cum = [0, *accumulate(h["generated_tokens"] for h in gr["history"])]
            ts = turns(folder / "transcripts" / f"{gr['game_id']}_p0.txt")
            series = []
            for i, (_, action, chars) in enumerate(ts):
                nxt = ts[i + 1][1] if i + 1 < len(ts) else len(gr["history"]) + 1
                done = min(len(cum) - 1, max(action, nxt) - 1)  # actions 1..done executed after this turn
                series.append([cum[done], round(chars / CHARS_PER_TOKEN)])
                last_call.append(chars / CHARS_PER_TOKEN)
            for i in range(1, len(series)):  # keep the token axis monotone
                series[i][0] = max(series[i][0], series[i - 1][0])
            games[gr["game_id"]] = series
            gaps.append(cum[-1] - (series[-1][0] if series else 0))
        result[run] = games
    (DATA / "thinking.json").write_text(json.dumps(result, separators=(",", ":")))
    n = sum(len(s) for g in result.values() for s in g.values())
    print(f"{len(result)} runs, {sum(len(g) for g in result.values())} games, {n} turns; "
          f"median last-call thinking {st.median(last_call):.0f} tokens; "
          f"median tokens after the last recorded turn {st.median(gaps):.0f}")


if __name__ == "__main__":
    main()
