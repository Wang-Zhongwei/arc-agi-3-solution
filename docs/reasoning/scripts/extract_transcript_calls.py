"""Per-call thinking tokens and actions for runs without request logs.

Runs such as our adaptive-reasoning-ucb v30 keep only transcripts, which hold
each call's thinking text but no token counts. This script re-tokenizes the
thinking text with the model's tokenizer and attributes actions per call, then
writes data/<name>_calls.csv for thinking_budget.py.

- Thinking tokens: the [THINKING] block of each call, encoded with the
  Qwen3.8-Flash-Next tokenizer. On Daniel's run, which logs exact
  usage.reasoning_tokens, this totals 99.9% of the logged count (r = 1.000).
- Actions: each turn header carries the action counter at turn start. A turn's
  actions are the jump to the next turn's counter, credited to the turn's last
  call; a turn ends once an action executes. This matches the request-log
  counter on 2,476 of Daniel's 2,477 calls.

Needs `pip install tokenizers`. The tokenizer file is downloaded on first use.

    python3 docs/reasoning/scripts/extract_transcript_calls.py
"""

import csv
import re
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

REPORT = Path(__file__).resolve().parents[1]
RUNS = REPORT.parents[2] / "duck-harness/ARC3-Inference/runs/kaggle"
SOURCES = {"v30": RUNS / "jonathanwang2022/adaptive-reasoning-ucb/v30"}
TOKENIZER_URL = ("https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
                 "/resolve/main/tokenizer.json")
TOKENIZER_PATH = Path.home() / ".cache/arc3-reasoning/qwen3.8-flash-next-tokenizer.json"

TURN = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|")
HEADER = re.compile(r"^\[[A-Z][A-Z _]*(?::[^\]]*)?\]$")


def tokenizer():
    if not TOKENIZER_PATH.exists():
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOKENIZER_URL, TOKENIZER_PATH)
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def summary_actions(run):
    totals = {}
    for line in (run / "summary.txt").read_text().splitlines():
        if "score=" in line:
            game, rest = line.strip().split(":", 1)
            kv = dict(part.split("=") for part in rest.replace(" ", "").split(","))
            totals[game] = int(kv["actions"])
    return totals


def transcript_calls(path, total_actions):
    turns, section, call = [], None, None
    for line in path.read_text().split("\n"):
        turn = TURN.match(line)
        if turn:
            turns.append({"action": int(turn.group(2)), "calls": []})
            section = "turn"
        elif HEADER.match(line):
            section = line
            if line == "[MODEL RESPONSE META]":
                call = {"thinking": [], "actions": 0}
                turns[-1]["calls"].append(call)
        elif section == "[THINKING]":
            call["thinking"].append(line)
    turns = [t for t in turns if t["calls"]]
    for turn, following in zip(turns, turns[1:] + [None]):
        end = following["action"] if following else total_actions + 1
        turn["calls"][-1]["actions"] = end - turn["action"]
    return [call for turn in turns for call in turn["calls"]]


def main():
    tok = tokenizer()
    for name, run in SOURCES.items():
        totals = summary_actions(run)
        rows = []
        for path in sorted((run / "transcripts").glob("*_p0.txt")):
            game = path.name.split("_p0")[0]
            calls = transcript_calls(path, totals[game])
            assert sum(c["actions"] for c in calls) == totals[game], game
            for i, call in enumerate(calls):
                text = "\n".join(call["thinking"]).strip()
                rows.append(dict(game=game.split("-")[0], call=i,
                                 thinking_tokens=len(tok.encode(text).ids),
                                 actions=call["actions"]))
        out = REPORT / "data" / f"{name}_calls.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} calls -> {out.relative_to(REPORT.parents[1])}")


if __name__ == "__main__":
    main()
