"""Per-call 'hmm' counts, output tokens and level-up positions for every run.

Each model call becomes one row: run, game, call index, analysis_step, level at
the turn, thinking tokens, output tokens (thinking + tool-call arguments), hmm
count, wait count. Level-ups come from the events sidecar: an action whose
level exceeds the previous level, or whose state turns WIN, is credited to the
end of the last call of its analysis_step.

Needs `pip install tokenizers` and the tokenizer from extract_transcript_calls.py.
Run from the main checkout, where ARC3-Inference/runs/ exists:

    python3 docs/reasoning/scripts/hmm_extract.py
"""

import csv
import gzip
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tokenizers import Tokenizer

RUNS = Path(__file__).resolve().parents[3] / "ARC3-Inference/runs/kaggle"
TOK = Path.home() / ".cache/arc3-reasoning/qwen3.8-flash-next-tokenizer.json"
OUT = Path(__file__).resolve().parents[1] / "data/hmm_calls.csv.gz"

TURN = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|")
HEADER = re.compile(r"^\[[A-Z][A-Z _]*(?::[^\]]*)?\]$")
HMM = re.compile(r"\bh+m{2,}\b", re.I)
WAIT = re.compile(r"\bwait\b", re.I)

_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(str(TOK))
    return _tok


def level_ups(events_path):
    """analysis_step -> list of (new_level, won) for actions that cleared a level."""
    ups, prev = {}, None
    for line in events_path.open():
        d = json.loads(line)
        if d.get("type") not in ("action", "initial"):
            continue
        try:
            lvl = int(d.get("level"))
        except (TypeError, ValueError):
            continue
        won = d.get("state") == "WIN"
        if d["type"] == "action" and prev is not None and (lvl > prev or (won and not ups.get("_won"))):
            step = d.get("analysis_step")
            if step not in (None, "None"):
                ups.setdefault(int(step), []).append(lvl)
            if won:
                ups["_won"] = True
        prev = lvl
    return ups


def tool_args(meta_lines):
    text = "\n".join(meta_lines)
    i = text.find("raw_tool_calls:")
    if i < 0:
        return ""
    try:
        calls = json.loads(text[i + len("raw_tool_calls:"):].strip())
        return "\n".join(c.get("function", {}).get("arguments", "") for c in calls)
    except Exception:
        return ""


def parse(job):
    run, transcript, events = job
    game = transcript.name.split("-")[0]
    ups = level_ups(events)
    won = ups.pop("_won", False)
    calls, section, step = [], None, None
    for line in transcript.read_text(errors="replace").split("\n"):
        m = TURN.match(line)
        if m:
            step = int(m.group(1))
            section = "turn"
            continue
        if HEADER.match(line):
            section = line
            if line == "[MODEL RESPONSE META]":
                calls.append({"step": step, "thinking": [], "meta": []})
            continue
        if section == "[THINKING]" and calls:
            calls[-1]["thinking"].append(line)
        elif section == "[MODEL RESPONSE META]" and calls:
            calls[-1]["meta"].append(line)
    if not calls:
        return []
    thinking = ["\n".join(c["thinking"]) for c in calls]
    args = [tool_args(c["meta"]) for c in calls]
    t_tok = [len(e.ids) for e in tok().encode_batch(thinking)]
    a_tok = [len(e.ids) for e in tok().encode_batch(args)]
    last_call_of_step = {}
    for i, c in enumerate(calls):
        last_call_of_step[c["step"]] = i
    lu_at = {}
    for s, lvls in ups.items():
        if s in last_call_of_step:
            lu_at[last_call_of_step[s]] = len(lvls)
    rows = []
    for i, c in enumerate(calls):
        rows.append({
            "run": run, "game": game, "gid": run + "/" + transcript.stem, "call": i, "step": c["step"],
            "think_tok": t_tok[i], "out_tok": t_tok[i] + a_tok[i],
            "hmm": len(HMM.findall(thinking[i])), "wait": len(WAIT.findall(thinking[i])),
            "levelups": lu_at.get(i, 0), "won": int(won), "n_calls": len(calls),
            "unmatched_ups": sum(len(v) for s, v in ups.items() if s not in last_call_of_step),
        })
    return rows


def main():
    jobs = []
    for run_dir in sorted(RUNS.glob("*/*/v*")):
        run = str(run_dir.relative_to(RUNS))
        for t in sorted((run_dir / "transcripts").glob("*_p0.txt")):
            ev = run_dir / "artifacts" / (t.stem + "_events.jsonl")
            if ev.exists():
                jobs.append((run, t, ev))
    print(len(jobs), "games", file=sys.stderr)
    with ProcessPoolExecutor(8) as pool, gzip.open(OUT, "wt", newline="") as f:
        w = None
        for rows in pool.map(parse, jobs, chunksize=4):
            for r in rows:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(r))
                    w.writeheader()
                w.writerow(r)


if __name__ == "__main__":
    main()
