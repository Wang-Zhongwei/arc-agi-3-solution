"""Base-model thinking tokens per action on levels 1-2, per game: tau of the GRPO reward's thinking factor.

For every first-pass attempt of the runs the token-budget analysis uses (25-game Kaggle runs scoring >= 30, no RFT
model):
- thinking tokens: the reasoning of every request sent on L1 or L2 (the transcript rebuilt by
  ARC3-Inference/inference/rft/context_rebuild.py, each request's level from the run's analysis events), tokenized
  with the model's tokenizer;
- actions: TAAF's actions_per_level[0] + actions_per_level[1] (benchmark.json, RESETs included).
tau = the median of thinking / actions over the attempts with at least 5 actions on these levels.

Transcripts keep reasoning normalised (blank lines dropped), so tau runs a little low; the bias is the same for every
rollout of a game and cancels in its group's advantages.

Writes docs/RL/data/thinking_per_action.json.

    PYTHONPATH=ARC3-Inference python docs/RL/scripts/thinking_per_action.py --tokenizer DIR [--runs RUNS_DIR]
"""

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(REPO, "ARC3-Inference")]
from inference.rft.context_rebuild import rebuild_file  # noqa: E402

OUT = os.path.join(REPO, "docs", "RL", "data", "thinking_per_action.json")
MIN_RUN_SCORE = 30  # as l1_l2_token_budget.py: drops crashed and 5-minute smoke runs
MIN_ACTIONS = 5
_TOKENIZER = None


def load_runs(runs_dir):
    """(relative run dir, evaluation) of the runs l1_l2_token_budget.py uses."""
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "*", "v*", "evaluation.json"))):
        run_dir = os.path.dirname(path)
        evaluation = json.load(open(path))
        if len(evaluation.get("games", {})) != 25 or evaluation["score"] < MIN_RUN_SCORE:
            continue
        if "/rft-model/" in run_dir:
            continue
        runs.append(run_dir)
    return runs


def step_levels(events_path):
    """analysis_step -> the level the game was on at that turn."""
    out = {}
    for line in open(events_path):
        event = json.loads(line)
        if event.get("type") == "analysis":
            out[int(event["analysis_step"])] = int(event["level"])
    return out


def one(job):
    global _TOKENIZER
    tokenizer_path, run_dir, transcript, actions_per_level = job
    if _TOKENIZER is None:
        from tokenizers import Tokenizer
        _TOKENIZER = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
    events = Path(run_dir) / "artifacts" / f"{Path(transcript).stem}_events.jsonl"
    if not events.exists():
        return None
    levels = step_levels(events)
    game = rebuild_file(Path(transcript))
    thinking = 0
    for request in game.requests:
        if levels.get(request.step) in (1, 2):
            text = request.reply.get("reasoning") or ""
            if text:
                thinking += len(_TOKENIZER.encode(text, add_special_tokens=False).ids)
    actions = int(actions_per_level[0]) + int(actions_per_level[1])
    return dict(run=os.path.relpath(run_dir), game=Path(transcript).stem[:4], thinking=thinking, actions=actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="directory with the model's tokenizer.json")
    parser.add_argument("--runs", default=os.path.join(REPO, "ARC3-Inference", "runs", "kaggle"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = []
    runs = load_runs(args.runs)
    for run_dir in runs:
        bench = os.path.join(run_dir, "benchmark.json")
        if not os.path.exists(bench):
            continue
        actions = {g["game_id"]: g.get("actions_per_level") or [] for g in json.load(open(bench))["game_runs"]}
        for transcript in sorted(glob.glob(os.path.join(run_dir, "transcripts", "*_p0.txt"))):
            game_id = Path(transcript).stem.rsplit("_p", 1)[0]
            apl = actions.get(game_id)
            if apl and len(apl) >= 2:
                jobs.append((args.tokenizer, run_dir, transcript, apl))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for r in pool.map(one, jobs, chunksize=4) if r]
    per_game = defaultdict(list)
    for r in rows:
        if r["actions"] >= MIN_ACTIONS:
            per_game[r["game"]].append(r["thinking"] / r["actions"])
    games = {}
    for game, ratios in sorted(per_game.items()):
        q = st.quantiles(ratios, n=4) if len(ratios) >= 4 else [None, None, None]
        games[game] = dict(tau=round(st.median(ratios), 1), attempts=len(ratios),
                           p25=round(q[0], 1) if q[0] is not None else None,
                           p75=round(q[2], 1) if q[2] is not None else None)
    out = dict(description="Base-model median thinking tokens per action on L1-L2 (thinking_per_action.py)",
               runs=len(runs), attempts=len(rows), min_actions=MIN_ACTIONS, games=games)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"{len(rows)} attempts from {len(runs)} runs -> {OUT}")
    for game, row in games.items():
        print(f"  {game}: tau {row['tau']} (p25 {row['p25']}, p75 {row['p75']}, {row['attempts']} attempts)")


if __name__ == "__main__":
    main()
