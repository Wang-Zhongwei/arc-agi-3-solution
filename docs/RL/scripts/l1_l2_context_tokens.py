"""Context tokens of each attempt at the L1 and L2 clears, as if the harness never trimmed.

GRPO rollouts on levels 1-2 run without context trimming, so the sequence the model and the
trainer see is the whole message stream: system prompt, every observation, tool result and image,
and every generated token. This rebuilds that stream from each run's transcripts
(ARC3-Inference/inference/rft/context_rebuild.py, token estimates within ~3% of request logs)
and records its size when the first L2 request is sent (L1 cleared), when the first L3 request is
sent (L2 cleared), or at the end of the game for an attempt that did not get there.

Writes docs/RL/data/l1_l2_context.json, read by l1_l2_token_budget.py.

    ARC3-Inference/.venv/bin/python docs/RL/scripts/l1_l2_context_tokens.py
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "ARC3-Inference")]
from l1_l2_token_budget import RUNS_DIR, load_runs  # noqa: E402
from inference.rft.context_rebuild import TOKENS_FIXED, estimate_tokens, rebuild  # noqa: E402

OUT = os.path.join(REPO, "docs", "RL", "data", "l1_l2_context.json")


def levels(events_path):
    """analysis_step -> current level (1-based) at that analyzer turn, and the highest level reached."""
    at_step, top = {}, 1
    for line in open(events_path):
        event = json.loads(line)
        if event.get("type") == "analysis":
            at_step[event["analysis_step"]] = event["level"]
        if isinstance(event.get("level"), int):
            top = max(top, event["level"])
    return at_step, top


def context_at(game, request):
    """Untrimmed context sent with this request: system prompt + every stream message before its reply."""
    return estimate_tokens([{"role": "system", "content": game.system}] + game.stream[:request.stream_end])


def one(job):
    run, path = job
    events = Path(path).parent.parent / "artifacts" / f"{Path(path).stem}_events.jsonl"
    if not events.exists():
        return None
    at_step, top = levels(events)
    game = rebuild(Path(path).read_text(encoding="utf-8"))
    if not game.requests:
        return None
    end = estimate_tokens([{"role": "system", "content": game.system}] + game.stream)
    first = {}
    for request in sorted(game.requests, key=lambda r: r.stream_end):
        level = at_step.get(request.step)
        if level is not None and level >= 2 and 2 not in first:
            first[2] = context_at(game, request)
        if level is not None and level >= 3 and 3 not in first:
            first[3] = context_at(game, request)
    l1 = first.get(2, end if top >= 2 else None)
    l2 = first.get(3, end if top >= 3 else None)
    # estimate_tokens adds a per-request overhead; a reply alone does not carry it
    generated = sum(estimate_tokens([r.reply]) - TOKENS_FIXED for r in game.requests
                    if at_step.get(r.step) in (1, 2))
    return {"run": run, "game": Path(path).stem[:4], "levels": top - 1,
            "l1": l1, "l2": l2, "end": end, "generated_l1_l2": generated}


def main():
    jobs = []
    for run, _ in load_runs():
        for path in sorted(Path(RUNS_DIR, run, "transcripts").glob("*_p0.txt")):
            jobs.append((run, str(path)))
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(one, jobs, chunksize=4) if r]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=0)
    print(f"{len(rows)} attempts from {len({r['run'] for r in rows})} runs ({len(jobs)} transcripts) -> {OUT}")


if __name__ == "__main__":
    main()
