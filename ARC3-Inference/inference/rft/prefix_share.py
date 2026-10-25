"""How much of each RFT sample lies before its first trained token: the savings of prefix stop-grad.

A sample is one segment (requests between two trims, which share their prefix) times one level: the
level's replies are trained, and everything before its first request's reply is context. With prefix
stop-grad that context runs forward only (no checkpoint, recompute or backward). Token counts are
context_rebuild's estimates; levels come from the run's events (analysis_step -> level).

    PYTHONPATH=. python inference/rft/prefix_share.py runs/kaggle/ticbudja/reset-baseline/v01 [...]
"""
import sys
from pathlib import Path
from statistics import median

from inference.rft.context_rebuild import estimate_tokens, rebuild_file, run_tokens_per_image

FULL, FORWARD = 3.5, 1.0  # relative cost per token: forward + recompute + backward vs forward only


def levels(events_path):
    """analysis_step -> level at the start of that analyzer turn, and the levels that were completed."""
    import json

    at_step, won = {}, set()
    for line in open(events_path):
        e = json.loads(line)
        if e.get("type") == "analysis":
            at_step[e["analysis_step"]] = e["level"]
        elif e.get("type") == "action" and e.get("level_completed"):
            won.add(e["level"] - 1 if e.get("level") is not None else None)  # level after the completing action
    return at_step, won


def samples(game, at_step, won, only_won, per_image):
    """(prefix, total, trained) per (segment, level): requests with the same history_start share a prefix."""
    groups = {}
    for r in game.requests:
        r.level = at_step.get(r.step)
        if r.uncertain or r.level is None or (only_won and r.level not in won):
            continue
        groups.setdefault((r.history_start, r.level), []).append(r)
    out = []
    for (_, level), reqs in groups.items():
        reqs.sort(key=lambda r: r.stream_end)
        prefix = estimate_tokens(reqs[0].messages, per_image)
        total = estimate_tokens(reqs[-1].messages, per_image) + estimate_tokens([reqs[-1].reply])
        trained = sum(estimate_tokens([r.reply]) for r in reqs)
        out.append((prefix, total, trained, len(reqs)))
    return out


for run in sys.argv[1:]:
  for only_won in (False, True):
      rows = []
      for path in sorted(Path(run, "transcripts").glob("*.txt")):
          events = Path(run, "artifacts", f"{path.stem}_events.jsonl")
          if not events.exists():
              continue
          try:
              at_step, won = levels(events)
              rows += samples(rebuild_file(path), at_step, won, only_won, run_tokens_per_image(Path(run)))
          except Exception as e:  # noqa: BLE001
              print("skip", path.name, type(e).__name__, str(e)[:80])
      if not rows:
          print(run, "no samples", only_won)
          continue
      prefix = sum(r[0] for r in rows)
      total = sum(r[1] for r in rows)
      trained = sum(r[2] for r in rows)
      shares = [r[0] / r[1] for r in rows]
      cost_now = FULL * total
      cost_stop = FORWARD * prefix + FULL * (total - prefix)
      print(f"{Path(run).parent.name}/{Path(run).name} {'won levels' if only_won else 'all levels'}: {len(rows)} samples (segment x level), "
            f"{total / 1e6:.1f}M tokens, trained {trained / total:.1%}; "
            f"prefix share: token-weighted {prefix / total:.1%}, median {median(shares):.1%}; "
            f"median sample {median(r[1] for r in rows) / 1e3:.0f}K tokens, {median(r[3] for r in rows)} requests; "
            f"step cost x{cost_now / cost_stop:.2f} faster with prefix stop-grad")
