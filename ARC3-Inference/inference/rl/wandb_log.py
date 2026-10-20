"""train_rft events in wandb: live from the training process, or replayed from a pulled run folder.

One mapping for both, so a replayed run looks like a live one:
- optimizer steps (x: train/step): train/loss, train/lr, train/grad_norm, train/trained_tokens, train/step_s;
- samples (x: sample/index): sample/loss, sample/tokens, sample/saved_tokens, sample/tokens_per_s,
  sample/forward_s, sample/backward_s, sample/layer (1 = recompute "layer"), sample/peak_gb,
  sample/arena_used_gb, sample/major_faults, sample/failed;
- validation (x: train/step): val/loss, val/tokens;
- GPU (replay only, x: gpu/elapsed_s; a live run gets wandb's own system metrics): gpu/util, gpu/power_w, gpu/memory_gb.
The key comes from WANDB_API_KEY (or a wandb login); it is never logged.

    python -m inference.rl.wandb_log RUN_DIR --project arc3-rft --name rft-train-v1-<pod> [--entity ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def step_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {"train/step": row["step"], "train/loss": row.get("loss"), "train/lr": row.get("lr"),
            "train/grad_norm": row.get("grad_norm"), "train/trained_tokens": row.get("trained_tokens"),
            "train/step_s": row.get("seconds"), "train/epoch": row.get("epoch")}


def sample_metrics(row: dict[str, Any], index: int) -> dict[str, float]:
    peak = max((row.get("peak_gb") or {}).values(), default=None)
    faults = row.get("major_faults")
    return {"sample/index": index, "sample/step": row.get("step"), "sample/loss": row.get("loss"),
            "sample/tokens": row.get("tokens"), "sample/saved_tokens": row.get("saved_tokens"),
            "sample/tokens_per_s": row.get("tokens_per_s"), "sample/forward_s": row.get("forward_s"),
            "sample/backward_s": row.get("backward_s"), "sample/layer": float(row.get("recompute_mode") == "layer"),
            "sample/peak_gb": peak, "sample/arena_used_gb": row.get("arena_used_gb"),
            "sample/major_faults": sum(faults) if faults else None, "sample/failed": float(bool(row.get("error")))}


class WandbLogger:
    """Turns train_rft log rows into wandb metrics; `run` is a wandb run (or a stand-in in tests)."""

    def __init__(self, run):
        self.run = run
        self.samples = 0
        self.step = 0
        for axis, pattern in (("train/step", "train/*"), ("sample/index", "sample/*"), ("train/step", "val/*"),
                              ("gpu/elapsed_s", "gpu/*")):
            run.define_metric(axis)
            run.define_metric(pattern, step_metric=axis)

    def event(self, row: dict[str, Any]) -> None:
        kind = row.get("event")
        if kind == "step":
            self.step = row["step"] + 1
            self._log(step_metrics(row))
        elif kind == "sample":
            self.samples += 1
            self._log(sample_metrics(row, self.samples))
        elif kind == "val":
            self._log({"train/step": self.step, "val/loss": row.get("loss"), "val/tokens": row.get("tokens")})
        elif kind == "arena":
            self.run.summary.update({f"arena/{k}": v for k, v in row.items() if k in
                                     ("cap_gb", "limit_gb", "ple_gb", "process_gb", "reserve_s")})
        elif kind in ("skip", "abort", "time_limit", "resumed"):
            self.run.summary[f"events/{kind}"] = self.run.summary.get(f"events/{kind}", 0) + 1

    def _log(self, metrics: dict[str, Any]) -> None:
        self.run.log({k: v for k, v in metrics.items() if v is not None})


def start(project: str, name: str | None, run_id: str | None, config: dict[str, Any], entity: str | None = None):
    import wandb
    return wandb.init(project=project, entity=entity, name=name, id=run_id, resume="allow" if run_id else None,
                      config=config)


def replay(run_dir: Path, logger: WandbLogger) -> dict[str, int]:
    log = run_dir / "train" / "train_log.jsonl"
    if not log.exists():
        log = run_dir / "results" / "train_log.jsonl"
    rows = [json.loads(line) for line in open(log)]
    for row in rows:
        logger.event(row)
    gpu_rows = 0
    first = next((r["wall_start"] for r in rows if r.get("wall_start")), None)
    gpu = run_dir / "results" / "gpu.csv"
    if gpu.exists():
        for rec in csv.reader(open(gpu)):
            try:
                stamp = datetime.strptime(rec[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                first = first if first is not None else stamp
                logger._log({"gpu/elapsed_s": stamp - first, "gpu/util": float(rec[1].strip().rstrip(" %")),
                             "gpu/power_w": float(rec[2].strip().rstrip(" W")),
                             "gpu/memory_gb": float(rec[3].strip().rstrip(" MiB")) / 1024})
                gpu_rows += 1
            except (ValueError, IndexError):
                continue
    return dict(events=len(rows), samples=logger.samples, gpu_rows=gpu_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a pulled run folder (train/train_log.jsonl, results/gpu.csv)")
    parser.add_argument("--project", default="arc3-rft")
    parser.add_argument("--entity")
    parser.add_argument("--name")
    parser.add_argument("--id", help="wandb run id (a replay into an existing run)")
    args = parser.parse_args(argv)
    log = args.run / "train" / "train_log.jsonl"
    rows = [json.loads(line) for line in open(log if log.exists() else args.run / "results" / "train_log.jsonl")]
    config = next((r.get("args") for r in rows if r.get("event") == "loaded"), {}) or {}
    run = start(args.project, args.name or args.run.name, args.id, dict(config, replayed_from=str(args.run)), args.entity)
    counts = replay(args.run, WandbLogger(run))
    print("replayed", counts, "->", getattr(run, "url", None))
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
