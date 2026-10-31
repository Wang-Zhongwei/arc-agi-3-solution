"""Summarize a rollout_env_pod.sh run: the server's counters over time, GPU busy time, CPU by process.

Reads RESULTS/metrics.jsonl ({"t": unix seconds, "metrics": {counter: value}} every 10 s), RESULTS/gpu.csv and
RESULTS/top.log. Prints one JSON object: per-minute generated tokens/s, running and waiting requests, tokens per
step and prefix-cache hit rate, the same over the steady part of the run (after the opening prefill, before the
first games stop), mean GPU utilization, and the processes that used the most CPU on average.

    python -m inference.grpo.rollout_env_report RESULTS [--steady-from-min 3 --steady-to-min 28]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path


def counters(path: Path) -> list[tuple[float, dict]]:
    rows = []
    for line in path.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append((float(d["t"]), d.get("metrics") or {}))
    return rows


def between(rows: list[tuple[float, dict]], t0: float, t1: float) -> dict:
    inside = [(t, m) for t, m in rows if t0 <= t <= t1]
    if len(inside) < 2:
        return {}
    (ta, a), (tb, b) = inside[0], inside[-1]
    dt = tb - ta
    d = lambda k: b.get(k, 0.0) - a.get(k, 0.0)  # noqa: E731
    drafts = d("spec_decode_num_drafts_total")
    queries = d("prefix_cache_queries_total")
    running = [m.get("num_requests_running", 0.0) for _, m in inside]
    return {"minutes": round(dt / 60, 1), "gen_tok_s": round(d("generation_tokens_total") / dt, 1),
            "prompt_tok_s": round(d("prompt_tokens_total") / dt, 1),
            "uncached_prompt_tok_s": round((d("prompt_tokens_total") - d("prefix_cache_hits_total")) / dt, 1),
            "prefix_hit_rate": round(d("prefix_cache_hits_total") / queries, 3) if queries else None,
            "tokens_per_step": round(1 + d("spec_decode_num_accepted_tokens_total") / drafts, 3) if drafts else None,
            "running_mean": round(st.mean(running), 1), "running_min": min(running),
            "waiting_mean": round(st.mean(m.get("num_requests_waiting", 0.0) for _, m in inside), 1),
            "kv_max": round(max(m.get("kv_cache_usage_perc", 0.0) for _, m in inside), 3),
            "requests_done": d("request_success_total"), "preemptions": d("num_preemptions_total")}


def gpu_util(path: Path) -> dict:
    util = defaultdict(list)
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[1].strip().isdigit():
                util[row[1].strip()].append(float(row[2].strip().split()[0]))
    return {f"gpu{k}": round(st.mean(v), 1) for k, v in util.items() if v}


def cpu_by_process(path: Path, top: int = 12) -> dict:
    """Mean %CPU per command over the top snapshots (a process missing from a snapshot counts as 0)."""
    snapshots, total, idle = 0, defaultdict(float), []
    for line in path.open():
        if re.match(r"^\d\d:\d\d:\d\d$", line.strip()):
            snapshots += 1
        elif line.startswith("%Cpu"):
            m = re.search(r"([\d.]+) id", line)
            if m:
                idle.append(float(m.group(1)))
        else:
            parts = line.split()
            if len(parts) >= 12 and parts[0].isdigit():
                try:
                    total[parts[11]] += float(parts[8])
                except ValueError:
                    pass
    means = sorted(((cmd, cpu / max(snapshots, 1)) for cmd, cpu in total.items()), key=lambda x: -x[1])
    return {"snapshots": snapshots, "machine_idle_pct_mean": round(st.mean(idle), 1) if idle else None,
            "top_commands_pct": {cmd: round(cpu, 1) for cmd, cpu in means[:top]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--steady-from-min", type=float, default=3)
    ap.add_argument("--steady-to-min", type=float, default=28)
    args = ap.parse_args()
    rows = counters(args.results / "metrics.jsonl")
    t0 = rows[0][0]
    per_minute = []
    for m0 in range(0, int((rows[-1][0] - t0) // 60) + 1):
        s = between(rows, t0 + 60 * m0, t0 + 60 * (m0 + 1) + 5)
        if s:
            per_minute.append({"minute": m0, **{k: s[k] for k in ("gen_tok_s", "running_mean", "waiting_mean",
                                                                   "uncached_prompt_tok_s", "tokens_per_step")}})
    out = {"steady": between(rows, t0 + 60 * args.steady_from_min, t0 + 60 * args.steady_to_min),
           "per_minute": per_minute}
    if (args.results / "gpu.csv").exists():
        out["gpu_util_mean"] = gpu_util(args.results / "gpu.csv")
    if (args.results / "top.log").exists():
        out["cpu"] = cpu_by_process(args.results / "top.log")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
