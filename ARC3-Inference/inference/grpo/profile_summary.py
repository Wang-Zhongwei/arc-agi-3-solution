"""Where a vLLM decode step's GPU time goes, from one server's torch-profiler traces (one file per rank).

Kernels are grouped by the launch that issued them (a CUDA-graph replay is one launch). Graph launches split by
GPU time into the target model's forward (the large ones) and smaller ones, which in decode are the drafter's
passes; kernels launched one by one (sampling, rejection, bookkeeping) are "eager". r = mean small graph / mean
target forward. Each class lists its top kernels and the graph launches a duration histogram, so the split can be
checked instead of trusted (a step that also prefills runs piecewise graphs, which land among the small ones).

    python -m inference.grpo.profile_summary TRACE_DIR > profile_summary.json
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

TARGET_SHARE = 0.3  # a graph launch is the target forward when it takes >= 30% of the longest graph launch


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def summarize(path: Path) -> dict:
    events = load(path)
    api = {}  # correlation id -> the runtime call that launched it
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                api[corr] = e.get("name", "?")
    launches = defaultdict(lambda: {"us": 0.0, "n": 0, "kernels": Counter()})
    first, last, busy = math.inf, 0.0, 0.0
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            corr = (e.get("args") or {}).get("correlation")
            dur = float(e.get("dur", 0))
            g = launches[corr]
            g["us"] += dur
            g["n"] += 1
            g["kernels"][e.get("name", "?")] += dur
            first, last, busy = min(first, e["ts"]), max(last, e["ts"] + dur), busy + dur
    graphs = {c: g for c, g in launches.items() if "graph" in api.get(c, "").lower()}
    longest = max((g["us"] for g in graphs.values()), default=0.0)
    classes = defaultdict(lambda: {"n": 0, "us": 0.0, "kernels": Counter()})
    for c, g in launches.items():
        if c in graphs:
            kind = "target_forward" if g["us"] >= TARGET_SHARE * longest else "small_graph"
        else:
            kind = "eager"
        k = classes[kind]
        k["n"] += 1
        k["us"] += g["us"]
        k["kernels"].update(g["kernels"])
    bins = Counter(int(math.floor(math.log2(max(g["us"], 1.0)))) for g in graphs.values())
    wall = last - first if last > first else 0.0
    out = {"file": path.name, "wall_ms": round(wall / 1e3, 1), "kernel_busy_ms": round(busy / 1e3, 1),
           "graph_launch_us_histogram": {f"{2 ** b}-{2 ** (b + 1)}": n for b, n in sorted(bins.items())}, "classes": {}}
    for kind, k in classes.items():
        out["classes"][kind] = {
            "launches": k["n"], "total_ms": round(k["us"] / 1e3, 1), "share_of_busy": round(k["us"] / busy, 3) if busy else None,
            "mean_ms_per_launch": round(k["us"] / 1e3 / k["n"], 3) if k["n"] else None,
            "top_kernels": [[name[:90], round(us / 1e3, 1)] for name, us in k["kernels"].most_common(8)]}
    t, s = out["classes"].get("target_forward"), out["classes"].get("small_graph")
    if t and s and t["launches"]:
        out["small_graphs_per_target_forward"] = round(s["launches"] / t["launches"], 2)
        out["r_per_small_graph"] = round(s["mean_ms_per_launch"] / t["mean_ms_per_launch"], 4)
    return out


def main() -> None:
    root = Path(sys.argv[1])
    files = sorted(p for p in root.rglob("*") if p.is_file() and (p.name.endswith(".json") or p.name.endswith(".json.gz")))
    print(json.dumps([summarize(p) for p in files], indent=1))


if __name__ == "__main__":
    main()
