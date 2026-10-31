"""Numbers behind docs/serving/drafters.md: draft-checkpoint bytes and speculative-decoding acceptance.

bytes: tensor bytes by role and dtype, read from each file's safetensors header (two HTTP range requests per file,
no weights downloaded). URLs are Hugging Face `resolve/<commit>/<file>` links or Kaggle model-file links
(https://www.kaggle.com/api/v1/models/<owner>/<model>/<framework>/<variation>/<version>/download/<file>, read with
KAGGLE_API_TOKEN or ~/.kaggle/access_token). `--prefix mtp.` keeps the draft head's tensors of a full checkpoint.

accept: mean acceptance length and per-position rates from a vLLM server log ("SpecDecoding metrics" lines,
token-weighted), or SGLang's mean "accept len" over its decode lines. `--until M` keeps the lines of the first M minutes
after the server's first metrics line (vLLM only), so runs of different lengths compare over the same window.

timeline: one row per `--bucket` minutes after the first metrics line of a vLLM log: the means of the 10 s engine
lines (prompt and generation tok/s, KV cache usage with its maximum, prefix-cache hit rate, waiting requests) and the
acceptance of that bucket.

steps: decode-step time from a vLLM server log's 10-s stats lines. Each interval's steps are its drafts (drafted
tokens / k) over the running requests, so step time = interval x running / drafts. Prefill tokens per step are the
interval's prompt tokens over its steps. A least-squares line, step = fixed + slope x prefill tokens, splits off the
prefill. Intervals with fewer than --min-running requests are skipped (start-up and wind-down).

    python -m inference.tools.drafter_stats bytes URL [URL ...] [--prefix mtp.]
    python -m inference.tools.drafter_stats accept runs/kaggle/<owner>/<notebook>/vNN/vllm-openai-server.log [--until M]
    python -m inference.tools.drafter_stats timeline runs/kaggle/<owner>/<notebook>/vNN/vllm-openai-server.log [--bucket 5]
    python -m inference.tools.drafter_stats steps runs/kaggle/<owner>/<notebook>/vNN/vllm-openai-server.log
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROLES = (  # first match wins
    ("embed_tokens", "embed_tokens"), ("lm_head", "lm_head"), ("mlp.experts.", "routed experts"),
    ("shared_expert", "shared expert"), ("attn", "attention"), ("mlp", "router / dense MLP"),
)


def role(name: str) -> str:
    return next((label for key, label in ROLES if key in name), "fc / norms / hyper-connections / other")


def header(url: str) -> dict:
    headers = {}
    if "kaggle.com" in url:
        token = os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "access_token").read_text().strip()
        headers["Authorization"] = f"Bearer {token}"

    def get(span: str) -> bytes:
        request = urllib.request.Request(url, headers={**headers, "Range": f"bytes={span}"})
        return urllib.request.urlopen(request, timeout=120).read()

    size = struct.unpack("<Q", get("0-7"))[0]
    return json.loads(get(f"8-{7 + size}"))


def tensor_bytes(urls: list[str], prefix: str = "") -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for url in urls:
        for name, meta in header(url).items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            start, stop = meta["data_offsets"]
            out[role(name)][meta["dtype"]] += stop - start
    return out


STAMP = re.compile(r" (\d\d-\d\d \d\d:\d\d:\d\d) ")


def vllm_lines(log: Path):
    """(minutes after the first engine metrics line, line) for vLLM's periodic metrics lines."""
    start = None
    for line in open(log, errors="ignore"):
        if "SpecDecoding metrics" not in line and "Avg prompt throughput" not in line:
            continue
        m = STAMP.search(line)
        t = datetime.strptime("2000-" + m.group(1), "%Y-%m-%d %H:%M:%S") if m else None  # leap year: 02-29 parses
        if start is None and t is not None and "Avg prompt throughput" in line:
            start = t
        minute = (t - start).total_seconds() / 60 if (t is not None and start is not None) else 0.0
        yield minute, line


def spec_summary(lines: list[str]) -> dict:
    accepted = drafted = 0
    positions: list[list[float]] = []
    for line in lines:
        m = re.search(r"Accepted: (\d+) tokens, Drafted: (\d+) tokens", line)
        accepted, drafted = accepted + int(m.group(1)), drafted + int(m.group(2))
        m = re.search(r"Per-position acceptance rate: ([\d., ]+)", line)
        if m:
            positions.append([float(x) for x in m.group(1).strip(", ").split(",") if x.strip()])
    if not drafted:
        return dict(engine=None)
    k = len(positions[0]) if positions else 3
    return dict(engine="vllm", lines=len(positions), length=round(1 + accepted / (drafted / k), 3),
                per_position=[round(sum(p[i] for p in positions) / len(positions), 3) for i in range(k)])


def acceptance(log: Path, until: float | None = None) -> dict:
    spec = [line for minute, line in vllm_lines(log)
            if "SpecDecoding metrics" in line and (until is None or minute < until)]
    if spec:
        return spec_summary(spec)
    sglang = [float(re.search(r"accept len: ([\d.]+)", line).group(1))
              for line in open(log, errors="ignore") if "accept len:" in line]
    if sglang:
        return dict(engine="sglang", lines=len(sglang), length=round(sum(sglang) / len(sglang), 3))
    return dict(engine=None)


def step_times(log: Path, min_running: int = 12) -> dict:
    engine: dict[str, tuple[float, int]] = {}
    spec: dict[str, tuple[int, int]] = {}
    for line in open(log, errors="ignore"):
        stamp = re.search(r" (\d\d-\d\d \d\d:\d\d:\d\d) ", line)
        if not stamp:
            continue
        if m := re.search(r"Avg prompt throughput: ([\d.]+) tokens/s.*Running: (\d+) reqs", line):
            engine[stamp.group(1)] = (float(m.group(1)), int(m.group(2)))
        elif m := re.search(r"Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+)", line):
            spec[stamp.group(1)] = (int(m.group(1)), len([x for x in m.group(2).split(",") if x.strip()]))
    clock = lambda s: datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S")
    stamps = sorted(set(engine) & set(spec))
    rows = []  # (step ms, prefill tokens per step)
    for before, now in zip(stamps, stamps[1:]):
        seconds = (clock(now) - clock(before)).total_seconds()
        prompt, running = engine[now]
        drafted, k = spec[now]
        if running < min_running or not drafted or seconds <= 0:
            continue
        steps = drafted / k / running
        rows.append((1000 * seconds / steps, prompt * seconds / steps))
    n = len(rows)
    if n < 2:
        return dict(intervals=n)
    mean_t, mean_p = sum(t for t, _ in rows) / n, sum(p for _, p in rows) / n
    slope = sum((p - mean_p) * (t - mean_t) for t, p in rows) / sum((p - mean_p) ** 2 for _, p in rows)
    fixed = mean_t - slope * mean_p
    residual = sum((t - fixed - slope * p) ** 2 for t, p in rows)
    r2 = 1 - residual / sum((t - mean_t) ** 2 for t, _ in rows)
    return dict(intervals=n, step_ms=round(mean_t, 1), prefill_tokens_per_step=round(mean_p),
                fixed_ms=round(fixed, 1), us_per_prefill_token=round(1000 * slope, 1), r2=round(r2, 2))


ENGINE_FIELDS = (("prompt", "Avg prompt throughput"), ("gen", "Avg generation throughput"),
                 ("kv", "GPU KV cache usage"), ("hit", "Prefix cache hit rate"), ("waiting", "Waiting"))


def timeline(log: Path, bucket: float = 5.0) -> list[dict]:
    rows: dict[int, dict] = {}
    for minute, line in vllm_lines(log):
        row = rows.setdefault(int(minute // bucket), dict(engine=[], spec=[]))
        if "SpecDecoding metrics" in line:
            row["spec"].append(line)
        else:
            row["engine"].append({k: float(re.search(label + r": ([\d.]+)", line).group(1)) for k, label in ENGINE_FIELDS})
    out = []
    for b in sorted(rows):
        engine = rows[b]["engine"]
        if not engine:
            continue
        mean = {k: round(sum(e[k] for e in engine) / len(engine), 1) for k, _ in ENGINE_FIELDS}
        spec = spec_summary(rows[b]["spec"]) if rows[b]["spec"] else {}
        out.append(dict(minutes=[b * bucket, (b + 1) * bucket], **mean, kv_max=max(e["kv"] for e in engine),
                        length=spec.get("length"), per_position=spec.get("per_position")))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bytes")
    b.add_argument("urls", nargs="+")
    b.add_argument("--prefix", default="")
    a = sub.add_parser("accept")
    a.add_argument("log", type=Path)
    s = sub.add_parser("steps")
    s.add_argument("log", type=Path)
    s.add_argument("--min-running", type=int, default=12)
    a.add_argument("--until", type=float, default=None, help="minutes after the first metrics line")
    t = sub.add_parser("timeline")
    t.add_argument("log", type=Path)
    t.add_argument("--bucket", type=float, default=5.0, help="minutes per row")
    args = parser.parse_args(argv)
    if args.command == "accept":
        print(json.dumps(acceptance(args.log, args.until)))
        return 0
    if args.command == "timeline":
        print(f"{'minutes':>9s} {'prompt':>7s} {'gen':>6s} {'KV%':>5s} {'KVmax':>5s} {'hit%':>5s} {'wait':>5s} "
              f"{'accept':>6s}  per position")
        for r in timeline(args.log, args.bucket):
            pos = " / ".join(f"{x:.3f}" for x in r["per_position"]) if r["per_position"] else ""
            length = f"{r['length']:.3f}" if r["length"] else ""
            print(f"{r['minutes'][0]:4.0f}-{r['minutes'][1]:<4.0f} {r['prompt']:7.0f} {r['gen']:6.0f} {r['kv']:5.1f} "
                  f"{r['kv_max']:5.1f} {r['hit']:5.1f} {r['waiting']:5.1f} {length:>6s}  {pos}")
        return 0
    if args.command == "steps":
        print(json.dumps(step_times(args.log, args.min_running)))
        return 0
    rows = tensor_bytes(args.urls, args.prefix)
    total = 0
    for label, dtypes in sorted(rows.items(), key=lambda kv: -sum(kv[1].values())):
        size = sum(dtypes.values())
        total += size
        print(f"{label:42s} {size / 1e9:7.3f} GB  " + ", ".join(f"{d} {v / 1e9:.3f}" for d, v in sorted(dtypes.items())))
    print(f"{'total':42s} {total / 1e9:7.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
