"""Throughput report for one or more runs, read from SGLang's serve.log.

Franzen tuned serving by measuring it (prefix reuse, prefill share, decode
speed); this puts the same numbers side by side for our runs, so a serving
change is judged by what it did to the server and not only by the score.

Usage:
    python kaggle/serve_report.py <run_dir> [<run_dir> ...]
where each run_dir holds the run's serve.log (and optionally summary.txt).

Weighting: SGLang writes a "Decode batch" line once every 40 decode
iterations (`decode_log_interval`), and its `#running-req` is the batch at
that one iteration. Small batches step faster, so they write more lines, and
a plain mean over lines over-weights them. Each line's `gen throughput` is the
tokens generated since the previous line over the wall time since then, so
that time is recovered as 40 x #running-req x accept len / gen throughput
(the same step time the bench uses), and the decode rows below are weighted
by it. A window's time includes any prefill run inside it, and idle time
before decoding resumes. The per-line rows are kept as engine diagnostics.

Reported per run:
- harness tok/s: generated tokens / job wallclock from summary.txt (the
  number that actually buys levels). Judge serving changes by this first.
- decode tok/s (time-weighted): tokens over decode-window wall time, and the
  time-weighted median / P90 of the per-window speed.
- batch (time-weighted): mean requests running and the share of decode time
  with the batch full. Decode speed scales with N, so idle slots are lost
  throughput.
- KV full / queued (time-weighted): share of decode time with the KV pool at
  >= 95%, and with requests waiting.
- per decode log (engine diagnostics): the same as plain statistics over
  lines, and the median speed with the batch full, which isolates engine
  speed from how well the scheduler kept slots busy.
- spec accept: mean speculative tokens accepted per step (MTP).
- prefix reuse: cached / total prompt tokens over all requests.
- prefill share: summed time requests spent in initial prefill, divided by
  their summed time on the GPU (prefill + decode).
- queue: total and P99 time requests waited before being scheduled.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

DECODE = re.compile(
    r"Decode batch, #running-req: (\d+).*?full token usage: ([\d.]+).*?accept len: ([\d.]+)"
    r".*?gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
REQUEST = re.compile(
    r"ReqTimeStats\(.*?input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?\): "
    r"queue_duration=([\d.]+)ms, initial_prefill_elapsed=([\d.]+)ms.*?forward_duration=([\d.]+)ms"
)


def _pct(values: list[float], q: float) -> float:
    """The q-quantile (0..1) of `values` by nearest rank; 0.0 when empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


DECODE_LOG_INTERVAL = 40  # SGLang's default; the launcher does not change it


def window_seconds(running: int, accept: float, tps: float) -> float:
    """Wall time one decode line covers: the tokens it counted over its speed."""
    if tps <= 0:
        return 0.0
    return DECODE_LOG_INTERVAL * running * max(accept, 1.0) / tps


def _weighted_pct(pairs: list[tuple[float, float]], q: float) -> float:
    """The q-quantile of (value, weight) pairs by weight; 0.0 when no weight."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    seen = 0.0
    for value, weight in sorted(pairs):
        seen += weight
        if seen >= q * total:
            return value
    return max(v for v, _ in pairs)


def report(run_dir: Path) -> dict:
    """Parse one run's serve.log (and summary.txt) into the metrics above."""
    decode, requests = [], []
    for line in (run_dir / "serve.log").read_text(errors="replace").splitlines():
        if (m := DECODE.search(line)):
            decode.append((int(m[1]), float(m[2]), float(m[3]), float(m[4]), int(m[5])))
        elif (m := REQUEST.search(line)):
            requests.append(tuple(float(x) for x in m.groups()))
    # The first requests are SGLang's own warmup; keep only real prompts.
    requests = [r for r in requests if r[0] > 1000]
    out: dict = {"run": run_dir.name, "decode_logs": len(decode), "requests": len(requests)}
    if decode:
        speeds = [d[3] for d in decode]
        running = [d[0] for d in decode]
        top = max(running)
        # Time-weighted: each line stands for the wall time since the last one.
        weights = [window_seconds(d[0], d[2], d[3]) for d in decode]
        span = sum(weights)
        if span > 0:
            def share(flags):
                return sum(w for f, w in zip(flags, weights) if f) / span
            out["decode_tps"] = sum(d[3] * w for d, w in zip(decode, weights)) / span
            out["decode_median"] = _weighted_pct(list(zip(speeds, weights)), 0.5)
            out["decode_p90"] = _weighted_pct(list(zip(speeds, weights)), 0.9)
            out["batch_mean"] = sum(r * w for r, w in zip(running, weights)) / span
            out["batch_full"] = share(r == top for r in running)
            out["kv_full"] = share(d[1] >= 0.95 for d in decode)
            out["queued"] = share(d[4] > 0 for d in decode)
            out["decode_minutes"] = span / 60
        out["batch_max"] = top
        # Per line (one sample per 40 iterations): engine diagnostics.
        out["decode_median_logs"] = st.median(speeds)
        out["decode_p90_logs"] = _pct(speeds, 0.9)
        # Speed at a full batch isolates engine changes from scheduling luck.
        out["decode_full_batch"] = st.median(d[3] for d in decode if d[0] == top)
        out["batch_mean_logs"] = st.mean(running)
        out["batch_full_logs"] = sum(r == top for r in running) / len(running)
        out["accept_len"] = st.mean(d[2] for d in decode)
        out["kv_full_logs"] = sum(d[1] >= 0.95 for d in decode) / len(decode)
        out["queued_logs"] = sum(d[4] > 0 for d in decode) / len(decode)
    if requests:
        out["prefix_reuse"] = sum(r[1] for r in requests) / max(1.0, sum(r[0] for r in requests))
        out["prefill_share"] = sum(r[4] for r in requests) / max(1.0, sum(r[5] for r in requests))
        out["queue_total_min"] = sum(r[3] for r in requests) / 60000
        out["queue_p99_s"] = _pct([r[3] for r in requests], 0.99) / 1000
        out["output_mean"] = st.mean(r[2] for r in requests)
    summary = run_dir / "summary.txt"
    if summary.exists():
        text = summary.read_text()
        if (m := re.search(r"generated tokens/sec: ([\d.]+)", text)):
            out["harness_tps"] = float(m[1])
        if (m := re.search(r"mean score:\s+([\d.]+)", text)):
            out["mean_score"] = float(m[1])
    return out


ROWS = [
    ("harness tok/s (wallclock)", "harness_tps", "{:.0f}"),
    ("mean score", "mean_score", "{:.2f}"),
    ("-- time-weighted", None, ""),
    ("decode tok/s", "decode_tps", "{:.0f}"),
    ("decode tok/s median", "decode_median", "{:.0f}"),
    ("decode tok/s P90", "decode_p90", "{:.0f}"),
    ("batch mean / max", ("batch_mean", "batch_max"), "{:.2f} / {}"),
    ("batch full (share)", "batch_full", "{:.0%}"),
    ("KV >= 95% full (share)", "kv_full", "{:.1%}"),
    ("requests queued (share)", "queued", "{:.0%}"),
    ("decode window time (min)", "decode_minutes", "{:.0f}"),
    ("-- per decode log (diagnostic)", None, ""),
    ("decode tok/s median", "decode_median_logs", "{:.0f}"),
    ("decode tok/s P90", "decode_p90_logs", "{:.0f}"),
    ("decode tok/s at full batch", "decode_full_batch", "{:.0f}"),
    ("batch mean", "batch_mean_logs", "{:.2f}"),
    ("batch full (share)", "batch_full_logs", "{:.0%}"),
    ("KV >= 95% full (share)", "kv_full_logs", "{:.1%}"),
    ("decode logs", "decode_logs", "{}"),
    ("-- requests", None, ""),
    ("spec accept len", "accept_len", "{:.2f}"),
    ("prefix reuse", "prefix_reuse", "{:.1%}"),
    ("prefill share of GPU time", "prefill_share", "{:.1%}"),
    ("queue wait total (min)", "queue_total_min", "{:.1f}"),
    ("queue wait P99 (s)", "queue_p99_s", "{:.1f}"),
    ("mean output tokens/request", "output_mean", "{:.0f}"),
    ("requests", "requests", "{}"),
]


def main() -> None:
    runs = [report(Path(arg)) for arg in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)
    width = max(len(r["run"]) for r in runs) + 2
    print(" " * 28 + "".join(r["run"].rjust(width) for r in runs))
    for label, key, fmt in ROWS:
        if key is None:
            print(label)
            continue
        cells = []
        for r in runs:
            keys = key if isinstance(key, tuple) else (key,)
            cells.append(fmt.format(*(r[k] for k in keys)) if all(k in r for k in keys) else "-")
        print(label.ljust(28) + "".join(c.rjust(width) for c in cells))


if __name__ == "__main__":
    main()
