"""Per-sample report of a train_rft run: where each step's time went, checked against the pass criteria.

Reads train/train_log.jsonl (or results/train_log.jsonl) and results/gpu.csv (nvidia-smi every second)
from a pulled run folder. Criteria (2026-10-20 rehearsal; judgment calls, see Logs.md):
- main-thread major page faults < 1,000 per sample (after the first): the PLE table stays in the page cache;
- cgroup anon + PLE table <= limit: the arena leaves the table its page cache;
- GPU busy (utilization > 0) >= 80% of each sample's seconds;
- "parts" samples: time <= 1.3 x the cost model c (prefix + 3.5 suffix), c = 1.65e-4 s (H200, 10-20);
- no skipped samples, at most one OOM retry.

With --project (check_rft_data's per-sample JSON lines), the dataset's average throughput: each mode's step time
fitted on the run's warm samples as a x prefix + b x tokens after the prefix, applied to every train sample in the
mode the arena would pick for it.

    python -m inference.rl.train_report RUN_DIR [--ple-gb 95.5] [--project check.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

COST_PER_UNIT_S = 16.1 / (0.8 * 64893 + 3.5 * 0.2 * 64893)  # H200 64K step, 80% prefix, prefix stop-grad


def gpu_samples(path: Path) -> list[tuple[float, float]]:
    """(unix time, utilization %) per nvidia-smi row."""
    out = []
    if not path.exists():
        return out
    for row in csv.reader(open(path)):
        if len(row) < 2:
            continue
        try:
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            out.append((stamp, float(row[1].strip().rstrip(" %"))))
        except ValueError:
            continue
    return out


def fit(rows: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """Least squares time = a x prefix + b x saved (no intercept) over (prefix, saved, time) rows."""
    if len(rows) < 2:
        return None
    spp = sum(p * p for p, _, _ in rows)
    sss = sum(s * s for _, s, _ in rows)
    sps = sum(p * s for p, s, _ in rows)
    spt = sum(p * t for p, _, t in rows)
    sst = sum(s * t for _, s, t in rows)
    det = spp * sss - sps * sps
    if abs(det) < 1e-9:
        return None
    return (spt * sss - sst * sps) / det, (sst * spp - spt * sps) / det


def project(args, warm: list[dict], arena: dict) -> None:
    rows = [json.loads(line) for line in open(args.project)]
    train = [r for r in rows if r.get("split", "train") == "train" and r["tokens"] <= args.max_tokens]
    per_mode = {}
    for mode in ("parts", "layer"):
        data = [(s["tokens"] - s["saved_tokens"], s["saved_tokens"], s["forward_s"] + s["backward_s"])
                for s in warm if s.get("recompute_mode") == mode]
        coef = fit(data)
        if coef is None or min(coef) < 0:  # too few samples: the cost model scaled by the measured ratio
            ratio = statistics.median([t / (COST_PER_UNIT_S * (p + 3.5 * v)) for p, v, t in data]) if data else None
            coef = (COST_PER_UNIT_S * ratio, 3.5 * COST_PER_UNIT_S * ratio) if ratio else None
        per_mode[mode] = coef
        print(f"{mode}: {len(data)} warm samples, time = {coef[0] * 1e6:.1f} us x prefix + {coef[1] * 1e6:.1f} us x saved"
              if coef else f"{mode}: no samples")
    bytes_per_token = {}
    for mode in ("parts", "layer"):
        sizes = [((s.get("d2h_gb") or 0) + (s.get("overflow_gb") or 0)) * 2**30 / s["saved_tokens"]
                 for s in warm if s.get("recompute_mode") == mode and s.get("saved_tokens")]
        bytes_per_token[mode] = statistics.median(sizes) if sizes else {"parts": 2.4e6, "layer": 1.2e6}[mode]
    cap = (arena.get("cap_gb") or 0) * 2**30
    total_time = total_tokens = 0.0
    modes = {"parts": 0, "layer": 0}
    for r in train:
        saved = r["tokens"] - r["prefix"]
        mode = "parts" if saved * bytes_per_token["parts"] <= cap else "layer"
        coef = per_mode.get(mode) or per_mode.get("parts") or per_mode.get("layer")
        if coef is None:
            print("no fitted mode: cannot project")
            return
        modes[mode] += 1
        total_time += coef[0] * r["prefix"] + coef[1] * saved
        total_tokens += r["tokens"]
    print(f"bytes per saved token: parts {bytes_per_token['parts'] / 1e6:.2f} MB, layer {bytes_per_token['layer'] / 1e6:.2f} MB;"
          f" arena {cap / 2**30:.0f} GiB -> {modes['parts']} samples parts, {modes['layer']} layer")
    print(f"projected dataset average: {total_tokens / total_time:.0f} tok/s, {total_time / 3600:.2f} h per epoch of "
          f"{len(train)} samples ({total_tokens / 1e6:.1f}M tokens), steps only")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--ple-gb", type=float, default=95.5)
    parser.add_argument("--project", type=Path, help="check_rft_data --out file: project the dataset's average")
    parser.add_argument("--max-tokens", type=int, default=131072)
    args = parser.parse_args(argv)
    log = args.run / "train" / "train_log.jsonl"
    if not log.exists():
        log = args.run / "results" / "train_log.jsonl"
    events = [json.loads(line) for line in open(log)]
    last_load = max(i for i, e in enumerate(events) if e["event"] == "loaded")
    events = events[last_load:]
    gpu = gpu_samples(args.run / "results" / "gpu.csv")
    arena = next((e for e in events if e["event"] == "arena"), {})
    print(f"arena cap {arena.get('cap_gb')} GiB (limit {arena.get('limit_gb')}, PLE {arena.get('ple_gb')}, "
          f"process {arena.get('process_gb')}), reserved in {arena.get('reserve_s')} s")
    for e in events:
        if e["event"] == "val":
            print(f"val epoch {e.get('epoch')}: loss {e['loss']} over {e['tokens']} tokens, {e['samples']} samples")
    samples = [e for e in events if e["event"] == "sample"]
    print(f"{'tokens':>7} {'saved':>6} {'mode':>5} {'fwd':>6} {'bwd':>6} {'tok/s':>6} {'model':>5} {'flt f/b':>11} "
          f"{'ple':>5} {'pre_ple':>7} {'gpu%':>5} {'ovfl':>5} {'arena':>6} {'anon':>6} {'file':>6} {'graphs':>6}  error")
    ratios, busy_all, faults_after_first, failures = [], [], [], []
    limit = arena.get("limit_gb")
    memory_ok = True
    for k, s in enumerate(samples):
        if s.get("error"):
            failures.append(s)
            print(f"{s.get('tokens'):>7} {s.get('saved_tokens')!s:>6}  ERROR {s['error'][:100]}")
            continue
        tokens, saved = s["tokens"], s.get("saved_tokens") or s["tokens"]
        prefix = tokens - saved
        took = s["forward_s"] + s["backward_s"]
        ratio = took / (COST_PER_UNIT_S * (prefix + 3.5 * saved))
        window = [u for t, u in gpu if s.get("wall_start", 0) <= t <= s.get("wall_end", 0)]
        busy = 100 * sum(u > 0 for u in window) / len(window) if window else float("nan")
        faults = s.get("major_faults") or [0, 0]
        cgroup = s.get("cgroup") or {}
        if limit and cgroup.get("anon_gb") is not None and cgroup["anon_gb"] + args.ple_gb > limit:
            memory_ok = False
        if k:
            faults_after_first.append(sum(faults))
        if s.get("recompute_mode") == "parts" and k:
            ratios.append(ratio)
        if window and k:
            busy_all.append(busy)
        pre = (s.get("prefetch") or {}).get("ple_s")
        print(f"{tokens:>7} {saved:>6} {s.get('recompute_mode', '?'):>5} {s['forward_s']:>6.1f} {s['backward_s']:>6.1f} "
              f"{s['tokens_per_s']:>6.0f} {ratio:>5.2f} {faults[0]:>5}/{faults[-1]:<5} {s.get('ple_lookup_s', 0):>5.2f} "
              f"{pre if pre is not None else '-':>7} {busy:>5.0f} {s.get('overflow_gb') or 0:>5.1f} "
              f"{s.get('arena_used_gb') or 0:>6.1f} {cgroup.get('anon_gb', '-'):>6} {cgroup.get('file_gb', '-'):>6} "
              f"{s.get('graphs_compiled', '-'):>6}  {'; '.join(s.get('errors') or [])[:80]}")
    retries = sum(1 for s in samples if s.get("errors"))
    checks = [
        ("major faults < 1,000 per sample after the first", bool(faults_after_first) and max(faults_after_first) < 1000,
         f"max {max(faults_after_first) if faults_after_first else '-'}"),
        ("anon + PLE <= limit", memory_ok, f"limit {limit}"),
        ("GPU busy >= 80% per sample after the first", bool(busy_all) and min(busy_all) >= 80,
         f"min {min(busy_all):.0f}%" if busy_all else "no gpu.csv"),
        ("parts samples <= 1.3x cost model", bool(ratios) and max(ratios) <= 1.3,
         f"median {statistics.median(ratios):.2f}, max {max(ratios):.2f}" if ratios else "no parts samples"),
        ("no skipped samples, <= 1 OOM retry",
         not failures and retries <= 1 and not any(e["event"] == "skip" for e in events),
         f"{len(failures)} failed, {retries} retried"),
    ]
    total_tokens = sum(s["tokens"] for s in samples if not s.get("error"))
    total_time = sum(s["forward_s"] + s["backward_s"] for s in samples if not s.get("error"))
    print(f"\n{len(samples)} samples, {total_tokens:,} tokens in {total_time:.0f} s of steps: "
          f"{total_tokens / max(total_time, 1):.0f} tok/s overall")
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({detail})")
    if args.project:
        project(args, [s for s in samples[1:] if not s.get("error")], arena)
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
