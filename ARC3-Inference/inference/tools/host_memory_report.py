"""Say what holds host RAM over a Kaggle run, from the watchdog's ``metrics/host_memory.jsonl``.

Each sample records MemAvailable, the cgroup's memory.stat split and resident memory per process
role (``read_memory_breakdown`` in the Lord launcher's watchdog). The report lists them at fixed
minutes and the growth of each role from ``--from-minute`` to the end, then projects each role's
growth rate onto ``--project-minutes`` of play. Every game is a thread of the one notebook-kernel
process, and the agent takes as many steps as the server's throughput allows, however many games
are queued, so the harness grows with play time rather than with the game count.

    python -m inference.tools.host_memory_report runs/kaggle/<user>/<slug>/vNN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 1024**3
STAT_FIELDS = ("anon", "file", "shmem", "unevictable")


def load_samples(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics" / "host_memory.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_at(samples: list[dict], minute: float) -> dict | None:
    """The first sample at or after ``minute`` from the first sample, or None past the end."""
    start = samples[0]["ts"]
    return next((sample for sample in samples if sample["ts"] - start >= minute * 60), None)


def row(sample: dict, groups: list[str]) -> dict[str, float | None]:
    stat = sample.get("cgroup_memory_stat") or {}
    values: dict[str, float | None] = {"MemAvailable": sample["host_available_bytes"] / GIB}
    for field in STAT_FIELDS:
        values[f"cg.{field}"] = stat[field] / GIB if field in stat else None
    process_groups = sample.get("process_groups") or {}
    for group in groups:
        values[group] = process_groups.get(group, {}).get("anon_bytes", 0) / GIB
    shm = sample.get("dev_shm_used_bytes")
    values["/dev/shm"] = shm / GIB if shm is not None else None
    return values


def report(samples: list[dict], minutes: list[float], from_minute: float, project_minutes: float) -> str:
    groups = sorted({group for sample in samples for group in sample.get("process_groups") or {}})
    if not groups:
        return "No per-process breakdown in these samples (the run predates read_memory_breakdown)."
    duration = (samples[-1]["ts"] - samples[0]["ts"]) / 60
    columns = [(f"{minute:g}m", sample_at(samples, minute)) for minute in minutes if minute < duration]
    columns.append((f"end {duration:.0f}m", samples[-1]))
    rows = {label: row(sample, groups) for label, sample in columns if sample}
    names = list(next(iter(rows.values())))
    lines = ["GiB (process groups: RssAnon)".ljust(22) + "".join(label.rjust(11) for label in rows)]
    for name in names:
        cells = (rows[label][name] for label in rows)
        lines.append(name.ljust(22) + "".join("—".rjust(11) if cell is None else f"{cell:11.2f}" for cell in cells))

    base = sample_at(samples, from_minute)
    if base is None:
        return "\n".join(lines)
    hours = (samples[-1]["ts"] - base["ts"]) / 3600
    start, end = row(base, groups), row(samples[-1], groups)
    lines.append(f"\nGrowth from {from_minute:g} min to the end ({hours:.2f} h), GiB: total, per hour, over {project_minutes:g} min")
    for name in names:
        if start[name] is not None and end[name] is not None:
            growth = end[name] - start[name]
            rate = growth / hours if hours else 0.0
            lines.append(f"  {name:20s} {growth:+7.2f} {rate:+7.2f} {rate * project_minutes / 60:+8.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minutes", type=float, nargs="+", default=[10, 30, 60, 90])
    parser.add_argument(
        "--from-minute", type=float, default=40,
        help="start of the growth window; keep it past the server start, which took 9-35 min",
    )
    parser.add_argument("--project-minutes", type=float, default=532, help="play time to project growth onto (competition rerun)")
    args = parser.parse_args()
    print(report(load_samples(args.run_dir), args.minutes, args.from_minute, args.project_minutes))


if __name__ == "__main__":
    main()
