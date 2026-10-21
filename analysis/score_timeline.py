"""Score, levels, tokens and actions of saved runs as a function of wall-clock time.

For each run directory ``runs/kaggle/<kernel>/v<N>/`` this rebuilds every game's
official score at every moment of the run, on the wall-clock axis (seconds since the
benchmark started, since the notebook started and since the model server answered
``/health``), and the run's mean over all games. Runs can then be compared at matched
wall-clock minutes (``--at 25,60,121``) and overlaid in one HTML chart.

Sources, all read in place (``.xz`` streamed with lzma, or plain):

- ``benchmark.json``: per game ``started_at``, ``history`` (one record per action with
  ``wallclock_seconds`` since the game started and ``generated_tokens``),
  ``actions_per_level``, ``levels_completed``, ``base_actions_per_level`` and the
  framework's ``final_score``. The framework increments ``actions_per_level[level]``
  for every action, so level *i* was cleared by action number
  ``sum(actions_per_level[:i+1])``. ``--check-events`` confirms this per level against
  ``artifacts/<game>_p0_events.jsonl``.
- ``<kernel>.log``: the notebook log (Kaggle's JSON list of ``{time, data}``, ``time`` in
  seconds since the notebook started). ``READY after Ns`` gives the server-ready time and
  the ``[finished] <game>`` lines tie the benchmark clock to the notebook clock.
- ``summary.txt``: the final per-game score, levels, actions and tokens. The rebuilt
  final values are checked against it for every game; any mismatch is reported.

Scoring is the framework's ``GameRun._compute_final_score`` (taaf/game.py, which mirrors
``arc_agi.scorecard``): level score ``min(115, (baseline/actions)^2 * 100)`` for a cleared
level, 0 otherwise, weighted by level number; the game score is the weighted mean, capped
at ``100 * (weight of scored levels) / (total weight)``.

Standard library only. From the repo root::

    nice -n 15 python3 analysis/score_timeline.py \\
        --runs fz-arms/v4 fz-ledger-on-trim/v1 --at 25,60,121 \\
        --out research/archive/data/score-timeline --html research/archive/data/score-timeline/full-runs.html
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO / "runs" / "kaggle"

# --- official score -----------------------------------------------------------


def official_score(
    number_of_levels: int,
    base_actions_per_level: list[int] | None,
    levels_completed: int,
    actions_per_level: list[int],
) -> float:
    """``taaf.game.GameRun._compute_final_score``, line for line."""
    if base_actions_per_level is None or number_of_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(number_of_levels):
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < levels_completed
        actions = actions_per_level[level_idx] if level_idx < len(actions_per_level) else 0
        baseline = base_actions_per_level[level_idx]
        if completed and actions > 0:
            level_score = min(115.0, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    if total_weights == 0:
        return 0.0
    score = total_score / total_weights
    max_score = max_weights / total_weights * 100
    return min(score, max_score)


# --- reading a run ---------------------------------------------------------------


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def find_file(run_dir: Path, name: str) -> Path | None:
    for candidate in (run_dir / name, run_dir / f"{name}.xz"):
        if candidate.exists():
            return candidate
    return None


@dataclass
class Game:
    key: str  # game_id, or game_id#p<k> when a run has several passes
    game_id: str
    number_of_levels: int
    baselines: list[int] | None
    state: str
    offset_s: float  # game start minus benchmark start
    action_t: list[float]  # benchmark seconds of each action
    action_tokens: list[int]
    final_t: float  # benchmark seconds when finish_game ran
    final_tokens: int  # no-move tokens of the last turn
    actions_per_level: list[int]
    levels_completed: int
    framework_score: float | None
    level_actions: list[int] = field(default_factory=list)  # action number that cleared each level

    def __post_init__(self) -> None:
        total = 0
        self.level_actions = []
        for i in range(self.levels_completed):
            total += self.actions_per_level[i]
            self.level_actions.append(total)
        self._token_prefix = [0]
        for tok in self.action_tokens:
            self._token_prefix.append(self._token_prefix[-1] + tok)

    def at(self, t: float) -> dict:
        """State after every action taken at or before benchmark second ``t``.

        Only cleared levels score, and a cleared level's action count is fixed when it is
        cleared, so the score at ``t`` uses the final ``actions_per_level`` for the levels
        cleared by then and 0 for the rest.
        """
        n = _count_le(self.action_t, t)
        levels = sum(1 for a in self.level_actions if a <= n)
        per_level = [self.actions_per_level[i] if i < levels else 0 for i in range(self.number_of_levels)]
        score = official_score(self.number_of_levels, self.baselines, levels, per_level)
        if t < self.offset_s:
            state = "waiting"
        elif t < self.final_t or self.state == "playing":
            state = "playing"
        else:
            state = self.state
        return {"levels": levels, "score": score, "actions": n, "tokens": self._token_prefix[n], "state": state}

    def final(self) -> dict:
        return {
            "levels": self.levels_completed,
            "score": official_score(self.number_of_levels, self.baselines, self.levels_completed,
                                    self.actions_per_level),
            "actions": len(self.action_t),
            "tokens": sum(self.action_tokens),
            "final_tokens": self.final_tokens,
        }


def _count_le(sorted_times: list[float], t: float) -> int:
    lo, hi = 0, len(sorted_times)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_times[mid] <= t:
            lo = mid + 1
        else:
            hi = mid
    return lo


@dataclass
class Run:
    name: str
    path: Path
    start_time: datetime
    end_s: float  # benchmark end, seconds after its start
    games: list[Game]
    bench_nb_s: float | None = None  # benchmark start on the notebook clock
    ready_nb_s: float | None = None  # server READY on the notebook clock
    summary: dict | None = None

    @property
    def game_ids(self) -> list[str]:
        return sorted({g.game_id for g in self.games})

    def axis_shift(self, axis: str) -> float | None:
        """Seconds to add to benchmark time to get time on ``axis``."""
        if axis == "bench":
            return 0.0
        if axis == "notebook":
            return self.bench_nb_s
        if axis == "ready":
            if self.bench_nb_s is None or self.ready_nb_s is None:
                return None
            return self.bench_nb_s - self.ready_nb_s
        raise ValueError(axis)

    def at(self, t: float) -> dict:
        """Run totals at benchmark second ``t``; the mean is over game ids (passes averaged)."""
        per_game = {g.key: g.at(t) for g in self.games}
        by_id: dict[str, list[float]] = {}
        for g in self.games:
            by_id.setdefault(g.game_id, []).append(per_game[g.key]["score"])
        mean = statistics.fmean(statistics.fmean(v) for v in by_id.values()) if by_id else 0.0
        return {
            "mean_score": mean,
            "levels": sum(s["levels"] for s in per_game.values()),
            "tokens": sum(s["tokens"] for s in per_game.values()),
            "actions": sum(s["actions"] for s in per_game.values()),
            "games_finished": sum(1 for g in self.games if g.state != "playing" and t >= g.final_t),
            "per_game": per_game,
        }


def run_name(run_dir: Path) -> str:
    return f"{run_dir.parent.name}/{run_dir.name}"


def resolve_run(arg: str) -> Path:
    p = Path(arg)
    for candidate in (p, RUNS_ROOT / arg, REPO / arg):
        if (candidate / "summary.txt").exists() or find_file(candidate, "benchmark.json"):
            return candidate.resolve()
    raise SystemExit(f"no run at {arg} (looked under {RUNS_ROOT})")


def load_run(run_dir: Path) -> Run:
    bench_path = find_file(run_dir, "benchmark.json")
    if bench_path is None:
        raise FileNotFoundError(f"{run_dir}: no benchmark.json(.xz)")
    with _open_text(bench_path) as f:
        bench = json.load(f)
    runs = bench.get("game_runs") or []
    if not runs:
        raise ValueError(f"{bench_path}: no game_runs")
    started = [datetime.fromisoformat(r["started_at"]) for r in runs if r.get("started_at")]
    start = datetime.fromisoformat(bench["start_time"]) if bench.get("start_time") else min(started)
    n_passes = int(bench.get("n_passes") or 1)
    per_pass = len(runs) // n_passes if len(runs) % n_passes == 0 else len(runs)
    games = []
    for i, r in enumerate(runs):
        offset = (datetime.fromisoformat(r["started_at"]) - start).total_seconds() if r.get("started_at") else 0.0
        hist = r.get("history") or []
        times = [offset + float(h["wallclock_seconds"]) for h in hist]
        # Actions are appended in order, so their times are already sorted; guard anyway.
        if any(b < a for a, b in zip(times, times[1:])):
            times = list(_running_max(times))
        final_t = offset + float(r.get("final_wallclock_seconds") or 0.0)
        if final_t <= offset and times:
            final_t = times[-1]
        key = r["game_id"] if n_passes == 1 else f"{r['game_id']}#p{i // per_pass}"
        games.append(Game(
            key=key, game_id=r["game_id"], number_of_levels=int(r["number_of_levels"]),
            baselines=r.get("base_actions_per_level"), state=str(r.get("state")), offset_s=offset,
            action_t=times, action_tokens=[int(h.get("generated_tokens") or 0) for h in hist],
            final_t=final_t, final_tokens=int(r.get("final_generated_tokens") or 0),
            actions_per_level=[int(a) for a in (r.get("actions_per_level") or [])],
            levels_completed=int(r.get("levels_completed") or 0), framework_score=r.get("final_score"),
        ))
    if bench.get("end_time"):
        end_s = (datetime.fromisoformat(bench["end_time"]) - start).total_seconds()
    else:
        end_s = max([g.final_t for g in games] + [t for g in games for t in g.action_t[-1:]] + [0.0])
    run = Run(name=run_name(run_dir), path=run_dir, start_time=start, end_s=end_s, games=games)
    run.bench_nb_s, run.ready_nb_s = notebook_clock(run_dir, run)
    summary_path = run_dir / "summary.txt"
    if summary_path.exists():
        run.summary = parse_summary(summary_path.read_text(encoding="utf-8", errors="replace"))
    return run


def _running_max(values: list[float]):
    best = -math.inf
    for v in values:
        best = max(best, v)
        yield best


SUMMARY_GAME = re.compile(
    r"^\s+(?P<gid>[\w-]+): score=(?P<score>[-\d.]+), levels=(?P<levels>[\d.]+)/(?P<n>\d+), "
    r"actions=(?P<actions>\d+), tokens=(?P<tokens>\d+)"
)


def parse_summary(text: str) -> dict:
    out: dict = {"games": {}}
    for line in text.splitlines():
        m = SUMMARY_GAME.match(line)
        if m:
            out["games"][m["gid"]] = {
                "score": float(m["score"]), "levels": float(m["levels"]),
                "actions": int(m["actions"]), "tokens": int(m["tokens"]),
            }
        elif line.startswith("mean score:"):
            out["mean_score"] = float(line.split(":", 1)[1])
        elif line.startswith("duration:"):
            out["duration"] = line.split(":", 1)[1].strip()
    return out


# --- notebook clock -------------------------------------------------------------

READY = re.compile(r"READY after (\d+)s")
FINISHED = re.compile(r"\[finished\] (\S+) state=")


def _log_entries(run_dir: Path):
    """(notebook seconds, text) from Kaggle's JSON log, or nothing if absent/unreadable."""
    logs = sorted(p for p in run_dir.iterdir()
                  if p.name.endswith(".log") or p.name.endswith(".log.xz"))
    logs = [p for p in logs if not p.name.startswith("serve.")]
    for path in logs:
        try:
            with _open_text(path) as f:
                data = json.load(f)
        except (OSError, ValueError, EOFError, lzma.LZMAError):
            continue
        if isinstance(data, list):
            for e in data:
                if isinstance(e, dict) and "time" in e:
                    yield float(e["time"]), str(e.get("data", ""))
            return


def notebook_clock(run_dir: Path, run: Run) -> tuple[float | None, float | None]:
    """Benchmark start and server READY on the notebook clock (seconds since notebook start).

    The benchmark start is fitted from each ``[finished] <game>`` line: that line prints
    when ``finish_game`` runs, at benchmark second ``final_t``. The median of
    ``log_time - final_t`` is robust to print buffering; ``Starting benchmark...`` is the
    fallback.
    """
    finals = {g.game_id: g.final_t for g in run.games}
    ready = None
    starting = None
    deltas = []
    for t, text in _log_entries(run_dir):
        for line in text.splitlines():
            if ready is None:
                m = READY.search(line)
                if m:
                    ready = t
            if starting is None and line.startswith("Starting benchmark"):
                starting = t
            m = FINISHED.search(line)
            if m and m.group(1) in finals:
                deltas.append(t - finals[m.group(1)])
    bench = statistics.median(deltas) if deltas else starting
    if ready is None:
        # Notebooks built after 2026-10-08 log score_timeline.jsonl, whose rows carry
        # both the notebook clock and the time since the server was ready.
        live = find_file(run_dir, "score_timeline.jsonl")
        if live is not None:
            with _open_text(live) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("t_notebook_s") is not None and row.get("t_ready_s") is not None:
                        ready = row["t_notebook_s"] - row["t_ready_s"]
                        break
    return bench, ready


# --- level-boundary cross-check against events.jsonl ------------------------------


def check_events(run: Run) -> list[str]:
    """Compare the benchmark.json level boundaries with the per-action events log."""
    problems = []
    for g in run.games:
        if "#p" in g.key and not g.key.endswith("#p0"):
            continue
        path = find_file(run.path / "artifacts", f"{g.game_id}_p0_events.jsonl")
        if path is None:
            continue  # not every run saved events; nothing to compare
        boundaries, prev = [], 0
        with _open_text(path) as f:
            for line in f:
                if '"type":"action"' not in line and '"type": "action"' not in line:
                    continue
                d = json.loads(line)
                if d.get("type") != "action":
                    continue
                s = int(d.get("score") or 0)
                if s > prev:
                    boundaries.append(int(d["action_num"]))
                prev = s
        if boundaries != g.level_actions:
            problems.append(f"{run.name} {g.game_id}: events clear levels at {boundaries}, "
                            f"benchmark.json at {g.level_actions}")
    return problems


# --- cross-check against summary.txt ----------------------------------------------


def cross_check(run: Run) -> list[str]:
    """Final rebuilt values vs summary.txt and the framework's final_score, per game."""
    problems = []
    by_id: dict[str, list[dict]] = {}
    for g in run.games:
        fin = g.final()
        if g.framework_score is not None and abs(fin["score"] - float(g.framework_score)) > 1e-9:
            problems.append(f"{run.name} {g.key}: rebuilt {fin['score']:.6f} != final_score {g.framework_score}")
        by_id.setdefault(g.game_id, []).append(fin)
    if run.summary is None:
        return problems + [f"{run.name}: no summary.txt to check against"]
    sgames = run.summary["games"]
    for gid, fins in by_id.items():
        s = sgames.get(gid)
        if s is None:
            problems.append(f"{run.name} {gid}: missing from summary.txt")
            continue
        mine = {k: statistics.fmean(f[k] for f in fins) for k in ("score", "levels", "actions", "tokens", "final_tokens")}
        if abs(mine["score"] - s["score"]) > 0.005 + 1e-9:
            problems.append(f"{run.name} {gid}: score {mine['score']:.4f} vs summary {s['score']:.2f}")
        if abs(mine["levels"] - s["levels"]) > 0.05:
            problems.append(f"{run.name} {gid}: levels {mine['levels']} vs summary {s['levels']}")
        if round(mine["actions"]) != s["actions"]:
            problems.append(f"{run.name} {gid}: actions {mine['actions']} vs summary {s['actions']}")
        # Current summaries count action tokens only; older framework builds (the 2026-09
        # duck-* runs) also added the no-move tokens of the last turn. Either is a match.
        if s["tokens"] not in (round(mine["tokens"]), round(mine["tokens"] + mine["final_tokens"])):
            problems.append(f"{run.name} {gid}: tokens {mine['tokens']} vs summary {s['tokens']}")
    for gid in sgames:
        if gid not in by_id:
            problems.append(f"{run.name} {gid}: in summary.txt but not in benchmark.json")
    if "mean_score" in run.summary:
        final_mean = run.at(math.inf)["mean_score"]
        if abs(final_mean - run.summary["mean_score"]) > 0.005 + 1e-9:
            problems.append(f"{run.name}: mean {final_mean:.4f} vs summary {run.summary['mean_score']:.2f}")
    return problems


# --- outputs ----------------------------------------------------------------------


def grid(run: Run, step_s: float) -> list[float]:
    n = int(math.ceil(run.end_s / step_s))
    return [i * step_s for i in range(n + 1)]


def _fmt_min(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds / 60:.2f}"


def slug(run: Run) -> str:
    return run.name.replace("/", "_")


def write_csvs(run: Run, out_dir: Path, step_s: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nb, rd = run.axis_shift("notebook"), run.axis_shift("ready")
    with open(out_dir / f"{slug(run)}_games.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "t_bench_min", "t_notebook_min", "t_ready_min", "game", "levels", "n_levels",
                    "score", "tokens", "actions", "state"])
        for t in grid(run, step_s):
            for g in run.games:
                s = g.at(t)
                w.writerow([run.name, _fmt_min(t), _fmt_min(None if nb is None else t + nb),
                            _fmt_min(None if rd is None else t + rd), g.key, s["levels"], g.number_of_levels,
                            f"{s['score']:.4f}", s["tokens"], s["actions"], s["state"]])
    with open(out_dir / f"{slug(run)}_mean.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "t_bench_min", "t_notebook_min", "t_ready_min", "mean_score", "levels", "tokens",
                    "actions", "games_finished", "n_games"])
        for t in grid(run, step_s):
            s = run.at(t)
            w.writerow([run.name, _fmt_min(t), _fmt_min(None if nb is None else t + nb),
                        _fmt_min(None if rd is None else t + rd), f"{s['mean_score']:.4f}", s["levels"],
                        s["tokens"], s["actions"], s["games_finished"], len(run.game_ids)])
    with open(out_dir / f"{slug(run)}_levels.csv", "w", newline="") as f:
        # One row per cleared level, at the exact second of the clearing action.
        w = csv.writer(f)
        w.writerow(["run", "game", "level", "t_bench_min", "t_notebook_min", "action_num", "level_actions",
                    "baseline", "level_score", "game_score_after", "run_mean_after"])
        rows = []
        for g in run.games:
            for i, a in enumerate(g.level_actions):
                t = g.action_t[a - 1]
                base = g.baselines[i] if g.baselines else None
                acts = g.actions_per_level[i]
                lvl = min(115.0, (base / acts) ** 2 * 100) if base and acts else 0.0
                rows.append((t, g, i, a, base, acts, lvl))
        rows.sort(key=lambda r: r[0])
        for t, g, i, a, base, acts, lvl in rows:
            w.writerow([run.name, g.key, i + 1, f"{t / 60:.3f}", "" if nb is None else f"{(t + nb) / 60:.3f}",
                        a, acts, base, f"{lvl:.2f}", f"{g.at(t)['score']:.4f}", f"{run.at(t)['mean_score']:.4f}"])


def at_minutes(runs: list[Run], minutes: list[float], axis: str) -> list[dict]:
    rows = []
    for run in runs:
        shift = run.axis_shift(axis)
        for m in minutes:
            t = None if shift is None else m * 60 - shift
            val = None if t is None else run.at(t)
            rows.append({"run": run.name, "axis": axis, "minute": m,
                         "mean_score": None if val is None else val["mean_score"],
                         "levels": None if val is None else val["levels"],
                         "tokens": None if val is None else val["tokens"],
                         "actions": None if val is None else val["actions"],
                         "past_end": t is not None and t > run.end_s})
    return rows


def print_at(runs: list[Run], minutes: list[float], axis: str) -> str:
    rows = at_minutes(runs, minutes, axis)
    width = max(len(r.name) for r in runs)
    head = f"{'run':<{width}}  " + "  ".join(f"{m:>7g}m" for m in minutes) + "    final  summary"
    lines = [f"mean score at wall-clock minute (axis: {axis})", head]
    for run in runs:
        cells = []
        for r in rows:
            if r["run"] != run.name:
                continue
            v = r["mean_score"]
            cells.append("     n/a" if v is None else f"{v:7.2f}" + ("*" if r["past_end"] else " "))
        final = run.at(math.inf)["mean_score"]
        summ = run.summary.get("mean_score") if run.summary else None
        lines.append(f"{run.name:<{width}}  " + "  ".join(cells) + f"  {final:7.2f}  "
                     + ("    -" if summ is None else f"{summ:7.2f}"))
    if any(r["past_end"] for r in rows):
        lines.append("* past the run's end: its final score")
    return "\n".join(lines)


# --- HTML overlay -----------------------------------------------------------------


def html_payload(runs: list[Run], step_s: float) -> dict:
    out = []
    for run in runs:
        ts = grid(run, step_s)
        series = {"mean_score": [], "levels": [], "tokens": [], "actions": [], "games_finished": []}
        games: dict[str, list] = {g.key: [] for g in run.games}
        for t in ts:
            s = run.at(t)
            for k in series:
                v = s[k]
                series[k].append(round(v, 4) if isinstance(v, float) else v)
            for key, st in s["per_game"].items():
                games[key].append(round(st["score"], 2))
        out.append({
            "name": run.name, "n_games": len(run.game_ids), "step_min": step_s / 60, "end_min": run.end_s / 60,
            "shift": {a: (None if run.axis_shift(a) is None else run.axis_shift(a) / 60)
                      for a in ("bench", "notebook", "ready")},
            "summary_mean": run.summary.get("mean_score") if run.summary else None,
            "series": series, "games": games,
        })
    return {"runs": out, "generated": datetime.now().isoformat(timespec="seconds")}


def write_html(runs: list[Run], path: Path, step_s: float, title: str, at: list[float]) -> None:
    payload = json.dumps(html_payload(runs, step_s), separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__TITLE__", escape(title)).replace("__AT__", json.dumps(at)).replace(
        "__DATA__", payload.replace("</", "<\\/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


HTML_TEMPLATE = (Path(__file__).resolve().parent / "score_timeline.html").read_text(encoding="utf-8") \
    if (Path(__file__).resolve().parent / "score_timeline.html").exists() else "__DATA__"


# --- CLI ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs, as <kernel>/v<N> under runs/kaggle or a path")
    ap.add_argument("--at", default="", help="comma-separated minutes, e.g. 25,60,121: print each run's mean")
    ap.add_argument("--axis", choices=("bench", "notebook", "ready"), default="bench",
                    help="clock for --at: since benchmark start (default), notebook start or server ready")
    ap.add_argument("--step", type=float, default=60.0, help="grid step in seconds for the CSVs and chart")
    ap.add_argument("--out", type=Path, help="directory for the per-run CSVs and at_minutes.csv")
    ap.add_argument("--html", type=Path, help="write the overlay chart here")
    ap.add_argument("--title", default="Score over wall-clock")
    ap.add_argument("--check-events", action="store_true",
                    help="also check level boundaries against artifacts/*_events.jsonl")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any mismatch")
    args = ap.parse_args(argv)

    runs = [load_run(resolve_run(r)) for r in args.runs]
    problems: list[str] = []
    for run in runs:
        problems += cross_check(run)
        if args.check_events:
            problems += check_events(run)
    sets = {tuple(r.game_ids) for r in runs}
    if len(sets) > 1:
        print("note: the runs play different game sets, so their means are not directly comparable",
              file=sys.stderr)
    for run in runs:
        nb = "?" if run.bench_nb_s is None else f"{run.bench_nb_s / 60:.1f}"
        rd = "?" if run.ready_nb_s is None else f"{run.ready_nb_s / 60:.1f}"
        print(f"{run.name}: {len(run.games)} games, benchmark {run.end_s / 60:.1f} min, "
              f"starts at notebook minute {nb}, server ready at notebook minute {rd}")
    minutes = [float(m) for m in args.at.split(",") if m.strip()]
    if minutes:
        print(print_at(runs, minutes, args.axis))
    if args.out:
        for run in runs:
            write_csvs(run, args.out, args.step)
        if minutes:
            with open(args.out / "at_minutes.csv", "w", newline="") as f:
                rows = []
                for axis in ("bench", "notebook", "ready"):
                    rows += at_minutes(runs, minutes, axis)
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                for r in rows:
                    w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and k == "mean_score" else v)
                                for k, v in r.items()})
    if args.html:
        write_html(runs, args.html, args.step, args.title, minutes or [25, 60, 121])
        print(f"wrote {args.html}")
    if problems:
        print(f"{len(problems)} cross-check mismatches:")
        for p in problems:
            print("  " + p)
    else:
        print(f"cross-check: every game's final score, levels, actions and tokens match summary.txt "
              f"and the framework final_score ({sum(len(r.games) for r in runs)} games)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
