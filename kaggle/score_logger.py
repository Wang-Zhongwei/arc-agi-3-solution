# --- score timeline: one JSON line per game every minute ----------------------
# kaggle/build_notebook.py pastes this file into its own notebook cell (after the
# system monitor), followed by a call to start_score_logger. It is also imported by
# kaggle/tests/test_score_logger.py, so it must stay standard-library only.
#
# A daemon thread reads the live taaf GameRun objects in bm.game_runs (Benchmark.run
# appends one per game when it starts them, and the framework updates
# history / actions_per_level / levels_completed / state in place as moves are made).
# It only reads: no locks, no calls into the solver, the engine or the model server,
# so it cannot change a game decision, a prompt or a request. Every read is wrapped,
# so a half-updated GameRun or a write error costs at most one line, never the run.
#
# Every `interval_s` it appends to `path` one line per game:
#   {"kind": "game", "t_notebook_s", "t_ready_s", "t_bench_s", "wall", "game", "pass",
#    "levels", "n_levels", "actions", "score", "tokens", "state", "run_state"}
# and one {"kind": "run", ..., "mean_score", "levels", "actions", "tokens",
# "games_finished", "n_games"} line. Every `print_every_s` it prints
#   [score] t=37m nb=49m mean=12.40 lv=31 act=1234 tok=1.23M fin=3/25 | ft09 3L 40.1 | ...
# t is minutes since the benchmark started, nb since the notebook started.
# score is the official formula (taaf GameRun._compute_final_score): level score
# min(115, (baseline/actions)^2 * 100), level i weighted by i, uncleared levels 0,
# capped at 100 * (weight of scored levels) / (total weight). None when the engine
# hides the baselines (competition rerun).
import json
import threading
import time
from datetime import datetime


def _official_score(number_of_levels, base_actions_per_level, levels_completed, actions_per_level):
    if base_actions_per_level is None or number_of_levels == 0:
        return None
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(number_of_levels):
        weight = level_idx + 1
        total_weights += weight
        actions = actions_per_level[level_idx] if level_idx < len(actions_per_level) else 0
        if level_idx < levels_completed and actions > 0:
            level_score = min(115.0, (base_actions_per_level[level_idx] / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100)


_FINISHED_STATES = {"won", "gave_up", "cancelled"}


class _TokenCounter:
    """Running sum of generated tokens per GameRun: each tick reads only new history records."""

    def __init__(self):
        self._seen = {}  # id(run) -> (records counted, tokens)

    def tokens(self, run, n_records):
        count, total = self._seen.get(id(run), (0, 0))
        if n_records < count:  # not expected (history only grows); recount
            count, total = 0, 0
        hist = run.history
        for i in range(count, n_records):
            total += int(getattr(hist[i], "generated_tokens", 0) or 0)
        self._seen[id(run)] = (n_records, total)
        return total


def score_snapshot(bm, *, notebook_start=None, server_ready=None, now=None, counter=None):
    """One dict per game plus a final "run" dict, read from the live benchmark. Never raises
    for a single game: a game that cannot be read gets {"kind": "game", "error": ...}."""
    now = time.time() if now is None else now
    runs = list(getattr(bm, "game_runs", None) or [])
    start = getattr(bm, "start_time", None)
    clock = {
        "wall": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        "t_notebook_s": None if notebook_start is None else round(now - notebook_start, 1),
        "t_ready_s": None if server_ready is None else round(now - server_ready, 1),
        "t_bench_s": None if start is None else round((datetime.fromtimestamp(now) - start).total_seconds(), 1),
    }
    counter = counter or _TokenCounter()
    n_passes = max(1, int(getattr(bm, "n_passes", 1) or 1))
    per_pass = len(runs) // n_passes if runs and len(runs) % n_passes == 0 else len(runs)
    rows = []
    for i, run in enumerate(runs):
        row = {"kind": "game", **clock, "game": str(getattr(run, "game_id", f"#{i}")),
               "pass": i // per_pass if per_pass else 0}
        try:
            # levels before the counts: a level cleared between the two reads then
            # scores on the next tick instead of with a short action count.
            levels = int(run.levels_completed)
            per_level = list(run.actions_per_level)
            n_records = len(run.history)
            raw_state = str(run.state)
            final = run.final_score
            score = float(final) if final is not None else _official_score(
                int(run.number_of_levels), run.base_actions_per_level, levels, per_level)
            row.update({
                "levels": levels, "n_levels": int(run.number_of_levels), "actions": n_records,
                "score": None if score is None else round(score, 4),
                "tokens": counter.tokens(run, n_records),
                "state": ("playing" if raw_state == "playing" else
                          "finished" if raw_state in _FINISHED_STATES else raw_state),
                "run_state": raw_state,
            })
        except Exception as exc:  # a half-built or odd GameRun: report, keep going
            row["error"] = repr(exc)[:200]
        rows.append(row)
    ok = [r for r in rows if "error" not in r]
    by_game = {}
    for r in ok:
        by_game.setdefault(r["game"], []).append(r["score"] or 0.0)
    mean = (sum(sum(v) / len(v) for v in by_game.values()) / len(by_game)) if by_game else None
    rows.append({
        "kind": "run", **clock,
        "mean_score": None if mean is None else round(mean, 4),
        "levels": sum(r["levels"] for r in ok), "actions": sum(r["actions"] for r in ok),
        "tokens": sum(r["tokens"] for r in ok),
        "games_finished": sum(1 for r in ok if r["state"] != "playing"),
        "n_games": len({r["game"] for r in rows if r["kind"] == "game"}),
        "unreadable": len(rows) - len(ok),
    })
    return rows


def format_score_line(rows):
    """[score] t=37m nb=49m mean=12.40 lv=31 act=1234 tok=1.23M fin=3/25 | ft09 3L 40.1 | ..."""
    run = rows[-1]

    def minutes(s):
        return "?" if s is None else f"{s / 60:.0f}m"

    def num(v):
        return "?" if v is None else f"{v:.1f}"

    head = (f"[score] t={minutes(run['t_bench_s'])} nb={minutes(run['t_notebook_s'])} "
            f"mean={'?' if run['mean_score'] is None else format(run['mean_score'], '.2f')} "
            f"lv={run['levels']} act={run['actions']} tok={run['tokens'] / 1e6:.2f}M "
            f"fin={run['games_finished']}/{run['n_games']}")
    games = []
    for r in sorted((r for r in rows if r["kind"] == "game"), key=lambda r: (r["game"], r["pass"])):
        name = r["game"].split("-")[0] + (f"#{r['pass']}" if r["pass"] else "")
        if "error" in r:
            games.append(f"{name} ?")
        else:
            mark = "" if r["state"] == "playing" else ("*" if r["state"] == "finished" else "!")
            games.append(f"{name} {r['levels']}L {num(r['score'])}{mark}")
    return " | ".join([head, *games])


def start_score_logger(get_bm, path, *, notebook_start=None, get_server_ready=None,
                       interval_s=60.0, print_every_s=300.0, out=print):
    """Start the daemon thread; returns an Event that stops it. `get_bm` and
    `get_server_ready` are called every tick, so they may return None until bm exists /
    the server answers. Logging starts when bm.game_runs is non-empty and ends one tick
    after bm.end_time is set (once a run has been seen in progress). The thread is
    the returned Event's `.thread`."""
    stop = threading.Event()
    counter = _TokenCounter()
    state = {"last_print": None, "errors": 0, "running": False}

    def tick():
        bm = get_bm()
        if bm is None or getattr(bm, "start_time", None) is None or not getattr(bm, "game_runs", None):
            return False
        # Benchmark.run clears end_time when it starts; only stop after seeing a run in progress.
        if getattr(bm, "end_time", None) is None:
            state["running"] = True
        elif not state["running"]:
            return False
        ready = get_server_ready() if get_server_ready else None
        rows = score_snapshot(bm, notebook_start=notebook_start, server_ready=ready, counter=counter)
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
        now = time.monotonic()
        ended = getattr(bm, "end_time", None) is not None
        if ended or state["last_print"] is None or now - state["last_print"] >= print_every_s - 1:
            out(format_score_line(rows), flush=True)
            state["last_print"] = now
        return ended

    def loop():
        while not stop.is_set():
            try:
                if tick():
                    return
            except Exception as exc:
                state["errors"] += 1
                if state["errors"] <= 3:  # say it, but never flood the log
                    out(f"[score] logger tick failed ({state['errors']}): {exc!r}", flush=True)
            stop.wait(interval_s)

    stop.thread = threading.Thread(target=loop, name="score-timeline", daemon=True)
    stop.thread.start()
    return stop
