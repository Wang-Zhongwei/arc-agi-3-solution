"""Victory lap (ARC3_VICTORY_LAP): replay a won game without its failed attempts.

The official scorer (arc_agi 0.9.8/0.9.9, scorecard.py) keeps every PLAY of a
game and scores the game as the best one:

- ``EnvironmentScoreList.score`` is ``max(run.score for run in self.runs)``
  (scorecard.py:237-241), and the scorecard's total is the mean of those
  per-game maxima (scorecard.py:612-617).
- A RESET whose response says ``full_reset`` opens a new play with fresh
  counters (``update_scorecard`` -> ``new_play`` -> ``Card.inc_play_count``,
  scorecard.py:839-843 and 688-695), and arcengine full-resets on any RESET
  sent in the WIN state, whatever ONLY_RESET_LEVELS says
  (base_game.py:311-314).

So once a game is won, a second play that clears each level in fewer actions
can only raise the game's score, and a second play that stops part-way scores
less than the first and is ignored. This module plans that second play from
the first play's own transitions - no model, no tokens - and executes it with
every step checked against what the first play saw.

Two route planners:

- ``last`` (what the harness runs): each level's last attempt verbatim - the
  actions after the level's last RESET, which is the attempt that cleared it.
  It drops the failed attempts and nothing else, so every step repeats a
  sequence that really ran from the same state. Replaying the 77 wins of 11
  saved runs this way reproduced every one with every frame identical.
- ``bfs``: the shortest route through the graph of every observed transition
  of the level, loop-free. It also cuts detours inside the winning attempt,
  but a detour can carry state the frame does not show (a click that changes
  nothing visible but arms a mode, a step counter that moves only every few
  actions), so on the same 77 wins 19 replays diverged from the recorded
  frames. It adds about 0.01 points per run over ``last`` when nothing
  diverges and loses whole laps when something does, so it is kept for
  analysis (analysis/victory_lap_replay.py --mode bfs) only.

Pure functions plus a small executor that takes the engine step as a callable,
so the solver hook stays thin and everything here is unit-testable.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

GAME_OVER = "GAME_OVER"
WIN = "WIN"


def state_hash(grid: Sequence[Sequence[int]], border: int = 0) -> str:
    """Identity of a board state: md5 of the final visible frame.

    ``border`` > 0 hashes only the board area inside a frame of that width,
    the same crop the harness's no-op guard calls the board
    (ARC3_NOOP_GUARD_BORDER, 4). Many games draw a step counter or progress
    bar in that frame, and the counter is the one thing a shorter route is
    supposed to change. A grid too small for the crop is hashed whole."""
    rows = [list(row) for row in grid]
    if border > 0 and len(rows) > 2 * border and rows and len(rows[0]) > 2 * border:
        rows = [row[border:len(row) - border] for row in rows[border:len(rows) - border]]
    digest = hashlib.md5()
    for row in rows:
        digest.update(bytes(int(cell) & 0xFF for cell in row))
        digest.update(b"|")
    return digest.hexdigest()


def action_key(name: str, data: dict[str, Any] | None) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """An action as the engine takes it: its name and its data (ACTION6 x/y)."""
    return str(name), tuple(sorted((str(k), v) for k, v in (data or {}).items()))


@dataclass(frozen=True)
class Transition:
    """One executed engine action of the first play."""

    prev: str
    action: str
    data: tuple[tuple[str, Any], ...]
    next: str
    levels_before: int
    levels_after: int
    state: str  # the engine state after the action (NOT_FINISHED, GAME_OVER, WIN)

    @classmethod
    def make(
        cls,
        *,
        prev: str,
        action: str,
        data: dict[str, Any] | None,
        next: str,  # noqa: A002 - the field name
        levels_before: int,
        levels_after: int,
        state: str,
    ) -> "Transition":
        name, items = action_key(action, data)
        return cls(prev, name, items, next, int(levels_before), int(levels_after), str(state))


@dataclass(frozen=True)
class Step:
    action: str
    data: tuple[tuple[str, Any], ...]
    expect_hash: str
    expect_levels: int

    def data_dict(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass
class RoutePlan:
    routes: list[list[Step]] = field(default_factory=list)
    start_hash: str = ""
    skip_reason: str | None = None

    @property
    def lengths(self) -> list[int]:
        return [len(route) for route in self.routes]


def plan_routes(
    transitions: Sequence[Transition],
    *,
    start_hash: str,
    number_of_levels: int,
    mode: str = "last",
) -> RoutePlan:
    """A verified route through every level, or a reason there is none.

    ``mode="last"`` (the default, and what the harness runs): each level's
    last attempt verbatim (_plan_last_attempts). ``mode="bfs"``: per level,
    a directed graph of the first play's transitions that stayed on that
    level (RESETs and deaths excluded); BFS from the level's start state to
    the state the level-up action was taken from, then that action. A (state,
    action) pair seen leading to two different states is dropped as
    ambiguous. Loop-free by construction.

    Either way every step carries the state the first play reached, so the
    executor can check it. The level-start state is the game's first state for
    level 1 and the state the level-up action produced for later levels -
    exactly what play 2 will see on arriving there. A level with no route
    that starts there has no verified route, and the whole lap is skipped:
    replaying an action from any other state would be a guess.
    """
    if number_of_levels <= 0:
        return RoutePlan(skip_reason="no levels")
    if mode == "last":
        return _plan_last_attempts(transitions, start_hash=start_hash, number_of_levels=number_of_levels)
    starts: dict[int, str] = {0: start_hash}
    win_edges: dict[int, Transition] = {}
    edges: dict[int, dict[str, dict[tuple, str]]] = {}
    ambiguous: dict[int, set[tuple[str, tuple]]] = {}
    for t in transitions:
        level = t.levels_before
        if t.action == "RESET":
            continue
        if t.levels_after > level:
            if level not in win_edges:
                win_edges[level] = t
            if t.levels_after < number_of_levels and t.levels_after not in starts:
                starts[t.levels_after] = t.next
            continue
        if t.state == GAME_OVER or t.levels_after != level:
            continue
        key = (t.action, t.data)
        out = edges.setdefault(level, {}).setdefault(t.prev, {})
        seen = out.get(key)
        if seen is not None and seen != t.next:
            ambiguous.setdefault(level, set()).add((t.prev, key))
        elif seen is None:
            out[key] = t.next
    plan = RoutePlan(start_hash=start_hash)
    for level in range(number_of_levels):
        win = win_edges.get(level)
        if win is None or level not in starts:
            return RoutePlan(skip_reason=f"level {level + 1}: no recorded level-up")
        graph = edges.get(level, {})
        bad = ambiguous.get(level, set())
        source, target = starts[level], win.prev
        parent: dict[str, tuple[str, tuple] | None] = {source: None}
        queue = deque([source])
        while queue and target not in parent:
            node = queue.popleft()
            for key, nxt in graph.get(node, {}).items():
                if (node, key) in bad or nxt in parent:
                    continue
                parent[nxt] = (node, key)
                queue.append(nxt)
        if target not in parent:
            return RoutePlan(skip_reason=f"level {level + 1}: winning state not reachable from the level start")
        path: list[Step] = []
        node = target
        while parent[node] is not None:
            prev, key = parent[node]
            path.append(Step(key[0], key[1], node, level))
            node = prev
        path.reverse()
        path.append(Step(win.action, win.data, win.next, win.levels_after))
        plan.routes.append(path)
    return plan


def _plan_last_attempts(
    transitions: Sequence[Transition], *, start_hash: str, number_of_levels: int
) -> RoutePlan:
    """Each level's last attempt verbatim: the actions after its last RESET.

    For comparison with the shortest route (analysis/victory_lap_replay.py
    --mode last): it removes failed attempts but never a detour inside the
    winning attempt, and it replays only sequences that really ran."""
    attempts: dict[int, list[Transition]] = {}
    for t in transitions:
        level = t.levels_before
        if t.action == "RESET":
            attempts[level] = []
            continue
        attempts.setdefault(level, []).append(t)
    starts = {0: start_hash}
    plan = RoutePlan(start_hash=start_hash)
    for level in range(number_of_levels):
        attempt = attempts.get(level) or []
        if not attempt or attempt[-1].levels_after <= level:
            return RoutePlan(skip_reason=f"level {level + 1}: no recorded level-up")
        if attempt[0].prev != starts.get(level):
            return RoutePlan(skip_reason=f"level {level + 1}: last attempt does not start at the level start")
        plan.routes.append([Step(t.action, t.data, t.next, t.levels_after) for t in attempt])
        starts[level + 1] = attempt[-1].next
    return plan


def level_score(baseline: float, actions: int) -> float:
    """arc_agi's per-level score: min(115, 100 * (baseline / actions) ** 2)."""
    if actions <= 0:
        return 0.0
    return min(115.0, (float(baseline) / float(actions)) ** 2 * 100.0)


def game_score(
    baselines: Sequence[float] | None,
    actions_per_level: Sequence[int],
    levels_completed: int,
) -> float | None:
    """The official game score of one play (taaf GameRun._compute_final_score,
    which mirrors arc_agi's EnvironmentScoreCalculator). None without baselines."""
    if not baselines:
        return None
    n = len(baselines)
    total = weights = scored = 0.0
    for i in range(n):
        weight = i + 1
        weights += weight
        actions = actions_per_level[i] if i < len(actions_per_level) else 0
        score = level_score(baselines[i], actions) if i < levels_completed and actions > 0 else 0.0
        if score > 0:
            scored += weight
        total += score * weight
    if weights == 0:
        return 0.0
    return min(total / weights, scored / weights * 100.0)


@dataclass
class LapResult:
    status: str  # "won", "aborted", "skipped"
    reason: str = ""
    levels_completed: int = 0
    actions_per_level: list[int] = field(default_factory=list)
    actions: int = 0
    reset_sent: bool = False


# step(action_name, data) -> (state_hash, levels_completed, engine_state_name)
StepFn = Callable[[str, dict[str, Any]], tuple[str, int, str]]


def run_lap(
    plan: RoutePlan,
    step: StepFn,
    *,
    should_stop: Callable[[], bool],
) -> LapResult:
    """RESET (a new play, since the game is in WIN), then each level's route.

    Stops at the first surprise - a state hash other than the one the first
    play recorded, a death, a level count other than expected, a stop request
    or an exception from the engine - and reports why. Never retries and never
    improvises: under max-over-plays an unfinished play costs nothing, and an
    unverified action is the only way the lap could do something the first
    play did not."""
    result = LapResult(status="aborted", actions_per_level=[0] * len(plan.routes))
    if plan.skip_reason:
        return LapResult(status="skipped", reason=plan.skip_reason)
    if should_stop():
        return LapResult(status="skipped", reason="stop requested before the lap")
    try:
        got_hash, levels, state = step("RESET", {})
    except Exception as exc:  # noqa: BLE001 - the lap must never take the game down
        result.reason = f"exception on RESET: {type(exc).__name__}: {exc}"
        return result
    result.reset_sent = True
    if levels != 0 or state in (GAME_OVER, WIN) or got_hash != plan.start_hash:
        result.reason = (
            f"RESET did not restore the first play's start (levels={levels}, state={state}, "
            f"hash {'matches' if got_hash == plan.start_hash else 'differs'})"
        )
        return result
    for level, route in enumerate(plan.routes):
        for index, planned in enumerate(route):
            if should_stop():
                result.reason = f"stop requested at level {level + 1} step {index + 1}"
                return result
            try:
                got_hash, levels, state = step(planned.action, planned.data_dict())
            except Exception as exc:  # noqa: BLE001
                result.reason = f"exception at level {level + 1} step {index + 1}: {type(exc).__name__}: {exc}"
                return result
            result.actions += 1
            result.actions_per_level[level] += 1
            if state == GAME_OVER:
                result.reason = f"GAME_OVER at level {level + 1} step {index + 1}"
                return result
            if levels != planned.expect_levels or got_hash != planned.expect_hash:
                result.reason = (
                    f"mismatch at level {level + 1} step {index + 1}/{len(route)} "
                    f"({planned.action}): levels {levels} vs {planned.expect_levels}, "
                    f"hash {'matches' if got_hash == planned.expect_hash else 'differs'}"
                )
                return result
        result.levels_completed = level + 1
    if state != WIN:
        result.reason = f"routes done but the engine state is {state}"
        return result
    result.status = "won"
    return result
