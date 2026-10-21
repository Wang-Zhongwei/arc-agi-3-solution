# Modified from Tycho tycho/workspace/wmlib_template.py @f68912a
# (Apache-2.0, copyright 2026 Jens Lehmann, Andrei Aioanei, Sahar Vahdati).
# Changes: extracted the planner block (BFS/A* with diagnostics, validate_plan, the search-key
# projection) and the abstention-aware render scorer (_cell_stats/_coverage_status/
# _predicts_change) from the 1773-line workspace library; the workspace file readers and verify()
# are not carried. See PROVENANCE.md.
"""Search over a world model that exposes `actions(state)`, `transition(state, action)` and
`outcome(state)`, plus the coverage bucket that makes an abstaining prediction visible.

The model object here is whatever v9 builds from certified rules (v9/planner.py); the contract is
Tycho's: a plan is accepted only when canonical replay reaches the target, so any fast or
approximate search stays safe.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np

UNKNOWN = -1
OUTCOME_ONGOING = "ongoing"
OUTCOME_LEVEL_COMPLETE = "level_complete"
OUTCOME_GAME_OVER = "game_over"
VALID_OUTCOMES = {OUTCOME_ONGOING, OUTCOME_LEVEL_COMPLETE, OUTCOME_GAME_OVER}


def clone(state):
    """Deep-copy a state so transition() mutates a copy (the planner holds many states at once)."""
    import copy as _copy
    return _copy.deepcopy(state)


def outcome(model, state) -> str:
    """Strictly read the model's terminal/status output: one of ongoing / level_complete / game_over."""
    if not hasattr(model, "outcome") or not callable(model.outcome):
        raise AttributeError("world model must define outcome(state)")
    value = model.outcome(state)
    if value not in VALID_OUTCOMES:
        raise ValueError(
            "outcome(state) must return exactly one of "
            f"{sorted(VALID_OUTCOMES)}, got {value!r}"
        )
    return value


def is_level_complete(model, state) -> bool:
    return outcome(model, state) == OUTCOME_LEVEL_COMPLETE


def is_game_over(model, state) -> bool:
    return outcome(model, state) == OUTCOME_GAME_OVER


def _hashable(o):
    """Stable, hashable view of an arbitrary value — handles dataclass/object/dict/list/tuple/numpy/
    sets. The planner uses this to dedup search nodes, generically over whatever fields your State has."""
    if isinstance(o, np.ndarray):
        return ("ndarray", str(o.dtype), tuple(o.shape), o.tobytes(order="C"))
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "__dict__") and not isinstance(o, type):
        return ("obj", tuple(sorted((k, _hashable(v)) for k, v in vars(o).items())))
    if isinstance(o, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in o.items()))
    if isinstance(o, (list, tuple)):
        return tuple(_hashable(v) for v in o)
    if isinstance(o, set):
        return tuple(sorted(_hashable(v) for v in o))
    try:
        hash(o); return o
    except TypeError:
        return repr(o)


# --- abstention-aware render scoring ---------------------------------------------------------


def _cell_stats(pred, obs):
    """Cell-level diagnostics for a render candidate.

    UNKNOWN (-1) means "this cell is intentionally unclaimed by render(state)". Accepted matching
    ignores unknown cells, but strict/cell diagnostics still expose the abstention through coverage.
    """
    if pred is None or pred.shape != obs.shape:
        return {"shape_ok": False, "full_cell_ok": 0, "known_cell_ok": 0, "known_cell_n": 0,
                "unknown_cell_n": 0, "cell_n": int(obs.size), "coverage": None,
                "coverage_status": "unobserved",
                "known_cell_accuracy": None, "partial_ok": False, "strict_ok": False}
    unknown = (pred == UNKNOWN)
    known = ~unknown
    cell_n = int(obs.size)
    known_cell_n = int(np.count_nonzero(known))
    unknown_cell_n = cell_n - known_cell_n
    full_cell_ok = int(np.count_nonzero(pred == obs))
    known_cell_ok = int(np.count_nonzero((pred == obs) & known))
    coverage = known_cell_n / cell_n if cell_n else None
    known_acc = known_cell_ok / known_cell_n if known_cell_n else None
    return {"shape_ok": True,
            "full_cell_ok": full_cell_ok,
            "known_cell_ok": known_cell_ok,
            "known_cell_n": known_cell_n,
            "unknown_cell_n": unknown_cell_n,
            "cell_n": cell_n,
            "coverage": coverage,
            "coverage_status": _coverage_status(coverage),
            "known_cell_accuracy": known_acc,
            "partial_ok": known_cell_n > 0 and known_cell_ok == known_cell_n,
            "strict_ok": bool(np.array_equal(pred, obs))}


def _coverage_status(coverage):
    """Human-facing coverage bucket for UNKNOWN-aware renders.

    complete: render() claimed every cell.
    partial : render() abstained on some cells with UNKNOWN=-1.
    vacuous : render() abstained on every cell; this is not evidence of a correct model.
    """
    if coverage is None:
        return "unobserved"
    if coverage <= 0:
        return "vacuous"
    if coverage >= 0.999999:
        return "complete"
    return "partial"


def _predicts_change(pred, prev) -> bool:
    """Whether a rendered prediction claims any visible change from prev.

    UNKNOWN cells are abstentions, not claimed changes.
    """
    if pred is None:
        return False
    if pred.shape != prev.shape:
        return True
    known = pred != UNKNOWN
    return bool(np.any((pred != prev) & known))


# --- search ----------------------------------------------------------------------------------


def _planner_key(model, state):
    """Return a search key, using an optional exact semantic projection.

    Pixel-heavy models often keep immutable terrain or render caches in ``State``. Hashing those
    fields at every node is correct but needlessly expensive. A model may therefore define
    ``planner_key(state)`` returning only fields that distinguish future dynamics and outcomes.
    The generic full-state key remains the default.
    """
    hook = getattr(model, "planner_key", None)
    value = hook(state) if callable(hook) else state
    return _hashable(value)


def _planner_successor(model, state, action):
    """Apply one planner edge without copying twice when the model promises a fresh result."""
    source = state if getattr(model, "TRANSITION_RETURNS_NEW_STATE", False) else clone(state)
    return model.transition(source, action)


def _acts(model, state):
    try:
        raw_actions = list(model.actions(state))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"actions(state) raised {type(e).__name__}: {e}") from e
    if raw_actions:
        return [_norm_action(a, strict=True) for a in raw_actions]
    return [{"action": x, "row": None, "col": None} for x in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")]


def _norm_action(a, *, strict=False):
    if isinstance(a, dict):
        if strict and "action" not in a:
            raise ValueError(
                "actions(state) must return dicts with an 'action' key; "
                f"got {a!r}"
            )
        return {"action": a.get("action"), "row": a.get("row"), "col": a.get("col")}
    if strict:
        raise ValueError(
            "actions(state) must return full action dicts "
            '{"action": "...", "row": None, "col": None}; '
            f"got {type(a).__name__} {a!r}"
        )
    return {"action": a, "row": None, "col": None}


def _target(model, target):
    return target or (lambda s: is_level_complete(model, s))


def _is_target(target, state):
    try:
        return bool(target(state))
    except Exception:  # noqa: BLE001
        return False


def _should_prune(model, state):
    try:
        return is_game_over(model, state)
    except Exception:  # noqa: BLE001
        return False


def plan_bfs_with_diagnostics(model, start, target=None, max_nodes=20000, prune_game_over=True):
    """BFS with explicit no-plan diagnostics.

    status is:
      found                 plan reached target
      no_plan_exhausted     reachable state graph was exhausted within max_nodes
      no_plan_within_budget max_nodes stopped search while a frontier remained
    """
    target = _target(model, target)
    q = deque([(start, [])]); seen = {_planner_key(model, start)}; n = 0; errors = []
    while q and n < max_nodes:
        st, path = q.popleft(); n += 1
        if _is_target(target, st):
            return {"status": "found", "plan": path, "expanded_nodes": n,
                    "frontier_remaining": bool(q), "seen_states": len(seen), "errors": errors}
        if prune_game_over and _should_prune(model, st):
            continue
        try:
            actions = _acts(model, st)
        except Exception as e:  # noqa: BLE001
            return {"status": "actions_error", "plan": None, "expanded_nodes": n,
                    "frontier_remaining": bool(q), "seen_states": len(seen),
                    "errors": [f"{type(e).__name__}: {e}"]}
        for a in actions:
            try:
                nx = _planner_successor(model, st, a)
            except Exception as e:  # noqa: BLE001
                if len(errors) < 5:
                    errors.append(f"transition({a}) raised {type(e).__name__}: {e}")
                continue
            k = _planner_key(model, nx)
            if k not in seen:
                seen.add(k); q.append((nx, path + [a]))
    status = "no_plan_within_budget" if q else "no_plan_exhausted"
    return {"status": status, "plan": None, "expanded_nodes": n,
            "frontier_remaining": bool(q), "seen_states": len(seen), "errors": errors}


def plan_bfs(model, start, target=None, max_nodes=20000, prune_game_over=True):
    """Shortest action sequence from `start` to a target state, or None if unreachable within max_nodes."""
    diag = plan_bfs_with_diagnostics(model, start, target, max_nodes, prune_game_over)
    return diag.get("plan") if diag.get("status") == "found" else None


def plan_astar_with_diagnostics(model, start, target=None, heuristic=None,
                                max_nodes=20000, prune_game_over=True):
    """A* with explicit no-plan diagnostics (same statuses as plan_bfs_with_diagnostics)."""
    target = _target(model, target)
    h = heuristic or (lambda _s: 0); cnt = 0
    pq = [(h(start), 0, cnt, start, [])]
    best = {_planner_key(model, start): 0}; n = 0; errors = []
    while pq and n < max_nodes:
        _, g, _, st, path = heapq.heappop(pq)
        if best.get(_planner_key(model, st)) != g:
            continue
        n += 1
        if _is_target(target, st):
            return {"status": "found", "plan": path, "expanded_nodes": n,
                    "frontier_remaining": bool(pq), "seen_states": len(best), "errors": errors}
        if prune_game_over and _should_prune(model, st):
            continue
        try:
            actions = _acts(model, st)
        except Exception as e:  # noqa: BLE001
            return {"status": "actions_error", "plan": None, "expanded_nodes": n,
                    "frontier_remaining": bool(pq), "seen_states": len(best),
                    "errors": [f"{type(e).__name__}: {e}"]}
        for a in actions:
            try:
                nx = _planner_successor(model, st, a)
            except Exception as e:  # noqa: BLE001
                if len(errors) < 5:
                    errors.append(f"transition({a}) raised {type(e).__name__}: {e}")
                continue
            ng = g + 1; k = _planner_key(model, nx)
            if k in best and best[k] <= ng:
                continue
            best[k] = ng; cnt += 1
            heapq.heappush(pq, (ng + h(nx), ng, cnt, nx, path + [a]))
    status = "no_plan_within_budget" if pq else "no_plan_exhausted"
    return {"status": status, "plan": None, "expanded_nodes": n,
            "frontier_remaining": bool(pq), "seen_states": len(best), "errors": errors}


def plan_astar(model, start, target=None, heuristic=None, max_nodes=20000, prune_game_over=True):
    """A* from `start` to a target state, guided by heuristic(state)->estimated moves."""
    diag = plan_astar_with_diagnostics(model, start, target, heuristic, max_nodes, prune_game_over)
    return diag.get("plan") if diag.get("status") == "found" else None


def validate_plan(model, start, plan, target=None, capture_renders=False):
    """Replay a candidate action sequence through the canonical executable model.

    A plan is accepted when every canonical transition succeeds and the final canonical state
    satisfies ``target`` (by default, ``outcome(state) == "level_complete"``).
    """
    target = _target(model, target)
    state = clone(start)
    actions = []
    renders = []
    errors = []
    for index, raw in enumerate(plan or []):
        try:
            action = _norm_action(raw, strict=True)
        except Exception as e:  # noqa: BLE001
            return {
                "status": "invalid_action",
                "plan": None,
                "steps_validated": index,
                "final_outcome": None,
                "errors": [f"action {index + 1}: {type(e).__name__}: {e}"],
                "renders": renders,
            }
        try:
            state = _planner_successor(model, state, action)
            if capture_renders:
                renders.append(np.asarray(model.render(state), dtype=np.int16).copy())
        except Exception as e:  # noqa: BLE001
            return {
                "status": "transition_error",
                "plan": None,
                "steps_validated": index,
                "final_outcome": None,
                "errors": [f"step {index + 1} {action}: {type(e).__name__}: {e}"],
                "renders": renders,
            }
        actions.append(action)
    try:
        final_outcome = outcome(model, state)
        reached = _is_target(target, state)
    except Exception as e:  # noqa: BLE001
        final_outcome = None
        reached = False
        errors.append(f"final target/outcome: {type(e).__name__}: {e}")
    return {
        "status": "found" if reached else "goal_not_reached",
        "plan": actions if reached else None,
        "steps_validated": len(actions),
        "final_outcome": final_outcome,
        "errors": errors,
        "renders": renders,
    }
