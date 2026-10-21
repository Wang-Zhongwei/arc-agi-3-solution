# Modified from Tycho tycho/harness/harness.py @f68912a
# (Apache-2.0, copyright 2026 Jens Lehmann, Andrei Aioanei, Sahar Vahdati).
# Changes: `run_env` keeps Tycho's engine loop, per-level action accounting, 5x-baseline budget,
# no-op/revisit diagnostics and the TraceStep/EnvRunRecord/_build_record records. Removed: the
# resume journal and actor snapshot/restore, workspace callbacks (record_solved/record_terminal/
# record_game_over), the status store and builder counters. Added: `start_level` (a per-level test
# jumps the local engine with set_level before the first frame), `deadline` (monotonic seconds;
# stop_reason "deadline"), `on_step` (per-action callback for the runner's trace), and a None
# budget when baselines are hidden (competition mode); `note_final_observation` is also called on
# every non-terminal exit. See PROVENANCE.md.
"""Evaluation harness: drives the arc-agi engine, enforces the RHAE action budget, computes
scores, and writes a structured result record.

Design notes (Tycho's, still true here)
- `available_actions` from a frame are game-action ints; RESET is a separate API control.
- Level transitions are implicit: `levels_completed` increments and the engine advances;
  `state == WIN` means all levels done. A level completion is a rise in `levels_completed`, and
  the actions since the last completion are attributed to that level.
- Budget: official eval terminates a level after `5 * h` actions. Enforced here so local scores
  are not optimistically biased. With hidden baselines (Kaggle) there is no per-level budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from arcengine import FrameData, GameAction, GameState

from .actions import actor_actions, normalize_game_actions
from .agent import ActionChoice, Agent
from .scoring import EnvResult, LevelResult

ACTION_BUDGET_MULTIPLIER = 5  # official eval cutoff: 5 * human_baseline per level

log = logging.getLogger("v9.harness")


def _frame_key(frame) -> int:
    """Hash the innermost grid of a frame (1xHxW). Used for no-op/loop detection."""
    grid = frame[-1] if frame is not None and len(frame) else []
    return hash(tuple(tuple(int(v) for v in row) for row in grid))


# Keys that are large or analysis-only: retained in the trace but never sent to the engine
# (arcengine caps ActionInput.reasoning at 16384 bytes).
_VIZ_ONLY_KEYS = {"llm_calls", "world_model_code", "objects", "plan", "mechanics", "prompt"}


def _engine_reasoning(reasoning: Optional[dict]) -> Optional[dict]:
    """A small, engine-safe subset of the agent's reasoning (drops viz-only/large
    keys; truncates as a backstop) so we never exceed the engine's 16KB cap."""
    if not reasoning:
        return reasoning
    small = {k: v for k, v in reasoning.items() if k not in _VIZ_ONLY_KEYS}
    s = str(small)
    if len(s) > 8000:  # backstop: keep only short scalar fields
        small = {k: v for k, v in small.items() if isinstance(v, (int, float, bool, str)) and len(str(v)) < 300}
    return small


def _grid_to_list(frame) -> Optional[list]:
    """Innermost grid as plain nested lists (JSON-safe; frame rows may be ndarrays). Uses frame[-1] —
    the resulting PLAYABLE state after the action (intermediate animation frames are decorative)."""
    if frame is None or not len(frame):
        return None
    grid = frame[-1]
    return [list(map(int, row)) for row in grid]


def _grid_to_first(frame) -> Optional[list]:
    """frame[0] as plain nested lists. On a level-completing action the engine returns
    [goal_reached_terminal, ...animation..., new_level_init]; frame[0] is the terminal BEFORE the
    advance — the goal-reached configuration. (frame[-1] is the next level, which _grid_to_list returns.)"""
    if frame is None or not len(frame):
        return None
    grid = frame[0]
    return [list(map(int, row)) for row in grid]


@dataclass
class TraceStep:
    turn: int
    action: str
    x: Optional[int]
    y: Optional[int]
    state: str
    levels_completed: int
    frame_changed: bool  # did this action alter the grid? (no-op detection)
    frame_key: int  # hash of the resulting grid (loop / revisit detection)
    frame: Optional[list] = None  # resulting HxW grid — only when keep_frames (viz)
    reasoning: Optional[dict] = None  # agent's per-step internal state, if it attached any


@dataclass
class EnvRunRecord:
    game_id: str
    n_levels: int
    baselines: list[int]
    levels_completed: int
    total_actions: int
    resets: int
    final_state: str
    env_score: float
    level_results: list[dict]
    wall_clock_s: float
    truncated_levels: list[int]  # levels cut by the action budget
    completed_level_actions: int = 0
    unfinished_level_index: Optional[int] = None
    unfinished_level_actions: int = 0
    total_actions_including_unfinished: int = 0
    reset_actions_counted: bool = True
    # Goal-agnostic behavioural diagnostics.
    noop_actions: int = 0  # scored actions that didn't change the grid
    distinct_frames: int = 0  # unique grid states seen
    revisits: int = 0  # transitions back into an already-seen frame
    actions_before_first_level: Optional[int] = None  # exploration cost proxy
    error: Optional[str] = None  # set if the game ended on an exception (partial record)
    stop_reason: Optional[str] = None
    start_level: int = 1  # v9: 1-indexed level the run started on (per-level tests)
    trace: list[dict] = field(default_factory=list)
    partial: bool = False


def run_env(
    agent: Agent,
    env,
    game_id: str,
    baselines: Optional[list[int]],
    *,
    n_levels: Optional[int] = None,
    first_frame: Optional[FrameData] = None,
    start_level: int = 1,
    keep_trace: bool = True,
    keep_frames: bool = False,
    deadline: Optional[float] = None,
    max_actions: Optional[int] = None,
    on_step: Optional[Callable[[dict], None]] = None,
) -> EnvRunRecord:
    """Play one already-made environment end-to-end under the action budget; return a record.

    `env` is an arc_agi environment wrapper (from `arcade.make`). `first_frame` is the frame to
    start from (the caller has already called `env.reset()` or jumped to `start_level`); when it
    is None the harness resets the environment itself. `baselines` may be None (hidden on
    Kaggle), in which case `n_levels` is required and there is no per-level budget.
    `deadline` is a `time.monotonic()` instant after which the run stops before the next action.
    `on_step(info)` is called after every executed action with the trace row as a dict.
    """
    keep_trace = keep_trace or keep_frames
    if baselines is None and n_levels is None:
        raise ValueError("run_env needs baselines or n_levels")
    n_levels = int(n_levels if n_levels is not None else len(baselines))
    t0 = time.time()

    frame = first_frame if first_frame is not None else env.reset()
    frames: list[FrameData] = [frame]
    game_avail = normalize_game_actions(frame.available_actions)
    agent.reset(game_id, game_avail)

    level_results: list[LevelResult] = []
    truncated: list[int] = []
    trace: list[TraceStep] = []

    cur_level = max(0, int(start_level) - 1)  # 0-indexed level currently being played
    actions_this_level = 0
    resets = 0
    turn = 0

    # Diagnostic counters (computed even without a full trace).
    noop_actions = 0  # scored actions that left the grid unchanged
    seen_keys: set[int] = set()
    revisits = 0  # transitions into an already-seen frame
    actions_before_first_level: Optional[int] = None
    prev_key = _frame_key(frame.frame)
    seen_keys.add(prev_key)
    run_error: Optional[str] = None  # set if a mid-game exception ends the run early
    stop_reason: Optional[str] = None

    while True:
        state = frame.state
        completed_so_far = frame.levels_completed

        if state == GameState.WIN:
            stop_reason = "win"
            break
        if cur_level >= n_levels:
            stop_reason = "all_levels_accounted"
            break
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "deadline"
            break
        if max_actions is not None and turn >= max_actions:
            stop_reason = "max_actions"
            break
        if agent.is_done(frames, frame):
            stop_reason = getattr(agent, "done_reason", None) or "agent_done"
            break

        # Budget for the current level (None when baselines are hidden).
        budget = None
        if baselines is not None and cur_level < len(baselines) and baselines[cur_level]:
            budget = ACTION_BUDGET_MULTIPLIER * int(baselines[cur_level])

        # A mid-game exception (for example a provider outage or an agent bug)
        # must NOT discard the trace collected so far — break with a partial record
        # instead of propagating.
        try:
            if state == GameState.GAME_OVER:
                # Only RESET is valid. The initial env.reset() above starts the play for free, but
                # this in-play recovery RESET is one scored action in the official scorecard.
                if hasattr(agent, "note_external_reset"):
                    try:
                        agent.note_external_reset()
                    except Exception:  # noqa: BLE001
                        pass
                choice = ActionChoice(action=GameAction.RESET)
                resets += 1
            else:
                avail = actor_actions(frame.available_actions)
                choice = agent.choose_action(frames, frame, avail)

            next_frame = env.step(
                choice.action,
                data=getattr(choice, "data", None),
                reasoning=_engine_reasoning(choice.reasoning),  # engine-safe subset (16KB cap)
            )
        except Exception as e:  # noqa: BLE001 — preserve partial progress on any failure
            run_error = f"{type(e).__name__}: {e}"
            stop_reason = "exception"
            log.warning("%s: run aborted at turn %d: %s", game_id, turn, run_error, exc_info=True)
            break

        turn += 1
        # Official scorecard accounting charges every in-play environment control, including RESET.
        actions_this_level += 1

        if next_frame is None:
            log.warning("%s: engine returned None at turn %d", game_id, turn)
            stop_reason = "engine_returned_none"
            break
        if int(next_frame.levels_completed) < int(completed_so_far):
            run_error = (
                "LevelRegressionError: ARC engine reported levels_completed decrease "
                f"{completed_so_far}->{next_frame.levels_completed} after {choice.action.name}; "
                "refusing to continue from a full-game reset state"
            )
            stop_reason = "level_regression"
            log.warning("%s: %s", game_id, run_error)
            break

        if choice.action == GameAction.RESET and state != GameState.GAME_OVER:
            resets += 1
            if hasattr(agent, "note_actor_reset"):
                try:
                    agent.note_actor_reset()
                except Exception:  # noqa: BLE001
                    pass

        # Most agents observe an action's result on the next choose_action() call. Terminal states
        # do not necessarily have a next decision, so give opt-in agents one final evidence callback.
        if next_frame.state in (GameState.WIN, GameState.GAME_OVER) and hasattr(agent, "note_final_observation"):
            try:
                agent.note_final_observation(
                    next_frame,
                    normalize_game_actions(next_frame.available_actions),
                )
            except Exception:  # noqa: BLE001
                log.exception("%s: terminal observation callback failed at turn %d", game_id, turn)

        new_key = _frame_key(next_frame.frame)
        changed = new_key != prev_key
        if choice.action != GameAction.RESET and not changed:
            noop_actions += 1
        if new_key in seen_keys and changed:
            revisits += 1
        seen_keys.add(new_key)

        step = TraceStep(
            turn=turn,
            action=choice.action.name,
            x=getattr(choice, "x", None),
            y=getattr(choice, "y", None),
            state=str(next_frame.state),
            levels_completed=next_frame.levels_completed,
            frame_changed=changed,
            frame_key=new_key,
            frame=(
                ((_grid_to_first(next_frame.frame) or _grid_to_list(next_frame.frame))
                 if next_frame.state == GameState.GAME_OVER else _grid_to_list(next_frame.frame))
                if keep_frames else None
            ),
            reasoning=choice.reasoning,
        )
        if keep_trace:
            trace.append(step)
        if on_step is not None:
            try:
                on_step({**asdict(step), "level_index": cur_level + 1,
                         "actions_this_level": actions_this_level, "elapsed_s": time.time() - t0})
            except Exception:  # noqa: BLE001 — the runner's logging must not break play
                log.exception("%s: on_step callback failed at turn %d", game_id, turn)
        prev_key = new_key

        # Did a level just complete? (levels_completed rose)
        level_completed = next_frame.levels_completed > completed_so_far
        budget_exhausted = False
        if level_completed:
            if not level_results:  # first completion: exploration-cost proxy
                actions_before_first_level = actions_this_level
            level_results.append(
                LevelResult(
                    level_index=cur_level + 1,
                    completed=True,
                    actions_taken=actions_this_level,
                    baseline_actions=int(baselines[cur_level]) if baselines else 0,
                )
            )
            cur_level += 1
            actions_this_level = 0
        elif budget is not None and actions_this_level >= budget:
            # Budget exhausted on this level without completing it.
            level_results.append(
                LevelResult(
                    level_index=cur_level + 1,
                    completed=False,
                    actions_taken=actions_this_level,
                    baseline_actions=int(baselines[cur_level]) if baselines else 0,
                )
            )
            truncated.append(cur_level + 1)
            cur_level += 1
            actions_this_level = 0
            stop_reason = "level_action_budget"
            budget_exhausted = True
            if hasattr(agent, "note_final_observation"):
                try:
                    agent.note_final_observation(
                        next_frame,
                        normalize_game_actions(next_frame.available_actions),
                    )
                except Exception:  # noqa: BLE001
                    log.exception("%s: final observation callback failed at action budget", game_id)

        frame = next_frame
        frames.append(frame)
        if budget_exhausted:
            # Without env-level seeking we cannot skip ahead; stop this env.
            break

    # v9: every exit hands the agent the last frame once, so the final action is graded and
    # recorded even when the stop was a budget, a deadline or max_actions rather than a terminal.
    if hasattr(agent, "note_final_observation") and frame.state not in (GameState.WIN, GameState.GAME_OVER):
        try:
            agent.note_final_observation(frame, normalize_game_actions(frame.available_actions))
        except Exception:  # noqa: BLE001
            log.exception("%s: final observation callback failed at exit", game_id)

    # Any levels never reached are implicit 0s; scoring handles missing indices.
    return _build_record(
        game_id, n_levels, baselines or [0] * n_levels, level_results, frame, resets, time.time() - t0,
        truncated, noop_actions, len(seen_keys), revisits, actions_before_first_level,
        run_error, [asdict(s) for s in trace] if keep_trace else [], partial=False,
        stop_reason=stop_reason, start_level=start_level,
        current_level_index=cur_level + 1, current_level_actions=actions_this_level)


def _build_record(game_id, n_levels, baselines, level_results, frame, resets, elapsed_s,
                  truncated, noop_actions, distinct_frames, revisits, actions_before_first_level,
                  run_error, trace, *, partial, current_level_index=None, current_level_actions=0,
                  stop_reason=None, start_level=1) -> EnvRunRecord:
    """Assemble an EnvRunRecord from loop state."""
    er = EnvResult(game_id=game_id, n_levels=n_levels, levels=level_results,
                   state=str(frame.state), resets=resets)
    recorded_indices = {l.level_index for l in level_results}
    unfinished_level_index = None
    unfinished_level_actions = 0
    if (
        current_level_actions
        and current_level_index is not None
        and 1 <= current_level_index <= n_levels
        and current_level_index not in recorded_indices
        and str(frame.state) != "GameState.WIN"
    ):
        unfinished_level_index = current_level_index
        unfinished_level_actions = current_level_actions
    completed_level_actions = sum(l.actions_taken for l in level_results if l.completed)
    return EnvRunRecord(
        game_id=game_id,
        n_levels=n_levels,
        baselines=list(baselines),
        levels_completed=int(frame.levels_completed),
        total_actions=er.total_actions,
        completed_level_actions=completed_level_actions,
        unfinished_level_index=unfinished_level_index,
        unfinished_level_actions=unfinished_level_actions,
        total_actions_including_unfinished=er.total_actions + unfinished_level_actions,
        reset_actions_counted=True,
        resets=resets,
        final_state=str(frame.state),
        env_score=er.env_score(),
        level_results=[asdict(l) for l in level_results],
        wall_clock_s=round(elapsed_s, 2),
        truncated_levels=truncated,
        noop_actions=noop_actions,
        distinct_frames=distinct_frames,
        revisits=revisits,
        actions_before_first_level=actions_before_first_level,
        error=run_error,
        stop_reason=stop_reason,
        start_level=int(start_level),
        trace=trace,
        partial=partial,
    )
