"""The turn graph: recall -> world -> goal -> mode -> propose -> commit -> end.

Each node is a function over `TurnState` (a thin view onto the GameSession). Perception and
the deterministic delta happen at engine-step granularity in session.observe(); this graph runs
once per model turn, when the batch queue is empty or a prediction was violated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .graph import END, Graph
from .modes import ModeInputs, ROLE_PLAN, commit_gate, mode_of
from .types import Confidences, Mode


@dataclass
class TurnState:
    session: Any
    turn: int
    mode: Mode = Mode.EXPLORE
    mode_reason: str = ""
    roles: list[tuple[str, int]] = field(default_factory=list)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    batch: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    plan_validated: bool = False
    harness_explores: bool = False
    harness_measures: bool = False
    novelty: float = 0.0
    disagreement: bool = False


def recall(st: TurnState) -> TurnState:
    """Pull the relevant memory: priors on a new game, the level model on a new level."""
    s = st.session
    # Not `turn_index == 0`: `run_turn` increments it before the graph runs, so inside a node it
    # is never 0 and this never fired. The whole cross-game store was write-only because of it,
    # and keyed on the empty family string besides.
    if not getattr(s, "priors_loaded", False):
        s.load_priors()
    st.novelty = s.novelty_score()
    return st


def world(st: TurnState) -> TurnState:
    """Harness-side theory upkeep: control rules from the control map, rescoring, staleness."""
    s = st.session
    s.refresh_rules()
    st.plan_validated = s.plan_ready()
    return st


def goal(st: TurnState) -> TurnState:
    """Goal model upkeep: win signatures, per-level novelty facts. No model call."""
    st.session.refresh_goal_facts()
    return st


def mode(st: TurnState) -> TurnState:
    s = st.session
    st.disagreement = s.hypotheses_disagree()
    inp = ModeInputs(
        turn=st.turn, ledger_len=len(s.ledger), level_actions=s.level_actions, new_level=s.new_level_pending,
        novelty=st.novelty, last_violated=s.last_violated, last_unexpected=s.last_unexpected, doom=s.doom_detected(),
        disagreement=st.disagreement, confidences=s.confidences, plan_validated=st.plan_validated,
        informative=bool(getattr(s, "last_violation_informative", False)),
        consecutive_recover=s.consecutive_recover,
    )
    st.mode, st.mode_reason = mode_of(inp)
    # Published now, not at the end of the turn: the JOB line in every prompt this turn reads it,
    # and reading last turn's reason beside this turn's mode is worse than reading nothing.
    s._mode_reason = st.mode_reason
    st.roles = list(ROLE_PLAN[st.mode])
    if s.config_flag("merged_turn") and st.mode in (Mode.EXPLORE, Mode.VERIFY):
        st.roles = [r for r in st.roles if r[0] != "interpret"]
    # The solver can only search for a goal it has been given. If there is none, or the last one
    # was unreachable and the body is now identified, spend one small call to set one.
    if (st.mode in (Mode.EXPLORE, Mode.VERIFY) and s.tracker.self_tid
            and not any(r[0] == "goal" for r in st.roles)
            and (s.symbolic_goal is None
                 or s.last_plan.status in ("bad_goal", "goal_reached"))):
        # A goal the ledger already supports is free; only ask the model when it does not.
        if s.adopt_harness_goal("no goal the solver could use"):
            # Solve it now, in this turn. The plan status was computed before the goal existed, so
            # without this the harness adopts a goal and then spends the turn as though it had
            # none -- which is what "plan is the blocking channel" meant on a run where a goal had
            # just been set two lines earlier.
            st.plan_validated = s.plan_ready()
            if st.plan_validated:
                st.mode, st.mode_reason = Mode.EXECUTE, "a goal the harness could set and solve"
                s._mode_reason = st.mode_reason
                st.roles = []
        else:
            st.roles = [("goal", 300)] + st.roles
    # The planner runs at near-zero budget: with a validated harness plan, EXECUTE spends no
    # request at all unless --model-executes asks for the model to commit it.
    if st.mode == Mode.EXECUTE and st.plan_validated and not s.config_flag("model_executes"):
        st.roles = []
    # When exploring has been buying one or two actions a turn, the harness can plan a better use
    # of the same turn itself and spend nothing on it. It only takes the turn after the model has
    # had two to do better, and only while there is no goal worth solving.
    st.harness_explores = (st.mode in (Mode.EXPLORE, Mode.VERIFY) and not st.plan_validated
                           and s.starving() and not s.config_flag("no_harness_explore"))
    # Before anything else that costs a request: if a control has never been read, read it. A
    # direction the harness cannot say anything about is a direction the planner does not have,
    # and the model will not spend its turns fixing that on its own.
    st.harness_measures = (not st.plan_validated
                           and bool(s.control.unmeasured(s.available))
                           and not s.config_flag("no_harness_measure"))
    if st.harness_measures or st.harness_explores:
        st.roles = []
    return st


def propose(st: TurnState) -> TurnState:
    """Run the roles for this mode; every output is validated before it is applied."""
    s = st.session
    for role, max_tokens in st.roles:
        out = s.call_role(role, st.mode, max_tokens)
        if out is None:
            st.errors.append(f"{role}: no valid output")
            if role in ("decide", "decide_lean"):
                break
            continue
        st.outputs[role] = out
        try:
            s.apply_role_output(role, out, st)
            if role == "goal" and s.symbolic_goal:
                # Solve it now, in the turn it was set. The plan status was computed in mode(),
                # before this role had run, so without this the harness takes a goal from the
                # model and then spends the rest of the turn as though it had none -- and does not
                # try it until the next turn. The same fault was fixed for the harness's own goals
                # earlier; this is the other half of it.
                st.plan_validated = s.plan_ready()
        except Exception as exc:  # noqa: BLE001
            # Applying an interpretation must never end the game. A re86 run died on a KeyError
            # thirty-two actions in and lost every level it might still have reached; the reply
            # that caused it was worth dropping, the run was not.
            st.errors.append(f"{role}: could not be applied ({exc!r})"[:160])
            s._log_row({"kind": "role_unapplied", "turn": st.turn, "role": role,
                        "error": repr(exc)[:200]})
            st.outputs.pop(role, None)
    return st


def commit(st: TurnState) -> TurnState:
    """Queue the batch with one prediction per step; a missing decide output falls back to one probe."""
    s = st.session
    decide = st.outputs.get("decide") or st.outputs.get("decide_lean")
    if decide is None and getattr(st, "harness_measures", False) and \
            s.queue_measurement("a control the harness has never been able to read"):
        pass
    elif decide is None and getattr(st, "harness_explores", False) and \
            s.queue_explore("the last turns bought one or two actions each"):
        pass
    elif decide is None and st.mode == Mode.EXECUTE and st.plan_validated and not st.roles:
        s.queue_plan()
    elif decide is None and s.last_plan.status in ("found", "partial"):
        s.queue_plan()   # better than a blind probe: these steps were replayed before committing
    elif decide is None and s.queue_explore("no decide output; " + ("; ".join(st.errors))[:80]):
        pass
    elif decide is None:
        s.queue_fallback_probe(reason="; ".join(st.errors) or "no decide output")
    else:
        s.queue_batch(decide, st.mode)
    s.log_turn(st)
    return st


def build_graph() -> Graph:
    g = Graph()
    g.add("recall", recall, "world")
    g.add("world", world, "goal")
    g.add("goal", goal, "mode")
    g.add("mode", mode, "propose")
    g.add("propose", propose, "commit")
    g.add("commit", commit, END)
    g.check()
    return g
