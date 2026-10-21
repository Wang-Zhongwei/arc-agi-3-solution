"""Mode controller and commit gate: pure functions of harness state.

ORIENT   nothing known yet, a new level, or real novelty
RECOVER  a trusted prediction just failed, or a doom loop
EXECUTE  the commit gate passes (every blocking channel confident, plan validated)
VERIFY   two hypotheses disagree on an available action, or a channel blocks EXECUTE
EXPLORE  otherwise: important uncertainty, chase the open questions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .types import Confidences, Mode

GATE = {"perception": 0.70, "control": 0.80, "world_model": 0.70, "goal": 0.60}
NOVELTY_THRESHOLD = 0.6


@dataclass
class ModeInputs:
    turn: int
    ledger_len: int
    level_actions: int
    new_level: bool
    novelty: float
    last_violated: bool
    last_unexpected: bool
    doom: bool
    disagreement: bool
    confidences: Confidences
    plan_validated: bool
    # Whether the last miss contradicted something the control map had settled. Recorded because
    # it is worth knowing; NOT used to gate RECOVER. Gating on it was tried and measured as an
    # exact no-op -- every miss in the run was on a settled action -- and it broke the rule that a
    # surprise without a plan recovers. Two thirds of all model calls still go to RECOVER, so the
    # cost is real, but this was not the way to cut it.
    informative: bool = False
    consecutive_recover: int = 0


def commit_gate(conf: Confidences, plan_validated: bool) -> tuple[bool, Optional[str]]:
    """Returns (passes, lowest blocking channel by name).

    A validated plan is itself the evidence the channels are proxies for: it exists only when the
    body is identified, the actions in it have certified effects, the goal compiles, and canonical
    replay reaches that goal. Requiring the channels *as well* double-counts, and did: on ls20
    level 2 the harness held a validated plan at plan=0.9 for turn after turn while control sat at
    0.5 and the gate refused. Perception still has to hold, because a plan drawn over a misread
    board is worth nothing.
    """
    if plan_validated and conf.perception >= GATE["perception"]:
        return True, None
    worst: Optional[tuple[float, str]] = None
    for name, threshold in GATE.items():
        value = getattr(conf, name)
        if value < threshold:
            margin = value - threshold
            if worst is None or margin < worst[0]:
                worst = (margin, name)
    if worst is not None:
        return False, worst[1]
    if not plan_validated:
        return False, "plan"
    return True, None


def mode_of(inp: ModeInputs) -> tuple[Mode, str]:
    """Returns (mode, one-line reason)."""
    if inp.ledger_len == 0 or inp.new_level:
        return Mode.ORIENT, "no transitions yet" if inp.ledger_len == 0 else "new level"
    if inp.novelty > NOVELTY_THRESHOLD:
        return Mode.ORIENT, f"novelty {inp.novelty:.2f}"
    if inp.doom:
        return Mode.RECOVER, "doom loop: same batch, same result"
    # A validated plan outranks a recent surprise. The plan is recomputed every turn from the
    # post-surprise world model and confirmed by canonical replay, so a model that can still solve
    # its own goal is not a broken model. Checking RECOVER first meant that on ls20 level 2 the
    # harness held a validated four-step plan at 0.9 and spent the turn on one key instead.
    ok, blocking = commit_gate(inp.confidences, inp.plan_validated)
    if ok and inp.plan_validated:
        return Mode.EXECUTE, "a validated plan for the goal you set"
    # A violated claim is a wrong model and worth stopping for. An unexpected change is not: the
    # coverage term says predictions account for about 15% of what moves in a frame, so something
    # unclaimed changes on nearly every step, and treating that as a surprise sent the harness to
    # RECOVER on 38 of 67 turns -- 57% of a run, at three model calls each -- while the batch it
    # bought averaged 3.25 actions. A harness that investigates more often than it plays cannot
    # outpace one that plays every turn, and the recorded baseline that clears ls20 level 1 does
    # exactly that. Unexplained change still reaches the model: it is in the outcome line, and it
    # still drives the world-model channel down.
    if inp.last_violated and inp.consecutive_recover < 2:
        return Mode.RECOVER, "last prediction violated"
    if ok:
        return Mode.EXECUTE, "commit gate passed"
    if inp.disagreement:
        return Mode.VERIFY, "hypotheses disagree on an available action"
    if blocking in ("goal", "plan") and inp.confidences.world_model >= GATE["world_model"] and inp.confidences.control >= GATE["control"]:
        return Mode.VERIFY, f"{blocking} is the blocking channel"
    return Mode.EXPLORE, f"{blocking} is the blocking channel" if blocking else "uncertain"


# Per-mode role sequence and budgets (max_tokens); thinking off everywhere by default.
# Token budgets are generous because a truncated answer is not a short answer: it is a lost call
# plus a repair round that re-sends the whole prompt. On the local GPU those budgets bought turns
# and were worth defending; through OpenRouter an extra 800 output tokens costs about a quarter of
# a cent, and the doubled call costs a turn. Every role here was seen hitting its old ceiling.
ROLE_PLAN: dict[Mode, list[tuple[str, int]]] = {
    # decide_lean, not decide, on both of the turns that used to run the full form. The full
    # memory patch has twelve required arrays and the model fills whatever room the grammar
    # offers: measured live on ls20 it came back finish=length at 1400 on BOTH attempts of one
    # ORIENT turn -- 631 s then 1266 s, thirty-one minutes, for nothing, on a level's only
    # orientation turn. The lean form carries everything the harness consumes (batch, step
    # predictions, confidences, labels, rules, hypotheses, questions, disputes, promote/demote,
    # goals); what it drops is the model-authored slot machinery, which no decision reads.
    # DECIDE_SCHEMA is still built and validated, behind --full-patch, for a later A/B.
    Mode.ORIENT: [("orient", 2400), ("goal", 400), ("decide_lean", 2400)],
    # decide_lean is the workhorse: it runs in EXPLORE and VERIFY, which are most turns. Live on
    # the GB10 it hit exactly 1700 twice, and a truncated answer there costs 760 seconds and then
    # a repair round that re-sends the whole prompt, so the ceiling was buying nothing and
    # spending a quarter of an hour each time it bit.
    Mode.EXPLORE: [("interpret", 800), ("decide_lean", 2400)],
    Mode.VERIFY: [("interpret", 800), ("decide_lean", 2400)],
    Mode.EXECUTE: [("decide_lean", 1500)],
    Mode.RECOVER: [("critic", 700), ("hypothesis", 1100), ("decide_lean", 2400)],
}
