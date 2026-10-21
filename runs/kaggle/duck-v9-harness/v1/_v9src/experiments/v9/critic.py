"""Critic mapping: the model classifies a failure; the harness decides what happens next.

The eight kinds are the plan's: perception, identity, control, dynamics, hidden_state, temporal,
goal, plan. Each maps to a deterministic follow-up so a small model never rewrites a good world
model for the wrong reason.
"""

from __future__ import annotations

FOLLOW_UP = {
    "perception": "retrack",        # re-run the tracker with a wider match radius
    "identity": "retrack",
    "control": "demote_control",    # halve the control-map confidence for the action; force a single-key probe
    "dynamics": "open_question",    # a rule is wrong: quarantine the violated rule, open a question
    "hidden_state": "open_question",
    "temporal": "open_question",
    "goal": "regoal",               # goal reasoner: re-rank goal hypotheses
    "plan": "replan",               # re-plan only, no model call
}


def follow_up(kind: str, fix_hint: str = "none") -> str:
    """The harness's own mapping wins; the model's `fix` is only a tie-breaker when it agrees."""
    return FOLLOW_UP.get(kind, "open_question")
