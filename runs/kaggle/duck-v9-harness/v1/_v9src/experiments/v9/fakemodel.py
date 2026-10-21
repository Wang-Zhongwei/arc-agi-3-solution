"""A scripted stand-in for the model: exercises the whole loop against the real engine with no
LLM. It reads the live session (bound by the session) and answers every role with valid JSON.

Policy: press each available action twice (orientation), then random-walk with the control
map's expected shifts as predictions. Every 7th decide step it deliberately predicts the wrong
shift, so batch halting is exercised on purpose. Deterministic under its seed.
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable, Optional

from .client import CallResult


class FakeModel:
    def __init__(self, seed: int = 0, wrong_every: int = 7) -> None:
        self.rng = random.Random(seed)
        self.wrong_every = wrong_every
        self.session: Any = None
        self.calls = 0
        self.total_usage = {"in": 0, "out": 0}
        self.thinking = False
        self.model = "fake"

    def bind(self, session: Any) -> None:
        self.session = session

    # -- role answers -----------------------------------------------------------------------------

    def _decide(self, vocab) -> dict[str, Any]:
        s = self.session
        actions = sorted(a for a in vocab.actions if a != "RESET") or ["ACTION1"]
        game_actions = [a for a in actions if a != "ACTION6"] or actions
        n_seen = len(s.ledger)
        if n_seen < 2 * len(game_actions):
            action = game_actions[n_seen % len(game_actions)]
            batch = [{"action": action, "x": None, "y": None}]
        else:
            k = self.rng.choice([1, 2, 3, 4])
            batch = [{"action": self.rng.choice(game_actions), "x": None, "y": None} for _ in range(k)]
        preds = []
        self_tid = s.tracker.self_tid
        for i, a in enumerate(batch, start=1):
            shift = s.control.expected_shift(a["action"]) if hasattr(s.control, "expected_shift") else None
            expect = []
            if self_tid and shift is not None:
                dr, dc = shift
                if self.calls % self.wrong_every == 0:
                    dr, dc = dr + 5, dc + 5  # deliberate misprediction
                expect.append({"var": f"obj:{self_tid}.pos", "op": "delta", "value": f"({dr},{dc})", "p": 0.8})
            else:
                expect.append({"var": "cells", "op": "changed", "value": "", "p": 0.6})
            preds.append({"step": i, "expect": expect, "summary": f"probe {a['action']}"})
        return {
            "mode_ack": s.current_mode.value if s.current_mode else "EXPLORE",
            "hypothesis_tested": "none",
            "batch": batch,
            "step_predictions": preds,
            "memory_patch": {"promote": [], "demote": [], "new_rules": [], "new_hypotheses": [], "disputes": [], "answered": [],
                             "new_questions": [], "labels": [], "goals": [], "new_slots": [], "entries": [],
                             "new_patterns": []},
            "confidence": {"perception": 0.6, "control": 0.5, "world_model": 0.4, "goal": 0.2, "plan": 0.2},
        }

    def _decide_lean(self, vocab) -> dict[str, Any]:
        """The routine-turn form: the same decision with the memory-restructuring fields absent."""
        full = self._decide(vocab)
        mp = full["memory_patch"]
        full["memory_patch"] = {k: mp[k] for k in ("new_rules", "new_hypotheses", "answered",
                                                   "new_questions", "labels")}
        return full

    def _goal(self, vocab) -> dict[str, Any]:
        """A symbolic goal over whatever class is most common on the board, with a simpler fallback."""
        counts: dict[str, int] = {}
        for t in self.session.tracker.live():
            if t.blob is None or t.tid == self.session.tracker.self_tid:
                continue
            from v9.symbols import shape_name
            from v9.types import colour_letter
            key = f"{colour_letter(t.colour)} {shape_name(t.blob.cells, t.last_bbox)}"
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return {"predicate": "touched", "target": "obj:t1", "row": None, "col": None,
                    "op": "<=", "n": None, "onto": "", "why": "scripted: nothing to aim at", "p": 0.1}
        best = max(counts, key=lambda k: (counts[k], k))
        return {"predicate": "all_touched", "target": f"class:{best}", "row": None, "col": None,
                "op": "<=", "n": None, "onto": "", "why": f"scripted: reach every {best}", "p": 0.4}

    def _interpret(self, vocab) -> dict[str, Any]:
        tids = sorted(vocab.tids)[:4]
        return {"labels": [{"tid": t, "label": "thing", "role": "unknown", "p": 0.3} for t in tids],
                "salient": tids[:2], "new_questions": [], "note": "scripted"}

    def _orient(self, vocab) -> dict[str, Any]:
        tids = sorted(vocab.tids)[:4]
        acts = sorted(a for a in vocab.actions if a != "RESET")[:4]
        return {"control_model": [{"action": a, "effect": "unknown", "p": 0.3} for a in acts],
                "entities": [{"tid": t, "label": "thing", "role": "unknown", "p": 0.3} for t in tids],
                "memory_plan": {"track": tids[:2], "watch": tids[2:4], "ignore": []},
                "questions": ["which action moves the controlled object?"],
                "goal_hypotheses": [{"text": "reach the other object", "p": 0.3}], "summary": "scripted orientation"}

    def _critic(self, vocab) -> dict[str, Any]:
        return {"kind": "control", "explanation": "scripted: the expected shift was wrong", "fix": "demote_control"}

    def _hypothesis(self, vocab) -> dict[str, Any]:
        acts = sorted(a for a in vocab.actions if a not in ("RESET", "ACTION6")) or ["ACTION1"]
        return {"hypotheses": [
            {"claim": "the action moves the object one cell", "discriminator": "press it once",
             "test_action": acts[0], "predicted_var": "cells", "predicted_op": "changed", "predicted_value": ""},
            {"claim": "the action does nothing here", "discriminator": "press it once",
             "test_action": acts[0], "predicted_var": "cells", "predicted_op": "unchanged", "predicted_value": ""},
        ]}

    def call_json(self, *, role: str, system: str, user: str, schema: dict[str, Any], max_tokens: int,
                  thinking: Optional[bool] = None, validate: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
                  on_attempt: Optional[Callable[[dict[str, Any]], None]] = None) -> CallResult:
        self.calls += 1
        vocab = self.session.current_vocab
        obj = {"decide": self._decide, "decide_lean": self._decide_lean, "interpret": self._interpret,
               "orient": self._orient, "critic": self._critic, "hypothesis": self._hypothesis,
               "goal": self._goal}[role](vocab)
        error = validate(obj) if validate else None
        text = json.dumps(obj)
        usage = {"in": len(user) // 4, "out": len(text) // 4}
        self.total_usage["in"] += usage["in"]
        self.total_usage["out"] += usage["out"]
        if on_attempt:
            on_attempt({"role": role, "attempt": 1, "usage": usage, "finish": "stop", "reasoning": "", "text": text,
                        "error": error, "latency_s": 0.0})
        return CallResult(obj=None if error else obj, text=text, reasoning="", usage=usage, latency_s=0.0,
                          finish="stop", error=error)
