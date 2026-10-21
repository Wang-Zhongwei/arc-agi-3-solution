"""Harness-owned memory: hypotheses, open questions, goal hypotheses, model-authored slots,
the cross-level level model, and the on-disk cross-game store.

Facts live in the ledger (ledger.py); rules in rules.py. Everything here is theory or plan, and
every write goes through a method that stamps provenance (turn, level) and enforces the caps.
The model proposes; these methods decide.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .schemas import MAX_ENTRIES_PER_SLOT, MAX_SLOTS_PER_GAME, SLOT_FIELD_TYPES, SLOT_KINDS, SLOT_SCOPES
from .types import GoalHypothesis, Hypothesis, MemorySlot, Question


class Memory:
    def __init__(self) -> None:
        self.hypotheses: dict[str, Hypothesis] = {}
        self.questions: dict[str, Question] = {}
        self.goals: dict[str, GoalHypothesis] = {}
        self.slots: dict[str, MemorySlot] = {}
        self.labels: dict[str, tuple[str, str, float]] = {}   # tid -> (label, role, p)
        self.memory_plan: dict[str, set[str]] = {"track": set(), "watch": set(), "ignore": set()}
        self.level_log: list[dict[str, Any]] = []             # harness-written entry per cleared level
        self.tried: list[dict[str, Any]] = []                  # batches that changed nothing
        self.priors: list[dict[str, Any]] = []                 # cross-game patterns loaded at ORIENT
        self._counter = {"h": 0, "q": 0, "g": 0}

    # -- ids -----------------------------------------------------------------------------------

    def _next(self, prefix: str) -> str:
        self._counter[prefix] += 1
        return f"{prefix}{self._counter[prefix]}"

    # -- hypotheses / questions / goals --------------------------------------------------------

    def add_hypothesis(self, claim: str, discriminator: str, turn: int) -> Hypothesis:
        claim = " ".join(claim.split())[:160]
        for h in self.hypotheses.values():
            if h.status == "open" and h.claim.lower() == claim.lower():
                return h
        h = Hypothesis(hid=self._next("h"), claim=claim, discriminator=" ".join(discriminator.split())[:160], created_turn=turn)
        self.hypotheses[h.hid] = h
        return h

    def open_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.status == "open"]

    def add_question(self, text: str, turn: int) -> Question:
        text = " ".join(text.split())[:120]
        for q in self.questions.values():
            if q.status == "open" and q.text.lower() == text.lower():
                return q
        q = Question(qid=self._next("q"), text=text, created_turn=turn)
        self.questions[q.qid] = q
        return q

    def answer_question(self, qid: str, turn: int, answer: str = "") -> None:
        q = self.questions.get(qid)
        if q is not None and q.status == "open":
            q.status = "answered"
            q.answered_turn = turn
            q.answer = answer[:160]

    def open_questions(self) -> list[Question]:
        return [q for q in self.questions.values() if q.status == "open"]

    def add_goal(self, text: str, p: float) -> GoalHypothesis:
        text = " ".join(text.split())[:160]
        for g in self.goals.values():
            if g.text.lower() == text.lower():
                g.p = max(0.0, min(1.0, float(p)))
                return g
        g = GoalHypothesis(gid=self._next("g"), text=text, p=max(0.0, min(1.0, float(p))))
        self.goals[g.gid] = g
        return g

    def top_goals(self, k: int = 3) -> list[GoalHypothesis]:
        return sorted((g for g in self.goals.values() if g.status == "open"), key=lambda g: -g.p)[:k]

    # -- labels and the memory plan (model proposes, harness decides) ----------------------------

    def set_label(self, tid: str, label: str, role: str, p: float) -> None:
        prev = self.labels.get(tid)
        # A label only moves when the new proposal is at least as confident as the standing one.
        if prev is None or float(p) >= prev[2] - 0.05:
            self.labels[tid] = (label[:24], role, max(0.0, min(1.0, float(p))))

    def plan_move(self, tid: str, to: str) -> None:
        for bucket in self.memory_plan.values():
            bucket.discard(tid)
        self.memory_plan[to].add(tid)

    # -- model-authored slots ------------------------------------------------------------------

    def create_slot(self, spec: dict[str, Any], turn: int) -> Optional[str]:
        """Returns a rejection reason or None on success."""
        name = str(spec["name"]).strip()
        if not name or name in self.slots:
            return f"slot {name!r} exists"
        if len(self.slots) >= MAX_SLOTS_PER_GAME:
            return "slot cap reached"
        if spec["kind"] not in SLOT_KINDS or spec["scope"] not in SLOT_SCOPES:
            return "bad kind or scope"
        fields = {}
        for f in spec["fields"]:
            if f["type"] not in SLOT_FIELD_TYPES:
                return f"bad field type {f['type']}"
            fields[str(f["name"])[:24]] = f["type"]
        if not fields:
            return "a slot needs at least one field"
        self.slots[name] = MemorySlot(name=name, kind=spec["kind"], fields=fields, scope=spec["scope"],
                                      reason=str(spec.get("reason", ""))[:160], created_turn=turn)
        return None

    def add_entry(self, slot_name: str, values: dict[str, Any], *, turn: int, level: int,
                  ledger_len: int, pin: bool = False, tids: set[str] = frozenset()) -> Optional[str]:
        slot = self.slots.get(slot_name)
        if slot is None:
            return f"unknown slot {slot_name!r}"
        entry: dict[str, Any] = {}
        for fname, ftype in slot.fields.items():
            if fname not in values:
                return f"entry missing field {fname!r}"
            v = values[fname]
            try:
                if ftype == "int":
                    v = int(v)
                elif ftype == "bool":
                    v = bool(v)
                elif ftype == "step":
                    v = int(v)
                    if not 0 <= v < max(1, ledger_len):
                        return f"field {fname} cites step {v} outside the ledger"
                elif ftype == "tid":
                    v = str(v)
                    if v not in tids:
                        return f"field {fname} names unknown track {v!r}"
                elif ftype == "colour":
                    v = str(v)[:1]
                else:
                    v = str(v)[:120]
            except (TypeError, ValueError):
                return f"field {fname} has the wrong type"
            entry[fname] = v
        if slot.kind in ("transition", "rule") and not any(t == "step" for t in slot.fields.values()):
            return f"a {slot.kind}-kind slot must have a step field citing the ledger"
        entry["_turn"] = turn
        entry["_level"] = level
        entry["_pin"] = bool(pin)
        slot.entries.append(entry)
        if len(slot.entries) > MAX_ENTRIES_PER_SLOT:
            # Oldest unpinned first.
            for i, e in enumerate(slot.entries):
                if not e.get("_pin"):
                    slot.entries.pop(i)
                    break
            else:
                slot.entries.pop(0)
        return None

    # -- level boundary --------------------------------------------------------------------------

    def on_level_clear(self, level: int, turn: int, facts: dict[str, Any]) -> None:
        """Harness-written level entry: zero requests, never wiped."""
        self.level_log.append({"level": level, "turn": turn, **facts})
        for slot in self.slots.values():
            if slot.scope == "level":
                slot.entries = [e for e in slot.entries if e.get("_pin")]

    def note_tried(self, description: str, step: int, level: int) -> None:
        for t in self.tried:
            if t["what"] == description and t["level"] == level:
                t["count"] += 1
                t["steps"].append(step)
                return
        self.tried.append({"what": description, "level": level, "count": 1, "steps": [step]})

    # -- rendering ------------------------------------------------------------------------------

    def render_questions(self, k: int = 4) -> str:
        qs = self.open_questions()[-k:]
        return "; ".join(f"{q.qid}: {q.text}" for q in qs) if qs else "none"

    def render_settled(self, k: int = 3) -> str:
        """Hypotheses the harness has already decided by running their own discriminating test.
        Shown so the model stops re-proposing them and can build on what is now known."""
        rows = [h for h in self.hypotheses.values() if h.status in ("supported", "refuted")][-k:]
        return "; ".join(
            f"{'confirmed' if h.status == 'supported' else 'ruled out'}: {h.claim}" for h in rows
        ) if rows else ""

    def render_hypotheses(self, k: int = 3) -> str:
        hs = self.open_hypotheses()[-k:]
        return "; ".join(f"{h.hid}: {h.claim} [test: {h.discriminator}]" for h in hs) if hs else "none"

    def render_goals(self) -> str:
        gs = self.top_goals()
        return "; ".join(f"{g.gid} p={g.p:.2f}: {g.text}" for g in gs) if gs else "none yet"

    def render_slots(self, max_chars: int = 400) -> str:
        if not self.slots:
            return ""
        parts = []
        for s in self.slots.values():
            tail = s.entries[-3:]
            rows = "; ".join(", ".join(f"{k}={v}" for k, v in e.items() if not k.startswith("_")) for e in tail)
            parts.append(f"{s.name} ({s.kind}, {s.scope}, {len(s.entries)} entries): {rows or 'empty'}")
        text = " | ".join(parts)
        return text[:max_chars]

    def render_level_log(self, current_level: int) -> str:
        parts = []
        clears = [e for e in self.level_log if e.get("summary")]
        if clears:
            last = clears[-1]
            parts.append(f"L{last['level']} ended: {last['summary'][:200]}")
        notes = [e for e in self.level_log if e.get("orientation") and e.get("level") == current_level]
        if notes:
            parts.append(f"your orientation note: {notes[-1]['orientation'][:200]}")
        return " | ".join(parts)

    def render_tried(self, level: int, k: int = 3) -> str:
        rows = [t for t in self.tried if t["level"] == level][-k:]
        return "; ".join(f"{t['what']} x{t['count']} (steps {','.join(map(str, t['steps'][-3:]))})" for t in rows) if rows else ""

    def from_json(self, d: dict[str, Any]) -> None:
        """Restore theory and plan from a dump (a previous run of the same game): used to seed a
        per-level test with what the harness knew at the boundary. Ledger facts are NOT
        restored; they belong to the run that observed them."""
        for h in d.get("hypotheses", []):
            self.hypotheses[h["hid"]] = Hypothesis(**h)
        for q in d.get("questions", []):
            self.questions[q["qid"]] = Question(**q)
        for g in d.get("goals", []):
            self.goals[g["gid"]] = GoalHypothesis(**g)
        for sl in d.get("slots", []):
            self.slots[sl["name"]] = MemorySlot(**sl)
        self.labels = {k: tuple(v) for k, v in d.get("labels", {}).items()}
        self.memory_plan = {k: set(v) for k, v in d.get("memory_plan", {}).items()} or self.memory_plan
        self.level_log = list(d.get("level_log", []))
        self.tried = list(d.get("tried", []))
        self.priors = list(d.get("priors", []))
        for prefix, items in (("h", self.hypotheses), ("q", self.questions), ("g", self.goals)):
            nums = [int(k[1:]) for k in items if k[1:].isdigit()]
            self._counter[prefix] = max(nums) if nums else 0

    def to_json(self) -> dict[str, Any]:
        return {
            "hypotheses": [asdict(h) for h in self.hypotheses.values()],
            "questions": [asdict(q) for q in self.questions.values()],
            "goals": [asdict(g) for g in self.goals.values()],
            "slots": [asdict(s) for s in self.slots.values()],
            "labels": self.labels, "memory_plan": {k: sorted(v) for k, v in self.memory_plan.items()},
            "level_log": self.level_log, "tried": self.tried, "priors": self.priors,
        }


# --- cross-game store ------------------------------------------------------------------------


@dataclass
class GameStore:
    """Patterns the model proposed across games, keyed by control-signature family.

    A prior is a hypothesis for a later game: shown at ORIENT with confidence halved and
    re-scored against that game's ledger before it can become a rule. Accepted only when the
    evidence cites at least two games the store has seen.
    """
    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    def load(self) -> "GameStore":
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {}
        self.data.setdefault("games", {})
        self.data.setdefault("patterns", [])
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str))
        os.replace(tmp, self.path)

    def known_games(self) -> set[str]:
        return set(self.data.get("games", {}))

    def note_game(self, game_id: str, family: str, summary: dict[str, Any]) -> None:
        self.data.setdefault("games", {})[game_id] = {"family": family, **summary}

    def add_pattern(self, pattern: str, evidence_games: list[str], confidence: float, *, family: str) -> Optional[str]:
        games = [g for g in evidence_games if g in self.known_games()]
        if len(games) < 2:
            return "needs two known games"
        self.data.setdefault("patterns", []).append({"pattern": pattern[:200], "evidence": games,
                                                     "confidence": max(0.0, min(1.0, float(confidence))), "family": family})
        return None

    def priors_for(self, family: str, k: int = 3) -> list[dict[str, Any]]:
        rows = [p for p in self.data.get("patterns", []) if p.get("family") == family]
        rows.sort(key=lambda p: -p["confidence"])
        return [{**p, "confidence": p["confidence"] / 2.0} for p in rows[:k]]
