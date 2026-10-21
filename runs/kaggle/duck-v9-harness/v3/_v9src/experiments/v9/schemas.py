"""JSON schemas and validators for every model role.

Every schema is a flat, hand-written dict (no $ref, no oneOf) so the server's grammar engine
accepts it, and every field is required with additionalProperties false so the model can only
fill slots the harness defined. Nested free structures (a rule's `when`/`then`, a slot entry's
values) travel as JSON strings and are parsed and re-validated here.

Rejection policy (plan): unknown entity -> reject; confidence outside [0,1] -> reject; rule with
no evidence -> reject (checked later by rules.RuleStore against the ledger); any attempt to
write the ledger is unrepresentable (no such field). A batch longer than MAX_BATCH is truncated,
not rejected (NOOA's commit rule: a rejected round-trip costs a whole request).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import ACTION_NAMES, RULE_TYPES, Mode

# --- what one model call may ask for -------------------------------------------------------
# Every ceiling the model can push against, in one place. They exist because a call is the scarce
# resource: on the GB10 one costs minutes, and an answer that runs long is not a fuller answer, it
# is a truncated one plus a repair round. The time budget in client.budget_for trims these further
# when the endpoint is slow.
MAX_BATCH = 12              # actions committed from a single decide
MAX_PLAN_STEPS = 20         # steps taken from one harness plan (schemas.MAX_PLAN_BATCH)
MAX_ORIENT_ENTITIES = 8     # objects the orientation call may name
MAX_ORIENT_QUESTIONS = 3    # questions it may open
MAX_STEP_PREDICTIONS = 6    # steps of a batch it is worth grading
# A harness plan is replayed to the goal before it is committed and every step of it carries a
# checkable claim, so it halts itself the moment the world model is wrong. The model's own batch
# has neither property and stays at MAX_BATCH. Turns are the scarce resource here (about thirty
# per run at ten minutes each) while actions are not (five times the human baseline, 615 on ls20
# level 2), so a plan the harness trusts is allowed to spend more of the cheap one.
MAX_PLAN_BATCH = MAX_PLAN_STEPS
MAX_SLOTS_PER_GAME = 8
MAX_ENTRIES_PER_SLOT = 32
SLOT_KINDS = ("object", "transition", "rule", "question", "goal")
SLOT_FIELD_TYPES = ("int", "str", "bool", "tid", "step", "colour")
SLOT_SCOPES = ("level", "game", "cross_game")
ENTITY_ROLES = ("player", "wall", "floor", "collectable", "hud", "goal", "hazard", "switch", "door",
                "counter", "decor", "unknown")
CRITIC_KINDS = ("perception", "identity", "control", "dynamics", "hidden_state", "temporal", "goal", "plan")
CRITIC_FIXES = ("retrack", "demote_control", "open_question", "replan", "regoal", "none")
CLAIM_OPS = ("delta", "equals", "unchanged", "changed")


def _s(**kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "string"}
    d.update(kw)
    return d


def _arr(items: dict[str, Any], *, max_items: Optional[int] = None,
         min_items: Optional[int] = None) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "array", "items": items}
    if max_items is not None:
        d["maxItems"] = max_items
    if min_items is not None:
        d["minItems"] = min_items
    return d


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


P01 = {"type": "number", "minimum": 0, "maximum": 1}
INT_OR_NULL = {"type": ["integer", "null"]}
# Grammar-enforced ids: the server's grammar engine honours `pattern`, so the model cannot emit
# free text where a track / hypothesis / question id belongs (seen live: "position of " as a tid).
TID = _s(pattern="^t[0-9]{1,4}$", maxLength=6)
HID_OR_NONE = _s(pattern="^(none|h[0-9]{1,4})$", maxLength=6)
QID = _s(pattern="^q[0-9]{1,4}$", maxLength=6)
ANY_ID = _s(pattern="^[thrq][0-9]{1,4}$", maxLength=6)
# Variables a claim may name: exactly the scalar registry plus per-track attributes.
VAR = _s(pattern=r"^(cells|objs_total|level|state|step|veil_frac|(cells|objs|consumed|remaining)\[[A-Za-z0-9_]{1,12}\]|hud\[[a-z]+\]\.(filled|len)|pos\[(self|t[0-9]{1,4})\]\.(r|c)|obj:t[0-9]{1,4}\.(pos|colour|gone))$", maxLength=40)

ACTION_SCHEMA = _obj({"action": _s(enum=list(ACTION_NAMES)), "x": INT_OR_NULL, "y": INT_OR_NULL})
CLAIM_SCHEMA = _obj({"var": VAR, "op": _s(enum=list(CLAIM_OPS)), "value": _s(maxLength=32), "p": P01})
STEP_PREDICTION_SCHEMA = _obj({"step": {"type": "integer", "minimum": 1}, "expect": _arr(CLAIM_SCHEMA, max_items=6),
                               "summary": _s(maxLength=100)})
NEW_RULE_SCHEMA = _obj({"type": _s(enum=list(RULE_TYPES)), "when": _s(maxLength=400), "then": _s(maxLength=300),
                        "because": _arr({"type": "integer"}, max_items=8), "text": _s(maxLength=120)})
NEW_HYPOTHESIS_SCHEMA = _obj({"claim": _s(maxLength=160), "discriminator": _s(maxLength=160)})
LABEL_SCHEMA = _obj({"tid": TID, "label": _s(maxLength=24), "role": _s(enum=list(ENTITY_ROLES)), "p": P01})
GOAL_SCHEMA = _obj({"text": _s(maxLength=160), "p": P01})
SLOT_FIELD_SCHEMA = _obj({"name": _s(maxLength=24), "type": _s(enum=list(SLOT_FIELD_TYPES))})
NEW_SLOT_SCHEMA = _obj({"kind": _s(enum=list(SLOT_KINDS)), "name": _s(maxLength=24),
                        "fields": _arr(SLOT_FIELD_SCHEMA, max_items=6), "scope": _s(enum=list(SLOT_SCOPES)),
                        "reason": _s(maxLength=160)})
ENTRY_SCHEMA = _obj({"slot": _s(maxLength=24), "values": _s(maxLength=400), "pin": {"type": "boolean"}})
PATTERN_SCHEMA = _obj({"pattern": _s(maxLength=200), "evidence_games": _arr(_s(maxLength=16), max_items=6), "confidence": P01})

MEMORY_PATCH_SCHEMA = _obj({
    "promote": _arr(ANY_ID, max_items=8),
    "demote": _arr(ANY_ID, max_items=8),
    "new_rules": _arr(NEW_RULE_SCHEMA, max_items=3),
    "new_hypotheses": _arr(NEW_HYPOTHESIS_SCHEMA, max_items=3),
    # Not only adding to memory: arguing with it. Each entry names something the harness currently
    # holds -- a label, a rule, a question, a goal -- and says why the record does not support it.
    # Without this the model can only ever pile on; a memory that only grows is one nobody is
    # allowed to be wrong in, and every wrong entry stays until a level ends.
    "disputes": _arr(_obj({"target": ANY_ID, "why": _s(maxLength=160)}), max_items=3),
    "answered": _arr(QID, max_items=6),
    "new_questions": _arr(_s(maxLength=120), max_items=3),
    "labels": _arr(LABEL_SCHEMA, max_items=8),
    "goals": _arr(GOAL_SCHEMA, max_items=3),
    "new_slots": _arr(NEW_SLOT_SCHEMA, max_items=2),
    "entries": _arr(ENTRY_SCHEMA, max_items=6),
    "new_patterns": _arr(PATTERN_SCHEMA, max_items=2),
})

CONFIDENCE_SCHEMA = _obj({"perception": P01, "control": P01, "world_model": P01, "goal": P01, "plan": P01})

# A routine turn does not restructure memory, so it does not pay for the fields that do. Every
# key is required (additionalProperties is false), so an unused field costs output tokens on
# every single turn, and at ~5 tok/s an output token is a second of the run. ORIENT and RECOVER
# keep the full patch; EXPLORE, VERIFY and EXECUTE use this one.
MEMORY_PATCH_LEAN_SCHEMA = _obj({
    # promote/demote and goals are here because the harness actually consumes them (the memory
    # plan, rule retirement, the goal ranking). What is NOT here is the memory-restructuring
    # trio -- new_slots, entries, new_patterns -- which is theory the harness owns anyway and
    # which nothing in the decision path reads. Measured live on ls20: the full form came back
    # finish=length on both attempts of one ORIENT turn, 631 s then 1266 s, half an hour for
    # nothing, on the turn that is a level's only orientation. The model fills the room it is
    # given, so the room is the thing to fix.
    "promote": _arr(ANY_ID, max_items=6),
    "demote": _arr(ANY_ID, max_items=6),
    "goals": _arr(GOAL_SCHEMA, max_items=2),
    "new_rules": _arr(NEW_RULE_SCHEMA, max_items=2),
    "new_hypotheses": _arr(NEW_HYPOTHESIS_SCHEMA, max_items=2),
    # Arguing with memory is not a restructuring field: it is the only way a wrong entry ever
    # leaves, and the lean patch runs on EXPLORE, VERIFY and EXECUTE, which is most turns. Left
    # out of this schema, the dispute channel existed only on ORIENT and RECOVER -- and
    # `fill_patch` dropped it on the way through besides, so it was unreachable from here twice.
    "disputes": _arr(_obj({"target": ANY_ID, "why": _s(maxLength=160)}), max_items=2),
    "answered": _arr(QID, max_items=4),
    "new_questions": _arr(_s(maxLength=120), max_items=2),
    "labels": _arr(LABEL_SCHEMA, max_items=4),
})

DECIDE_SCHEMA = _obj({
    "mode_ack": _s(enum=[m.value for m in Mode]),
    "hypothesis_tested": HID_OR_NONE,
    # min 1 in the grammar: an empty batch cost two model calls and a fallback probe on ls20.
    # The ceiling used to be MAX_BATCH * 2 on the reasoning that a long batch is truncated
    # harness-side rather than bounced. But the grammar is also an invitation: the model filled
    # the room, ran into the token limit part-way through the JSON, and the whole answer was lost
    # rather than trimmed. Truncating something that arrived beats trimming nothing.
    "batch": _arr(ACTION_SCHEMA, max_items=MAX_BATCH, min_items=1),
    "step_predictions": _arr(STEP_PREDICTION_SCHEMA, max_items=MAX_STEP_PREDICTIONS),
    "memory_patch": MEMORY_PATCH_SCHEMA,
    "confidence": CONFIDENCE_SCHEMA,
})

INTERPRET_SCHEMA = _obj({
    "labels": _arr(LABEL_SCHEMA, max_items=8),
    "salient": _arr(TID, max_items=6),
    "new_questions": _arr(_s(maxLength=120), max_items=3),
    "note": _s(maxLength=240),
})

ORIENT_SCHEMA = _obj({
    "control_model": _arr(_obj({"action": _s(enum=list(ACTION_NAMES)), "effect": _s(maxLength=100), "p": P01}), max_items=8),
    "entities": _arr(LABEL_SCHEMA, max_items=MAX_ORIENT_ENTITIES),
    "memory_plan": _obj({"track": _arr(TID, max_items=12), "watch": _arr(TID, max_items=12),
                         "ignore": _arr(TID, max_items=12)}),
    "questions": _arr(_s(maxLength=100), max_items=MAX_ORIENT_QUESTIONS),
    "goal_hypotheses": _arr(GOAL_SCHEMA, max_items=MAX_ORIENT_QUESTIONS),
    "summary": _s(maxLength=200),
})

CRITIC_SCHEMA = _obj({
    "kind": _s(enum=list(CRITIC_KINDS)),
    "explanation": _s(maxLength=240),
    "fix": _s(enum=list(CRITIC_FIXES)),
})

HYPOTHESIS_SCHEMA = _obj({
    "hypotheses": _arr(_obj({
        "claim": _s(maxLength=160),
        "discriminator": _s(maxLength=160),
        "test_action": _s(enum=list(ACTION_NAMES)),
        "predicted_var": VAR,
        "predicted_op": _s(enum=list(CLAIM_OPS)),
        "predicted_value": _s(maxLength=32),
    }), max_items=3),
})

DECIDE_LEAN_SCHEMA = _obj({
    "mode_ack": _s(enum=[m.value for m in Mode]),
    "hypothesis_tested": HID_OR_NONE,
    # The same ceilings as the full form, for the same reason: `truncate_batch` drops everything
    # past MAX_BATCH and `validate_decide` drops predictions past step MAX_BATCH, so a grammar
    # that invites 24 actions and 12 predictions is asking for output tokens that are thrown away
    # -- and at 2 tok/s a thrown-away token is half a second of the run.
    "batch": _arr(ACTION_SCHEMA, max_items=MAX_BATCH, min_items=1),
    "step_predictions": _arr(STEP_PREDICTION_SCHEMA, max_items=MAX_STEP_PREDICTIONS),
    "memory_patch": MEMORY_PATCH_LEAN_SCHEMA,
    "confidence": CONFIDENCE_SCHEMA,
})

# The one bounded question the symbolic pipeline asks: what would winning look like, written in
# the closed goal algebra? The harness compiles the answer and searches for it; the model never
# plans and never writes code.
# A FLAT form, not JSON-in-a-string. Measured on ls20 level 2: asked for a nested term, the model
# returned the predicate name alone ("adjacent", "and", "or") on six attempts out of six, while its
# own `why` field showed it had chosen correctly ("t35 is the designated goal marker... move t42 to
# touch it"). It can pick from a list and it cannot write syntax, so the harness assembles the term.
GOAL_PREDICATES = ("all_touched", "touched", "at", "adjacent", "count", "covered")
GOAL_SCHEMA = _obj({
    "predicate": _s(enum=list(GOAL_PREDICATES)),
    "target": _s(pattern=r"^(obj:t[0-9]{1,4}|class:[A-Za-z]([ ][a-z]+)?)$", maxLength=24),
    "row": INT_OR_NULL,                 # `at` only
    "col": INT_OR_NULL,                 # `at` only
    "op": _s(enum=["<=", "<", "==", ">=", ">"]),   # `count` only
    "n": INT_OR_NULL,                   # `count` only
    # `covered` only: the kind of place every piece named by `target` has to end up on. Eight of
    # twenty-four public games end on exactly that, and none of them ends on anything the rest of
    # this form can say.
    "onto": _s(pattern=r"^(class:[A-Za-z]([ ][a-z]+)?)?$", maxLength=24),
    "why": _s(maxLength=200),
    "p": P01,
})

SCHEMAS = {"decide": DECIDE_SCHEMA, "decide_lean": DECIDE_LEAN_SCHEMA, "interpret": INTERPRET_SCHEMA,
           "orient": ORIENT_SCHEMA, "critic": CRITIC_SCHEMA, "hypothesis": HYPOTHESIS_SCHEMA,
           "goal": GOAL_SCHEMA}

FULL_PATCH_KEYS = ("promote", "demote", "new_rules", "new_hypotheses", "disputes", "answered",
                   "new_questions", "labels", "goals", "new_slots", "entries", "new_patterns")


def fill_patch(mp: dict[str, Any]) -> dict[str, Any]:
    """A lean patch is a full patch with the memory-restructuring fields empty."""
    return {key: mp.get(key, []) for key in FULL_PATCH_KEYS}


# --- validators ---------------------------------------------------------------------------------


@dataclass
class Vocab:
    """What the model is allowed to name. Built by the session from live harness state."""
    tids: set[str] = field(default_factory=set)
    actions: set[str] = field(default_factory=set)         # available this turn (+ RESET)
    scalars: set[str] = field(default_factory=set)         # registry names
    rule_ids: set[str] = field(default_factory=set)
    hypothesis_ids: set[str] = field(default_factory=set)
    question_ids: set[str] = field(default_factory=set)
    slot_names: set[str] = field(default_factory=set)
    known_games: set[str] = field(default_factory=set)
    ledger_len: int = 0

    def var_ok(self, var: str) -> bool:
        if var in self.scalars or var == "cells":
            return True
        if var.startswith("obj:"):
            body = var[4:]
            tid, _, attr = body.partition(".")
            return tid in self.tids and attr in ("pos", "colour", "gone")
        return False


def validate_shape(obj: Any, schema: dict[str, Any]) -> Optional[str]:
    """Check a reply has the fields its schema says are required, before anything reads them.

    Constrained decoding is a promise, not a guarantee. Through a gateway the request is routed to
    whichever provider will take it, and `provider.require_parameters` only asserts that the
    provider accepts `response_format`, not that it honours it. A real re86 run crashed on
    `KeyError: 'discriminator'` for a field every schema here marks required, thirty-two actions
    in, and took the whole game with it.

    List entries that are missing a required field are dropped, because one malformed hypothesis
    is not a reason to lose the others. A missing top-level field is an error, since that is the
    answer itself and there is nothing to salvage.
    """
    if not isinstance(obj, dict):
        return "the reply was not a JSON object"
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in obj:
            return f"the reply is missing the required field {name!r}"
    for name, spec in props.items():
        if name not in obj or spec.get("type") != "array":
            continue
        items = spec.get("items") or {}
        need = items.get("required")
        if not need or not isinstance(obj[name], list):
            continue
        obj[name] = [e for e in obj[name]
                     if isinstance(e, dict) and all(k in e for k in need)]
    return None


def parse_jsonish(text: str) -> Any:
    """Parse a JSON string, tolerating the Python-literal spelling the model actually emits.

    Observed live: `when` came back as "{'action_is': 'ACTION1'}" — single quotes — and strict
    json.loads rejected the whole turn over an optional field. `ast.literal_eval` evaluates
    literals only (no calls, no names), so it is safe here and costs nothing when the model gets
    it right.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(text)


def _parse_value(op: str, raw: str) -> Any:
    """'3' -> 3; '(-1,0)' -> (-1, 0); 'R' -> 'R'. Free text stays a string."""
    s = str(raw).strip()
    if op in ("changed", "unchanged"):
        return None
    if s.startswith("(") and s.endswith(")"):
        parts = [p.strip() for p in s[1:-1].split(",")]
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return s
    try:
        return int(s)
    except ValueError:
        return s


def validate_claims(expect: list[dict[str, Any]], vocab: Vocab) -> Optional[str]:
    for c in expect:
        if not vocab.var_ok(c["var"]):
            return f"unknown variable {c['var']!r}; use a name from VARS or obj:<tid>.pos/.colour/.gone"
        if not 0.0 <= float(c["p"]) <= 1.0:
            return f"probability {c['p']} outside [0,1]"
    return None


def validate_memory_patch(mp: dict[str, Any], vocab: Vocab) -> Optional[str]:
    """Reject only what would corrupt facts (a rule with bad evidence, an entry that cannot be
    typed, a pattern without evidence). Low-stakes lists are filtered in place, because a
    rejected round-trip costs a whole request (seen live: 'answered q9' with no q9)."""
    known = vocab.tids | vocab.rule_ids | vocab.hypothesis_ids
    mp["promote"] = [t for t in mp.get("promote", []) if t in known]
    mp["demote"] = [t for t in mp.get("demote", []) if t in known]
    for lab in mp.get("labels", []):
        if lab["tid"] not in vocab.tids:
            return f"label names unknown track {lab['tid']!r}"
    mp["answered"] = [q for q in mp.get("answered", []) if q in vocab.question_ids]
    # A malformed or unsupported rule is dropped, never fatal: the rule is an optional extra and
    # the batch is the turn's actual work, so losing the turn over it wastes a whole request.
    kept_rules = []
    for r in mp.get("new_rules", []):
        try:
            parse_jsonish(r["when"]); parse_jsonish(r["then"])
        except (ValueError, SyntaxError, TypeError):
            continue
        if not r.get("because"):
            continue
        if any(int(i) < 0 or int(i) >= max(1, vocab.ledger_len) for i in r["because"]):
            continue
        kept_rules.append(r)
    mp["new_rules"] = kept_rules
    # Everything below is theory the harness owns anyway: a duplicate slot, an entry the harness
    # cannot read, a cross-game pattern with one game behind it. None of it can corrupt a fact, so
    # none of it is worth bouncing the whole answer over. A rejected round-trip costs a full
    # request, and on this server that is ten minutes: one ls20 turn was spent entirely on a
    # pattern that cited a single game, throwing away the batch that came with it.
    seen_slots = set(vocab.slot_names)
    kept_slots = []
    for slot in mp.get("new_slots", []):
        if slot["name"] in seen_slots or len(seen_slots) >= MAX_SLOTS_PER_GAME:
            continue
        seen_slots.add(slot["name"])
        kept_slots.append(slot)
    mp["new_slots"] = kept_slots
    kept_entries = []
    for e in mp.get("entries", []):
        if e["slot"] not in seen_slots:
            continue  # an entry for a slot the model never declared
        try:
            vals = json.loads(e["values"])
        except json.JSONDecodeError:
            continue
        if not isinstance(vals, dict):
            continue
        kept_entries.append(e)
    mp["entries"] = kept_entries
    mp["new_patterns"] = [p for p in mp.get("new_patterns", [])
                          if len([g for g in p.get("evidence_games", []) if g in vocab.known_games]) >= 2]
    return None


def validate_decide(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    if obj["hypothesis_tested"] not in ("none", "") and obj["hypothesis_tested"] not in vocab.hypothesis_ids:
        obj["hypothesis_tested"] = "none"  # a hypothesis id that does not exist yet: the batch still stands
    if not obj["batch"]:
        return "batch must contain at least one action"
    for a in obj["batch"][:MAX_BATCH]:
        if a["action"] not in vocab.actions:
            return f"action {a['action']} is not available now; available: {sorted(vocab.actions)}"
        if a["action"] == "ACTION6" and (a.get("x") is None or a.get("y") is None):
            return "ACTION6 needs integer x and y in 0..63"
        if a["action"] == "ACTION6" and not (0 <= int(a["x"]) <= 63 and 0 <= int(a["y"]) <= 63):
            return "ACTION6 x and y must be in 0..63"
    # A prediction for a step the batch does not have is surplus, not corruption: the model
    # planned further than it committed. Bouncing the whole answer over it costs a turn, and one
    # was lost to exactly this -- "step 2 is outside the batch (1..1)" on an otherwise good reply.
    top = min(MAX_BATCH, len(obj["batch"]))
    obj["step_predictions"] = [sp for sp in obj["step_predictions"] if 1 <= int(sp["step"]) <= top]
    for sp in obj["step_predictions"]:
        problem = validate_claims(sp["expect"], vocab)
        if problem:
            return problem
    for k, v in obj["confidence"].items():
        if not 0.0 <= float(v) <= 1.0:
            return f"confidence.{k} outside [0,1]"
    return validate_memory_patch(obj["memory_patch"], vocab)


def validate_interpret(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    for lab in obj["labels"]:
        if lab["tid"] not in vocab.tids:
            return f"label names unknown track {lab['tid']!r}"
    # Low-stakes lists are filtered, not bounced: the harness decides the memory plan anyway.
    obj["salient"] = [t for t in obj["salient"] if t in vocab.tids]
    return None


def validate_orient(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    for lab in obj["entities"]:
        if lab["tid"] not in vocab.tids:
            return f"entities names unknown track {lab['tid']!r}"
    for key in ("track", "watch", "ignore"):
        obj["memory_plan"][key] = [t for t in obj["memory_plan"][key] if t in vocab.tids]
    obj["control_model"] = [cm for cm in obj["control_model"] if cm["action"] in vocab.actions or cm["action"] == "RESET"]
    return None


def validate_hypothesis(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    # One unusable hypothesis is not a reason to lose the others. Live on ls20: the model named
    # `obj:t42.pos` for a track that had just left the listing, and `state`, which is not a
    # registry scalar; both bounced the whole call, and a bounced call is a whole turn. A
    # hypothesis the harness cannot test is dropped, and only an empty set is an error.
    usable = [h for h in obj.get("hypotheses", [])
              if h["test_action"] in vocab.actions and vocab.var_ok(h["predicted_var"])]
    if not usable:
        return ("give at least one hypothesis whose test names an available action and a variable "
                "from VARS or obj:<tid>.pos/.colour/.gone")
    obj["hypotheses"] = usable
    return None


def validate_critic(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    return None


def validate_decide_lean(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    obj["memory_patch"] = fill_patch(obj["memory_patch"])
    return validate_decide(obj, vocab)


def goal_term(obj: dict[str, Any]) -> dict[str, Any]:
    """Assemble the goal-algebra term from the filled-in form. The model never writes syntax."""
    predicate = obj["predicate"]
    target = str(obj.get("target") or "")
    if predicate == "at":
        return {"at": ["$self", [int(obj.get("row") or 0), int(obj.get("col") or 0)]]}
    if predicate == "adjacent":
        return {"adjacent": ["$self", target]}
    if predicate == "count":
        return {"count": [target, obj.get("op") or "<=", int(obj.get("n") or 0)]}
    if predicate == "covered":
        return {"covered": [target, str(obj.get("onto") or "")]}
    return {predicate: target}


def validate_goal(obj: dict[str, Any], vocab: Vocab) -> Optional[str]:
    """Only what the form itself can get wrong; whether the goal is *solvable* is the harness's
    business, and is reported back to the model on the next turn rather than rejected here."""
    predicate = obj.get("predicate")
    if predicate == "at":
        for key in ("row", "col"):
            value = obj.get(key)
            if value is None or not 0 <= int(value) <= 63:
                return f"predicate 'at' needs {key} in 0..63"
    elif predicate == "covered":
        if not str(obj.get("onto") or "").startswith("class:"):
            return "predicate 'covered' needs onto set to the class every piece must end up on"
    elif predicate == "count":
        if obj.get("n") is None:
            return "predicate 'count' needs n, the number of things that may be left unreached"
        if not str(obj.get("target") or "").startswith("class:"):
            return "predicate 'count' counts a class, so target must be class:<X>"
    elif not str(obj.get("target") or "").strip():
        return f"predicate {predicate!r} needs a target from the symbol table"
    return None


VALIDATORS = {"decide": validate_decide, "decide_lean": validate_decide_lean,
              "interpret": validate_interpret, "orient": validate_orient,
              "critic": validate_critic, "hypothesis": validate_hypothesis,
              "goal": validate_goal}


def truncate_batch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """NOOA's commit rule: keep the useful prefix, report the cut."""
    if len(batch) <= MAX_BATCH:
        return batch, False
    return batch[:MAX_BATCH], True
