"""The prediction contract: what the model claimed before an action, graded against the
observed Delta after it.

Every claim becomes exactly one of MATCHED, VIOLATED (with the observed value) or ABSTAINED
(the variable was veiled or is not observable from this delta). Every observed change with no
claim on it is UNEXPECTED. From those, a surprise number and a coverage status. The wording of
the outcome line the model reads back is fixed here (Schema's format) so it can be
tested byte-for-byte.

Claim variables
- any scalar-registry name: cells[X], objs[X], objs_total, hud[edge].filled, pos[tid].r,
  pos[tid].c, consumed[X], remaining[X], level, state, ...
- obj:<tid>.pos     value (dr,dc) for op delta, (r,c) for op equals (needs pos[tid].r/.c scalars)
- obj:<tid>.colour  value = colour int for op equals; changed/unchanged
- obj:<tid>.gone    value True/False for op equals; changed means "became gone"
- cells             the changed-cell count: delta/equals on the count, or changed/unchanged
Ops: delta, equals, unchanged, changed.

Stdlib only.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

from v9.types import Claim, Comparison, Delta, Prediction

__all__ = ["compare", "outcome_line", "P_UNCLAIMED", "MISPREDICTED"]

P_UNCLAIMED = 0.05          # probability charged for a change nobody claimed
P_FLOOR = 0.01              # a violated claim never costs more than -log2(P_FLOOR)
MISPREDICTED = ("world model MISPREDICTED the step just taken (action {n}); "
                "the rest of the committed plan was dropped.")

_OBJ_RE = re.compile(r"^obj:(?P<tid>[^.]+)\.(?P<attr>pos|colour|color|gone)$")
_POS_RE = re.compile(r"^pos\[(?P<tid>[^\]]+)\]\.(?P<axis>r|c)$")

_ABSTAIN = object()   # sentinel: the variable is unobservable in this delta


def _norm_value(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_norm_value(x) for x in v)
    return v


def _diff(before: Any, after: Any) -> Any:
    if isinstance(before, (tuple, list)) and isinstance(after, (tuple, list)):
        return tuple(a - b for a, b in zip(after, before))
    return after - before


def _observe(var: str, delta: Delta, before: dict[str, int], after: dict[str, int]) -> Any:
    """Return (value_before, value_after) for a claim var, or _ABSTAIN. For obj:*.pos the
    'values' are (0,0) and the shift, so a delta reads directly as the observed shift."""
    objs = delta.objects
    if var == "cells":
        return (0, delta.cells.n)
    m = _OBJ_RE.match(var)
    if m:
        tid, attr = m.group("tid"), m.group("attr")
        if tid in objs.occluded or tid in objs.ambiguous:
            return _ABSTAIN
        if attr == "pos":
            shift = next((s for t, s in objs.moved if t == tid), None)
            if shift is None and (tid in objs.vanished or tid in objs.consumed):
                return ("present", "gone")
            return ((0, 0), tuple(shift) if shift else (0, 0))
        if attr in ("colour", "color"):
            rc = next((x for x in objs.recoloured if x[0] == tid), None)
            if rc is not None:
                return (rc[1], rc[2])
            key = f"colour[{tid}]"
            if key in before or key in after:
                return (before.get(key, after.get(key)), after.get(key, before.get(key)))
            if tid in objs.vanished or tid in objs.consumed:
                return _ABSTAIN
            return ("same", "same")           # not recoloured; absolute value unknown
        if attr == "gone":
            gone_after = tid in objs.vanished or tid in objs.consumed
            appeared = tid in objs.appeared or tid in objs.reappeared
            return (True, False) if appeared else (False, gone_after)
    if var in before or var in after:
        b = before.get(var, after.get(var))
        a = after.get(var, before.get(var))
        return (b, a)
    return _ABSTAIN


def _grade(claim: Claim, observed: Any) -> tuple[str, Any]:
    """Grade one claim against (before, after). Returns (verdict, observed_value)."""
    b, a = observed
    op = claim.op
    value = _norm_value(claim.value)
    if b in ("present", "same") and isinstance(a, str):
        # obj vanished when a position was claimed, or a colour that did not change
        if b == "present":            # position claim on a track that went away
            if op == "changed":
                return ("MATCHED", "gone")
            return ("VIOLATED", "gone")
        if b == "same":               # colour with no recolour event
            if op == "unchanged":
                return ("MATCHED", "unchanged")
            if op == "changed":
                return ("VIOLATED", "unchanged")
            return ("ABSTAINED", None)
    if op == "delta":
        d = _diff(b, a)
        if isinstance(value, (int, float)) and isinstance(d, tuple):
            return ("VIOLATED", d)
        return ("MATCHED" if d == value else "VIOLATED", d)
    if op == "equals":
        return ("MATCHED" if a == value else "VIOLATED", a)
    if op == "unchanged":
        d = _diff(b, a) if not isinstance(b, bool) else (b != a)
        zero = (d == 0) or (isinstance(d, tuple) and all(x == 0 for x in d)) or d is False
        return ("MATCHED" if zero else "VIOLATED", d)
    if op == "changed":
        # `obj:<tid>.gone` is one-directional: "changed" means the thing became gone, not that
        # its goneness moved in either direction. Graded symmetrically, a claim that a track will
        # vanish was MATCHED when it REAPPEARED instead -- which both suppresses the halt the
        # batch is supposed to take on a surprise and counts the step as clean.
        if str(getattr(claim, "var", "")).endswith(".gone"):
            return ("MATCHED" if a is True and b is not True else "VIOLATED", a)
        d = _diff(b, a) if not isinstance(b, bool) else (b != a)
        zero = (d == 0) or (isinstance(d, tuple) and all(x == 0 for x in d)) or d is False
        return ("VIOLATED" if zero else "MATCHED", d)
    return ("ABSTAINED", None)


def _covers(claimed: set[str], name: str) -> bool:
    """Whether a scalar/object change `name` is covered by any claim var."""
    if name in claimed:
        return True
    m = _POS_RE.match(name)
    if m and f"obj:{m.group('tid')}.pos" in claimed:
        return True
    m = _OBJ_RE.match(name)
    if m and m.group("attr") == "pos":
        tid = m.group("tid")
        if f"pos[{tid}].r" in claimed or f"pos[{tid}].c" in claimed:
            return True
    return False


def compare(prediction: Prediction, delta: Delta, scalars_before: dict[str, int],
            scalars_after: dict[str, int], *, veiled_vars: set[str] = frozenset()) -> Comparison:
    """Grade a prediction against the observed delta.

    Each claim is MATCHED, VIOLATED (with what was seen) or ABSTAINED (its var is in
    `veiled_vars`, in the prediction's own abstain list, or unobservable here). UNEXPECTED
    lists scalar names that changed with no claim covering them (hud[*] and the positions of
    tracks in an ambiguity group are ignored), plus tracks that moved/vanished/recoloured
    with no claim. surprise = sum of -log2(p) over violated claims (p = the probability the
    model left for the other outcome, 1 - claim.p, floored at P_FLOOR) and unexpected changes
    (p = P_UNCLAIMED). coverage = graded / (graded + unexpected) with graded = matched +
    violated; status complete when nothing unclaimed changed, partial otherwise, vacuous for a
    prediction with no claims.
    """
    matched: list[Claim] = []
    violated: list[tuple[Claim, Any]] = []
    abstained: list[Claim] = []
    abstain_names = set(prediction.abstain) | set(veiled_vars)
    claimed_vars: set[str] = set()

    for claim in prediction.claims:
        claimed_vars.add(claim.var)
        if claim.var in abstain_names or _is_veiled(claim.var, abstain_names):
            abstained.append(claim)
            continue
        obs = _observe(claim.var, delta, scalars_before, scalars_after)
        if obs is _ABSTAIN:
            abstained.append(claim)
            continue
        verdict, seen = _grade(claim, obs)
        if verdict == "MATCHED":
            matched.append(claim)
        elif verdict == "VIOLATED":
            violated.append((claim, seen))
        else:
            abstained.append(claim)

    unexpected: list[str] = []
    ambiguous = set(delta.objects.ambiguous)
    for name in sorted(delta.scalars):
        # Diagnostics and bookkeeping, not board facts. `hud[...]` was already excluded;
        # `veil_frac` moves on any animated frame and `level`/`step`/`state` move on every
        # level-up, and charging them as "a change nobody claimed" costs -log2(0.05) = 4.32
        # surprise each and drags `coverage` down permanently -- and coverage is three tenths
        # of the world-model channel the commit gate reads.
        if name.startswith("hud[") or name in ("veil_frac", "level", "step", "state"):
            continue
        m = _POS_RE.match(name)
        if m and m.group("tid") in ambiguous:
            continue
        if not _covers(claimed_vars, name) and name not in abstain_names:
            unexpected.append(name)
    pos_scalar_tids = {m.group("tid") for m in map(_POS_RE.match, delta.scalars) if m}
    for tid, _shift in delta.objects.moved:
        if tid in ambiguous or tid in pos_scalar_tids:
            continue
        name = f"obj:{tid}.pos"
        if not _covers(claimed_vars, name) and name not in abstain_names:
            unexpected.append(name)
    for tid in list(delta.objects.vanished) + list(delta.objects.consumed):
        name = f"obj:{tid}.gone"
        if tid not in ambiguous and name not in claimed_vars and name not in abstain_names:
            unexpected.append(name)
    for tid, _f, _t in delta.objects.recoloured:
        name = f"obj:{tid}.colour"
        if tid not in ambiguous and name not in claimed_vars and f"obj:{tid}.color" not in claimed_vars \
                and name not in abstain_names:
            unexpected.append(name)

    surprise = 0.0
    for claim, _seen in violated:
        p = max(P_FLOOR, 1.0 - min(1.0, max(0.0, float(claim.p))))
        surprise += -math.log2(p)
    surprise += len(unexpected) * -math.log2(P_UNCLAIMED)

    graded = len(matched) + len(violated)
    unclaimed = len(unexpected)
    if not prediction.claims:
        coverage, status = 0.0, "vacuous"
    elif graded + unclaimed == 0:
        coverage, status = 0.0, "partial"
    else:
        coverage = graded / (graded + unclaimed)
        status = "complete" if unclaimed == 0 else "partial"
    return Comparison(matched=matched, violated=violated, abstained=abstained, unexpected=unexpected,
                      surprise=round(surprise, 3), coverage=round(coverage, 3), coverage_status=status)


def _is_veiled(var: str, veiled: set[str]) -> bool:
    """obj:<tid>.* is veiled when 'obj:<tid>' or the pos[<tid>] scalars are veiled."""
    m = _OBJ_RE.match(var)
    if m:
        tid = m.group("tid")
        return f"obj:{tid}" in veiled or f"pos[{tid}].r" in veiled or f"pos[{tid}].c" in veiled
    m = _POS_RE.match(var)
    if m:
        return f"obj:{m.group('tid')}" in veiled
    return False


def _fmt(v: Any) -> str:
    if isinstance(v, tuple):
        return "(" + ",".join(str(x) for x in v) + ")"
    return str(v)


def _short_action(a: str) -> str:
    return a.replace("ACTION", "A") if isinstance(a, str) else str(a)


def outcome_line(prediction: Optional[Prediction], comparison: Optional[Comparison], action: str,
                 executed: int, requested: int, level_before: int, level_after: int, *,
                 batch: Optional[list[str]] = None, reason: Optional[str] = None) -> str:
    """The one line the model reads about its last commit, in Schema's wording:

    `Result of your last commit: committed N action(s) [..] — executed k; stopped because
    <reason>. Net: level a→b. matched m/n claims (hyp H); violated: <var> expected X got Y;
    unexpected: ...; coverage <status>; surprise s.s`

    When a claim was violated the reason is the fixed MISPREDICTED sentence naming the step
    just taken (action k). Otherwise `reason` is used if given, else derived: the level
    changed / the batch ran to its end / the batch was cut short.
    """
    violated = bool(comparison and comparison.violated)
    if violated:
        why = MISPREDICTED.format(n=executed)
    elif reason:
        why = reason
    elif level_after != level_before:
        why = "the level changed"
    elif executed >= requested:
        why = "the batch ran to its end"
    else:
        why = "the batch was cut short"
    listing = ", ".join(_short_action(a) for a in (batch if batch is not None else [action] * requested))
    if not why.endswith("."):
        why += "."
    head = (f"Result of your last commit: committed {requested} action(s) [{listing}] — executed {executed}; "
            f"stopped because {why} Net: level {level_before}→{level_after}.")
    if comparison is None or prediction is None:
        return head + " no prediction was registered for this step."
    n = len(prediction.claims)
    hyps = sorted({c.hypothesis for c in prediction.claims if c.hypothesis and c.hypothesis != "none"})
    hyp = f" (hyp {', '.join(hyps)})" if hyps else ""
    parts = [f"matched {len(comparison.matched)}/{n} claims{hyp}"]
    if comparison.violated:
        parts.append("violated: " + "; ".join(
            f"{c.var} expected {_fmt(_norm_value(c.value)) if c.op in ('delta', 'equals') else c.op} got {_fmt(seen)}"
            for c, seen in comparison.violated))
    if comparison.abstained:
        parts.append("abstained: " + ", ".join(c.var for c in comparison.abstained))
    if comparison.unexpected:
        parts.append("unexpected: " + ", ".join(comparison.unexpected))
    parts.append(f"coverage {comparison.coverage_status}")
    parts.append(f"surprise {comparison.surprise:.1f}")
    return head + " " + "; ".join(parts)
