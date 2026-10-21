"""Harness-computed fact lines for traversal / collectable games, all generic:

- Passability: per target colour, how often the self moved into it vs was blocked, with the
  minority cases listed by cell and the scalar state at the time (the documented failure is an
  inverted floor/wall reading; a mixed colour is the precondition signal).
- Consumption: per class, consumed/remaining counts and the order things were consumed in.
- PreconditionMiner: a contact table (boolean features at each contact, success or not);
  each single feature scored as "success iff f" with a Laplace count; the top separators
  become a fact line.
- tried_ledger_line: batches that changed nothing, grouped, so the model stops re-trying.

Stdlib only.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Iterable, Optional

from v9.types import COLOUR_CHARS, Cell, Record, colour_letter

__all__ = ["Passability", "Consumption", "PreconditionMiner", "class_key", "tried_ledger_line"]

_OBJ_RE = re.compile(r"^obj:([^.]+)\.")
_FORM_HASH_RE = re.compile(r"^[0-9a-f]{4,}$")     # perception's fnv1a form_hash
MAX_ORDER_SHOWN = 12
MAX_TIDS_SHOWN = 4


def class_key(x: object) -> Optional[str]:
    """Validate a class key, the same key `class:<X>` names in the rule DSL: one COLOUR_CHARS
    letter, or a perception form hash (lowercase hex). Anything else — a role label, a
    tracker klass ("unknown"), an empty string, None — is not a class and returns None, so it
    never reaches the scalar registry as `consumed[unknown]`."""
    if not isinstance(x, str) or not x:
        return None
    if len(x) == 1:
        return x if x in COLOUR_CHARS else None
    return x if _FORM_HASH_RE.match(x) else None


def _state_text(state: Optional[dict[str, int]]) -> str:
    if not state:
        return ""
    keys = [k for k in state if k.startswith(("consumed[", "remaining[", "hud["))] or sorted(state)
    return ", ".join(f"{k}={state[k]}" for k in sorted(keys)[:3])


# --- passability --------------------------------------------------------------------------------


class Passability:
    """Open/blocked counts per target colour learned from the self track's moves."""

    def __init__(self) -> None:
        self.open: Counter = Counter()
        self.blocked: Counter = Counter()
        self.exceptions: dict[int, list[tuple[str, Cell, str, str]]] = defaultdict(list)
        self.visited: set[Cell] = set()
        self.n = 0

    def observe(self, action: str, self_track_before_bbox: tuple[int, int, int, int], moved: bool,
                target_cells_colours: list[int], *, target_cells: Optional[list[Cell]] = None,
                state: Optional[dict[str, int]] = None) -> None:
        """One move attempt: the colours of the cells the self would have entered, and whether
        it moved. Every distinct target colour gets the observation (a blocked attempt against
        a mixed target is charged to each colour). `target_cells` and `state` only feed the
        exception listing."""
        self.n += 1
        r0, c0, r1, c1 = self_track_before_bbox
        if moved:
            self.visited.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        for col in sorted(set(int(x) for x in target_cells_colours)):
            if moved:
                self.open[col] += 1
            else:
                self.blocked[col] += 1
            where = (target_cells[0] if target_cells else (r0, c0))
            self.exceptions[col].append(("open" if moved else "blocked", tuple(where), action, _state_text(state)))

    def colours(self) -> list[int]:
        return sorted(set(self.open) | set(self.blocked))

    def verdict(self, colour: int) -> str:
        """WALL | OPEN | MIXED | unknown for one colour."""
        o, b = self.open.get(colour, 0), self.blocked.get(colour, 0)
        if o + b == 0:
            return "unknown"
        if b and not o:
            return "WALL"
        if o and not b:
            return "OPEN"
        return "MIXED"

    def status(self, adjacent_colours: Optional[Iterable[int]] = None) -> str:
        """none (no observations) | partial | complete (every colour in `adjacent_colours`, or
        every observed colour when not given, has >= 3 observations)."""
        if self.n == 0:
            return "none"
        cols = set(adjacent_colours) if adjacent_colours is not None else set(self.colours())
        if not cols:
            return "partial"
        for col in cols:
            if self.open.get(col, 0) + self.blocked.get(col, 0) < 3:
                return "partial"
        return "complete"

    def render(self) -> str:
        """`colour G: open 240/240 | colour c: blocked 31/31 -> WALL | colour Y: open 3/4
        (blocked once at (19,30) with consumed[Y]=0)`."""
        if self.n == 0:
            return "passability: no moves observed yet"
        parts = []
        for col in self.colours():
            o, b = self.open.get(col, 0), self.blocked.get(col, 0)
            L = colour_letter(col)
            if b and not o:
                parts.append(f"colour {L}: blocked {b}/{b} -> WALL")
            elif o and not b:
                parts.append(f"colour {L}: open {o}/{o}")
            else:
                minority = "blocked" if b <= o else "open"
                cases = [e for e in self.exceptions[col] if e[0] == minority]
                shown = []
                for _kind, cell, _action, st in cases[:2]:
                    s = f"at ({cell[0]},{cell[1]})"
                    if st:
                        s += f" with {st}"
                    shown.append(s)
                times = {1: "once", 2: "twice"}.get(len(cases), f"{len(cases)} times")
                head = f"colour {L}: open {o}/{o + b}"
                if len(cases) == 1:
                    parts.append(f"{head} ({minority} once {shown[0]})")
                else:
                    parts.append(f"{head} ({minority} {times}: " + "; ".join(shown) + ")")
        return " | ".join(parts)


# --- consumption -------------------------------------------------------------------------------


class Consumption:
    """Per-class consumed/remaining counts plus the consumption order.

    A class is a colour letter or a form hash (`class_key`); a track whose class cannot be
    determined is counted as unclassified and kept out of the scalar registry entirely,
    because every scalar name here is shown to the model and is claimable in a prediction.
    `remaining[X]` counts the live members of X at the last `census`, not population minus
    consumed: objects spawn, so the difference is not the population.
    """

    def __init__(self) -> None:
        self.population: dict[str, set[str]] = defaultdict(set)   # class -> tids ever seen
        self.consumed_tids: dict[str, list[str]] = defaultdict(list)
        self.order: list[tuple[int, str, str]] = []                # (step, tid, class)
        self.klass_of: dict[str, str] = {}
        self.live_counts: dict[str, int] = {}
        self.censused = False
        self.unclassified = 0

    # -- class resolution ------------------------------------------------------------------

    def _resolve(self, tid: str, hint: object = None, snap: object = None) -> Optional[str]:
        """Class of a track: an explicit valid hint, else what we already know about the tid,
        else the colour letter of its snapshot track. None when it cannot be determined."""
        k = class_key(hint)
        if k:
            return k
        k = self.klass_of.get(tid)
        if k:
            return k
        tracks = getattr(snap, "tracks", None) if snap is not None else None
        t = tracks.get(tid) if isinstance(tracks, dict) else None
        if t is not None:
            return class_key(getattr(t, "letter", None)) or class_key(colour_letter(getattr(t, "colour", -1)))
        return None

    def register(self, tid: str, klass: str) -> bool:
        """Declare a track as a member of a class. Returns False (and records nothing) when
        `klass` is not a colour letter or a form hash."""
        k = class_key(klass)
        if not k:
            return False
        self.klass_of[tid] = k
        self.population[k].add(tid)
        return True

    def consume(self, tid: str, step: int, klass: Optional[str] = None) -> bool:
        """Record that a track was consumed at `step` (idempotent per tid). Returns False when
        the class cannot be determined; the event then counts as unclassified only."""
        k = self._resolve(tid, klass)
        if not k:
            self.unclassified += 1
            return False
        if tid in self.consumed_tids[k]:
            return True
        self.register(tid, k)
        self.consumed_tids[k].append(tid)
        self.order.append((step, tid, k))
        return True

    # -- observation -----------------------------------------------------------------------

    def observe(self, record: Record, klass_of: Optional[dict[str, str]] = None) -> None:
        """Pull CONSUME events (and consumed tids) out of one ledger record. `klass_of` is an
        optional {tid: class key} map (colour letter or form hash); anything else is ignored
        and the class is read from the record's before snapshot instead."""
        self.observe_delta(record.delta, record.before_snap, record.step, klass_of)

    def observe_delta(self, delta: Any, before_snap: Any, step: int,
                      klass_of: Optional[dict[str, str]] = None) -> None:
        """The same reading, from the delta directly.

        The caller needs this before a `Record` exists, because `consumed[X]` and `remaining[X]`
        are shown to the model and claimable in a prediction: updated after the comparison, they
        were identical in the before and after readings of every step, so the one claim that
        settles a collection game -- "this action consumes a Y" -- graded VIOLATED every time it
        was right.
        """
        km = klass_of or {}
        snap = before_snap
        seen: set[str] = set()
        for ev in delta.events:
            if ev.kind == "CONSUME":
                self.consume(ev.tid, ev.step, self._resolve(ev.tid, km.get(ev.tid) or ev.detail.get("class"), snap))
                seen.add(ev.tid)
        for tid in delta.objects.consumed:
            if tid not in seen:
                self.consume(tid, step, self._resolve(tid, km.get(tid), snap))

    def census(self, live: object, klass_of: Optional[dict[str, str]] = None) -> None:
        """Count the live members of each class now. `live` is a Snapshot (anything with
        `.tracks`) or a {tid: class key} mapping. Only classes we already know about get a
        `remaining[X]`, so a census never invents a scalar for every colour on the board."""
        km = klass_of or {}
        counts: dict[str, int] = {}
        if isinstance(live, dict):
            items = [(tid, self._resolve(tid, km.get(tid) or k)) for tid, k in live.items()]
        else:
            tracks = getattr(live, "tracks", {}) or {}
            items = [(tid, self._resolve(tid, km.get(tid), live))
                     for tid, t in tracks.items() if getattr(t, "state", "LIVE") == "LIVE"]
        for tid, k in items:
            if not k or tid in self.consumed_tids.get(k, ()):
                continue
            counts[k] = counts.get(k, 0) + 1
            if k in self.population or k in self.consumed_tids:
                self.register(tid, k)
        self.live_counts = counts
        self.censused = True

    # -- readings --------------------------------------------------------------------------

    def classes(self) -> list[str]:
        """Every class with a consumption or a registered member, sorted."""
        return sorted(set(self.population) | set(self.consumed_tids))

    def scalars(self) -> dict[str, int]:
        """consumed[X] and remaining[X] for every known class. Every name here is a valid
        class key, so the model can claim it and a rule can name it."""
        out: dict[str, int] = {}
        for k in self.classes():
            c = len(self.consumed_tids.get(k, []))
            out[f"consumed[{k}]"] = c
            if self.censused:
                out[f"remaining[{k}]"] = int(self.live_counts.get(k, 0))
            else:
                out[f"remaining[{k}]"] = max(0, len(self.population.get(k, ())) - c)
        return out

    def render(self) -> str:
        """`consumed: Y 2/5 (t3@41, t7@44), R 0/3 | order: Y, Y`, plus an unclassified count
        when tracks were consumed whose class we could not determine."""
        if not self.population and not self.order:
            tail = f" ({self.unclassified} of unknown class)" if self.unclassified else ""
            return "consumed: nothing tracked" + tail
        parts = []
        steps = {tid: s for s, tid, _ in self.order}
        for k in self.classes():
            got = self.consumed_tids.get(k, [])
            total = len(self.population.get(k, ()))
            shown = got[-MAX_TIDS_SHOWN:]
            detail = ""
            if got:
                lead = "..., " if len(got) > len(shown) else ""
                detail = " (" + lead + ", ".join(f"{t}@{steps.get(t, '?')}" for t in shown) + ")"
            parts.append(f"{k} {len(got)}/{total}{detail}")
        line = "consumed: " + ", ".join(parts)
        if self.order:
            tail_order = self.order[-MAX_ORDER_SHOWN:]
            lead = "..., " if len(self.order) > len(tail_order) else ""
            line += " | order: " + lead + ", ".join(k for _, _, k in tail_order)
        if self.unclassified:
            line += f" | {self.unclassified} consumed of unknown class (not scored)"
        return line


# --- precondition miner ------------------------------------------------------------------------


class PreconditionMiner:
    """A contact table; each single boolean feature scored as 'success iff f'."""

    def __init__(self) -> None:
        self.rows: list[tuple[dict[str, bool], bool]] = []

    def record_contact(self, features: dict[str, bool], success: bool) -> None:
        """One contact: the boolean state at the moment of contact and whether it succeeded."""
        self.rows.append(({k: bool(v) for k, v in features.items()}, bool(success)))

    def separators(self, top: int = 2) -> list[tuple[str, float, int]]:
        """Top features by Laplace agreement (agree+1)/(n+2) with 'success iff f' (or iff not f,
        reported as `!f`), where n = contacts that carry the feature. Ties break toward
        coverage, then name. Features constant across all rows are skipped."""
        if not self.rows:
            return []
        names = sorted({k for feats, _ in self.rows for k in feats})
        scored: list[tuple[str, float, int]] = []
        for f in names:
            rows = [(feats[f], ok) for feats, ok in self.rows if f in feats]
            n = len(rows)
            # Constant across the WHOLE table, not just among the rows carrying it. Features here
            # are sparse -- `target_is_Y` is recorded only on contacts whose target is Y, and
            # absence is what encodes False -- so "the value never varies among carrying rows" is
            # true of every real separator and the old test leaned on the outcomes instead. With
            # `and` binding tighter than `or`, it skipped a constant only when the outcomes
            # varied, which is the opposite of what it needed: in the ordinary early state, where
            # every contact fails, `self_is_b` and `action_3` scored (n+1)/(n+2) with full
            # coverage as `!f` and the alphabetically first one was asserted to the model as the
            # precondition and handed to induce_contact_rules as a rule guard.
            true_rows = sum(1 for feats, _ok in self.rows if feats.get(f))
            if n == 0 or true_rows == 0 or true_rows == len(self.rows):
                continue
            agree = sum(1 for v, ok in rows if v == ok)
            disagree = n - agree
            if agree >= disagree:
                scored.append((f, (agree + 1) / (n + 2), n))
            else:
                scored.append((f"!{f}", (disagree + 1) / (n + 2), n))
        scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return scored[:top]

    def coverage_of(self, feature: str) -> float:
        f = feature[1:] if feature.startswith("!") else feature
        n = sum(1 for feats, _ in self.rows if f in feats)
        return n / len(self.rows) if self.rows else 0.0

    def render(self, top: int = 2) -> str:
        """`contact rule: success iff consumed[Y]>=2 (7/7, coverage complete); next: adjacent=up (5/7)`."""
        if not self.rows:
            return "contact rule: no contacts observed yet"
        seps = self.separators(top)
        if not seps:
            return f"contact rule: {len(self.rows)} contacts, no separating feature yet"
        total = len(self.rows)
        bits = []
        for i, (f, score, n) in enumerate(seps):
            rows = [(feats[f.lstrip('!')], ok) for feats, ok in self.rows if f.lstrip('!') in feats]
            agree = sum(1 for v, ok in rows if (v == ok) != f.startswith("!"))
            cov = "complete" if n == total else f"{n}/{total} rows"
            label = "success iff " + (f"not {f[1:]}" if f.startswith("!") else f)
            bits.append((f"{label} ({agree}/{n}, coverage {cov})") if i == 0 else f"{label} ({agree}/{n})")
        return "contact rule: " + bits[0] + ("; next: " + "; ".join(bits[1:]) if len(bits) > 1 else "")


# --- tried-and-failed --------------------------------------------------------------------------


def _target_of(rec: Record) -> Optional[str]:
    if rec.prediction is not None:
        for c in rec.prediction.claims:
            m = _OBJ_RE.match(c.var)
            if m and rec.before_snap is not None and m.group(1) != rec.before_snap.self_tid:
                return m.group(1)
            if m and rec.before_snap is None:
                return m.group(1)
    return None


def tried_ledger_line(ledger: Any, *, level: Optional[int] = None, max_groups: int = 4) -> str:
    """Summarise records that changed nothing (noop / blocked tag or zero changed cells),
    grouped by action, target object (from the prediction's obj:<tid> claims) and the
    consumed[*] state: `tried: ACTION3 onto t2 at consumed[Y]=0 x2, steps 41,44; ...`."""
    recs = list(ledger.records) if hasattr(ledger, "records") else list(ledger)
    if level is not None:
        recs = [r for r in recs if r.level == level]
    groups: "OrderedDict[tuple, list[int]]" = OrderedDict()
    for r in recs:
        dead = r.delta.cells.n == 0 or "noop" in r.delta.tags or "blocked" in r.delta.tags
        if not dead:
            continue
        state = ""
        if r.before_snap is not None:
            state = _state_text({k: v for k, v in r.before_snap.scalars.items() if k.startswith("consumed[")})
        key = (r.action, _target_of(r), state)
        groups.setdefault(key, []).append(r.step)
    if not groups:
        return "tried: nothing has failed to change the board yet"
    parts = []
    for (action, target, state), steps in list(groups.items())[:max_groups]:
        s = action
        if target:
            s += f" onto {target}"
        if state:
            s += f" at {state}"
        s += f" x{len(steps)}, step{'s' if len(steps) > 1 else ''} " + ",".join(str(x) for x in steps[-4:])
        parts.append(s)
    more = f"; +{len(groups) - max_groups} more" if len(groups) > max_groups else ""
    return "tried: " + "; ".join(parts) + more
