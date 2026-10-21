"""Per-action control map: what each action has done to the self track, counted from the
ledger's object deltas. Keyed on tracks, not colours (astra_infra's ControlMap keyed on
colour cells; here the tracker already says which object is the self).

Effect signatures per observation:
  moved (dr,dc)   the self track moved by exactly (dr,dc)
  blocked         the delta carries the `blocked` tag (or the self did not move while the
                  action is otherwise known to move it, if the caller tags it so)
  no-op           nothing changed on the board
  other           something else changed (recolours / vanishes / appearances), with counts

Stdlib only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Any, Optional

from v9.types import Delta

__all__ = ["ControlMap", "UNINFORMATIVE"]

# Signatures that say "we could not read this action", as opposed to a real effect.
UNINFORMATIVE = ("other", "hud-only")


def _short(action: str) -> str:
    return action.replace("ACTION", "A")


def _fmt_count(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


class ControlMap:
    """Counts of effect signatures per action, with a dominant reading per action."""

    def __init__(self) -> None:
        self.counts: dict[str, Counter] = defaultdict(Counter)       # action -> {sig: weight}
        self.detail: dict[str, Counter] = defaultdict(Counter)       # action -> {"recolour": n, ...}
        self.inherited: set[str] = set()
        # The same reading, split by the state the body was in. A button whose meaning depends on
        # which way the body is facing is unlearnable pooled: every press contradicts the last,
        # `known` never settles, and control confidence falls back towards 0.5 however many times
        # it is pressed. Split by context each reading is consistent, and the pooled counts stay
        # as the fallback for when a context has not been seen often enough to stand on its own.
        self.by_context: dict[tuple[str, str], Counter] = defaultdict(Counter)

    # -- observation ---------------------------------------------------------------------------

    @staticmethod
    def classify(delta: Delta, self_tid: Optional[str], *, self_renamed: bool = False) -> tuple[str, dict[str, int]]:
        """Signature key for one delta, plus the other-change counts."""
        objs = delta.objects
        other = {"recolour": len(objs.recoloured), "vanish": len(objs.vanished) + len(objs.consumed),
                 "appear": len(objs.appeared)}
        if "blocked" in delta.tags:
            return "blocked", other
        if self_tid is not None:
            shift = next((s for t, s in objs.moved if t == self_tid), None)
            if shift is not None and tuple(shift) != (0, 0):
                return f"moved ({shift[0]},{shift[1]})", other
        # A lone mover is what the button moved, whether or not the tracker is calling it the self.
        #
        # This used to fire only on a step where the body had just been renamed, on the reasoning
        # that a lone mover which is not the self is somebody else. The evidence says otherwise and
        # it is decisive: on re86, pressing the five actions in rotation left the control map with
        # ZERO readable controls, because one of those actions changes which piece the others move,
        # so "did the self move" answers differently every time and never converges. A map that
        # cannot read a single button is worth less than one that occasionally credits a button
        # with an autonomous object's movement -- and the ledger re-scores every rule anyway, while
        # a plan built on a wrong shift halts at its first step.
        real = [(t, sh) for t, sh in objs.moved if tuple(sh) != (0, 0)]
        # One movement, however many tracks report it. A sprite made of several components moves
        # as several tids with the same displacement, so "exactly one thing moved" is almost never
        # true of a real piece: on re86 the four directions stayed unreadable through every warmup
        # because its pieces are rings and each press moved three or four tracks at once. What
        # makes it one movement is that they all moved the same way.
        if real and len({tuple(sh) for _t, sh in real}) == 1:
            sh = real[0][1]
            return f"moved ({sh[0]},{sh[1]})", other
        if delta.cells.n == 0 or "noop" in delta.tags:
            return "no-op", other
        if "hud_only" in delta.tags:
            return "hud-only", other
        return "other", other

    def observe(self, action: str, delta: Delta, self_tid: Optional[str], *,
                self_renamed: bool = False, context: str = "") -> str:
        """Record one action's effect. Returns the signature key it was filed under.

        `context` is whatever the harness can say about the body's state at the moment of the
        press -- its shape, for a body that turns. It is recorded alongside the pooled count, and
        never instead of it.
        """
        key, other = self.classify(delta, self_tid, self_renamed=self_renamed)
        if context:
            self.by_context[(action, context)][key] += 1
        self.counts[action][key] += 1
        for k, v in other.items():
            if v:
                self.detail[action][k] += v
        return key

    # -- readings ------------------------------------------------------------------------------

    def actions(self) -> list[str]:
        """Actions with at least one observation."""
        return sorted(a for a, c in self.counts.items() if sum(c.values()) > 0)

    def signature(self, action: str) -> dict[str, Any]:
        """{"counts": {sig: n}, "total": n, "dominant": sig|None, "confidence": dominant/(total+1),
        "shift": (dr,dc)|None, "other": {"recolour": n, ...}}."""
        c = self.counts.get(action, Counter())
        total = float(sum(c.values()))
        if total <= 0:
            return {"counts": {}, "total": 0, "dominant": None, "confidence": 0.0, "shift": None,
                    "other": {}}
        # "other" is the absence of a reading, not a competing one, so it must never win a tie
        # against a real one. On re86 A1 and A4 each read `moved (-3,0) x5, other x5`, the tie
        # broke on the alphabet -- "o" sorts after "m" -- the dominant came out "other", and the
        # shift was discarded. Two of the four directions were thrown away on a spelling, the
        # planner was left with down and left, and every arrangement search reported
        # "no_plan_exhausted" on a board that needs four directions.
        # Count first, then informativeness, then the name. Putting informativeness first let a
        # single move beat six unreadable frames, which is the opposite mistake and a worse one.
        dominant, n = max(c.items(), key=lambda kv: (kv[1], kv[0] not in UNINFORMATIVE, kv[0]))
        return {"counts": dict(c), "total": total if not total.is_integer() else int(total),
                "dominant": dominant, "confidence": n / (total + 1.0),
                "shift": self._shift_of(dominant), "other": dict(self.detail.get(action, {}))}

    @staticmethod
    def _shift_of(key: Optional[str]) -> Optional[tuple[int, int]]:
        if key and key.startswith("moved ("):
            a, b = key[7:-1].split(",")
            return (int(a), int(b))
        return None

    def expected_shift(self, action: str) -> Optional[tuple[int, int]]:
        """The self shift this action produces when it produces one.

        The dominant reading first; failing that, the most-seen move. A direction key in a game
        with walls spends much of its life blocked, and "blocked" can easily be the single most
        common outcome of a button that moves the body perfectly reliably whenever there is
        somewhere to move to.
        """
        dominant = self.signature(action)["dominant"]
        direct = self._shift_of(dominant)
        if direct is not None:
            return direct
        if dominant in UNINFORMATIVE:
            # The frames could not be read. `move_signatures` still offers the candidates to rule
            # induction, which scores them against the ledger; asserting one here would put noise
            # straight into a plan.
            return None
        moves = self.move_signatures(action)
        return moves[0][0] if moves else None

    def move_signatures(self, action: str) -> list[tuple[tuple[int, int], float]]:
        """Every self-shift this action has produced, most seen first, as ((dr,dc), count).

        `expected_shift` reports only the dominant effect, so an action whose delta is usually
        filed as "other" (a noisy tracker, an animation frame, something else moving in the
        same step) reports no shift at all even after a dozen clean moves. Rule induction
        needs the candidates, not just the winner, and then lets the ledger judge them.
        """
        out = []
        for key, n in self.counts.get(action, Counter()).items():
            shift = self._shift_of(key)
            if shift is not None:
                out.append((shift, float(n)))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out

    def slide_direction(self, action: str) -> Optional[tuple[int, int]]:
        """The unit direction of an action whose travel varies, or None.

        A body that slides until it hits something does not have "a shift": the same button moves
        it one cell here and nine there, so every distance is its own signature, no single one can
        reach the agreement `known` asks for, and the action reads as unlearnable no matter how
        many times it is pressed. What is constant is the direction. This reports that only when
        every non-zero move seen for the action lies along one axis in one sign AND at least two
        different distances have been seen; a single repeated distance is an ordinary step and is
        left to `expected_shift`.
        """
        moves = [shift for shift, _n in self.move_signatures(action)]
        if len(moves) < 2:
            return None
        units, mags = set(), set()
        for dr, dc in moves:
            if dr and dc:
                return None                     # diagonal or mixed: not a slide along an axis
            dist = abs(dr) + abs(dc)
            if not dist:
                continue
            units.add((dr // dist if dr else 0, dc // dist if dc else 0))
            mags.add(dist)
        if len(units) != 1 or len(mags) < 2:
            return None
        return next(iter(units))

    def slide_agreement(self, action: str, *, movers_only: bool = False) -> float:
        """Share of observations that are moves along this action's slide direction.

        Over readable observations by default; over the presses that actually moved the body when
        `movers_only`, which is the question to ask when the rest were blocked.
        """
        unit = self.slide_direction(action)
        if unit is None:
            return 0.0
        counts = self.counts.get(action, Counter())
        readable = {k: n for k, n in counts.items() if k not in UNINFORMATIVE
                    and (not movers_only or self._shift_of(k) is not None)}
        total = float(sum(readable.values()))
        if total <= 0:
            return 0.0
        agree = 0.0
        for key, n in readable.items():
            shift = self._shift_of(key)
            if shift is None:
                continue
            dist = abs(shift[0]) + abs(shift[1])
            if dist and (shift[0] // dist if shift[0] else 0,
                         shift[1] // dist if shift[1] else 0) == unit:
                agree += n
        return agree / total

    def confidence(self, action: str) -> float:
        return float(self.signature(action)["confidence"])

    # -- the same readings, in context --------------------------------------------------------

    def contexts(self, action: str) -> list[str]:
        return sorted(c for (a, c) in self.by_context if a == action)

    def context_signature(self, action: str, context: str) -> dict[str, Any]:
        """`signature` restricted to presses made in one body state."""
        c = self.by_context.get((action, context), Counter())
        total = float(sum(c.values()))
        if total <= 0:
            return {"counts": {}, "total": 0, "dominant": None, "confidence": 0.0, "shift": None}
        dominant, n = max(c.items(), key=lambda kv: (kv[1], kv[0]))
        return {"counts": dict(c), "total": int(total) if total.is_integer() else total,
                "dominant": dominant, "confidence": n / (total + 1.0),
                "shift": self._shift_of(dominant)}

    def known_in(self, action: str, context: str, *, min_obs: int = 2, min_agree: float = 0.8) -> bool:
        """Is the action's effect settled in this body state? Falls back to the pooled reading
        when this state has not been seen enough times to say anything on its own."""
        if not context:
            return self.known(action)
        c = self.by_context.get((action, context), Counter())
        readable = {k: n for k, n in c.items() if k not in UNINFORMATIVE}
        total = float(sum(readable.values()))
        if total < min_obs:
            return self.known(action)
        dom = max(readable, key=lambda k: (readable[k], k))
        return dom not in UNINFORMATIVE and float(readable[dom]) / total >= min_agree

    def shift_in(self, action: str, context: str) -> Optional[tuple[int, int]]:
        """The self shift this action produces in this body state, else the pooled one."""
        if context:
            sig = self.context_signature(action, context)
            if sig["total"] >= 2 and sig["shift"] is not None:
                return sig["shift"]
        return self.expected_shift(action)

    def context_helps(self, action: str) -> bool:
        """True when splitting by body state turns an unreadable action into a readable one.

        This is the whole justification for keeping the split: if the pooled reading already
        settles, the context adds nothing but a way to be wrong on thin evidence.
        """
        if self.known(action):
            return False
        seen = [c for c in self.contexts(action)
                if sum(self.by_context[(action, c)].values()) >= 2]
        return bool(seen) and all(self.known_in(action, c) for c in seen)

    def known(self, action: str, *, min_obs: int = 2, min_agree: float = 0.8,
              min_readable: float = 0.5) -> bool:
        """Do we know what this action does?

        These games are deterministic, so two observations that agree are real evidence, while
        the Laplace-smoothed `confidence` (dominant/(total+1)) tops out at 0.75 after three and
        would keep the commit gate shut forever. "Known" is stated directly instead: at least
        `min_obs` readable observations, at least `min_agree` of them showing the same effect,
        and at least `min_readable` of all observations readable at all.
        """
        counts = self.counts.get(action, Counter())
        # "other" is the absence of a reading, not a contradiction: the frame changed in ways this
        # map could not attribute to the self (an animation, another body, a moment of lost
        # identity). It neither proves nor disproves what the action does, so it is excluded from
        # both sides. Counting it as a dominant effect would open the gate on an action we cannot
        # plan with (on one re86 run A1 read "other" 29 times of 34); counting it against would
        # keep the gate shut on an action seen moving the body 23 times with no counter-evidence.
        readable = {k: n for k, n in counts.items() if k not in UNINFORMATIVE}
        total = float(sum(readable.values()))
        seen = float(sum(counts.values()))
        if total < min_obs:
            return False
        if seen and total / seen < min_readable:
            # Mostly blind: 5 clean readings in 34 presses is an action that only sometimes does
            # the thing, and a plan cannot lean on it. 23 in 34 is an action we can plan with.
            return False
        dom = max(readable, key=lambda k: (readable[k], k))
        if dom in UNINFORMATIVE:
            return False
        if float(readable[dom]) / total >= min_agree:
            return True
        # A blocked press is not the button contradicting itself. It is the button doing exactly
        # what it does, against a wall, and the passability map already accounts for it. Counting
        # it as disagreement is why control confidence climbed to 1.0 and fell back to 0.5 in
        # every ls20 run: in a game with walls a direction key is blocked often, and pooled
        # agreement can never stay above four fifths however reliable the button is.
        #
        # So: among the presses where the body actually went somewhere, do they agree? A button
        # that has never moved anything is untouched by this and still reads as "always blocked"
        # or "always a no-op", which is knowledge of a different kind.
        moves = {k: n for k, n in readable.items() if self._shift_of(k) is not None}
        moved = float(sum(moves.values()))
        if moved >= min_obs:
            best = max(moves.values())
            if best / moved >= min_agree:
                return True
            if self.slide_agreement(action, movers_only=True) >= min_agree:
                return True
        # A slide is knowledge even though no single distance dominates: the harness can say
        # exactly where the body ends up, because the planner simulates the travel against the
        # same wall map it plans with.
        return self.slide_agreement(action) >= min_agree

    def known_fraction(self, actions: Iterable[str]) -> float:
        """Share of the given actions whose effect is known. This is the control channel the
        commit gate reads, so it reaches 1.0 exactly when every control has been pinned down."""
        acts = [a for a in actions if a != "RESET"]
        if not acts:
            return 0.0
        return sum(1.0 for a in acts if self.known(a)) / len(acts)

    def readable_count(self, action: str) -> float:
        """Presses of this action the map could actually read. "other" is a frame it could not
        attribute, and no number of those teaches anything about the button."""
        c = self.counts.get(action, Counter())
        return float(sum(n for k, n in c.items() if k not in UNINFORMATIVE))

    def unmeasured(self, actions: Iterable[str], *, min_obs: int = 2,
                   max_presses: int = 6) -> list[str]:
        """Available actions the harness still cannot say anything about, least-seen first.

        re86 level 1, twelve turns in: A1 had been pressed sixteen times and moved the body the
        same way thirteen of them, A2 nine times; A3, A4 and A5 had two presses between them and
        one readable reading each. Three of the five controls were simply unknown, so control
        confidence sat at 0.4, and the planner was left with up and down. Nothing in the loop was
        trying to fix that: the model spends its turns on what is already working.

        Bounded by `max_presses`: some actions cannot be read however often they are pressed,
        because what they do is not something the map can attribute. Without the bound the harness
        would keep taking the turn to press them and never call the model again, which is what
        happened the first time this was wired in: a whole offline replay finished with zero model
        requests.
        """
        out = [a for a in actions if a != "RESET" and self.readable_count(a) < min_obs
               and sum(self.counts.get(a, Counter()).values()) < max_presses]
        return sorted(out, key=lambda a: (self.readable_count(a), a))

    def blocked_count(self, action: str) -> float:
        return float(self.counts.get(action, Counter()).get("blocked", 0))

    # -- carry over ----------------------------------------------------------------------------

    def render_contexts(self, max_rows: int = 6) -> str:
        """`A1 when body=abc: moved self (0,-1) x4 | A1 when body=xyz: moved self (1,0) x3`,
        shown only for actions the split actually rescues."""
        parts = []
        for action in self.actions():
            if not self.context_helps(action):
                continue
            for ctx in self.contexts(action):
                sig = self.context_signature(action, ctx)
                if not sig["dominant"] or sig["total"] < 2:
                    continue
                label = str(sig["dominant"]).replace("moved ", "moved self ")
                parts.append(f"{_short(action)} when body={ctx}: {label} x{_fmt_count(sig['total'])}")
        return " | ".join(parts[:max_rows])

    def inherit(self, other: "ControlMap", factor: float = 0.5) -> None:
        """Carry another map's counts in at reduced weight (cross-level prior)."""
        for action, c in other.counts.items():
            for key, n in c.items():
                self.counts[action][key] += n * factor
            self.inherited.add(action)
        for action, d in other.detail.items():
            for key, n in d.items():
                self.detail[action][key] += n * factor
        # The context split carries over too. A button whose meaning depends on which way the
        # body is facing does not stop depending on it at a level boundary, and `_on_level_change`
        # builds a fresh map and inherits into it -- so leaving this out emptied `by_context`
        # permanently from level 2 on: `context_helps` returned False for every action,
        # `render_contexts` returned "" for the rest of the game, and `known_in` fell back to the
        # pooled reading every time. The whole mechanism existed on level 1 only.
        for key, c in other.by_context.items():
            for sig, n in c.items():
                self.by_context[key][sig] += n * factor

    def demote(self, action: str, factor: float = 0.5) -> None:
        """Halve what is believed about one action, so it has to be read again.

        The critic's `demote_control` verdict: a prediction failed because the harness has the
        wrong idea of what a button does. Halving rather than clearing keeps the shape of the
        evidence while dropping `readable_count` below `known`'s bar, which is what puts the
        action back into `unmeasured()` and buys it a fresh probe.
        """
        for store in (self.counts, self.detail):
            if action in store:
                for key in list(store[action]):
                    store[action][key] *= factor
        for key in list(self.by_context):
            if key[0] == action:
                for sig in list(self.by_context[key]):
                    self.by_context[key][sig] *= factor

    def reset(self) -> None:
        self.counts.clear()
        self.detail.clear()
        self.inherited.clear()
        self.by_context.clear()

    # -- rendering -----------------------------------------------------------------------------

    def render(self) -> str:
        """`A1: moved self (-1,0) x4, blocked x2 | A2: ... | A5: no-op x3`."""
        parts = []
        for action in self.actions():
            c = self.counts[action]
            bits = []
            for key, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
                label = key.replace("moved ", "moved self ")
                if key == "other":
                    d = self.detail.get(action, {})
                    inner = ", ".join(f"{k} {_fmt_count(v)}" for k, v in sorted(d.items()) if v)
                    label = f"other ({inner})" if inner else "other"
                bits.append(f"{label} x{_fmt_count(n)}")
            tag = " (prior)" if action in self.inherited else ""
            parts.append(f"{_short(action)}: " + ", ".join(bits) + tag)
        return " | ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {"counts": {a: dict(c) for a, c in self.counts.items()},
                "detail": {a: dict(c) for a, c in self.detail.items()},
                "by_context": {f"{a}\u0000{ctx}": dict(c) for (a, ctx), c in self.by_context.items()},
                "inherited": sorted(self.inherited)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ControlMap":
        m = cls()
        for a, c in d.get("counts", {}).items():
            m.counts[a].update(c)
        for a, c in d.get("detail", {}).items():
            m.detail[a].update(c)
        for joined, c in d.get("by_context", {}).items():
            a, _, ctx = joined.partition("\u0000")
            m.by_context[(a, ctx)].update(c)
        m.inherited = set(d.get("inherited", []))
        return m
