"""The deterministic delta: what actually changed, at three granularities, plus its rendering.

`compute_delta` is one of the two trusted inputs of the whole harness (the other is
`perceive`). Everything a rule is scored against, and everything a prediction is compared
with, comes from here -- so it is a pure function of two `Percept`s and the tracker's
events, with no model in the loop and no heuristics that could quietly disagree with the
frame.

Three granularities, because a prediction can be right at one and wrong at another:

* **cells** -- how many changed, from which colour to which, and a hash of the mask.
* **objects** -- the tracker's events, regrouped into the shape a prompt and a rule want.
* **scalars** -- every registry name whose integer value moved, as (before, after).

Plus **tags**, the four cheap facts that decide a mode: `noop`, `blocked`, `hud_only`,
`veiled`. Each is a claim the harness is willing to defend, which is why `blocked` is
deliberately conservative -- see its comment in `compute_delta`.

The two renderers exist because the model never sees the grid (plan, "What the model
sees"). It sees an object table of at most 24 rows and a one-line diff. Both are budgeted:
`render_object_table` folds runs of identical static blocks into a single row so a board of
forty identical wall tiles costs one line, not forty.
"""

from __future__ import annotations

from typing import Optional

from .perception import band_cells, fnv1a
from .tracker import ObjectTracker, _tid_key
from .types import (Cell, CellDelta, Delta, Event, ObjectDelta, Percept, Track, TrackState,
                    colour_letter)


def compute_delta(prev: Percept, cur: Percept, events: list[Event], tracker: ObjectTracker,
                  *, action: Optional[str], acted: bool) -> Delta:
    """Everything that changed between two frames, as one immutable record.

    `events` must be the list `tracker.update(cur, ...)` just returned; passing a stale
    list would silently describe the wrong step, so the caller keeps them adjacent.
    """
    changed: set[Cell] = set()
    by_pair: dict[str, int] = {}
    height = min(len(prev.grid), len(cur.grid))
    for r in range(height):
        before_row, after_row = prev.grid[r], cur.grid[r]
        width = min(len(before_row), len(after_row))
        for c in range(width):
            a, b = int(before_row[c]), int(after_row[c])
            if a == b:
                continue
            changed.add((r, c))
            key = "%s>%s" % (colour_letter(a), colour_letter(b))
            by_pair[key] = by_pair.get(key, 0) + 1

    cells = CellDelta(
        n=len(changed),
        by_pair=dict(sorted(by_pair.items())),
        changed=frozenset(changed),
        mask_hash=fnv1a(repr(tuple(sorted(changed)))),
    )

    objects = ObjectDelta(
        moved=[(e.tid, (int(e.detail["d"][0]), int(e.detail["d"][1])))
               for e in events if e.kind == "MOVE"],
        recoloured=[(e.tid, int(e.detail["from"]), int(e.detail["to"]))
                    for e in events if e.kind == "RECOLOUR"],
        reshaped=[e.tid for e in events if e.kind == "RESHAPE"],
        appeared=[e.tid for e in events if e.kind == "SPAWN"],
        vanished=[e.tid for e in events if e.kind == "VANISH"],
        occluded=[e.tid for e in events if e.kind == "OCCLUDE"],
        reappeared=[e.tid for e in events if e.kind == "REAPPEAR"],
        consumed=[e.tid for e in events if e.kind == "CONSUME"],
        ambiguous=sorted((t.tid for t in tracker.live()
                          if t.state is TrackState.AMBIGUOUS), key=_tid_key),
    )

    scalars: dict[str, tuple[int, int]] = {}
    for name in sorted(set(prev.scalars) | set(cur.scalars)):
        if name in ("step",):
            continue
        before = int(prev.scalars.get(name, 0))
        after = int(cur.scalars.get(name, 0))
        if before != after:
            scalars[name] = (before, after)
    # Per-track positions are not a property of a frame, so they are reconstructed from the
    # tracker's own before/after snapshot rather than from either percept.
    was, now = tracker.previous_positions(), tracker.positions()
    for name in sorted(set(was) | set(now)):
        before, after = was.get(name), now.get(name)
        if before is None or after is None or before == after:
            continue
        scalars[name] = (int(before), int(after))

    hud = band_cells(cur.hud_bands)
    tags: list[str] = []
    if cells.n == 0:
        tags.append("noop")
    if cur.veil:
        tags.append("veiled")
    if cells.n and hud and changed <= hud:
        tags.append("hud_only")
    # `blocked` is the harness saying "you pressed a direction and your object did not go".
    # It is deliberately conservative: it needs an executed action, a self we have already
    # identified, no MOVE for it, and no change outside the HUD. A frame where something
    # else moved is not evidence that we were blocked, and calling it that would teach a
    # passability map a wall that is not there -- the exact inverted-mask failure the plan
    # records four times.
    self_tid = tracker.self_tid
    if acted and self_tid and not any(e.kind == "MOVE" and e.tid == self_tid for e in events):
        if not changed or (hud and changed <= hud) or cells.n == 0:
            tags.append("blocked")
    if cur.level != prev.level:
        tags.append("level_up")

    touched = {e.tid for e in events}
    unchanged = sum(1 for t in tracker.live() if t.tid not in touched)

    return Delta(cells=cells, objects=objects, scalars=scalars, events=list(events),
                 tags=tags, unchanged=unchanged)


# --- rendering -----------------------------------------------------------------------------


def _row(track: Track, self_tid: Optional[str]) -> str:
    r0, c0, r1, c1 = track.last_bbox
    bits = ["%s %s %dpx @(%d,%d) %dx%d %s" % (
        track.tid, colour_letter(track.colour), track.blob.pixels if track.blob else 0,
        r0, c0, r1 - r0 + 1, c1 - c0 + 1, track.klass)]
    group = getattr(track, "group", None)
    if group is not None:
        # One token that says "this does not move on its own" -- cheaper than repeating the
        # displacement on every member's row, and it is what the model needs to plan with.
        bits.append("g%d" % group)
    if track.state is TrackState.OCCLUDED:
        bits.append("hidden")
    elif track.state is TrackState.AMBIGUOUS:
        bits.append("ambiguous")
    if track.flags.get("blinking"):
        bits.append("blinking")
    if track.label:
        bits.append('"%s"' % track.label[:24])
    if self_tid and track.tid == self_tid:
        bits.append("<-self")
    return " ".join(bits)


def render_object_table(tracker: ObjectTracker, percept: Percept, *, max_rows: int = 24,
                        self_tid: Optional[str] = None) -> str:
    """The object table the model sees instead of the grid, at most `max_rows` lines.

    Rows read `t3 c 9px @(12,30) 3x3 static`. Identical static blocks -- same colour, same
    size, same class, three or more of them -- collapse to one line `c x41 all static 5x5`,
    because forty wall tiles listed individually would eat the whole budget and say nothing
    the folded line does not. Everything interesting (the self, movers, anything occluded,
    ambiguous or blinking) is listed individually and sorted first, so the fold can only
    ever drop repetition.
    """
    self_tid = self_tid if self_tid is not None else tracker.self_tid
    live = tracker.live()
    if not live:
        return "none (step %d, level %d)" % (percept.step, percept.level)

    folded: dict[tuple[int, tuple[int, int], str], list[Track]] = {}
    singles: list[Track] = []
    for track in live:
        r0, c0, r1, c1 = track.last_bbox
        size = (r1 - r0 + 1, c1 - c0 + 1)
        interesting = (track.tid == self_tid or track.state is not TrackState.LIVE
                       or track.flags.get("blinking") or track.label
                       or track.klass not in ("static", "background"))
        if interesting:
            singles.append(track)
        else:
            folded.setdefault((track.colour, size, track.klass), []).append(track)

    # The self body first, then anything that moves or is labelled, then the rest: the table is
    # clipped at max_rows, and the controlled object must never fall below the fold (seen live:
    # the model "probed t1" for two turns because the self group was among the +10 not shown).
    self_group = set(tracker.self_group()) if hasattr(tracker, "self_group") and self_tid else set()

    def _rank(track: Track) -> tuple[int, int]:
        num = int(track.tid[1:]) if track.tid[1:].isdigit() else 0
        if track.tid == self_tid or track.tid in self_group:
            return (0, num)
        if track.klass in ("controllable", "autonomous") or any(e.kind == "MOVE" for e in track.events[-3:]):
            return (1, num)
        if track.label or track.state is not TrackState.LIVE:
            return (2, num)
        return (3, num)

    singles.sort(key=_rank)
    lines: list[str] = []
    for track in singles:
        lines.append(_row(track, self_tid))
    extra: list[str] = []
    for key in sorted(folded):
        group = folded[key]
        colour, size, klass = key
        if len(group) >= 3:
            extra.append("%s x%d all %s %dx%d" % (colour_letter(colour), len(group), klass,
                                                  size[0], size[1]))
        else:
            extra.extend(_row(t, self_tid) for t in group)
    lines.extend(extra)

    # No "OBJ" token: the prompt already prefixes this block with one, and repeating it
    # spends a token on saying the same thing twice.
    head = "step %d level %d, %d live" % (percept.step, percept.level, len(live))
    group = tracker.self_group()
    if len(group) > 1:
        head += ", self = %s" % "+".join(group)
    if len(lines) > max_rows:
        dropped = len(lines) - max_rows
        lines = lines[:max_rows]
        lines.append("(+%d more not shown)" % dropped)
    return "\n".join([head] + lines)


def render_diff(delta: Delta) -> str:
    """The one-line diff, e.g. `moved t1 (-1,0); gone t7; recoloured t2 Y>R; cells 18; unchanged 11`.

    One line, because it is read every turn and the budget for the whole state block is
    ~1200 tokens. Order is fixed (movement, then identity changes, then arrivals and
    departures, then the cell count) so a model reading two consecutive turns compares like
    with like.

    Tracks that moved by the same displacement are folded into one entry
    (`moved t3,t4,t5,t7,t9 (0,-1)`). A rigid body of five components otherwise spends five
    entries repeating one fact, and a camera pan spends one per object on screen.
    """
    parts: list[str] = []
    by_shift: dict[tuple[int, int], list[str]] = {}
    for tid, d in delta.objects.moved:
        by_shift.setdefault(d, []).append(tid)
    for d in sorted(by_shift):
        parts.append("moved %s (%d,%d)" % (",".join(by_shift[d]), d[0], d[1]))
    for tid, before, after in delta.objects.recoloured:
        parts.append("recoloured %s %s>%s" % (tid, colour_letter(before), colour_letter(after)))
    if delta.objects.reshaped:
        parts.append("reshaped " + ",".join(delta.objects.reshaped))
    if delta.objects.appeared:
        parts.append("new " + ",".join(delta.objects.appeared))
    if delta.objects.consumed:
        parts.append("consumed " + ",".join(delta.objects.consumed))
    if delta.objects.vanished:
        parts.append("gone " + ",".join(delta.objects.vanished))
    if delta.objects.occluded:
        parts.append("hidden " + ",".join(delta.objects.occluded))
    if delta.objects.reappeared:
        parts.append("back " + ",".join(delta.objects.reappeared))
    if delta.objects.ambiguous:
        parts.append("ambiguous " + ",".join(delta.objects.ambiguous))
    parts.append("cells %d" % delta.cells.n)
    parts.append("unchanged %d" % delta.unchanged)
    if delta.tags:
        parts.append("[" + " ".join(delta.tags) + "]")
    return "; ".join(parts)
