"""Facts about the attempt that just ended, from the frames the harness holds.

Pure python, standard library only, no I/O. Two flags read them
(``tool_agent`` places the lines):

- **ARC3_DEATH_VERDICT** (``death_verdict_lines``): on the GAME OVER turn
  message, the budget bar's reading in the death frame and the actions of the
  attempt that the game undid. U tu93 L4 read an empty-bar budget death (bar
  64 -> 0 over the 20 actions of the attempt) as a ranged kill and spent ~72k
  tokens on it; Ucore r11l L2 died at the hidden fifth undone move of an
  attempt and, told "this action itself was fatal", hunted a contradiction for
  ~85k tokens.
- **ARC3_UNDONE_FACT** (``undone_lines``): in an ordinary turn message, the
  actions of the just-executed sequence that the game undid.

A budget bar is one colour in one edge band (the cells within ``border`` of
that grid edge, the no-op guard's border) whose cell count, over the game's
history, falls within attempts on a steady share of actions, rarely rises
within one (ls20's bar refills when an item is taken) and is restored when a
RESET or a new level starts an attempt. It is identified from the bands
without their corner squares, so the few cells of a bar that reach into a
side band never pass for one. An undone action left the board outside the
border band as it was before the action while its animation changed cells
there and changed them back.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

Grid = Sequence[Sequence[int]]

# An empty bar is read as the budget running out only when it emptied at
# most this many actions before the game over (g50t's and ka59's engines end
# the attempt up to two actions after the last cell goes).
MAX_EMPTY_GAP = 3
# A bar falls on at least this share of the actions of the attempts that
# start with it (the slowest bar in the public games, re86's, falls on ~1
# action in 4) and rises within an attempt at most this share of the times it
# falls.
MIN_FALL_SHARE = 0.15
MAX_RISE_SHARE = 0.25
MIN_FALLS = 3
MIN_CELLS = 3
MAX_UNDONE_LINES = 2
_CLICK_RE = re.compile(r"row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)", re.I)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def band_counts(grid: Grid, border: int, *, corners: bool = True) -> Counter:
    """Cells of each colour in each edge band: {(edge, colour): count}.

    With ``corners`` the bands overlap at the corner squares, as the no-op
    guard's border frame does, so a band's count is every cell of that edge
    strip; without them each band is its strip minus the two corner squares
    (``border`` x ``border``) it shares with the side bands."""
    counts: Counter = Counter()
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if border <= 0 or rows <= 2 * border or cols <= 2 * border:
        return counts
    inner = slice(0, cols) if corners else slice(border, cols - border)
    for r in range(rows):
        row = grid[r]
        edge_row = r < border or r >= rows - border
        if edge_row:
            edge = "top" if r < border else "bottom"
            for value in row[inner]:
                counts[(edge, int(value))] += 1
            if not corners:
                continue
        for c in range(border):
            counts[("left", int(row[c]))] += 1
            counts[("right", int(row[cols - 1 - c]))] += 1
    return counts


def _result(entry: Any) -> dict:
    return getattr(entry, "result", None) or {}


def _grid(entry: Any) -> Grid | None:
    frame = getattr(entry, "frame", None)
    return getattr(frame, "grid", None) if frame is not None else None


def _level(entry: Any) -> Any:
    frame = getattr(entry, "frame", None)
    return getattr(frame, "level", None) if frame is not None else None


def _name(entry: Any) -> str:
    return str(_result(entry).get("action_name") or getattr(entry, "action", "") or "").strip().upper()


def is_attempt_start(history: Sequence[Any], index: int) -> bool:
    """Whether the board after entry ``index`` starts an attempt: the seeded
    initial frame, a RESET (the model's, the automatic one after a game over,
    the warmup one), an action that completed a level, or one under which the
    level changed (``runtime_state.actions_since_reset``'s boundaries)."""
    if index <= 0:
        return True
    entry = history[index]
    name = _name(entry)
    if not name or name == "RESET" or _result(entry).get("level_completed"):
        return True
    return _level(history[index - 1]) != _level(entry)


def attempt_start(history: Sequence[Any], end: int) -> int:
    """Index of the entry whose board started the attempt that ``end`` is in."""
    index = max(0, min(end, len(history) - 1))
    while index > 0 and not is_attempt_start(history, index):
        index -= 1
    return index


def action_number(entry: Any, fallback: int) -> int:
    try:
        return int(_result(entry).get("action_num"))
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------- budget bar


@dataclass(frozen=True)
class Bar:
    edge: str
    colour: int
    falls: int


def find_bar(history: Sequence[Any], end: int, border: int) -> Bar | None:
    """The budget bar seen over ``history[:end + 1]``, or None.

    Counts are the bands without their corner squares. Within attempts a key
    must fall at least ``MIN_FALLS`` times, on at least ``MIN_FALL_SHARE`` of
    the actions of the attempts it was present at the start of, rise at most
    ``MAX_RISE_SHARE`` of the times it falls, and be restored (higher at an
    attempt start than at the end of the attempt before) at least once; the
    bar is the key with the most falls that is present at the start of the
    attempt ``end`` is in. (A few cells that empty early and stay empty for
    the rest of a long attempt, sk48's, fall on too small a share.)"""
    counts: list[Counter] = []
    for entry in history[: end + 1]:
        grid = _grid(entry)
        counts.append(band_counts(grid, border, corners=False) if grid else Counter())
    rises: Counter = Counter()
    falls: Counter = Counter()
    present: Counter = Counter()
    restored: Counter = Counter()
    opened = [key for key, value in counts[0].items() if value > 0] if counts else []
    for index in range(1, len(counts)):
        before, after = counts[index - 1], counts[index]
        if is_attempt_start(history, index):
            for key, value in after.items():
                if value > before.get(key, 0):
                    restored[key] += 1
            opened = [key for key, value in after.items() if value > 0]
            continue
        present.update(opened)
        for key in set(before) | set(after):
            old, new = before.get(key, 0), after.get(key, 0)
            if new > old:
                rises[key] += 1
            elif new < old:
                falls[key] += 1
    best: tuple | None = None
    start = counts[attempt_start(history, end)] if counts else Counter()
    for key, fell in falls.items():
        if fell < MIN_FALLS or not restored.get(key) or start.get(key, 0) < MIN_CELLS:
            continue
        if fell < MIN_FALL_SHARE * present[key] or rises[key] > MAX_RISE_SHARE * fell:
            continue
        rank = (fell, start.get(key, 0))
        if best is None or rank > best[0]:
            best = (rank, key)
    if best is None:
        return None
    (fell, _), (edge, colour) = best
    return Bar(edge=edge, colour=colour, falls=fell)


@dataclass(frozen=True)
class BarReading:
    edge: str
    colour: int
    start: int     # cells at the attempt's first board
    before: int    # cells on the board before the fatal action
    death: int     # cells in the death frame
    actions: int   # actions of the attempt, the fatal one included
    used: int      # cells lost over the attempt (refills not netted out)

    @property
    def per_action(self) -> float:
        return self.used / self.actions if self.actions else 0.0

    @property
    def empty(self) -> bool:
        return self.death == 0


def read_bar(history: Sequence[Any], death: int, border: int) -> BarReading | None:
    """The budget bar over the attempt that ended at entry ``death``, counted
    over its whole edge strip (corner squares included).

    The bar is identified from everything up to the board that restarted the
    level after the death (``death + 1`` when it is there), so a first death
    already shows the refill."""
    if death <= 0 or death >= len(history):
        return None
    end = death + 1 if death + 1 < len(history) and is_attempt_start(history, death + 1) else death
    bar = find_bar(history, end, border)
    if bar is None:
        return None
    start = attempt_start(history, death)
    actions = death - start
    if actions <= 0:
        return None
    key = (bar.edge, bar.colour)
    series = []
    for index in range(start, death + 1):
        grid = _grid(history[index])
        series.append(band_counts(grid, border).get(key, 0) if grid else 0)
    used = sum(max(0, old - new) for old, new in zip(series, series[1:]))
    if series[-1] > series[-2]:
        # refilled by the fatal action: su15's death frame shows a bar of 50-60
        # cells right after one of 2-12, so the death frame's count says
        # nothing about the budget
        return None
    if series[-1] == 0:
        # the last time it emptied (a refill can follow an earlier empty spell)
        emptied = max((i for i, value in enumerate(series) if value > 0), default=-1) + 1
        if len(series) - 1 - emptied > MAX_EMPTY_GAP:
            return None  # empty long before the game over: no reading of its cause
    return BarReading(
        edge=bar.edge, colour=bar.colour, start=series[0], before=series[-2],
        death=series[-1], actions=actions, used=used,
    )


def bar_line(reading: BarReading) -> str:
    rate = f"about {reading.per_action:.1f}"
    if reading.empty:
        return (
            f"Budget bar ({reading.edge} edge): {reading.start} cells at the start of this attempt, "
            f"0 in the death frame, {rate} fewer per action: the bar is empty in the death frame, so "
            f"this game over is the action budget running out on action {reading.actions} of this "
            "attempt."
        )
    per_action = f", {rate} per action" if reading.used else ""
    return (
        f"Budget bar ({reading.edge} edge): {reading.death} of {reading.start} cells left in the "
        f"death frame{per_action}: not a budget death."
    )


# ---------------------------------------------------------------- undone actions


def _click(display: str) -> tuple[int, int] | None:
    match = _CLICK_RE.search(display or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _bbox(entry: Any) -> list[int] | None:
    animation = _result(entry).get("animation") or {}
    try:
        if int(animation.get("transient_pixels") or 0) <= 0:
            return None
    except (TypeError, ValueError):
        return None
    bbox = animation.get("transient_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return [int(v) for v in bbox]


def is_undone(entry: Any, border: int, *, skip_click_flash: bool = True) -> bool:
    """The action left the board outside the border band as it was, while its
    animation changed cells there and changed them back.

    ``skip_click_flash``: a click whose changed-back cells are centred on the
    clicked cell is the game marking the click (su15's ripple, lf52's
    cursor box), not a move it undid."""
    result = _result(entry)
    if result.get("gameplay_changed") is not False or result.get("game_over"):
        return False
    if _name(entry) == "RESET":
        return False
    bbox = _bbox(entry)
    if bbox is None:
        return False
    grid = _grid(entry)
    rows = len(grid) if grid else 64
    cols = len(grid[0]) if grid else 64
    top, left, bottom, right = bbox
    if bottom < border or top >= rows - border or right < border or left >= cols - border:
        return False  # only the border band blinked
    click = _click(str(getattr(entry, "action", "")))
    if skip_click_flash and click is not None:
        row, col = click
        if abs((top + bottom) / 2 - row) <= 1.5 and abs((left + right) / 2 - col) <= 1.5:
            return False
    return True


def undone_in_attempt(history: Sequence[Any], end: int, border: int) -> list[int]:
    """Indexes of the undone actions from the attempt's start through ``end``."""
    start = attempt_start(history, end)
    return [i for i in range(start + 1, end + 1) if is_undone(history[i], border)]


def _span(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _numbers(values: Sequence[int]) -> str:
    text = [str(v) for v in values]
    if len(text) <= 2:
        return " and ".join(text)
    return ", ".join(text[:-1]) + " and " + text[-1]


def undone_attempt_line(history: Sequence[Any], death: int, border: int) -> str:
    """The GAME OVER line: the attempt's undone actions, or ""."""
    undone = undone_in_attempt(history, death - 1, border) if death > 0 else []
    if not undone:
        return ""
    numbers = [action_number(history[i], i) for i in undone]
    fatal = action_number(history[death], death)
    count = len(numbers)
    if count == 1:
        what, label = "1 action that the game undid (it animated", "action"
    else:
        what, label = f"{count} actions that the game undid (they animated", "actions"
    return (
        f"This attempt had {what}, then the board returned to how it was): {label} "
        f"{_numbers(numbers)}; the game over came on action {fatal}."
    )


def death_verdict_lines(history: Sequence[Any], border: int) -> tuple[list[str], bool]:
    """The verdict lines for a GAME OVER turn and whether they replace the
    BAR RULE sentence (an empty bar, or at least one undone action).

    ``history`` ends with the automatic RESET after the death; the death is
    the last entry with ``game_over``."""
    death = next(
        (i for i in range(len(history) - 1, 0, -1) if _result(history[i]).get("game_over")), None
    )
    if death is None:
        return [], False
    lines: list[str] = []
    reading = read_bar(history, death, border)
    if reading is not None:
        lines.append(bar_line(reading))
    undone = undone_attempt_line(history, death, border)
    if undone:
        lines.append(undone)
    return lines, bool(undone) or bool(reading is not None and reading.empty)


def undone_lines(
    history: Sequence[Any], first_action: int | None, last_action: int | None, border: int,
    *, min_ordinal: int = 1, max_lines: int = MAX_UNDONE_LINES,
) -> list[str]:
    """One line per undone action of the sequence ``first_action``..``last_action``
    (action numbers) in the attempt still running, the latest ``max_lines``;
    an action is named only from the ``min_ordinal``-th undone action of its
    attempt. Actions before a RESET or a level change inside the sequence
    belong to an attempt that is over and are left out."""
    if first_action is None or last_action is None or not history:
        return []
    live = attempt_start(history, len(history) - 1)
    picked: list[str] = []
    for index in range(len(history) - 1, live, -1):
        number = action_number(history[index], index)
        if number < first_action:
            break
        if number > last_action or not is_undone(history[index], border):
            continue
        nth = len(undone_in_attempt(history, index, border))
        if nth < min_ordinal:
            continue
        top, left, bottom, right = _bbox(history[index]) or [0, 0, 0, 0]
        picked.append(
            f"Action {number} {history[index].action} was undone by the game: it animated, then the "
            f"board returned to how it was ({ordinal(nth)} undone action in this attempt; cells that "
            f"changed during the animation: rows {_span(top, bottom)}, cols {_span(left, right)})."
        )
        if len(picked) >= max_lines:
            break
    return list(reversed(picked))
