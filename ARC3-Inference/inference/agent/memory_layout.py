"""Memory in plain words, and in every request. Pure python, no I/O.

Two things the user asked for on 2026-10-10 after the memory audit (PR #148):

- **ARC3_MEM_LEGIBLE**: every memory line in plain words, one fact per line:
  full action names, ``x`` counts, clicks as ``click (row, col)``, colour
  names, labels. ``"L1: 17 acts [U L D11 L4], spent 25, 0 deaths"`` becomes
  ``"Level 1: cleared in 17 actions (25 spent, no deaths). Route: UP, LEFT,
  DOWN x11, LEFT x4."``; ``"UP: moved 5, changed 17"`` becomes ``"UP (22 presses
  on this level): moved cells 5 times, changed cell colours 17 times."``
- **ARC3_MEM_EVERY_CALL**: the memory in every request, in two places
  (``tool_agent`` places them):

  * ``render_previous_levels``: the ``<previous_levels>`` message right after
    the system prompt, one entry per cleared level. It is drawn only when a
    level is committed (cleared and its recap settled) and at trims, so every
    other request extends the previous one byte for byte.
  * ``render_this_level``: the ``<this_level>`` block at the end of each turn
    message, facts about the current level only. It sits in the newest
    message, so it never breaks the cached prefix.

The counts come from ``memory_facts.FactRecorder``; nothing here reads frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from inference.agent.memory_facts import (
    ANIM,
    CHANGE,
    CLEAR,
    DEATH,
    HUD,
    NOOP,
    SHIFT,
    STILL,
    _CLASS_ORDER,
    cut_words,
    parse_click,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS

# The legend the system prompt gives for the board's letters.
COLOUR_NAMES = {
    "W": "white", "w": "light gray", "g": "gray", "G": "dark gray", "c": "charcoal", "B": "black",
    "M": "magenta", "P": "pink", "R": "red", "b": "blue", "S": "sky blue", "Y": "yellow",
    "O": "orange", "r": "dark red", "N": "light green", "p": "purple",
}
assert set(COLOUR_NAMES) == set(ARC_COLOR_CHARS)

# What one action did, in words (memory_facts.classify's classes).
EFFECT_WORDS = {
    CLEAR: "cleared the level",
    DEATH: "ended in GAME OVER",
    SHIFT: "moved cells",
    CHANGE: "changed cell colours",
    ANIM: "animated and changed nothing",
    HUD: "changed only the edge",
    NOOP: "did nothing",
}

def effect_words_v2(border: int) -> dict[str, str]:
    """ARC3_MEM_EFFECT_WORDS: what each class says, without implying the
    classes exclude each other. ``classify`` puts an action that moved a piece
    and changed any colour count under CHANGE alone (52% of run B's CHANGE
    actions moved an object), and an animation that left the edge changed
    under ANIM; ``border`` is the band the classes crop (ARC3_MEM_STATE_BORDER)."""
    band = f"the {int(border)}-cell border"
    return {
        **EFFECT_WORDS,
        SHIFT: "moved cells (no colour count changed)",
        CHANGE: "changed colour counts (cells may also have moved)",
        ANIM: f"animated, then left the board inside {band} as it was",
        HUD: f"changed only cells in {band}",
    }


PREVIOUS_TAG = "previous_levels"
THIS_LEVEL_TAG = "this_level"
# ARC3_MEM_APPEND_COMMIT: a newly cleared level's entry, in the newest message
# until it is folded into <previous_levels> at the next trim
CLEARED_TAG = "cleared_level"
# Size caps (chars). The this-level block stays near 300 tokens (Qwen's
# tokenizer averages ~3.6 chars per token on this text).
HEAD_CHARS = 6000
LEVEL_CHARS = 1100
ROUTE_CHARS = 240
# ARC3_MEM_ROUTE_FORMAT=readable: colour names make a click longer, so the
# readable route's default cap is 320. On run B's 108 entries that cuts 23
# routes (240 cut 24 of B's), 8,665 route tokens against B's 7,952, and every
# final <previous_levels> block stays under HEAD_CHARS (max 5,789 chars).
READABLE_ROUTE_CHARS = 320
# A cut readable route keeps its last runs: the model quoted route ends ("The
# route ended with 'click (38,21), LEFT x2'", bp35; tn36's L6 tail past 160
# chars) to recall how it won.
READABLE_TAIL_RUNS = 2
NOTE_CHARS = 200
# Per level in <previous_levels>: notes kept, newest first by kind.
NOTES_PER_LEVEL = 4


# ---------------------------------------------------------------- words


def colour_name(letter: str) -> str:
    return COLOUR_NAMES.get(str(letter), f"colour {letter}")


def times(n: int) -> str:
    n = int(n)
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


def plural(n: int, word: str, words: str | None = None) -> str:
    n = int(n)
    return f"{n} {word if n == 1 else (words or word + 's')}"


def deaths_words(n: int) -> str:
    return "no deaths" if int(n) == 0 else plural(n, "death")


def action_words(sig: str) -> str:
    """``UP``, or ``click (12, 30)`` for a click (row, column)."""
    cell = parse_click(sig)
    if cell is not None:
        return f"click ({cell[0]}, {cell[1]})"
    return str(sig)


def key_words(key: str) -> str:
    """An effect-table key: ``UP``, or ``clicks on red cells`` for ``MOUSE@R``."""
    if str(key).startswith("MOUSE@"):
        return f"clicks on {colour_name(key.split('@', 1)[1])} cells"
    return str(key)


def _word_runs(words: Sequence[str]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for word in words:
        if out and out[-1][0] == word:
            out[-1] = (word, out[-1][1] + 1)
        else:
            out.append((word, 1))
    return out


def _runs(sigs: Sequence[str]) -> list[tuple[str, int]]:
    return _word_runs([action_words(sig) for sig in sigs])


def readable_action_words(sig: str, key: str = "") -> str:
    """ARC3_MEM_ROUTE_FORMAT=readable: ``UP``, or ``click (48,57) red`` for a
    click, named by the colour of the clicked cell before the click (its
    effect key ``MOUSE@R``), in the ``(row,col)`` form the model writes."""
    cell = parse_click(sig)
    if cell is None:
        return str(sig)
    word = f"click ({cell[0]},{cell[1]})"
    letter = str(key).split("@", 1)[1] if str(key).startswith("MOUSE@") else ""
    return f"{word} {COLOUR_NAMES[letter]}" if letter in COLOUR_NAMES else word


def route_words(sigs: Sequence[str], max_chars: int = ROUTE_CHARS, *,
                keys: Sequence[str] | None = None) -> str:
    """``RIGHT x2, SPACE, DOWN x8, click (12, 30)``; cut at a whole run with
    ``... and N more actions`` when longer than ``max_chars``. With ``keys``
    (ARC3_MEM_ROUTE_FORMAT=readable, one effect key per action) clicks read
    ``click (12,30) red`` and repeat as ``click (12,30) red x4``."""
    if keys is None:
        runs = _runs(sigs)
    else:
        keys = [*keys, *[""] * (len(sigs) - len(keys))]
        runs = _word_runs([readable_action_words(sig, key) for sig, key in zip(sigs, keys)])
    parts = [word if n == 1 else f"{word} x{n}" for word, n in runs]
    full = ", ".join(parts)
    if len(full) <= max_chars:
        return full
    if keys is not None:
        return _cut_keeping_tail(parts, runs, len(sigs), max_chars)
    shown: list[str] = []
    covered = 0
    total = len(sigs)
    for part, (_, n) in zip(parts, runs):
        rest = total - covered - n
        tail = f", ... and {plural(rest, 'more action')}"
        if len(", ".join([*shown, part])) + len(tail) > max_chars:
            break
        shown.append(part)
        covered += n
    rest = total - covered
    return ", ".join([*shown, f"... and {plural(rest, 'more action')}"]) if shown else f"{plural(total, 'action')}"


def _cut_keeping_tail(parts: list[str], runs: list[tuple[str, int]], total: int, max_chars: int) -> str:
    """A readable route over ``max_chars``: the first whole runs that fit,
    ``... N more actions ...``, then the last ``READABLE_TAIL_RUNS`` runs
    (fewer when they do not fit); never a cut inside a run."""
    for tail_runs in range(min(READABLE_TAIL_RUNS, len(runs) - 1), -1, -1):
        tail = parts[len(parts) - tail_runs:] if tail_runs else []
        tail_n = sum(n for _, n in runs[len(runs) - tail_runs:]) if tail_runs else 0
        head: list[str] = []
        covered = 0
        for part, (_, n) in zip(parts[: len(parts) - tail_runs], runs):
            gap = f"... {plural(total - covered - n - tail_n, 'more action')} ..."
            if len(", ".join([*head, part, gap, *tail])) > max_chars:
                break
            head.append(part)
            covered += n
        text = ", ".join([*head, f"... {plural(total - covered - tail_n, 'more action')} ...", *tail])
        if head and len(text) <= max_chars:
            return text
    return f"{plural(total, 'action')} ..."


def counts_words(counts: Sequence[tuple[str, int]], words: dict[str, str] | None = None) -> str:
    """``moved cells 10 times, did nothing twice``. With ``words``
    (ARC3_MEM_EFFECT_WORDS) the count comes first, ``10 times moved cells
    (no colour count changed)``: after "... as it was" a trailing "once"
    would read as part of the phrase."""
    if words is None:
        return ", ".join(f"{EFFECT_WORDS.get(cls, cls)} {times(n)}" for cls, n in counts)
    return ", ".join(f"{times(n)} {words.get(cls, cls)}" for cls, n in counts)


# ---------------------------------------------------------------- legible facts


def solved_text(level: int, path: Sequence[str], spent: int, deaths: int, *,
                route_chars: int = ROUTE_CHARS, winning: int | None = None,
                keys: Sequence[str] | None = None) -> str:
    """``Level 2: cleared in 16 actions (23 spent, no deaths). Route: RIGHT x2, ...``.

    ``route_chars`` 0 leaves the route out. ``winning`` (ARC3_MEM_PATH): the
    winning attempt's own loop-free length, when the route is a shorter one
    on record. ``keys``: the readable route (``route_words``)."""
    text = (f"Level {level}: cleared in {plural(len(path), 'action')} "
            f"({int(spent)} spent, {deaths_words(deaths)}).")
    if route_chars > 0 and path:
        text += f" Route: {route_words(path, route_chars, keys=keys)}."
    if winning:
        text += f" The winning attempt itself took {plural(winning, 'action')}."
    return text


def solved_words(level: int, path: Sequence[str], spent: int, deaths: int, attempts: int, *,
                 route_chars: int = ROUTE_CHARS, keys: Sequence[str] | None = None) -> str:
    """ARC3_MEM_HEAD_WORDS: ``Level 2: cleared. 20 actions spent in 2
    attempts, no deaths. Winning route without loops (12 actions): ...``.
    Every count says what it counts ("cleared in 12 actions (20 spent)" was
    read as spend summed over attempts). ``route_chars`` 0 keeps only the
    route's length."""
    text = (f"Level {level}: cleared. {plural(spent, 'action')} spent in {plural(attempts, 'attempt')}, "
            f"{deaths_words(deaths)}.")
    length = plural(len(path), "action")
    if route_chars > 0 and path:
        text += f" Winning route without loops ({length}): {route_words(path, route_chars, keys=keys)}."
    else:
        text += f" Winning route without loops: {length}."
    return text


def hazard_text(level: int, sig: str, first_index: int, attempt: int, length: int, *,
                act: int | None = None, first_act: int | None = None) -> str:
    """How an attempt ended, never why (memory_facts.hazard_line in words).

    ``act`` and ``first_act`` (ARC3_MEM_SEEN_GAME_ACTION): the game's action
    numbers of the fatal action and of the board it was played from, given
    first, with the attempt's own count after them."""
    if act is not None:
        where = ("the attempt's start board" if first_index <= 0 or first_act is None
                 else f"the board first reached after action {first_act} (action {first_index} of that attempt)")
        return (f"Level {level}: attempt {attempt} ended in GAME OVER at action {act} "
                f"(action {length} of that attempt). The last action was {action_words(sig)}, from {where}.")
    where = ("the attempt's start board" if first_index <= 0
             else f"the board first reached after action {first_index} of that attempt")
    return (f"Level {level}: attempt {attempt} ended in GAME OVER after {plural(length, 'action')}. "
            f"The last action was {action_words(sig)}, from {where}.")


def first_effect_words(key: str, cls: str, sig: str) -> str:
    who = (f"a click on a {colour_name(key.split('@', 1)[1])} cell ({action_words(sig)})"
           if str(key).startswith("MOUSE@") else str(key))
    return f"{who} {EFFECT_WORDS.get(cls, cls)} for the first time in this game"


def clear_words(level: int, sig: str, attempt: int, actions: int, spent: int) -> str:
    return (f"Level {level} was cleared by {action_words(sig)} (action {actions} of attempt {attempt}; "
            f"{plural(spent, 'action')} on the level)")


def _counts(counter: Any, key: str) -> list[tuple[str, int]]:
    return [(cls, counter[(key, cls)]) for cls in _CLASS_ORDER if counter[(key, cls)]]


def _key_row(effects: Any, key: str, *, game: bool, words: dict[str, str] | None = None) -> str:
    level = _counts(effects.level_counts, key)
    whole = _counts(effects.game_counts, key)
    if not level:
        row = f"{key}: not tried on this level."
        if game and whole:
            row += f" In the whole game: {counts_words(whole, words)}."
        return row
    total = sum(n for _, n in level)
    if all(cls in STILL for cls, _ in level):
        boards = len(effects.level_boards.get(key, ()))
        row = (f"{key} ({plural(total, 'press', 'presses')} on this level): the board inside the edge "
               f"never changed, on {plural(boards, 'board')}.")
    else:
        row = f"{key} ({plural(total, 'press', 'presses')} on this level): {counts_words(level, words)}."
    if game and whole != level:
        row += f" In the whole game: {counts_words(whole, words)}."
    return row


def _click_row(effects: Any, key: str, words: dict[str, str] | None = None) -> str:
    count_words = words
    words = EFFECT_WORDS if words is None else words
    targets = list(effects.targets.get(key, {}).items())
    counts = _counts(effects.level_counts, key)
    total = sum(n for _, n in counts)
    row = (f"{key_words(key).capitalize()} ({plural(total, 'click')} on {plural(len(targets), 'cell')}): "
           f"{counts_words(counts, count_words)}.")
    if targets:
        (r, c), cls = targets[-1]
        row += f" Last: ({r}, {c}) {words.get(cls, cls)}."
        other = next(((cell, k) for cell, k in reversed(targets[:-1]) if k != cls), None)
        if other is not None:
            (r2, c2), cls2 = other
            row += f" Before that: ({r2}, {c2}) {words.get(cls2, cls2)}."
    return row


def effect_rows(
    effects: Any,
    valid_actions: Sequence[str],
    *,
    board_colours: Sequence[str] = (),
    game: bool = True,
    max_chars: int = 1000,
    max_click_rows: int = 6,
    words: dict[str, str] | None = None,
    colours_first: bool = False,
    new_colours: dict[str, Sequence[Sequence[tuple[int, int]]]] | None = None,
    lead_row: str | None = None,
    clicked_also: bool = False,
) -> list[str]:
    """The legible ``<actions>`` rows (memory_facts.ActionEffects.render in words).

    ``game``: add the whole game's counts where they differ (the card's rows);
    the this-level block passes False and lists only this level.

    ``colours_first`` (ARC3_MEM_COLOURS_FIRST): the not-clicked colours row
    comes first and outside ``max_chars``, a row that does not fit is skipped
    rather than ending the list, and the colours in ``new_colours`` (absent
    from the level's first board; ``FactRecorder.new_colour_groups``) are
    listed first, each with where it is now.

    ``lead_row`` (ARC3_MEM_CONTACT_ROW; not None when on): replaces the
    not-clicked colours row; it comes first (when not "") and outside
    ``max_chars``, and a row that does not fit is skipped, as with
    ``colours_first``.

    ``clicked_also`` (ARC3_MEM_CLICKED_ALSO): when click rows are left out
    (by ``max_chars`` or past ``max_click_rows``), a short row after the shown
    click rows names their colours with this level's click counts, ``Also
    clicked on this level: yellow x1, purple x2.``, outside ``max_chars``."""
    valid = [str(a).upper() for a in valid_actions or []]
    keyboard = [a for a in valid if a not in ("MOUSE", "RESET")]
    for key in effects.order:
        if not key.startswith("MOUSE@") and key not in keyboard and key != "RESET":
            keyboard.append(key)
    rows: list[str] = []
    untried: list[str] = []
    for key in keyboard:
        tried_level = any(k == key for k, _ in effects.level_counts)
        tried_game = any(k == key for k, _ in effects.game_counts)
        if tried_level or (game and tried_game):
            rows.append(_key_row(effects, key, game=game, words=words))
        elif key in valid:
            untried.append(key)
    clicked = [k for k in effects.order if k.startswith("MOUSE@") and any(kk == k for kk, _ in effects.level_counts)]
    # ARC3_MEM_CLICKED_ALSO: the click key of each click row, by row index
    click_keys: dict[int, str] = {}

    def rank(key: str) -> tuple[int, int]:
        urgent = effects.level_counts[(key, CLEAR)] + effects.level_counts[(key, DEATH)]
        total = sum(n for (k, _), n in effects.level_counts.items() if k == key)
        return (0 if urgent else 1, -total)

    if clicked:
        ranked = sorted(clicked, key=rank)
        for key in ranked[:max_click_rows]:
            click_keys[len(rows)] = key
            rows.append(_click_row(effects, key, words))
    if untried:
        rows.append("Not tried on this level: " + ", ".join(untried) + ".")
    if lead_row is not None:
        # ARC3_MEM_CONTACT_ROW: the object-level row replaces the colours row
        out = [lead_row] if lead_row else []
        shown = _fit_rows(out, rows, max_chars, skip=True)
        return _with_clicked_also(out, effects, clicked, rank, click_keys, shown) if clicked_also else out
    colours_row = ""
    if "MOUSE" in valid and board_colours:
        used = {k.split("@", 1)[1] for k in clicked}
        if colours_first:
            fresh = new_colours or {}
            unused_chars = [c for c in board_colours if c not in used]
            unused = [
                f"{colour_name(c)} (new on this level; {colour_place_words(fresh[c])})"
                for c in unused_chars if c in fresh
            ] + [colour_name(c) for c in unused_chars if c not in fresh]
        else:
            unused = [colour_name(c) for c in board_colours if c not in used]
        if unused:
            colours_row = "Colours on the board not clicked on this level: " + ", ".join(unused) + "."
            if not colours_first:
                rows.append(colours_row)
    out: list[str] = [colours_row] if colours_first and colours_row else []
    shown = _fit_rows(out, rows, max_chars, skip=colours_first)
    return _with_clicked_also(out, effects, clicked, rank, click_keys, shown) if clicked_also else out


def _fit_rows(out: list[str], rows: Sequence[str], max_chars: int, *, skip: bool) -> list[int]:
    """Append ``rows`` to ``out`` within ``max_chars`` (``out``'s rows already
    there are outside it): a row that does not fit ends the list, or with
    ``skip`` is left out. Returns, per row index, its index in ``out`` (-1
    when left out)."""
    shown = [-1] * len(rows)
    total = 0
    for index, row in enumerate(rows):
        if total + len(row) + 1 > max_chars:
            if skip:
                continue
            break
        shown[index] = len(out)
        out.append(row)
        total += len(row) + 1
    return shown


def _with_clicked_also(out: list[str], effects: Any, clicked: Sequence[str], rank: Any,
                       click_keys: dict[int, str], shown: Sequence[int]) -> list[str]:
    """ARC3_MEM_CLICKED_ALSO: ``out`` with ``Also clicked on this level:
    yellow x1, purple x2.`` after the last shown click row (or where the
    click rows would be) when any clicked colour has no row in ``out``."""
    present = {key for index, key in click_keys.items() if shown[index] >= 0}
    missing = [key for key in sorted(clicked, key=rank) if key not in present]
    if not missing:
        return out
    counts = [
        f"{colour_name(key.split('@', 1)[1])} x{sum(n for (k, _), n in effects.level_counts.items() if k == key)}"
        for key in missing
    ]
    row = "Also clicked on this level: " + ", ".join(counts) + "."
    places = [shown[index] for index in click_keys if shown[index] >= 0]
    if places:
        at = max(places) + 1
    else:
        # before the first row that comes after the click rows, else at the end
        later = [shown[index] for index in range(max(click_keys, default=-1) + 1, len(shown)) if shown[index] >= 0]
        at = min(later) if later else len(out)
    return [*out[:at], row, *out[at:]]


def _span_words(word: str, lo: int, hi: int) -> str:
    return f"{word} {lo}" if lo == hi else f"{word}s {lo}-{hi}"


def _box_words(cells: Sequence[tuple[int, int]]) -> str:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return f"{_span_words('row', min(rows), max(rows))}, {_span_words('col', min(cols), max(cols))}"


def colour_place_words(groups: Sequence[Sequence[tuple[int, int]]]) -> str:
    """Where a colour's cells are: ``rows 43-54, cols 12-14`` for one group;
    ``40 cells in 3 groups, the largest at rows 43-54, cols 12-14`` when
    they are scattered (groups largest first, as ``colour_groups`` gives)."""
    groups = [g for g in groups if g]
    if not groups:
        return ""
    if len(groups) == 1:
        return _box_words(groups[0])
    total = sum(len(g) for g in groups)
    return (f"{plural(total, 'cell')} in {len(groups)} groups, "
            f"the largest at {_box_words(groups[0])}")


def _visits_words(visits: Sequence[tuple[int, int]], attempt: int, acts: dict | None = None) -> str:
    """``acts`` (ARC3_MEM_SEEN_GAME_ACTION): (attempt, index) -> the game's
    action number, given first: ``after action 144 (action 5 of this attempt)``."""
    bits: list[str] = []
    for a, index in visits:
        owner = "this attempt" if a == attempt else f"attempt {a}"
        act = acts.get((a, index)) if acts is not None else None
        if act is None:
            bits.append(f"at the start of {owner}" if index == 0 else f"after action {index} of {owner}")
        elif index == 0:
            bits.append(f"at the start of {owner} (after action {act})" if act else f"at the start of {owner}")
        else:
            bits.append(f"after action {act} (action {index} of {owner})")
    return "; ".join(bits)


def _board_words(seen: Any, board: str, attempt: int, *, game_actions: bool = False) -> str:
    entry = seen.get(board)
    if entry is None or not entry.visits:
        return "moved to another board"
    pool = [v for v in entry.visits if v[0] == attempt] or list(entry.visits)
    a, index = next((v for v in pool if v[1] == 0), pool[-1])
    owner = "" if a == attempt else f" of attempt {a}"
    act = entry.acts.get((a, index)) if game_actions else None
    if index == 0:
        return f"reached the start board{owner}"
    if act is not None:
        who = "this attempt" if a == attempt else f"attempt {a}"
        return f"reached the board after action {act} (action {index} of {who})"
    return f"reached the board after action {index}{owner}"


def seen_text(recorder: Any, valid_actions: Sequence[str], *, hide_fatal: bool = False,
              max_chars: int = 320, game_actions: bool = False) -> str:
    """The legible ``Seen before`` line: when the current board was seen
    earlier on this level, which actions were tried from it and what each
    did, and the actions not tried from it; ``""`` on a first visit or at an
    attempt's start board."""
    position = recorder.position
    if position is None or position[1] == 0:
        return ""
    entry = recorder.seen.get(recorder.current_h)
    if entry is None or not any(v != position for v in entry.visits):
        return ""
    attempt = recorder.attempts.attempt
    earlier = [v for v in entry.visits if v != position]
    acts = entry.acts if game_actions else None
    head = f"The current board was seen before on this level: {_visits_words(earlier, attempt, acts)}."
    items: list[str] = []
    tried: set[str] = set()
    for sig, (kind, next_board, at) in reversed(entry.tried.items()):
        tried.add(sig)
        if hide_fatal and kind == "death":
            continue
        if kind == "clear":
            what = "cleared the level"
        elif kind == "death":
            what = f"GAME OVER in attempt {at}"
        elif kind == "noop":
            what = "did nothing"
        else:
            what = _board_words(recorder.seen, next_board, attempt, game_actions=game_actions)
        items.append(f"{action_words(sig)} {what}")
    valid = [str(a).upper() for a in valid_actions or []]
    untried = [a for a in valid if a not in ("MOUSE", "RESET") and a not in tried]
    tail = f" Not tried from it: {', '.join(untried)}." if untried else ""
    for limit in range(min(4, len(items)), -1, -1):
        shown = items[:limit]
        middle = ""
        if shown:
            more = len(items) - limit
            middle = " From it: " + "; ".join(shown) + (f"; and {more} more" if more else "") + "."
        text = head + middle + tail
        if len(text) <= max_chars:
            return text
    return head if len(head) <= max_chars else ""


# ---------------------------------------------------------------- previous levels


@dataclass
class LevelEntry:
    """One cleared level as ``<previous_levels>`` shows it, frozen when committed."""

    level: int
    path: list[str]
    spent: int
    deaths: int
    winning: int | None = None
    # ARC3_CLEARED_BOARDS: which frame is the board before the clear and
    # where the board it produced is (cleared_boards.clear_line)
    cleared: str = ""
    # ARC3_MEM_DEBRIEF: the model's recap from its first reasoning after the clear
    recap: str = ""
    # ARC3_LEVEL_DEBRIEF (arm/level-debrief): the debrief call's GOAL / MECHANICS
    goal: str = ""
    rules: str = ""
    # ARC3_MEM_RECAP_ASK_V2: (N, claim) for each later level N whose recap
    # says it corrects this level's recap
    corrected: list[tuple[int, str]] = field(default_factory=list)
    # (label, text): the model's own notes written on the level
    notes: list[tuple[str, str]] = field(default_factory=list)
    # ARC3_MEM_ROUTE_FORMAT=readable: each path action's effect key
    keys: list[str] = field(default_factory=list)
    # ARC3_MEM_HEAD_WORDS: attempts on the level (0: not recorded)
    attempts: int = 0
    # ARC3_MEM_RULES: the rules the model confirmed on the level, verbatim,
    # frozen with the entry
    confirmed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HeadOptions:
    """How ``<previous_levels>`` entries are drawn; set once per run on the
    ledger (memory v2's flags). The defaults are what pf-keep-10x draws."""

    # a recap, goal or rules line at stage 0 (later stages cut to 160)
    recap_chars: int = 300
    # what ends a cut recap, goal or rules line (ARC3_MEM_RECAP_CHARS: " ...")
    cut_marker: str = "..."
    route_chars: int = ROUTE_CHARS
    # ARC3_MEM_ROUTE_FORMAT=readable: clicks named by colour (``route_words``)
    readable_routes: bool = False
    # ARC3_MEM_HEAD_WORDS: the head line says what each count is (``solved_words``)
    head_words: bool = False
    # ARC3_MEM_COMPRESS_ROUTES_FIRST: over the cap, every older entry loses its
    # route before any loses recap text (``_entry_lines_routes_first``)
    routes_first: bool = False


DEFAULT_HEAD = HeadOptions()


def _head_line(entry: LevelEntry, *, legible: bool, route: bool, options: HeadOptions) -> str:
    if legible:
        keys = list(entry.keys) if options.readable_routes else None
        if options.head_words and entry.attempts > 0 and not entry.winning:
            return solved_words(entry.level, entry.path, entry.spent, entry.deaths, entry.attempts,
                                route_chars=options.route_chars if route else 0, keys=keys)
        return solved_text(entry.level, entry.path, entry.spent, entry.deaths,
                           route_chars=options.route_chars if route else 0, winning=entry.winning, keys=keys)
    from inference.agent.memory_facts import SOLVED_PATH_CHARS, solved_head

    return solved_head(entry.level, entry.path, entry.spent, entry.deaths,
                       path_chars=SOLVED_PATH_CHARS if route else 0, winning=entry.winning)


def _confirmed_lines(entry: LevelEntry) -> list[str]:
    """ARC3_MEM_RULES: ``  Rules you confirmed: "..." | "..."`` ([] without rules)."""
    if not entry.confirmed:
        return []
    from inference.agent.confirmed_rules import quoted

    return [f"  Rules you confirmed: {quoted(entry.confirmed)}"]


def this_level_rules_line(rules: Sequence[str]) -> str:
    """ARC3_MEM_RULES: the ``<this_level>`` line ("" without rules)."""
    if not rules:
        return ""
    from inference.agent.confirmed_rules import quoted

    return f"Rules you confirmed on this level: {quoted(list(rules))}"


def _corrected_lines(entry: LevelEntry, cap: int, mark: str) -> list[str]:
    """ARC3_MEM_RECAP_ASK_V2: ``  Corrected by level 4: <claim>`` for each
    later recap that says it corrects this level's."""
    return [f"  Corrected by level {n}: {cut_words(claim, cap, marker=mark)}" for n, claim in entry.corrected]


def _entry_lines_routes_first(entry: LevelEntry, stage: int, *, legible: bool,
                              options: HeadOptions) -> list[str]:
    """ARC3_MEM_COMPRESS_ROUTES_FIRST stages. 0: everything; 1: no route (the
    head line keeps the route's length) and notes cut to 2; 2: the goal,
    rules and recap cut to 300 chars; 3: the goal cut to 160, or the recap
    when there is no goal, and no notes; 4: the head line."""
    lines = [_head_line(entry, legible=legible, route=stage == 0, options=options)]
    if stage >= 4:
        return lines
    pad, mark = "  ", options.cut_marker
    cap = options.recap_chars if stage <= 1 else 300 if stage == 2 else 160
    label = "Your recap" if legible else "debrief"
    if entry.cleared and stage <= 2:
        lines.append(f"{pad}{entry.cleared}")
    if stage == 3:
        if entry.goal:
            lines.append(f"{pad}Goal: {cut_words(entry.goal, cap, marker=mark)}")
        elif entry.recap:
            lines.append(f"{pad}{label}: {cut_words(entry.recap, cap, marker=mark)}")
        elif entry.rules:
            lines.append(f"{pad}Rules: {cut_words(entry.rules, cap, marker=mark)}")
        return lines
    if entry.goal:
        lines.append(f"{pad}Goal: {cut_words(entry.goal, cap, marker=mark)}")
    if entry.rules:
        lines.append(f"{pad}Rules: {cut_words(entry.rules, cap, marker=mark)}")
    if entry.recap:
        lines.append(f"{pad}{label}: {cut_words(entry.recap, cap, marker=mark)}")
    lines.extend(_corrected_lines(entry, cap, mark))
    if stage <= 2:
        lines.extend(_confirmed_lines(entry))
    for note_label, text in (entry.notes if stage == 0 else entry.notes[:2]):
        lines.append(f"{pad}{note_label}: {cut_words(text, NOTE_CHARS)}")
    return lines


def _entry_lines(entry: LevelEntry, stage: int, *, legible: bool,
                 options: HeadOptions = DEFAULT_HEAD) -> list[str]:
    """Stage 0: everything; 1: notes cut to 2 and the recap to 160 chars; 2: no
    route and no notes; 3: the head line and the goal only; 4: the head line."""
    if options.routes_first:
        return _entry_lines_routes_first(entry, stage, legible=legible, options=options)
    lines = [_head_line(entry, legible=legible, route=stage < 2, options=options)]
    if stage >= 4:
        return lines
    pad = "  "
    cap, mark = options.recap_chars if stage == 0 else 160, options.cut_marker
    if entry.cleared and stage <= 2:
        lines.append(f"{pad}{entry.cleared}")
    if entry.goal:
        lines.append(f"{pad}Goal: {cut_words(entry.goal, cap, marker=mark)}")
    if stage >= 3:
        return lines
    if entry.rules:
        lines.append(f"{pad}Rules: {cut_words(entry.rules, cap, marker=mark)}")
    if entry.recap:
        label = "Your recap" if legible else "debrief"
        lines.append(f"{pad}{label}: {cut_words(entry.recap, cap, marker=mark)}")
    lines.extend(_corrected_lines(entry, cap, mark))
    lines.extend(_confirmed_lines(entry))
    if stage >= 2:
        return lines
    notes = entry.notes if stage == 0 else entry.notes[:2]
    for label, text in notes:
        lines.append(f"{pad}{label}: {cut_words(text, NOTE_CHARS)}")
    return lines


def render_entry(entry: LevelEntry, *, legible: bool = True, options: HeadOptions = DEFAULT_HEAD) -> str:
    """One entry exactly as ``render_previous_levels`` draws it uncompressed."""
    return "\n".join(_entry_lines(entry, 0, legible=legible, options=options))


def render_previous_levels(entries: Sequence[LevelEntry], *, max_chars: int = HEAD_CHARS,
                           legible: bool = True, options: HeadOptions = DEFAULT_HEAD) -> str:
    """The ``<previous_levels>`` block, ``""`` before the first clear.

    Over ``max_chars`` the oldest entries are compressed first, one stage at a
    time (``_entry_lines``), so the newest level keeps its notes longest; the
    newest entry is compressed last."""
    entries = sorted(entries, key=lambda e: e.level)
    if not entries:
        return ""
    stages = [0] * len(entries)

    def build() -> str:
        lines: list[str] = []
        for entry, stage in zip(entries, stages):
            lines.extend(_entry_lines(entry, stage, legible=legible, options=options))
        return f"<{PREVIOUS_TAG}>\n" + "\n".join(lines) + f"\n</{PREVIOUS_TAG}>"

    text = build()
    while options.routes_first and len(text) > max_chars:
        # breadth-first: every older entry reaches stage k (oldest first)
        # before any reaches k + 1; the newest entry is compressed last
        older = [stage for stage in stages[:-1]]
        if older and min(older) < 4:
            stages[older.index(min(older))] += 1
        elif stages[-1] < 4:
            stages[-1] += 1
        elif len(entries) > 1:
            entries, stages = entries[1:], stages[1:]
        else:
            break
        text = build()
    while len(text) > max_chars:
        index = next((i for i, s in enumerate(stages) if s < 4 and i < len(entries) - 1), None)
        if index is None:
            index = len(entries) - 1 if stages[-1] < 4 else None
        if index is None:
            # every entry is down to its head line: drop the oldest
            if len(entries) <= 1:
                break
            entries, stages = entries[1:], stages[1:]
        else:
            stages[index] += 1
        text = build()
    return text


# ---------------------------------------------------------------- this level


def render_this_level(
    *,
    level: int,
    attempt: int,
    actions: int,
    deaths: int,
    rows: Sequence[str],
    hazards: Sequence[str],
    seen: str,
    notes: Sequence[tuple[str, str]],
    max_chars: int = LEVEL_CHARS,
    attempt_actions: int | None = None,
    extra: Sequence[str] = (),
) -> str:
    """The ``<this_level>`` block for the end of a turn message.

    Lines, in order: a summary, the action and click rows, how attempts
    ended (newest 2), the Seen-before line, the model's notes on this level
    (newest first). Over ``max_chars`` cut in this order: older hazards,
    action rows from the end, notes from the oldest, the Seen-before line;
    the summary line always stays. ``extra`` (ARC3_MEM_RULES, ARC3_MEM_MOVED;
    each capped where it is made) follows the notes, outside ``max_chars``."""
    if actions and attempt_actions is not None and attempt >= 2:
        # ARC3_MEM_ATTEMPT_COUNTS: the level's count was read as this attempt's
        summary = (f"Level {level} so far: {plural(actions, 'action')} over {plural(attempt, 'attempt')}, "
                   f"{int(attempt_actions)} of them in this attempt (attempt {attempt}), {deaths_words(deaths)}.")
    elif actions:
        summary = (f"Level {level} so far: {plural(actions, 'action')}, attempt {attempt}, "
                   f"{deaths_words(deaths)}.")
    else:
        summary = f"Level {level} so far: no actions yet."
    rows = list(rows)
    hazards = list(hazards)[-2:]
    notes = [f"Your note ({label.lower()}): {cut_words(text, NOTE_CHARS)}" for label, text in notes]

    extra = [line for line in extra if line]
    tail = len("\n".join(extra)) + 1 if extra else 0

    def build() -> str:
        lines = [summary, *rows, *hazards, *([seen] if seen else []), *notes, *extra]
        return f"<{THIS_LEVEL_TAG}>\n" + "\n".join(lines) + f"\n</{THIS_LEVEL_TAG}>"

    text = build()
    while len(text) - tail > max_chars:
        if len(hazards) > 1:
            hazards.pop(0)
        elif len(rows) > 2:
            rows.pop()
        elif len(notes) > 2:
            notes.pop()
        elif seen:
            seen = ""
        elif rows:
            rows.pop()
        elif notes:
            notes.pop()
        elif hazards:
            hazards.pop()
        else:
            break
        text = build()
    return text
