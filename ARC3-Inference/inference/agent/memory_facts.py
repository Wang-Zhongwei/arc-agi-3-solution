"""Facts the harness reads off the frames for memory-v2. Pure python, no I/O.

The model-written memory channel is nearly empty: 2.7-3.9 tags per 100 calls,
and most ledger lines were harness fragments of its thinking. The evidence the
model wastes is already in the game data: 9.0% of actions repeat an exact
(board, action) pair, 27% sit on net-zero loops, and 15% of tokens go to
attempts that die. So the harness records it here, from every executed action
(batch members included), and ``tool_agent`` shows it:

- ``FactRecorder.feed`` walks the solver's history entries once each, keeping
  per attempt only hashes and counts (no grid beyond the latest board).
- ``AttemptTracker`` knows the attempt, its steps and the level's spend.
- ``ActionEffects`` counts what each action (or click colour) did, per level
  and per game. Counts, never verdicts: a key that never moved anything is
  reported as "no board change on 4 presses across 2 boards".
- ``SeenStates`` remembers, per level, which boards were visited and what was
  tried from each, so a revisit can be reported with what happened before.
- ``loop_free`` + ``rle_path`` turn a winning attempt into a short path,
  ``distill_think`` picks one whole sentence of the clearing turn's reasoning.

Board identity is ``state_hash`` with the no-op guard's border (4) cropped, so a
ticking budget bar does not make every board new. The notes built from it are
advisory and block nothing, so merging two boards that differ only in hidden
state costs a misleading line, never a refused action.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from inference.agent.noop_repeat_guard import action_signature
from inference.utils.grid_utils import ARC_COLOR_CHARS

Grid = Sequence[Sequence[int]]

CLEAR, DEATH, NOOP, HUD, ANIM, SHIFT, CHANGE = (
    "clear", "death", "noop", "hud", "anim", "shift", "change",
)
# Classes whose board interior did not change.
STILL = (NOOP, HUD, ANIM)
# Model-facing words for each class; plain descriptions, no verdicts.
LABELS = {
    CLEAR: "clear",
    DEATH: "death",
    NOOP: "no change",
    HUD: "edge only",
    ANIM: "anim only",
    SHIFT: "moved",
    CHANGE: "changed",
}
_CLASS_ORDER = (CLEAR, DEATH, SHIFT, CHANGE, ANIM, HUD, NOOP)
_LETTERS = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R", "SPACE": "S"}
_CLICK_RE = re.compile(r"^([A-Z0-9_]+)\((-?\d+),(-?\d+)\)$")
# Count caps: sets of board hashes saturate rather than grow with the game.
_MAX_BOARDS_PER_KEY = 64
_MAX_CELL_SAMPLES = 64


# ---------------------------------------------------------------- boards


def crop(grid: Grid, border: int) -> list[tuple[int, ...]]:
    """The board without its outer ``border`` cells; small grids stay whole."""
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    if rows > 2 * border and cols > 2 * border:
        return [tuple(row[border:cols - border]) for row in grid[border:rows - border]]
    return [tuple(row) for row in grid]


def state_hash(grid: Grid, border: int) -> str:
    """Identity of a board for "have I been here before" questions.

    ``border`` cells are cropped first. With the guard's 4 a timer bar in the
    edge band does not split one position into many; with 0 (the death
    ledger's default) every tick makes a new board.
    """
    return hashlib.sha1(repr(crop(grid, max(0, int(border)))).encode("utf-8")).hexdigest()


def _histogram(core: list[tuple[int, ...]]) -> Counter:
    return Counter(cell for row in core for cell in row)


def colour_groups(grid: Grid, value: int, border: int) -> list[list[tuple[int, int]]]:
    """The 4-connected groups of ``value`` cells inside the ``border`` crop
    (the crop ``crop`` takes), as (row, col) in board coordinates, largest
    first (ties: top-most, then left-most)."""
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    lo = border if rows > 2 * border and cols > 2 * border else 0
    r_hi, c_hi = rows - lo, cols - lo
    cells = {
        (r, c)
        for r in range(lo, r_hi)
        for c in range(lo, min(c_hi, len(grid[r])))
        if int(grid[r][c]) == value
    }
    groups: list[list[tuple[int, int]]] = []
    while cells:
        seed = min(cells)
        cells.discard(seed)
        group, stack = [seed], [seed]
        while stack:
            r, c = stack.pop()
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in cells:
                    cells.discard(nb)
                    group.append(nb)
                    stack.append(nb)
        groups.append(sorted(group))
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def changed_cells(pre_core: list[tuple[int, ...]], post_core: list[tuple[int, ...]]) -> int:
    """Cells that differ between two cropped boards (all of them on a shape change)."""
    if len(pre_core) != len(post_core) or any(len(a) != len(b) for a, b in zip(pre_core, post_core)):
        return sum(len(row) for row in post_core)
    return sum(1 for a, b in zip(pre_core, post_core) if a != b for x, y in zip(a, b) if x != y)


def classify(result: dict[str, Any], pre: Grid, post: Grid, border: int) -> str:
    """What one executed action did, first match wins.

    clear (level_completed or the final win) > death (game_over) > interior
    unchanged: anim (an animation played) > hud (only edge cells changed) >
    noop > shift (interior colour counts unchanged, so things moved) > change
    (counts changed: something appeared, vanished or was recoloured).

    An animation that leaves the interior as it was is checked before noop
    and hud: those two would otherwise always match first.
    """
    result = result or {}
    if result.get("level_completed") or result.get("run_complete"):
        return CLEAR
    if result.get("game_over"):
        return DEATH
    pre_core = crop(pre, border)
    post_core = crop(post, border)
    return _classify_cores(result, pre, post, pre_core, post_core)


def _classify_cores(result, pre, post, pre_core, post_core) -> str:
    if pre_core == post_core:
        if result.get("animation"):
            return ANIM
        same = len(pre) == len(post) and all(tuple(a) == tuple(b) for a, b in zip(pre, post))
        return NOOP if same else HUD
    if _histogram(pre_core) == _histogram(post_core):
        return SHIFT
    return CHANGE


# ---------------------------------------------------------------- actions


def parse_click(sig: str) -> tuple[int, int] | None:
    """(row, col) of a click signature such as ``MOUSE(4,7)``; None otherwise."""
    match = _CLICK_RE.match(str(sig or ""))
    if not match:
        return None
    return int(match.group(2)), int(match.group(3))


def effect_key(sig: str, pre: Grid | None) -> str:
    """The row an action is counted under: its name, or ``MOUSE@<letter>`` for
    a click, the letter being the colour of the clicked cell before the click."""
    cell = parse_click(sig)
    if cell is None:
        return sig
    row, col = cell
    letter = "?"
    try:
        if pre is not None and 0 <= row < len(pre) and 0 <= col < len(pre[row]):
            letter = ARC_COLOR_CHARS[max(0, min(15, int(pre[row][col])))]
    except (TypeError, ValueError, IndexError):
        letter = "?"
    return f"MOUSE@{letter}"


def action_item(sig: str) -> Any:
    """A signature as ``action()`` takes it: ``"RIGHT"``, or
    ``{"action": "MOUSE", "row": r, "col": c}`` for a click."""
    match = _CLICK_RE.match(str(sig or ""))
    if not match:
        return sig
    return {"action": match.group(1), "row": int(match.group(2)), "col": int(match.group(3))}


def short_action(sig: str) -> str:
    """``U D L R S`` for the arrows and space, ``M(r,c)`` for a click."""
    cell = parse_click(sig)
    if cell is not None:
        return f"M({cell[0]},{cell[1]})"
    return _LETTERS.get(sig, sig)


def display_action(sig: str) -> str:
    """Full name for a key (``DOWN``), ``M(r,c)`` for a click."""
    cell = parse_click(sig)
    return f"M({cell[0]},{cell[1]})" if cell is not None else sig


def _rle_tokens(sigs: Iterable[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for token in (short_action(s) for s in sigs):
        if runs and runs[-1][0] == token:
            runs[-1] = (token, runs[-1][1] + 1)
        else:
            runs.append((token, 1))
    return runs


def _run_text(token: str, count: int) -> str:
    if count == 1:
        return token
    return f"{token}{count}" if len(token) == 1 else f"{token}x{count}"


def rle_path(sigs: Iterable[str]) -> str:
    """Run-length path: ``L3 U4 R3 U3``; clicks as ``M(10,4)``, repeats ``x2``."""
    return " ".join(_run_text(token, count) for token, count in _rle_tokens(sigs))


@dataclass(frozen=True)
class Step:
    sig: str
    pre_h: str
    post_h: str
    cls: str
    # the effect key (``MOUSE@<letter>``: the colour of the clicked cell
    # before the click), for ARC3_MEM_ROUTE_FORMAT=readable's click labels
    key: str = ""


def loop_free(steps: Sequence[Step], start_h: str, *, keep_still: bool = False) -> list[str]:
    """The signatures of ``loop_free_steps``."""
    return [step.sig for step in loop_free_steps(steps, start_h, keep_still=keep_still)]


def loop_free_steps(steps: Sequence[Step], start_h: str, *, keep_still: bool = False) -> list[Step]:
    """The winning attempt with its loops cut out.

    Walks the chain of boards from the start. When an action returns to a
    board already on the chain, everything since that board is cut. Actions
    that left the interior unchanged (noop, hud, anim) are dropped.

    ``keep_still`` (ARC3_MEM_KEEP_EDGE): an action that changed only the edge
    band (hud) or played an animation (anim) is kept, and no loop is cut back
    across it. Such an action can change state the cropped interior does not
    show (ar25's selection marker sits in the edge, so dropping SPACE #106
    made its L3 path fail to clear in level-path-10x), and two boards with the
    same interior on either side of it may then behave differently. Only
    actions that changed no cell at all (noop) are dropped.
    """
    if keep_still:
        return _loop_free_sealed(steps, start_h)
    path: list[Step] = []
    chain = [start_h]
    for step in steps:
        if step.cls in STILL:
            continue
        if step.cls not in (CLEAR, DEATH) and step.post_h in chain:
            j = chain.index(step.post_h)
            del chain[j + 1:]
            del path[j:]
            continue
        path.append(step)
        chain.append(step.post_h)
    return path


def _loop_free_sealed(steps: Sequence[Step], start_h: str) -> list[Step]:
    """``loop_free`` that keeps hud and anim actions and never cuts across one.

    ``marks[k]`` is the path length when ``chain[k]`` was reached, so a cut
    back to ``chain[k]`` drops exactly the actions taken since. A kept still
    action restarts the chain at the current board: what came before it is
    sealed.
    """
    path: list[Step] = []
    chain = [start_h]
    marks = [0]
    for step in steps:
        if step.cls in STILL:
            if step.cls != NOOP:
                path.append(step)
                chain, marks = [chain[-1]], [len(path)]
            continue
        if step.cls not in (CLEAR, DEATH) and step.post_h in chain:
            j = chain.index(step.post_h)
            del chain[j + 1:]
            del marks[j + 1:]
            del path[marks[j]:]
            continue
        path.append(step)
        chain.append(step.post_h)
        marks.append(len(path))
    return path


# ---------------------------------------------------------------- routes

ROUTE_MAX_BOARDS = 1024
ROUTE_MAX_SIGS = 16
ROUTE_MAX_OUTCOMES = 4
# UNDO restores whatever came before it, so it links no fixed pair of boards;
# RESET is the attempt's start, not a step on a route.
NO_LINK_SIGS = frozenset({"UNDO", "ACTION7", "RESET"})


class LevelGraph:
    """Move links observed on one level and the fewest-link routes over them.

    Nodes are board hashes (``state_hash``: the interior, edge cropped). Only
    a board-changing action links one board to the next: ``links[pre][sig]``
    keeps up to 4 boards it reached, in arrival order. No-change actions,
    deaths, clears, RESET and UNDO add no link (UNDO depends on history, not
    on the board). A click keeps its exact signature, ``MOUSE(r,c)``.

    ``varied`` holds the (board, action) pairs seen with more than one
    outcome: two boards reached, or a move and also a death or no change. The
    routes use every move outcome; ``route_varied`` names the varied steps.

    Over a cap (1024 boards, 16 actions per board, 4 outcomes) the link is not
    added and ``full`` counts it: nothing is evicted, so a recorded link
    stays and, while the start board is the same, a distance never grows. A
    RESET that lands on another board moves the source (``set_start``), and
    distances from the new start can be longer. ``bfs`` recomputes the
    distances from ``start_h`` in full; it runs only when a link or the
    start changed, which gives the same distances as running it after every
    action.
    """

    def __init__(
        self,
        level: int,
        start_h: str,
        *,
        max_boards: int = ROUTE_MAX_BOARDS,
        max_sigs: int = ROUTE_MAX_SIGS,
        max_outcomes: int = ROUTE_MAX_OUTCOMES,
    ) -> None:
        self.level = level
        self.start_h = start_h
        self.max_boards = max_boards
        self.max_sigs = max_sigs
        self.max_outcomes = max_outcomes
        self.boards: set[str] = {start_h}
        self.links: dict[str, dict[str, list[str]]] = {}
        # (board, action) pairs that also ended in a death or no change
        self.other: set[tuple[str, str]] = set()
        self.varied: set[tuple[str, str]] = set()
        self.full = 0
        self.dist: dict[str, int] = {start_h: 0}
        self.parent: dict[str, tuple[str, str]] = {}

    def add_board(self, board: str) -> bool:
        """Put a board on record (no link); False when over the board cap."""
        if board in self.boards:
            return True
        if len(self.boards) >= self.max_boards:
            self.full += 1
            return False
        self.boards.add(board)
        return True

    def add_move(self, pre: str, sig: str, post: str) -> bool:
        """Record that ``sig`` from ``pre`` reached ``post``; True when the
        links changed (the distances are then recomputed)."""
        outcomes = self.links.get(pre, {}).get(sig)
        if outcomes is not None and post in outcomes:
            return False
        if outcomes or (pre, sig) in self.other:
            self.varied.add((pre, sig))
        if outcomes is not None and len(outcomes) >= self.max_outcomes:
            self.full += 1
            return False
        by_sig = self.links.get(pre)
        if outcomes is None and by_sig is not None and len(by_sig) >= self.max_sigs:
            self.full += 1
            return False
        if not (self.add_board(pre) and self.add_board(post)):
            return False
        self.links.setdefault(pre, {}).setdefault(sig, []).append(post)
        self.bfs()
        return True

    def note_other(self, pre: str, sig: str) -> None:
        """``sig`` from ``pre`` ended in a death or changed nothing."""
        if (pre, sig) in self.other:
            return
        if len(self.other) < self.max_boards * self.max_sigs:
            self.other.add((pre, sig))
        if self.links.get(pre, {}).get(sig):
            self.varied.add((pre, sig))

    def set_start(self, start_h: str) -> None:
        """The attempt's start board, the source of every route."""
        if start_h == self.start_h:
            return
        self.start_h = start_h
        self.add_board(start_h)
        self.bfs()

    def bfs(self) -> None:
        dist = {self.start_h: 0}
        parent: dict[str, tuple[str, str]] = {}
        queue = deque([self.start_h])
        while queue:
            board = queue.popleft()
            step = dist[board] + 1
            for sig, posts in self.links.get(board, {}).items():
                for post in posts:
                    if post not in dist:
                        dist[post] = step
                        parent[post] = (board, sig)
                        queue.append(post)
        self.dist, self.parent = dist, parent

    def route(self, board: str) -> list[str] | None:
        """The fewest-link route from the start board, as signatures; None
        when no recorded links reach ``board``."""
        if board not in self.dist:
            return None
        sigs: list[str] = []
        while board in self.parent:
            board, sig = self.parent[board]
            sigs.append(sig)
        return sigs[::-1]

    def route_varied(self, board: str) -> list[int]:
        """0-based indexes of the steps on ``route(board)`` whose (board,
        action) pair had more than one outcome on this level."""
        if board not in self.dist:
            return []
        steps: list[tuple[str, str]] = []
        while board in self.parent:
            board, sig = self.parent[board]
            steps.append((board, sig))
        steps.reverse()
        return [i for i, pair in enumerate(steps) if pair in self.varied]


# what one action did to the route; DEATH and CLEAR reuse the class names
ROUTE_NEW, ROUTE_KNOWN, ROUTE_SHORTER, ROUTE_STILL, ROUTE_START = (
    "new", "known", "shorter", "still", "start",
)
ROUTE_ROWS_KEPT = 20000


@dataclass(frozen=True)
class RouteRow:
    """The route after one executed action (or RESET).

    ``d`` is the route of the board the action reached (None: no recorded
    links reach it, or a death or clear); ``before`` that board's route
    before the action; ``pre_d`` the route of the board it was taken from.
    ``i`` is the action's index in the attempt and ``k`` the actions so far
    in it (both 0 on a RESET row; the level's spend counts a manual RESET,
    ``k`` does not). ``path`` is the route on record after the action, run
    length coded: to the board reached, or for a clear to the clearing board
    plus the clearing action.
    """

    act: int
    level: int
    attempt: int
    i: int
    sig: str
    d: int | None
    k: int
    change: str
    before: int | None = None
    pre_d: int | None = None
    pre: str = ""
    board: str = ""
    automatic: bool = False
    varied: tuple[int, ...] = ()
    path: str = ""


# ---------------------------------------------------------------- reasoning

_TAGS_RE = re.compile(
    r"<(world_model|hypothesis|level_note|scratchpad)(\s[^>]*)?>.*?</\1>", re.S | re.I
)
_FILLER_RE = re.compile(
    r"^(?:(?:so|then|hmm+|but|and|also|ok(?:ay)?|wait|now|let me|alright|well|actually|right|note)\b[\s,.:;!-]*"
    r"|(?:[a-z0-9]\)|\d+\.)\s+)+",
    re.I,
)
# The clearing turn mostly plans its last moves ("Just try UP now", "Execute:
# DOWN, DOWN, RIGHT", "One more UP should push it in"). A plan is not how the
# level was done, and an imperative may be read as advice on the next level,
# so sentences that open with a plan or command word, or hedge, are skipped.
_PLAN_START_RE = re.compile(
    r"^(?:try|just|let'?s?|i'?ll|i will|i should|i can|we'?ll|we can|should|next|maybe|need|"
    r"if|execute|press|click|do|run|check|verify|batch|be|include|navigate|move|handle|double|redefine|"
    r"simpler|better|first|careful|risks?|(?:the |my |remaining )*plan)\b",
    re.I,
)
_HEDGE_RE = re.compile(r"\b(?:should|would|might|maybe|perhaps|hopefully|i need|we need)\b", re.I)
# "Execute: DOWN, ..." and "For the MINI step: navigate to ..." start a plan
# after a short label.
_LABEL_RE = re.compile(r"^[^:]{1,40}?:\s+")
# Code and traces read as findings by the scores above but say nothing about
# the level: "Batch = p1 + p2 + ['RIGHT', ...]", "'DOWN' -> (45,34)".
_CODE_RE = re.compile(r"[=\[\]{}'\"`]|→|\w\(\)|\bdef\b|\bprint\b")
_ACTION_WORD_RE = re.compile(r"\b(?:UP|DOWN|LEFT|RIGHT|SPACE|UNDO|MOUSE|ACTION[1-7])\b|\bclick", re.I)
_ACTION_CASE_RE = re.compile(r"\b(?:UP|DOWN|LEFT|RIGHT|SPACE|UNDO|MOUSE|ACTION[1-7])\b")
_NUMBER_RE = re.compile(r"\d")
_FINDING_RE = re.compile(
    r"\b(so|therefore|means|because|confirm(?:s|ed)?|shows?|must|needs?|needed|requires?|"
    r"opens?|opened|blocked|reach(?:es|ed)?|goal|key|door|clears?|cleared|"
    r"solves?|solved|completes?|completed|works?|worked|matches|triggers?|toggles?)\b",
    re.I,
)


def _is_plan(sentence: str) -> bool:
    if sentence.endswith("!") or _HEDGE_RE.search(sentence) or _PLAN_START_RE.match(sentence):
        return True
    label = _LABEL_RE.match(sentence)
    return bool(label and _PLAN_START_RE.match(sentence[label.end():]))


def distill_think(text: str, max_chars: int = 160) -> str:
    """One whole sentence of reasoning that states how a level was done.

    Sentences are scored +2 for naming an action (arrow names in capitals,
    MOUSE, or a click), +1 for a number or coordinate, +1 for a finding word,
    and need 3: an action and a number or finding. Leading filler (so, then,
    hmm, ok, wait, now, let me, list markers...) is stripped. The
    best-scoring sentence wins, the LAST one on ties, because later sentences
    hold the settled view. Questions are skipped, and so is anything that
    looks like code or a move trace (``=``, brackets, quotes, arrows), or like
    a plan: an opening command or plan word (try, just, let's, I'll, should,
    next, execute, press, check...), also after a short ``label:``, a hedge
    (should, would, might, maybe) or a closing ``!``. A sentence is never
    cut, so one longer than ``max_chars`` cannot be chosen; when nothing
    qualifies the result is ``""`` (no line beats a fragment).

    On the r2 ledger run's 112 level clears this keeps a sentence for 34
    (84 before the plan filter and the score of 3), and none of the kept
    ones is an imperative.
    """
    flat = " ".join(_TAGS_RE.sub(" ", str(text or "")).split())
    if not flat or max_chars <= 0:
        return ""
    best = ""
    best_score = 3
    for raw in re.split(r"(?<=[.!?])\s+", flat):
        sentence = _FILLER_RE.sub("", raw.strip().lstrip("-*• ")).strip()
        if (
            len(sentence) < 20 or sentence.endswith("?") or len(sentence) > max_chars
            or _CODE_RE.search(sentence) or _is_plan(sentence)
        ):
            continue
        score = (
            2 * bool(_ACTION_CASE_RE.search(sentence) or re.search(r"\bclick", sentence, re.I))
            + bool(_NUMBER_RE.search(sentence))
            + bool(_FINDING_RE.search(sentence))
        )
        if score >= best_score:
            best, best_score = sentence, score
    if not best:
        return ""
    return best[0].upper() + best[1:]


# ARC3_MEM_DEBRIEF: the model's own recap of a level, from its first
# reasoning after the clear. A recap starts at "Level N solved/complete...",
# "previous level", or a "Key mechanics:" / "Mechanics confirmed:" label.
_DEBRIEF_START_RE = re.compile(
    r"\b(?:level|L)\s*\d+\s*(?:was\s+|is\s+|got\s+|has\s+been\s+)?(?:solved|completed?|cleared|won|passed|beaten)\b"
    r"|\b(?:previous|last)\s+level\b"
    r"|(?:^|(?<=[.!?] )|\b(?:level|L)\s*\d+\s+|\bkey\s+)"
    r"(?:mechanics?|rules?|learnings?|insights?|discovery)\s*"
    r"(?:established|confirmed|learned|recap|from\s+(?:L|level\s*)\d+)?\s*:"
    r"|\bsolution\s+insight\s*:",
    re.I,
)
# "Let me record the key mechanics in memory" announces a note; it is not one.
_DEBRIEF_RECORD_RE = re.compile(
    r"\b(?:let me|i'?ll|i will|now)\s+(?:record|save|note|write down)\b[^.:!]*(?:[.!]|(?=:))", re.I
)
# The recap names a goal or a rule; "Level 3 solved in 14 actions." does not.
_DEBRIEF_RULE_RE = re.compile(
    r"\b(?:goal|win|wins|rule|rules|mechanic|mechanics|means|because|when|whenever|must|needs?|needed|"
    r"requires?|key|triggers?|triggered|condition|confirmed|counter|each)\b|→",
    re.I,
)
# ARC3_MEM_DEBRIEF precision (the review of PR #148): 14 of 83 captures on three
# runs named another level than the one cleared ("Level 5 complete! ... Now
# level 5" for level 4, "Mechanics from L2" for level 3), and 4 were a
# celebration with no rule. Level numbers in the recap, in order:
_LEVEL_REF_RE = re.compile(r"\b(?:level|L)\s*(\d+)\b", re.I)
# the celebration a recap may open with ("Level 4 completed! Wait -", "LEVEL 3
# COMPLETE...", "Level 2 solved..."), cut before the rule check
_DEBRIEF_HEAD_RE = re.compile(
    r"^\W*(?:(?:level|L)\s*\d+\s*)?(?:was\s+|is\s+|got\s+|has\s+been\s+)?"
    r"(?:solved|completed?|cleared|won|passed|beaten)\b[\W_]*(?:(?:no\s+)?wait\b[\W_]*)?",
    re.I,
)
# a rule word in the rest of the recap; an arrow alone is a move trace
_DEBRIEF_RULE_WORD_RE = re.compile(_DEBRIEF_RULE_RE.pattern[: _DEBRIEF_RULE_RE.pattern.rfind("|")], re.I)
DEBRIEF_CHARS = 300


def _debrief_stop_re(new_level: int) -> re.Pattern:
    """Where the reasoning moves on to the new level."""
    return re.compile(
        rf"\bnow\b[^.]{{0,20}}?\b(?:level|L)\s*{int(new_level)}\b|\b(?:level|L)\s*{int(new_level)}\b"
        r"|\bnew (?:level|board|layout)\b|\bthe (?:new|current) board\b"
        r"|\blet me (?:inspect|look|parse|analy[sz]e|check|read|map|get|build|see|study|examine|view)\b",
        re.I,
    )


def extract_debrief(text: str, new_level: int, max_chars: int = DEBRIEF_CHARS, *, precise: bool = False) -> str:
    """The model's recap of the level it just cleared, in its own words, or ``""``.

    Taken from the first reasoning on the new level: from the sentence that
    opens the recap ("Level 1 solved: the goal was to make the two blocks
    meet", "Key mechanics learned: ...") up to where the reasoning turns to
    the new level ("Now level 2", "Level 2 board", "Let me inspect"). Memory
    tags and "let me record ..." announcements are removed first. Whole
    sentences up to ``max_chars``; a recap that names no goal or rule word is
    dropped (the spend and path are already on the solved line).

    On the level-path-10x, mem-v2-10x and user-combo-10x v1 runs this keeps a
    recap at 24, 28 and 31 of 96, 100 and 113 level-ups (19-20 of 25 games),
    from the first two reasonings after each clear. The "how" sentence it
    replaces came from the clearing turn's last plan: 0 of 25 "how" notes in
    level-path-10x described the level's route.

    ``precise`` (what ARC3_MEM_DEBRIEF uses; see ``_precise_debrief``) also
    drops a recap that names another level, one that is only a celebration,
    and the unfinished sentence a cut leaves.
    """
    flat = " ".join(_TAGS_RE.sub(" ", str(text or "")).replace("</memory>", " ").split())
    flat = " ".join(_DEBRIEF_RECORD_RE.sub(" ", flat).split())
    if not flat or max_chars <= 0:
        return ""
    start = _DEBRIEF_START_RE.search(flat)
    if start is None:
        return ""
    head = flat[: start.start()]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    # back up to the start of the sentence holding the match, if it is close
    begin = cut + 2 if cut >= 0 and start.start() - cut < 80 else (0 if start.start() < 80 else start.start())
    stop = _debrief_stop_re(new_level).search(flat, start.end())
    body = flat[begin: stop.start() if stop else len(flat)].strip(" -,;:")
    if len(body) > max_chars:
        window = body[:max_chars]
        dot = max(window.rfind(". "), window.rfind("! "), window.rfind(".)"))
        body = window[: dot + 1] if dot >= 40 else cut_words(body, max_chars)
    if len(body) < 25 or not _DEBRIEF_RULE_RE.search(body):
        return ""
    if precise:
        return _precise_debrief(body, int(new_level) - 1)
    return body


def debrief_about_level(text: str, level: int) -> bool:
    """Whether a recap is about ``level`` by the level numbers it names.

    No number: yes (it is the first reasoning after the clear). Otherwise the
    first number must be ``level``, or the new level's number followed by
    ``level`` ("Level 2 solved level 1", "Level 2 solved... no wait, LEVEL 1
    COMPLETED!"): the model sometimes calls the clear by the number it now
    sees on the board.
    """
    refs = [int(m.group(1)) for m in _LEVEL_REF_RE.finditer(str(text or ""))]
    if not refs or refs[0] == level:
        return True
    return refs[0] == level + 1 and len(refs) > 1 and refs[1] == level


def _precise_debrief(body: str, level: int) -> str:
    """``extract_debrief(precise=True)``'s extra checks on a found recap.

    - It must be about ``level`` (``debrief_about_level``).
    - Without its opening celebration ("Level 4 completed! Wait -") at least
      40 chars must remain and name a goal or rule word (an arrow alone is a
      move trace: "LEVEL 3 COMPLETE... wait, LEVEL COMPLETED ->"; "Level 2
      solved level 1 with the same mechanic." says nothing).
    - A recap cut mid-sentence by the new level's first words ends at its
      last whole sentence, or is dropped when less than 40 chars remain.
    """
    if not debrief_about_level(body, level):
        return ""
    rest = _DEBRIEF_HEAD_RE.sub("", body, count=1).strip()
    if len(rest) < 40 or not _DEBRIEF_RULE_WORD_RE.search(rest):
        return ""
    if not re.search(r"[.!?)\]✓]$", body):
        end = max(body.rfind(". "), body.rfind("! "), body.rfind("? "))
        if end + 1 < 40:
            return ""
        body = body[: end + 1]
    return body


# ARC3_MEM_RECAP_ASK: the level-start message asks for "Level N recap:", and
# the recap is read by `extract_recap`. On mem-layout-10x (2026-10-11) the
# precise extractor kept 24 of 106 recaps: it dropped a recap that names the
# new level ("Level 2 solved mechanics: ..." for level 1), one with no word from
# its rule list ("Level 1 solved: rotate ring so the yellow block reaches the
# yellow bracket slot"), and one written after "Now level N+1" under a label
# ("Level 3 mechanics learned (worth recording): ..."). `extract_recap` keeps
# 44 of 106 on the same reasonings, before any asked recap.
RECAP_CHARS = 420
_RECAP_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
# words that carry no rule: celebration, counting, filler
_RECAP_EMPTY_WORDS = frozenset("""
level levels solved solve completed complete cleared clear won passed beaten wait yes nice great finally
actions action total took spent steps step clicks click moves move presses press probes probe
the and with for this that was were are has had have now next then also just whatever which from
into onto our its it's let lets let's me my i'm i'll will well ok okay hmm actually so but not
one two three four five six seven eight nine ten first last again plan worked works same
""".split())
# sentences about the harness or the frame, not about the level's rules
_RECAP_JUNK_RE = re.compile(
    r"level_completed|current[ _]frame|it says|says level|next level's|level \d+'s start|\bwait\b|\?!|\?\s*$"
    r"|animation was confirmed|my (?:model|hypothesis) (?:was right|held)|the plan worked",
    re.I,
)
# an unlabelled recap states a rule: a relation (=, ->, :) or a goal/causal word
_RECAP_RULEISH_RE = re.compile(
    r"=|→|->|:|\b(?:goal|win|wins|rule|mechanics?|key|insight|means|because|when|whenever|if|so that|so\b|"
    r"must|needs?|requires?|by\b|match|matching|reach|reaches|moves?|push|pushes|click|clicking|fill|filled|"
    r"align|aligned|rotate|rotates|merge|deliver|cover|place|placed|swap|toggle|grows?|shrinks?)",
    re.I,
)
_RECAP_LABELS = r"(?:recap|summary|mechanics?|rules?|insights?|world model|learnings?|knowledge|takeaways?|lessons?)"


def _recap_flat(text: str) -> str:
    flat = " ".join(_TAGS_RE.sub(" ", str(text or "")).replace("</memory>", " ").split())
    return " ".join(_DEBRIEF_RECORD_RE.sub(" ", flat).split())


def _recap_stop_re(new_level: int) -> re.Pattern:
    """Where the reasoning turns to the new board."""
    n = int(new_level)
    return re.compile(
        rf"\bnow\b[^.]{{0,20}}?\b(?:on\s+)?(?:level|L)\s*{n}\b"
        rf"|\bnext (?:is )?(?:level|L)\s*{n}\b"
        rf"|\b(?:level|L)\s*{n}\s*(?:board|layout|analysis|structure|start|now|:|!|\.)"
        r"|\b(?:level|L)\s*\d+\s*(?:board|layout|analysis|structure)\b"
        r"|\bnew (?:level|board|layout|elements?)\b|\bthe (?:new|current) board\b"
        r"|\blet me (?:inspect|look|parse|analy[sz]e|check|read|map|get|build|see|study|examine|view|organi[sz]e|figure)\b",
        re.I,
    )


def _recap_content_words(text: str) -> list[str]:
    return [w for w in _RECAP_WORD_RE.findall(text) if w.lower() not in _RECAP_EMPTY_WORDS]


def _recap_fit(body: str, max_chars: int) -> str:
    body = body.strip(" -,;:")
    if len(body) > max_chars:
        window = body[:max_chars]
        dot = max(window.rfind(". "), window.rfind("! "), window.rfind(".)"), window.rfind("; "))
        body = window[: dot + 1] if dot >= 60 else cut_words(body, max_chars)
    return body


# ARC3_MEM_RECAP_CHARS: the recap is kept whole up to one cap, for capture and
# display alike. pf-keep-10x kept 420 chars at capture (cutting at the last
# sentence end, with no mark) and showed 300: 82 of 105 shown recaps ended in
# "...", and 20 of 100 lost every goal word, which the model writes last.
RECAP_CUT_MARK = " ..."
# a sentence that names what completed the level
_RECAP_GOAL_RE = re.compile(
    r"\b(?:goal|win|wins|won|complet\w*|solv\w*|target|objective|clear(?:s|ed)? (?:the|when|once)|"
    r"finish\w*|exit|reach(?:ed|ing)?)\b",
    re.I,
)
# a correction or doubt of what came before ("Wait - actually the win was ...").
# The tidy keeps it: dropping it alone would leave the claim it corrects
# standing as a fact, and dropping both loses the sentence that held the goal
# (vc33 and lp85 level 1 in pf-keep-10x)
_RECAP_CORRECTION_RE = re.compile(r"\b(?:wait|actually)\b", re.I)
# _RECAP_JUNK_RE without "wait": questions, harness and frame talk
_RECAP_TIDY_JUNK_RE = re.compile(
    r"level_completed|current[ _]frame|it says|says level|next level's|level \d+'s start|\?!|\?\s*$"
    r"|animation was confirmed|my (?:model|hypothesis) (?:was right|held)|the plan worked",
    re.I,
)
_RECAP_SPLIT_RE = re.compile(r"[()\[\]]|(?<=[.!?;])\s+")


def _recap_sentences(text: str) -> list[str]:
    """Sentences (and ``;`` clauses) of a recap, never split inside brackets,
    so "(19 clicks incl. 3 probe clicks)" stays in its sentence."""
    out: list[str] = []
    depth = start = 0
    for match in _RECAP_SPLIT_RE.finditer(text):
        token = match.group()
        if token in "([":
            depth += 1
        elif token in ")]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(text[start: match.start()])
            start = match.end()
    out.append(text[start:])
    return [x for x in out if x.strip()]


def _recap_tidy(body: str) -> str:
    """Drop the recap's sentences that carry no rule: those with no content
    word ("Hmm.", "Level 3 completed!") and questions or harness talk. A
    correction ("Wait - actually ...") is kept, with what it corrects."""
    kept = [
        x for x in _recap_sentences(" ".join(body.split()))
        if _RECAP_CORRECTION_RE.search(x)
        or (_recap_content_words(x) and not _RECAP_TIDY_JUNK_RE.search(x))
    ]
    return " ".join(kept)


def _recap_fit_marked(body: str, max_chars: int, mark: str = RECAP_CUT_MARK) -> str:
    """``_recap_fit`` for ARC3_MEM_RECAP_CHARS: any cut ends in ``mark`` and
    the result stays within ``max_chars``. When the plain cut would lose the
    only sentence naming the goal, the leading sentences that fit are kept,
    then " ... ", then the goal sentence."""
    body = body.strip(" -,;:")
    if len(body) <= max_chars:
        return body
    room = max_chars - len(mark)
    window = body[:room]
    dot = max(window.rfind(". "), window.rfind("! "), window.rfind(".)"), window.rfind("; "))
    fitted = window[: dot + 1] + mark if dot >= 60 else cut_words(body, max_chars, marker=mark)
    if not _RECAP_GOAL_RE.search(body) or _RECAP_GOAL_RE.search(fitted.removesuffix(mark)):
        return fitted
    sentences = _recap_sentences(body)
    at = next(i for i, x in enumerate(sentences) if _RECAP_GOAL_RE.search(x))
    tail = mark if at < len(sentences) - 1 else ""
    goal = sentences[at]
    if len(goal) + len(tail) > max_chars - 40:
        goal, tail = cut_words(goal, max_chars - 40, marker=mark), ""
    keep: list[str] = []
    for sentence in sentences[:at]:
        if len(" ".join([*keep, sentence])) + len(" ... ") + len(goal) + len(tail) > max_chars:
            break
        keep.append(sentence)
    return (" ".join(keep) + " ... " if keep else "... ") + goal + tail


# ARC3_MEM_RECAP_ASK_V2: the parts the ask names, in any order
_RECAP_PART_RE = re.compile(r"(?<![A-Za-z])(goal|rules|corrects\s+(?:level|L)\s*(\d+))\s*\**\s*:", re.I)


def split_recap(text: str, cleared: int) -> tuple[str, str, list[tuple[int, str]]] | None:
    """A recap written as ARC3_MEM_RECAP_ASK_V2 asks: ``(goal, rules,
    corrections)``, each correction ``(K, claim)`` for an earlier level K;
    None when the recap has neither a ``Goal:`` nor a ``Rules:`` part. Text
    before the first part goes into the rules; a ``Corrects level K:`` that
    does not name an earlier level stays in the part it follows."""
    flat = " ".join(str(text or "").split())
    marks = [m for m in _RECAP_PART_RE.finditer(flat)
             if not m.group(2) or 1 <= int(m.group(2)) < int(cleared)]
    if not any(m.group(1).lower() in ("goal", "rules") for m in marks):
        return None
    tidy = lambda s: s.strip(" -*,;:")  # noqa: E731
    lead = tidy(flat[: marks[0].start()]) if marks else ""
    goal = rules = ""
    corrections: list[tuple[int, str]] = []
    for i, mark in enumerate(marks):
        body = tidy(flat[mark.end(): marks[i + 1].start() if i + 1 < len(marks) else len(flat)])
        kind = mark.group(1).lower()
        if kind == "goal":
            goal = f"{goal} {body}".strip()
        elif kind == "rules":
            rules = f"{rules} {body}".strip()
        elif body:
            corrections.append((int(mark.group(2)), body))
    rules = f"{lead} {rules}".strip() if lead else rules
    return goal, rules, corrections


def extract_recap(text: str, cleared: int, max_chars: int = RECAP_CHARS, *, tidy: bool = False) -> str:
    """The model's recap of level ``cleared`` from a reasoning on the next level, or ``""``.

    In order:
    1. A labelled recap of the cleared level anywhere in the reasoning: the
       asked "Level N recap:", or "Level N summary for notes:", "Level N
       mechanics learned (...):"; "Level N+1 recap:" counts too (the model
       sometimes numbers the clear by the new board). It runs to where the
       reasoning turns to the new board.
    2. The precise PR #148 recap (``extract_debrief(precise=True)``).
    3. A reasoning that opens on the clear ("Level N solved: ...", or "Level
       N+1 solved ..." right after level N was cleared): the rest of its
       opening, without counting asides and sentences about the harness, when
       it states a rule.

    ``tidy`` (ARC3_MEM_RECAP_CHARS) on 1 and 3: sentences with no rule are
    dropped (``_recap_tidy``) and a cut is marked and keeps the goal
    (``_recap_fit_marked``). 2 is unchanged.
    """
    flat = _recap_flat(text)
    if not flat or max_chars <= 0:
        return ""
    n, nxt = int(cleared), int(cleared) + 1
    label = re.search(
        rf"\b(?:level|L)\s*{n}\s*(?:'s\s+)?{_RECAP_LABELS}\b[^:.!?]{{0,40}}:", flat, re.I
    ) or re.search(rf"\b(?:level|L)\s*{nxt}\s*(?:recap|summary)\b[^:.!?]{{0,40}}:", flat, re.I)
    if label:
        stop = _recap_stop_re(nxt).search(flat, label.end())
        body = flat[label.end(): stop.start() if stop else len(flat)]
        body = _recap_fit_marked(_recap_tidy(body), max_chars) if tidy else _recap_fit(body, max_chars)
        if len(_recap_content_words(body)) >= 4:
            return body
    old = extract_debrief(text, nxt, precise=True)
    if old:
        return old
    head = re.match(
        rf"^\W*(?:(?:level|L)\s*(?:{n}|{nxt})\b\s*(?:is\s+|was\s+|has\s+been\s+)?"
        r"(?:solved|completed?|cleared|won|passed|beaten|done)\b)[\W_]*"
        r"(?:(?:no\s+)?wait\b[^.!?]*[.!?]\s*)?",
        flat, re.I,
    )
    if not head:
        return ""
    stop = _recap_stop_re(nxt).search(flat, head.end())
    body = flat[head.end(): stop.start() if stop else len(flat)]
    # counting asides: "(level total 39 ...)", "in 25 actions"
    body = re.sub(r"\([^()]*\d[^()]*\)", " ", body)
    body = re.sub(
        r"\b(?:in|with|took|total(?: for level \d+)?:?)\s+~?\d+\s*(?:\+\s*\d+\s*)*(?:effective\s+)?"
        r"(?:actions?|clicks?|moves?|steps?|presses?)\b[^.!?]*[.!?]?",
        " ", body, flags=re.I,
    )
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
    body = " ".join(x for x in sentences if not _RECAP_JUNK_RE.search(x))
    if tidy:
        body = _recap_tidy(body)
    if not _RECAP_RULEISH_RE.search(body):
        return ""
    body = _recap_fit_marked(body, max_chars) if tidy else _recap_fit(body, max_chars)
    if len(body) < 30 or len(_recap_content_words(body)) < 5:
        return ""
    return body


def cut_words(text: str, limit: int, *, marker: str = "...") -> str:
    """Cut at a word boundary to at most ``limit`` chars, adding ``marker``."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    if limit <= len(marker):
        return ""
    head = text[: limit - len(marker)]
    if " " in head:
        head = head[: head.rfind(" ")]
    return head.rstrip(" ,;:") + marker


# ---------------------------------------------------------------- solved lines

SOLVED_LINE_CHARS = 165
SOLVED_PATH_CHARS = 64


def _fit_path(sigs: Sequence[str], max_chars: int) -> str:
    """The run-length path, cut at whole runs to fit, with ``...+N`` for the rest."""
    runs = _rle_tokens(sigs)
    full = " ".join(_run_text(t, n) for t, n in runs)
    if len(full) <= max_chars:
        return full
    shown: list[str] = []
    covered = 0
    for token, count in runs:
        text = _run_text(token, count)
        rest = len(sigs) - covered - count
        if len(" ".join([*shown, text])) + len(f" ...+{rest}") > max_chars:
            break
        shown.append(text)
        covered += count
    return " ".join([*shown, f"...+{len(sigs) - covered}"])


def solved_head(
    level: int, path: Sequence[str], spent: int, deaths: int, *,
    path_chars: int = SOLVED_PATH_CHARS, winning: int | None = None,
) -> str:
    """``winning`` (ARC3_MEM_PATH): the loop-free length of the winning
    attempt, given when ``path`` is a shorter route on record."""
    rendered = _fit_path(path, path_chars) if path_chars > 0 else ""
    bracket = f" [{rendered}]" if rendered else ""
    if winning:
        bracket += f" ({winning} in the winning attempt)"
    return (
        f"L{level}: {len(path)} acts{bracket}, spent {spent}, "
        f"{deaths} death{'' if deaths == 1 else 's'}"
    )


def solved_line(
    level: int,
    path: Sequence[str],
    spent: int,
    deaths: int,
    *,
    think: str = "",
    note: str = "",
    cap: int = SOLVED_LINE_CHARS,
    path_chars: int = SOLVED_PATH_CHARS,
    winning: int | None = None,
    debrief: str = "",
) -> tuple[str, dict[str, str]]:
    """One ``<solved>`` line, at most ``cap`` chars, and the parts it kept.

    The head (path, spend, deaths) is always kept. The optional parts are then
    admitted in priority order while they fit: the model's own level note,
    then one distilled sentence of the clearing turn's reasoning ("how").
    They are rendered in the order head | how | note.

    ``debrief`` (ARC3_MEM_DEBRIEF; callers then pass no ``think`` and no
    ``note``): the model's recap from ``extract_debrief``, outside ``cap``
    and at most ``DEBRIEF_CHARS``, rendered last.
    """
    head = solved_head(level, path, spent, deaths, path_chars=path_chars, winning=winning)
    remaining = cap - len(head)
    kept: dict[str, str] = {}
    if note:
        room = remaining - len(" | note: ")
        cut = cut_words(note, room) if room >= 20 else ""
        if cut:
            kept["note"] = cut
            remaining -= len(" | note: ") + len(cut)
    how = distill_think(think, max_chars=min(160, remaining - len(" | how: "))) if think else ""
    if how:
        kept["how"] = how
    if debrief:
        kept["debrief"] = cut_words(debrief, DEBRIEF_CHARS)
    return render_solved(head, kept), kept


def render_solved(head: str, parts: dict[str, str]) -> str:
    line = head
    for name in ("how", "note", "debrief"):
        if parts.get(name):
            line += f" | {name}: {parts[name]}"
    return line


# ---------------------------------------------------------------- trackers


class AttemptTracker:
    """The current level's attempt: its steps, start board and spend."""

    def __init__(self) -> None:
        self.level: int | None = None
        self.attempt = 0
        self.steps: list[Step] = []
        self.start_h = ""
        self.spent = 0
        self.deaths = 0
        self.first_index: dict[str, int] = {}
        # the game's action number after each action of this attempt
        # (acts[0]: the attempt's start), for ARC3_MEM_SEEN_GAME_ACTION
        self.acts: list[int] = [0]

    def start_level(self, level: int, start_h: str, act: int = 0) -> None:
        self.level = level
        self.attempt = 1
        self.steps = []
        self.start_h = start_h
        self.spent = 0
        self.deaths = 0
        self.first_index = {start_h: 0}
        self.acts = [int(act)]

    def restart(self, start_h: str, act: int = 0) -> None:
        self.attempt += 1
        self.steps = []
        self.start_h = start_h
        self.first_index = {start_h: 0}
        self.acts = [int(act)]

    def act_at(self, index: int) -> int | None:
        """The game's action number after action ``index`` of this attempt."""
        return self.acts[index] if 0 <= index < len(self.acts) else None

    @property
    def index(self) -> int:
        return len(self.steps)

    def add(self, step: Step, act: int | None = None) -> None:
        self.steps.append(step)
        self.spent += 1
        self.first_index.setdefault(step.post_h, len(self.steps))
        self.acts.append(int(act) if act is not None else self.acts[-1] + 1)


class ActionEffects:
    """Counts of what each action key did, this level and this game."""

    def __init__(self) -> None:
        self.level_counts: Counter = Counter()
        self.game_counts: Counter = Counter()
        self.level_boards: dict[str, set[str]] = {}
        self.game_boards: dict[str, set[str]] = {}
        self.fatal_boards: dict[str, set[str]] = {}
        self.level_cells: dict[str, list[int]] = {}
        # click colour -> {(row, col): class}, this level, newest last
        self.targets: dict[str, OrderedDict] = {}
        self.order: list[str] = []

    def new_level(self) -> None:
        self.level_counts = Counter()
        self.level_boards = {}
        self.level_cells = {}
        self.targets = {}

    def record(self, key: str, sig: str, cls: str, pre_h: str, cells: int) -> bool:
        """Count one action; True when (key, cls) is new for the game."""
        first = (key, cls) not in self.game_counts
        self.level_counts[(key, cls)] += 1
        self.game_counts[(key, cls)] += 1
        if key not in self.order:
            self.order.append(key)
        for store in (self.level_boards, self.game_boards):
            boards = store.setdefault(key, set())
            if len(boards) < _MAX_BOARDS_PER_KEY:
                boards.add(pre_h)
        if cls == DEATH:
            fatal = self.fatal_boards.setdefault(key, set())
            if len(fatal) < _MAX_BOARDS_PER_KEY:
                fatal.add(pre_h)
        if cls in (SHIFT, CHANGE):
            samples = self.level_cells.setdefault(key, [])
            if len(samples) < _MAX_CELL_SAMPLES:
                samples.append(cells)
        cell = parse_click(sig)
        if cell is not None:
            targets = self.targets.setdefault(key, OrderedDict())
            targets.pop(cell, None)
            targets[cell] = cls
            while len(targets) > 64:
                targets.popitem(last=False)
        return first

    # ---- rendering

    @staticmethod
    def _counts(counter: Counter, key: str) -> list[tuple[str, int]]:
        return [(cls, counter[(key, cls)]) for cls in _CLASS_ORDER if counter[(key, cls)]]

    def _part(self, counts: list[tuple[str, int]], boards: int, key: str, scope_game: bool) -> str:
        total = sum(n for _, n in counts)
        if counts and all(cls in STILL for cls, _ in counts):
            return (
                f"no interior change on {total} press{'' if total == 1 else 'es'} "
                f"across {boards} board{'' if boards == 1 else 's'}"
            )
        bits = []
        for cls, n in counts:
            text = f"{LABELS[cls]} {n}"
            if cls == DEATH and scope_game:
                fatal = len(self.fatal_boards.get(key, ()))
                text += f" ({fatal} board{'' if fatal == 1 else 's'})"
            bits.append(text)
        return ", ".join(bits)

    def key_row(self, key: str) -> str:
        level = self._counts(self.level_counts, key)
        game = self._counts(self.game_counts, key)
        game_part = self._part(game, len(self.game_boards.get(key, ())), key, True) if game else ""
        if not level:
            return f"{key}: not tried this level" + (f" | game {game_part}" if game_part else "")
        level_part = self._part(level, len(self.level_boards.get(key, ())), key, True)
        if game == level:
            return f"{key}: {level_part}"
        return f"{key}: {level_part} | game {game_part}"

    def click_row(self, key: str) -> str:
        targets = self.targets.get(key, OrderedDict())
        counts = self._counts(self.level_counts, key)
        text = f"{key} {len(targets)} target{'' if len(targets) == 1 else 's'}: " + ", ".join(
            f"{LABELS[cls]} {n}" for cls, n in counts
        )
        cells = self.level_cells.get(key)
        if cells:
            text += f"; median {int(statistics.median(cells))} cells"
        newest = list(reversed(targets.items()))
        picked = newest[:1]
        if newest:
            other = [t for t in newest[1:] if t[1] != newest[0][1]]
            picked += other[:1] or newest[1:2]
        examples = [f"({row},{col}) {LABELS[cls]}" for (row, col), cls in picked]
        if examples:
            text += "; e.g. " + ", ".join(examples)
        return text

    def render(
        self,
        valid_actions: Sequence[str],
        *,
        board_colours: Sequence[str] = (),
        max_chars: int = 1000,
        max_click_rows: int = 11,
    ) -> list[str]:
        """Rows for the card's ``<actions>`` section, within ``max_chars``."""
        valid = [str(a).upper() for a in valid_actions or []]
        rows: list[str] = []
        keyboard = [a for a in valid if a not in ("MOUSE", "RESET")]
        for key in self.order:
            if not key.startswith("MOUSE@") and key not in keyboard and key != "RESET":
                keyboard.append(key)
        never: list[str] = []
        for key in keyboard:
            if key == "RESET":
                continue
            if any(k == key for k, _ in self.game_counts):
                rows.append(self.key_row(key))
            elif key in valid:
                never.append(key)
        if never:
            rows.append("not tried yet this game: " + ", ".join(never))
        clicked = [k for k in self.order if k.startswith("MOUSE@") and any(
            kk == k for kk, _ in self.level_counts
        )]
        if clicked:
            def rank(key: str) -> tuple[int, int]:
                urgent = self.level_counts[(key, CLEAR)] + self.level_counts[(key, DEATH)]
                total = sum(n for (k, _), n in self.level_counts.items() if k == key)
                return (0 if urgent else 1, -total)

            for key in sorted(clicked, key=rank)[:max_click_rows]:
                rows.append(self.click_row(key))
        if "MOUSE" in valid and board_colours:
            used = {k.split("@", 1)[1] for k in clicked}
            unused = [c for c in board_colours if c not in used]
            if unused:
                rows.append("colours on the board not clicked this level: " + " ".join(unused))
        out: list[str] = []
        total = 0
        for row in rows:
            if total + len(row) + 1 > max_chars:
                break
            out.append(row)
            total += len(row) + 1
        return out


@dataclass
class _Board:
    visits: list[tuple[int, int]] = field(default_factory=list)
    tried: OrderedDict = field(default_factory=OrderedDict)
    # (attempt, index) -> the game's action number then (ARC3_MEM_SEEN_GAME_ACTION)
    acts: dict = field(default_factory=dict)


class SeenStates:
    """Boards visited on this level and what was tried from each (LRU-capped)."""

    def __init__(self, max_boards: int = 512, max_actions: int = 16, max_visits: int = 4) -> None:
        self.max_boards = max_boards
        self.max_actions = max_actions
        self.max_visits = max_visits
        self.boards: OrderedDict[str, _Board] = OrderedDict()

    def reset(self) -> None:
        self.boards = OrderedDict()

    def _entry(self, board: str) -> _Board:
        entry = self.boards.get(board)
        if entry is None:
            entry = _Board()
            self.boards[board] = entry
            while len(self.boards) > self.max_boards:
                self.boards.popitem(last=False)
        else:
            self.boards.move_to_end(board)
        return entry

    def get(self, board: str) -> _Board | None:
        return self.boards.get(board)

    def has_visit(self, board: str) -> bool:
        entry = self.boards.get(board)
        return bool(entry and entry.visits)

    def visit(self, board: str, attempt: int, index: int, act: int | None = None) -> None:
        entry = self._entry(board)
        if (attempt, index) not in entry.visits:
            entry.visits.append((attempt, index))
            del entry.visits[:-self.max_visits]
            if act is not None:
                entry.acts[(attempt, index)] = int(act)
                for key in [k for k in entry.acts if k not in entry.visits]:
                    del entry.acts[key]

    def tried_before(self, board: str, sig: str) -> bool:
        entry = self.boards.get(board)
        return bool(entry and sig in entry.tried)

    def try_action(self, board: str, sig: str, kind: str, next_board: str, attempt: int) -> None:
        """Record ``sig`` tried from ``board``: ``kind`` is clear, death, noop
        or move; what a move reached is worded when the line is rendered."""
        entry = self._entry(board)
        entry.tried.pop(sig, None)
        entry.tried[sig] = (kind, next_board, attempt)
        while len(entry.tried) > self.max_actions:
            entry.tried.popitem(last=False)


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _board_ref(visits: Sequence[tuple[int, int]], attempt: int) -> str:
    """Name a board by one of its visits, this attempt's if any: the start
    board as such, otherwise the latest visit."""
    pool = [v for v in visits if v[0] == attempt] or list(visits)
    a, index = next((v for v in pool if v[1] == 0), pool[-1])
    owner = "" if a == attempt else f" of attempt {a}"
    return f"the start board{owner}" if index == 0 else f"the board after action #{index}{owner}"


def _visits_phrase(visits: Sequence[tuple[int, int]], attempt: int) -> str:
    grouped: OrderedDict[int, list[int]] = OrderedDict()
    for a, index in visits:
        grouped.setdefault(a, []).append(index)
    phrases = []
    for a, indexes in grouped.items():
        owner = "this attempt" if a == attempt else f"attempt {a}"
        bits = []
        if 0 in indexes:
            bits.append("at the start")
        acts = [f"#{i}" for i in indexes if i]
        if acts:
            bits.append("after action " + _join(acts))
        phrases.append(" and ".join(bits) + f" of {owner}")
    return "; ".join(phrases)




# ---------------------------------------------------------------- recorder


@dataclass
class FactEvent:
    kind: str  # first | death | clear | level
    level: int
    data: dict[str, Any] = field(default_factory=dict)


class FactRecorder:
    """Feeds every executed action through the trackers, once each.

    ``feed(entries)`` takes the solver's full history (``HistoryEntry`` with
    ``action``, ``frame`` and ``result``), processes the entries it has not
    seen and returns what happened that the memory needs to say: a (key,
    class) pair new for the game, a death, a level clear (with the loop-free
    path and the spend), a level change.

    With ``route`` on (ARC3_MEM_PATH) it also keeps a ``LevelGraph`` per
    level and one ``RouteRow`` per executed action in ``route_rows``.
    """

    def __init__(self, border: int = 4, *, route: bool = False, keep_still: bool = False) -> None:
        self.border = max(0, int(border))
        self.route = bool(route)
        # ARC3_MEM_KEEP_EDGE: the solved path keeps edge-only and anim-only actions
        self.keep_still = bool(keep_still)
        # ARC3_MEM_ROUTE_FORMAT=readable: a clear also reports each path
        # action's effect key ("keys"), so a click can name its cell's colour
        self.route_keys = False
        # ARC3_EFFECT_MOVED_FIX: a key press whose interior changes are all
        # objects translating counts as SHIFT even when colour counts changed
        self.moved_fix = False
        self.graph: LevelGraph | None = None
        self.route_rows: list[RouteRow] = []
        # rows dropped from the front of route_rows; row numbers are absolute
        self.route_base = 0
        # the last board before this level's latest GAME OVER (routes on)
        self.death_board: str | None = None
        self._acts = 0
        self.cursor = 0
        self.attempts = AttemptTracker()
        self.effects = ActionEffects()
        self.seen = SeenStates()
        self.current_h = ""
        self.position: tuple[int, int] | None = None
        self.board_colours: list[str] = []
        # ARC3_MEM_COLOURS_FIRST: the latest board and this level's first
        # board, read only by ``new_colour_groups``
        self.board_grid: Grid | None = None
        self.level_start_grid: Grid | None = None
        self.stats: Counter = Counter()
        self._last_core: list[tuple[int, ...]] | None = None
        self._last_hash = ""
        self._last_grid: Grid | None = None

    # ---- feeding

    def feed(self, entries: Sequence[Any]) -> list[FactEvent]:
        events: list[FactEvent] = []
        if self.cursor == 0 and entries:
            first = entries[0].frame
            if first is not None:
                self._begin_level(int(first.level), first.grid, events, announce=False)
            self.cursor = 1
        for index in range(max(1, self.cursor), len(entries)):
            pre = entries[index - 1].frame
            entry = entries[index]
            post = entry.frame
            self.cursor = index + 1
            if pre is None or post is None:
                continue
            self._grid_step(entry, pre, post, events, act=index)
        if entries and entries[-1].frame is not None:
            self.board_colours = self._colours(entries[-1].frame.grid)
            self.board_grid = entries[-1].frame.grid
        return events

    def new_colour_groups(self) -> dict[str, list[list[tuple[int, int]]]]:
        """ARC3_MEM_COLOURS_FIRST: each of ``board_colours`` that is absent
        from this level's first board (both inside the ``border`` crop), with
        its 4-connected groups of cells on the latest board, largest first,
        in board coordinates. {} when either board is unknown."""
        start = getattr(self, "level_start_grid", None)
        grid = getattr(self, "board_grid", None)
        if start is None or grid is None:
            return {}
        before = set(_histogram(crop(start, self.border)))
        out: dict[str, list[list[tuple[int, int]]]] = {}
        for char in self.board_colours:
            value = ARC_COLOR_CHARS.find(char)
            if value < 0 or value in before:
                continue
            groups = colour_groups(grid, value, self.border)
            if groups:
                out[char] = groups
        return out

    def _colours(self, grid: Grid) -> list[str]:
        counts = _histogram(crop(grid, self.border))
        if not counts:
            return []
        background = counts.most_common(1)[0][0]
        return [
            ARC_COLOR_CHARS[max(0, min(15, int(value)))]
            for value, _ in sorted(counts.items())
            if value != background
        ]

    def _core(self, grid: Grid) -> tuple[list[tuple[int, ...]], str]:
        """Cropped board and its hash; the previous action's board is reused."""
        if self._last_grid is grid and self._last_core is not None:
            return self._last_core, self._last_hash
        core = crop(grid, self.border)
        return core, hashlib.sha1(repr(core).encode("utf-8")).hexdigest()

    def _grid_step(self, entry: Any, pre: Any, post: Any, events: list[FactEvent], *, act: int | None = None) -> None:
        result = dict(getattr(entry, "result", None) or {})
        if isinstance(result.get("action_num"), int):
            act = result["action_num"]
        action = str(getattr(entry, "action", "") or "").strip()
        pre_core, pre_h = self._core(pre.grid)
        post_core, post_h = self._core(post.grid)
        self._last_grid, self._last_core, self._last_hash = post.grid, post_core, post_h
        if action.upper() == "RESET":
            self.reset_step(int(post.level), post_h, post.grid, manual=not result.get("automatic"), events=events, act=act)
            return
        if result.get("level_completed") or result.get("run_complete"):
            cls = CLEAR
        elif result.get("game_over"):
            cls = DEATH
        else:
            cls = _classify_cores(result, pre.grid, post.grid, pre_core, post_core)
        sig = action_signature(action)
        if cls == CHANGE and self.moved_fix and parse_click(sig) is None:
            # a key press only: a click that grows or redraws a shape (s5i5's
            # arms) can pass the translation test too
            from inference.agent.object_facts import pure_move

            if pure_move(pre.grid, post.grid, self.border):
                cls = SHIFT
        cells = changed_cells(pre_core, post_core) if cls in (SHIFT, CHANGE) else 0
        self.step(
            int(pre.level), sig, effect_key(sig, pre.grid), cls, pre_h, post_h,
            cells=cells, pre_grid=pre.grid, post_grid=post.grid,
            post_level=int(post.level), events=events, act=act,
        )

    def _begin_level(self, level: int, grid: Grid | None, events: list[FactEvent], *, start_h: str | None = None, announce: bool = True) -> None:
        start = start_h if start_h is not None else (state_hash(grid, self.border) if grid is not None else "")
        self.level_start_grid = grid
        self.attempts.start_level(level, start, self._acts)
        self.effects.new_level()
        self.seen.reset()
        self.seen.visit(start, 1, 0, self._acts)
        self.current_h = start
        self.position = (1, 0)
        if self.route:
            self.graph = LevelGraph(level, start)
            self.death_board = None
        if announce:
            events.append(FactEvent("level", level))

    def reset_step(
        self, level: int, start_h: str, grid: Grid | None, *, manual: bool,
        events: list[FactEvent], act: int | None = None,
    ) -> None:
        """A RESET entry (automatic after a death, or the model's own) closes the attempt."""
        self._acts = int(act) if act is not None else self._acts + 1
        pre_h = self.current_h
        if self.attempts.level is None or level != self.attempts.level:
            self._begin_level(level, grid, events, start_h=start_h)
        else:
            if manual:
                self.attempts.spent += 1
            self.attempts.restart(start_h, self._acts)
            self.seen.visit(start_h, self.attempts.attempt, 0, self._acts)
            self.current_h = start_h
            self.position = (self.attempts.attempt, 0)
        if self.graph is not None:
            self._route_guarded(self._route_reset, pre_h, start_h, automatic=not manual)

    def step(
        self,
        level: int,
        sig: str,
        key: str,
        cls: str,
        pre_h: str,
        post_h: str,
        *,
        cells: int = 0,
        pre_grid: Grid | None = None,
        post_grid: Grid | None = None,
        post_level: int | None = None,
        events: list[FactEvent],
        act: int | None = None,
    ) -> None:
        """One executed action. Also the entry point for hash-only replays."""
        self._acts = int(act) if act is not None else self._acts + 1
        if self.attempts.level is None or level != self.attempts.level:
            self._begin_level(level, pre_grid, events, start_h=pre_h)
        tracker = self.attempts
        attempt = tracker.attempt
        self.stats["actions"] += 1
        if self.seen.tried_before(pre_h, sig):
            self.stats["exact_repeats"] += 1
        if cls == CLEAR:
            kind = "clear"
        elif cls == DEATH:
            kind = "death"
        elif cls in STILL or post_h == pre_h:
            kind = "noop"
        else:
            kind = "move"
            self.stats["moves"] += 1
            if self.seen.has_visit(post_h):
                self.stats["revisits"] += 1
        self.seen.try_action(pre_h, sig, kind, post_h, attempt)
        game_first = self.effects.record(key, sig, cls, pre_h, cells)
        tracker.add(Step(sig, pre_h, post_h, cls, key), self._acts)
        if self.graph is not None:
            self._route_guarded(self._route_step, sig, kind, pre_h, post_h)
        if game_first and cls not in (DEATH, CLEAR):
            events.append(FactEvent("first", level, {"key": key, "cls": cls, "sig": sig}))
        if cls == DEATH:
            tracker.deaths += 1
            first_index = tracker.first_index.get(pre_h, tracker.index - 1)
            events.append(FactEvent("death", level, {
                "key": key, "sig": sig, "attempt": attempt, "length": tracker.index,
                "first_index": first_index,
                "game_first": game_first,
                # the game's action numbers (ARC3_MEM_SEEN_GAME_ACTION words them)
                "act": self._acts, "first_act": tracker.act_at(first_index),
            }))
            self.current_h = post_h
            return
        if cls == CLEAR:
            kept = loop_free_steps(tracker.steps, tracker.start_h, keep_still=self.keep_still)
            path = [s.sig for s in kept]
            data = {
                "key": key, "sig": sig, "path": path, "spent": tracker.spent,
                "deaths": tracker.deaths, "attempt": attempt,
                "game_first": game_first, "actions": tracker.index,
            }
            route = self._route_guarded(self.graph.route, pre_h) if self.graph is not None else None
            if route is not None:
                shorter = len(route) + 1 < len(path)
                # loop_free keeps an UNDO that lands on a board cut from its
                # chain; UNDO adds no link, so such a path is not a route
                # on record (and replays wrongly from the start board)
                undo = any(s in NO_LINK_SIGS for s in path[:-1])
                if shorter or undo:
                    # the route on record to the clearing board, then the
                    # clearing action; the winning attempt's own length is
                    # kept only when it was longer
                    data["path"] = [*route, sig]
                    if shorter:
                        data["winning"] = len(path)
            if self.route_keys:
                if data["path"] is path:
                    data["keys"] = [s.key for s in kept]
                else:
                    # a route on record: each action's key as last seen on this attempt
                    last = {s.sig: s.key for s in tracker.steps}
                    data["keys"] = [last.get(s, "") for s in data["path"]]
            events.append(FactEvent("clear", level, data))
            next_level = post_level if post_level is not None and post_level != level else level + 1
            self._begin_level(next_level, post_grid, events, start_h=post_h)
            return
        if post_h != pre_h and cls not in STILL:
            self.seen.visit(post_h, attempt, tracker.index, self._acts)
            self.position = (attempt, tracker.index)
        self.current_h = post_h

    # ---- routes (ARC3_MEM_PATH)

    def _route_guarded(self, hook: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a route hook. On any error the error is counted and this
        level's graph is dropped (the next level starts a new one), so the
        memory-v2 facts come out as they would without routes; None then."""
        try:
            return hook(*args, **kwargs)
        except Exception:  # noqa: BLE001 - routes are an extra
            self.stats["route_errors"] += 1
            self.graph = None
            return None

    def _route_add(self, row: RouteRow) -> None:
        self.route_rows.append(row)
        if len(self.route_rows) > ROUTE_ROWS_KEPT:
            drop = len(self.route_rows) - ROUTE_ROWS_KEPT // 2
            del self.route_rows[:drop]
            self.route_base += drop

    @property
    def route_total(self) -> int:
        """Rows recorded so far (absolute, including dropped ones)."""
        return self.route_base + len(self.route_rows)

    def route_rows_since(self, start: int) -> list[RouteRow]:
        return self.route_rows[max(0, int(start) - self.route_base):]

    def _route_step(self, sig: str, kind: str, pre_h: str, post_h: str) -> None:
        """Link a move into this level's graph and record the route after it."""
        graph = self.graph
        tracker = self.attempts
        full = graph.full
        pre_d = graph.dist.get(pre_h)
        before = graph.dist.get(post_h)
        known = post_h in graph.boards
        link = sig not in NO_LINK_SIGS
        path: list[str] | None = None
        if kind == "move":
            if link:
                graph.add_move(pre_h, sig, post_h)
            else:
                graph.add_board(post_h)
            d = graph.dist.get(post_h)
            if not known:
                change = ROUTE_NEW
            elif d is not None and before is not None and d < before:
                change = ROUTE_SHORTER
            else:
                # on record already; this also covers a board whose first
                # route this link made (reached before only by UNDO or from
                # a board with no route): 0 such rows in r2 or fz-experts8
                change = ROUTE_KNOWN
            path = graph.route(post_h)
        elif kind == "noop":
            if link:
                graph.note_other(pre_h, sig)
            d, change = graph.dist.get(post_h), ROUTE_STILL
            path = graph.route(post_h)
        elif kind == "death":
            if link:
                graph.note_other(pre_h, sig)
            d, change = None, DEATH
            self.death_board = pre_h
        else:
            d, change = None, CLEAR
            route = graph.route(pre_h)
            path = None if route is None else [*route, sig]
        self.stats["graph_full"] += graph.full - full
        varied = tuple(graph.route_varied(post_h)) if kind in ("move", "noop") else ()
        self._route_add(RouteRow(
            act=self._acts, level=graph.level, attempt=tracker.attempt, i=tracker.index, sig=sig,
            d=d, k=tracker.index, change=change, before=before, pre_d=pre_d, pre=pre_h,
            board=post_h, varied=varied, path=_fit_path(path, 200) if path else "",
        ))

    def _route_reset(self, pre_h: str, start_h: str, *, automatic: bool) -> None:
        graph = self.graph
        full = graph.full
        pre_d = graph.dist.get(pre_h)
        graph.set_start(start_h)
        self.stats["graph_full"] += graph.full - full
        self._route_add(RouteRow(
            act=self._acts, level=graph.level, attempt=self.attempts.attempt, i=0, sig="RESET",
            d=graph.dist.get(start_h), k=0, change=ROUTE_START, before=graph.dist.get(start_h),
            pre_d=pre_d, pre=pre_h, board=start_h, automatic=automatic,
        ))

    def route_state(self) -> dict[str, Any] | None:
        """``level_path`` for the python sandbox: the route on record to the
        current board as a list ``action()`` accepts, its length and the
        indexes of its varied steps; path and length are None when no
        recorded links reach the board (or the level's graph was dropped
        after a route error). ``before_death`` is the route on record from
        the start board to the last board before this level's latest GAME
        OVER, in the same form: the turn line cuts it at 40 chars, and after
        the automatic RESET ``path`` is ``[]``. None before a GAME OVER on
        the level or when no recorded links reach that board. None with
        routes off."""
        if not self.route:
            return None
        none = {"path": None, "len": None, "varied": [], "before_death": None}
        graph = self.graph
        if graph is None:  # dropped after a route error: no route on record
            return none
        state = self._route_guarded(self._route_state_of, graph, self.current_h, self.death_board)
        return none if state is None else state

    @staticmethod
    def _route_state_of(graph: LevelGraph, board: str, death_board: str | None = None) -> dict[str, Any]:
        died = graph.route(death_board) if death_board is not None else None
        before_death = None if died is None else [action_item(sig) for sig in died]
        route = graph.route(board)
        if route is None:
            return {"path": None, "len": None, "varied": [], "before_death": before_death}
        return {
            "path": [action_item(sig) for sig in route],
            "len": len(route),
            "varied": graph.route_varied(board),
            "before_death": before_death,
        }

    def route_line(self, since: int, *, max_chars: int = 300) -> str:
        """The turn message's ``Shortest route on record`` line, or ``""``
        when no action ran on the current level since row ``since``.

        It gives the route to the current board, the attempt's actions so
        far, the route after each of this attempt's actions since ``since``
        (named "of attempt N" when an earlier attempt also acted since then),
        how many route steps had more than one outcome, the latest shorter
        route found, and how an attempt that ended in GAME OVER since then
        relates to the routes. Over ``max_chars`` the per-action list goes
        first, then the varied and shorter notes, then the GAME OVER part.
        """
        graph = self.graph
        if graph is None:
            return ""
        level = self.attempts.level
        rows = [row for row in self.route_rows_since(since) if row.level == level]
        if not rows:
            return ""
        current = self.current_h
        d = graph.dist.get(current)
        if d is None:
            route = "none on record"
        elif d == 0:
            route = "0 actions (this is the start board)"
        else:
            route = f"{_count(d, 'action')} [{_fit_path(graph.route(current) or [], SOLVED_PATH_CHARS)}]"
        core = f"Shortest route on record from this level's start board to the current board: {route}"
        if self.attempts.index:
            core += f"; this attempt so far: {_count(self.attempts.index, 'action')}"
        core += "."
        death = ""
        ended = [row for row in rows if row.change == DEATH]
        if ended:
            row = ended[-1]
            reach = graph.dist.get(row.pre)
            if reach is None:
                where = "its last board before GAME OVER has no route on record"
            elif reach == 0:
                where = "its last board before GAME OVER is the start board"
            else:
                where = (
                    f"its last board before GAME OVER is {_count(reach, 'action')} from the start on record "
                    f"[{_fit_path(graph.route(row.pre) or [], 40)}]"
                )
            death = f" Attempt {row.attempt} ended in GAME OVER after {_count(row.k, 'action')}; {where}."
        attempt = self.attempts.attempt
        shown = [row for row in rows if row.attempt == attempt and not row.automatic]
        # an earlier attempt's actions since then are left out: name the attempt
        earlier = any(row.attempt != attempt and not row.automatic for row in rows)
        listing = ""
        if shown:
            scope = f" of attempt {attempt}" if earlier else ""
            listing = (
                f" Route length after each action{scope} since the last message: "
                + " ".join(route_value(row) for row in shown) + "."
            )
        notes = ""
        varied = len(graph.route_varied(current)) if d else 0
        if varied:
            notes += f" {_count(varied, 'step')} on the route had more than one outcome on this level."
        shorter = [row for row in shown if row.change == ROUTE_SHORTER]
        if shorter:
            row = shorter[-1]
            notes += (
                f" A shorter route to a known board was found at action #{row.i} of this attempt "
                f"({row.before} -> {row.d})."
            )
        for text in (core + death + listing + notes, core + death + notes, core + death, core):
            if len(text) <= max_chars:
                return text
        return ""

    # ---- views

    def effects_rows(self, valid_actions: Sequence[str], *, max_chars: int = 1000) -> list[str]:
        return self.effects.render(valid_actions, board_colours=self.board_colours, max_chars=max_chars)

    def seen_line(self, valid_actions: Sequence[str], *, hide_fatal: bool = False, max_chars: int = 240) -> str:
        """``Seen before ...`` when the current board was visited earlier on
        this level; ``""`` at an attempt's start board or on a first visit."""
        position = self.position
        if position is None or position[1] == 0:
            return ""
        entry = self.seen.get(self.current_h)
        if entry is None or not any(v != position for v in entry.visits):
            return ""
        attempt = self.attempts.attempt
        head = f"Seen before (board interior matches): {_visits_phrase(entry.visits, attempt)}."
        items: list[str] = []
        tried_keys: set[str] = set()
        for sig, (kind, next_board, at) in reversed(entry.tried.items()):
            tried_keys.add(sig)
            if hide_fatal and kind == "death":
                continue
            items.append(f"{display_action(sig)} -> {self._outcome(kind, next_board, at, attempt)}")
        valid = [str(a).upper() for a in valid_actions or []]
        untried = [a for a in valid if a not in ("MOUSE", "RESET") and a not in tried_keys]
        tail = f" Untried: {', '.join(untried)}." if untried else ""
        for limit in range(min(4, len(items)), -1, -1):
            shown = items[:limit]
            more = len(items) - limit
            middle = ""
            if shown:
                middle = " From it you tried: " + ", ".join(shown) + (f", +{more} more" if more else "") + "."
            elif items:
                middle = f" From it you tried {len(items)} action{'' if len(items) == 1 else 's'}."
            text = head + middle + tail
            if len(text) <= max_chars:
                return text
        return head[:max_chars] if len(head) <= max_chars else ""

    def _outcome(self, kind: str, next_board: str, at: int, attempt: int) -> str:
        """What a tried action did, worded now rather than when it was tried:
        a board that was new then has been seen since, so a move names the
        board it reached by one of that board's visits."""
        if kind == "clear":
            return "CLEAR"
        if kind == "death":
            return f"GAME OVER (attempt {at})"
        if kind == "noop":
            return "no board change"
        reached = self.seen.get(next_board)
        if reached is None or not reached.visits:
            return "moved"  # that board was evicted from the LRU
        return _board_ref(reached.visits, attempt)


# ---------------------------------------------------------------- route text

ROUTE_SUFFIX_CHARS = 80


def _num(value: int | None) -> str:
    return "-" if value is None else str(value)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def route_record(row: RouteRow) -> dict[str, Any]:
    """One line of ``*_level_path.jsonl``: the route after one action."""
    return {
        "act": row.act, "level": row.level, "attempt": row.attempt, "i": row.i, "sig": row.sig,
        "board": row.board[:10], "d": row.d, "k": row.k, "change": row.change,
        "varied": list(row.varied), "path": row.path,
    }


def route_value(row: RouteRow) -> str:
    """One action's entry in a route list: the route length, X for GAME
    OVER, C for a clear, - when no recorded links reach the board."""
    if row.change == DEATH:
        return "X"
    if row.change == CLEAR:
        return "C"
    return _num(row.d)


def route_change_text(row: RouteRow) -> str:
    if row.change == ROUTE_NEW:
        return "new board"
    if row.change == ROUTE_SHORTER:
        return f"shorter than {row.before} on record"
    if row.change == ROUTE_STILL:
        return "no board change"
    if row.change == ROUTE_START:
        return "start board"
    if row.change == DEATH:
        if row.pre_d is None:
            return "GAME OVER from a board with no route on record"
        return f"GAME OVER from a board at route {row.pre_d}"
    if row.change == CLEAR:
        return "level cleared" if row.pre_d is None else f"level cleared; {row.pre_d + 1} on record"
    return "known board"


def route_suffix(rows: Sequence[RouteRow], *, max_chars: int = ROUTE_SUFFIX_CHARS) -> str:
    """What an action(...) call's echo line gains: `` | route 9 (new board)``
    for one action, `` | route 0->3 (1 2 1 2 3)`` for a batch (the route
    before the call, then after it and after each action; more than 12 show
    the first and last 5; `` | route - -> 3 (- 3)`` when a side has no
    route). Automatic RESET rows are left out. ``""`` when no action ran."""
    rows = [row for row in rows if not row.automatic]
    if not rows:
        return ""
    if len(rows) == 1:
        row = rows[0]
        value = "-" if row.change in (DEATH, CLEAR) else _num(row.d)
        return f" | route {value} ({route_change_text(row)})"[:max_chars]
    values = [route_value(row) for row in rows]
    first, last = _num(rows[0].pre_d), values[-1]
    # "-->-" is hard to read: spaces around the arrow when a side is "-"
    arrow = " -> " if "-" in (first, last) else "->"
    head = f" | route {first}{arrow}{last}"
    note = "; shorter route found" if any(row.change == ROUTE_SHORTER for row in rows) else ""
    for keep in (5, 3, 2):
        shown = values if len(values) <= 12 and keep == 5 else [*values[:keep], "...", *values[-keep:]]
        if len(values) <= 2 * keep + 1:
            shown = values
        text = f"{head} ({' '.join(shown)}{note})"
        if len(text) <= max_chars:
            return text
    text = f"{head} ({len(values)} actions{note})"
    return text if len(text) <= max_chars else head[:max_chars]


def hazard_line(level: int, sig: str, first_index: int, attempt: int, length: int) -> str:
    """How an attempt ended, never why: the harness cannot tell a hazard from a
    budget that ran out (17 of 44 Fable-run deaths had the step bar empty or
    nearly so), so the last action is named but not blamed. ``first_index``
    and ``length`` count actions within the attempt."""
    where = "its start board" if first_index <= 0 else f"the board first reached at action {first_index} of the attempt"
    return (
        f"L{level}: attempt {attempt} ended in GAME OVER after {length} action{'' if length == 1 else 's'}; "
        f"last action {display_action(sig)} from {where}"
    )


def first_effect_text(key: str, cls: str, sig: str) -> str:
    text = f"{key}: first {LABELS.get(cls, cls)}"
    if key.startswith("MOUSE@"):
        text += f" at {display_action(sig)}"
    return text


def clear_text(level: int, sig: str, attempt: int, actions: int, spent: int) -> str:
    return (
        f"L{level} cleared by {display_action(sig)} (attempt {attempt}, action {actions}; "
        f"{spent} actions on the level)"
    )
