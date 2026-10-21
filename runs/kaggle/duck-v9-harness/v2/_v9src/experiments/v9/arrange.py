"""A planner for the games this benchmark actually contains.

`v9.planner` searches over one body walking a fixed board and asks what it reaches. Reading the
condition that guards a level advance in all 24 public games says that describes none of them:
eight need every piece of a kind to end up somewhere particular, and not one ends on "how many of
a kind are left", which is the predicate the rest of the goal layer is built around.

So this searches the other thing. The state is where every movable piece is and which of them the
controls are currently driving; a direction action moves the driven piece, and an action that
moves nothing may be the one that changes which piece that is. On re86, read from its source, that
is exactly the game: ACTION5 advances a selection cursor by recolouring a centre pixel, the
direction keys move whichever piece is selected, and the level ends when the pieces compose into a
target picture.

Nothing here is assumed about which action cycles the selection. Every action that has never been
seen to move anything is offered to the search as a candidate, the plan is replayed before it is
committed, and every step carries a prediction, so a wrong guess halts the batch at the first step
instead of spending it. That is the same contract the rest of the harness runs on: propose,
validate by replay, and let the ledger decide.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# The search is exponential in this and blind: breadth-first over the joint position of every
# piece plus which one is driven, with no heuristic. Measured on re86 with every object taken as a
# piece, all six target classes came back `no_plan_within_budget` at 20,000 nodes, so the number
# below is not a performance knob, it is the honest limit of this search. Making it useful on a
# board with eight rings needs a different search -- a heuristic, or placing pieces one at a time
# and holding the ones already placed -- not a bigger budget.
MAX_PIECES = 6
MAX_NODES = 30000


@dataclass(frozen=True)
class ArrangeState:
    """Where every piece is, and which one the controls drive."""
    spots: tuple[tuple[int, int], ...]     # top-left of each piece, in piece order
    driven: int
    steps: int = 0


@dataclass
class ArrangeModel:
    """An executable model of "move the piece you are driving; some button changes which".

    Honours the Tycho planner contract (`actions`, `transition`, `outcome`, `planner_key`), so the
    same vendored search and the same `validate_plan` that check a walking plan check this one.
    """
    grid: list[list[int]]
    shapes: tuple[frozenset[tuple[int, int]], ...]   # each piece's cells, relative to its top-left
    shifts: dict[str, tuple[int, int]]
    cyclers: tuple[str, ...]
    wall_colours: frozenset[int]
    targets: frozenset[tuple[int, int]]              # cells a piece may come to rest on
    # How the targets have to be satisfied. "touch" is the weak reading -- every piece is standing
    # on some target -- and it is what `covered` says. "cover" is what a template asks for: every
    # target cell has a piece on it. re86's win predicate composites the pieces and compares the
    # picture, so two pieces on one square satisfies the first and cannot satisfy the second.
    mode: str = "touch"
    max_steps: int = 200
    TRANSITION_RETURNS_NEW_STATE = True

    def __post_init__(self) -> None:
        self._h, self._w = len(self.grid), len(self.grid[0]) if self.grid else 0

    # -- the contract --------------------------------------------------------------------------

    def actions(self, state: ArrangeState) -> list[dict[str, Any]]:
        # Deduped: `transition` dispatches on cyclers first, so an action that is in both lists
        # has exactly one meaning here -- the cycler one. Listing it twice bought nothing but a
        # second identical successor for every node of every search.
        names = list(self.cyclers) + [a for a in self.shifts if a not in self.cyclers]
        return [{"action": a, "row": None, "col": None} for a in names]

    def cells_at(self, piece: int, pos: tuple[int, int]) -> frozenset:
        return frozenset((pos[0] + r, pos[1] + c) for r, c in self.shapes[piece])

    def _free(self, piece: int, pos: tuple[int, int], spots: tuple) -> bool:
        """A square is free if it is on the board, not a wall, and not already held by another
        piece. Pieces collide: two of them cannot stand in the same place, and a search that let
        them would produce plans the board refuses."""
        mine = self.cells_at(piece, pos)
        for r, c in mine:
            if not (0 <= r < self._h and 0 <= c < self._w):
                return False
            if self.grid[r][c] in self.wall_colours:
                return False
        for other, spot in enumerate(spots):
            if other != piece and self.cells_at(other, spot) & mine:
                return False
        return True

    def transition(self, state: ArrangeState, action: dict[str, Any]) -> ArrangeState:
        name = action["action"]
        if name in self.cyclers:
            return ArrangeState(state.spots, (state.driven + 1) % max(1, len(state.spots)),
                                state.steps + 1)
        dr, dc = self.shifts[name]
        pos = state.spots[state.driven]
        nxt = (pos[0] + dr, pos[1] + dc)
        if not self._free(state.driven, nxt, state.spots):
            return ArrangeState(state.spots, state.driven, state.steps + 1)
        spots = list(state.spots)
        spots[state.driven] = nxt
        return ArrangeState(tuple(spots), state.driven, state.steps + 1)

    def outcome(self, state: ArrangeState) -> str:
        if state.steps > self.max_steps:
            return "game_over"
        if self.mode == "one":
            only = getattr(self, "_only", 0)
            return ("level_complete" if self.cells_at(only, state.spots[only]) & self.targets
                    else "ongoing")
        if self.mode == "cover":
            filled: set = set()
            for piece, pos in enumerate(state.spots):
                filled |= self.cells_at(piece, pos)
            return "level_complete" if self.targets <= filled else "ongoing"
        for piece, pos in enumerate(state.spots):
            if not (self.cells_at(piece, pos) & self.targets):
                return "ongoing"
        return "level_complete"

    def progress(self, state: ArrangeState) -> int:
        """How many target cells are filled, for reporting how close a failed search came."""
        filled: set = set()
        for piece, pos in enumerate(state.spots):
            filled |= self.cells_at(piece, pos)
        return len(self.targets & filled)

    def planner_key(self, state: ArrangeState):
        # The driven piece is part of the state: the same arrangement with a different piece
        # selected is a different position to search from.
        return (state.spots, state.driven)


@dataclass
class ArrangeResult:
    status: str
    actions: list[str] = field(default_factory=list)
    expanded: int = 0
    note: str = ""
    placed: int = 0
    total: int = 0
    # One entry per action: (tid of the piece this step moves, its displacement), or None for a
    # step that only changes which piece is driven. The harness writes the batch's predictions
    # from these, so an arrangement plan is checked exactly as strictly as a walking one: the
    # named piece must move by the named amount, and the batch halts at the first step it does
    # not. A step that claims nothing would be a step nothing could falsify.
    step_claims: list[Optional[tuple[str, tuple[int, int]]]] = field(default_factory=list)


def build(session: Any, pieces: list[str], targets: list[str],
          mode: str = "touch") -> Optional[ArrangeModel]:
    """Assemble the model from certified facts: the pieces' shapes and where they stand, the
    shifts the control map has read, the colours the passability map calls walls, and every action
    that has never moved anything as a candidate for changing which piece is driven."""
    tr = session.tracker
    shapes, spots = [], []
    for tid in pieces[:MAX_PIECES]:
        track = tr.tracks.get(tid)
        if track is None or track.blob is None:
            return None
        r0, c0 = min(r for r, _c in track.blob.cells), min(c for _r, c in track.blob.cells)
        shapes.append(frozenset((r - r0, c - c0) for r, c in track.blob.cells))
        spots.append((r0, c0))
    if not shapes:
        return None

    shifts: dict[str, tuple[int, int]] = {}
    cyclers: list[str] = []
    for a in session.available:
        # ACTION6 is a click and carries coordinates; every step queued from here is sent with
        # x=None, and `int(None)` in the agent ends the whole game run through the vendored
        # runner's exception path. The walking planner already refuses it for the same reason.
        if a in ("RESET", "ACTION6"):
            continue
        known = session.control.known(a)
        # The same ladder `planner.build_model` climbs, in the same order. Two planners that
        # disagree about what a button does plan against two different worlds, and only one of
        # them is the one the step claims are graded against.
        shift = None
        if hasattr(session.rules, "shift_for"):
            shift = session.rules.shift_for(a)
        ctx = session.body_state() if hasattr(session, "body_state") else ""
        if shift is None and hasattr(session.control, "known_in") and session.control.known_in(a, ctx):
            shift = session.control.shift_in(a, ctx)
        if shift is None and known:
            shift = session.control.expected_shift(a)
        if shift is not None and tuple(shift) != (0, 0):
            shifts[a] = tuple(shift)
        elif known or session.control.readable_count(a) >= 2:
            # It has been pressed and never moved anything: the plainest thing a selector can look
            # like from outside.
            cyclers.append(a)
    # And the other thing a selector looks like. re86's ACTION5 recolours one pixel of whichever
    # piece it selects, which the tracker reports as a one-cell move while the real direction keys
    # move three. An action whose step is a different size from the others is not doing the same
    # job as the others, so it is offered as a candidate for changing which piece is driven. It
    # Promoted, it is read as a selector and only as a selector: `transition` dispatches on
    # cyclers first, so the shift it also has is unreachable. Every step still carries a
    # prediction, so a wrong guess halts the batch after one action.
    outliers: list[str] = []
    if shifts:
        sizes = [abs(dr) + abs(dc) for dr, dc in shifts.values()]
        modal = max(set(sizes), key=sizes.count)
        for a, (dr, dc) in shifts.items():
            if abs(dr) + abs(dc) != modal and a not in cyclers:
                outliers.append(a)
    if len(shifts) < 2:
        return None

    from .types import colour_letter
    letters = session.rules.wall_colours() if hasattr(session.rules, "wall_colours") else set()
    walls = frozenset(i for i in range(16) if colour_letter(i) in letters)
    if not walls:
        # The same fallback the walking planner has. The rule store is empty for the first turns
        # of a level, which is exactly when an arrangement goal is first set; with no walls every
        # wall is passable to the search and the first step of the plan walks into one.
        passability = getattr(session, "passability", None)
        if passability is not None:
            walls = frozenset(c for c in passability.colours() if passability.verdict(c) == "WALL")
    spot_cells: set = set()
    for tid in targets:
        track = tr.tracks.get(tid)
        if track is not None and track.blob is not None:
            spot_cells |= set(track.blob.cells)
    if not spot_cells:
        return None
    # The modal-size outliers go first: they are the specific signal ("this action does not do
    # what the others do"), while "pressed and never moved anything" is the weak one. Truncating
    # a list that ended with the outliers dropped the real selector whenever two other actions
    # happened to read as inert.
    cyclers = outliers + [a for a in cyclers if a not in outliers]
    model = ArrangeModel(grid=session.percept.grid, shapes=tuple(shapes), shifts=shifts,
                         cyclers=tuple(cyclers[:2]), wall_colours=walls,
                         targets=frozenset(spot_cells), mode=mode)
    model._start = tuple(spots)   # type: ignore[attr-defined]
    return model


def solve_incremental(session: Any, pieces: list[str], targets: list[str],
                      max_nodes: int = 6000, mode: str = "cover") -> ArrangeResult:
    """Place the pieces one at a time, holding the ones already placed.

    The joint search is exponential in piece count and measured useless past about six: on re86
    with every object taken as a piece, every target class came back `no_plan_within_budget`.

    The decomposition is not an approximation here, which is the point. In a game where the
    controls drive one piece at a time, moving the driven piece never moves any other, so where
    piece B ends up does not depend on the route piece A took to get there. Each placement is its
    own small search over one piece's position and the selection index, and the plans concatenate.
    What the pieces already placed contribute is that they are obstacles, and the next search sees
    them where they now stand.
    """
    from .vendor.tycho.planner import plan_bfs_with_diagnostics
    model = build(session, pieces, targets, mode=mode)
    if model is None:
        return ArrangeResult(status="no_model", note="needs two actions with a known effect")
    spots = list(model._start)        # type: ignore[attr-defined]
    order = list(pieces[:MAX_PIECES])
    remaining = set(model.targets)
    actions: list[str] = []
    claims: list[Optional[tuple[str, tuple[int, int]]]] = []
    driven = 0
    expanded = 0
    placed = 0
    for index in range(len(spots)):
        here = model.cells_at(index, spots[index]) & remaining
        if here:                                   # already standing on somewhere it can stay
            remaining -= here
            placed += 1
            continue
        one = _one_piece_goal(model, index, remaining)
        start = ArrangeState(spots=tuple(spots), driven=driven)
        diag = plan_bfs_with_diagnostics(one, start, max_nodes=max_nodes)
        expanded += diag["expanded_nodes"]
        if diag["status"] != "found":
            continue                               # this one cannot be placed from here; try the rest
        state = start
        for step in diag["plan"]:
            nxt = one.transition(state, step)
            moved = next((i for i in range(len(state.spots)) if state.spots[i] != nxt.spots[i]), None)
            if moved is None:
                claims.append(None)
            else:
                before, after = state.spots[moved], nxt.spots[moved]
                claims.append((order[moved], (after[0] - before[0], after[1] - before[1])))
            actions.append(step["action"])
            state = nxt
        spots = list(state.spots)
        driven = state.driven
        remaining -= model.cells_at(index, spots[index])
        placed += 1
    if not actions:
        return ArrangeResult(status="no_plan_incremental", expanded=expanded, placed=placed,
                             total=len(spots), note="no piece could be placed from where it stands")
    # Replay the concatenated plan through the full model before calling it found. The
    # decomposition assumes moving the driven piece never moves another, and the sub-model that
    # places piece k keeps every cycler, so a later placement can walk an earlier piece off its
    # target while the bookkeeping records both as placed. `solve` validates by replay and this
    # did not, yet this is the path every arrangement goal takes -- and the prompt told the model
    # the plan was "validated by replay" either way. The condition checked is the one the goal
    # algebra states: every piece standing on some target, sharing allowed.
    state = ArrangeState(spots=tuple(model._start), driven=0)   # type: ignore[attr-defined]
    for name in actions:
        state = model.transition(state, {"action": name, "row": None, "col": None})
    held = sum(1 for piece, pos in enumerate(state.spots) if model.cells_at(piece, pos) & model.targets)
    if held < placed:
        placed = held
    if placed < len(spots) or len(order) < len(pieces):
        # `found` is a promise that the goal will hold when the plan ends, and a run that placed
        # one piece of six does not make it. Returned as found, it committed up to twenty actions
        # on a plan that provably could not end the level, and the goal was retired two turns
        # later for failing a promise it should never have made. `pieces` longer than MAX_PIECES
        # is the same shortfall from the other end: the search never saw the rest.
        return ArrangeResult(status="partial", actions=actions, expanded=expanded, placed=placed,
                             total=len(pieces), step_claims=claims,
                             note=f"{placed} of {len(pieces)} pieces placed; the rest could not be "
                                  f"placed from where they stand")
    return ArrangeResult(status="found", actions=actions, expanded=expanded, placed=placed,
                         total=len(spots), step_claims=claims,
                         note=f"{placed} of {len(spots)} pieces placed, one at a time")


def _one_piece_goal(model: ArrangeModel, index: int, targets: frozenset) -> ArrangeModel:
    """The same board, asking only that piece `index` comes to rest on one of `targets`."""
    import copy
    sub = copy.copy(model)
    sub.targets = frozenset(targets)
    sub.mode = "one"
    sub._only = index                              # type: ignore[attr-defined]
    return sub


def solve(session: Any, pieces: list[str], targets: list[str],
          max_nodes: int = MAX_NODES, mode: str = "touch") -> ArrangeResult:
    """Search for the actions that satisfy the targets: every piece on one, or every one filled."""
    from .vendor.tycho.planner import plan_bfs_with_diagnostics, validate_plan
    model = build(session, pieces, targets, mode=mode)
    if model is None:
        return ArrangeResult(status="no_model",
                             note="needs two actions with a known effect, pieces the tracker can "
                                  "see, and somewhere for them to go")
    start = ArrangeState(spots=model._start, driven=0)   # type: ignore[attr-defined]
    total = len(model.shapes)
    placed = (model.progress(start) if mode == "cover"
              else sum(1 for i, p in enumerate(start.spots) if model.cells_at(i, p) & model.targets))
    total = len(model.targets) if mode == "cover" else total
    if model.outcome(start) == "level_complete":
        return ArrangeResult(status="already", placed=placed, total=total,
                             note="every piece is already on a target and the level did not end")
    diag = plan_bfs_with_diagnostics(model, start, max_nodes=max_nodes)
    if diag["status"] != "found":
        return ArrangeResult(status=diag["status"], expanded=diag["expanded_nodes"],
                             placed=placed, total=total,
                             note="; ".join(diag.get("errors", []))[:160])
    check = validate_plan(model, start, diag["plan"])
    if check["status"] != "found":
        return ArrangeResult(status="invalid", expanded=diag["expanded_nodes"],
                             placed=placed, total=total, note=check["status"])
    names = [a["action"] for a in diag["plan"]]
    claims: list[Optional[tuple[str, tuple[int, int]]]] = []
    state = start
    order = pieces[:MAX_PIECES]
    for name in names:
        nxt = model.transition(state, {"action": name, "row": None, "col": None})
        moved = next((i for i in range(len(state.spots)) if state.spots[i] != nxt.spots[i]), None)
        if moved is None:
            claims.append(None)
        else:
            before, after = state.spots[moved], nxt.spots[moved]
            claims.append((order[moved], (after[0] - before[0], after[1] - before[1])))
        state = nxt
    return ArrangeResult(status="found", actions=names, expanded=diag["expanded_nodes"],
                         placed=placed, total=total, step_claims=claims)
