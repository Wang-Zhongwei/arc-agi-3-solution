"""The solver at the end of the symbolic pipeline.

    [symbols] -> (model predicts a symbolic goal) -> THIS -> [actions]

A small executable world model is built from certified facts only -- the controlled body's shape,
the shifts each action is known to produce, and the colours the passability map calls walls --
and searched with the vendored Tycho planner (plan_bfs / validate_plan). The goal is whatever
expression `v9.goals` compiled, so the harness never assumes what winning means; it solves for
what the model said winning means, and refuses the plan if canonical replay does not reach it.
Zero model requests.

The model here is deliberately narrow: it plans over the controlled object's position and the
set of remaining "collectable" tracks, using the control map's certified shifts and the
passability verdicts (WALL colours block, floor colours pass). It only claims a plan when the
goal is one the harness can state deterministically ("every collectable consumed" or "self
overlaps a labelled goal object"), and a plan is offered to the model as a suggestion the
DecideSchema can commit; the model may still choose otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import colour_letter
from .vendor.tycho.planner import plan_bfs_with_diagnostics, validate_plan

DIRS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


@dataclass(frozen=True)
class PlanState:
    pos: tuple[int, int]            # top-left of the body's bounding box
    touched: frozenset[str]         # tids the body has reached (a goal is written over these)
    steps: int = 0


@dataclass
class GridModel:
    """`actions/transition/outcome` over PlanState, honouring the Tycho planner contract."""
    grid: list[list[int]]
    self_cells: frozenset[tuple[int, int]]   # relative to pos
    shifts: dict[str, tuple[int, int]]
    wall_colours: frozenset[int]
    targets: dict[str, frozenset[tuple[int, int]]]   # tid -> cells of everything reachable
    # Actions whose travel is "keep going in this direction until something stops you". The map
    # holds the unit direction; the distance comes out of the simulation, which is the only place
    # it can come from, since it depends on where the body is standing.
    slides: dict[str, tuple[int, int]] = field(default_factory=dict)
    goal: Optional[Any] = None               # compiled predicate (pos, touched) -> bool
    max_steps: int = 400
    TRANSITION_RETURNS_NEW_STATE = True

    def __post_init__(self) -> None:
        # One cell -> tid index, built once. Checking "what did the body just cover" against every
        # target's cell set, for every cell of the body, for every step of a slide, for every node
        # of the search, is what made the first turn of a run hang for sixteen minutes without a
        # single model request: a slide is up to 64 steps, so it multiplied the cost of the whole
        # search by the width of the board.
        self._at: dict[tuple[int, int], list[str]] = {}
        for tid, cells in self.targets.items():
            for cell in cells:
                self._at.setdefault(cell, []).append(tid)
        self._free_cache: dict[tuple[int, int], bool] = {}

    def actions(self, state: PlanState) -> list[dict[str, Any]]:
        return [{"action": a, "row": None, "col": None} for a in self.shifts]

    def _cells_at(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        return [(pos[0] + r, pos[1] + c) for r, c in self.self_cells]

    def _free(self, pos: tuple[int, int]) -> bool:
        hit = self._free_cache.get(pos)
        if hit is not None:
            return hit
        h, w = len(self.grid), len(self.grid[0])
        ok = True
        for r, c in self._cells_at(pos):
            if not (0 <= r < h and 0 <= c < w) or self.grid[r][c] in self.wall_colours:
                ok = False
                break
        self._free_cache[pos] = ok
        return ok

    def transition(self, state: PlanState, action: dict[str, Any]) -> PlanState:
        """One move. Blocked by a wall or the border leaves the body where it was, which is what
        the passability map actually measured; anything the body passes over counts as reached.

        A slide keeps going in its direction until the next cell is not free, so the distance is a
        consequence of the board rather than a property of the button. Everything swept on the way
        is touched: a body that slides across a collectable has reached it.
        """
        name = action["action"]
        unit = self.slides.get(name)
        step = unit if unit is not None else self.shifts[name]
        pos = state.pos
        touched = set(state.touched)
        limit = max(len(self.grid), len(self.grid[0])) + 1
        at = self._at
        for _ in range(limit if unit is not None else 1):
            nxt = (pos[0] + step[0], pos[1] + step[1])
            if not self._free(nxt):
                break
            pos = nxt
            if at:
                for cell in self._cells_at(pos):
                    here = at.get(cell)
                    if here:
                        touched.update(here)
        return PlanState(pos, frozenset(touched), state.steps + 1)

    def outcome(self, state: PlanState) -> str:
        if state.steps > self.max_steps:
            return "game_over"
        if self.goal is not None and self.goal(state.pos, state.touched):
            return "level_complete"
        return "ongoing"

    def planner_key(self, state: PlanState):
        return (state.pos, state.touched)


@dataclass
class PlanResult:
    status: str
    actions: list[str] = field(default_factory=list)
    expanded: int = 0
    note: str = ""
    goal_text: str = ""
    # The exact shift the search assumed for each action. The harness predicts the plan with
    # THESE, not with a second lookup: when the two sources disagreed (the rule store said one
    # thing, the control map another, and the control map said nothing at all for an action whose
    # dominant reading was "other"), a plan step was committed carrying an empty prediction, so
    # nothing could falsify it and the same one-step plan was re-solved for eight turns running.
    shifts: dict[str, tuple[int, int]] = field(default_factory=dict)
    # What the simulation says the body's top-left does on each step of `actions`. For a slide
    # this is the only honest prediction: the distance depends on where the body is standing, so
    # no per-action number could state it.
    step_shifts: list[tuple[int, int]] = field(default_factory=list)
    # For an arrangement plan: per step, which piece moves and by how much, or None for a step
    # that only changes which piece is driven.
    piece_claims: list[Any] = field(default_factory=list)


def _body(tracker: Any) -> frozenset:
    """Every cell of the controlled body, not just the track that happens to carry its name."""
    # The stable body, not the maximal one: a plan built around parts that are not there when it
    # arrives runs exactly as predicted and reaches nothing.
    cells = tracker.stable_body_cells() if hasattr(tracker, "stable_body_cells") else frozenset()
    if not cells and hasattr(tracker, "body_cells"):
        cells = tracker.body_cells()
    if cells:
        return cells
    track = tracker.tracks.get(tracker.self_tid) if tracker.self_tid else None
    return frozenset(track.blob.cells) if track is not None and track.blob is not None else frozenset()


def body_origin(tracker: Any) -> Optional[tuple[int, int]]:
    """The one origin a goal is ever evaluated at: the top-left of the WHOLE body.

    `plan()` seeds its search from here and `goal_world` writes `self_cells` relative to here, so
    anything that grades a goal has to use the same point. Reading the top-left of the single
    track that carries the body's name instead is wrong by however far the other fragments reach:
    on a five-component body the promise check then evaluated a met goal at the wrong square,
    counted it unmet, and retired a correct goal after two turns of it.
    """
    body = _body(tracker)
    if not body:
        return None
    return (min(r for r, _c in body), min(c for _r, c in body))


def goal_world(session: Any):
    """The symbol world a goal is written over: the body's shape and every other live thing."""
    from .goals import GoalWorld
    from .symbols import shape_name
    tr = session.tracker
    self_tid = tr.self_tid
    if not self_tid or self_tid not in tr.tracks or tr.tracks[self_tid].blob is None:
        return None
    body = _body(tr)
    origin = body_origin(tr)
    if origin is None:
        return None
    r0, c0 = origin
    group = set(tr.self_group())
    classes: dict[str, str] = {}
    cells: dict[str, frozenset] = {}
    for t in tr.live():
        if t.tid in group or t.blob is None:
            continue
        classes[t.tid] = f"{colour_letter(t.colour)} {shape_name(t.blob.cells, t.last_bbox)}"
        cells[t.tid] = frozenset(t.blob.cells)
    return GoalWorld(self_cells=frozenset((r - r0, c - c0) for r, c in body),
                     classes=classes, cells=cells)


def build_model(session: Any, goal=None) -> Optional[GridModel]:
    """Assemble a GridModel from the session's certified facts, or None with a reason."""
    tr = session.tracker
    self_tid = tr.self_tid
    if not self_tid or self_tid not in tr.tracks or tr.tracks[self_tid].blob is None:
        return None
    # The rule store is the stricter source: a shift it reports has survived scoring against the
    # whole ledger. It is empty early on, so the control map remains the fallback.
    shifts: dict[str, tuple[int, int]] = {}
    for a in session.available:
        if a not in DIRS:
            continue
        shift = None
        if hasattr(session.rules, "shift_for"):
            shift = session.rules.shift_for(a)
        # Ask what the button does in the state the body is in now, not on average over every
        # state it has been in.
        ctx = session.body_state() if hasattr(session, "body_state") else ""
        if shift is None and hasattr(session.control, "known_in") and session.control.known_in(a, ctx):
            shift = session.control.shift_in(a, ctx)
        if shift is None and session.control.known(a):
            shift = session.control.expected_shift(a)
        if shift is not None and tuple(shift) != (0, 0):
            shifts[a] = tuple(shift)
    slides: dict[str, tuple[int, int]] = {}
    for a in session.available:
        if a not in DIRS or not session.control.known(a):
            continue
        unit = session.control.slide_direction(a)
        if unit is not None:
            slides[a] = unit
            shifts.setdefault(a, unit)      # so the action is offered to the search at all
    if len(shifts) < 2:
        return None
    letters = session.rules.wall_colours() if hasattr(session.rules, "wall_colours") else set()
    walls = frozenset(i for i in range(16) if colour_letter(i) in letters)
    if not walls:
        walls = frozenset(c for c in session.passability.colours()
                          if session.passability.verdict(c) == "WALL")
    body = _body(tr)
    if not body:
        return None
    r0 = min(r for r, _c in body)
    c0 = min(c for _r, c in body)
    self_cells = frozenset((r - r0, c - c0) for r, c in body)
    group = set(tr.self_group())
    targets = {t.tid: frozenset(t.blob.cells) for t in tr.live()
               if t.tid not in group and t.blob is not None}
    if goal is None:
        return None
    return GridModel(grid=session.percept.grid, self_cells=self_cells, shifts=shifts,
                     wall_colours=walls, targets=targets, goal=goal, slides=slides)


def plan(session: Any, max_nodes: int = 8000) -> PlanResult:
    """Solve for the symbolic goal the model most recently set, if the facts allow it."""
    from .goals import compile_goal, describe, GoalReject
    goal_js = getattr(session, "symbolic_goal", None)
    if not goal_js:
        return PlanResult(status="no_goal", note="no symbolic goal has been set yet")
    world = goal_world(session)
    if world is None:
        return PlanResult(status="no_model", note="the controlled body is not identified yet")
    compiled = compile_goal(goal_js, world)
    if isinstance(compiled, GoalReject):
        return PlanResult(status="bad_goal", note=str(compiled), goal_text=describe(goal_js))
    if isinstance(goal_js, dict) and "covered" in goal_js:
        # The one goal shape that is not about the body. It gets its own search, over where the
        # pieces are and which of them the controls drive.
        from . import arrange
        arg = goal_js["covered"]
        pieces = world.members(str(arg[0])) if isinstance(arg, list) and len(arg) == 2 else []
        spots = world.members(str(arg[1])) if isinstance(arg, list) and len(arg) == 2 else []
        if pieces and spots:
            # One piece at a time, not a joint search over all of them. The joint form is
            # exponential in piece count and returned "no plan within budget" for every class on
            # re86; the incremental form found plans of 22 and 5 actions on the same board, and it
            # is exact here rather than approximate because moving the driven piece never moves
            # another.
            # The caller's node budget reaches both searches. Neither took it before:
            # `solve_incremental` kept its own default per piece (six of them) and the joint
            # fallback kept 30,000, so a call the caller thought it had capped at 8,000 could
            # expand 66,000 nodes.
            per_piece = max(500, int(max_nodes) // max(1, min(len(pieces), arrange.MAX_PIECES)))
            # "touch", not "cover": `covered` in the goal algebra is every piece standing on
            # SOME target, sharing allowed (goals.py), while "cover" asks that every target cell
            # be filled. The incremental search was asking one question and the joint fallback
            # beside it the other, and neither was the one the goal states.
            got = arrange.solve_incremental(session, pieces, spots, mode="touch",
                                            max_nodes=per_piece)
            if got.status != "found":
                got = arrange.solve(session, pieces, spots, max_nodes=int(max_nodes))
            if got.status in ("found", "partial"):
                # A partial arrangement is still worth a turn -- the pieces it does place stay
                # placed -- but it is labelled partial, so nothing downstream reads it as a
                # promise that the goal will hold when the batch ends.
                return PlanResult(status=got.status, actions=got.actions, expanded=got.expanded,
                                  goal_text=describe(goal_js),
                                  note=f"{got.placed} of {got.total} pieces onto their places",
                                  step_shifts=[], shifts={},
                                  piece_claims=list(got.step_claims))
            if got.status == "already":
                return PlanResult(status="goal_reached", goal_text=describe(goal_js), note=got.note)
            return PlanResult(status=got.status, expanded=got.expanded,
                              goal_text=describe(goal_js),
                              note=f"{got.placed} of {got.total} pieces are already placed; "
                                   f"{got.note}"[:200])
        # The harness can state this goal and check it; it cannot yet solve it. The search moves
        # one body over a fixed board, and this asks where OTHER things end up, which needs a model
        # in which several objects move. Saying so is worth more than quietly planning a walk that
        # cannot achieve it, which is what every other goal shape did for a day.
        me = _body(tr) if (tr := session.tracker) else frozenset()
        held = compiled((min(r for r, _c in me) if me else 0,
                         min(c for _r, c in me) if me else 0), frozenset())
        return PlanResult(status="goal_reached" if held else "unsolvable_here",
                          goal_text=describe(goal_js),
                          note=("that is already true and the level did not end, so it was not the "
                                "condition" if held else
                                "the solver moves your body over a fixed board; this asks where "
                                "other things end up, which it cannot plan yet. It is still worth "
                                "stating: the harness will tell you when it becomes true."))
    model = build_model(session, goal=compiled)
    if model is None:
        return PlanResult(status="no_model", note="two actions with a known effect are required",
                          goal_text=describe(goal_js))
    tr = session.tracker
    body = _body(tr)
    if not body:
        return PlanResult(status="no_model", note="the controlled body is not identified yet",
                          goal_text=describe(goal_js))
    # Seed with what the body has actually reached, so a collection goal advances instead of
    # re-solving its first step forever.
    reached = getattr(session, "reached", None)
    already = frozenset(reached({t: model.targets[t] for t in model.targets})) if reached \
        else frozenset(getattr(session, "touched_tids", set()) & set(model.targets))
    start = PlanState(pos=(min(r for r, _c in body), min(c for _r, c in body)), touched=already)
    if compiled(start.pos, start.touched):
        return PlanResult(status="goal_reached", expanded=0, goal_text=describe(goal_js),
                          note="everything this goal asks for has already been reached, and the "
                               "level did not end: the goal was wrong, set another")
    text = describe(goal_js)
    diag = plan_bfs_with_diagnostics(model, start, max_nodes=max_nodes)
    if diag["status"] != "found":
        # No full solution. Getting closer is still worth a turn, and it is the difference
        # between committing twelve actions and committing two: on ls20 level 2 the solver kept
        # answering "no_plan_exhausted" while the agent spent whole eight-minute turns on single
        # probes. A partial plan is labelled as partial and never claims to reach the goal.
        partial = _closest_approach(model, start, goal_js, world, max_nodes=min(max_nodes, 4000))
        if partial is not None:
            actions, gained = partial
            return PlanResult(status="partial", actions=actions, expanded=diag["expanded_nodes"],
                              note=f"no full solution ({diag['status']}); this gets {gained} closer",
                              goal_text=text, shifts=dict(model.shifts),
                              step_shifts=_simulate(model, start, actions))
        return PlanResult(status=diag["status"], expanded=diag["expanded_nodes"],
                          note="; ".join(diag["errors"])[:200], goal_text=text)
    check = validate_plan(model, start, diag["plan"])
    if check["status"] != "found":
        return PlanResult(status="invalid", expanded=diag["expanded_nodes"], note=check["status"],
                          goal_text=text)
    found = [a["action"] for a in diag["plan"]]
    return PlanResult(status="found", actions=found, expanded=diag["expanded_nodes"], goal_text=text,
                      shifts=dict(model.shifts), step_shifts=_simulate(model, start, found))


def _simulate(model: GridModel, start: PlanState, actions: list[str]) -> list[tuple[int, int]]:
    """Replay a plan through the model and report the displacement each step produces."""
    out, state = [], start
    for name in actions:
        nxt = model.transition(state, {"action": name, "row": None, "col": None})
        out.append((nxt.pos[0] - state.pos[0], nxt.pos[1] - state.pos[1]))
        state = nxt
    return out


def explore(session: Any, *, visited: frozenset = frozenset(), max_steps: int = 20,
            max_nodes: int = 4000) -> PlanResult:
    """A route that walks the body over ground it has not stood on, using only certified facts.

    When no goal survives, the harness still has to spend the turn, and a one-action probe is the
    worst way to spend it: on ls20 the batch collapsed to one or two actions after the first
    surprise and stayed there, so a turn bought a turn's worth of evidence instead of a batch's.

    This costs no model request. It plans over the same executable model the goal solver uses, so
    every step carries a checkable prediction and the batch still halts itself the moment the
    world model is wrong -- which on unfamiliar ground is the point, not a risk.
    """
    world = goal_world(session)
    if world is None:
        return PlanResult(status="no_model", note="the controlled body is not identified yet")
    model = build_model(session, goal=lambda _pos, _touched: False)
    if model is None:
        return PlanResult(status="no_model", note="two actions with a known effect are required")
    body = _body(session.tracker)
    start = PlanState(pos=(min(r for r, _c in body), min(c for _r, c in body)), touched=frozenset())

    from collections import deque
    seen = {start.pos}
    best_path: list[str] = []
    best_gain = 0
    queue = deque([(start, [], frozenset({start.pos}))])
    expanded = 0
    while queue and expanded < max_nodes:
        state, path, stood = queue.popleft()
        expanded += 1
        if len(path) >= max_steps:
            continue
        for action in model.actions(state):
            try:
                nxt = model.transition(state, action)
            except Exception:  # noqa: BLE001
                continue
            if nxt.pos == state.pos:
                continue                      # blocked: the route learns nothing by standing still
            if nxt.pos in seen:
                continue
            seen.add(nxt.pos)
            route = path + [action["action"]]
            fresh = stood | {nxt.pos}
            gain = len(fresh - visited)
            if gain > best_gain or (gain == best_gain and len(route) < len(best_path)):
                best_gain, best_path = gain, route
            queue.append((nxt, route, fresh))
    if not best_path:
        return PlanResult(status="nowhere_to_go",
                          note="every direction from here is blocked on the map the harness has")
    actions = best_path[:max_steps]
    return PlanResult(status="explore", actions=actions, expanded=expanded,
                      goal_text="stand on ground you have not stood on",
                      note=f"{best_gain} squares the body has never occupied",
                      shifts=dict(model.shifts), step_shifts=_simulate(model, start, actions))


def _goal_cells(goal_js: Any, world) -> frozenset:
    """The cells the goal is about, so "closer" has a meaning even when the goal is unreachable."""
    if not isinstance(goal_js, dict) or len(goal_js) != 1:
        return frozenset()
    (name, arg), = goal_js.items()
    if name == "at" and isinstance(arg, list) and len(arg) == 2:
        return frozenset({(int(arg[1][0]), int(arg[1][1]))})
    term = arg[1] if name == "adjacent" and isinstance(arg, list) and len(arg) == 2 else arg
    if name == "count" and isinstance(arg, list):
        term = arg[0]
    members = world.members(str(term)) if hasattr(world, "members") else []
    cells: set = set()
    for tid in members:
        cells |= set(world.cells.get(tid, frozenset()))
    return frozenset(cells)


def _closest_approach(model: GridModel, start: PlanState, goal_js: Any, world,
                      *, max_nodes: int = 20000) -> Optional[tuple[list[str], int]]:
    """Breadth-first over the same certified model, keeping the state that gets nearest the goal.

    Nearness is the smallest Chebyshev gap between the body's footprint and the goal's cells, with
    the number of goal members already reached taking precedence. Returns None when no reachable
    state improves on standing still, so a turn is never spent going nowhere.
    """
    from collections import deque
    targets = _goal_cells(goal_js, world)
    if not targets:
        return None
    # One multi-source pass over the board instead of |body| x |targets| arithmetic per node.
    # Chebyshev distance IS 8-connected BFS distance, so the field is exact, and it turns the
    # per-node cost from ~2,100 operations (a 53-cell body against a 40-cell target class) into
    # 53 lookups. This runs only after the 8,000-node search has already failed, so it was the
    # slow path behind the slow path -- the same shape as the hang that cost sixteen minutes.
    INF = 10 ** 6
    height = len(model.grid)
    width = len(model.grid[0]) if height else 0
    field = [[INF] * width for _ in range(height)]
    ring: Any = deque()
    for r, c in targets:
        if 0 <= r < height and 0 <= c < width and field[r][c]:
            field[r][c] = 0
            ring.append((r, c))
    while ring:
        r, c = ring.popleft()
        step = field[r][c] + 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and field[rr][cc] > step:
                    field[rr][cc] = step
                    ring.append((rr, cc))
    target_tids = set(model.targets)

    def score(state: PlanState) -> tuple[int, int]:
        reached = len(state.touched & target_tids)
        gap = min((field[r][c] for r, c in model._cells_at(state.pos)
                   if 0 <= r < height and 0 <= c < width), default=INF)
        return (-reached, gap)

    best_score = score(start)
    best_path: list[str] = []
    seen = {model.planner_key(start)}
    queue = deque([(start, [])])
    expanded = 0
    while queue and expanded < max_nodes:
        state, path = queue.popleft()
        expanded += 1
        for action in model.actions(state):
            try:
                nxt = model.transition(state, action)
            except Exception:  # noqa: BLE001
                continue
            key = model.planner_key(nxt)
            if key in seen:
                continue
            seen.add(key)
            here = score(nxt)
            if here < best_score:
                best_score, best_path = here, path + [action["action"]]
            queue.append((nxt, path + [action["action"]]))
    if not best_path:
        return None
    start_score = score(start)
    gained = (start_score[1] - best_score[1]) or (start_score[0] - best_score[0])
    return best_path[:MAX_PARTIAL], int(gained)


MAX_PARTIAL = 12


def render_plan(result: PlanResult, max_show: int = 12) -> str:
    if result.status == "unsolvable_here":
        return f'your goal "{result.goal_text}" is understood but cannot be planned: {result.note}'
    if result.status == "explore":
        seq = " ".join(result.actions[:max_show])
        return (f"no goal survives, so the harness plans for evidence instead: {len(result.actions)} "
                f"steps over {result.note}, each one predicted and validated by replay: {seq}")
    if result.status == "goal_reached":
        return f'your goal "{result.goal_text}" is already met and the level did not end: {result.note}'
    if result.status == "partial":
        seq = " ".join(result.actions[:max_show])
        return (f'harness could not solve "{result.goal_text}" in full, but these {len(result.actions)} '
                f"steps get closer and were validated by replay: {seq}")
    if result.status != "found":
        goal = f' for "{result.goal_text}"' if result.goal_text else ""
        return f"harness plan{goal}: none ({result.status}{': ' + result.note if result.note else ''})"
    seq = result.actions
    shown = " ".join(seq[:max_show]) + (f" ... (+{len(seq) - max_show})" if len(seq) > max_show else "")
    return (f'harness solved your goal "{result.goal_text}" in {len(seq)} steps '
            f"(validated by replay): {shown}")
