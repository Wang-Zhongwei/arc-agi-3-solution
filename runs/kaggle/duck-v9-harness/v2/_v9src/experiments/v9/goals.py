"""Layer 3: a symbolic goal the model predicts and Python solves.

    [raw grid] -> (parser) -> [symbols] -> (small model) -> [symbolic goal] -> (solver) -> [action]

The model is never asked to plan. It is asked one bounded question -- *what would winning look
like, written in this algebra* -- and the harness searches for the actions that make the
expression true, then validates the plan by canonical replay before spending a single move.

The algebra is closed and tiny, because an 8B model fills forms well and writes code badly:

    {"all_touched": "class:Y"}          every yellow thing has been reached
    {"touched": "obj:t7"}               that particular thing has been reached
    {"at": ["$self", [12, 30]]}         the body is standing there
    {"adjacent": ["$self", "obj:t7"]}   the body is next to it
    {"count": ["class:Y", "<=", 0]}     no yellow things left unreached
    {"covered": ["class:Y", "class:B"]} every yellow thing is standing on a blue one
    {"and": [ ... ]}  {"or": [ ... ]}   at most two levels deep

Every term is evaluated against the harness's own executable model, so a goal that cannot be
expressed is rejected at compile time rather than believed and acted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

COMPARISONS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
               ">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "!=": lambda a, b: a != b}
PREDICATES = ("all_touched", "touched", "at", "adjacent", "count", "covered", "and", "or")
MAX_DEPTH = 2


class GoalReject(str):
    """A rejection carrying the reason, so the model is told what to fix."""


@dataclass(frozen=True)
class GoalWorld:
    """What a goal may ask about: the body's shape, what each thing is, and where it is."""
    self_cells: frozenset[tuple[int, int]]        # relative to the body's top-left
    classes: dict[str, str]                       # tid -> class key, e.g. "Y dot"
    cells: dict[str, frozenset[tuple[int, int]]]  # tid -> absolute cells

    def members_of_goal(self, js: Any) -> list[str]:
        """Every tid a goal expression names, so telemetry can say how many of them were reached."""
        out: list[str] = []
        stack = [js]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict) or len(node) != 1:
                continue
            (name, arg), = node.items()
            if name in ("and", "or") and isinstance(arg, list):
                stack.extend(arg)
                continue
            term = arg
            if name == "adjacent" and isinstance(arg, list) and len(arg) == 2:
                term = arg[1]
            elif name == "count" and isinstance(arg, list) and arg:
                term = arg[0]
            if isinstance(term, str):
                out.extend(self.members(term))
        return sorted(set(out))

    def members(self, term: str) -> list[str]:
        """Resolve `obj:<tid>` or `class:<key>` to the tids it names."""
        if term.startswith("obj:"):
            tid = term[4:]
            return [tid] if tid in self.classes else []
        if term.startswith("class:"):
            key = term[6:]
            return sorted(tid for tid, k in self.classes.items()
                          if k == key or k.split(" ")[0] == key)
        return []


CompiledGoal = Callable[[tuple[int, int], frozenset], bool]


def compile_goal(js: Any, world: GoalWorld, *, depth: int = 0) -> "CompiledGoal | GoalReject":
    """Turn the model's JSON goal into a predicate over (body position, touched set)."""
    if depth > MAX_DEPTH:
        return GoalReject(f"goal nested deeper than {MAX_DEPTH}")
    if not isinstance(js, dict) or len(js) != 1:
        return GoalReject("each goal term is one object with exactly one key")
    (name, arg), = js.items()
    if name not in PREDICATES:
        return GoalReject(f"unknown goal predicate {name!r}; use one of {list(PREDICATES)}")

    if name in ("and", "or"):
        if not isinstance(arg, list) or not arg:
            return GoalReject(f"{name} takes a non-empty list of terms")
        parts = []
        for item in arg:
            sub = compile_goal(item, world, depth=depth + 1)
            if isinstance(sub, GoalReject):
                return sub
            parts.append(sub)
        if name == "and":
            return lambda pos, touched, _p=parts: all(p(pos, touched) for p in _p)
        return lambda pos, touched, _p=parts: any(p(pos, touched) for p in _p)

    if name == "all_touched":
        members = world.members(str(arg))
        if not members:
            return GoalReject(f"all_touched names {arg!r}, which matches nothing on the board")
        return lambda pos, touched, _m=frozenset(members): _m <= touched

    if name == "touched":
        members = world.members(str(arg))
        if len(members) != 1:
            return GoalReject(f"touched needs exactly one object; {arg!r} matches {len(members)}")
        return lambda pos, touched, _t=members[0]: _t in touched

    if name == "at":
        if not (isinstance(arg, list) and len(arg) == 2 and str(arg[0]) == "$self"
                and isinstance(arg[1], (list, tuple)) and len(arg[1]) == 2):
            return GoalReject('at takes ["$self", [row, col]]')
        return lambda pos, touched, _t=(int(arg[1][0]), int(arg[1][1])): pos == _t

    if name == "adjacent":
        if not (isinstance(arg, list) and len(arg) == 2 and str(arg[0]) == "$self"):
            return GoalReject('adjacent takes ["$self", "obj:<tid>" or "class:<X>"]')
        members = world.members(str(arg[1]))
        if not members:
            return GoalReject(f"adjacent names {arg[1]!r}, which matches nothing on the board")
        wanted: frozenset = frozenset().union(*[world.cells.get(t, frozenset()) for t in members])
        near = frozenset((r + dr, c + dc) for r, c in wanted
                         for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)))
        return lambda pos, touched, _n=near, _b=world.self_cells: any(
            (pos[0] + r, pos[1] + c) in _n for r, c in _b)

    if name == "covered":
        # The condition eight of twenty-four public games actually end on, and the first one here
        # that is not about the body at all: every piece of one kind has to be sitting on a piece
        # of another. `touched` and `all_touched` ask what the body reached; this asks where
        # things ended up, which is what these games are.
        if not (isinstance(arg, list) and len(arg) == 2):
            return GoalReject('covered takes ["class:<X>", "class:<Y>"]: every X on some Y')
        pieces = world.members(str(arg[0]))
        targets = world.members(str(arg[1]))
        if not pieces:
            return GoalReject(f"covered names {arg[0]!r}, which matches nothing on the board")
        if not targets:
            return GoalReject(f"covered names {arg[1]!r}, which matches nothing on the board")
        spots: frozenset = frozenset().union(*[world.cells.get(t, frozenset()) for t in targets])
        # Evaluated against where the pieces are now, because the body's position says nothing
        # about it. A goal the body cannot affect by walking is still a goal worth stating; the
        # harness reports whether it holds and the planner refuses what it cannot solve.
        need = {p: frozenset(world.cells.get(p, frozenset())) for p in pieces}
        return lambda pos, touched, _n=need, _s=spots: all(c & _s for c in _n.values())

    if name == "count":
        if not (isinstance(arg, list) and len(arg) == 3 and arg[1] in COMPARISONS):
            return GoalReject('count takes ["class:<X>", "<=", n] with a comparison from '
                              f"{sorted(COMPARISONS)}")
        members = world.members(str(arg[0]))
        if not members:
            return GoalReject(f"count names {arg[0]!r}, which matches nothing on the board")
        try:
            k = int(arg[2])
        except (TypeError, ValueError):
            return GoalReject("count's third element must be a whole number")
        return lambda pos, touched, _m=frozenset(members), _op=COMPARISONS[arg[1]], _k=k: _op(
            len(_m - touched), _k)

    return GoalReject(f"unhandled predicate {name!r}")


def describe(js: Any) -> str:
    """The goal in words, for the prompt and the log."""
    if not isinstance(js, dict) or len(js) != 1:
        return str(js)
    (name, arg), = js.items()
    if name == "and":
        return " and ".join(describe(a) for a in arg)
    if name == "or":
        return " or ".join(describe(a) for a in arg)
    if name == "all_touched":
        return f"reach every {arg}"
    if name == "touched":
        return f"reach {arg}"
    if name == "at":
        return f"stand at ({arg[1][0]},{arg[1][1]})"
    if name == "adjacent":
        return f"stand next to {arg[1]}"
    if name == "covered":
        return f"put every {arg[0]} on a {arg[1]}"
    if name == "count":
        return f"leave {arg[1]} {arg[2]} of {arg[0]} unreached"
    return str(js)


GOAL_HINT = (
    'Goal algebra: {"all_touched":"class:Y"} reach every yellow thing; {"touched":"obj:t7"}; '
    '{"at":["$self",[12,30]]}; {"adjacent":["$self","obj:t7"]}; {"count":["class:Y","<=",0]}; '
    'combine with {"and":[...]} or {"or":[...]}, at most two levels deep. Name only classes and '
    "objects that appear in the symbol table."
)
