"""Count which frame_diff keys and entry fields python tool calls read.

usage: python -I frame_diff_fields.py TRANSCRIPT...

Entry fields share names with segmentation node fields (color, pixels, hash,
at, bbox), so a regex cannot tell them apart. Instead the code is parsed and
values are traced from `frame_diff(...)` to a category list (`d['moved']`) to
an entry (`for m in d['moved']`), and only reads on traced values count.

Per call it records:
- explicit reads: `d['moved']`, `m['from']`, `d.get(k)` with `k` looping over
  a literal tuple of keys, ...
- whole views: a traced value printed or stored whole (`print(d)`,
  `print(m)`, `for k, v in d.items(): print(k, v)`), which shows every field
  without the code naming one.
"""
import ast
import json
import re
import sys
from collections import Counter

TOP = ["changed_cell_count", "moved", "rotated", "appeared", "disappeared",
       "changed_color", "resized"]
CATS = TOP[1:]
FIELDS = {
    "moved": ["color", "pixels", "hash", "from", "to", "rotated_by"],
    "rotated": ["color", "pixels", "hash", "at", "rotated_by"],
    "appeared": ["color", "pixels", "hash", "shape_hash", "at", "bbox"],
    "disappeared": ["color", "pixels", "hash", "shape_hash", "at", "bbox"],
    "changed_color": ["at", "pixels", "before_color", "after_color",
                      "before_hash", "after_hash"],
    "resized": ["at_before", "at_after", "color", "before_pixels",
                "after_pixels", "before_hash", "after_hash"],
}
ALL = frozenset(CATS)
ARG = re.compile(r'^\s*"arguments":\s*(".*")\s*,?\s*$')
FD = re.compile(r"\bframe_diff\s*\(")
# Calls whose argument is only inspected, not shown.
QUIET = {"len", "bool", "isinstance", "id", "type", "any", "all", "sum",
         "enumerate", "sorted", "list", "zip", "reversed"}


def codes(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        m = ARG.match(line)
        if not m:
            continue
        try:
            args = json.loads(json.loads(m.group(1)))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(args, dict) and isinstance(args.get("code"), str):
            yield args["code"]


class Tracer(ast.NodeVisitor):
    """Taints: ("diff",), ("list", cats), ("entry", cats), ("key", cats)."""

    def __init__(self):
        self.env = {}
        self.top = set()          # explicit top-level key reads
        self.fields = set()       # explicit (category, field) reads
        self.whole_top = set()    # categories shown whole via a generic dump
        self.whole_entry = set()  # categories whose entries are shown whole
        self.dynamic = False      # keys taken from d itself (d.items(), for k in d)

    # -- expression taint -------------------------------------------------
    def taint(self, e):
        if isinstance(e, ast.Name):
            return self.env.get(e.id)
        if isinstance(e, ast.Call):
            f = e.func
            if isinstance(f, ast.Name) and f.id == "frame_diff":
                return ("diff",)
            if isinstance(f, ast.Attribute) and f.attr == "get" and e.args:
                return self.index(self.taint(f.value), e.args[0])
            if isinstance(f, ast.Name) and f.id in ("list", "sorted", "reversed") and e.args:
                t = self.taint(e.args[0])
                return t if t and t[0] == "list" else None
            return None
        if isinstance(e, ast.Subscript):
            return self.index(self.taint(e.value), e.slice)
        if isinstance(e, ast.BoolOp):
            for v in e.values:
                t = self.taint(v)
                if t:
                    return t
        if isinstance(e, ast.IfExp):
            return self.taint(e.body) or self.taint(e.orelse)
        if isinstance(e, ast.NamedExpr):
            return self.taint(e.value)
        return None

    def keys_of(self, s):
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            return {s.value}, False
        t = self.taint(s)
        if t and t[0] == "key":
            return set(t[1]), t[2]
        return set(TOP), True

    def index(self, t, s):
        if not t:
            return None
        if t[0] == "diff":
            keys, generic = self.keys_of(s)
            if generic:
                self.dynamic = True
            else:
                self.top |= keys
            cats = frozenset(keys & ALL)
            return ("list", cats) if cats else None
        if t[0] == "list":
            if isinstance(s, ast.Slice):
                return t
            return ("entry", t[1])
        if t[0] == "entry":
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                for c in t[1]:
                    self.fields.add((c, s.value))
            return None
        return None

    # -- binding ----------------------------------------------------------
    def bind(self, target, t):
        if isinstance(target, ast.Name):
            if t:
                self.env[target.id] = t
            else:
                self.env.pop(target.id, None)

    def iter_bind(self, target, it):
        """Bind loop variables for `for target in it`."""
        lits = None
        if isinstance(it, (ast.Tuple, ast.List, ast.Set)):
            vals = [x.value for x in it.elts if isinstance(x, ast.Constant)]
            if vals and all(isinstance(v, str) for v in vals) and set(vals) & set(TOP):
                lits = frozenset(vals)
        if lits is not None:
            self.bind(target, ("key", lits, False))
            return
        # d.items() / d.keys() / d.values() / iterating d itself
        call = it if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) else None
        base = self.taint(call.func.value) if call else self.taint(it)
        if base and base[0] == "diff":
            self.dynamic = True
            self.whole_top |= ALL
            meth = call.func.attr if call else "keys"
            if meth == "items" and isinstance(target, ast.Tuple) and len(target.elts) == 2:
                self.bind(target.elts[0], ("key", frozenset(TOP), True))
                self.bind(target.elts[1], ("list", ALL))
            elif meth == "values":
                self.bind(target, ("list", ALL))
            else:
                self.bind(target, ("key", frozenset(TOP), True))
            return
        t = self.taint(it)
        if t and t[0] == "list":
            self.bind(target, ("entry", t[1]))
        elif isinstance(target, ast.Name):
            self.env.pop(target.id, None)

    def visit_Assign(self, node):
        self.visit(node.value)
        t = self.taint(node.value)
        for tgt in node.targets:
            self.bind(tgt, t)

    def visit_For(self, node):
        self.visit(node.iter)
        self.iter_bind(node.target, node.iter)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_comp(self, node, elts):
        for g in node.generators:
            self.visit(g.iter)
            self.iter_bind(g.target, g.iter)
            for cond in g.ifs:
                self.visit(cond)
        for e in elts:
            self.shown(e)  # collected for output, e.g. [m for m in d['moved']]
            self.visit(e)

    def visit_ListComp(self, node):
        self.visit_comp(node, [node.elt])

    visit_SetComp = visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self.visit_comp(node, [node.key, node.value])

    # -- uses -------------------------------------------------------------
    def shown(self, e):
        """A traced value used whole (printed, stored, formatted)."""
        t = self.taint(e)
        if not t:
            return
        if t[0] == "diff":
            self.whole_top |= ALL
            self.whole_entry |= ALL
        elif t[0] == "list":
            self.whole_entry |= t[1]
        elif t[0] == "entry":
            self.whole_entry |= t[1]

    def visit_Call(self, node):
        f = node.func
        quiet = isinstance(f, ast.Name) and f.id in QUIET
        if isinstance(f, ast.Attribute) and f.attr in ("get", "items", "keys", "values"):
            self.taint(node)  # records the read
            self.visit(f.value) if not isinstance(f.value, ast.Name) else None
            for a in node.args:
                self.visit(a)
            return
        if isinstance(f, ast.Attribute):
            self.visit(f.value)
        for a in list(node.args) + [k.value for k in node.keywords]:
            if not quiet:
                self.shown(a)
            self.visit(a)

    def visit_Subscript(self, node):
        self.taint(node)
        if not isinstance(node.value, ast.Name):
            self.visit(node.value)
        self.visit(node.slice)

    def visit_FormattedValue(self, node):
        self.shown(node.value)
        self.visit(node.value)

    def visit_Dict(self, node):
        for v in node.values:
            self.shown(v)
        self.generic_visit(node)

    def visit_Tuple(self, node):
        if isinstance(node.ctx, ast.Load):
            for v in node.elts:
                self.shown(v)
        self.generic_visit(node)

    visit_List = visit_Tuple


def analyse(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tr = Tracer()
    tr.visit(tree)
    return tr


def main(paths):
    calls = fd_calls = unparsed = 0
    top, top_any = Counter(), Counter()
    field, field_any = Counter(), Counter()
    whole_cat, dynamic, dump_all = Counter(), 0, 0
    games = {}
    for p in paths:
        game = p.rsplit("/", 1)[-1].split("_p")[0]
        for code in codes(p):
            calls += 1
            if not FD.search(code):
                continue
            fd_calls += 1
            tr = analyse(code)
            if tr is None:
                unparsed += 1
                continue
            games.setdefault(game, 0)
            games[game] += 1
            dynamic += tr.dynamic
            dump_all += tr.whole_top == ALL and tr.whole_entry == ALL
            for k in TOP:
                exp = k in tr.top
                top[k] += exp
                top_any[k] += exp or k in tr.whole_top or (
                    k == "changed_cell_count" and bool(tr.whole_top))
            for c in CATS:
                whole_cat[c] += c in tr.whole_entry
                for f in FIELDS[c]:
                    exp = (c, f) in tr.fields
                    field[c, f] += exp
                    field_any[c, f] += exp or c in tr.whole_entry
            stray = {cf for cf in tr.fields if cf[1] not in FIELDS[cf[0]]}
            for c, f in stray:
                field[c, f + " (not a field)"] += 1

    n = max(fd_calls - unparsed, 1)
    print(f"python calls {calls}, calls calling frame_diff {fd_calls} "
          f"({fd_calls / max(calls, 1):.1%}), unparsed {unparsed}, "
          f"games {len(games)}")
    print(f"generic key iteration (d.items(), for k in d): {dynamic} ({dynamic / n:.0%}); "
          f"whole diff shown: {dump_all} ({dump_all / n:.0%})")
    print(f"\n{'top-level key':22}{'named':>8}{'% fd':>7}{'named or shown':>16}{'% fd':>7}")
    for k in TOP:
        print(f"{k:22}{top[k]:>8}{top[k] / n:>7.0%}{top_any[k]:>16}{top_any[k] / n:>7.0%}")
    print(f"\n{'category.field':30}{'named':>8}{'% fd':>7}{'named or entry shown':>22}{'% fd':>7}")
    for c in CATS:
        print(f"{c + ' (entries shown whole)':30}{'':>8}{'':>7}{whole_cat[c]:>22}{whole_cat[c] / n:>7.0%}")
        for f in FIELDS[c]:
            print(f"  {c + '.' + f:28}{field[c, f]:>8}{field[c, f] / n:>7.0%}"
                  f"{field_any[c, f]:>22}{field_any[c, f] / n:>7.0%}")
    extra = {k: v for k, v in field.items() if k[1].endswith("(not a field)")}
    if extra:
        print("\nreads of keys the entry does not have:",
              ", ".join(f"{c}.{f.split()[0]} {v}" for (c, f), v in sorted(extra.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
