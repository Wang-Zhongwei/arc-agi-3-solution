"""Count which fields python tool calls read in Lord Hansolo's harness.

usage: python -I lord_fields.py RUN_DIR

Lord's harness has no frame_diff. Change information comes from the action
result (`last_action_result`, the return value of `action(...)`, and the
compact `history[i].result`), its `animation` reading and
`largest_changed_regions`. Objects come from segmentation nodes. Values are
traced through the code like frame_diff_fields.py does, so a key only counts
when it is read on the right kind of object. For example, `bbox` on a node and
`bbox` on an animation are counted apart.

Also reports what the harness showed without being asked: the compact
`last_action_result` attached to every tool result.
"""
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CALL = re.compile(r"^\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>",
                  re.S | re.M)
RESULT = re.compile(r"^\[TOOL RESULT: python\]\n(.*?)(?=^\[(?:MODEL RESPONSE META|THINKING|TOOL CALL: \w+|"
                    r"TOOL RESULT: \w+|ANALYZER STATUS|SYSTEM PROMPT|USER PROMPT|IMAGE|ASSISTANT)\]|\Z)",
                    re.S | re.M)

DOC = {
    "result": ["changed_pixels", "total_pixels", "largest_changed_regions", "changed_region_count",
               "per_step_changed_pixels", "per_step_frames", "animation", "executed", "done",
               "level_completed", "game_over", "run_complete", "valid_actions", "state", "score"],
    "history.result": ["score", "changed_pixels", "level_completed", "game_over"],
    "animation": ["frame_count", "bbox", "type", "from", "to", "points", "cycles", "parts",
                  "hash", "color", "pixel_count"],
    "region": ["bbox", "pixel_count"],
    "node": ["id", "color", "hash", "pixel_count", "bbox", "mask", "boundary", "children"],
    "frame": ["ascii", "rows", "crop", "step", "level", "shape", "segmentation", "frame_count",
              "animation_frames"],
    "history entry": ["action", "frame", "result", "plan"],
    "transition": ["action", "before_frame", "after_frame", "frame", "result", "plan"],
    "game_over": ["actions", "action_count", "frame"],
}
GLOBALS = {
    "last_action_result": ("result", "last_action_result"),
    "current_frame": ("frame",), "previous_frame": ("frame",), "latest_frame": ("frame",),
    "history": ("list", "history entry"), "transitions": ("list", "transition"),
    "last_transition": ("obj", "transition"), "game_overs": ("list", "game_over"),
}
QUIET = {"len", "bool", "isinstance", "id", "type", "any", "all", "sum", "enumerate", "sorted",
         "list", "zip", "reversed", "range", "min", "max", "set", "tuple"}


class Tracer(ast.NodeVisitor):
    def __init__(self):
        self.env = {}
        self.reads = set()   # (object kind, key)
        self.whole = set()   # object kinds shown whole
        self.result_src = set()

    def kind_of(self, t):
        if not t:
            return None
        if t[0] == "result":
            return "history.result" if t[1] == "history.result" else "result"
        if t[0] in ("obj",):
            return t[1]
        return {"frame": "frame", "anim": "animation", "region": "region", "node": "node"}.get(t[0])

    def taint(self, e):
        if isinstance(e, ast.Name):
            return self.env.get(e.id, GLOBALS.get(e.id))
        if isinstance(e, ast.Call):
            f = e.func
            if isinstance(f, ast.Name) and f.id == "action":
                return ("result", "action()")
            if isinstance(f, ast.Attribute) and f.attr == "get" and e.args:
                return self.index(self.taint(f.value), e.args[0])
            if isinstance(f, ast.Name) and f.id in ("list", "sorted", "reversed") and e.args:
                t = self.taint(e.args[0])
                return t if t and t[0] in ("list", "regions", "nodes", "frames") else None
            if isinstance(f, ast.Attribute):
                self.attr(f)
            return None
        if isinstance(e, ast.Attribute):
            return self.attr(e)
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

    def attr(self, e):
        t = self.taint(e.value)
        k = self.kind_of(t)
        a = e.attr
        if k == "frame":
            self.reads.add(("frame", a))
            if a == "segmentation":
                return ("seg",)
            if a == "animation_frames":
                return ("frames",)
            return None
        if k in ("history entry", "transition", "game_over"):
            self.reads.add((k, a))
            if a == "result":
                # Only last_transition carries the full result; history entries and
                # earlier transitions carry the compact one.
                full = isinstance(e.value, ast.Name) and e.value.id == "last_transition"
                return ("result", "last_transition.result" if full else "history.result")
            if a in ("frame", "before_frame", "after_frame"):
                return ("frame",)
        return None

    def key(self, s):
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            return s.value
        return None

    def index(self, t, s):
        if not t:
            return None
        k = self.key(s)
        if t[0] == "result":
            if k:
                self.reads.add((self.kind_of(t), k))
                self.result_src.add(t[1])
                if k == "animation":
                    return ("anim",)
                if k == "largest_changed_regions":
                    return ("regions",)
            return None
        if t[0] == "anim":
            if k:
                self.reads.add(("animation", k))
            return None
        if t[0] == "seg":
            if k == "nodes":
                return ("nodes",)
            return None
        if t[0] in ("regions", "nodes", "list", "frames"):
            if isinstance(s, ast.Slice):
                return t
            return {"regions": ("region",), "nodes": ("node",), "frames": ("frame",)}.get(
                t[0], ("obj", t[1]) if t[0] == "list" else None)
        if t[0] in ("region", "node"):
            if k:
                self.reads.add((t[0], k))
            return None
        return None

    def bind(self, target, t):
        if isinstance(target, ast.Name):
            if t:
                self.env[target.id] = t
            else:
                self.env.pop(target.id, None)

    def element(self, t):
        if not t:
            return None
        if t[0] == "list":
            return ("obj", t[1])
        return {"regions": ("region",), "nodes": ("node",), "frames": ("frame",)}.get(t[0])

    def iter_bind(self, target, it):
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "enumerate"
                and it.args and isinstance(target, ast.Tuple) and len(target.elts) == 2):
            self.bind(target.elts[1], self.element(self.taint(it.args[0])))
            return
        self.bind(target, self.element(self.taint(it)))

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
            self.shown(e)
            self.visit(e)

    def visit_ListComp(self, node):
        self.visit_comp(node, [node.elt])

    visit_SetComp = visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node):
        self.visit_comp(node, [node.key, node.value])

    def shown(self, e):
        t = self.taint(e)
        if not t:
            return
        if t[0] in ("regions",):
            self.whole.add("region")
        elif t[0] == "nodes":
            self.whole.add("node")
        else:
            k = self.kind_of(t)
            if k and k != "frame":
                self.whole.add(k)

    def visit_Call(self, node):
        f = node.func
        quiet = isinstance(f, ast.Name) and f.id in QUIET
        self.taint(node)
        if isinstance(f, ast.Attribute):
            self.visit(f.value)
        for a in list(node.args) + [k.value for k in node.keywords]:
            if not quiet:
                self.shown(a)
            self.visit(a)

    def visit_Attribute(self, node):
        self.attr(node)
        self.visit(node.value)

    def visit_Subscript(self, node):
        self.taint(node)
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


def main(run):
    calls = unparsed = 0
    reads, whole, touched = Counter(), Counter(), Counter()
    games = defaultdict(set)
    srcs = Counter()
    echo_calls, echo_keys, anim_types = 0, Counter(), Counter()
    for path in sorted(Path(run, "transcripts").glob("*.txt")):
        game = path.stem.split("_p")[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        for code, res in zip(CALL.findall(text), RESULT.findall(text)):
            calls += 1
            try:
                tree = ast.parse(code)
            except SyntaxError:
                unparsed += 1
                continue
            tr = Tracer()
            tr.visit(tree)
            for r in tr.reads:
                reads[r] += 1
                games[r].add(game)
            for k in {k for k, _ in tr.reads} | tr.whole:
                touched[k] += 1
            for k in tr.whole:
                whole[k] += 1
            srcs.update(tr.result_src)
            # One action() call attaches its result directly; several attach
            # {"action_calls": n, "last_action_result": {...}}.
            try:
                ar = json.loads(res.strip()).get("action_result") or {}
                lar = ar.get("last_action_result", ar) if isinstance(ar, dict) else None
            except (json.JSONDecodeError, AttributeError):
                lar = None
            if isinstance(lar, dict) and lar:
                echo_calls += 1
                echo_keys.update(k for k, v in lar.items() if v not in (None, [], {}))
                a = lar.get("animation")
                anim_types[a.get("type") if isinstance(a, dict) else "none"] += 1

    n = max(calls - unparsed, 1)
    ngames = len(list(Path(run, "transcripts").glob("*.txt")))
    print(f"python calls {calls}, unparsed {unparsed}")
    print("result read via:", dict(srcs.most_common()))
    for kind, keys in DOC.items():
        print(f"\n{kind:22}{'calls':>7}{'% all':>7}   (calls touching it: {touched[kind]}, "
              f"shown whole: {whole[kind]})")
        extra = sorted({k for (kd, k) in reads if kd == kind} - set(keys))
        for k in keys + extra:
            c = reads[kind, k]
            mark = "" if k in keys else "  (undocumented)"
            print(f"  {k:20}{c:>7}{c / n:>7.0%}  games {len(games[kind, k])}{mark}")
    print(f"\ntool results carrying the compact last_action_result: {echo_calls}")
    print("  non-empty keys:", dict(echo_keys.most_common()))
    print("  animation type:", dict(anim_types.most_common()))


if __name__ == "__main__":
    main(sys.argv[1])
