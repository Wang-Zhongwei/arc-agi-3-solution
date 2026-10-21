"""The read-only REPL child: a persistent subprocess with immutable views of harness memory.

Protocol (stdin/stdout JSON lines): the parent sends {"type":"snapshot", ...} to refresh the
views and {"type":"exec", "code": ...} to run a snippet; the child replies {"type":"result",
"stdout":..., "result":..., "error":...}. Imports are limited to the 14-module allowlist; the
views are tuples and MappingProxyType, and there is no handle to the parent's objects, so the
model can look at the ledger and never edit it.
"""

import builtins
import contextlib
import io
import json
import sys
import traceback
from types import MappingProxyType

SAFE_MODULES = {"bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json",
                "math", "operator", "random", "re", "statistics", "string"}
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int",
    "isinstance", "len", "list", "map", "max", "min", "next", "print", "range", "repr", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "hasattr", "getattr", "type",
}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name or "").split(".", 1)[0]
    if root not in SAFE_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in the REPL.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def main() -> None:
    env = {"__builtins__": {n: getattr(builtins, n) for n in SAFE_BUILTINS}}
    env["__builtins__"]["__import__"] = _safe_import
    persistent = {}  # model-defined helpers survive across exec calls within a game
    out = sys.stdout
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "snapshot":
            for key in ("objects", "ledger", "rules", "frame", "control", "scalars", "questions"):
                env[key] = _freeze(msg.get(key))

            def diff(a, b, _f=env.get("frame")):
                """Cells that differ between two grids: [(r, c, a_val, b_val)] (at most 200)."""
                rows = []
                for r, (ra, rb) in enumerate(zip(a, b)):
                    for c, (va, vb) in enumerate(zip(ra, rb)):
                        if va != vb:
                            rows.append((r, c, va, vb))
                            if len(rows) >= 200:
                                return rows
                return rows

            def count(pred):
                """Number of ledger records for which pred(record) is true."""
                return sum(1 for rec in env["ledger"] if pred(rec))

            def test_rule(rid):
                """Support/contradict counts the harness recorded for a rule id (a read, not a re-score)."""
                for r in env["rules"]:
                    if r["rid"] == rid:
                        return {"support": r["score"]["support"], "contradict": r["score"]["contradict"],
                                "status": r["score"]["status"]}
                return None

            env["diff"] = diff
            env["count"] = count
            env["test_rule"] = test_rule
            env.update(persistent)
            continue
        if msg.get("type") != "exec":
            continue
        buf = io.StringIO()
        result = None
        error = ""
        try:
            code = compile(str(msg.get("code", "")), "<repl>", "exec")
            scope = dict(env)
            scope["result"] = None
            with contextlib.redirect_stdout(buf):
                exec(code, scope, scope)
            result = scope.get("result")
            for k, v in scope.items():
                if callable(v) and k not in env and not k.startswith("_"):
                    persistent[k] = v
                    env[k] = v
        except Exception as exc:  # noqa: BLE001
            tb = traceback.extract_tb(exc.__traceback__)
            user = [f for f in tb if f.filename == "<repl>"]
            error = "".join(f'  line {f.lineno}\n' for f in user) + f"{type(exc).__name__}: {exc}"
        try:
            payload = json.dumps({"type": "result", "stdout": buf.getvalue()[:4000], "result": result, "error": error}, default=str)
        except (TypeError, ValueError):
            payload = json.dumps({"type": "result", "stdout": buf.getvalue()[:4000], "result": str(result)[:2000], "error": error})
        out.write(payload + "\n")
        out.flush()


if __name__ == "__main__":
    main()
