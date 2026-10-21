"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.grid_utils import format_grid_rle as _format_grid_rle


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        runtime_globals["action"] = action

        def note(**fields):
            # Update the harness-owned world model / hypothesis note; returns what was accepted.
            if not fields:
                raise ValueError("note(...) needs at least one field, e.g. note(prediction='no change').")
            _send({"type": "note", "fields": _json_safe(fields)})
            reply = _recv()
            if reply.get("type") == "note_error":
                raise RuntimeError(str(reply.get("error", "note failed")))
            if reply.get("type") != "note_result":
                raise RuntimeError("Invalid note response from sandbox host.")
            runtime_globals["world_model"] = dict(reply.get("world_model") or {})
            return {"accepted": list(reply.get("accepted") or []), "rejected": dict(reply.get("rejected") or {})}

        runtime_globals["note"] = note

        def remember(**facts):
            # Store short facts in the harness memory; they are shown in every later prompt and survive history trimming.
            if not facts:
                raise ValueError("remember(...) needs at least one keyword, e.g. remember(arm_step='6 cols per RIGHT').")
            _send({"type": "remember", "facts": _json_safe(facts)})
            reply = _recv()
            if reply.get("type") != "remember_result":
                raise RuntimeError("Invalid remember response from sandbox host.")
            return dict(reply.get("facts") or {})

        runtime_globals["remember"] = remember

        def checkpoint(name="cp"):
            # Record the current board as a named checkpoint (free: no action). restore(name) brings the game back to it.
            _send({"type": "checkpoint", "name": str(name)})
            reply = _recv()
            if reply.get("type") != "checkpoint_result":
                raise RuntimeError(str(reply.get("error", "checkpoint failed")))
            return dict(reply.get("result") or {})

        def restore(name="cp", exact=False, via="auto"):
            # Return to a checkpoint by real actions (they are scored): via='undo' presses ACTION7 back to it, via='replay'
            # does RESET + the level's actions up to the checkpoint (no-ops dropped and LEFT/RIGHT, UP/DOWN pairs
            # cancelled unless exact=True); 'auto' picks undo when it is available and cheap, else replay. The result
            # says whether the board matches the checkpoint ('restored'); if a compressed replay missed, it retries exactly.
            _send({"type": "restore", "name": str(name), "exact": bool(exact), "via": str(via)})
            reply = _recv()
            if reply.get("type") != "restore_result":
                raise RuntimeError(str(reply.get("error", "restore failed")))
            result = dict(reply.get("result") or {})
            if reply.get("state"):
                _refresh_state(reply.get("state") or {})
                if result.get("action_result"):
                    action_results.append(dict(result["action_result"]))
            return result

        runtime_globals["checkpoint"] = checkpoint
        runtime_globals["restore"] = restore
        runtime_globals["world_model"] = dict((initial.get("state") or {}).get("world_model") or {})
        runtime_globals["facts"] = dict((initial.get("state") or {}).get("facts") or {})

        def plan(goal, actions=None, max_depth=6, max_nodes=4000, start=None):
            # Breadth-first search over the model's own predict(lines, action) -> lines; returns the shortest action list reaching goal(lines).
            predict_fn = runtime_globals.get("predict")
            if not callable(predict_fn):
                raise RuntimeError("plan(...) needs a predict(lines, action) function defined first.")
            candidates = list(actions) if actions else [a for a in runtime_globals.get("valid_actions", []) if a not in ("MOUSE", "RESET")]
            cur = runtime_globals.get("current_frame")
            start_lines = list(start) if start is not None else (cur.ascii.split("\n") if cur is not None else [])
            if goal(start_lines):
                return []
            seen = {"\n".join(start_lines)}
            frontier = [(start_lines, [])]
            expanded = 0
            while frontier and expanded < max_nodes:
                next_frontier = []
                for lines, path in frontier:
                    if len(path) >= max_depth:
                        continue
                    for act in candidates:
                        try:
                            nxt = predict_fn(list(lines), act)
                        except Exception:
                            continue
                        expanded += 1
                        if nxt is None:
                            continue
                        nxt = [str(x) for x in nxt]
                        key = "\n".join(nxt)
                        if key in seen:
                            continue
                        seen.add(key)
                        if goal(nxt):
                            return path + [act]
                        next_frontier.append((nxt, path + [act]))
                        if expanded >= max_nodes:
                            break
                frontier = next_frontier
            return None

        runtime_globals["plan"] = plan

        def objects(color=None, min_pixels=1, frame=None, include_background=False, composite=False):
            # Compact object list from the frame's segmentation, largest first: id, color(s), pixels, bbox, centre, height, width.
            # composite=True returns grouped objects (a bordered shape with its interior, a checkerboard trail) instead of raw components.
            fr = frame if frame is not None else runtime_globals.get("current_frame")
            if fr is None:
                return []
            if composite:
                out = []
                for o in fr.segmentation.get("objects") or []:
                    if o.get("background") and not include_background:
                        continue
                    if int(o.get("pixels", 0)) < int(min_pixels):
                        continue
                    if color is not None and color not in o.get("colors", []):
                        continue
                    out.append({"id": o["id"], "color": "".join(o["colors"]), "colors": o["colors"], "kind": o["kind"], "parts": len(o["parts"]), "pixels": o["pixels"], "bbox": o["bbox"], "centre": o["centre"], "height": o["height"], "width": o["width"]})
                out.sort(key=lambda o: (-int(o["pixels"] or 0), o["id"]))
                return out
            nodes = list(fr.segmentation.get("nodes") or [])
            board_cells = 1
            try:
                board_cells = max(1, int(fr.shape[0]) * int(fr.shape[1]))
            except Exception:
                pass
            out = []
            for n in nodes:
                if color is not None and n.get("color") != color:
                    continue
                if int(n.get("pixels", 0)) < int(min_pixels):
                    continue
                if not include_background and int(n.get("pixels", 0)) > 0.3 * board_cells:
                    continue
                out.append({k: n.get(k) for k in ("id", "color", "pixels", "bbox", "centre", "height", "width")})
            out.sort(key=lambda o: (-int(o["pixels"] or 0), o["id"]))
            return out

        def board_rle(frame=None):
            # The whole board as run-length rows (identical rows folded); a few hundred tokens for a 64x64 board.
            fr = frame if frame is not None else runtime_globals.get("current_frame")
            return _format_grid_rle(getattr(fr, "_grid", None) or []) if fr is not None else "(no frame)"

        def board_summary(max_items=24, frame=None, composite=True):
            # One line per object ("R 4x4 at r19-22 c42-45 [id 7]") when the board has few objects; a per-colour census otherwise.
            # composite=True (default) lists grouped objects: a bordered piece with its interior, or a textured trail, as one line.
            objs = objects(frame=frame, composite=composite)
            if len(objs) <= max_items:
                return "\n".join(
                    f"{o['color']} {o['height']}x{o['width']} at r{o['bbox'][0]}-{o['bbox'][2]} c{o['bbox'][1]}-{o['bbox'][3]}"
                    + (f" ({o['kind']}, {o['parts']} parts)" if o.get('kind') in ('composite', 'textured') else "") + f" [id {o['id']}]"
                    for o in objs
                ) or "(no objects besides the background)"
            census = {}
            for o in objs:
                key = (o["color"], f"{o['height']}x{o['width']}")
                census[key] = census.get(key, 0) + 1
            by_color = {}
            for (col, shape), n in census.items():
                by_color.setdefault(col, []).append((n, shape))
            lines = [f"{len(objs)} objects; per colour:"]
            for col, items in sorted(by_color.items(), key=lambda kv: -sum(n for n, _ in kv[1])):
                items.sort(reverse=True)
                lines.append(f"  {col}: " + ", ".join(f"{shape} x{n}" for n, shape in items[:6]) + (" ..." if len(items) > 6 else ""))
            big = [o for o in objs if int(o["pixels"]) >= 4][:8]
            if big:
                lines.append("largest: " + "; ".join(f"{o['color']} {o['height']}x{o['width']} at r{o['bbox'][0]} c{o['bbox'][1]}" for o in big))
            return "\n".join(lines)

        runtime_globals["objects"] = objects
        runtime_globals["board_summary"] = board_summary
        runtime_globals["board_rle"] = board_rle

        def scene_graph(frame=None, max_objects=24):
            # Object-centric encoding: nodes are grouped objects (colours + '#' + id), edges are spatial relations written
            # as arrows: inside, overlaps, touches, and each object's nearest neighbour left/right (same rows) and
            # above/below (same cols) with the gap in cells.
            objs = objects(frame=frame, composite=True)[:max_objects]
            def name(o):
                return f"{o['color']}#{o['id']}"
            lines = ["objects:"]
            for o in objs:
                lines.append(f"  {name(o)}: {o['height']}x{o['width']} at r{o['bbox'][0]}-{o['bbox'][2]} c{o['bbox'][1]}-{o['bbox'][3]} ({o['kind']})")
            lines.append("relations:")
            def inside(b, a):
                ar0, ac0, ar1, ac1 = a["bbox"]; br0, bc0, br1, bc1 = b["bbox"]
                return br0 >= ar0 and br1 <= ar1 and bc0 >= ac0 and bc1 <= ac1 and b["pixels"] < a["pixels"]
            for a in objs:
                nearest = {"right": None, "left": None, "below": None, "above": None}
                ar0, ac0, ar1, ac1 = a["bbox"]
                for b in objs:
                    if a is b:
                        continue
                    br0, bc0, br1, bc1 = b["bbox"]
                    rows_overlap = not (br0 > ar1 or br1 < ar0)
                    cols_overlap = not (bc0 > ac1 or bc1 < ac0)
                    if inside(b, a):
                        lines.append(f"  {name(b)} --inside--> {name(a)}")
                        continue
                    if inside(a, b):
                        continue
                    if rows_overlap and cols_overlap:
                        if a["id"] < b["id"]:
                            lines.append(f"  {name(a)} --overlaps--> {name(b)}")
                        continue
                    if rows_overlap:
                        d = (bc0 - ac1 - 1) if bc0 > ac1 else (ac0 - bc1 - 1)
                        side = "right" if bc0 > ac1 else "left"
                        if d == 0 and a["id"] < b["id"]:
                            lines.append(f"  {name(a)} --touches--> {name(b)}")
                        cur = nearest[side]
                        if cur is None or d < cur[0]:
                            nearest[side] = (d, b)
                    elif cols_overlap:
                        d = (br0 - ar1 - 1) if br0 > ar1 else (ar0 - br1 - 1)
                        side = "below" if br0 > ar1 else "above"
                        if d == 0 and a["id"] < b["id"]:
                            lines.append(f"  {name(a)} --touches--> {name(b)}")
                        cur = nearest[side]
                        if cur is None or d < cur[0]:
                            nearest[side] = (d, b)
                for side, hit in nearest.items():
                    if hit is not None:
                        d, b = hit
                        axis = "same rows" if side in ("left", "right") else "same cols"
                        lines.append(f"  {name(a)} --{side} by {d} cells, {axis}--> {name(b)}")
            return "\n".join(lines)

        runtime_globals["scene_graph"] = scene_graph

        def frame_diff(before=None, after=None, min_pixels=1):
            # Object-level diff between two frames (default: previous_frame -> current_frame) using the shape+colour
            # signature ('hash') as a stable object id: moved (with displacement), appeared, disappeared, changed_cells.
            b = before if before is not None else runtime_globals.get("previous_frame")
            a = after if after is not None else runtime_globals.get("current_frame")
            if a is None or b is None:
                return {"error": "need two frames (previous_frame is None on the first turn)"}
            def nodes(fr):
                cells = 1
                try:
                    cells = max(1, int(fr.shape[0]) * int(fr.shape[1]))
                except Exception:
                    pass
                return [n for n in fr.segmentation.get("nodes") or [] if int(n.get("pixels", 0)) >= min_pixels and int(n.get("pixels", 0)) <= 0.3 * cells]
            nb, na = nodes(b), nodes(a)
            by_hash_a = {}
            for n in na:
                by_hash_a.setdefault(n["hash"], []).append(n)
            moved, disappeared, appeared = [], [], []
            for n in nb:
                cands = by_hash_a.get(n["hash"], [])
                if not cands:
                    disappeared.append({"color": n["color"], "size": f"{n['height']}x{n['width']}", "at": n["bbox"][:2]})
                    continue
                best = min(cands, key=lambda m: abs(m["bbox"][0] - n["bbox"][0]) + abs(m["bbox"][1] - n["bbox"][1]))
                cands.remove(best)
                dr, dc = best["bbox"][0] - n["bbox"][0], best["bbox"][1] - n["bbox"][1]
                if dr or dc:
                    moved.append({"color": n["color"], "size": f"{n['height']}x{n['width']}", "from": n["bbox"][:2], "to": best["bbox"][:2], "d": [dr, dc]})
            for rest in by_hash_a.values():
                for m in rest:
                    appeared.append({"color": m["color"], "size": f"{m['height']}x{m['width']}", "at": m["bbox"][:2]})
            bg, ag = getattr(b, "_grid", None) or [], getattr(a, "_grid", None) or []
            changed = sum(1 for r in range(min(len(bg), len(ag))) for c in range(min(len(bg[r]), len(ag[r]))) if bg[r][c] != ag[r][c])
            return {"changed_cells": changed, "moved": moved, "appeared": appeared, "disappeared": disappeared}

        runtime_globals["frame_diff"] = frame_diff

        # --- a small STRIPS planner for PDDL the model writes (and/not, optional '- type' tokens ignored, BFS) ---
        def _sexp(text):
            tokens = text.replace("(", " ( ").replace(")", " ) ").split()
            def parse(i):
                out = []
                while i < len(tokens):
                    t = tokens[i]
                    if t == "(":
                        sub, i = parse(i + 1)
                        out.append(sub)
                    elif t == ")":
                        return out, i + 1
                    else:
                        out.append(t.lower()); i += 1
                return out, i
            tree, _ = parse(0)
            return tree[0] if tree else []

        def _strip_types(items):
            out = []; i = 0
            while i < len(items):
                if items[i] == "-":
                    i += 2; continue
                out.append(items[i]); i += 1
            return out

        def _literals(expr):
            pos, neg = [], []
            if not expr:
                return pos, neg
            if expr[0] == "and":
                for sub in expr[1:]:
                    p, n = _literals(sub); pos += p; neg += n
            elif expr[0] == "not":
                p2, _ = _literals(expr[1]); neg += p2
            else:
                pos.append(tuple(expr))
            return pos, neg

        def pddl_plan(domain, problem, max_nodes=50000):
            # Breadth-first STRIPS search over the grounded actions of a PDDL domain/problem (subset: :predicates,
            # :action with :parameters/:precondition/:effect using and/not; :objects, :init, :goal). Returns a list of
            # action strings such as 'move hero l1 l2', [] when the goal already holds, or None if no plan is found.
            from itertools import product as _product
            from collections import deque as _deque
            d = _sexp(domain); p = _sexp(problem)
            actions = []
            for sec in d[2:]:
                if sec and sec[0] == ":action":
                    spec = {"name": sec[1], "params": [], "pre": ([], []), "eff": ([], [])}
                    i = 2
                    while i < len(sec):
                        key = sec[i]; val = sec[i + 1] if i + 1 < len(sec) else []
                        if key == ":parameters":
                            spec["params"] = _strip_types(val)
                        elif key == ":precondition":
                            spec["pre"] = _literals(val)
                        elif key == ":effect":
                            spec["eff"] = _literals(val)
                        i += 2
                    actions.append(spec)
            objs, init, goal = [], set(), ([], [])
            for sec in p[2:]:
                if sec and sec[0] == ":objects":
                    objs = _strip_types(sec[1:])
                elif sec and sec[0] == ":init":
                    init = {tuple(lit) for lit in sec[1:]}
                elif sec and sec[0] == ":goal":
                    goal = _literals(sec[1])
            def ground(lits, binding):
                return [tuple(binding.get(t, t) for t in lit) for lit in lits]
            grounded = []
            for spec in actions:
                for combo in _product(objs, repeat=len(spec["params"])):
                    b = dict(zip(spec["params"], combo))
                    pre_p, pre_n = spec["pre"]; eff_p, eff_n = spec["eff"]
                    grounded.append((f"{spec['name']} {' '.join(combo)}".strip(), set(ground(pre_p, b)), set(ground(pre_n, b)), set(ground(eff_p, b)), set(ground(eff_n, b))))
            goal_p, goal_n = set(goal[0]), set(goal[1])
            start = frozenset(init)
            if goal_p <= start and not (goal_n & start):
                return []
            frontier = _deque([start]); parent = {start: None}; expanded = 0
            while frontier and expanded < max_nodes:
                state = frontier.popleft(); expanded += 1
                for name_, pre_p, pre_n, eff_p, eff_n in grounded:
                    if not (pre_p <= state) or (pre_n & state):
                        continue
                    nxt = frozenset((state - eff_n) | eff_p)
                    if nxt in parent:
                        continue
                    parent[nxt] = (state, name_)
                    if goal_p <= nxt and not (goal_n & nxt):
                        plan = []; cur = nxt
                        while parent[cur] is not None:
                            prev, act = parent[cur]; plan.append(act); cur = prev
                        return plan[::-1]
                    frontier.append(nxt)
            return None

        runtime_globals["pddl_plan"] = pddl_plan
        _refresh_state(initial.get("state") or {})

        prelude_error = ""
        prelude = str(initial.get("prelude", "") or "")
        if prelude.strip():
            try:
                exec(compile(prelude, "<persistent_helpers>", "exec"), runtime_globals, runtime_globals)
            except Exception as exc:
                prelude_error = "persistent helpers failed to load: " + _sanitize_exception(exc)

        def _score_predict(max_checks=30):
            predict_fn = runtime_globals.get("predict")
            if not callable(predict_fn):
                return []
            checks = []
            trans = runtime_globals.get("transitions") or []
            cur = runtime_globals.get("current_frame")
            level = cur.level if cur is not None else None
            for t in trans[-max_checks:]:
                if level is not None and (t.before_frame.level != level or t.after_frame.level != level):
                    continue
                before = t.before_frame.ascii.split("\n"); after = t.after_frame.ascii.split("\n")
                try:
                    got = predict_fn(list(before), t.action)
                    got = [str(x) for x in got] if got is not None else None
                    err = ""
                except Exception as exc:
                    got, err = None, _sanitize_exception(exc)[:120]
                if got is None:
                    checks.append({"action": t.action, "ok": False, "cell_acc": 0.0, "error": err or "predict returned None"})
                    continue
                total = sum(len(r) for r in after); same = 0
                for r in range(min(len(after), len(got))):
                    a, g = after[r], got[r]
                    same += sum(1 for c in range(min(len(a), len(g))) if a[c] == g[c])
                acc = same / total if total else 0.0
                checks.append({"action": t.action, "ok": got == after, "cell_acc": round(acc, 4), "error": ""})
            return checks

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "predict_checks": _json_safe(_score_predict()),
                    "prelude_error": prelude_error,
                }
            )
        except Exception as exc:
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    "predict_checks": _json_safe(_score_predict()),
                    "prelude_error": prelude_error,
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
    note_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    remember_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    checkpoint_handler: Callable[[str], dict[str, Any]] | None = None,
    restore_handler: Callable[[str, bool, str], dict[str, Any]] | None = None,
    prelude: str = "",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
                "prelude": prelude,
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "checkpoint":
                if checkpoint_handler is None:
                    _send_json_line(process.stdin, {"type": "checkpoint_error", "error": "checkpoint(...) is not available in this session."})
                    continue
                try:
                    _send_json_line(process.stdin, {"type": "checkpoint_result", "result": checkpoint_handler(str(message.get("name") or "cp"))})
                except Exception as exc:  # noqa: BLE001
                    _send_json_line(process.stdin, {"type": "checkpoint_error", "error": f"checkpoint failed: {exc}"})
                continue
            if msg_type == "restore":
                if restore_handler is None:
                    _send_json_line(process.stdin, {"type": "restore_error", "error": "restore(...) is not available in this session."})
                    continue
                try:
                    payload = restore_handler(str(message.get("name") or "cp"), bool(message.get("exact")), str(message.get("via") or "auto"))
                except Exception as exc:  # noqa: BLE001
                    _send_json_line(process.stdin, {"type": "restore_error", "error": " ".join(str(exc).split()) or "restore failed"})
                    continue
                raw = (payload.get("result") or {}).get("action_result")
                if isinstance(raw, dict):
                    host_action_results.append(dict(raw))
                _send_json_line(process.stdin, {"type": "restore_result", "result": payload.get("result") or {}, "state": payload.get("state") or {}})
                continue
            if msg_type == "remember":
                facts_now: dict[str, Any] = {}
                if remember_handler is not None:
                    try:
                        facts_now = dict(remember_handler(dict(message.get("facts") or {})) or {})
                    except Exception:  # noqa: BLE001
                        facts_now = {}
                _send_json_line(process.stdin, {"type": "remember_result", "facts": facts_now})
                continue
            if msg_type == "note":
                if note_handler is None:
                    _send_json_line(process.stdin, {"type": "note_error", "error": "note(...) is not available in this session."})
                    continue
                try:
                    note_payload = note_handler(dict(message.get("fields") or {}))
                except Exception as exc:  # noqa: BLE001
                    _send_json_line(process.stdin, {"type": "note_error", "error": f"note failed: {exc}"})
                    continue
                _send_json_line(
                    process.stdin,
                    {
                        "type": "note_result",
                        "accepted": list(note_payload.get("accepted") or []),
                        "rejected": dict(note_payload.get("rejected") or {}),
                        "world_model": dict(note_payload.get("world_model") or {}),
                    },
                )
                continue
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                except (RuntimeError, ValueError, TypeError) as exc:  # deliberate rejections carry their message
                    _send_json_line(process.stdin, {"type": "action_error", "error": " ".join(str(exc).split()) or "action failed in sandbox host."})
                    continue
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "predict_checks": list(message.get("predict_checks") or []),
                    "prelude_error": str(message.get("prelude_error", "") or ""),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
