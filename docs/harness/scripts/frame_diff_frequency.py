"""How often python tool calls use frame_diff, and when.

usage: python -I frame_diff_frequency.py RUN_DIR...

Reads RUN_DIR/transcripts/*.txt for the calls and
RUN_DIR/artifacts/*_events.jsonl for each turn's level (one `analysis` event
per transcript turn). Several RUN_DIRs are pooled.
"""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

TURN = re.compile(r"^--- analysis_step=\d+ \|.*?---$", re.M)
CALL = re.compile(r"^\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>",
                  re.S | re.M)
RESULT = re.compile(r"^\[TOOL RESULT: python\]\n(.*?)(?=^\[(?:MODEL RESPONSE META|THINKING|TOOL CALL: python|"
                    r"TOOL RESULT: python|ANALYZER STATUS|SYSTEM PROMPT|USER PROMPT)\]|\Z)", re.S | re.M)
FD = re.compile(r"\bframe_diff\s*\(")
ACTION = re.compile(r"\baction\s*\(")
ERROR = re.compile(r"Traceback|^\w*Error\b|\bError:", re.M)
LOOPS = (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)


def invocations(code):
    """(no-arg?, inside a loop?, preceded by action()?) per frame_diff call."""
    tree = ast.parse(code)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    action_lines = [n.lineno for n in calls if n.func.id == "action"]
    out = []
    for n in calls:
        if n.func.id != "frame_diff":
            continue
        p, loop = parents.get(n), False
        while p is not None and not loop:
            loop = isinstance(p, LOOPS)
            p = parents.get(p)
        out.append((not n.args and not n.keywords, loop, any(l <= n.lineno for l in action_lines)))
    return out


def levels(run, game):
    path = next(Path(run, "artifacts").glob(game + "_p*_events.jsonl"), None)
    if path is None:
        return None
    return [e["level"] for e in map(json.loads, open(path, encoding="utf-8"))
            if e.get("type") == "analysis"]


def main(runs):
    c = Counter()
    phase, phase_n = Counter(), Counter()
    games, games_fd = 0, 0
    for run in runs:
        for path in sorted(Path(run, "transcripts").glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            heads = list(TURN.finditer(text))
            lv = levels(run, path.stem.split("_p")[0])
            if lv is not None and len(lv) != len(heads):
                lv = None
            games += 1
            used, cur, k = False, None, 0
            for i, h in enumerate(heads):
                body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
                codes, results = CALL.findall(body), RESULT.findall(body)
                c["turns"] += 1
                turn_fd = False
                for j, code in enumerate(codes):
                    res = results[j] if j < len(results) else ""
                    fd, act, err = bool(FD.search(code)), bool(ACTION.search(code)), bool(ERROR.search(res))
                    c["calls"] += 1
                    c["action calls"] += act
                    c["errors fd" if fd else "errors other"] += err
                    if not fd:
                        continue
                    turn_fd = used = True
                    c["fd calls"] += 1
                    c["fd calls with action"] += act
                    try:
                        inv = invocations(code)
                    except SyntaxError:
                        inv = [(None, None, None)] * len(FD.findall(code))
                    c["invocations"] += len(inv)
                    c["no-arg"] += sum(a is True for a, _, _ in inv)
                    c["in loop"] += sum(l is True for _, l, _ in inv)
                    c["after action"] += sum(b is True for _, _, b in inv)
                c["fd turns"] += turn_fd
                if lv is not None:
                    k = k + 1 if lv[i] == cur else 1
                    cur = lv[i]
                    key = ("level 1" if cur == 1 else "level 2+", "turns 1-3" if k <= 3 else "turns 4+")
                    phase_n[key] += 1
                    phase[key] += turn_fd
            games_fd += used

    pct = lambda a, b: f"{a} / {b} ({100 * a / max(b, 1):.1f}%)"
    print(f"games using frame_diff: {pct(games_fd, games)}")
    print(f"calls using frame_diff: {pct(c['fd calls'], c['calls'])}")
    print(f"turns with a frame_diff call: {pct(c['fd turns'], c['turns'])}")
    print(f"action() calls that also call frame_diff: {pct(c['fd calls with action'], c['action calls'])}")
    print(f"invocations: {c['invocations']}; no-arg {pct(c['no-arg'], c['invocations'])}; "
          f"after action() {pct(c['after action'], c['invocations'])}; in a loop {pct(c['in loop'], c['invocations'])}")
    other = c["calls"] - c["fd calls"]
    print(f"errors: frame_diff calls {pct(c['errors fd'], c['fd calls'])}, others {pct(c['errors other'], other)}")
    for key in sorted(phase_n):
        print(f"turns using frame_diff, {key[0]}, {key[1]}: {pct(phase[key], phase_n[key])}")


if __name__ == "__main__":
    main(sys.argv[1:])
