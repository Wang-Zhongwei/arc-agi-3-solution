"""Count how often python tool calls reference each segmentation node field.

usage: python -I node_fields.py FIELD,FIELD,... TRANSCRIPT...
"""
import json
import re
import sys
from collections import Counter

fields = sys.argv[1].split(",")
paths = sys.argv[2:]

ARG = re.compile(r'^\s*"arguments":\s*(".*")\s*,?\s*$')


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


def refs(code, f):
    return re.search(
        rf"""\[\s*['"]{f}['"]\s*\]|\.get\(\s*['"]{f}['"]|['"]{f}['"]\s*\)""", code
    ) is not None


calls = seg_calls = 0
hit = Counter()
games = {f: set() for f in fields}
for p in paths:
    game = p.rsplit("/", 1)[-1].split("_p")[0]
    for code in codes(p):
        calls += 1
        if "segmentation" not in code:
            continue
        seg_calls += 1
        for f in fields:
            if refs(code, f):
                hit[f] += 1
                games[f].add(game)

ngames = len({p.rsplit("/", 1)[-1].split("_p")[0] for p in paths})
print(f"python calls {calls}, calls touching segmentation {seg_calls} ({seg_calls / max(calls, 1):.0%}), games {ngames}")
print(f"{'field':22}{'calls':>7}{'% seg calls':>13}{'games':>7}")
for f in sorted(fields, key=lambda f: -hit[f]):
    print(f"{f:22}{hit[f]:>7}{hit[f] / max(seg_calls, 1):>12.0%}{len(games[f]):>5}/{ngames}")
