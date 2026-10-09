"""Count open() and JSON calls submitted to the Python tool in saved runs.

Usage: python -I docs/harness/scripts/file_io_frequency.py RUN_DIR...

Counts tool calls containing each API, not dynamic invocations: a call inside a
function definition or dead branch still counts as submitted code. Confirmed
open failures come separately from the Python tool results.
"""

import ast
import re
import sys
from collections import Counter
from pathlib import Path


TOOL_CALL = re.compile(
    r"^\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n"
    r"<parameter=code>\n(.*?)\n</parameter>",
    re.M | re.S,
)
OPEN_ERROR = re.compile(r"NameError: name 'open' is not defined")
NAMES = ("open", "json.load", "json.dump", "json.loads", "json.dumps")


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def count_run(run):
    totals = Counter()
    by_game = {}
    for path in sorted((run / "transcripts").glob("*.txt")):
        game = Counter()
        transcript = path.read_text(encoding="utf-8", errors="replace")
        game["open_errors"] = len(OPEN_ERROR.findall(transcript))
        for code in TOOL_CALL.findall(transcript):
            game["python_calls"] += 1
            try:
                tree = ast.parse(code)
            except SyntaxError:
                game["unparsed_calls"] += 1
                if re.search(r"\b(?:open|json)\s*(?:\(|\.)", code):
                    game["unparsed_candidates"] += 1
                continue
            names = Counter(
                call_name(node.func)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
            )
            for name in NAMES:
                if names[name]:
                    game[name + "_calls"] += 1
                    game[name + "_sites"] += names[name]
        totals.update(game)
        totals["games"] += 1
        by_game[path.stem] = game
    return totals, by_game


def main(paths):
    header = "run | games | python calls | open | open errors | json.load/dump | json.loads/dumps | unparsed"
    print(header)
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    combined = Counter()
    for path in paths:
        run = Path(path)
        counts, games = count_run(run)
        combined.update(counts)
        print(
            f"{run.parent.name}/{run.name} | {counts['games']} | {counts['python_calls']} | "
            f"{counts['open_calls']} | {counts['open_errors']} | "
            f"{counts['json.load_calls']}/{counts['json.dump_calls']} | "
            f"{counts['json.loads_calls']}/{counts['json.dumps_calls']} | "
            f"{counts['unparsed_calls']}"
        )
        for name, game in games.items():
            if game["open_calls"] or game["json.load_calls"] or game["json.dump_calls"]:
                print(
                    f"  {name}: open={game['open_calls']}, "
                    f"open errors={game['open_errors']}, "
                    f"json.load/dump={game['json.load_calls']}/{game['json.dump_calls']}"
                )
    if len(paths) > 1:
        print(
            f"TOTAL | {combined['games']} | {combined['python_calls']} | "
            f"{combined['open_calls']} | {combined['open_errors']} | "
            f"{combined['json.load_calls']}/{combined['json.dump_calls']} | "
            f"{combined['json.loads_calls']}/{combined['json.dumps_calls']} | "
            f"{combined['unparsed_calls']}"
        )
    if combined["unparsed_candidates"]:
        print(f"Warning: {combined['unparsed_candidates']} unparsed calls mention open or json.")


if __name__ == "__main__":
    main(sys.argv[1:])
