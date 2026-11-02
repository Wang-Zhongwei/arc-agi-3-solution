#!/usr/bin/env python3
"""Compare run v06's real pinned confirmed lines with the harness filters, by phase.

Reads <v06 run>/rules_audit: claims_v06.json (prepare_v06.py) and cold_result_v06.json (readers, cold_read_workflow.js).
These lines have reader labels only. Nobody checked whether they are true.

current: the line's level is being played. The harness shows no confirmed line here.
past:    the line's level was cleared. The harness applies the filters and shows the rest.
"""
import collections as C
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO / "ARC3-Inference"), str(REPO / "tufa-arc-agi-framework/src")]
from inference.agent.confirmed_rules import confirmed_line_filters

ALWAYS = ("hedged_next", "generic_artifact", "fragment", "snapshot")
# Tried for the level being played, and not in the harness: confirmed lines show only for cleared levels.
LIVE_ONLY = ("agitated", "impossibility")
PAST_ONLY = ("dead_coords", "dead_layout", "dead_literal")
IMPOSSIBILITY = re.compile(r"unsolvable|impossib|dead end|\btrap\b|no exit|no way|can't be (reached|solved)|cannot be (reached|solved)|must be wrong|is wrong", re.I)


def active(phase):
    return ALWAYS + (LIVE_ONLY if phase == "current" else PAST_ONLY)


def filters_hit(line, next_sentence):
    """Every filter that matches, whatever its phase."""
    hit = dict.fromkeys(ALWAYS + LIVE_ONLY + PAST_ONLY, False)
    hit.update(dict.fromkeys(confirmed_line_filters(line, next_sentence), True))
    hit["agitated"] = "CONFIRMED" in line or line.count("✓") >= 2
    hit["impossibility"] = bool(IMPOSSIBILITY.search(line))
    return hit


OUT = REPO / "ARC3-Inference/runs/kaggle/jonathanwang2022/main-tuned-drafter/v06/rules_audit"
cold = {x["id"]: x for x in json.load(open(OUT / "cold_result_v06.json"))["result"]["lines"]}
claims = json.load(open(OUT / "claims_v06.json"))
conf = [c for c in claims if c["kind"] == "confirmed"]
text = lambda c: c["v7_text"] or c["shown"]
hit = lambda c: filters_hit(text(c), c["next1"])


def kept(c, phase):
    """The content filter leaves no v7_text; the rest are in filters_hit."""
    return bool(c["v7_text"]) and not any(hit(c)[f] for f in active(phase))


def row(tag, lines, reader):
    good = sum(bool((cold[c["id"]].get(reader) or {}).get("meaningful")) for c in lines)
    print("  %-26s lines %3d meaningful %3d (%2.0f%%) games %2d" % (tag, len(lines), good, 100 * good / max(1, len(lines)), len({c["game"] for c in lines})))


print("lines", len(claims), "confirmed", len(conf), "ruled out", len(claims) - len(conf), "games", len({c["game"] for c in claims}))
for phase, reader in (("current", "same"), ("past", "later")):
    uni = conf if phase == "current" else [c for c in conf if c["level"] < c["max_level_seen"]]
    survivors = [c for c in uni if kept(c, phase)]
    print(f"\nphase {phase}, {reader}-level reader")
    row("run v06", uni, reader)
    row("filters", survivors, reader)
    row("dropped", [c for c in uni if c not in survivors], reader)
    row("content filter", [c for c in uni if not c["v7_text"]], reader)
    for f in active(phase):
        row(f, [c for c in uni if hit(c)[f]], reader)
    print("  kinds, all / kept:")
    for k, n in C.Counter((cold[c["id"]].get(reader) or {}).get("info_type") for c in uni).most_common():
        print("    %-18s %3d %3d" % (k, n, sum(kept(c, phase) for c in uni if (cold[c["id"]].get(reader) or {}).get("info_type") == k)))
print("\nnever-cleared levels, never shown by the harness")
row("confirmed lines", [c for c in conf if c["level"] >= c["max_level_seen"]], "same")
