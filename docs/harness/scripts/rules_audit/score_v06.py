#!/usr/bin/env python3
"""Compare run v06's real pinned lines with what the candidate filter would keep, by phase.

Reads <v06 run>/rules_audit: claims_v06.json, wf_result_v06.json (verifiers), cold_result_v06.json (readers).
Writes joined_v06.json.

Line labels:
  bad:    verifiers call it wrong, or misleading.
  useful: correct, or partly correct and not misleading, and meaningful to that phase's reader.
  noise:  everything else.
"""
import collections as C
import json
import pathlib
import random

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
load = lambda name: (lambda d: d.get("result", d))(json.load(open(OUT / name)))
verify = {x["id"]: x for x in load("wf_result_v06.json")["claims"]}
cold = {x["id"]: x for x in load("cold_result_v06.json")["lines"]}
rows = []
for c in json.load(open(OUT / "claims_v06.json")):
    f = (verify.get(c["id"]) or {}).get("final") or {}
    k = cold.get(c["id"]) or {}
    rows.append(dict(c, **{x: f.get(x) for x in ("claim_verdict", "next_role", "consequence_verdict", "usefulness", "confidence", "evidence")}, same=k.get("same"), later=k.get("later")))
json.dump(rows, open(OUT / "joined_v06.json", "w"), indent=1, ensure_ascii=False)

conf = [r for r in rows if r["kind"] == "confirmed"]
print("lines", len(rows), "confirmed", len(conf), "missing verdict", sum(not r["claim_verdict"] for r in rows), "missing reader", sum(not (r["same"] and r["later"]) for r in rows))
wf = list(verify.values())
print("verifiers agree on verdict", sum((x["in_run"] or {}).get("claim_verdict") == (x["cross_run"] or {}).get("claim_verdict") for x in wf), "of", len(wf), "sent to judge", sum(bool(x.get("disputed")) for x in wf))
conf = [r for r in conf if r["claim_verdict"] and r["same"] and r["later"]]
ok = lambda r: r["claim_verdict"] == "correct" or (r["claim_verdict"] == "partially_correct" and r["usefulness"] != "misleading")
bad = lambda r: r["claim_verdict"] == "incorrect" or r["usefulness"] == "misleading"
label = lambda r, rd: "bad" if bad(r) else "useful" if ok(r) and r[rd]["meaningful"] else "noise"
text = lambda r: r["v7_text"] or r["shown"]


def keeps(r, phase, without=()):
    """The content filter drops a line with no v7_text; the rest are in filters_hit."""
    if not r["v7_text"]:
        return False
    hit = filters_hit(text(r), r["next1"])
    return not any(hit[f] for f in active(phase) if f not in without)


def score(ls, rd):
    c = C.Counter(label(r, rd) for r in ls)
    return dict(n=len(ls), useful=c["useful"], bad=c["bad"], noise=c["noise"], correct=sum(r["claim_verdict"] == "correct" for r in ls), wrong=sum(r["claim_verdict"] == "incorrect" for r in ls))


def show(tag, s):
    n = max(1, s["n"])
    print("  %-30s lines %3d useful %3d (%2.0f%%) bad %3d (%2.0f%%) noise %3d (%2.0f%%) correct %2.0f%% wrong %2.0f%%" % (tag, s["n"], s["useful"], 100 * s["useful"] / n, s["bad"], 100 * s["bad"] / n, s["noise"], 100 * s["noise"] / n, 100 * s["correct"] / n, 100 * s["wrong"] / n))


def boot(uni, kept, rd):
    games = sorted({r["game"] for r in uni})
    random.seed(0)
    out = {"useful": [], "bad": [], "noise": []}
    for _ in range(4000):
        a, b = [], []
        for g in [random.choice(games) for _ in games]:
            a += [r for r in uni if r["game"] == g]
            b += [r for r in kept if r["game"] == g]
        if a and b:
            x, y = score(a, rd), score(b, rd)
            for k in out:
                out[k].append(100 * y[k] / y["n"] - 100 * x[k] / x["n"])
    q = lambda v, p: sorted(v)[int(p * len(v))]
    return "  ".join("%s %+.1f [%+.1f, %+.1f]" % (k, q(v, .5), q(v, .025), q(v, .975)) for k, v in out.items())


for ph, rd in (("current", "same"), ("past", "later")):
    uni = conf if ph == "current" else [r for r in conf if r["level"] < r["max_level_seen"]]
    kept = [r for r in uni if keeps(r, ph)]
    print(f"\nphase {ph}")
    show("run v06", score(uni, rd))
    show("candidate filter", score(kept, rd))
    print("  change:", boot(uni, kept, rd))
    print("  kept share: useful %d/%d, bad %d/%d, noise %d/%d" % tuple(x for k in ("useful", "bad", "noise") for x in (score(kept, rd)[k], score(uni, rd)[k])))
    print("  by filter, over all v06 lines in this phase (lines can match several):")
    hits = [("content", [r for r in uni if not r["v7_text"]])] + [(f, [r for r in uni if filters_hit(text(r), r["next1"])[f]]) for f in active(ph)]
    for f, hit in hits:
        s = score(hit, rd)
        only = [r for r in hit if sum(filters_hit(text(r), r["next1"])[g] for g in active(ph)) + (not r["v7_text"]) == 1]
        print("    %-17s lines %3d useful %3d bad %3d noise %3d games %2d wrong %3d | caught by this filter alone %3d (useful %d)" % (f, s["n"], s["useful"], s["bad"], s["noise"], len({r["game"] for r in hit}), s["wrong"], len(only), score(only, rd)["useful"]))
    if ph == "current":
        kept2 = [r for r in uni if keeps(r, ph, without=("agitated",))]
        show("filter without agitated", score(kept2, rd))
        print("  change:", boot(uni, kept2, rd))
    kept3 = [r for r in uni if keeps(r, ph, without=("hedged_next",))]
    show("filter without hedged_next", score(kept3, rd))
    print("  by kind, lines dropped / kept (reader's label):")
    for k, n in C.Counter(r[rd]["info_type"] for r in uni).most_common():
        s = [r for r in uni if r[rd]["info_type"] == k]
        print("    %-18s all %3d kept %3d" % (k, n, sum(r in kept for r in s)))
    print("  examples of useful lines dropped:")
    for r in [r for r in uni if r not in kept and label(r, rd) == "useful"][:12]:
        h = filters_hit(text(r), r["next1"])
        print("    ", r["id"], [f for f in active(ph) if h[f]] or ["content"], "|", r["shown"][:110])
    print("  examples of bad lines kept:")
    for r in [r for r in kept if label(r, rd) == "bad"][:6]:
        print("    ", r["id"], "|", r["shown"][:120])

# The harness shows confirmed lines only for cleared levels, so a line of a level the run never cleared never shows.
never = [r for r in conf if r["level"] >= r["max_level_seen"]]
live_useful = [r for r in conf if label(r, "same") == "useful"]
print("\ndecided design: confirmed lines shown on the level being played: 0 (v06 %d)" % len(conf))
show("never-cleared levels, never shown", score(never, "same"))
print("  useful to the same-level reader, no longer pinned while that level is played: %d of %d" % (len(live_useful), len(conf)))
print("  of those, on a level the run later cleared: %d" % sum(r["level"] < r["max_level_seen"] for r in live_useful))

ref = [r for r in rows if r["kind"] == "refuted"]
print("\nruled-out lines", len(ref), C.Counter(r["claim_verdict"] for r in ref).most_common(), "| v7 extraction keeps", sum(bool(r["v7_text"]) for r in ref))
