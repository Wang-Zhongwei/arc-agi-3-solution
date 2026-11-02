#!/usr/bin/env python3
"""Held-out detail on v01: replication of the v02 findings, each filter against the bar, and exploratory variants.

The variants (v9 without A, and so on) were chosen after seeing v01, so v01 is not a clean test of them.
Usage: holdout_extra.py
"""
import collections as C
import json
import random

import rule_v8 as R8
import rule_v9 as R9
from audit_paths import DATA, DATA_V01

LOAD = {"v02": DATA / "joined_v02.json", "v01": DATA_V01 / "joined_v01.json"}
ALL = {k: json.load(open(p)) for k, p in LOAD.items()}
ok = lambda r: r["claim_verdict"] == "correct" or (r["claim_verdict"] == "partially_correct" and r["usefulness"] != "misleading")
bad = lambda r: r["claim_verdict"] == "incorrect" or r["usefulness"] == "misleading"
useful = lambda r, rd: ok(r) and r[rd]["meaningful"] and not bad(r)
label = lambda r, rd: "bad" if bad(r) else "useful" if useful(r, rd) else "noise"
pct = lambda a, b: f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"
PHASES = (("current", "same"), ("past", "later"))


def verdicts(ls):
    return {k: pct(v, len(ls)) for k, v in C.Counter(r["claim_verdict"] for r in ls).most_common()}


def universe(rows, ph):
    conf = [r for r in rows if r["kind"] == "confirmed" and r.get("same") and r.get("later") and r.get("claim_verdict")]
    return conf if ph == "current" else [r for r in conf if r["level"] < r["max_level_seen"]]


def sc(ls, rd):
    c = C.Counter(label(r, rd) for r in ls)
    return dict(n=len(ls), useful=c["useful"], bad=c["bad"], noise=c["noise"], correct=sum(r["claim_verdict"] == "correct" for r in ls), wrong=sum(r["claim_verdict"] == "incorrect" for r in ls))


def show(tag, s):
    n = max(1, s["n"])
    print("  %-30s lines %3d useful %3d (%2.0f%%) bad %3d (%2.0f%%) noise %3d (%2.0f%%) correct %2.0f%% wrong %2.0f%%" % (tag, s["n"], s["useful"], 100 * s["useful"] / n, s["bad"], 100 * s["bad"] / n, s["noise"], 100 * s["noise"] / n, 100 * s["correct"] / n, 100 * s["wrong"] / n))


def boot(uni, kept, rd):
    games = sorted({r["game"] for r in uni})
    random.seed(0)
    db, du, dn = [], [], []
    for _ in range(4000):
        A, B = [], []
        for g in [random.choice(games) for _ in games]:
            A += [r for r in uni if r["game"] == g]
            B += [r for r in kept if r["game"] == g]
        if A and B:
            a, b = sc(A, rd), sc(B, rd)
            db.append(100 * b["bad"] / b["n"] - 100 * a["bad"] / a["n"])
            du.append(100 * b["useful"] / b["n"] - 100 * a["useful"] / a["n"])
            dn.append(100 * b["noise"] / b["n"] - 100 * a["noise"] / a["n"])
    q = lambda v, p: sorted(v)[int(p * len(v))]
    f = lambda v: "%+.1f [%+.1f, %+.1f]" % (q(v, .5), q(v, .025), q(v, .975))
    return f"useful {f(du)}  bad {f(db)}  noise {f(dn)}"


def no_a(r, ph):
    """v9 without the agitated filter."""
    f = R9.reasons(r["shown"], r["next1"])
    if f["H"] or f["M"] or f["fragment"] or f["snapshot"]:
        return False
    if ph == "current":
        return not f["I"]
    return not (f["dead_coords"] or f["dead_layout"] or f["dead_literal"])


def hm_useless(r, ph):
    """H, M and the useless-line filters that met the bar on both runs: no A, no I."""
    f = R9.reasons(r["shown"], r["next1"])
    if f["H"] or f["M"] or f["fragment"] or f["snapshot"]:
        return False
    return ph == "current" or not f["dead_coords"]


for run, rows in ALL.items():
    conf = [r for r in rows if r["kind"] == "confirmed"]
    ref = [r for r in rows if r["kind"] == "refuted"]
    real = [r for r in ref if r["claim_verdict"] != "not_a_claim"]
    print(f"\n#### {run}")
    print("confirmed", verdicts(conf))
    print("refuted real", verdicts(real), "| not a claim", sum(r["claim_verdict"] == "not_a_claim" for r in ref), "of", len(ref), "| in section", verdicts([r for r in ref if r["in_section"]]))
    print("next role", {k: pct(v, len(conf)) for k, v in C.Counter(r["next_role"] for r in conf).most_common()})
    print("consequence truth", verdicts([{"claim_verdict": r["consequence_verdict"]} for r in conf if r["next_role"] == "consequence"]))
    h = lambda r: bool(R8.HEDGE.search(r["next1"] or ""))
    print("no hedge after", verdicts([r for r in conf if not h(r)]))
    print("hedge after", verdicts([r for r in conf if h(r)]))
    ag = lambda r: R8.agitated(r["shown"])
    last = lambda r: r["level"] >= r["max_level_seen"]
    print("agitated, last level seen", verdicts([r for r in conf if ag(r) and last(r)]), "| calm, last level seen", verdicts([r for r in conf if not ag(r) and last(r)]))
    print("agitated, cleared", verdicts([r for r in conf if ag(r) and not last(r)]))
    im = lambda r: bool(R8.IMPOSS.search(r["shown"]))
    print("impossibility", verdicts([r for r in conf if im(r)]))
    w = C.Counter(r["game"] for r in conf if r["claim_verdict"] == "incorrect")
    print("wrong by game", w.most_common(6))
    both = [r for r in conf if r.get("same") and r.get("later")]
    print("meaningful same", pct(sum(r["same"]["meaningful"] for r in both), len(both)), "later", pct(sum(r["later"]["meaningful"] for r in both), len(both)))
    b = [r for r in both if bad(r)]
    print("bad rated meaningful by same reader", pct(sum(r["same"]["meaningful"] for r in b), len(b)))
    for k, n in C.Counter(r["same"]["info_type"] for r in both).most_common():
        s = [r for r in both if r["same"]["info_type"] == k]
        print("  %-18s %3d (%2.0f%%) own %3.0f%% later %3.0f%% correct %3.0f%%" % (k, n, 100 * n / len(both), 100 * sum(r["same"]["meaningful"] for r in s) / n, 100 * sum(r["later"]["meaningful"] for r in s) / n, 100 * sum(r["claim_verdict"] == "correct" for r in s) / n))
    for ph, rd in PHASES:
        uni = universe(rows, ph)
        print(f"\n phase {ph}")
        show("v7", sc(uni, rd))
        k8 = [r for r in uni if R8.keep(r["shown"], r["next1"], ph)]
        k9 = [r for r in uni if R9.keep(r["shown"], r["next1"], ph)]
        kx = [r for r in uni if no_a(r, ph)]
        ky = [r for r in uni if hm_useless(r, ph)]
        show("v8", sc(k8, rd))
        show("v9", sc(k9, rd))
        show("v9 minus A (exploratory)", sc(kx, rd))
        show("H M frag snap coords (expl.)", sc(ky, rd))
        print("  v8 vs v7:", boot(uni, k8, rd))
        print("  v9 vs v7:", boot(uni, k9, rd))
        print("  v9 vs v8:", boot(k8, k9, rd))
        print("  v9-A vs v7:", boot(uni, kx, rd))
        print("  filters over the v7 lines (junk = bad + noise; bar: >=4 lines, >=3 games, >=70% junk):")
        for f in ("H", "M", "A", "I", "fragment", "snapshot", "dead_coords", "dead_layout", "dead_literal"):
            base = k8 if f in ("fragment", "snapshot", "dead_coords", "dead_layout", "dead_literal") else uni
            hit = [r for r in base if R9.reasons(r["shown"], r["next1"])[f]]
            if hit:
                s = sc(hit, rd)
                print("    %-13s n %3d useful %3d bad %3d noise %3d games %2d junk %3.0f%% wrong %3.0f%%" % (f, s["n"], s["useful"], s["bad"], s["noise"], len({r["game"] for r in hit}), 100 * (s["bad"] + s["noise"]) / s["n"], 100 * s["wrong"] / s["n"]))
        if run == "v01":
            print("  useful lines the useless-line filters remove:")
            for r in k8:
                u = R9.useless_reasons(r["shown"])
                fired = [k for k, v in u.items() if v and (ph == "past" or k in ("fragment", "snapshot"))]
                if fired and label(r, rd) == "useful":
                    print("    ", r["id"], fired, "|", r["shown"][:150])

print("\n#### checks and level split")
import pathlib  # noqa: E402
import re  # noqa: E402

for run, rows in ALL.items():
    folder = LOAD[run].parent.parent
    won = {g for g, got, total in re.findall(r"^\s*(\w+)-\w+: score=[\d.]+, levels=([\d.]+)/(\d+)", (folder / "summary.txt").read_text(), re.M) if float(got) == float(total)}
    conf = [r for r in rows if r["kind"] == "confirmed"]
    never = [r for r in conf if r["level"] >= r["max_level_seen"] and r["game"] not in won]
    cleared = [r for r in conf if r not in never]
    lv = lambda ls: len({(r["game"], r["level"]) for r in ls})
    print(run, "won", len(won))
    print("  cleared", len(cleared), "levels", lv(cleared), "per level %.1f" % (len(cleared) / lv(cleared)), verdicts(cleared))
    print("  never cleared", len(never), "levels", lv(never), "per level %.1f" % (len(never) / lv(never)), verdicts(never))
    ag = lambda r: R8.agitated(r["shown"])
    print("  never cleared, agitated", verdicts([r for r in never if ag(r)]), "| calm", verdicts([r for r in never if not ag(r)]))
    squash = lambda t: re.sub(r"[\s\-*#>•]+", "", t)
    found = 0
    for r in rows:
        tr = list(open(folder / "transcripts" / pathlib.Path(r["transcript"]).name, encoding="utf-8", errors="replace"))
        found += squash(r["raw"])[:60] in squash("".join(tr[r["thinking_line"] - 1 : r["thinking_line"] + 400]))
    print("  claims found at their transcript line", pct(found, len(rows)))
wf = json.load(open(DATA_V01 / "wf_result_v01.json"))
wf = wf.get("result", wf)["claims"]
print("v01 lenses agree on verdict", pct(sum((x["in_run"] or {}).get("claim_verdict") == (x["cross_run"] or {}).get("claim_verdict") for x in wf), len(wf)), "sent to judge", sum(bool(x.get("disputed")) for x in wf))

print("\n#### pooled v01 + v02, exploratory")
pool = ALL["v01"] + ALL["v02"]
for g in pool:
    g["game"] = g["game"]
for ph, rd in PHASES:
    out = []
    for run, rows in ALL.items():
        out += [dict(r, game=run + ":" + r["game"]) for r in universe(rows, ph)]
    print("phase", ph)
    show("v7", sc(out, rd))
    for tag, fn in (("v8", lambda r: R8.keep(r["shown"], r["next1"], ph)), ("v9", lambda r: R9.keep(r["shown"], r["next1"], ph)), ("v9 minus A", lambda r: no_a(r, ph))):
        k = [r for r in out if fn(r)]
        show(tag, sc(k, rd))
        print("   ", boot(out, k, rd))
