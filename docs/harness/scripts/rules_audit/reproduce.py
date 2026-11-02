#!/usr/bin/env python3
"""Recompute every number in docs/harness/confirmed-rules-audit.md from the audit data.

Usage: reproduce.py [DATA_DIR]
DATA_DIR defaults to the v02 rules_audit folder (see audit_paths.py).
"""
import collections as C
import json
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule_v8 as R  # noqa: E402

from audit_paths import DATA as DEFAULT_DATA  # noqa: E402

DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
rows = json.load(open(DATA / "joined_v02.json"))
conf = [r for r in rows if r["kind"] == "confirmed"]
ref = [r for r in rows if r["kind"] == "refuted"]


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def verdicts(ls):
    c = C.Counter(r["claim_verdict"] for r in ls)
    return {k: pct(v, len(ls)) for k, v in c.most_common()}


ok = lambda r: r["claim_verdict"] == "correct" or (r["claim_verdict"] == "partially_correct" and r["usefulness"] != "misleading")
bad = lambda r: r["claim_verdict"] == "incorrect" or r["usefulness"] == "misleading"
useful = lambda r, rd: ok(r) and r[rd]["meaningful"] and not bad(r)
label = lambda r, rd: "bad" if bad(r) else "useful" if useful(r, rd) else "noise"
hedged = lambda r: bool(R.HEDGE.search(r["next1"] or ""))
agitated = lambda r: R.agitated(r["shown"])

print("## 1 dataset")
print("claims", len(rows), dict(C.Counter((r["kind"], r["in_section"]) for r in rows)))
print("games", len({r["game"] for r in rows}))

print("\n## 2 accuracy")
print("confirmed", verdicts(conf))
real = [r for r in ref if r["claim_verdict"] != "not_a_claim"]
print("refuted all", len(ref), verdicts(ref))
print("refuted real claims", verdicts(real))
print("not_a_claim in section", sum(r["claim_verdict"] == "not_a_claim" and r["in_section"] for r in ref))
print("refuted in section", verdicts([r for r in ref if r["in_section"]]))

print("\n## 3 next sentence")
for name, ls in (("confirmed", conf), ("refuted", ref)):
    print(name, {k: pct(v, len(ls)) for k, v in C.Counter(r["next_role"] for r in ls).most_common()})
for role in ("consequence", "elaboration"):
    print(role, "truth", verdicts([{"claim_verdict": r["consequence_verdict"]} for r in conf if r["next_role"] == role]))

print("\n## 4 hedge after")
print("confirmed kept", len([r for r in conf if not hedged(r)]), verdicts([r for r in conf if not hedged(r)]))
print("confirmed dropped", len([r for r in conf if hedged(r)]), verdicts([r for r in conf if hedged(r)]))
print("refuted real, no hedge", verdicts([r for r in real if not hedged(r)]))
print("refuted real, hedge", verdicts([r for r in real if hedged(r)]))

print("\n## 5 cleared vs not (level < last level seen)")
cl = [r for r in conf if r["level"] < r["max_level_seen"]]
nc = [r for r in conf if r["level"] >= r["max_level_seen"]]
lv = lambda ls: len({(r["game"], r["level"]) for r in ls})
for name, ls in (("cleared", cl), ("not cleared", nc)):
    print(name, len(ls), "levels", lv(ls), "lines/level %.1f" % (len(ls) / lv(ls)), "agitated", pct(sum(map(agitated, ls)), len(ls)), verdicts(ls))
print("agitated cleared", verdicts([r for r in cl if agitated(r)]))
print("agitated not cleared", verdicts([r for r in nc if agitated(r)]))
print("calm not cleared", verdicts([r for r in nc if not agitated(r)]))

print("\n## 6 cold read")
both = [r for r in conf if r.get("same") and r.get("later")]
print("lines with both readers", len(both))
print("meaningful same", pct(sum(r["same"]["meaningful"] for r in both), len(both)), "later", pct(sum(r["later"]["meaningful"] for r in both), len(both)))
print("fully clear same", pct(sum(r["same"]["comprehensible"] == "clear" for r in both), len(both)), "later", pct(sum(r["later"]["comprehensible"] == "clear" for r in both), len(both)))
print("%-18s %5s %6s %9s %9s %8s" % ("kind (same reader)", "n", "share", "own-level", "later", "correct"))
for k, n in C.Counter(r["same"]["info_type"] for r in both).most_common():
    s = [r for r in both if r["same"]["info_type"] == k]
    print("%-18s %5d %5.0f%% %8.0f%% %8.0f%% %7.0f%%" % (k, n, 100 * n / len(both), 100 * sum(r["same"]["meaningful"] for r in s) / n, 100 * sum(r["later"]["meaningful"] for r in s) / n, 100 * sum(r["claim_verdict"] == "correct" for r in s) / n))
badl = [r for r in both if bad(r)]
print("bad lines", len(badl), "rated meaningful by same reader", sum(r["same"]["meaningful"] for r in badl))
print("reader agreement meaningful", pct(sum(r["same"]["meaningful"] == r["later"]["meaningful"] for r in both), len(both)))


def sc(ls, rd):
    c = C.Counter(label(r, rd) for r in ls)
    n = len(ls)
    return dict(n=n, useful=c["useful"], bad=c["bad"], noise=c["noise"], correct=sum(r["claim_verdict"] == "correct" for r in ls), wrong=sum(r["claim_verdict"] == "incorrect" for r in ls))


def line(tag, s):
    n = max(1, s["n"])
    print("%-14s lines %3d useful %3d (%2.0f%%) bad %3d (%2.0f%%) noise %3d (%2.0f%%) correct %2.0f%% wrong %2.0f%%" % (tag, s["n"], s["useful"], 100 * s["useful"] / n, s["bad"], 100 * s["bad"] / n, s["noise"], 100 * s["noise"] / n, 100 * s["correct"] / n, 100 * s["wrong"] / n))


print("\n## 7 frozen rule v8, in sample")
for ph, uni, rd in (("current", both, "same"), ("past", [r for r in both if r["level"] < r["max_level_seen"]], "later")):
    keep = [r for r in uni if R.keep(r["shown"], r["next1"], ph)]
    print("phase", ph)
    line("v7", sc(uni, rd))
    line("rule v8", sc(keep, rd))
    for f in "HMAI":
        hit = [r for r in uni if R.reasons(r["shown"], r["next1"])[f]]
        if hit:
            line("  catches " + f, sc(hit, rd))
    games = sorted({r["game"] for r in uni})
    random.seed(0)
    db, du = [], []
    for _ in range(4000):
        A, B = [], []
        for g in [random.choice(games) for _ in games]:
            A += [r for r in uni if r["game"] == g]
            B += [r for r in keep if r["game"] == g]
        if A and B:
            a, b = sc(A, rd), sc(B, rd)
            db.append(100 * b["bad"] / b["n"] - 100 * a["bad"] / a["n"])
            du.append(100 * b["useful"] / b["n"] - 100 * a["useful"] / a["n"])
    db.sort()
    du.sort()
    q = lambda v, p: v[int(p * len(v))]
    print("  game bootstrap 95%%: bad %+.1f [%+.1f, %+.1f]  useful %+.1f [%+.1f, %+.1f]" % (q(db, .5), q(db, .025), q(db, .975), q(du, .5), q(du, .025), q(du, .975)))

    print("\n## 8 noise left after rule v8,", ph)
    noise = [r for r in keep if label(r, rd) == "noise"]
    print("noise", len(noise), "verdicts", dict(C.Counter(r["claim_verdict"] for r in noise)))
    print("by kind", C.Counter(r[rd]["info_type"] for r in noise).most_common())
    print("kept lines by kind and label")
    t = C.defaultdict(C.Counter)
    for r in keep:
        t[r[rd]["info_type"]][label(r, rd)] += 1
    for k, v in t.items():
        print("  ", k, dict(v))
    print()

print("## 9 corrections and checks")
RUNS = DATA.parent.parent
summary = (DATA.parent / "summary.txt").read_text()
won = {g for g, got, total in re.findall(r"^\s*(\w+)-\w+: score=[\d.]+, levels=([\d.]+)/(\d+)", summary, re.M) if float(got) == float(total)}
print("won games", len(won))
finals = [r for r in nc if r["game"] in won]
really = [r for r in nc if r["game"] not in won]
print("'not cleared' lines on the last level of a won game", len(finals), "levels", lv(finals), verdicts(finals))
print("truly never cleared", len(really), "levels", lv(really), "lines/level %.1f" % (len(really) / lv(really)), "agitated", pct(sum(map(agitated, really)), len(really)), verdicts(really))
truly_cleared = cl + finals
print("truly cleared", len(truly_cleared), "levels", lv(truly_cleared), "lines/level %.1f" % (len(truly_cleared) / lv(truly_cleared)), "agitated", pct(sum(map(agitated, truly_cleared)), len(truly_cleared)), verdicts(truly_cleared))
print("agitated truly never cleared", verdicts([r for r in really if agitated(r)]))
print("calm truly never cleared", verdicts([r for r in really if not agitated(r)]))
print("agitated truly cleared", verdicts([r for r in truly_cleared if agitated(r)]))
print("doubted confirmed", verdicts([r for r in conf if r["next_role"] == "doubt_about_claim"]))

print("\nlater-reader column, grouped by the later reader's own kind")
for k, n in C.Counter(r["later"]["info_type"] for r in both).most_common():
    s = [r for r in both if r["later"]["info_type"] == k]
    print("  %-18s %3d meaningful %3.0f%%" % (k, n, 100 * sum(r["later"]["meaningful"] for r in s) / n))
inref = [r for r in ref if r["in_section"]]
print("refuted in section, meaningful to same reader", pct(sum(bool(r.get("same") and r["same"]["meaningful"]) for r in inref), len(inref)), [r["game"] for r in inref])

wf = json.load(open(DATA / "wf_result.json"))["claims"]
print("\nverifier lenses agree on verdict", pct(sum((x["in_run"] or {}).get("claim_verdict") == (x["cross_run"] or {}).get("claim_verdict") for x in wf), len(wf)), "sent to judge", sum(bool(x.get("disputed")) for x in wf))

# integrity: the joined file equals its sources, and each claim sits where it says in the raw transcript
V = {x["id"]: x for x in json.load(open(DATA / "verified_v02.json"))}
cold = json.load(open(DATA / "cold_result.json"))["result"]["lines"]
K = {x["id"]: x for x in cold}
print("joined verdict == verified", all(r["claim_verdict"] == V[r["id"]]["claim_verdict"] for r in rows), "| joined same/later == cold", all(r.get("same") == K[r["id"]].get("same") and r.get("later") == K[r["id"]].get("later") for r in rows if r["id"] in K))
print("joined verdict == wf final", all((x.get("final") or {}).get("claim_verdict") == V[x["id"]]["claim_verdict"] for x in wf))
found = 0
for r in rows:
    tr = list(open(DATA.parent / "transcripts" / pathlib.Path(r["transcript"]).name, encoding="utf-8", errors="replace"))
    window = "".join(tr[r["thinking_line"] - 1 : r["thinking_line"] + 400])
    # raw may join a heading to the lines under it, so compare with all whitespace and list marks removed
    squash = lambda t: re.sub(r"[\s\-*#>•]+", "", t)
    found += squash(r["raw"])[:60] in squash(window)
print("claims found at their transcript line", pct(found, len(rows)))
key = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
wrong = C.Counter(r["game"] for r in conf if r["claim_verdict"] == "incorrect")
print("wrong confirmed lines by game", wrong.most_common(), "top 3 share", pct(sum(v for _, v in wrong.most_common(3)), sum(wrong.values())))
print("distinct confirmed lines", len({(r["game"], key(r["shown"])) for r in conf}), "of", len(conf))

print("\n## 10 scores")


def per_game(v):
    return {m[0]: float(m[1]) for m in re.findall(r"^\s*(\w+)-\w+: score=([\d.]+)", (RUNS / v / "summary.txt").read_text(), re.M)}


for a, b in (("v06", "v03"),):
    x, y = per_game(a), per_game(b)
    d = [x[g] - y[g] for g in sorted(set(x) & set(y))]
    m = sum(d) / len(d)
    sd = (sum((i - m) ** 2 for i in d) / (len(d) - 1)) ** 0.5
    t = m / (sd / len(d) ** 0.5)
    try:
        from scipy import stats

        p = "%.2f" % (2 * stats.t.sf(abs(t), len(d) - 1))
    except ImportError:
        p = "n/a (scipy missing)"
    print(a, "%.2f" % (sum(x.values()) / len(x)), b, "%.2f" % (sum(y.values()) / len(y)), "paired n %d diff %+.2f SD %.2f t %.2f p %s" % (len(d), m, sd, t, p))

print("\n## 11 pinned section in v06, last copy per game")
import glob  # noqa: E402

conf_n, ref_n, sizes, with_ref = [], [], [], 0
for f in sorted(glob.glob(str(RUNS / "v06/transcripts/*.txt"))):
    text = open(f, encoding="utf-8", errors="replace").read()
    i = text.rfind("## Confirmed and ruled-out rules from trimmed turns")
    if i < 0:
        continue
    j = re.search(r"\n\[[A-Z]|\n--- analysis_step", text[i:])
    sec = text[i : i + j.start()] if j else text[i:]
    c = r_ = mode = 0
    for ln in sec.split("\n"):
        if ln.startswith("Confirmed:"):
            mode = 1
        elif ln.startswith("Ruled out:"):
            mode = 2
        elif ln.startswith("Level "):
            mode = 0
        elif ln.startswith("- "):
            c += mode == 1
            r_ += mode == 2
    conf_n.append(c)
    ref_n.append(r_)
    sizes.append(len(sec))
    with_ref += r_ > 0
print("games", len(conf_n), "confirmed lines", sum(conf_n), "ruled out", sum(ref_n), "median confirmed", sorted(conf_n)[len(conf_n) // 2], "games with ruled-out", with_ref, "largest section chars", max(sizes))

print("\n## 12 rule v9 (v8 plus useless-line filters), in sample")
import rule_v9 as R9  # noqa: E402


def boot(uni, keep, rd):
    games = sorted({r["game"] for r in uni})
    random.seed(0)
    db, du = [], []
    for _ in range(4000):
        A, B = [], []
        for g in [random.choice(games) for _ in games]:
            A += [r for r in uni if r["game"] == g]
            B += [r for r in keep if r["game"] == g]
        if A and B:
            a, b = sc(A, rd), sc(B, rd)
            db.append(100 * b["bad"] / b["n"] - 100 * a["bad"] / a["n"])
            du.append(100 * b["useful"] / b["n"] - 100 * a["useful"] / a["n"])
    db.sort()
    du.sort()
    q = lambda v, p: v[int(p * len(v))]
    return "bad %+.1f [%+.1f, %+.1f]  useful %+.1f [%+.1f, %+.1f]" % (q(db, .5), q(db, .025), q(db, .975), q(du, .5), q(du, .025), q(du, .975))


for ph, uni, rd in (("current", both, "same"), ("past", [r for r in both if r["level"] < r["max_level_seen"]], "later")):
    k8 = [r for r in uni if R.keep(r["shown"], r["next1"], ph)]
    k9 = [r for r in uni if R9.keep(r["shown"], r["next1"], ph)]
    print("phase", ph)
    line("v7", sc(uni, rd))
    line("rule v8", sc(k8, rd))
    line("rule v9", sc(k9, rd))
    print("  v9 vs v7 bootstrap:", boot(uni, k9, rd))
    print("  v9 vs v8 bootstrap:", boot(k8, k9, rd))
    for f in ("fragment", "snapshot") + (("dead_coords", "dead_layout", "dead_literal") if ph == "past" else ()):
        hit = [r for r in k8 if R9.useless_reasons(r["shown"])[f]]
        if hit:
            line("  " + f[:12], sc(hit, rd))
    lost = [r for r in k8 if r not in k9 and label(r, rd) == "useful"]
    print("  useful lines v9 removes beyond v8:", [(r["id"], r["shown"][:90]) for r in lost])
    left = [r for r in k9 if label(r, rd) == "noise"]
    print("  noise left", len(left), C.Counter(r[rd]["info_type"] for r in left).most_common())

print("\n## 13 cost of A and I on top of H and M, current phase")
hm = [r for r in both if not (hedged(r) or R.META.search(r["shown"]))]
hmai = [r for r in both if R.keep(r["shown"], r["next1"], "current")]
for tag, ls in (("keep all", both), ("H+M", hm), ("H+M+A+I", hmai)):
    line(tag, sc(ls, "same"))
