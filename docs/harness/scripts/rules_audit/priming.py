#!/usr/bin/env python3
"""Does showing the rules section make the model say "confirmed" more often?

Split each game of the treated run at the thinking-character count where the pinned section first appears.
Split the same game of a control run (no section) at the same count. Rates are per 100K thinking characters.
Usage: priming.py RUNS_DIR treated control [control...]
"""
import collections as C
import glob
import os
import random
import re
import sys

HDR = re.compile(r"^(\[(THINKING|TOOL CALL: \w+|TOOL RESULT: \w+|ASSISTANT|ANALYZER STATUS|USER PROMPT|SYSTEM PROMPT|MODEL RESPONSE META)\]|--- analysis_step=(\d+).*---)\s*$")
SECTION = "rules from trimmed turns"
WORDS = {"confirmed": re.compile(r"\bconfirmed\b", re.I), "hmm (placebo)": re.compile(r"\bhmm+\b", re.I), "hypothes* (placebo)": re.compile(r"\bhypothes\w*", re.I), "refuted family": re.compile(r"\b(refut\w*|disprov\w*|falsif\w*|debunk\w*|ruled out|rules out|rule out)\b", re.I)}


def blocks(path):
    """Yield (section name, text) for each transcript block."""
    sec, buf = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = HDR.match(line.rstrip("\n"))
        if m:
            if sec:
                yield sec, "\n".join(buf)
            sec, buf = m.group(2), []
        else:
            buf.append(line.rstrip("\n"))
    if sec:
        yield sec, "\n".join(buf)


def load(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "transcripts", "*.txt"))):
        g = os.path.basename(f).split("-")[0]
        chars, first, th = 0, None, []
        for sec, text in blocks(f):
            if sec == "THINKING":
                th.append(text)
                chars += len(text)
            elif first is None and SECTION in text:
                first = chars
        out[g] = (th, first)
    return out


def split(th, at):
    """Blocks before and after a thinking-character position."""
    pre, post, c = [], [], 0
    for t in th:
        (pre if c < at else post).append(t)
        c += len(t)
    return "\n".join(pre), "\n".join(post)


runs_dir, treated, *controls = sys.argv[1:]
T = load(os.path.join(runs_dir, treated))
CT = {c: load(os.path.join(runs_dir, c)) for c in controls}
at = {g: f for g, (_, f) in T.items() if f is not None}
print("games with a section in", treated, len(at), "of", len(T))
for name, rx in WORDS.items():
    print("\n##", name)
    rec = {}
    for run, data in [(treated, T)] + list(CT.items()):
        for g, s in at.items():
            if g not in data:
                continue
            pre, post = split(data[g][0], s)
            rec.setdefault(run, {})[g] = (len(rx.findall(pre)), len(pre), len(rx.findall(post)), len(post))
    def rr(run, games):
        a, b, c, d = [sum(rec[run][g][i] for g in games) for i in range(4)]
        return (100000 * a / b if b else 0), (100000 * c / d if d else 0)
    games_all = sorted(at)
    for run in rec:
        p, q = rr(run, [g for g in games_all if g in rec[run] and rec[run][g][3] > 0])
        print("%-5s pre %.2f post %.2f ratio %.2f" % (run, p, q, q / p if p else 0))
    for c in controls:
        gs = [g for g in games_all if g in rec[treated] and g in rec[c] and rec[c][g][3] > 0 and rec[treated][g][3] > 0]
        random.seed(0)
        vals = []
        for _ in range(4000):
            s = [random.choice(gs) for _ in gs]
            tp, tq = rr(treated, s)
            cp, cq = rr(c, s)
            if tp and cp and cq:
                vals.append((tq / tp) / (cq / cp))
        vals.sort()
        tp, tq = rr(treated, gs)
        cp, cq = rr(c, gs)
        print("ratio of ratios vs %s, %d games: %.2f [%.2f, %.2f]" % (c, len(gs), (tq / tp) / (cq / cp), vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]))
