"""Rule-filter variants on SF's 512 labelled quotes and MR's 40 judged this-level quotes."""
import json, re, sys
from collections import Counter
sys.path.insert(0, sys.argv[1])
from inference.agent.confirmed_rules import _rule_ok_v2, _MARK_RE
R = json.load(open(sys.argv[2]))
J = [json.loads(l) for l in open(sys.argv[3]) if json.loads(l)['type'] == 'rules']
GOALNEG = re.compile(r"goal is not|not the goal|\bimpossible\b", re.I)
NEG = re.compile(r"\b(?:never|cannot|can't|can not)\b", re.I)
REQ = re.compile(r"\b(?:requires?|required|must|critical)\b", re.I)
FRAG_STRICT = re.compile(r"^(?:[✓✔✅]|\()")
FRAG_LOWER = re.compile(r"^[a-z]")
POSS = re.compile(r"\bpossible:|^also possible|\bconsistent with\b", re.I)
VARIANTS = {
    'v3 (as built)': dict(goalneg=1, neg=1, req=1, frag='lower', poss=1),
    'no lowercase-fragment': dict(goalneg=1, neg=1, req=1, frag='strict', poss=1),
    'no marked never/cannot': dict(goalneg=1, neg=0, req=1, frag='strict', poss=1),
    'goal-neg + poss only': dict(goalneg=1, neg=0, req=0, frag='strict', poss=1),
}
def ok(text, v):
    marked = bool(_MARK_RE.search(text))
    if not _rule_ok_v2(text, marked=marked): return False
    if v['goalneg'] and GOALNEG.search(text): return False
    if v['neg'] and NEG.search(text): return False
    if v['req'] and not marked and REQ.search(text): return False
    if v['frag'] == 'lower' and (FRAG_LOWER.match(text) or FRAG_STRICT.match(text)): return False
    if v['frag'] == 'strict' and FRAG_STRICT.match(text): return False
    if v['poss'] and POSS.search(text): return False
    return True
for name, v in VARIANTS.items():
    kept = [d for d in R if ok(d['rule'], v)]
    c = Counter(d['cls'] for d in kept)
    jk = [j for j in J if ok(j['text'], v)]; jd = [j for j in J if not ok(j['text'], v)]
    bad = lambda js: sum(j['verdict'] in ('wrong', 'misleading', 'stale') for j in js)
    good = lambda js: sum(j['verdict'] == 'correct' for j in js)
    print(f"{name:24s} SF kept {len(kept)} M {c['M']} H {c['H']} T {c['T']} contra {sum(d['contra'] for d in kept)} M-share {c['M']/len(kept):.3f} H-share {c['H']/len(kept):.3f}"
          f" | MR judged: dropped bad {bad(jd)}/{bad(J)}, dropped correct {good(jd)}/{good(J)}, dropped noise {sum(j['verdict']=='noise' for j in jd)}")
