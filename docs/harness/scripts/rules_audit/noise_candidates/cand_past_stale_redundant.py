"""Round 3 (final): staleness and redundancy concepts. Scored by noise_harness.py on v02 only.
Each rule reads only text, next1, level, prior."""
import re

WORD = re.compile(r"[a-z][a-z'-]+")
STOP = set("the a an and or of to in on at is are was were be been it its this that these those for with by from as so "
           "now also then confirmed confirm confirmation model rule mechanic".split())
def toks(s): return {w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 2}

PAIR = re.compile(r"\(\s*[-+]?\d+\s*,\s*[-+]?\d+\s*\)")
RANGE = re.compile(r"\b(?:rows?|cols?|columns?)\s+\d+", re.I)
def ncoord(s): return len(PAIR.findall(s)) + len(RANGE.findall(s))

# --- STALENESS ---
# A. Coordinates with no input verb, motion verb or condition: it says where things were, not what they do.
EVENT = re.compile(r"\b(click\w*|press\w*|mov\w*|push\w*|drag\w*|when|after|if|per|each)\b|→", re.I)
RULEWORD = re.compile(r"\b(mechanics?|rules?)\b", re.I)
def static_coords(l):
    t = l["text"]
    return ncoord(t) >= 1 and not EVENT.search(t)
def static_coords_keeprule(l):
    return static_coords(l) and not RULEWORD.search(l["text"])

# B. The named subject is the board's layout.
LAYOUT = re.compile(r"\b(map|geometry|layout)\b", re.I)
LAYOUT_WIDE = re.compile(r"\b(map|geometry|layout|grid|board|schedule|offset)\b", re.I)
def layout_noun(l): return bool(LAYOUT.search(l["text"]))
def layout_wide(l): return bool(LAYOUT_WIDE.search(l["text"]))

# C. Private symbols given literal values: set-builder formulas, X = number, quoted grid drawings.
SYMBOL = re.compile(r"(?<![\w-])[A-Z][A-Za-z]?[-+]?\s*=\s*(?:\{|[-+]?\d|[\"'])|\{[^}]*[<>=]|[\"'][.A-Za-z]{2,}\s?/")
def symbol_literal(l): return bool(SYMBOL.search(l["text"]))

# D. A trace of one attempt: opens on a coordinate, "Step N" or "Then", or cites step numbers.
TRACE = re.compile(r"^\W*(step\s*\d+\b|then\b|\(\s*\d+\s*,\s*\d+\s*\))|\b(at\s+steps?|steps)\s+\d+", re.I)
def episode_trace(l): return bool(TRACE.search(l["text"]))

# E. No claim in the sentence: header cut off at a colon, or "Confirmed" with no subject after it.
SUBJECTLESS = re.compile(r"^\W*confirmed\s+(again\b|twice\b|earlier\b|\w+ly\b)", re.I)
SUBJECTLESS_WIDE = re.compile(r"^\W*confirmed\s+(\(|as\b|again\b|twice\b|earlier\b|\w+ly\b)", re.I)
def no_claim_wide(l):
    t = l["text"].rstrip()
    return t.endswith(":") or bool(SUBJECTLESS_WIDE.search(t))
def no_claim(l):
    t = l["text"].rstrip()
    return t.endswith(":") or bool(SUBJECTLESS.search(t))

def past_union(l):
    return static_coords_keeprule(l) or layout_noun(l) or symbol_literal(l) or episode_trace(l) or no_claim(l)
def current_union(l):
    return episode_trace(l) or no_claim(l)

# --- concepts expected to fail, measured anyway ---
NAMED_LEVEL = re.compile(r"\blevels?[- ]\d", re.I)
def named_level(l): return bool(NAMED_LEVEL.search(l["text"]))
def num_density(th):
    def f(l):
        t = l["text"].split()
        return bool(t) and sum(any(c.isdigit() for c in w) for w in t) / len(t) >= th
    return f
TRANSIENT = re.compile(r"\b(now|remains?|stayed|still|currently)\b", re.I)
def transient_state(l): return bool(TRANSIENT.search(l["text"]))
GENERAL = re.compile(r"\b(when|whenever|if|each|every|per|always|never|only|must|can't|cannot|not|no|rule|mechanic\w*|by)\b", re.I)
def instance_no_general(l): return ncoord(l["text"]) >= 1 and not GENERAL.search(l["text"])

# --- REDUNDANCY against line["prior"] ---
def jacc(a, b): return len(a & b) / len(a | b) if a | b else 0
def contain(a, b): return len(a & b) / len(a) if a else 0
def redun(metric, th, same_level=False, mintok=2):
    def f(l):
        a = toks(l["text"])
        if len(a) < mintok: return False
        return any(metric(a, toks(p["text"])) >= th for p in l["prior"] if not same_level or p["level"] == l["level"])
    return f
def shared_coords(k):
    def f(l):
        a = set(re.sub(r"\s", "", x) for x in PAIR.findall(l["text"]))
        return len(a) >= k and any(p["level"] == l["level"] and len(a & set(re.sub(r"\s", "", x) for x in PAIR.findall(p["text"]))) >= k for p in l["prior"])
    return f

CANDIDATES = {
    "static_coords": ("both", static_coords),
    "static_coords_keeprule": ("both", static_coords_keeprule),
    "layout_noun": ("past", layout_noun),
    "layout_wide": ("past", layout_wide),
    "layout_noun_cur": ("current", layout_noun),
    "symbol_literal": ("both", symbol_literal),
    "episode_trace": ("both", episode_trace),
    "no_claim": ("both", no_claim),
    "FAIL_no_claim_wide": ("past", no_claim_wide),
    "trailing_colon_only": ("both", lambda l: l["text"].rstrip().endswith(":")),
    "past_union": ("past", past_union),
    "current_union": ("current", current_union),
    "FAIL_named_level": ("past", named_level),
    "FAIL_num_density_25": ("past", num_density(.25)),
    "FAIL_num_density_35": ("past", num_density(.35)),
    "FAIL_transient_state": ("past", transient_state),
    "FAIL_instance_no_general": ("past", instance_no_general),
    "RED_jacc_30": ("both", redun(jacc, .3)),
    "RED_jacc_40": ("both", redun(jacc, .4)),
    "RED_jacc_50": ("both", redun(jacc, .5)),
    "RED_jacc_samelevel_30": ("both", redun(jacc, .3, True)),
    "RED_contain_50": ("both", redun(contain, .5)),
    "RED_contain_60": ("both", redun(contain, .6)),
    "RED_subset_3tok": ("both", redun(contain, 1.0, False, 3)),
    "RED_shared_coords_2": ("both", shared_coords(2)),
    "RED_shared_coords_1": ("both", shared_coords(1)),
}
