"""Critic's shortlist: what survived the attack on the four designers' candidates.
Text-only language patterns; fn(line) -> True drops the line. Uses only line['text'].
Written after reading the v02 labels, so the scores below are optimistic until the held-out run.

current_rule: fragment or snapshot.
past_rule:    current_rule or dead_coords or dead_layout or dead_literal. The dead_* rules are for cleared levels only:
              readers want these lines while the level is live."""
import re

def text(l): return l['text'].strip()

def mark_cells(s):
    """Spell a (r,c) cell as <r,c> so it survives the aside stripping below."""
    return re.sub(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)", r"<\1,\2>", s)

def main_clause(s):
    """Text without asides, nested ones included; a bare cell stays because it is a value, not an aside."""
    s, prev = mark_cells(s), None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", s)
    return s

# A location: <r,c>, "rows 3", "col 2".
PLACE = re.compile(r"<[-+]?\d+,[-+]?\d+>|\b(rows?|cols?|columns?)\s+\d+", re.I)
# The line says something holds in general or names an input. Either way it is not a read-out.
GENERAL = re.compile(r"→|\b(when|if|until|per|each|every|always)\b", re.I)
INPUT = re.compile(r"\b(click|press|mov|push)\w*", re.I)
def generalises(s): return bool(GENERAL.search(s) or INPUT.search(s))

# --- FRAGMENT: the sentence lost its content. A heading whose list was cut off, or a short sentence whose subject is a cell.
CELL_OPEN = re.compile(r"^\W*(then\s+)?\(\s*-?\d+\s*,\s*\d+\s*\)", re.I)
def fragment(l):
    s = text(l)
    return s.endswith(':') or (bool(CELL_OPEN.search(s)) and len(s.split()) <= 12)

# --- SNAPSHOT: where something is at one instant. A right-now word with a location, a bare location predicate, or a step tie.
# These markers count only in the main clause: "dots move rigidly (now at col 15)" still states a rule.
MOMENT = re.compile(r"\bnow\b(?!\s+(confirmed|verified))|\bcurrently\b|\bremain(s|ing)?\b|\bfresh\b|\binitial\b", re.I)
LOCATED = re.compile(r"\b(is|are|was|were|occup(y|ies|ied))\s+(at\s+)?(rows?\s*\d|cols?\s*\d|<)", re.I)
AT_STEP = re.compile(r"\bat steps?\s+\d|^\W*step\s+\d+\s*:", re.I)
def snapshot(l):
    s = text(l)
    if generalises(s):
        return False
    m = main_clause(s)
    return bool((MOMENT.search(m) and PLACE.search(mark_cells(s))) or LOCATED.search(m) or AT_STEP.search(m))

# --- DEAD_COORDS: a place is stated in the main clause and nothing is said to happen there.
EVENT = re.compile(r"\b(?!confirm|verifi)\w{3,}ed\b", re.I)    # a past-tense event other than the claim word
NAMED_RULE = re.compile(r"\b(mechanics?|rules?)\b", re.I)      # the line calls itself a rule
CELL_VERB = re.compile(r">\s+[a-z]{3,}s\b")                    # a verb right after the cell: "<2,2> turns"
def dead_coords(l):
    s = text(l)
    m = main_clause(s)
    return bool(PLACE.search(m)) and not (CELL_VERB.search(m) or generalises(s) or EVENT.search(s) or NAMED_RULE.search(s))

# --- DEAD_LAYOUT: one level's layout was checked, and it comes with numbers.
LAYOUT_WORD = r"(map|geometry|layout|regions?)"
LAYOUT = re.compile(r"\b" + LAYOUT_WORD + r"\s+(is\s+)?confirmed|\bconfirmed\s+(the\s+)?([\w-]+\s+){0,2}?" + LAYOUT_WORD + r"\b", re.I)
def dead_layout(l):
    s = text(l)
    return bool(LAYOUT.search(s)) and bool(re.search(r"\d", s))

# --- DEAD_LITERAL: a private symbol set to a literal value, a set-builder formula, or a quoted grid drawing.
LITERAL = re.compile(r"(?<![\w-])[A-Z][-+]?\s*=\s*[{\d+-]|\{[^}]*[<>=]|[\"'][.A-Za-z]{2,}\s?/")
def dead_literal(l): return bool(LITERAL.search(text(l)))

def current_rule(l): return fragment(l) or snapshot(l)
def past_rule(l): return current_rule(l) or dead_coords(l) or dead_layout(l) or dead_literal(l)

CANDIDATES = {
    'fragment':     ('both', fragment),
    'snapshot':     ('both', snapshot),
    'dead_coords':  ('past', dead_coords),
    'dead_layout':  ('past', dead_layout),
    'dead_literal': ('past', dead_literal),
    'current_rule': ('current', current_rule),
    'past_rule':    ('past', past_rule),
}
