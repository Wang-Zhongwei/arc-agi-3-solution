"""Critic's shortlist: survivors of the attack on the four designers' candidates. Text-only general language patterns.
fn(line) -> True drops the line. Uses only line['text']. Scored on dev run v02 only; written after reading its labels."""
import re

NUMBER = re.compile(r"\d")
# A location, in the form main_clause() leaves it: "<r,c>", "rows 3", "col 2".
MAIN_COORD = re.compile(r"<[-+]?\d+,[-+]?\d+>|\b(rows?|cols?|columns?)\s+\d+", re.I)
# The line says something happens or holds in general: a condition, a quantifier, an arrow, or a named input.
GENERAL = re.compile(r"→|\b(when|if|until|per|each|every|always)\b", re.I)
INPUT = re.compile(r"\b(click|press|mov|push)\w*", re.I)
# A past-tense event other than the claim word itself.
EVENT = re.compile(r"\b(?!confirm|verifi)\w{3,}ed\b", re.I)
# The line calls itself a rule.
NAMED_RULE = re.compile(r"\b(mechanics?|rules?)\b", re.I)

def text(l): return l['text'].strip()
def general(s): return bool(GENERAL.search(s) or INPUT.search(s))

def main_clause(s):
    """Text without asides, nested ones included. A bare (r,c) cell is kept: it is a value, not an aside."""
    s = re.sub(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)", r"<\1,\2>", s)
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", s)
    return s

def main_clause_keep_all(s):
    """Same spelling of cells as main_clause(), with asides left in: a location may sit inside an aside."""
    return re.sub(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)", r"<\1,\2>", s)

# --- FRAGMENT: the sentence lost its content. Both phases.
# A heading whose list was cut off, or a short sentence whose subject is only a cell.
CELL_OPEN = re.compile(r"^\W*(then\s+)?\(\s*-?\d+\s*,\s*\d+\s*\)", re.I)
def fragment(l):
    s = text(l)
    return s.endswith(':') or (bool(CELL_OPEN.search(s)) and len(s.split()) <= 12)

# --- SNAPSHOT: where something is, or what value it has, at one instant. Both phases.
# The marker and a location must be in the line, the marker in the main clause, and the line must not generalise.
MOMENT = re.compile(r"\bnow\b(?!\s+(confirmed|verified))|\bcurrently\b|\bremain(s|ing)?\b|\bfresh\b|\binitial\b", re.I)
LOCATED = re.compile(r"\b(is|are|was|were|occup(y|ies|ied))\s+(at\s+)?(rows?\s*\d|cols?\s*\d|<)", re.I)
AT_STEP = re.compile(r"\bat steps?\s+\d|^\W*step\s+\d+\s*:", re.I)
def snapshot(l):
    s = text(l)
    if general(s):
        return False
    m = main_clause(s)
    located_now = MOMENT.search(m) and MAIN_COORD.search(main_clause_keep_all(s))
    return bool(located_now or LOCATED.search(m) or AT_STEP.search(m))

# --- The rest are for cleared levels only. Readers want these lines while the level is live.
# DEAD_COORDS: a place is stated in the main clause and nothing is said to happen there.
# A word in -s right after the cell is its verb: "(2,2) turns", "at (4,4) controls".
CELL_VERB = re.compile(r">\s+[a-z]{3,}s\b")
def dead_coords(l):
    s = text(l)
    m = main_clause(s)
    return bool(MAIN_COORD.search(m)) and not CELL_VERB.search(m) and not general(s) and not EVENT.search(s) and not NAMED_RULE.search(s)

# DEAD_LAYOUT: one level's layout was checked, and it comes with numbers.
LAYOUT_WORD = r"(map|geometry|layout|regions?)"
LAYOUT = re.compile(r"\b" + LAYOUT_WORD + r"\s+(is\s+)?confirmed|\bconfirmed\s+(the\s+)?([\w-]+\s+){0,2}?" + LAYOUT_WORD + r"\b", re.I)
def dead_layout(l):
    s = text(l)
    return bool(LAYOUT.search(s)) and bool(NUMBER.search(s))

# DEAD_LITERAL: a private symbol set to a literal value, a set-builder formula, or a quoted grid drawing.
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
