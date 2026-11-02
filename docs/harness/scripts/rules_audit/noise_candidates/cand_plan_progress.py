"""Plan and progress concepts, final round. fn(line) -> True drops the line.
Line fields used: text, next1, level. No game name, colour, object word or specific coordinate."""
import re
COORD = r"\(-?\d+\s*,\s*-?\d+\)"
HAS_COORDS = re.compile(COORD + r"|\brows?\s+\d+|\bcols?\s+\d+", re.I)
def t(l): return l['text']
def n(l): return l['next1'] or ''
def main_clause(s):
    """Text with asides removed, nested ones included."""
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", s)
    return s

# MOMENT: the main clause says what is true right now, not what is always true.
# An aside does not count: "dots move rigidly (now at cols 15-23)" still states a rule.
# "now confirmed" is about the check, not the board, so it does not count either.
MOMENT = re.compile(r"\bnow\b(?!\s+(confirmed|verified))|\bcurrently\b|\bremain(s|ing)?\b|\bfresh\b|\binitial\b|\bat this (step|point|moment)\b", re.I)
moment = lambda l: bool(MOMENT.search(main_clause(t(l))))

# LOC: the sentence's subject is a location, so it reports a cell rather than an object or a rule.
LOC_STRICT = re.compile(r"^\W*(then\s+)?" + COORD, re.I)
LOC_WIDE = re.compile(r"^\W*(then\s+|confirmed\s+for\s+|confirmed:\s*)?(" + COORD + r"|row\s+\d+|at\s+steps?\s+\d+)", re.I)

# STEP: tied to a numbered step of the run.
STEP = re.compile(r"\bsteps?\s+\d+|\bat this step\b", re.I)
step_main = lambda l: bool(STEP.search(main_clause(t(l))))
step_any = lambda l: bool(STEP.search(t(l)))
# Same idea with other clock words; tested to see if the concept widens.
CLOCK = re.compile(r"\b(steps?|frames?|batch|turn|round|attempt)\s+\d+|\bat this step\b", re.I)
clock_main = lambda l: bool(CLOCK.search(main_clause(t(l))))

# SELF_CHECK: compares the agent's own model or prediction with what it saw, and the evidence is coordinates.
SELF_WORDS = re.compile(r"\b(model|prediction|predicted|simulat\w+|expected|match(es|ing)|observed|obs|actual|real)\b", re.I)
self_check = lambda l: bool(SELF_WORDS.search(t(l)) and HAS_COORDS.search(t(l)))
# The same, using only the words of the concept's name. Tests whether "obs" and "real" carry it.
SELF_NARROW = re.compile(r"\b(model|prediction|predicted)\b", re.I)
self_narrow = lambda l: bool(SELF_NARROW.search(t(l)) and HAS_COORDS.search(t(l)))

# BARE_OBJ: "confirmed" attaches to a token rather than a stated claim.
BARE_OBJ = re.compile(r"\(\w+\s+confirmed\)|\bstep\s+\d+\s+confirmed|\[confirmed\b|\bconfirmed earlier\b", re.I)
bare_obj = lambda l: bool(BARE_OBJ.search(t(l)))

# LAYOUT_CHECK: says one level's layout was checked. Dead once that level is cleared.
LAYOUT = r"(state|map|geometry|layout|regions?)"
LAYOUT_CHECK = re.compile(r"\b" + LAYOUT + r"\s+(is\s+)?confirmed|\bconfirmed\s+(the\s+)?([\w-]+\s+){0,2}?" + LAYOUT + r"\b", re.I)
layout_check = lambda l: bool(LAYOUT_CHECK.search(t(l)))

# NEXT_ACT: next sentence turns to tooling or a labelled plan: the check closes with no consequence drawn.
NEXT_ACT = re.compile(r"^\W*(now,?\s+)?let me\b|^\W*(route|plan|target)\b[^:]{0,40}:", re.I)
next_act = lambda l: bool(NEXT_ACT.search(n(l)))
# Controls for the same idea: other ways of moving on.
NEXT_NOW = re.compile(r"^\W*now\b", re.I)
NEXT_STATE = re.compile(r"^\W*(current|state)\b", re.I)
NEXT_ACK = re.compile(r"^\W*(ok|okay|good|great|nice)\b", re.I)

# PLAN_STEP: opener that marks one step of a sequence. BRANCH: an alternative.
PLAN_STEP = re.compile(r"^\W*(then|next|first|finally|afterwards?|after that|step\s+\d+)\b", re.I)
BRANCH = re.compile(r"\bor\s+" + COORD + r"|\beither\b|\balternativ", re.I)

# FIRST_PERSON: first-person record of an action.
FIRST_PERSON = re.compile(r"\b(i|we)\s+(have\s+|'ve\s+|now\s+)?(moved|pressed|clicked|matched|tested|tried|ran|checked|confirmed|poured|did)\b|\bi've\b", re.I)

# EVENT_CITE: evidence cited as a numbered event.
EVENT = re.compile(r"\b(actions?|attempts?|try|tries)\s+\d+|\b\d+(st|nd|rd|th)\s+\w+", re.I)

# PAST_EVENT: a past-tense verb in the main clause plus coordinates: one event recorded.
def past_event(l):
    m = main_clause(t(l))
    return bool(HAS_COORDS.search(m)) and any(w.lower() not in ('confirmed','verified','red') for w in re.findall(r"\b[a-zA-Z]{3,}ed\b", m))

# NEXT_BULLET: the line is an item of a list.
NEXT_BULLET = re.compile(r"^\s*[-*]\s")

# RESTATES: most content words already appeared in an earlier line of the game.
WORD = re.compile(r"[a-z]{4,}")
STOP = {'confirmed','that','this','with','from','then','have','were','when','also','only'}
def bag(s): return set(WORD.findall(s.lower())) - STOP
def restates(l):
    b = bag(t(l))
    return len(b) >= 3 and any(len(b & bag(p['text'])) / len(b) >= 0.6 for p in l['prior'])

# OTHER_LEVEL: mentions a level number other than the line's own: the check happened elsewhere.
def other_level(l):
    return any(int(x) != l['level'] for x in re.findall(r"\blevel[- ](\d+)", t(l), re.I))

heading = lambda l: t(l).rstrip().endswith(':')
loc_strict = lambda l: bool(LOC_STRICT.search(t(l)))
loc_wide = lambda l: bool(LOC_WIDE.search(t(l)))

def any_of(*fs): return lambda l: any(f(l) for f in fs)

CANDIDATES = {
 # passing and recommended
 'moment':        ('both', moment),
 'loc_strict':    ('current', loc_strict),
 'loc_wide':      ('past', loc_wide),
 'step_any':      ('past', step_any),
 'self_check':    ('both', self_check),
 'bare_obj':      ('both', bare_obj),
 'layout_check':  ('past', layout_check),
 'union_current': ('current', any_of(moment, loc_strict, self_check)),
 'union_past':    ('past', any_of(moment, loc_wide, step_any, self_check, bare_obj, layout_check)),
 'union_current_no_self': ('current', any_of(moment, loc_strict)),
 'union_past_no_self':    ('past', any_of(moment, loc_wide, step_any, bare_obj, layout_check)),
 # passing but not recommended
 'next_act':      ('both', next_act),
 'other_level':   ('both', other_level),
 'step_main':     ('both', step_main),
 'heading':       ('both', heading),
 # failed
 'loc_wide_cur':  ('current', loc_wide),
 'clock_main':    ('both', clock_main),
 'self_narrow':   ('both', self_narrow),
 'plan_step':     ('both', lambda l: bool(PLAN_STEP.search(t(l)))),
 'branch':        ('both', lambda l: bool(BRANCH.search(t(l)))),
 'first_person':  ('both', lambda l: bool(FIRST_PERSON.search(t(l)))),
 'event_cite':    ('both', lambda l: bool(EVENT.search(t(l)))),
 'past_event':    ('both', past_event),
 'next_now':      ('both', lambda l: bool(NEXT_NOW.search(n(l)))),
 'next_state':    ('both', lambda l: bool(NEXT_STATE.search(n(l)))),
 'next_ack':      ('both', lambda l: bool(NEXT_ACK.search(n(l)))),
 'next_bullet':   ('both', lambda l: bool(NEXT_BULLET.search(n(l)))),
 'restates':      ('both', restates),
}
