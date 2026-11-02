"""Round 3 (final): momentary-state-snapshot candidates. General language patterns only.
The separator is not "has coordinates" but "reports a location or value and states no action, condition or cause"."""
import re
# A location read-out: (r,c) pair, "rows 3-11", "cols 15", "col 2", "row 4".
COORD = re.compile(r"\(-?\d+\s*,\s*[+-]?\d+\)|\b(rows?|cols?|columns?)\s*\d+", re.I)
# Something happens or is conditional: connectives, arrows, the common input verbs.
CAUSE = re.compile(r"→|\b(when|if|while|after|per|so|by|click\w*|press\w*|mov(?:e|es|ed|ing)|push\w*|drag\w*)\b", re.I)
# Any past-tense event verb other than the claim word itself.
EVENT_ED = re.compile(r"\b(?!confirm|verifi)\w{3,}ed\b", re.I)
# Tied to one instant of the run, by step number only.
STEP = re.compile(r"\b(at|during|in) (this )?steps?\b|\bsteps? \d+\s*(-|and)\s*\d+|^\W*step \d+\b", re.I)
# Same concept widened to any tick word.
TICK = re.compile(r"\b(at|in|after|during)\s+(this\s+)?(steps?|batch|frames?|turns?|actions?)\b(\s*\d+)?|\bsteps? \d+\s*(-|and)\s*\d+|^\W*step \d+\b", re.I)
# "Right now" words.
NOW = re.compile(r"\b(now|currently|still|remains?|fresh|initial)\b", re.I)
NOW_NO_START = re.compile(r"\b(now|currently|still|remains?)\b", re.I)
NOW_WITH_START = re.compile(r"\b(now|currently|still|remains?|fresh|initial|start\w*)\b", re.I)
# Subject + bare location or value predicate.
LOCATIVE = re.compile(r"\b(is|are|was|were|now|remains?|occup(?:y|ies|ied))\s+(at\s+)?(rows?\s*\d|cols?\s*\d|\(\d)", re.I)
# Opens with a bare cell: a piece of a list or plan.
LEADCOORD = re.compile(r"^\W*(then\s+|and\s+|also\s+|next\s+)?\(-?\d+\s*,\s*\d+\)", re.I)
# A check against the agent's own model or prediction.
MODEL = re.compile(r"\b(model|prediction|predicted|expected)\b", re.I)
# A dump of values: quoted grid text, three or more short list items, or two or more key=value pairs.
DUMP = re.compile(r"[\"'][^\"']*/[^\"']*[\"']|(\b\w{1,2}\b,\s*){3,}|(\b\w+\s*[:=]\s*[^,;]{1,8}[,;]\s*){2,}", re.I)
# A bare count right after the claim word.
COUNT = re.compile(r"^\W*confirmed:?\s+(only\s+)?(\d+|one|two|three|four|five)\b", re.I)
# A value going from one number to another.
TRANSITION = re.compile(r"\d+\s*(→|->)\s*\d+")

def t(l): return l['text']
def coord_nocause(l): return bool(COORD.search(t(l))) and not CAUSE.search(t(l))
def now_coord(l): return coord_nocause(l) and bool(NOW.search(t(l)))
def locative(l): return coord_nocause(l) and bool(LOCATIVE.search(t(l)))
def coord_noevent(l): return coord_nocause(l) and not EVENT_ED.search(t(l))
def step(l): return bool(STEP.search(t(l)))
def tick(l): return bool(TICK.search(t(l)))
def lead(l): return bool(LEADCOORD.search(t(l)))
def dump(l): return bool(DUMP.search(t(l)))
def core(l): return now_coord(l) or locative(l) or step(l) or lead(l)
def past_rule(l): return core(l) or coord_noevent(l) or dump(l)
CANDIDATES = {
 # components
 'now_coord_nocause': ('both', now_coord),
 'locative_nocause':  ('both', locative),
 'step_strict':       ('both', step),
 'lead_coord':        ('both', lead),
 'coord_no_event':    ('both', coord_noevent),
 'value_dump':        ('both', dump),
 # proposed rules
 'SNAP_CORE':         ('both', core),
 'SNAP_PAST':         ('past', past_rule),
 # robustness of the "right now" word list
 'core_no_fresh':     ('current', lambda l: (coord_nocause(l) and bool(NOW_NO_START.search(t(l)))) or locative(l) or step(l) or lead(l)),
 'core_with_start':   ('current', lambda l: (coord_nocause(l) and bool(NOW_WITH_START.search(t(l)))) or locative(l) or step(l) or lead(l)),
 # concepts that do not separate
 'FAIL_coord_no_cause': ('both', coord_nocause),
 'FAIL_tick_general':   ('both', tick),
 'FAIL_model_check':    ('both', lambda l: bool(MODEL.search(t(l)))),
 'FAIL_count':          ('both', lambda l: bool(COUNT.search(t(l)))),
 'FAIL_transition':     ('both', lambda l: bool(TRANSITION.search(t(l)))),
 'FAIL_no_cause_any':   ('both', lambda l: not CAUSE.search(t(l))),
}
