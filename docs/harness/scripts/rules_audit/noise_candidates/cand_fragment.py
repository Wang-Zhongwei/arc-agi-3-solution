"""Fragment / private-shorthand noise candidates (final, round 3). Designed on main-tuned-drafter v02 only.
Text-level language patterns: no game names, colours, object words or coordinates."""
import re

CONF = r"(?:confirmed|verified)"
def t(l): return l['text'].strip()

# Heading that lost its content: the sentence ends by promising what follows.
def heading(l): return t(l).endswith(':')

# Bare reference: the confirmation points at something gone. The pronoun must end the sentence,
# so "confirmed that the piece moves" (a stated claim) does not match.
BARE_RE = re.compile(r"\b" + CONF + r"\s+(it|that|this|earlier|above|before|previously)\W*$|\[\s*(also\s+)?" + CONF + r"[^\]]*\]", re.I)
def bare_ref(l): return bool(BARE_RE.search(t(l)))

# A single capital letter assigned or confirmed as if it were defined: "B = {...}", "(L confirmed)".
LABEL_RE = re.compile(r"(?<![\w.'\"])[A-Z][-+]?\s*=\s*\S|\(\s*[A-Z][-+]?\s+" + CONF + r"\b")
def letter_label(l): return bool(LABEL_RE.search(t(l)))

# The sentence opens with a coordinate, so its subject is a cell and not a thing.
COORD_OPEN = re.compile(r"^\W*\(\s*-?\d+\s*,\s*-?\d+\s*\)")
def coord_open(l): return bool(COORD_OPEN.match(t(l)))

# Union recommended for both phases.
def core(l): return heading(l) or bare_ref(l) or letter_label(l) or coord_open(l)

# Plan-step opener, under the bar alone.
STEP_OPEN = re.compile(r"^\W*(then|next|step\s+\d+|finally|after that)\b", re.I)
def plan_open(l): return bool(STEP_OPEN.match(t(l))) or coord_open(l)

# A model, state or map is named, then only a snapshot follows instead of the content.
SNAP = re.compile(r"\b(model|state|map)\b[^:]{0,20}" + CONF + r"\s*:[^:]*(\(\s*\d+\s*,\s*\d+\s*\)|\brows?\s+\d+)", re.I)
def label_snapshot(l): return bool(SNAP.search(t(l)))

# Short with no verb. Arrows and equals count as predicates.
VERB = re.compile(r"\b(is|are|was|were|be|been|has|have|had|do|does|did|can|can't|must|will|won't|don't|doesn't)\b|\b\w{3,}(ed|es|ing)\b", re.I)
def verbless(l): return not VERB.search(re.sub(r"\b" + CONF + r"\b", " ", t(l), flags=re.I))
def short_verbless(l):
    s = t(l)
    return verbless(l) and len(s.split()) <= 10 and not re.search(r"→|->|=", s)

# Symbol density: share of tokens that are real words, after removing the confirmation word.
RULE_WORD = re.compile(r"\b(cycle|mechanics?|rules?|conditions?|goal)\b", re.I)
def share(s):
    x = re.sub(r"\b" + CONF + r"\b:?", " ", s, flags=re.I).split()
    return len([w for w in x if re.fullmatch(r"\W*[A-Za-z]{3,}('s)?\W*", w)]) / max(1, len(x))
def sym_raw(l): return share(t(l)) < .45
def sym_dense(l): return share(t(l)) < .45 and not RULE_WORD.search(t(l))

# The evidence belongs to a different level.
OTHER = re.compile(r"\blevels?[- ](\d+)\b", re.I)
def other_level(l): return any(int(m.group(1)) != l['level'] for m in OTHER.finditer(t(l)))
def other_level_norule(l): return other_level(l) and not RULE_WORD.search(t(l))

# Concepts kept for the record of what failed.
NOSUBJ = re.compile(r"^\W*" + CONF + r"\s*(as\b|\(|\w+ly\b|twice\b|again\b|for\s*\()", re.I)
def no_subject(l): return bool(NOSUBJ.match(t(l)))
IDENT = re.compile(r"\b[A-Za-z]+\d+\b|\b[A-Za-z]+_[A-Za-z0-9_]+\b")
def ident(l): return bool(IDENT.search(t(l)))
def verbless_all(l): return verbless(l)
def dangling(l): return not re.search(r"[.!?:)\]✓\"']\s*$", t(l))
def next_list(l): return bool(re.match(r"^\s*(-|\*|\d+[.)])\s", l['next1'] or ''))
def next_colon(l): return (l['next1'] or '').strip().endswith(':')
def as_predicted(l): return bool(re.search(r"\bas (predicted|expected|assumed)\b", t(l), re.I))

CANDIDATES = {
    'core': ('both', core),
    'heading_colon': ('both', heading),
    'bare_ref': ('both', bare_ref),
    'letter_label': ('both', letter_label),
    'coord_open': ('both', coord_open),
    'plan_or_coord_open': ('both', plan_open),
    'label_snapshot': ('both', label_snapshot),
    'short_verbless': ('both', short_verbless),
    'sym_dense': ('both', sym_dense),
    'sym_raw': ('both', sym_raw),
    'other_level_norule': ('past', other_level_norule),
    'other_level': ('both', other_level),
    'core_or_sym': ('past', lambda l: core(l) or sym_dense(l)),
    'core_or_snapshot': ('both', lambda l: core(l) or label_snapshot(l)),
    'core_or_snap_or_sym': ('past', lambda l: core(l) or label_snapshot(l) or sym_dense(l)),
    'core_or_snap_or_verbless': ('current', lambda l: core(l) or label_snapshot(l) or short_verbless(l)),
    'x_no_subject': ('both', no_subject),
    'x_ident': ('both', ident),
    'x_verbless_all': ('both', verbless_all),
    'x_dangling': ('both', dangling),
    'x_next_list': ('both', next_list),
    'x_next_colon': ('both', next_colon),
    'x_as_predicted': ('both', as_predicted),
}
