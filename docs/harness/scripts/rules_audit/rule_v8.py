"""Frozen candidate rule for pinning confirmed claims (designed on main-tuned-drafter v02, 2026-11-01).

phase='current': the claim's level is the level being played (not yet cleared).
phase='past'   : the claim's level was cleared (the model is on a later level).
"""
import re
# H: the next sentence hedges / doubts / re-checks.
HEDGE = re.compile(r"^\W*(hmm+|wait|but|however|unless|actually|although|though|maybe|perhaps|or\b|not sure|hold on|let me (re-?check|double|verify|re-?read)|is that|did\b|does\b|was\b|could|might|need to (check|verify))|\?\s*$|\b(not sure|unclear|double[- ]check|re-?check|verify|uncertain|suspicious|contradict|doesn't (match|fit)|wrong)\b", re.I)
# M: a generic reference to an earlier artifact is called confirmed without restating its content.
ARTIFACT = r"(plan|model|map|board|state|layout|schedule|legend|mapping|shapes?|experiment|search|analysis|hypothesis|theory|table|picture|reading|interpretation|origin|geometry|order|sequence|solution|answer|prediction|setup|configuration|structure|assignment|rule set|facts?)"
META = re.compile(r"^\W*(so|ok|okay|good|great|yes|and|also)?[,\s]*(the|my|our|this|that|these|those)?\s*([\w-]+\s){0,2}?" + ARTIFACT + r"s?\b[^.:;]{0,30}?\b(is|are|was|were|has been|have been)?\s*(now\s+|also\s+|again\s+|fully\s+)?(confirmed|verified)\b(?!\s*:)[^:]{0,40}$"
                  r"|^\W*(i|we)\s+(have\s+|'ve\s+)?(now\s+)?(confirmed|verified)\s+(the|my|our)\s+([\w-]+\s){0,2}?" + ARTIFACT + r"s?\b[^:]{0,40}$"
                  r"|^\W*(the|my|our)\s+([\w-]+\s){0,2}?" + ARTIFACT + r"s?\s+confirmed\s+(the|that|it)\b[^:]{0,40}$", re.I)
# A: agitated confirmation (uppercase CONFIRMED or two or more check marks).
def agitated(s): return 'CONFIRMED' in s or s.count('✓') >= 2
# I: impossibility / dead-end verdicts.
IMPOSS = re.compile(r"unsolvable|impossib|dead end|\btrap\b|no exit|no way|can't be (reached|solved)|cannot be (reached|solved)|must be wrong|is wrong", re.I)

def reasons(shown, next1):
    return {'H': bool(HEDGE.search(next1 or '')), 'M': bool(META.search(shown)), 'A': agitated(shown), 'I': bool(IMPOSS.search(shown))}

def keep(shown, next1, phase):
    f = reasons(shown, next1)
    if f['H'] or f['M']:
        return False
    if phase == 'current' and (f['A'] or f['I']):
        return False
    return True
