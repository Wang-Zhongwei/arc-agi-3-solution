import re, sys, glob, os, json, collections
sys.path.insert(0, sys.argv[1])
from inference.agent import confirmed_rules as cr
run_dir, out = sys.argv[2], sys.argv[3]
HDR = re.compile(r'^(\[(THINKING|TOOL CALL: \w+|TOOL RESULT: \w+|ASSISTANT|ANALYZER STATUS|USER PROMPT|SYSTEM PROMPT|MODEL RESPONSE META)\]|--- analysis_step=(\d+).*---)\s*$')
LVL = re.compile(r'Current state: step (\d+), level (\d+)\.')
CONSEQ = re.compile(r"^(so\b(?! (now|let|i'll|i will|next))|this means|that means|which means|i\.e\.|therefore|thus|hence|in other words|meaning\b|→)", re.I)
LIST = re.compile(r"^\s*([-*•]|\d+[.)]\s)")
DOUBT = re.compile(r"^(but\b|however|wait|actually|although|though|unless|except|hold on|no[,.!]|or\b|hmm+,?\s*(but|wait|no|that|why|what|actually))", re.I)
MOVE = re.compile(r"^(now\b|next\b|let me|let's|i'll|i will|i need|then\b|ok\b|okay|good\b|great|alright|so (now|let|i'll|i will|next))", re.I)
def next_class(s):
    if s is None: return 'end'
    t = s.strip()
    if LIST.match(t): return 'list'
    t2 = re.sub(r'^\W+', '', t)
    if DOUBT.match(t2): return 'doubt'
    if MOVE.match(t2): return 'move_on'
    if CONSEQ.match(t2): return 'consequence'
    return 'other'
def drop_reason(raw):
    s = cr._LEAD.sub('', raw).strip(); p = s.replace('’', "'")
    if '?' in p: return 'question'
    if cr._HEDGED.search(p) or cr._REFUTED_EXTRA.search(p): return 'hedge_or_plan'
    if cr._SUPPORTS.search(p): return 'says_confirm_or_support'
    if cr._bare_verdict(p): return 'bare_verdict'
    if len(s.split()) < cr.MIN_WORDS: return 'short'
    if not cr._informative(cr._tidy(s)): return 'low_information'
    return None
claims = []
for f in sorted(glob.glob(os.path.join(run_dir, 'transcripts', '*.txt'))):
    game = os.path.basename(f).split('-')[0]
    sec=None; buf=[]; start=0; level=1; step=0; astep=None; maxlvl=1
    def flush():
        global level, step, maxlvl
        text='\n'.join(buf)
        if sec == 'USER PROMPT':
            m = LVL.findall(text)
            if m: step, level = int(m[-1][0]), int(m[-1][1]); maxlvl = max(maxlvl, level)
        elif sec == 'THINKING':
            segs = cr._segments(text, True)
            for i, piece in enumerate(segs):
                nxt = [s.strip() for s in segs[i+1:i+3]]
                base = dict(game=game, level=level, step=step, analysis_step=astep, transcript=os.path.abspath(f),
                            thinking_line=start, raw=piece.strip()[:900], next1=nxt[0][:600] if nxt else None,
                            next2=nxt[1][:600] if len(nxt) > 1 else None, next_class=next_class(nxt[0] if nxt else None))
                kinds = cr.classified_sentences(piece, confirmed=True, refuted=True)
                if kinds:
                    for kind, shown in kinds:
                        claims.append({**base, 'kind': kind, 'in_section': True, 'shown': shown})
                elif cr._REFUTED.search(piece.replace('’', "'")):
                    claims.append({**base, 'kind': 'refuted', 'in_section': False, 'shown': cr._tidy(cr._LEAD.sub('', piece).strip())[:500],
                                   'dropped_by': drop_reason(piece)})
    for n, line in enumerate(open(f, encoding='utf-8', errors='replace'), 1):
        m = HDR.match(line.rstrip('\n'))
        if m:
            flush(); buf=[]; sec=m.group(2); start=n
            if m.group(3): astep = int(m.group(3))
        else: buf.append(line.rstrip('\n'))
    flush()
    for c in claims:
        if c['game'] == game: c['max_level_seen'] = maxlvl
for i, c in enumerate(claims): c['id'] = f"{c['game']}-{i:03d}"
json.dump(claims, open(out, 'w'), indent=1)
k = collections.Counter((c['kind'], c['in_section']) for c in claims)
print('claims', len(claims), dict(k))
print('next_class (in section):', collections.Counter((c['kind'], c['next_class']) for c in claims if c['in_section']))
print('refuted dropped_by:', collections.Counter(c.get('dropped_by') for c in claims if not c['in_section']))
g = collections.Counter(c['game'] for c in claims); print('per game:', dict(sorted(g.items(), key=lambda x: -x[1])))
