import json,glob,re,sys,importlib.util
from collections import Counter
sp=importlib.util.spec_from_file_location('cr','/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/wt/stall-fix/ARC3-Inference/inference/agent/confirmed_rules.py')
cr=importlib.util.module_from_spec(sp); sys.modules['cr']=cr; sp.loader.exec_module(cr)
run=sys.argv[1]
TL=re.compile(r'<this_level>\n(.*?)\n</this_level>',re.S)
n=Counter()
for f in sorted(glob.glob(f'{run}/*.req.jsonl')):
    rows=[json.loads(l) for l in open(f)]
    shown_at={}  # i -> list of rules shown
    cur=[]; pl=''
    allrules=[]
    for r in rows:
        for m in r['new']:
            if m['role']=='user' and m['idx']==r['n_msgs']-1 and 'Current state: step' in m['text']:
                cur=[]
                for b in TL.findall(m['text']):
                    for ln in b.split('\n'):
                        if ln.startswith('Rules you confirmed on this level: '):
                            cur=re.findall(r'"(.*?)"(?: \| |$)',ln[35:])
        if r.get('pl'): pl=r['pl']
        prevrules=[q for ln in pl.split('\n') if ln.startswith('  Rules you confirmed: ') for q in re.findall(r'"(.*?)"(?: \| |$)',ln[23:])]
        shown_at[r['i']]=(list(cur),prevrules)
        allrules.extend(cur)
    for idx,r in enumerate(rows):
        reas=r.get('reasoning') or ''
        if not reas: continue
        ss=cr._sentences(reas)
        cur,prev=shown_at[r['i']]
        if cur:
            n['req_with_rules']+=1
            hit=any(cr.overlap(s,q)>=0.5 for s in ss for q in cur)
            n['restates_shown']+=hit
            # control: rules of this game shown >=5 requests LATER and not now
            fut=[q for j in range(idx+5,min(len(rows),idx+30)) for q in shown_at[rows[j]['i']][0] if q not in cur]
            fut=list(dict.fromkeys(fut))[:6]
            if fut:
                n['ctl_req']+=1; n['ctl_hit']+=any(cr.overlap(s,q)>=0.5 for s in ss for q in fut)
        if prev:
            n['req_with_prev']+=1
            n['restates_prev']+=any(cr.overlap(s,q)>=0.5 for s in ss for q in prev)
        n['own_recap']+=bool(re.search(r'Level \d+ recap:',reas))
print(run,dict(n))
print('restate shown this-level rule: %.0f%%; control (rules shown 5-30 req later): %.0f%%; restate previous-level rule: %.1f%%'%(100*n['restates_shown']/n['req_with_rules'],100*n['ctl_hit']/max(1,n['ctl_req']),100*n['restates_prev']/max(1,n['req_with_prev'])))
