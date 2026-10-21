import json,glob,re,sys
from collections import Counter
run=sys.argv[1]
SHOW={'rules':r'^Rules you confirmed on this level','moved':r'^Moved since this level started','colours':r'^Colours on the board not clicked','undone':r'was undone by the game','budget':r'^Budget bar','seen':r'^The current board was seen before','untried':r'^Not tried on this level','effects':r'presses? on this level\)','touch':r'^Same-colour objects that touch','prev':r'<previous_levels>|<cleared_level>'}
CITE={'rules':r'confirmed rules?|rules? (you|I) confirmed|harness note|\(established|ledger|the rules line',
 'moved':r'moved since|since (this|the) level started|not by an arrow key',
 'colours':r'not clicked|never clicked|unclicked|(haven.t|have not|not yet) (been )?clicked|colours on the board',
 'undone':r'\bundone\b|undid|reverted',
 'budget':r'budget bar|bar reading|not a budget death',
 'seen':r'seen before|seen-before|board was seen|same board as',
 'untried':r'not tried|untried|(haven.t|have not|never) tried|not yet tried',
 'effects':r'presses on this level|changed only the edge|changed cell colou?rs|moved cells \d|did nothing \d',
 'touch':r'touching|listed separately',
 'prev':r'recap|previous[_ ]levels|level[- ]\d+(.s)? (rule|recap|note|entry)|from level \d+'}
shown=Counter(); cited=Counter(); cited_any=Counter(); nreq=0
for f in sorted(glob.glob(f'{run}/*.req.jsonl')):
    last=''; pl=False
    for l in open(f):
        r=json.loads(l)
        for m in r['new']:
            if m['role']=='user' and m['idx']==r['n_msgs']-1 and 'Current state: step' in m['text']: last=m['text']
        if r.get('pl_len'): pl=True
        reas=(r.get('reasoning') or '')
        if not reas: continue
        nreq+=1
        for k in SHOW:
            pres = pl if k=='prev' else bool(re.search(SHOW[k],last,re.M))
            c=bool(re.search(CITE[k],reas,re.I))
            cited_any[k]+=c
            if pres:
                shown[k]+=1; cited[k]+=c
print(run,'requests with reasoning',nreq)
for k in SHOW: print('%-8s shown in %5d req, cited in %5d of them (%.0f%%); cited overall %5d'%(k,shown[k],cited[k],100*cited[k]/max(1,shown[k]),cited_any[k]))
