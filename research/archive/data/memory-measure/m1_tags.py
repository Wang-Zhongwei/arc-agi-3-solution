import json,glob,os,re,sys,collections
from sim import ML
TAG=ML._TAG_RE; STATUS=ML._STATUS_RE; ALIAS=ML._STATUS_ALIASES; tidy=ML._tidy
EXAMPLES=['the blue block is the piece you steer','up moves the blue block up one cell','mouse on a wall opens it','level 2 needed the key before the door']
def is_example(body):
    b=' '.join(body.lower().split())
    return any(e in b for e in EXAMPLES)
def tags_in(text):
    out=[]
    for tag,attrs,body in TAG.findall(text or ''):
        if is_example(body): continue
        tag=tag.lower(); st=None
        if tag=='hypothesis':
            m=STATUS.search(attrs or ''); st=ALIAS.get(m.group(1).lower(),'open') if m else 'open'
        out.append((tag,st,' '.join(body.split())))
    return out
STOP=set('the a an of to in on is are was were it its and or for with that this as at by be from not no if then so but when will can has have had do does did i we you my our'.split())
def toks(s): return set(w for w in re.findall(r'[a-z0-9_]+',s.lower()) if w not in STOP and len(w)>1)
def jac(a,b):
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)
def games(run):
    out={}
    for p in sorted(glob.glob(f'parsed/{run}__*.json')):
        b=os.path.basename(p)
        if '__req__' in b: continue
        g=b.split('__')[1][:-5]
        out[g]=json.load(open(p))
    return out
def analyze_run(run):
    G=games(run); res={}
    for g,T in G.items():
        calls=[(ti,t,c) for ti,t in enumerate(T) for c in t['calls']]
        rec=[]  # (call_idx, level, src, tag, status, text)
        n_code=0
        for k,(ti,t,c) in enumerate(calls):
            lvl=t['level']
            for src,txt in (('content',c['assistant']),('think',c['thinking'])):
                for tag,st,body in tags_in(txt):
                    rec.append((k,lvl,src,tag,st,body))
            n_code+=len(tags_in(c['code']))
        res[g]={'calls':len(calls),'rec':rec,'code_tags':n_code,'turns':len(T),'calls_with_tag':len(set(r[0] for r in rec))}
    return res
if __name__=='__main__':
    for run in ['fz-ledger-on-trim','fz-ledger-on-trim-r2']:
        R=analyze_run(run)
        json.dump(R,open(f'm1_{run}.json','w'))
        tot=collections.Counter(); calls=0; zero=0
        for g,r in R.items():
            calls+=r['calls']
            c=collections.Counter()
            for k,l,src,tag,st,body in r['rec']:
                c[tag if tag!='hypothesis' else 'hyp_'+st]+=1; c['src_'+src]+=1
            tot.update(c)
            if not r['rec']: zero+=1
        print(run,'calls',calls,'tags',sum(len(r['rec']) for r in R.values()),dict(tot),'zero-tag games',zero,'code-tags',sum(r['code_tags'] for r in R.values()))
