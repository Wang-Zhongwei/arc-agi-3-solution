import json,glob,os,re,collections
from trimsim import predict_trims
from m2_posttrim import CUE_RE,DEF
def run_stats(run,use_pred=True):
    post=collections.Counter(); base=collections.Counter(); pre=collections.Counter()
    ntr=0
    for p in sorted(glob.glob(f'parsed/{run}__*.json')):
        b=os.path.basename(p)
        if '__req__' in b or b.startswith('ev__'): continue
        T=json.load(open(p)); calls=[c for t in T for c in t['calls']]
        pt=predict_trims(T); ntr+=len(pt)
        inpost=set(x for k in pt for x in range(k,k+3))
        inpre=set(x for k in pt for x in range(k-3,k))
        defined=set()
        for k,c in enumerate(calls):
            names=set(DEF.findall(c['code'])); redef=len(names&defined)>0; defined|=names
            tgt = post if k in inpost else (pre if k in inpre else base)
            th=c['thinking']+'\n'+c['assistant']
            tgt['calls']+=1; tgt['chars']+=len(c['thinking'])
            for name,rx in CUE_RE.items():
                if name=='memory_word': continue
                if rx.search(th): tgt[name]+=1
            tgt['lost']+= bool(CUE_RE['lost_memory'].search(th))
            tgt['redef']+=redef
    return ntr,post,pre,base
for run in ['fz-ledger-on-trim','fz-ledger-on-trim-r2','fz-arms-v4-control','fz-arms-v5-temp06','fz-arms-v6-audit60k','fz-streams14']:
    ntr,post,pre,base=run_stats(run)
    f=lambda c,k: c[k]/max(1,c['calls'])
    print(f"{run:22s} predicted trims {ntr:3d} | post calls {post['calls']:3d} think chars/call post {post['chars']/max(1,post['calls']):6.0f} pre {pre['chars']/max(1,pre['calls']):6.0f} base {base['chars']/max(1,base['calls']):6.0f} | lost_memory cue post {f(post,'lost_memory'):.3f} base {f(base,'lost_memory'):.3f} | re_derive post {f(post,'re_derive'):.3f} base {f(base,'re_derive'):.3f} | re_verify post {f(post,'re_verify'):.3f} base {f(base,'re_verify'):.3f} | from_scratch post {f(post,'from_scratch'):.3f} base {f(base,'from_scratch'):.3f} | redef fn post {f(post,'redef'):.3f} base {f(base,'redef'):.3f}")
