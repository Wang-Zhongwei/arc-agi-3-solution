import json,re,collections,statistics
from align import *
import m3_revisit
CUES={
 'lost_memory':r"\b(i (don't|do not|can't|cannot|no longer) (remember|recall|have|see)|i (forgot|lost)|(earlier|previous|older) (context|history|turns?|messages?) (is|are|was|were|has been|have been) (gone|dropped|trimmed|truncated|lost|cleared)|(context|history) (was|has been|got) (trimmed|truncated|reset|dropped|cleared))",
 'from_scratch':r"\b(from scratch|start(ing)? over|start fresh|begin again|re-?start)\b",
 're_derive':r"\b(re-?(derive|derived|deriving|establish|learn|discover|determine|figure|understand)|figure out (again|what)|work out (again|what))",
 're_verify':r"\b(re-?(verify|check|confirm|examine|test|probe|inspect|read|visit)|double-?check|check again|verify again|test again|confirm again|look again)",
 'what_do_actions':r"\b(what (does|do|is) (each|every|the) (action|key|button|move|direction|arrow)|what the actions? (do|does|mean)|effects? of (each|the) action|which (action|key|button)s? (do|does|move|affect))",
 'memory_word':r"\b(memory|ledger|my notes|my earlier notes|notes? (say|says|said)|according to (my|the) notes)\b",
}
CUE_RE={k:re.compile(v,re.I) for k,v in CUES.items()}
DEF=re.compile(r'^\s*def\s+([A-Za-z_]\w*)\s*\(',re.M)
def sent_with(rx,text,width=160):
    m=rx.search(text)
    if not m: return None
    s=max(0,m.start()-width//2); e=min(len(text),m.end()+width//2)
    return text[s:e].replace('\n',' ')
def analyze(g):
    G=load(g); calls=G['calls']; R=G['R']
    acts,_=m3_revisit.analyze(g)
    act_by_call=collections.defaultdict(list)
    for a in acts:
        if a['call'] is not None: act_by_call[a['call']].append(a)
    # function definitions per call, with prior-defined check (game-level)
    defined=set(); redef={}
    for k,(ti,t,c) in enumerate(calls):
        names=set(DEF.findall(c['code']))
        redef[k]=len(names & defined); defined|=names
    ndef={k:len(set(DEF.findall(c['code']))) for k,(ti,t,c) in enumerate(calls)}
    def stats(ks):
        ks=[k for k in ks if 0<=k<len(calls)]
        out=collections.Counter()
        ex={}
        for k in ks:
            ti,t,c=calls[k]
            th=c['thinking']+'\n'+c['assistant']
            out['calls']+=1
            out['think_chars']+=len(c['thinking'])
            for name,rx in CUE_RE.items():
                n=len(rx.findall(th))
                if n: out['cue_'+name]+=1; out['cues_any']+= (name!='memory_word')
                if n and name not in ex: ex[name]=sent_with(rx,th)
            out['redef_calls']+= (redef[k]>0)
            out['def_calls']+=(ndef[k]>0)
            out['redef_fn']+=redef[k]
            for a in act_by_call.get(k,[]):
                out['actions']+=1
                if a.get('pair_repeat'): out['pair_repeat']+=1
                if a.get('forgotten'): out['pair_forgotten']+=1
        return out,ex
    res=[]
    for k in G['trims']:
        post,exp=stats(range(k,k+3)); pre,exb=stats(range(k-3,k))
        res.append({'g':g,'k':k,'post':dict(post),'pre':dict(pre),'ex':exp})
    # baseline all calls not in a post window (k..k+2) 
    in_post=set(x for k in G['trims'] for x in range(k,k+3))
    base,_=stats([k for k in range(len(calls)) if k not in in_post])
    return res,dict(base),len(calls)
if __name__=='__main__':
    allres=[]; base=collections.Counter(); ncalls=0
    for g in games():
        res,b,n=analyze(g); allres+=res; base.update(b); ncalls+=n
    json.dump({'res':allres,'base':dict(base)},open('m2_posttrim.json','w'))
    post=collections.Counter(); pre=collections.Counter()
    for r in allres:
        post.update(r['post']); pre.update(r['pre'])
    def rate(c,key,den='calls'): return c.get(key,0)/max(1,c[den])
    print('trims',len(allres))
    print('post calls',post['calls'],'pre calls',pre['calls'],'base calls',base['calls'])
    for key in ['think_chars','cues_any','cue_lost_memory','cue_from_scratch','cue_re_derive','cue_re_verify','cue_what_do_actions','cue_memory_word','redef_calls','redef_fn','def_calls']:
        print(f'{key:22s} post/call {rate(post,key):8.3f}  pre/call {rate(pre,key):8.3f}  base/call {rate(base,key):8.3f}')
    for key in ['actions','pair_repeat','pair_forgotten']:
        print(f'{key:22s} post {post[key]} ({post[key]/max(1,post["actions"]):.3f}) pre {pre[key]} ({pre[key]/max(1,pre["actions"]):.3f}) base {base[key]} ({base[key]/max(1,base["actions"]):.3f})')
