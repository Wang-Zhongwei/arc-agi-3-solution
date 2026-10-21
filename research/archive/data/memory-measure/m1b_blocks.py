import json,re,collections,statistics as st
from align import *
from m1_tags import tags_in
from sim import ML,simulate
tidy=ML._tidy
ENTRY=re.compile(r'<(world_model|hypothesis|level_note)([^>]*)>(.*?)</\1>',re.S)
FILL=re.compile(r'^(so|then|hmm+|but|wait|alternatively|actually|okay|ok|now|let|maybe|however|and|also|thus|therefore|since|if|that|this|it|hold on|oh|well|right|yes|no|perhaps|unless|first|next)\b',re.I)
def parse_block(b):
    out=[]
    for tag,attrs,body in ENTRY.findall(b or ''):
        st_=None
        if tag=='hypothesis':
            m=re.search(r'status="(\w+)"',attrs); st_=m.group(1) if m else None
        out.append((tag,st_,body.strip()))
    return out
def run_blocks(run,use_actual):
    res=[]
    for g in games(run) if use_actual else [p.split('__')[1][:-5] for p in sorted(__import__('glob').glob(f'parsed/{run}__*.json')) if '__req__' not in p and 'ev__' not in p]:
        T=json.load(open(f'parsed/{run}__{g}.json'))
        calls=[(ti,t,c) for ti,t in enumerate(T) for c in t['calls']]
        tagnotes=set(); allt=collections.Counter()
        for ti,t,c in calls:
            for src in (c['assistant'],c['thinking']):
                for tag,stt,body in tags_in(src):
                    if tag=='level_note': tagnotes.add(tidy(body))
        if use_actual:
            R=json.load(open(f'parsed/{run}__req__{g}.json'))
            blocks=[R[k]['mem'] for k in range(1,len(R)) if R[k]['head_hash']!=R[k-1]['head_hash']]
        else:
            from trimsim import predict_trims
            B=simulate(T); pt=predict_trims(T)
            blocks=[B[k-1] for k in pt if k-1<len(B)]
        for b in blocks:
            ents=parse_block(b)
            c=collections.Counter()
            c['chars']=len(b or '')
            for tag,stt,body in ents:
                if tag=='world_model': c['world']+=1
                elif tag=='hypothesis': c['conf' if stt=='confirmed' else 'ref']+=1
                else:
                    txt=re.sub(r'^level \d+:\s*','',body)
                    # is it a tag-written note?
                    istag=any(txt.startswith(n[:60]) or n.startswith(txt[:60]) for n in tagnotes)
                    if istag: c['note_tag']+=1
                    else:
                        c['note_auto']+=1
                        c['auto_fill']+=bool(FILL.match(txt))
                        c['auto_trunc']+=txt.endswith('...')
            c['entries']=len(ents); c['empty']=int(not ents)
            res.append((g,dict(c),b))
    return res
if __name__=='__main__':
    out={}
    for run,actual in [('fz-ledger-on-trim-r2',True),('fz-ledger-on-trim',False)]:
        res=run_blocks(run,actual)
        tot=collections.Counter()
        for g,c,b in res: tot.update(c)
        n=len(res)
        print(run,'blocks',n,'(actual)' if actual else '(simulated at predicted trims)')
        print('  empty ledgers at trim',tot['empty'],'; mean chars',round(tot['chars']/n),'median chars',st.median(c['chars'] for g,c,b in res),'max',max(c['chars'] for g,c,b in res),'p90',sorted(c['chars'] for g,c,b in res)[int(.9*(n-1))])
        print('  mean entries/block',round(tot['entries']/n,2),'world',round(tot['world']/n,2),'confirmed',round(tot['conf']/n,2),'refuted',round(tot['ref']/n,2),'level_note(tag)',round(tot['note_tag']/n,2),'level_note(auto)',round(tot['note_auto']/n,2))
        na=tot['note_auto']; print('  auto notes',na,'of level notes',na+tot['note_tag'],f'= {na/max(1,na+tot["note_tag"]):.2f}; auto starts w/ filler {tot["auto_fill"]/max(1,na):.2f}; truncated(...) {tot["auto_trunc"]/max(1,na):.2f}')
        ne=tot['entries']; print('  share of all ledger entries that are harness-auto-extracted think fragments',round(na/max(1,ne),2))
        print('  blocks with zero model-written content (no world/hyp/tag note):',sum(1 for g,c,b in res if c.get('world',0)+c.get('conf',0)+c.get('ref',0)+c.get('note_tag',0)==0),'of',n)
        out[run]=[(g,c) for g,c,b in res]
    json.dump(out,open('m1b.json','w'))
