import lzma,json,os,re,sys,csv,collections,glob
R='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle'
rows=[]
for d in sorted(glob.glob(R+'/*/v*')):
    p=None
    for n in ('benchmark.json.xz','benchmark.json'):
        if os.path.exists(d+'/'+n): p=d+'/'+n
    if not p: continue
    try:
        b=json.load(lzma.open(p) if p.endswith('xz') else open(p))
    except Exception as e: continue
    gr=b.get('game_runs') or []
    if not gr: continue
    k=d.split('/')[-2]+'/'+d.split('/')[-1]
    lv=0;win=0;tok=0;acts=0;ng=0;nreq=len(glob.glob(d+'/*_requests.jsonl*'))
    for g in gr:
        ng+=1
        lc=g.get('levels_completed') or 0; lv+=lc
        if lc>=g.get('number_of_levels',99) and lc>0: win+=1
        m=re.search(r'tokens=(\d+)',g.get('solver_note') or '')
        if m: tok+=int(m.group(1))
        acts+=len(g.get('history') or [])
    rows.append(dict(run=k,games=ng,wins=win,levels=lv,tokens=tok,actions=acts,req_files=nreq,start=b.get('start_time','')[:10],
      score=sum(g.get('final_score') or 0 for g in gr)/ng*100/1 if ng else 0))
json.dump(rows,open('inventory.json','w'),indent=1)
full=[r for r in rows if r['games']>=25]
print('runs with benchmark',len(rows),'full25',len(full))
for name,s in (('all',rows),('full25',full)):
    print(name,'levels',sum(r['levels'] for r in s),'wins',sum(r['wins'] for r in s),'tokens',sum(r['tokens'] for r in s),'actions',sum(r['actions'] for r in s),'reqfiles',sum(r['req_files'] for r in s))
print('with reqfiles',sum(1 for r in rows if r['req_files']), 'levels in those',sum(r['levels'] for r in rows if r['req_files']),'wins',sum(r['wins'] for r in rows if r['req_files']),'tokens',sum(r['tokens'] for r in rows if r['req_files']),'reqfiles',sum(r['req_files'] for r in rows))
print('by 2h-ish full25 sorted by wins')
for r in sorted(full,key=lambda r:-r['wins'])[:12]: print(r)
