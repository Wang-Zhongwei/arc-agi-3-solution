import json,hashlib,sys,os,glob,re
ROOT='/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs'
OUT='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memory-measure/parsed'
def txt(m):
    c=m.get('content')
    if isinstance(c,list):
        return ' '.join(p.get('text','') if isinstance(p,dict) and p.get('type')=='text' else ('[IMG]' if isinstance(p,dict) else str(p)) for p in c)
    return c or ''
def nimg(m):
    c=m.get('content')
    if isinstance(c,list): return sum(1 for p in c if isinstance(p,dict) and p.get('type')!='text')
    return 0
def mh(m): return hashlib.md5(json.dumps(m,sort_keys=True).encode()).hexdigest()[:10]
def run(run):
    for p in sorted(glob.glob(f'{ROOT}/{run}/*_p0_requests.jsonl')):
        g=os.path.basename(p).split('_')[0]
        outp=f'{OUT}/{run}__req__{g}.json'
        if os.path.exists(outp): continue
        rows=[]
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if d.get('event')!='request': continue
                ms=d['messages']
                hs=[mh(m) for m in ms]
                head=ms[1] if len(ms)>1 else {}
                ht=txt(head)
                mem=None
                if ht.startswith('<memory>'):
                    e=ht.find('</memory>')
                    mem=ht[:e+9]
                rows.append({'i':len(rows),'step':d['analysis_step'],'action':d['action'],'ridx':d['request_index_within_turn'],
                  'nmsg':len(ms),'chars':sum(len(txt(m))+len(json.dumps(m.get('tool_calls',''))) +len(m.get('reasoning_content') or '') for m in ms),
                  'nimg':sum(nimg(m) for m in ms),'head_hash':hs[1] if len(hs)>1 else None,'hs':hs,'mem':mem,
                  'roles':''.join(m['role'][0] for m in ms)})
        json.dump(rows,open(outp,'w'))
        print(run,g,len(rows),flush=True)
if __name__=='__main__': run(sys.argv[1])
