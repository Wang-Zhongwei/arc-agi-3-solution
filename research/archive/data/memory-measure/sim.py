import json,sys,glob,os
import importlib.util
_sp=importlib.util.spec_from_file_location('memory_ledger','/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/ledger/ARC3-Inference/inference/agent/memory_ledger.py')
ML=importlib.util.module_from_spec(_sp); sys.modules['memory_ledger']=ML; _sp.loader.exec_module(ML)
MemoryLedger=ML.MemoryLedger
def calls_of(T):
    out=[]
    for ti,t in enumerate(T):
        for c in t['calls']:
            out.append((ti,t,c))
    return out
def simulate(T, render_at=None):
    """Returns list of rendered blocks AFTER each call index (state before next request)."""
    L=MemoryLedger(); blocks=[]
    lastlevel=None
    seen_turn=-1
    for k,(ti,t,c) in enumerate(calls_of(T)):
        if ti!=seen_turn:
            seen_turn=ti
            if t['level'] is not None and t['level']!=lastlevel:
                if lastlevel is not None: L.on_level_change(t['level'])
                else: L.level=t['level']
                lastlevel=t['level']
        L.absorb(c['assistant']); L.absorb(c['thinking']); L.note_think(c['thinking'])
        blocks.append(L.render())
    return blocks
if __name__=='__main__':
    run='fz-ledger-on-trim-r2'
    tot=match=0
    for p in sorted(glob.glob(f'parsed/{run}__req__*.json')):
        g=os.path.basename(p).split('__')[2][:-5]
        R=json.load(open(p)); T=json.load(open(f'parsed/{run}__{g}.json'))
        B=simulate(T)
        for k in range(1,len(R)):
            if R[k]['head_hash']!=R[k-1]['head_hash']:
                actual=R[k]['mem'] or ''
                sim=B[k-1] if k-1<len(B) else None
                tot+=1
                ok = (sim==actual)
                match+=ok
                if not ok and tot<8: print(g,k,'\nACT',actual[:600],'\nSIM',(sim or '')[:600])
    print('match',match,'of',tot)
