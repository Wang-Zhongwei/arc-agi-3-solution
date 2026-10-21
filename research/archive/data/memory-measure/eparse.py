import json,glob,os,hashlib,sys
ROOT='/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs'
OUT='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memory-measure/parsed'
def hsh(rows):
    return hashlib.md5(b''.join(bytes(r) for r in rows)).hexdigest()[:12]
def run(run='fz-ledger-on-trim-r2'):
    for p in sorted(glob.glob(f'{ROOT}/{run}/artifacts/*_events.jsonl')):
        g=os.path.basename(p).split('_')[0]
        outp=f'{OUT}/ev__{run}__{g}.json'
        if os.path.exists(outp): continue
        rows=[]
        with open(p) as f:
            for line in f:
                e=json.loads(line)
                if e['type'] not in('action','initial'): continue
                b=e['board']
                interior=[r[4:-4] for r in b[4:-4]]
                d={k:e.get(k) for k in('type','action_num','analysis_step','action_display','action_name','level','state','game_over','level_completed','board_changed','run_complete','batch_index','batch_size','score')}
                d['ih']=hsh(interior); d['fh']=hsh(b)
                # bottom row / top band hud signature (count of non-background in last 4 rows)
                rows.append(d)
        json.dump(rows,open(outp,'w'))
        print(g,len(rows),sum(1 for r in rows if r['game_over']),flush=True)
if __name__=='__main__': run()
