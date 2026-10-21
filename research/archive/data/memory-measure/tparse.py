"""Parse Duck-harness transcripts into per-model-call records. Light, single pass."""
import re, json, sys, os, glob
ROOT='/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs'
OUT='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memory-measure/parsed'
HDR=re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\S+) \| (\S+) ---$',re.M)
MARK=re.compile(r'^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|ASSISTANT|TOOL CALL: \w+|TOOL RESULT: \w+|ANALYZER STATUS)\]\n?',re.M)

def parse(path):
    t=open(path,errors='replace').read()
    parts=HDR.split(t)
    turns=[]
    # parts: [pre, step, action, time, agent, body, step, ...]
    for k in range(1,len(parts),5):
        step,action,tm,agent,body=int(parts[k]),int(parts[k+1]),parts[k+2],parts[k+3],parts[k+4]
        segs=[]
        ms=list(MARK.finditer(body))
        for i,m in enumerate(ms):
            end=ms[i+1].start() if i+1<len(ms) else len(body)
            segs.append((m.group(1),body[m.end():end]))
        turn={'step':step,'action':action,'time':tm,'level':None,'cur_step':None,'user_head':'','executed':None,'calls':[],'game_over_prompt':False,'prompt_tokens':None}
        cur=None; last_pt=None
        for name,txt in segs:
            if name=='USER PROMPT':
                turn['user_head']=txt[:700]
                m=re.search(r'Current state: step (\d+), level (\d+)',txt)
                if m: turn['cur_step'],turn['level']=int(m.group(1)),int(m.group(2))
                m=re.search(r'The code executed (\d+) action',txt)
                if m: turn['executed']=int(m.group(1))
                if re.search(r'game.?over|GAME_OVER|was reset|RESET',txt[:900]): turn['game_over_prompt']=True
                turn['retained_fn_line']=bool(re.search(r'Your retained functions',txt))
                turn['has_memory_in_user']='<memory>' in txt
                turn['user_len']=len(txt)
            elif name=='ANALYZER STATUS':
                m=re.search(r'prompt_tokens (\d+)',txt)
                if m: last_pt=int(m.group(1)); turn['prompt_tokens']=last_pt
                if cur is not None and cur.get('pt') is None and m: cur['pt']=last_pt
            elif name=='MODEL RESPONSE META':
                cur={'finish':None,'content_chars':0,'reasoning_chars':0,'thinking':'','assistant':'','code':'','result':'','pt':last_pt}
                m=re.search(r'finish_reason: (\S+)',txt); cur['finish']=m.group(1) if m else None
                m=re.search(r'content_chars: (\d+)',txt); cur['content_chars']=int(m.group(1)) if m else 0
                m=re.search(r'reasoning_chars: (\d+)',txt); cur['reasoning_chars']=int(m.group(1)) if m else 0
                turn['calls'].append(cur)
            elif cur is not None and name=='THINKING': cur['thinking']=txt.strip()
            elif cur is not None and name=='ASSISTANT': cur['assistant']=txt.strip()
            elif cur is not None and name.startswith('TOOL CALL'): cur['code']=txt.strip()
            elif cur is not None and name.startswith('TOOL RESULT'): cur['result']=txt.strip()[:6000]
        turns.append(turn)
    return turns

if __name__=='__main__':
    run=sys.argv[1]
    os.makedirs(OUT,exist_ok=True)
    for p in sorted(glob.glob(f'{ROOT}/{run}/transcripts/*_p0.txt')):
        g=os.path.basename(p).split('_')[0]
        outp=f'{OUT}/{run}__{g}.json'
        if os.path.exists(outp): continue
        turns=parse(p)
        json.dump(turns,open(outp,'w'))
        print(run,g,len(turns),sum(len(t['calls']) for t in turns),flush=True)
