import json,sys,re
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
G='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/memv2-analysis/games'
game=sys.argv[1]
import glob
p=glob.glob(f'{R}/{game}*_p0_requests.jsonl')[0]
L=[json.loads(l) for l in open(p)]
# map calls -> transcript times: parse game md: '## step N | action M | HH:MM:SS' then '### thinking (call k)'
md=open(glob.glob(f'{G}/{game}*.md')[0]).read().split('\n')
calltime={}; cur=None
for line in md:
    m=re.match(r'## step (\d+) \| action (\d+) \| (\S+)',line)
    if m: cur=(int(m.group(1)),int(m.group(2)),m.group(3))
    m=re.match(r'### thinking \(call (\d+)\)',line)
    if m: calltime[int(m.group(1))]=cur
call=0; prev=None
for d in L:
    if d['event']=='request':
        nm=len(d['messages']); roles=[m['role'] for m in d['messages']]
        trim = 'system' in roles
        continue
    call+=1
    u=d.get('usage') or {}
    pt=u.get('prompt_tokens',0); ct=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
    flag='  <-- low cache' if pt and ct/pt<0.3 else ''
    if flag or (len(sys.argv)>2 and call in map(int,sys.argv[2].split(','))):
        print(call, calltime.get(call), 'step',d.get('analysis_step'),'act',d.get('action'),'prompt',pt,'cached',ct,'gen',u.get('completion_tokens'),'finish',d.get('finish_reason'),'resent_system' if trim else '', flag)
