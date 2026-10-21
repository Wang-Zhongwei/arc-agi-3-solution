import sys, json, lzma, glob, os, re
from collections import Counter, defaultdict
run = sys.argv[1]
pats = {
  'segmentation': r'\.segmentation',
  'frame_diff(': r'frame_diff\(',
  '.ascii': r'\.ascii',
  'previous_frame': r'previous_frame',
  'history[': r'history\[',
  'transitions/last_transition': r'transitions|last_transition',
  'animation_for/last_animation/animations': r'animation_for|last_animation|\banimations\b',
  'action(': r'\baction\(',
  'boundary': r'boundary',
  'adjacency': r'adjacency',
  'children': r"children",
  'hash': r"hash",
  'BFS/search': r'deque|bfs|BFS|heapq',
}
tot = Counter(); games = {}
first_turn = Counter(); first_turn_n = 0
per_game_seg = {}
usage = defaultdict(list)
for f in sorted(glob.glob(os.path.join(run, '*_requests.jsonl.xz'))):
    g = os.path.basename(f).split('_')[0]
    calls = {}
    with lzma.open(f, 'rt') as fh:
        for line in fh:
            r = json.loads(line)
            if r['event'] == 'response' and r.get('usage'):
                u = r['usage']; usage['prompt'].append(u.get('prompt_tokens',0)); usage['completion'].append(u.get('completion_tokens',0))
                usage['cached'].append((u.get('prompt_tokens_details') or {}).get('cached_tokens',0))
                usage['image'].append((u.get('prompt_tokens_details') or {}).get('image_tokens',0))
            if r['event'] != 'request': continue
            for m in r['messages']:
                if m.get('role') == 'assistant':
                    for tc in m.get('tool_calls') or []:
                        if tc['id'] in calls: continue
                        try:
                            code = json.loads(tc['function']['arguments']).get('code','')
                        except Exception:
                            code = tc['function']['arguments']
                        calls[tc['id']] = code
    c = Counter()
    for code in calls.values():
        for k, p in pats.items():
            if re.search(p, code): c[k] += 1
    c['_calls'] = len(calls)
    tot.update(c)
    per_game_seg[g] = (len(calls), c['segmentation'], c['frame_diff('], c['.ascii'], c['animation_for/last_animation/animations'])
print('total python calls', tot['_calls'])
for k in pats:
    print(f'{k:42s} {tot[k]:6d} {100*tot[k]/max(1,tot["_calls"]):5.1f}%')
print()
print('game calls seg fd ascii anim')
for g,v in per_game_seg.items(): print(g, *v)
import statistics as st
for k,v in usage.items():
    print(k, 'n',len(v),'mean',round(st.mean(v)),'median',st.median(v), 'sum', sum(v))
