"""Per game: every <memory> block and <memory_update> the model received (first-seen request index), the model's own memory tags, and the level timeline."""
import json, os, re, sys, glob, collections
run, out = sys.argv[1], sys.argv[2]
TAG = re.compile(r'<(world_model|hypothesis|level_note)[^>]*>(.*?)</\1>', re.S | re.I)
BLK = re.compile(r'<memory>.*?</memory>', re.S)
UPD = re.compile(r'<memory_update>.*?</memory_update>', re.S)
def text(m):
    x = m.get('content')
    if isinstance(x, list): return '\n'.join(p.get('text', '') for p in x if isinstance(p, dict))
    return str(x or '')
summary = {}
for p in sorted(glob.glob(run + '/*_p0_requests.jsonl')):
    game = os.path.basename(p).split('_')[0]
    reqs = [json.loads(l) for l in open(p) if l.strip()]
    reqs = [r for r in reqs if r.get('event') == 'request']
    seen_blk, seen_upd, own = {}, {}, collections.OrderedDict()
    for i, r in enumerate(reqs):
        for m in r.get('messages', []):
            t = text(m)
            if m.get('role') in ('user', 'tool'):
                for b in BLK.findall(t): seen_blk.setdefault(b, i)
                for u in UPD.findall(t): seen_upd.setdefault(u, i)
            if m.get('role') == 'assistant':
                blob = t + '\n' + str(m.get('reasoning_content') or '') + '\n' + json.dumps(m.get('tool_calls') or '')
                for k, v in TAG.findall(blob):
                    own.setdefault((k.lower(), ' '.join(v.split())[:300]), i)
    # level timeline from events
    ev = glob.glob(f'{run}/artifacts/{game}_p0_events.jsonl')
    timeline = []
    if ev:
        last = None
        for l in open(ev[0]):
            d = json.loads(l)
            if d.get('type') != 'action': continue
            lv = d.get('level'); st = d.get('state')
            if lv != last:
                timeline.append(f"action {d.get('action_num')}: level {lv} ({st})"); last = lv
            if st and 'GAME_OVER' in str(st): timeline.append(f"action {d.get('action_num')}: GAME OVER on level {lv}")
    o = [f'# {game}: memory the model received, and its own notes', f'requests: {len(reqs)}', '', '## level timeline (from events)'] + timeline
    o += ['', f'## model-written memory tags ({len(own)} distinct)'] + [f'- req {i} [{k}] {v}' for (k, v), i in own.items()]
    o += ['', f'## <memory> blocks received ({len(seen_blk)} distinct; shown after trims)']
    for b, i in sorted(seen_blk.items(), key=lambda x: x[1]): o += [f'### first seen in request {i}', b, '']
    o += [f'## <memory_update> blocks received ({len(seen_upd)} distinct)']
    for u, i in sorted(seen_upd.items(), key=lambda x: x[1]): o += [f'### request {i}', u, '']
    open(f'{out}/{game}.md', 'w').write('\n'.join(o))
    summary[game] = dict(requests=len(reqs), own_tags=len(own), blocks=len(seen_blk), updates=len(seen_upd))
    print(game, summary[game])
json.dump(summary, open(f'{out}/_summary.json', 'w'), indent=1)
