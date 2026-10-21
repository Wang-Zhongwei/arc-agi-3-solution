import sys, json, lzma, glob, os, re
from collections import Counter
run = sys.argv[1]
roles = ['player','avatar','agent','goal','target','key','door','wall','hazard','enemy','exit','counter','timer','button','switch','portal','cursor','HUD']
seen_calls, seen_reason = {}, set()
rc = Counter(); n_reason = 0
full_ascii = 0; ascii_slice = 0; ascii_calls = 0
for f in sorted(glob.glob(os.path.join(run, '*_requests.jsonl.xz'))):
    for line in lzma.open(f, 'rt'):
        r = json.loads(line)
        if r['event'] != 'request': continue
        for m in r['messages']:
            if m.get('role') != 'assistant': continue
            rs = m.get('reasoning_content') or ''
            key = hash(rs[:400] + rs[-200:])
            if rs and key not in seen_reason:
                seen_reason.add(key); n_reason += 1
                low = rs.lower()
                for w in roles:
                    if re.search(r'\b' + w.lower() + r's?\b', low): rc[w] += 1
            for tc in m.get('tool_calls') or []:
                if tc['id'] in seen_calls: continue
                try: code = json.loads(tc['function']['arguments']).get('code', '')
                except Exception: code = ''
                seen_calls[tc['id']] = 1
                if '.ascii' in code:
                    ascii_calls += 1
                    if re.search(r'print\(\s*current_frame\.ascii\s*\)', code): full_ascii += 1
                    if re.search(r"\.ascii\.split\(|splitlines\(\)", code): ascii_slice += 1
print('unique reasoning blocks', n_reason)
for w, c in rc.most_common(): print(f'{w:8s} {c:5d} {100*c/n_reason:5.1f}%')
print('ascii calls', ascii_calls, 'print(current_frame.ascii) whole', full_ascii, 'split/splitlines', ascii_slice)
