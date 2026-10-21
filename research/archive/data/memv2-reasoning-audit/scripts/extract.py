"""Compact per-game transcript: every model call's thinking, code and tool result, without the repeated system prompt."""
import re, sys, os, json
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
STEP = re.compile(r'^--- analysis_step=(\d+) \| action=(\d+) \| (\S+) \| ')
for fn in sorted(os.listdir(src)):
    if not fn.endswith('.txt'): continue
    game = fn.split('_')[0]
    lines = open(os.path.join(src, fn), errors='replace').read().split('\n')
    o = []; i = 0; n = len(lines); calls = 0
    sec = None
    while i < n:
        l = lines[i]
        m = STEP.match(l)
        if m:
            o.append(f"\n## step {m[1]} | action {m[2]} | {m[3]}"); sec = None; i += 1; continue
        if l.startswith('[SYSTEM PROMPT]'):
            sec = 'skip'; i += 1; continue
        if l.startswith('[USER PROMPT]'):
            sec = 'user'; o.append('### opener'); i += 1; continue
        if l.startswith('[THINKING]'):
            sec = 'think'; calls += 1; o.append(f'### thinking (call {calls})'); i += 1; continue
        if l.startswith('[CONTENT]') or l.startswith('[ASSISTANT CONTENT]'):
            sec = 'content'; o.append('### reply text'); i += 1; continue
        if l.startswith('[TOOL CALL: python]'):
            sec = 'code'; o.append('### code'); i += 1; continue
        if l.startswith('[TOOL RESULT: python]'):
            sec = 'result'; o.append('### result'); i += 1; continue
        if l.startswith('[ANALYZER STATUS]') or l.startswith('[MODEL RESPONSE META]') or l.startswith('raw_tool_calls'):
            sec = 'skip'; i += 1; continue
        if l.startswith('[') and l.endswith(']') and len(l) < 40:
            sec = 'skip'; i += 1; continue
        if sec in ('think', 'content', 'code'):
            if not l.startswith('<tool_call>') and not l.startswith('</tool_call>') and not l.startswith('<function') and not l.startswith('</function') and not l.startswith('<parameter') and not l.startswith('</parameter'):
                o.append(l)
        elif sec == 'result':
            o.append(l[:400])
        elif sec == 'user':
            # keep the dynamic facts, drop the boilerplate instructions
            if (l.startswith('Current state') or l.startswith('Executed actions') or l.startswith('Previous action') or 'LEVEL' in l or l.startswith('You are') or l.startswith('Valid actions')
                or l.startswith('Shortest route') or l.startswith('Seen before') or l.startswith('<memory') or l.startswith('  <') or l.startswith('</memory') or l.startswith('GAME OVER') or 'level' in l.lower()[:40]):
                o.append(l[:600])
        i += 1
    txt = '\n'.join(o)
    open(os.path.join(out, f'{game}.md'), 'w').write(f'# {game} (mem-v2-10x run, compact transcript: thinking, code, results)\n' + txt)
    print(game, calls, 'calls', round(len(txt) / 1e6, 2), 'MB')
