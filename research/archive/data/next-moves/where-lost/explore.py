"""Exploration vs decisive route on cleared levels: BFS shortest path through the agent's own observed state graph."""
import sys, json, lzma, hashlib
from collections import defaultdict, deque
from load import *

def initial_board(gid, rd):
    ev = find(rd / 'artifacts', f'{gid}_p0_events.jsonl')
    for l in opent(ev):
        r = json.loads(l)
        if r.get('type') == 'initial':
            return r['board']
    return None

def hud_mask(boards):
    """Mask edge rows/cols (outer 3) that change in >= 60% of transitions (step bars, timers)."""
    if len(boards) < 4:
        return set(), set()
    H = len(boards[0]); Wd = len(boards[0][0])
    rows = set(); cols = set()
    cand_r = [r for r in list(range(3)) + list(range(H - 3, H)) if 0 <= r < H]
    cand_c = [c for c in list(range(3)) + list(range(Wd - 3, Wd)) if 0 <= c < Wd]
    n = len(boards) - 1
    for r in cand_r:
        ch = sum(1 for a, b in zip(boards, boards[1:]) if a[r] != b[r])
        if ch >= 0.6 * n: rows.add(r)
    for c in cand_c:
        ch = sum(1 for a, b in zip(boards, boards[1:]) if any(a[i][c] != b[i][c] for i in range(H)))
        if ch >= 0.6 * n: cols.add(c)
    return rows, cols

def key(b, rows, cols):
    if not rows and not cols:
        s = json.dumps(b)
    else:
        s = json.dumps([[v for j, v in enumerate(row) if j not in cols] for i, row in enumerate(b) if i not in rows])
    return hashlib.md5(s.encode()).hexdigest()

def analyse(rel, mask=True):
    rd = ROOT / rel
    r = load_run(rel)
    out = []
    for g in r['games']:
        acts = g['acts']
        if not acts: continue
        b0 = initial_board(g['gid'], rd)
        for i in range(g['lc']):
            a0, a1 = g['bounds'][i]
            if a1 - a0 <= 0: continue
            start = b0 if a0 == 0 else acts[a0 - 1]['board']
            seq = [start] + [acts[j]['board'] for j in range(a0, a1 - 1)]  # states before each action
            rows, cols = hud_mask(seq) if mask else (set(), set())
            ks = [key(b, rows, cols) for b in seq]
            adj = defaultdict(set)
            for t in range(len(ks) - 1):
                adj[ks[t]].add(ks[t + 1])
            # BFS from ks[0] to ks[-1] (state from which the winning action was taken)
            dist = {ks[0]: 0}; dq = deque([ks[0]])
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1; dq.append(v)
            shortest = dist.get(ks[-1], len(ks) - 1) + 1
            n = a1 - a0
            noop = sum(1 for j in range(a0, a1) if not acts[j].get('board_changed', True))
            fa = failed_attempts(g)[i]
            out.append(dict(game=g['key'], level=i, n=n, base=g['base'][i], shortest=shortest, wasted=fa['wasted'],
                            noop=noop, uniq=len(set(ks)), masked=(len(rows), len(cols)), nl=g['nl']))
    return out

if __name__ == '__main__':
    allrows = {}
    for rel in sys.argv[1:]:
        rows = analyse(rel)
        allrows[rel] = rows
        N = sum(x['n'] for x in rows); SH = sum(x['shortest'] for x in rows); B = sum(x['base'] for x in rows)
        WA = sum(x['wasted'] for x in rows); NO = sum(x['noop'] for x in rows)
        over = [x for x in rows if x['n'] > x['base']]
        No = sum(x['n'] for x in over); SHo = sum(x['shortest'] for x in over); Bo = sum(x['base'] for x in over)
        WAo = sum(x['wasted'] for x in over)
        print(f"{rel}: cleared levels {len(rows)}; actions {N}, human base {B} (ratio {N/B:.2f}); shortest-in-own-graph {SH} -> exploration {1-SH/N:.1%}; pre-reset/death {WA} ({WA/N:.1%}); no-op {NO} ({NO/N:.1%})")
        print(f"   levels over baseline: {len(over)}/{len(rows)}; actions {No}, base {Bo}, shortest {SHo} (expl {1-SHo/max(1,No):.1%}), pre-reset {WAo}")
        # score if each cleared level played at its own shortest route
    json.dump(allrows, open(sys.argv[0].replace('.py','')+'-'+str(len(sys.argv)-1)+'.json', 'w'))
