"""Shared loaders: per-game level timing, failed attempts, official score."""
import json, lzma, hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path('/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/next-moves/om/runs/kaggle')
PARK_GAP = 240.0  # s; gap between consecutive actions counted as parked (near-miss report uses >4 min)

def opent(p):
    p = Path(p)
    if p.suffix == '.xz':
        return lzma.open(p, 'rt')
    return open(p)

def find(run_dir, name):
    for c in (run_dir / name, run_dir / (name + '.xz')):
        if c.exists():
            return c
    return None

def official(nl, base, lc, apl):
    tot = 0.0; W = 0; mw = 0
    for i in range(nl):
        w = i + 1; W += w
        a = apl[i] if i < len(apl) else 0
        s = min(115.0, (base[i] / a) ** 2 * 100) if (i < lc and a > 0) else 0.0
        if s > 0: mw += w
        tot += s * w
    return min(tot / W, mw / W * 100) if W else 0.0

def ts(s):
    return datetime.fromisoformat(s).timestamp()

def load_run(rel, run_len_s=None):
    rd = ROOT / rel
    d = json.load(opent(find(rd, 'benchmark.json')))
    t0 = ts(d['start_time'])
    t_end = ts(d['end_time']) - t0 if d.get('end_time') else None
    games = []
    for g in d['game_runs']:
        gid = g['game_id']; key = gid.split('-')[0]
        off = ts(g['started_at']) - t0 if g.get('started_at') else None
        hist = g['history']
        at = [off + h['wallclock_seconds'] for h in hist] if off is not None else []
        names = [h['action']['id'] for h in hist]
        apl = g['actions_per_level']; base = g['base_actions_per_level']; nl = g['number_of_levels']
        lc = g['levels_completed']
        # level boundaries by action count
        bounds = []  # (first_action_idx, last_action_idx_exclusive) per level, from apl
        c = 0
        for i in range(nl):
            bounds.append((c, c + apl[i])); c += apl[i]
        # events: game over and resets per level
        ev = find(rd / 'artifacts', f'{gid}_p0_events.jsonl')
        acts = []
        if ev:
            for l in opent(ev):
                r = json.loads(l)
                if r.get('type') == 'action':
                    acts.append(r)
        games.append(dict(key=key, gid=gid, nl=nl, base=base, apl=apl, lc=lc, state=g['state'],
                          score=g['final_score'] if g['final_score'] is not None else official(nl, base, lc, apl),
                          off=off, at=at, names=names, bounds=bounds, acts=acts,
                          tokens=[h['generated_tokens'] for h in hist],
                          final_wc=g.get('final_wallclock_seconds'), final_tok=g.get('final_generated_tokens') or 0))
    return dict(rel=rel, t_end=t_end, games=games, start=d['start_time'])

RATE = 55.0  # tok/s per stream: median tokens/second over gaps <= 4 min between a game's actions (10x v1 54.6, level-path 55.1)

def level_times(g, run_end):
    """Per level: active = tokens/RATE capped by the wall gap (slot time actually used), parked = rest of the wall time.
    Tokens of no-action turns are folded into the next action's generated_tokens (history total == run total)."""
    out = []
    if g['off'] is None:
        return out
    at = g['at']; tok = g['tokens']
    for i, (a0, a1) in enumerate(g['bounds']):
        if i > g['lc']:
            break
        t_start = g['off'] if a0 == 0 else at[a0 - 1]
        cleared = i < g['lc']
        act = park = 0.0; prev = t_start
        for j in range(a0, min(a1, len(at))):
            gap = at[j] - prev; a = min(gap, tok[j] / RATE)
            act += a; park += gap - a; prev = at[j]
        if not cleared:
            gap = max(0.0, run_end - prev); a = min(gap, (g.get('final_tok') or 0) / RATE)
            act += a; park += gap - a
        out.append(dict(level=i, t_start=t_start, t_end=(at[a1 - 1] if cleared and a1 > 0 else None),
                        active=act, parked=park, actions=a1 - a0, cleared=cleared))
    return out

def failed_attempts(g):
    """Per level index: actions before the last RESET/GAME_OVER restart on that level (wasted attempts), counts."""
    res = {}
    acts = g['acts']
    if not acts:
        return res
    # events 'level' is 1-indexed level being played when the action was taken? use apl bounds instead
    for i, (a0, a1) in enumerate(g['bounds']):
        seg = acts[a0:a1]
        last_restart = -1; n_reset = 0; n_go = 0
        for j, r in enumerate(seg):
            if r.get('action_name') == 'RESET':
                last_restart = j; n_reset += 1
            if r.get('game_over'):
                n_go += 1
                last_restart = max(last_restart, j)  # actions up to and incl. the death are a failed attempt
        res[i] = dict(wasted=last_restart + 1, n_reset=n_reset, n_go=n_go, n=a1 - a0)
    return res
