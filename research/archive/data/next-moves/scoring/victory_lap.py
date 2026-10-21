"""Replay recorded won games in the local engine, then full-reset after WIN and replay
(a) each level's last attempt (actions after the last RESET in that level) and
(b) the BFS-shortest route through the play-1 observed state graph (validated by execution).
Scores play 2 with the official formula; game score = max over plays (arc_agi scorecard.py:241)."""
import json, lzma, os, sys, importlib.util, hashlib, collections
os.environ["ONLY_RESET_LEVELS"] = "true"   # as in our runs (game_api.py:222)
from arcengine import ARCBaseGame, ActionInput, GameAction, GameState
ENV = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files'

def load(game_id):
    g, ver = game_id.split('-')
    d = f'{ENV}/{g}/{ver}'
    meta = json.load(open(f'{d}/metadata.json'))
    src = open(f'{d}/{g}.py').read()
    mod = type(sys)('m_' + g); exec(src, mod.__dict__)
    cls = [v for v in mod.__dict__.values() if isinstance(v, type) and issubclass(v, ARCBaseGame) and v is not ARCBaseGame][0]
    return cls, meta['baseline_actions']

def score(base, apl, done):
    n = len(base); tw = n*(n+1)/2; ts = 0; mw = 0
    for i in range(n):
        s = min(115.0, (base[i]/apl[i])**2*100) if i < done and apl[i] > 0 else 0
        if s > 0: mw += i+1
        ts += s*(i+1)
    return min(ts/tw, mw/tw*100)

def act(game, a):
    ai = ActionInput(id=GameAction[a['id']], data=a.get('data') or {})
    return game.perform_action(ai, raw=True)

def h(fr):
    import numpy as np
    return hashlib.md5(np.asarray(fr.frame[-1]).tobytes()).hexdigest()

def run_game(gr):
    cls, _ = load(gr['game_id'])
    base = gr['base_actions_per_level']
    try: game = cls(seed=0)
    except TypeError: game = cls()
    fr = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    hist = [x['action'] for x in gr['history']]
    # play 1: segment by level; track last attempt and graph per level
    level = 0; seg = collections.defaultdict(list); last = collections.defaultdict(list)
    graph = collections.defaultdict(dict); start = {0: h(fr)}; cur = h(fr); win_edge = {}
    for a in hist:
        prev = cur
        fr = act(game, a)
        if a['id'] == 'RESET':
            last[level] = []
            cur = h(fr) if fr.frame else cur
            continue
        if not fr.frame:
            return dict(err='empty frame in play1')
        cur = h(fr); seg[level].append(a); last[level].append(a)
        key = json.dumps(a, sort_keys=True)
        if fr.levels_completed > level:
            win_edge[level] = (prev, key)
            level = fr.levels_completed
            start[level] = cur
        else:
            graph[level].setdefault(prev, {})[key] = cur
    if fr.state != GameState.WIN:
        return dict(err=f'play1 not WIN: {fr.state} lvl {fr.levels_completed}')
    apl1 = [len(seg[i]) + 0 for i in range(len(base))]
    # play-1 official counts include RESETs: use benchmark actions_per_level
    apl1 = gr['actions_per_level']
    s1 = score(base, apl1, len(base))
    out = dict(game=gr['game_id'][:4], s1=round(s1, 2), apl1=apl1, base=base)
    # play 2a: last attempts
    def play2(routes):
        fr = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not fr.full_reset or fr.levels_completed != 0:
            return None, 'no full reset'
        apl = []
        for lv in range(len(base)):
            n = 0
            for a in routes[lv]:
                fr = act(game, a); n += 1
                if not fr.frame: return None, f'empty frame L{lv+1}'
                if fr.levels_completed > lv: break
            if fr.levels_completed <= lv:
                return None, f'route failed L{lv+1}'
            apl.append(n)
        return apl, 'ok' if fr.state == GameState.WIN else 'not win'
    apl2, st = play2([last[i] for i in range(len(base))])
    out['last_attempt'] = dict(status=st, apl=apl2, s2=round(score(base, apl2, len(base)), 2) if apl2 else None)
    # play 2b: BFS shortest route in observed graph to the win pre-state, + winning action
    routes = []
    for lv in range(len(base)):
        pre, wkey = win_edge[lv]
        s0 = start[lv]; q = collections.deque([s0]); par = {s0: None}
        while q:
            u = q.popleft()
            if u == pre: break
            for k, v in graph[lv].get(u, {}).items():
                if v not in par: par[v] = (u, k); q.append(v)
        if pre not in par:
            routes.append(last[lv]); continue
        path = []; u = pre
        while par[u] is not None:
            u2, k = par[u]; path.append(json.loads(k)); u = u2
        path = path[::-1] + [json.loads(wkey)]
        routes.append(path if len(path) < len(last[lv]) else last[lv])
    apl3, st3 = play2(routes)
    out['bfs'] = dict(status=st3, apl=apl3, s2=round(score(base, apl3, len(base)), 2) if apl3 else None)
    return out

if __name__ == '__main__':
    res = []
    for path in sys.argv[1:]:
        d = json.load(lzma.open(path))
        for gr in d['game_runs']:
            if gr['levels_completed'] == gr['number_of_levels']:
                try:
                    r = run_game(gr)
                except Exception as e:
                    r = dict(game=gr['game_id'][:4], err=repr(e)[:200])
                r['run'] = path.split('/runs/kaggle/')[-1].rsplit('/', 1)[0]
                print(json.dumps(r), flush=True)
