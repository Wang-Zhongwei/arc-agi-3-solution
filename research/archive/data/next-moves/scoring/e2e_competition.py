"""End-to-end check against the official arc_agi REST server in competition mode
(taaf CompetitionArcadeServer, arc_agi 0.9.9): play a recorded won game, RESET after WIN,
replay shortest routes, close the scorecard and read the official score.
Also checks: does a level RESET count as an action; is a zero-action RESET a no-op; can a new play
start mid-game."""
import json, lzma, os, sys, logging
SERVER_ONLY_RESET = os.environ.get('SRV_ONLY_RESET')
import arc_agi
from arcengine import GameAction, GameState
from taaf.competition_arcade import CompetitionArcadeServer
ENVDIR = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files'
RUN = sys.argv[1]; GAME = sys.argv[2]; VL = sys.argv[3]
gr = [g for g in json.load(lzma.open(RUN))['game_runs'] if g['game_id'] == GAME][0]
hist = [x['action'] for x in gr['history']]
vl = [json.loads(l) for l in open(VL)]
vl = [r for r in vl if r.get('game') == GAME[:4] and RUN.split('/runs/kaggle/')[-1].startswith(r['run'])][0]

srv = CompetitionArcadeServer(game_ids=[GAME], environments_dir=ENVDIR).start()
arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=srv.base_url, arc_api_key=srv.api_key, logger=logging.getLogger('x'))
card = arc.create_scorecard()
env = arc.make(GAME, scorecard_id=card)
def step(a):
    return env.step(GameAction[a['id']], data=a.get('data') or {})
# probe 1: zero-action RESET at game start (should be intercepted: no full reset, +1 action on scorecard)
r = env.step(GameAction.RESET)
print('zero-action RESET at start: levels', r.levels_completed, 'full_reset', getattr(r, 'full_reset', None))
n_extra = 1
for a in hist:
    r = step(a)
print('play1 end state', r.state, 'levels', r.levels_completed)
# RESET after WIN -> expect full reset (new play)
r = env.step(GameAction.RESET)
print('post-WIN RESET: levels', r.levels_completed, 'state', r.state, 'full_reset', getattr(r, 'full_reset', None))
# play 2: replay bfs routes reconstructed from the per-level counts by re-deriving routes in-process
sys.path.insert(0, os.path.dirname(VL)); import victory_lap as V
cls, _ = V.load(GAME)
# rebuild routes the same way as victory_lap (local engine), then send them to the server
import collections, hashlib
from arcengine import ActionInput
game = cls(seed=0) if 'seed' in cls.__init__.__code__.co_varnames else cls()
fr = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
level = 0; last = collections.defaultdict(list); graph = collections.defaultdict(dict); start = {0: V.h(fr)}; cur = V.h(fr); win_edge = {}
for a in hist:
    prev = cur; fr = V.act(game, a)
    if a['id'] == 'RESET': last[level] = []; cur = V.h(fr); continue
    cur = V.h(fr); last[level].append(a); key = json.dumps(a, sort_keys=True)
    if fr.levels_completed > level: win_edge[level] = (prev, key); level = fr.levels_completed; start[level] = cur
    else: graph[level].setdefault(prev, {})[key] = cur
routes = []
for lv in range(gr['number_of_levels']):
    pre, wkey = win_edge[lv]; s0 = start[lv]; q = collections.deque([s0]); par = {s0: None}
    while q:
        u = q.popleft()
        if u == pre: break
        for k, v in graph[lv].get(u, {}).items():
            if v not in par: par[v] = (u, k); q.append(v)
    path = []; u = pre
    while par[u] is not None: u2, k = par[u]; path.append(json.loads(k)); u = u2
    path = path[::-1] + [json.loads(wkey)]
    routes.append(path if len(path) < len(last[lv]) else last[lv])
for lv, rt in enumerate(routes):
    for a in rt:
        r = step(a)
        if r.levels_completed > lv: break
print('play2 end state', r.state, 'levels', r.levels_completed, 'route lens', [len(x) for x in routes])
sc = arc.close_scorecard(card)
envs = sc.environments
for e in envs:
    print('ENV', e.id, 'score(max over plays)=', round(e.score, 2), 'plays=', len(e.runs))
    for run in e.runs:
        print('   play score', round(run.score, 2), 'actions', run.actions, 'resets', run.resets, 'level_actions', run.level_actions)
print('scorecard score', round(sc.score, 2), 'benchmark final_score', round(gr['final_score'], 2), 'local apl', gr['actions_per_level'])
srv.stop()
