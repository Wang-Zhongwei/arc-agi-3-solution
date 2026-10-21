"""Simulator v2: continuous-time, event-driven replay of the ARC-AGI-3 harness scheduler.

Calibrated on analysis/calibration/ (calibrate.py; see calibration/README.md). Standard
library only. The harness's own priority code (priority_scheduler.py) is loaded by file
path, so the Franzen, easy-first and D' priorities are the harness functions themselves.

Model
-----
Session clock. Every game's clock starts at the benchmark start (t = 0) and stops at
`minutes * 60` (per-game deadline from session start, as solver.py). Waiting at the gate
counts against it. Requests in flight at the deadline are cut and their tokens lost.

Gate (tool_agent._PriorityGate). `slots` games hold a slot. play() acquires once per game:
Franzen, easy-first and UCB at 2,000,000 - 100 * game_index (every game starts before any
re-queued one, in dispatch order); D' at its own snapshot value (configurable). A game
keeps its slot while its prefix is intact and hands it over only at a context trim
(_maybe_handover): the release and re-enqueue are one step, so it may win the slot straight
back (a repick). With the refreshed queue (the competition config) every waiter that has a
snapshot is re-priced at each pump. Heap order is (-priority, arrival). A game that clears
its last level releases its slot. Easy-first's streak keeps the slot through a trim.

The first `slots` games to acquire are admitted at once; the rest wait in priority order.
Acquires arrive in dispatch order with a few places of jitter (`arrival_jitter`), and a share
of games (`arrival_late_p`) arrives up to `arrival_late_places` later still, as thread start-up
reorders them in the real runs (fitted to the five calibration runs' first waves, where one or
two of indices 5-9 lose their first-wave slot to index 10 or 11).

Each holding game loops:
  1. request: prefill its uncached tokens (the prompt growth since its last request, or the
     whole retained prompt after a trim: the re-prefill), then decode its completion
     tokens. Decode is processor sharing: every decoding stream runs at the measured
     per-request rate for the current number of decoding streams (token-weighted
     ReqTimeStats, so chunked-prefill interference is included; the 12- and 14-slot
     server configs have their own, slower curves).
  2. progress: the completion tokens accrue to the current level; actions accrue in
     proportion. A level clears when its sampled cost is reached (overflow carries on).
  3. client pause: sampled from a log-linear model of the measured gap (prompt size, first
     request of a turn, actions so far, per-game effect, empirical residuals).
  4. context: the prompt grows by the request's sampled increment. When the next prompt
     would exceed the budget (window - 12k output reserve), it is trimmed to the drain
     level and the next request pays a re-prefill; the game hands over before it.
Completion tokens, prompt increment and turn boundary are drawn jointly from consecutive
request pairs in the calibration requests (by first/steady segment and prompt size).

Level costs. Per (game, level), drawn independently per simulated run from a weighted
Kaplan-Meier estimate over the observed attempts of that game-level ACROSS RUNS (cleared
attempts are events, attempts still running at the end are censored). Where a game-level
has fewer than `prior_max_n` (3) attempts, a prior joins with total weight `prior_weight`
(1): the pooled-by-level attempts of all games, rescaled to this game's difficulty (a
Dirichlet-process-style posterior; for a level no run reached it is all there is). The prior
pools levels 1..6 by level and 7+ together (`deep_pool` 7); `deep_pool` 1 pools every level,
a depth-neutral prior (Kaplan-Meier medians by depth 1-7 are 14/28/22/20/39/34/31k on ledger
costs, so the by-depth pool makes unreached deep levels dearer than the data's level costs). The
draw is a (tokens, actions) pair of one observed clear, so efficiency comes with it. Mass
beyond the last observed clear is an unclearable level (tail="never") or a draw from the
pooled tail beyond it (tail="pooled"). A game's own earlier levels are never used (in the
data consecutive levels' cost ratios correlate at 0.05). The ledger condition uses the
ledger runs' attempts; the other group's join at half weight where the group has < 2.
`replay_run` instead keeps one run's realized costs (validate_sim2.py).

Stuck levels. A handful of runs per game-level under-counts the chance that a new run gets
stuck on a level every observed run cleared (each calibration run, left out, has 3-8 levels
it spent >= 60k on whose leave-one-out P(cost > spent) is below 0.1). Two options spread the
unclearable mass: `stuck_floor` makes every level unclearable with that probability on top
of the Kaplan-Meier draw, and `stuck_shrink` shrinks each level's unclearable share toward
its depth's attempt-weighted mean. Both raise the leave-one-run-out log score of the
calibration runs' level outcomes (-2.511 plain; -2.414 at floor 0.04; -2.461 at shrink 0.5),
but both bias the leave-one-run-out run scores low (floor 0.04: 6.8 points, z +1.31; shrink
0.5: z +0.28 against +0.13 plain), because stucks moved onto early levels cost more score
than they save where they were observed. So both default to 0 and serve as sensitivity
checks (holdout_sim2.py sections A and F).

Scorer: taaf GameRun._compute_final_score (the value inference.tools.eval reports), mean
over games.

Usage (from analysis/):
    nice -n 10 python3 sim2.py --policy franzen dprime --ledger on --reps 100
    nice -n 10 python3 sim2.py --policy franzen --games-from-run v4-panel-control \\
        --slots 4 --minutes 20 --ledger off
    nice -n 10 python3 sim2.py --policy franzen dprime --n-games 110 --minutes 532 --reps 20
"""
from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import importlib.util
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALIB = HERE / "calibration"
SCHEDULER_PY = HERE.parent / "ARC3-Inference" / "inference" / "agent" / "priority_scheduler.py"


def _load_scheduler():
    """The harness's priority_scheduler.py, by path (the package __init__ needs requests)."""
    spec = importlib.util.spec_from_file_location("_harness_priority_scheduler", SCHEDULER_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PS = _load_scheduler()

# tool_agent.py gate bands
PRIORITY_BAND = 1_000_000
PRIORITY_UNTRIMMED_BASE = 2_000_000
OUTPUT_RESERVE = 12 * 1024           # LOCAL_ANALYZER_MAX_OUTPUT reserve in the context budget
DEFAULT_CTX, DEFAULT_DRAIN = 128 * 1024, 58 * 1024
REQUEST_SAFETY_MARGIN = 512          # tool_agent._REQUEST_SAFETY_MARGIN_TOKENS


def drain_target(ctx_window: int, drain_tokens: int) -> int:
    """Tokens the harness keeps after a trim: the budget less the drain, the drain capped at
    half the budget (59,136 at the competition setting, where 58 Ki exceeds the cap)."""
    budget = ctx_window - OUTPUT_RESERVE - REQUEST_SAFETY_MARGIN
    return budget - min(drain_tokens, budget // 2)

# Competition notebook (kaggle/template.ipynb): refreshed queue, score normalization,
# remaining-level tail lookup, tail fade 0.2, no pace; tool_agent.py defaults otherwise.
FRANZEN_CONFIG = dict(human_actions=25.0, action_scale=115.0, token_scale=62000.0,
                      action_weight=0.25, tail_base=2.0, tail_cap=4.0, normalize_score=True,
                      tail_lookup="remaining", tail_efficiency=0.8)
FRANZEN_FADE = 0.2
DPRIME_FADE = 0.4                    # ARC3_PRIORITY_TAIL_FADE_FRACTION in the D' arms
EASY_OPTIONS = dict(reference_tokens=30000.0, power=0.5, max_boost=4.0)
EASY_STALL = 2.0


# ============================================================================ scorer

def level_score(baseline: float, actions: int) -> float:
    """Per-level score, taaf GameRun._compute_final_score."""
    if actions <= 0:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def game_score(levels_total: int, completed: int, actions_per_level, baselines) -> float:
    """taaf GameRun._compute_final_score: level-weighted (level i has weight i), capped at the
    weight share of the levels that scored. inference.tools.eval averages it over games."""
    total = tw = mw = 0.0
    for i in range(levels_total):
        w = i + 1
        tw += w
        a = actions_per_level[i] if i < len(actions_per_level) else 0
        s = level_score(baselines[i], a) if i < completed and a > 0 else 0.0
        if s > 0:
            mw += w
        total += s * w
    return 0.0 if tw == 0 else min(total / tw, mw / tw * 100.0)


# ============================================================================ calibration I/O

def read_csv(name: str, calib: Path = CALIB) -> list[dict]:
    with open(calib / name) as f:
        return list(csv.DictReader(f))


def load_meta(calib: Path = CALIB) -> dict:
    return json.loads((calib / "games_meta.json").read_text())


# ============================================================================ level costs

@dataclass(frozen=True)
class Obs:
    """One attempt at a level: generated tokens, cleared or censored, its weight in the KM.
    `actions` is the clear's action count; for a prior (pooled) observation `ratio` is its
    actions / human baseline instead, applied to the target level's baseline."""
    tokens: float
    event: bool
    actions: float = 0.0
    weight: float = 1.0
    ratio: float | None = None


class WeightedKM:
    """Kaplan-Meier estimate from weighted right-censored observations, sampled by inverse CDF.

    At a tied time, events count before censorings (the usual convention). The survival
    left after the last event time (`residual`) is mass the data cannot place."""

    def __init__(self, obs):
        obs = sorted(obs, key=lambda o: (o.tokens, not o.event))
        self.times: list[float] = []
        self.cdf: list[float] = []
        self.groups: list[list[Obs]] = []
        at_risk = sum(o.weight for o in obs)
        surv = 1.0
        i = 0
        while i < len(obs):
            t = obs[i].tokens
            events, d, c = [], 0.0, 0.0
            while i < len(obs) and obs[i].tokens == t:
                if obs[i].event:
                    events.append(obs[i])
                    d += obs[i].weight
                else:
                    c += obs[i].weight
                i += 1
            if d > 0 and at_risk > 0:
                surv *= max(0.0, 1.0 - d / at_risk)
                self.times.append(t)
                self.cdf.append(1.0 - surv)
                self.groups.append(events)
            at_risk -= d + c
        self.residual = surv
        self.max_time = obs[-1].tokens if obs else 0.0
        self.n = len(obs)

    def survival(self, t: float) -> float:
        """S(t) = P(cost > t)."""
        k = bisect.bisect_right(self.times, t)
        return 1.0 if k == 0 else 1.0 - self.cdf[k - 1]

    def sample(self, rng: random.Random, after: float = 0.0):
        """(tokens, Obs) of a draw conditional on cost > after; None for residual mass."""
        s0 = self.survival(after)
        if s0 <= 1e-12 or not self.times:
            return None
        u = 1.0 - rng.random()                     # (0, 1]
        target = (1.0 - s0) + s0 * u               # F(cost) >= target, strictly above F(after)
        k = bisect.bisect_left(self.cdf, target - 1e-12)
        if k >= len(self.times):
            return None
        group = self.groups[k]
        if len(group) == 1:
            return self.times[k], group[0]
        x = rng.random() * sum(o.weight for o in group)
        for o in group:
            x -= o.weight
            if x <= 0:
                return self.times[k], o
        return self.times[k], group[-1]

    def sample_cleared(self, rng: random.Random, after: float = 0.0):
        """(tokens, Obs) of a clear conditional on cost > after (the residual excluded); None
        when no observed clear lies beyond `after`."""
        lo, hi = 1.0 - self.survival(after), 1.0 - self.residual
        if hi - lo <= 1e-12 or not self.times:
            return None
        target = lo + (hi - lo) * (1.0 - rng.random())
        k = min(bisect.bisect_left(self.cdf, target - 1e-12), len(self.times) - 1)
        group = self.groups[k]
        if len(group) == 1:
            return self.times[k], group[0]
        x = rng.random() * sum(o.weight for o in group)
        for o in group:
            x -= o.weight
            if x <= 0:
                return self.times[k], o
        return self.times[k], group[-1]


@dataclass
class LevelDraw:
    tokens: float          # generated tokens to clear (inf = not in this run)
    actions: float         # actions the clear takes (scored against the baseline)
    rate: float            # actions per generated token while on the level
    source: str            # game | prior | tail | never | stuck | replay


FULL_RUN_GROUPS = {"all": None, "ledger": 1, "noledger": 0}


class LevelModel:
    """Per (game, level) cost distributions from levels.csv (full runs by default).

    group: "ledger" (ledger runs), "noledger", or "all". Out-of-group attempts join with
    `cross_weight` only where the group has fewer than `min_n` attempts at that game-level.
    prior_weight: total weight of the pooled-by-level prior, mixed in where the game-level
    has fewer than `prior_max_n` attempts (None: everywhere). A level with no attempts uses
    the prior alone.

    Replay (sample_game(..., replay_run=R)): levels run R attempted keep R's own outcome -
    a clear is its exact (tokens, actions); a level R was still on is drawn from the KM
    conditional on more than R's tokens - and levels R never reached are drawn as usual."""

    def __init__(self, rows: list[dict], meta: dict, *, group: str = "all", runs=None,
                 exclude_runs=(), tail: str = "never", prior_weight: float = 1.0,
                 prior_max_n: float | None = 3, cross_weight: float = 0.5, min_n: int = 2,
                 deep_pool: int = 7, stuck_floor: float = 0.0, stuck_shrink: float = 0.0):
        if tail not in ("never", "pooled"):
            raise ValueError("tail must be never or pooled")
        self.meta = meta["games"]
        self.tail = tail
        self.prior_weight = prior_weight
        self.prior_max_n = prior_max_n
        self.stuck_floor = stuck_floor
        self.stuck_shrink = stuck_shrink
        self._rbar: dict[int, float] = {}
        self.deep_pool = deep_pool
        self.replay_rows = {(r["run"], r["game"], int(r["level"])): r for r in rows}
        full = [r for r in rows if r["kind"] == "full" and r["run"] not in set(exclude_runs)]
        if runs:
            full = [r for r in full if r["run"] in set(runs)]
        want = FULL_RUN_GROUPS[group]
        in_group = [r for r in full if want is None or int(r["ledger"]) == want]
        out_group = [r for r in full if want is not None and int(r["ledger"]) != want]
        self.runs = sorted({r["run"] for r in in_group})

        by_gl = defaultdict(list)
        for r in in_group:
            by_gl[(r["game"], int(r["level"]))].append(self._obs(r, 1.0))
        counts = {k: len(v) for k, v in by_gl.items()}
        for r in out_group:
            key = (r["game"], int(r["level"]))
            if counts.get(key, 0) < min_n:
                by_gl[key].append(self._obs(r, cross_weight))
        self.game_obs = dict(by_gl)

        # game difficulty: shrunk geometric-mean ratio of its clears to the level medians
        pool_rows = in_group
        cleared = defaultdict(list)
        for r in pool_rows:
            if r["cleared"] == "1":
                cleared[self._depth(int(r["level"]))].append(float(r["tokens"]))
        med = {d: st.median(v) for d, v in cleared.items()}
        logs = defaultdict(list)
        for r in pool_rows:
            if r["cleared"] == "1":
                logs[r["game"]].append(math.log(float(r["tokens"]) / med[self._depth(int(r["level"]))]))
        self.scale = {g: math.exp(sum(v) / (len(v) + 4.0)) for g, v in logs.items()}

        # pooled-by-depth prior, in unit-difficulty tokens, efficiency as actions/baseline
        pool = defaultdict(list)
        for r in pool_rows:
            g, lvl = r["game"], int(r["level"])
            base = self.meta[g]["baselines"][lvl - 1]
            pool[self._depth(lvl)].append(Obs(
                float(r["tokens"]) / self.scale.get(g, 1.0), r["cleared"] == "1",
                float(r["actions"]), 1.0, float(r["actions"]) / max(1.0, base)))
        self.pool = dict(pool)

        # actions per token while on a level, for proration (incl. unclearable draws)
        rate_gl, rate_g = defaultdict(list), defaultdict(list)
        for r in full:
            t, a = float(r["tokens"]), float(r["actions"])
            if t > 0:
                rate_gl[(r["game"], int(r["level"]))].append(a / t)
                rate_g[r["game"]].append(a / t)
        self.rate_gl = {k: st.median(v) for k, v in rate_gl.items()}
        self.rate_g = {k: st.median(v) for k, v in rate_g.items()}
        self._km: dict = {}

    @staticmethod
    def _obs(r: dict, weight: float) -> Obs:
        return Obs(float(r["tokens"]), r["cleared"] == "1", float(r["actions"]), weight)

    def _depth(self, level: int) -> int:
        return min(level, self.deep_pool)

    def _prior(self, game: str, level: int, weight: float) -> list[Obs]:
        """The pooled attempts at this depth, rescaled to `game`, sharing total `weight`."""
        pool = self.pool.get(self._depth(level)) or [o for v in self.pool.values() for o in v]
        s = self.scale.get(game, 1.0)
        w = weight / len(pool)
        return [Obs(o.tokens * s, o.event, o.actions, w, o.ratio) for o in pool]

    def km(self, game: str, level: int) -> WeightedKM:
        key = (game, level)
        if key not in self._km:
            own = self.game_obs.get(key, [])
            n = sum(o.weight for o in own)
            if not own:
                prior = self._prior(game, level, 1.0)
            elif self.prior_weight > 0 and (self.prior_max_n is None or n < self.prior_max_n):
                prior = self._prior(game, level, self.prior_weight)
            else:
                prior = []
            self._km[key] = WeightedKM(own + prior)
        return self._km[key]

    def attempts(self, game: str, level: int) -> float:
        return sum(o.weight for o in self.game_obs.get((game, level), []))

    def rate(self, game: str, level: int) -> float:
        return self.rate_gl.get((game, level)) or self.rate_g.get(game) or 1 / 700.0

    def mean_residual(self, level: int) -> float:
        """Kaplan-Meier residual (unclearable share) at this depth, averaged over the game-levels
        the runs reached, weighted by their attempts (levels nobody reached do not count)."""
        d = self._depth(level)
        if d not in self._rbar:
            num = den = 0.0
            for (g, lv), obs in self.game_obs.items():
                w = sum(o.weight for o in obs)
                if self._depth(lv) == d and w > 0 and g in self.meta:
                    num += w * self.km(g, lv).residual
                    den += w
            self._rbar[d] = num / den if den else 0.0
        return self._rbar[d]

    def stuck_probability(self, game: str, level: int) -> float:
        """P(unclearable): the KM residual shrunk toward its depth's mean by `stuck_shrink`, plus
        the floor. With both 0 it is the residual itself."""
        km = self.km(game, level)
        if km.residual >= 1.0 - 1e-12:
            return 1.0
        q = (1.0 - self.stuck_shrink) * km.residual + self.stuck_shrink * self.mean_residual(level)
        return self.stuck_floor + (1.0 - self.stuck_floor) * q

    def survival(self, game: str, level: int, t: float) -> float:
        """P(cost > t): the stuck probability plus the clears beyond t, in the KM's proportions."""
        km = self.km(game, level)
        if not (self.stuck_floor or self.stuck_shrink):
            return km.survival(t)
        p = self.stuck_probability(game, level)
        if p >= 1.0:
            return 1.0
        return p + (1.0 - p) * max(0.0, km.survival(t) - km.residual) / (1.0 - km.residual)

    def sample_level(self, game: str, level: int, rng: random.Random, after: float = 0.0) -> LevelDraw:
        km = self.km(game, level)
        base = self.meta[game]["baselines"][level - 1]
        rate = self.rate(game, level)
        if self.stuck_floor or self.stuck_shrink:
            # unclearable with the stuck probability's share of the survival beyond `after`,
            # otherwise one of the clears beyond it
            surv = self.survival(game, level, after)
            if surv <= 0 or rng.random() < self.stuck_probability(game, level) / surv:
                got = None
            else:
                got = km.sample_cleared(rng, after=after)
        else:
            got = km.sample(rng, after=after)
        source = "game"
        if got is None and self.tail == "pooled":
            prior = WeightedKM(self._prior(game, level, 1.0))
            got = prior.sample(rng, after=max(after, km.max_time))
            source = "tail"
        if got is None:
            return LevelDraw(math.inf, 0.0, rate, "never")
        tokens, o = got
        if o.ratio is not None:
            actions = max(1.0, round(o.ratio * base))
            source = "prior" if source == "game" else source
        else:
            actions = max(1.0, o.actions)
        return LevelDraw(tokens, actions, actions / max(1.0, tokens), source)

    def sample_game(self, game: str, rng: random.Random, replay_run: str | None = None) -> list[LevelDraw]:
        """Every level of `game`, each drawn independently (never from its own earlier draws)."""
        draws = []
        for lvl in range(1, self.meta[game]["levels"] + 1):
            r = self.replay_rows.get((replay_run, game, lvl)) if replay_run else None
            if r is not None and r["cleared"] == "1":
                t, a = float(r["tokens"]), max(1.0, float(r["actions"]))
                draws.append(LevelDraw(t, a, a / max(1.0, t), "replay"))
            elif r is not None:
                draws.append(self.sample_level(game, lvl, rng, after=float(r["tokens"])))
            else:
                draws.append(self.sample_level(game, lvl, rng))
        return draws


# ============================================================================ requests and serving

def prompt_bin(prompt: float) -> int:
    """The data card's client-gap prompt bins: <30k, 30-60k, 60-90k, 90k+."""
    return 0 if prompt < 30_000 else 1 if prompt < 60_000 else 2 if prompt < 90_000 else 3


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small normal equations)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[c][c] != 0:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


# a request's place in its turn: inside an invocation, opening a new turn, or opening a
# resumed invocation after a yield (the request model's third element)
CONT, FRESH, RESUME = 0, 1, 2


class GapModel:
    """log(gap) = b0 + b_bin[prompt bin] + b_first * first_in_turn + b_act * log1p(actions/50)
    + game effect (shrunk mean residual) + an empirical residual. Fitted on the non-trim
    requests of the requests runs (fz-ledger-on-trim-r2, the v4 panels, fz-experts8).

    resume_aware (with `kinds`, the request kinds of prefill_levers.csv): a fresh turn and a
    yield resumption get separate coefficients. Pooled, the turn-start term is mostly fresh
    turns, whose gap holds the action's env step (r2 non-trim mean gap: 5.05 s fresh, 2.62 s
    resumption, 2.08 s inside an invocation), so the pooled model overstates what removing a
    resumption saves. Requests of runs without kinds count as fresh."""

    def __init__(self, req_rows: list[dict], shrink: float = 20.0, kinds: dict | None = None):
        self.resume_aware = kinds is not None
        xs, ys, games = [], [], []
        for r in with_occurrences(req_rows):
            if not r["gap_before_s"] or r["trim"] != "0" or not r["t_entry_s"]:
                continue
            xs.append(self.features(float(r["prompt_tokens"]), request_kind(r, kinds or {}),
                                    float(r["action"] or 0), self.resume_aware))
            ys.append(math.log(max(0.05, float(r["gap_before_s"]))))
            games.append(r["game"])
        k = len(xs[0])
        ata = [[sum(x[i] * x[j] for x in xs) for j in range(k)] for i in range(k)]
        aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(k)]
        self.beta = _solve(ata, aty)
        res = [y - sum(b * xi for b, xi in zip(self.beta, x)) for x, y in zip(xs, ys)]
        per = defaultdict(list)
        for g, e in zip(games, res):
            per[g].append(e)
        self.game_effect = {g: sum(v) / (len(v) + shrink) for g, v in per.items()}
        self.residuals = [e - self.game_effect[g] for g, e in zip(games, res)]
        self.n = len(ys)

    @staticmethod
    def features(prompt: float, first, actions: float, resume_aware: bool = False) -> list[float]:
        """first: a bool (turn start) or a request kind (CONT / FRESH / RESUME)."""
        b = prompt_bin(prompt)
        head = [1.0, float(b == 1), float(b == 2), float(b == 3)]
        if resume_aware:
            head += [float(first == FRESH), float(first == RESUME)]       # True == FRESH
        else:
            head.append(float(bool(first)))
        return head + [math.log1p(max(0.0, actions) / 50.0)]

    def sample(self, rng: random.Random, game: str, prompt: float, first, actions: float) -> float:
        x = self.features(prompt, first, actions, self.resume_aware)
        mu = sum(b * xi for b, xi in zip(self.beta, x)) + self.game_effect.get(game, 0.0)
        return math.exp(mu + rng.choice(self.residuals))


def with_occurrences(rows: list[dict]) -> list[dict]:
    """Copies of request rows tagged with `_occ`, the row's count among earlier rows of the same
    (run, game, step, action, req_in_turn) in file order: a resumption repeats all four."""
    seen, out = defaultdict(int), []
    for r in rows:
        key = (r["run"], r["game"], r["step"], r["action"], r["req_in_turn"])
        out.append(dict(r, _occ=seen[key]))
        seen[key] += 1
    return out


def request_key(r: dict) -> tuple:
    return (r["run"], r["game"], r["step"], r["action"], r["req_in_turn"], r["_occ"])


def lever_kinds(rows: list[dict]) -> dict:
    """request key -> CONT / FRESH / RESUME, from the prefill_levers.csv rows."""
    out = {}
    for r in with_occurrences([r for r in rows if r.get("answered", "1") == "1"]):
        out[request_key(r)] = {"fresh": FRESH, "resume": RESUME}.get(r["kind"], CONT)
    return out


def request_kind(r: dict, kinds: dict) -> int:
    """A requests.csv row's kind: from the lever table where it covers the run, else a turn
    start (req_in_turn 1) counts as fresh."""
    k = kinds.get(request_key(r))
    if k is not None:
        return k
    return FRESH if r["req_in_turn"] == "1" else CONT


def lever_adjustments(lever: str, rows: list[dict]) -> dict:
    """(run, game, step, action, req_in_turn, occurrence) -> (increment change, opens or None)
    from the prefill_levers.csv rows (answered requests, in file order like requests.csv)."""
    col = f"d_{lever}"
    if not rows or col not in rows[0]:
        known = sorted(k[2:] for k in rows[0] if k.startswith("d_")) if rows else []
        raise ValueError(f"unknown prefill lever {lever!r}; known: {known}")
    ykey = lever.split("+")[0]
    ocol = f"open_{ykey}" if f"open_{ykey}" in rows[0] else None
    return {request_key(r): (round(float(r[col])), (r[ocol] == "1") if ocol else None)
            for r in with_occurrences([r for r in rows if r.get("answered", "1") == "1"])}


class RequestModel:
    """Joint (completion tokens, next prompt increment, next request starts a turn) drawn from
    consecutive request pairs, by segment kind (first context segment / after a trim) and
    prompt size (20k bins); plus the trim drain level and the cached system prefix."""

    BIN = 20_000

    def __init__(self, req_rows: list[dict], seg_rows: list[dict], lever: str | None = None,
                 lever_rows: list[dict] | None = None, kinds: dict | None = None):
        """lever: a column pair of calibration/prefill_levers.csv (prefill_levers.py). Each
        request's prompt increment changes by `d_<lever>` (the new tokens the lever adds or
        removes on that request) and, for a yield lever, whether it opens an invocation comes
        from `open_<yield key>`. Requests of runs the table does not cover are unchanged."""
        adjust = lever_adjustments(lever, lever_rows) if lever else {}
        self.lever = lever
        self.adjusted = 0
        kinds = kinds or {}
        by_game = defaultdict(list)
        for r in with_occurrences(req_rows):
            if r["t_entry_s"] and r["prompt_tokens"]:
                by_game[(r["run"], r["game"])].append(r)
        cells = defaultdict(list)
        first_prompts = []
        for reqs in by_game.values():
            reqs.sort(key=lambda r: float(r["t_entry_s"]))
            first_prompts.append(float(reqs[0]["prompt_tokens"]))
            seen_trim = False
            for a, b in zip(reqs, reqs[1:]):
                seen_trim = seen_trim or a["trim"] == "1"
                if b["trim"] == "1":
                    continue
                dp = int(b["prompt_tokens"]) - int(a["prompt_tokens"])
                if dp < 0:
                    continue
                first = request_kind(b, kinds)
                adj = adjust.get(request_key(b))
                if adj is not None:
                    d, opens = adj
                    dp = max(0, dp + d)
                    if opens is not None and first != FRESH:
                        first = RESUME if opens else CONT
                    self.adjusted += 1
                kind = "steady" if seen_trim else "first"
                cells[(kind, int(float(a["prompt_tokens"]) // self.BIN))].append(
                    (int(a["completion_tokens"]), dp, first))
        self.cells = dict(cells)
        self.kinds = {k: sorted(b for kk, b in self.cells if kk == k) for k in ("first", "steady")}
        self.first_prompt = st.median(first_prompts)
        trims = [s for s in seg_rows if s["method"] == "requests" and s["ended_by_trim"] == "1"
                 and s["next_prompt"] and s["reprefill_uncached"]]
        self.drain_prompts = [float(s["next_prompt"]) for s in trims]          # at drain 59392
        self.cached_prefix = st.median(float(s["next_prompt"]) - float(s["reprefill_uncached"])
                                       for s in trims)

    def sample(self, rng: random.Random, kind: str, prompt: float):
        bins = self.kinds[kind] or self.kinds["first"]
        want = int(prompt // self.BIN)
        b = min(bins, key=lambda x: (abs(x - want), x))
        return rng.choice(self.cells[(kind if self.kinds[kind] else "first", b)])

    def drain_prompt(self, rng: random.Random, drain_tokens: int, ctx_window: int = DEFAULT_CTX) -> float:
        """Prompt after a trim: an observed one (at the default setting), scaled by the retained
        target the harness drains to, budget - min(drain, budget // 2) (tool_agent.py
        _trim_messages_for_context; budget = window - output reserve - 512). Scaling by the
        drain itself, as before, was right only where drain = budget / 2 (every calibration
        run); a smaller drain retains MORE history, not less."""
        return rng.choice(self.drain_prompts) * drain_target(ctx_window, drain_tokens) / drain_target(
            DEFAULT_CTX, DEFAULT_DRAIN)


# The phase-3 calibration set: Franzen-scheduler full runs on the standard 10-slot server.
# Every fitted piece (level costs, decode curve, request shape, client gaps) comes from these
# under fit="franzen"; UCB, D' (v4-stack-v1), the 12- and 14-slot runs and the panels are held
# out. fit="all" is the phase-2 calibration (every run).
CALIB_NOLEDGER = ("fz-arms-v4-control", "fz-arms-v5-temp06", "fz-arms-v6-audit60k")
CALIB_LEDGER = ("fz-ledger-on-trim", "fz-ledger-on-trim-r2")
FIT_SETS = {"franzen": CALIB_NOLEDGER + CALIB_LEDGER, "all": None}

# Serving configurations with their own decode curve. At the same number of running requests
# the 12- and 14-slot runs decoded 10-12% slower than the 10-slot ones (token-weighted
# ReqTimeStats; one run each, so config and machine cannot be told apart).
CURVE_RUNS = {
    "std": ("fz-arms-v4-control", "fz-arms-v5-temp06", "fz-arms-v6-audit60k", "fz-ledger-on-trim",
            "fz-ledger-on-trim-r2", "fz-ucb", "fz-tput-base", "fz-experts8", "v4-panel-control",
            "v4-panel-easy-first", "v4-panel-dprime"),
    "s12": ("fz-ledger-streams12",),
    "s14": ("fz-streams14",),
}


def curve_for_slots(slots: int) -> str:
    return "std" if slots <= 10 else "s12" if slots <= 12 else "s14"


def _pava_decreasing(points):
    """Weighted pooled-adjacent-violators: [(n, value, weight)] -> {n: value}, non-increasing."""
    blocks = []
    for n, v, w in points:
        blocks.append([v, float(w), [n]])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return {n: v for v, _, ns in blocks for n in ns}


def decode_curves(reqstats: list[dict], summary: dict, min_count: int = 20,
                  fit_runs=None, decode_lines: list[dict] | None = None) -> dict[str, dict[int, float]]:
    """Per-stream decode tok/s by number of running requests, per serving configuration.

    Token-weighted (sum of output / sum of decode s) over ReqTimeStats requests with >= 64
    output tokens, by mean #running-req rounded; this is the rate requests actually got,
    chunked-prefill interference included. 1-2 streams (rare) come from the decode-batch
    lines. A configuration's buckets below its own data are the standard curve times the
    ratio where both have data. Monotone non-increasing (weighted PAVA); past the last
    bucket the aggregate is flat.

    fit_runs: only these runs' lines are used. A configuration with no run left gets no curve
    (the simulator then uses `std`, flat aggregate past its last bucket). The 1-2 stream rates
    then come from those runs' decode-batch lines (`decode_lines`, serve_decode.csv)."""
    groups = {g: tuple(r for r in runs if fit_runs is None or r in fit_runs) for g, runs in CURVE_RUNS.items()}
    groups = {g: runs for g, runs in groups.items() if runs}
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    for r in reqstats:
        o, d = float(r["output"]), float(r["decode_s"])
        if o < 64 or d <= 0:
            continue
        n = round(float(r["running_mean"]))
        for group, runs in groups.items():
            if r["run"] in runs:
                x = sums[(group, n)]
                x[0] += o
                x[1] += d
                x[2] += 1
    raw = {}
    for group in groups:
        raw[group] = {n: (x[0] / x[1], x[2]) for (g, n), x in sums.items() if g == group and x[2] >= min_count}
    if fit_runs is None or decode_lines is None:
        pooled = summary["serving"]["decode_lines_pooled_std10"]
        for n in (1, 2):
            if n not in raw["std"]:
                raw["std"][n] = (pooled[str(n)]["per_stream_tps_mean"], pooled[str(n)]["lines"])
    else:
        # buckets the request lines leave thin (the full runs rarely have < 6 streams running):
        # the decode-batch per-stream rate of the same runs, times the request/batch ratio
        # where both have data (prefill interference)
        lines = {}
        for n in range(1, 1 + max(int(r["running_req"]) for r in decode_lines)):
            rows = [r for r in decode_lines if r["run"] in groups["std"] and int(r["running_req"]) == n]
            w = sum(int(r["n_lines"]) for r in rows)
            if w:
                lines[n] = (sum(float(r["agg_tps_mean"]) / n * int(r["n_lines"]) for r in rows) / w, w)
        both = [n for n in raw["std"] if n in lines]
        ratio = (sum(raw["std"][n][0] * raw["std"][n][1] for n in both)
                 / sum(lines[n][0] * raw["std"][n][1] for n in both)) if both else 1.0
        lo = min(raw["std"], default=max(lines) + 1)
        for n, (v, w) in lines.items():
            if n < lo:
                raw["std"][n] = (v * ratio, min(w, min_count))
    curves = {"std": _pava_decreasing(sorted((n, v, w) for n, (v, w) in raw["std"].items()))}
    for group in [g for g in ("s12", "s14") if g in groups]:
        common = [n for n in raw[group] if n in raw["std"] and raw[group][n][1] >= 30 and raw["std"][n][1] >= 30]
        ratio = (sum(raw[group][n][0] * raw[group][n][1] for n in common)
                 / sum(raw["std"][n][0] * raw[group][n][1] for n in common)) if common else 1.0
        pts = dict(raw[group])
        for n, (v, w) in raw["std"].items():
            if n < min(raw[group]):
                pts[n] = (v * ratio, w)
        curves[group] = _pava_decreasing(sorted((n, v, w) for n, (v, w) in pts.items()))
    return curves


@dataclass
class Serving:
    curve: dict[int, float]
    speed_scale: float = 1.0
    prefill_tps: float = 11_600.0     # large-prompt prefill (re-prefill 53.6k in 4.7 s)
    prefill_overhead_s: float = 0.08  # small prompts: 0.13 s below 1k uncached tokens

    def rate(self, n: int) -> float:
        if n <= 0:
            return 0.0
        top = max(self.curve)
        v = self.curve[n] if n in self.curve else self.curve[top] * top / n
        return v * self.speed_scale

    def prefill_s(self, uncached: float) -> float:
        return self.prefill_overhead_s + max(0.0, uncached) / self.prefill_tps


class Calibration:
    """Everything the simulator reads from analysis/calibration/, loaded once.

    fit: "franzen" (default) fits every piece on the Franzen calibration runs only (FIT_SETS);
    "all" on every run (the phase-2 calibration). Replay (`replay_run`) and the run settings
    (dispatch order, baselines) read any run either way."""

    def __init__(self, calib: Path = CALIB, fit: str = "franzen", resume_gaps: bool = False):
        """resume_gaps: fit the client gap with separate fresh-turn and resumption terms
        (GapModel resume_aware); off reproduces the phase-3 calibration."""
        self.dir = calib
        self.fit = fit
        self.fit_runs = FIT_SETS[fit]
        keep = (lambda r: True) if self.fit_runs is None else (lambda r: r["run"] in self.fit_runs)
        self.meta = load_meta(calib)
        self.summary = json.loads((calib / "summary.json").read_text())
        self.level_rows = read_csv("levels.csv", calib)
        req = [r for r in read_csv("requests.csv", calib) if keep(r)]
        seg = [r for r in read_csv("segments.csv", calib) if keep(r)]
        self._lever_rows = read_csv("prefill_levers.csv", calib) if (calib / "prefill_levers.csv").exists() else []
        self._kinds = lever_kinds(self._lever_rows)
        self.gaps = GapModel(req, kinds=self._kinds if resume_gaps else None)
        self.requests = RequestModel(req, seg, kinds=self._kinds)
        self._req_rows, self._seg_rows = req, seg
        self._lever_models: dict = {}
        self.curves = decode_curves(read_csv("reqstats.csv", calib), self.summary, fit_runs=self.fit_runs,
                                    decode_lines=read_csv("serve_decode.csv", calib))
        self._levels: dict = {}

    def request_model(self, lever: str | None) -> "RequestModel":
        """The request model with a prefill lever applied (None: the calibrated one)."""
        if not lever:
            return self.requests
        if lever not in self._lever_models:
            self._lever_models[lever] = RequestModel(self._req_rows, self._seg_rows, lever, self._lever_rows,
                                                     kinds=self._kinds)
        return self._lever_models[lever]

    def curve(self, name: str) -> dict[int, float]:
        """A serving configuration's decode curve; `std` when it has none (held out)."""
        return self.curves.get(name) or self.curves["std"]

    def levels(self, **kw) -> LevelModel:
        if self.fit_runs is not None:
            kw.setdefault("runs", tuple(self.fit_runs))
        key = json.dumps(kw, sort_keys=True, default=list)
        if key not in self._levels:
            self._levels[key] = LevelModel(self.level_rows, self.meta, **kw)
        return self._levels[key]


# ============================================================================ policies

def level_weighted_reward(levels_completed: int, total_levels) -> float:
    """ff0630c priority_scheduler.level_weighted_reward (UCB)."""
    total = PS.parse_total_levels(total_levels) or 10
    done = min(max(0, int(levels_completed)), total)
    return (done * (done + 1) / 2) / (total * (total + 1) / 2)


UCB_VALUE_CAP = 400_000


@dataclass(frozen=True)
class UCBSnapshot:
    game_tokens: float
    reward: float
    recent_rate: float | None


def ucb_value(state: UCBSnapshot, *, total_tokens: float, c: float = 0.1,
              unit: float = 100_000.0, recent_weight: float = 0.5) -> int:
    """ff0630c priority_scheduler.ucb_value: reward rate + c * sqrt(ln(1 + T/unit) / (1 + n/unit))."""
    n = max(0.0, float(state.game_tokens)) / unit
    lifetime = float(state.reward) / max(n, 1e-9) if n > 0 else 0.0
    recent = lifetime if state.recent_rate is None else float(state.recent_rate)
    rate = (1.0 - recent_weight) * lifetime + recent_weight * recent
    bonus = c * math.sqrt(math.log1p(max(0.0, total_tokens) / unit) / (1.0 + n))
    return max(1, min(UCB_VALUE_CAP, int((rate + bonus) * 100_000)))


class Policy:
    """Gate pricing. fresh(): play()'s initial acquire; handover(): _maybe_handover at a trim;
    reprice(): the refreshed queue (_snapshot_priority) at each pump; hold(): keep the slot.
    Optional: reprice_many(snapshots, now) prices every waiter in one call (the gate then uses
    it instead of reprice); bind(simulator) gives run-wide observables."""

    name = "base"
    refresh = True

    def __init__(self, deadline: float, start: float = 0.0):
        self.clock_start, self.clock_deadline = start, deadline

    def tail_fraction(self, now: float, fade: float | None) -> float | None:
        if fade is None:
            return None
        window = fade * max(0.0, self.clock_deadline - self.clock_start)
        remaining = max(0.0, self.clock_deadline - now)
        return remaining / window if window > 0 else 0.0

    def fresh(self, g, now: float):
        return PRIORITY_UNTRIMMED_BASE - 100 * g.index, None

    def handover(self, g, now: float):
        snap = self.snapshot(g)
        return self.reprice(snap, now), snap

    def snapshot(self, g):
        raise NotImplementedError

    def reprice(self, snap, now: float) -> int:
        raise NotImplementedError

    def hold(self, g) -> bool:
        return False


class Franzen(Policy):
    name = "franzen"

    def __init__(self, deadline, start=0.0, fade=FRANZEN_FADE):
        super().__init__(deadline, start)
        self.fade = fade

    def snapshot(self, g):
        return PS.PrioritySnapshot(g.level + 1, int(g.actions_on_level()), float(g.spent), 1.0,
                                   g.n_levels)

    def reprice(self, snap, now):
        return PS.priority_value(snap, tail_fraction=self.tail_fraction(now, self.fade), **FRANZEN_CONFIG)


class EasyFirst(Franzen):
    """ARC3_EASY_FIRST on the competition config: boosted price and the streak hold."""
    name = "easy-first"

    def snapshot(self, g):
        boost = PS.easy_boost(g.cleared_levels, g.cleared_tokens, **EASY_OPTIONS)
        cost = g.cleared_tokens / g.cleared_levels if g.cleared_levels else 0.0
        return PS.PrioritySnapshot(g.level + 1, int(g.actions_on_level()), float(g.spent), 1.0,
                                   g.n_levels, boost, cost)

    def hold(self, g):
        return PS.easy_streak(g.cleared_levels, g.cleared_tokens, g.spent,
                              reference_tokens=EASY_OPTIONS["reference_tokens"], stall_multiple=EASY_STALL)


class DPrime(Policy):
    """ARC3_PRIORITY_POLICY=dprime. fresh_first=False is the harness (unstarted games are ranked
    by the formula at level 1 with no spend); True puts them in the 2M band like Franzen."""
    name = "dprime"

    def __init__(self, deadline, start=0.0, fade=DPRIME_FADE, fresh_first=False):
        super().__init__(deadline, start)
        self.fade, self.fresh_first = fade, fresh_first
        if fresh_first:
            self.name = "dprime-ff"

    def fresh(self, g, now):
        if self.fresh_first:
            return super().fresh(g, now)
        snap = PS.PrioritySnapshot(1, 0, 0.0, 1.0, g.n_levels)
        return self.reprice(snap, now), snap

    def snapshot(self, g):
        cost = g.cleared_tokens / g.cleared_levels if g.cleared_levels else 0.0
        return PS.PrioritySnapshot(g.level + 1, int(g.actions_on_level()), float(g.spent), 1.0,
                                   g.n_levels, 1.0, cost)

    def reprice(self, snap, now):
        return PS.d_prime_priority(snap, tail_fraction=self.tail_fraction(now, self.fade))


class UCB(Policy):
    """ARC3_PRIORITY_POLICY=ucb at ff0630c: no tail fade; T sums each game's generated tokens
    as recorded at its last handover; recent rate over the last 64k tokens of level-up marks."""
    name = "ucb"

    def __init__(self, deadline, start=0.0, c=0.1, recent_tokens=64_000.0):
        super().__init__(deadline, start)
        self.c, self.window = c, recent_tokens
        self.recorded: dict[int, float] = {}

    def snapshot(self, g):
        tokens = float(g.tokens)
        reward = level_weighted_reward(g.level, g.n_levels)
        recent = None
        if tokens >= self.window:
            cutoff = tokens - self.window
            earlier = [r for t, r in g.marks if t <= cutoff]
            recent = (reward - (earlier[-1] if earlier else 0.0)) / (self.window / 100_000.0)
        return UCBSnapshot(tokens, reward, recent)

    def handover(self, g, now):
        snap = self.snapshot(g)
        self.recorded[g.index] = snap.game_tokens
        return self.reprice(snap, now), snap

    def reprice(self, snap, now):
        return ucb_value(snap, total_tokens=sum(self.recorded.values()), c=self.c)


class FIFO(Policy):
    """Reference: every re-queued game at the same price (arrival order), fresh first."""
    name = "fifo"
    refresh = False

    def handover(self, g, now):
        return 1, None


POLICIES = {
    "franzen": lambda d, kw: Franzen(d, fade=kw.get("fade", FRANZEN_FADE)),
    "easy-first": lambda d, kw: EasyFirst(d, fade=kw.get("fade", FRANZEN_FADE)),
    "dprime": lambda d, kw: DPrime(d, fade=kw.get("fade", DPRIME_FADE)),
    "dprime-ff": lambda d, kw: DPrime(d, fade=kw.get("fade", DPRIME_FADE), fresh_first=True),
    "ucb": lambda d, kw: UCB(d, c=kw.get("ucb_c", 0.1)),
    "fifo": lambda d, kw: FIFO(d),
}


def make_policy(name: str, deadline: float, **kw) -> Policy:
    kw = {k: v for k, v in kw.items() if v is not None}
    return POLICIES[name](deadline, kw)


# ============================================================================ gate

class Gate:
    """tool_agent._PriorityGate, single-threaded: each call returns the games it admitted."""

    def __init__(self, slots: int, policy: Policy):
        self.free = max(1, int(slots))
        self.policy = policy
        self.waiting: list[tuple[int, int, int]] = []   # (-priority, arrival, game index)
        self.snapshots: dict[int, object] = {}
        self.seq = 0

    def acquire(self, game: int, priority: int, snapshot, now: float) -> list[int]:
        return self._enqueue(game, priority, snapshot, now)

    def handover(self, game: int, priority: int, snapshot, now: float) -> list[int]:
        """Release and re-enqueue as one step: the caller may win its slot straight back."""
        self.free += 1
        return self._enqueue(game, priority, snapshot, now)

    def release(self, now: float) -> list[int]:
        self.free += 1
        return self._pump(now)

    def _enqueue(self, game, priority, snapshot, now):
        self.seq += 1
        if snapshot is not None:
            self.snapshots[self.seq] = snapshot
        heapq.heappush(self.waiting, (-int(priority), self.seq, game))
        return self._pump(now)

    def _pump(self, now: float) -> list[int]:
        if self.policy.refresh and self.free > 0:
            many = getattr(self.policy, "reprice_many", None)
            if many is not None:
                # joint pricing (e.g. ranks among the waiters): one call over every snapshot
                toks = [tok for _, tok, _ in self.waiting if tok in self.snapshots]
                prices = dict(zip(toks, many([self.snapshots[tok] for tok in toks], now)))
                self.waiting = [(-prices[tok], tok, g) if tok in prices else (p, tok, g)
                                for p, tok, g in self.waiting]
            else:
                self.waiting = [
                    (-self.policy.reprice(self.snapshots[tok], now), tok, g) if tok in self.snapshots
                    else (p, tok, g) for p, tok, g in self.waiting]
            heapq.heapify(self.waiting)
        admitted = []
        while self.free > 0 and self.waiting:
            _, tok, g = heapq.heappop(self.waiting)
            self.snapshots.pop(tok, None)
            self.free -= 1
            admitted.append(g)
        return admitted

    def queued(self) -> list[int]:
        return [g for _, _, g in self.waiting]


# ============================================================================ simulation

@dataclass
class SimConfig:
    games: list[str]                       # dispatch order; names may repeat (copies)
    policy: str = "franzen"
    slots: int = 10
    minutes: float = 121.0
    ledger: str = "all"                    # level-cost group: ledger | noledger | all
    ctx_window: int = DEFAULT_CTX
    drain_tokens: int = DEFAULT_DRAIN
    speed_scale: float = 1.0               # multiplies the decode curve (16 experts: 0.85)
    curve: str | None = None               # serving config: std | s12 | s14 (None: by slots)
    fade: float | None = None              # None: the policy's own (Franzen 0.2, D' 0.4)
    ucb_c: float = 0.1
    first_request_s: float = 1.0           # benchmark start -> a game's first request
    arrival_jitter: float = 3.0            # acquire order = dispatch index + U(0, this) ...
    arrival_late_p: float = 0.2            # ... and, for this share of games, + U(0, late places)
    arrival_late_places: float = 10.0
    server_ready_s: float = 0.0            # no prefill before this (startup race)
    tail: str = "never"
    prior_weight: float = 1.0
    prior_max_n: float | None = 3          # prior only where a game-level has < this many attempts
    stuck_floor: float = 0.0               # any level unclearable with this probability on top
    stuck_shrink: float = 0.0              # unclearable share shrunk toward its depth's mean
    deep_pool: int = 7                     # prior pooled by level up to this depth (1: depth-neutral)
    exclude_runs: tuple = ()
    replay_run: str | None = None          # keep this run's own outcome on the levels it attempted
    lever: str | None = None               # prefill lever (calibration/prefill_levers.csv column)


def arrival_keys(n: int, rng: random.Random, cfg: SimConfig) -> list[float]:
    """Arrival position of each dispatch index at the gate (sorted, it is the acquire order)."""
    keys = []
    for i in range(n):
        x = i + rng.uniform(0.0, cfg.arrival_jitter)
        if cfg.arrival_late_p > 0 and rng.random() < cfg.arrival_late_p:
            x += rng.uniform(0.0, cfg.arrival_late_places)
        keys.append(x)
    return keys


@dataclass
class SimGame:
    index: int
    name: str
    n_levels: int
    baselines: list[int]
    draws: list[LevelDraw]
    rng: random.Random
    level: int = 0                 # current level, 0-based (= levels completed)
    spent: float = 0.0             # tokens on the current level
    tokens: float = 0.0            # generated tokens credited
    cleared_tokens: float = 0.0
    cleared_actions: float = 0.0
    actions_per_level: list[int] = field(default_factory=list)
    marks: list[tuple[float, float]] = field(default_factory=list)   # UCB level-up marks
    prompt: float = 0.0
    kind: str = "first"
    reprefill: bool = False
    pending: tuple | None = None   # (completion, prompt increment, next request starts a turn)
    next_prompt: float = 0.0
    holding: bool = False
    fresh: bool = True
    done: bool = False
    first_admit: float | None = None
    level_up_s: list[float] = field(default_factory=list)
    queued_at: float | None = None
    waits: list[float] = field(default_factory=list)
    n_trims: int = 0
    n_switch_outs: int = 0
    n_requests: int = 0
    uncached: float = 0.0          # prompt tokens prefilled (new tokens plus re-prefills)
    active_s: float = 0.0
    held_since: float | None = None
    switched_out_at: float | None = None
    readmitted_at: float | None = None

    @property
    def cleared_levels(self) -> int:
        return self.level

    def actions_on_level(self) -> float:
        if self.level >= self.n_levels:
            return 0.0
        d = self.draws[self.level]
        return min(d.actions, d.rate * self.spent) if math.isfinite(d.tokens) else d.rate * self.spent

    def actions_total(self) -> float:
        return self.cleared_actions + self.actions_on_level()

    def score(self) -> float:
        return game_score(self.n_levels, self.level, self.actions_per_level, self.baselines)


@dataclass
class SimResult:
    score: float
    games: list[SimGame]
    total_tokens: float
    lost_tokens: float
    deadline: float
    slots: int
    switch_outs: int
    trims: int
    never_readmitted: int
    starts: list[float | None]
    mean_decoding: float
    requests: int
    uncached_tokens: float = 0.0

    def summary(self) -> dict:
        starts = [s for s in self.starts if s is not None]
        return {
            "score": self.score, "total_tokens": self.total_tokens, "lost_tokens": self.lost_tokens,
            "switch_outs": self.switch_outs, "trims": self.trims, "never_readmitted": self.never_readmitted,
            "started_after_30min": sum(1 for s in starts if s > 1800), "never_started": len(self.starts) - len(starts),
            "last_start_s": max(starts) if starts else None, "mean_decoding": self.mean_decoding,
            "requests": self.requests, "uncached_tokens": self.uncached_tokens,
            "tok_per_active_s": self.total_tokens / max(1.0, sum(g.active_s for g in self.games)),
            "levels": sum(g.level for g in self.games),
        }


class Simulator:
    """One simulated run. Event-driven; decode is processor sharing in virtual service time."""

    def __init__(self, cfg: SimConfig, calib: Calibration, seed: int, levels: LevelModel | None = None):
        self.cfg = cfg
        self.cal = calib
        self.deadline = cfg.minutes * 60.0
        self.policy = make_policy(cfg.policy, self.deadline, fade=cfg.fade, ucb_c=cfg.ucb_c)
        if hasattr(self.policy, "bind"):
            # run-wide observables (slots, unfinished games, total tokens) for policies that use them
            self.policy.bind(self)
        self.gate = Gate(cfg.slots, self.policy)
        name = cfg.curve or curve_for_slots(cfg.slots)
        curve = calib.curve(name) if hasattr(calib, "curve") else calib.curves[name]
        self.serving = Serving(curve, cfg.speed_scale)
        self.budget = cfg.ctx_window - OUTPUT_RESERVE
        self.requests = calib.request_model(cfg.lever) if hasattr(calib, "request_model") else calib.requests
        levels = levels or calib.levels(group=cfg.ledger, tail=cfg.tail, prior_weight=cfg.prior_weight,
                                        prior_max_n=cfg.prior_max_n, exclude_runs=tuple(cfg.exclude_runs),
                                        stuck_floor=cfg.stuck_floor, stuck_shrink=cfg.stuck_shrink,
                                        deep_pool=cfg.deep_pool)
        # common random numbers: level draws depend only on (seed, game slot), not the policy
        meta = calib.meta["games"]
        self.seed = seed
        self.games = []
        for i, name in enumerate(cfg.games):
            lrng = random.Random(f"levels:{seed}:{i}:{name}")
            self.games.append(SimGame(
                index=i, name=name, n_levels=meta[name]["levels"], baselines=meta[name]["baselines"],
                draws=levels.sample_game(name, lrng, cfg.replay_run), rng=random.Random(f"requests:{seed}:{i}:{name}")))
        self.events: list = []
        self.seq = 0
        self.now = 0.0
        # decode pool: virtual service V advances at the per-stream rate
        self.decoding: list = []        # heap (target V, seq, game)
        self.vtime = 0.0
        self.vt_at = 0.0
        self.busy_integral = 0.0

    # ---- event plumbing
    def push(self, t: float, kind: str, g: int) -> None:
        self.seq += 1
        heapq.heappush(self.events, (t, self.seq, kind, g))

    def advance(self, t: float) -> None:
        n = len(self.decoding)
        if n:
            self.vtime += self.serving.rate(n) * (t - self.vt_at)
            self.busy_integral += n * (t - self.vt_at)
        self.vt_at = t

    def next_decode(self) -> float:
        if not self.decoding:
            return math.inf
        return self.vt_at + max(0.0, self.decoding[0][0] - self.vtime) / self.serving.rate(len(self.decoding))

    # ---- game flow
    def admit(self, gi: int, t: float) -> None:
        g = self.games[gi]
        g.holding = True
        g.held_since = t
        if g.queued_at is not None:
            g.waits.append(t - g.queued_at)
            g.queued_at = None
        if not g.fresh and g.switched_out_at is not None:
            g.readmitted_at = t
        if g.fresh:
            g.fresh = False
            g.first_admit = t
            g.prompt = self.requests.first_prompt
            self.push(t + self.cfg.first_request_s, "request", gi)
        else:
            self.push(t, "request", gi)

    def admit_all(self, admitted: list[int], t: float) -> None:
        for gi in admitted:
            self.admit(gi, t)

    def start_request(self, g: SimGame, t: float) -> None:
        if g.n_requests == 0:
            uncached = g.prompt
        elif g.reprefill:
            uncached = g.prompt - self.requests.cached_prefix
        else:
            uncached = g.pending[1] if g.pending else 0.0
        g.reprefill = False
        g.pending = self.requests.sample(g.rng, g.kind, g.prompt)
        g.n_requests += 1
        g.uncached += uncached
        start = max(t, self.cfg.server_ready_s)
        self.push(start + self.serving.prefill_s(uncached), "prefilled", g.index)

    def finish_decode(self, g: SimGame, t: float) -> None:
        completion, increment, first = g.pending
        self.credit(g, completion, t)
        if g.done:
            self.end_holding(g, t)
            self.admit_all(self.gate.release(t), t)
            return
        g.next_prompt = g.prompt + increment
        gap = self.cal.gaps.sample(g.rng, g.name, g.next_prompt, first, g.actions_total())
        self.push(t + gap, "gap", g.index)

    def credit(self, g: SimGame, tokens: float, t: float) -> None:
        g.tokens += tokens
        while tokens > 0 and g.level < g.n_levels:
            d = g.draws[g.level]
            need = d.tokens - g.spent
            if tokens < need:
                g.spent += tokens
                return
            tokens -= need
            g.cleared_tokens += d.tokens
            g.cleared_actions += d.actions
            g.actions_per_level.append(int(d.actions))
            g.level += 1
            g.spent = 0.0
            g.level_up_s.append(t)
            g.marks.append((g.tokens - tokens, level_weighted_reward(g.level, g.n_levels)))
        if g.level >= g.n_levels:
            g.done = True

    def end_holding(self, g: SimGame, t: float) -> None:
        if g.held_since is not None:
            g.active_s += t - g.held_since
        g.held_since = None
        g.holding = False

    def after_gap(self, g: SimGame, t: float) -> None:
        if g.next_prompt > self.budget:
            # trim to the drain level; the next request re-prefills; hand over before it
            g.prompt = self.requests.drain_prompt(g.rng, self.cfg.drain_tokens, self.cfg.ctx_window)
            g.kind = "steady"
            g.reprefill = True
            g.n_trims += 1
            if self.policy.hold(g):
                self.start_request(g, t)
                return
            priority, snap = self.policy.handover(g, t)
            self.end_holding(g, t)
            g.queued_at = t
            admitted = self.gate.handover(g.index, priority, snap, t)
            if g.index not in admitted:
                g.n_switch_outs += 1
                g.switched_out_at, g.readmitted_at = t, None
            self.admit_all(admitted, t)
            return
        g.prompt = g.next_prompt
        self.start_request(g, t)

    def run(self) -> SimResult:
        # every play() acquires at the session start; thread start-up jitter reorders arrivals,
        # which decides the first `slots` admissions and breaks ties between equal prices. A
        # few places of jitter alone cannot give the calibration runs' first waves (temp06 left
        # out index 6 and r1 index 5 while index 10 got in; r2 admitted index 11 first), so a
        # share of games is late by up to `arrival_late_places` more (section 0 of
        # holdout_sim2.py: log-likelihood of the five first waves -16.7 vs -38.9 for jitter alone)
        keys = arrival_keys(len(self.games), random.Random(f"arrival:{self.seed}"), self.cfg)
        order = sorted(self.games, key=lambda g: keys[g.index])
        for g in order:
            priority, snap = self.policy.fresh(g, 0.0)
            g.queued_at = 0.0
            self.admit_all(self.gate.acquire(g.index, priority, snap, 0.0), 0.0)
        while True:
            t_ev = self.events[0][0] if self.events else math.inf
            t_dec = self.next_decode()
            t = min(t_ev, t_dec)
            if t >= self.deadline or t == math.inf:
                break
            self.advance(t)
            self.now = t
            if t_dec <= t_ev:
                _, _, gi = heapq.heappop(self.decoding)
                self.finish_decode(self.games[gi], t)
                continue
            _, _, kind, gi = heapq.heappop(self.events)
            g = self.games[gi]
            if kind == "request":
                self.start_request(g, t)
            elif kind == "prefilled":
                self.seq += 1
                heapq.heappush(self.decoding, (self.vtime + g.pending[0], self.seq, gi))
            elif kind == "gap":
                self.after_gap(g, t)
        self.advance(self.deadline)
        return self.result()

    def result(self) -> SimResult:
        T = self.deadline
        # cut at the deadline: decoding, or prefilled-but-not-yet-decoding, requests
        in_flight = {gi for _, _, gi in self.decoding}
        in_flight |= {gi for _, _, kind, gi in self.events if kind == "prefilled"}
        lost = sum(self.games[gi].pending[0] for gi in in_flight)
        for g in self.games:
            if g.held_since is not None:
                g.active_s += T - g.held_since
                g.held_since = None
        # switched out and not re-admitted at least 120 s before the deadline (the data card's rule)
        never = sum(1 for g in self.games if g.switched_out_at is not None
                    and (g.readmitted_at is None or g.readmitted_at > T - 120.0))
        return SimResult(
            score=st.fmean(g.score() for g in self.games), games=self.games,
            total_tokens=sum(g.tokens for g in self.games), lost_tokens=lost, deadline=T,
            slots=self.cfg.slots, switch_outs=sum(g.n_switch_outs for g in self.games),
            trims=sum(g.n_trims for g in self.games), never_readmitted=never,
            starts=[g.first_admit for g in self.games], mean_decoding=self.busy_integral / T,
            requests=sum(g.n_requests for g in self.games),
            uncached_tokens=sum(g.uncached for g in self.games))


def simulate(cfg: SimConfig, calib: Calibration, seed: int) -> SimResult:
    return Simulator(cfg, calib, seed).run()


# ============================================================================ CLI

def game_list(calib: Calibration, games=None, from_run=None, n_games=None, seed=0) -> list[str]:
    """Dispatch order: the full-run order (benchmark game_runs), a run's panel, or an explicit
    list. n_games > len draws extra copies (with replacement, fixed seed) after the base list."""
    if games:
        base = list(games)
    else:
        base = calib.meta["runs"][from_run or "fz-arms-v4-control"]["dispatch_order"]
    if n_games and n_games != len(base):
        if n_games < len(base):
            return base[:n_games]
        rng = random.Random(f"games:{seed}")
        return base + [rng.choice(base) for _ in range(n_games - len(base))]
    return base


def describe(results: list[SimResult]) -> dict:
    scores = [r.score for r in results]
    sums = [r.summary() for r in results]

    def agg(key):
        vals = [s[key] for s in sums if s[key] is not None]
        return (st.fmean(vals), st.pstdev(vals)) if vals else (math.nan, math.nan)
    per_game = defaultdict(list)
    per_level = defaultdict(list)
    starts = defaultdict(list)
    for r in results:
        for g in r.games:
            key = f"{g.name}#{g.index}"
            per_game[key].append(g.score())
            per_level[key].append(g.level)
            starts[key].append(g.first_admit)
    return {
        "reps": len(results), "score_mean": st.fmean(scores),
        "score_sd": st.stdev(scores) if len(scores) > 1 else 0.0,
        "score_se": (st.stdev(scores) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0,
        "agg": {k: agg(k) for k in sums[0]},
        "per_game": {k: (st.fmean(v), st.fmean(per_level[k]),
                         st.fmean([s for s in starts[k] if s is not None]) if any(s is not None for s in starts[k]) else None)
                     for k, v in per_game.items()},
        "start_sorted_mean": [st.fmean(x) for x in zip(*[sorted(s if s is not None else math.inf for s in r.starts)
                                                         for r in results])],
    }


def print_report(name: str, d: dict, verbose: bool = True) -> None:
    a = d["agg"]
    print(f"\n=== {name}: {d['reps']} reps  score {d['score_mean']:.2f}  sd {d['score_sd']:.2f}  "
          f"se {d['score_se']:.2f}")
    print(f"  levels {a['levels'][0]:.1f}  tokens {a['total_tokens'][0] / 1e6:.2f}M (lost {a['lost_tokens'][0]:.0f})  "
          f"requests {a['requests'][0]:.0f}  tok/active-s {a['tok_per_active_s'][0]:.1f}  "
          f"mean decoding {a['mean_decoding'][0]:.2f}")
    print(f"  trims {a['trims'][0]:.1f}  switch-outs {a['switch_outs'][0]:.1f} (sd {a['switch_outs'][1]:.1f})  "
          f"never re-admitted {a['never_readmitted'][0]:.1f}")
    print(f"  starts: after 30 min {a['started_after_30min'][0]:.1f}  never {a['never_started'][0]:.1f}  "
          f"last {a['last_start_s'][0] / 60:.1f} min")
    s = d["start_sorted_mean"]
    print("  start times (min, sorted, mean over reps): " + " ".join(
        "-" if not math.isfinite(x) else f"{x / 60:.0f}" for x in s))
    if verbose:
        print("  per game (score, levels, first admit min):")
        items = list(d["per_game"].items())
        for i in range(0, len(items), 5):
            print("   " + "  ".join(f"{k.split('#')[0]:>5} {v[0]:5.1f} {v[1]:4.1f} "
                                     f"{(v[2] / 60 if v[2] is not None else math.nan):4.0f}"
                                     for k, v in items[i:i + 5]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs="+", default=["franzen"], choices=sorted(POLICIES))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--minutes", type=float, default=121.0)
    ap.add_argument("--ledger", choices=["on", "off", "all"], default="all",
                    help="level costs from ledger runs, no-ledger runs, or all full runs")
    ap.add_argument("--games", nargs="*", help="explicit games, in dispatch order")
    ap.add_argument("--games-from-run", help="that run's dispatch order (e.g. v4-panel-control)")
    ap.add_argument("--n-games", type=int, help="truncate, or extend with copies (e.g. 110)")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--drain", type=int, default=DEFAULT_DRAIN)
    ap.add_argument("--speed-scale", type=float, default=1.0, help="0.85 ~ 16 experts per token")
    ap.add_argument("--fade", type=float, help="tail fade fraction (default per policy)")
    ap.add_argument("--ucb-c", type=float, default=0.1)
    ap.add_argument("--tail", choices=["never", "pooled"], default="never")
    ap.add_argument("--prior-weight", type=float, default=1.0)
    ap.add_argument("--stuck-floor", type=float, default=0.0,
                    help="probability any level is unclearable on top of the KM draw")
    ap.add_argument("--stuck-shrink", type=float, default=0.0,
                    help="shrink each level's unclearable share toward its depth's mean")
    ap.add_argument("--deep-pool", type=int, default=7,
                    help="pool the prior by level up to this depth; 1 pools every level (depth-neutral)")
    ap.add_argument("--prior-max-n", type=float, default=3.0,
                    help="mix the pooled prior into game-levels with fewer attempts (-1: always)")
    ap.add_argument("--exclude-run", nargs="*", default=[], help="leave these runs out of level costs")
    ap.add_argument("--replay-run", help="keep this run's own outcome on the levels it attempted")
    ap.add_argument("--server-ready", type=float, default=0.0)
    ap.add_argument("--resume-gaps", action="store_true",
                    help="separate fresh-turn and resumption client gaps (GapModel resume_aware)")
    ap.add_argument("--lever", help="prefill lever: a d_<lever> column of calibration/prefill_levers.csv "
                    "(e.g. dedup, y6144, y6144+dedup)")
    ap.add_argument("--calib", choices=sorted(FIT_SETS), default="franzen",
                    help="fit on the Franzen calibration runs only (default) or on every run")
    ap.add_argument("--json", help="write the summaries here")
    ap.add_argument("--quiet", action="store_true", help="no per-game table")
    a = ap.parse_args(argv)
    calib = Calibration(fit=a.calib, resume_gaps=a.resume_gaps)
    games = game_list(calib, a.games, a.games_from_run, a.n_games, a.seed)
    group = {"on": "ledger", "off": "noledger", "all": "all"}[a.ledger]
    out = {}
    per_policy_scores = {}
    for pol in a.policy:
        cfg = SimConfig(games=games, policy=pol, slots=a.slots, minutes=a.minutes, ledger=group,
                        ctx_window=a.ctx, drain_tokens=a.drain, speed_scale=a.speed_scale, fade=a.fade,
                        ucb_c=a.ucb_c, tail=a.tail, prior_weight=a.prior_weight,
                        prior_max_n=None if a.prior_max_n < 0 else a.prior_max_n, stuck_floor=a.stuck_floor,
                        stuck_shrink=a.stuck_shrink, deep_pool=a.deep_pool,
                        exclude_runs=tuple(a.exclude_run), server_ready_s=a.server_ready,
                        replay_run=a.replay_run, lever=a.lever)
        results = [simulate(cfg, calib, a.seed * 100_003 + rep) for rep in range(a.reps)]
        d = describe(results)
        per_policy_scores[pol] = [r.score for r in results]
        print_report(f"{pol} | {len(games)} games, {a.slots} slots, {a.minutes:g} min, levels={group}", d,
                     verbose=not a.quiet)
        out[pol] = d
    if len(a.policy) > 1:
        ref = a.policy[0]
        print("\npaired differences (same level draws per rep):")
        for pol in a.policy[1:]:
            diff = [x - y for x, y in zip(per_policy_scores[pol], per_policy_scores[ref])]
            se = st.stdev(diff) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
            print(f"  {pol} - {ref}: {st.fmean(diff):+.2f} (se {se:.2f})")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
