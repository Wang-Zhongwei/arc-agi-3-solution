"""Unit tests for sim2.py: scorer parity, level-cost sampling, gate semantics, event loop.

Pure python. Run from analysis/ with either
    python3 test_sim2.py                    (built-in runner, no pytest needed)
    python -m pytest test_sim2.py
"""
from __future__ import annotations

import csv
import math
import random
import statistics as st
from collections import defaultdict
from types import SimpleNamespace

import sim2
from sim2 import Gate, LevelDraw, Obs, SimConfig, Simulator, WeightedKM

CAL = None


def calib() -> sim2.Calibration:
    global CAL
    if CAL is None:
        CAL = sim2.Calibration()
    return CAL


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# D' and easy-first call priority_scheduler functions that live on the stacked scheduler
# arms (d_prime_priority, easy_boost), not on main. Their tests skip where those are absent.
STACKED_SCHEDULER = all(hasattr(sim2.PS, n) for n in ("d_prime_priority", "easy_boost"))


def needs_stacked_scheduler(fn):
    def wrapper():
        if not STACKED_SCHEDULER:
            try:
                import pytest
            except ImportError:
                print(f"skip  {fn.__name__}: needs the D'/easy-first priority_scheduler")
                return
            pytest.skip("needs the D'/easy-first priority_scheduler (stacked arms)")
        fn()
    wrapper.__name__ = fn.__name__
    return wrapper


# ============================================================================ scorer

def test_scorer_reproduces_every_benchmark_final_score():
    """actions_per_level rebuilt from levels.csv, scored, against benchmark.json final_score."""
    meta = calib().meta
    actions = defaultdict(dict)
    for r in csv.DictReader(open(sim2.CALIB / "levels.csv")):
        if r["cleared"] == "1":
            actions[(r["run"], r["game"])][int(r["level"])] = int(float(r["actions"]))
    checked = 0
    for run, info in meta["runs"].items():
        for game, final in info["final_score"].items():
            g = meta["games"][game]
            done = info["levels_completed"][game]
            apl = [actions[(run, game)].get(i + 1, 0) for i in range(g["levels"])]
            assert close(sim2.game_score(g["levels"], done, apl, g["baselines"]), final), (run, game)
            checked += 1
    assert checked >= 18 * 10


def test_scorer_caps_level_and_game():
    # one action on a 10-action level scores 115, but the game is capped at the weight share
    assert sim2.level_score(10, 1) == 115.0
    assert close(sim2.game_score(3, 1, [1, 0, 0], [10, 10, 10]), 100.0 / 6)
    assert close(sim2.game_score(3, 3, [5, 10, 20], [10, 10, 10]), (115 * 1 + 100 * 2 + 25 * 3) / 6)
    assert sim2.game_score(4, 0, [], [1, 1, 1, 1]) == 0.0


# ============================================================================ Kaplan-Meier

def test_km_without_censoring_is_the_empirical_distribution():
    km = WeightedKM([Obs(t, True, a) for t, a in [(1, 10), (2, 20), (3, 30), (4, 40)]])
    assert [round(km.survival(t), 9) for t in (0, 1, 2.5, 4)] == [1.0, 0.75, 0.5, 0.0]
    rng = random.Random(1)
    counts = defaultdict(int)
    for _ in range(20000):
        t, o = km.sample(rng)
        assert o.actions == 10 * t          # the clear's own action count comes with it
        counts[t] += 1
    assert all(abs(c / 20000 - 0.25) < 0.015 for c in counts.values())


def test_km_with_censoring_matches_the_textbook_estimate():
    km = WeightedKM([Obs(1, True), Obs(2, False), Obs(3, True), Obs(4, True)])
    assert close(km.survival(1), 0.75)
    assert close(km.survival(3), 0.75 * 0.5)
    assert km.survival(4) == 0.0 and km.residual == 0.0


def test_km_residual_mass_is_unclearable():
    assert WeightedKM([Obs(5, False), Obs(9, False)]).sample(random.Random(0)) is None
    km = WeightedKM([Obs(1, True), Obs(5, False)])          # S = 1/2 beyond the last clear
    rng = random.Random(2)
    nones = sum(km.sample(rng) is None for _ in range(20000))
    assert abs(nones / 20000 - 0.5) < 0.02


def test_km_conditional_draws_exceed_the_spend():
    km = WeightedKM([Obs(t, True) for t in range(1, 11)])
    rng = random.Random(3)
    draws = [km.sample(rng, after=6.5) for _ in range(4000)]
    assert all(d[0] > 6.5 for d in draws)
    assert abs(sum(d[0] == 7 for d in draws) / 4000 - 0.25) < 0.03


def test_km_weights_count_as_repeated_observations():
    a = WeightedKM([Obs(1, True, weight=2.0), Obs(2, True), Obs(3, False)])
    b = WeightedKM([Obs(1, True), Obs(1, True), Obs(2, True), Obs(3, False)])
    for t in (0.5, 1, 2, 3):
        assert close(a.survival(t), b.survival(t))


# ============================================================================ level model

def _rows(spec):
    """levels.csv-like rows from (run, ledger, game, level, tokens, actions, cleared)."""
    return [{"run": r, "kind": "full", "ledger": str(l), "game": g, "level": str(lv),
             "tokens": str(t), "actions": str(a), "cleared": "1" if c else "0"}
            for r, l, g, lv, t, a, c in spec]


META = {"games": {"aa00": {"levels": 3, "baselines": [10, 20, 30]},
                  "bb00": {"levels": 2, "baselines": [5, 5]}}}


def test_levels_drawn_independently_never_from_own_earlier_levels():
    spec = []
    for i in range(8):            # level 1 and 2 costs anti-aligned across runs on purpose
        spec += [(f"r{i}", 0, "aa00", 1, 1000 * (i + 1), 10, True),
                 (f"r{i}", 0, "aa00", 2, 1000 * (8 - i), 20, True),
                 (f"r{i}", 0, "aa00", 3, 500, 30, True)]
    model = sim2.LevelModel(_rows(spec), META, group="noledger")
    rng = random.Random(4)
    games = [model.sample_game("aa00", rng) for _ in range(4000)]
    x = [g[0].tokens for g in games]
    y = [g[1].tokens for g in games]
    mx, my = st.fmean(x), st.fmean(y)
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    assert abs(r) < 0.05                      # the source runs had r = -1
    assert all(g[2].tokens == 500 and g[2].actions == 30 for g in games)


def test_ledger_condition_uses_ledger_runs():
    spec = [(f"L{i}", 1, "aa00", 1, 1000, 10, True) for i in range(3)]
    spec += [(f"N{i}", 0, "aa00", 1, 9000, 90, True) for i in range(3)]
    rng = random.Random(5)
    led = sim2.LevelModel(_rows(spec), META, group="ledger")
    off = sim2.LevelModel(_rows(spec), META, group="noledger")
    assert {led.sample_level("aa00", 1, rng).tokens for _ in range(200)} == {1000.0}
    assert {off.sample_level("aa00", 1, rng).tokens for _ in range(200)} == {9000.0}
    # a level the group never attempted borrows the other group's attempts at half weight
    sparse = _rows([("N0", 0, "aa00", 2, 4000, 40, True)] + spec)
    assert led.attempts("aa00", 2) == 0
    assert sim2.LevelModel(sparse, META, group="ledger").attempts("aa00", 2) == 0.5


def test_stuck_floor_makes_any_level_unclearable_with_its_probability():
    spec = [(f"r{i}", 0, "aa00", 1, 1000 * (i + 1), 10, True) for i in range(3)]
    model = sim2.LevelModel(_rows(spec), META, group="noledger", stuck_floor=0.25)
    rng = random.Random(0)
    draws = [model.sample_level("aa00", 1, rng) for _ in range(4000)]
    stuck = sum(1 for d in draws if not math.isfinite(d.tokens)) / len(draws)
    assert 0.22 < stuck < 0.28 and {d.source for d in draws} == {"never", "game"}
    assert close(model.survival("aa00", 1, 2500.0), 0.25 + 0.75 / 3)
    # beyond every clear only the floor is left: a replayed stuck level stays stuck
    assert all(not math.isfinite(model.sample_level("aa00", 1, rng, after=5000.0).tokens) for _ in range(50))
    plain = sim2.LevelModel(_rows(spec), META, group="noledger")
    assert all(math.isfinite(plain.sample_level("aa00", 1, rng).tokens) for _ in range(200))


def test_stuck_shrink_moves_unclearable_mass_toward_the_depth_mean():
    # aa00 L1 always clears; bb00 L1 is stuck in half the runs: depth-1 mean residual 0.25
    spec = [(f"r{i}", 0, "aa00", 1, 1000 * (i + 1), 10, True) for i in range(4)]
    spec += [(f"r{i}", 0, "bb00", 1, 1000 * (i + 1), 10, i < 2) for i in range(4)]
    spec = [x if x[4] < 3000 or x[6] else x[:4] + (90_000,) + x[5:] for x in spec]
    base = sim2.LevelModel(_rows(spec), META, group="noledger", prior_max_n=0)
    assert close(base.km("aa00", 1).residual, 0.0) and close(base.km("bb00", 1).residual, 0.5)
    half = sim2.LevelModel(_rows(spec), META, group="noledger", prior_max_n=0, stuck_shrink=0.5)
    rbar = half.mean_residual(1)
    assert close(half.stuck_probability("aa00", 1), 0.5 * rbar)
    assert close(half.stuck_probability("bb00", 1), 0.25 + 0.5 * rbar)
    rng = random.Random(1)
    d = [half.sample_level("aa00", 1, rng) for _ in range(4000)]
    assert abs(sum(not math.isfinite(x.tokens) for x in d) / len(d) - 0.5 * rbar) < 0.03
    assert all(x.tokens in (1000, 2000, 3000, 4000) for x in d if math.isfinite(x.tokens))
    # survival is continuous with the clears and ends at the stuck probability
    assert close(half.survival("bb00", 1, 0.0), 1.0) and close(half.survival("bb00", 1, 1e9), half.stuck_probability("bb00", 1))


def test_unattempted_level_draws_from_the_rescaled_pool():
    # bb00 is twice as expensive as aa00 at level 1; aa00 level 2 is the only level-2 data
    spec = [(f"r{i}", 0, "aa00", 1, 1000, 10, True) for i in range(4)]
    spec += [(f"r{i}", 0, "bb00", 1, 2000, 5, True) for i in range(4)]
    spec += [(f"r{i}", 0, "aa00", 2, 3000, 40, True) for i in range(4)]
    model = sim2.LevelModel(_rows(spec), META, group="all")
    rng = random.Random(6)
    d = model.sample_level("bb00", 2, rng)
    assert d.source == "prior"
    # scaled by bb00/aa00 difficulty; actions keep aa00's actions/baseline ratio (40/20 = 2)
    assert d.tokens > 3000 and d.actions == 2 * 5
    # a game-level with >= prior_max_n attempts ignores the prior entirely
    assert {model.sample_level("aa00", 1, rng).source for _ in range(200)} == {"game"}


def test_replay_keeps_a_runs_own_costs():
    spec = [(f"r{i}", 0, "aa00", 1, 1000 * (i + 1), 10 + i, True) for i in range(4)]
    spec += [("r0", 0, "aa00", 2, 7000, 50, False)]
    spec += [(f"r{i}", 0, "aa00", 2, 1000 * (i + 5), 20, True) for i in range(1, 4)]
    model = sim2.LevelModel(_rows(spec), META, group="all")
    rng = random.Random(7)
    for _ in range(100):
        g = model.sample_game("aa00", rng, replay_run="r0")
        assert (g[0].tokens, g[0].actions, g[0].source) == (1000.0, 10.0, "replay")
        assert g[1].tokens > 7000                  # the level r0 was stuck on: beyond its spend


# ============================================================================ gate

class _Const(sim2.Policy):
    """Snapshot = a number, or a function of time for re-pricing tests."""
    refresh = True

    def reprice(self, snap, now):
        return int(snap(now) if callable(snap) else snap)


def test_fresh_games_start_in_dispatch_order_before_requeued_ones():
    pol = sim2.Franzen(deadline=7260.0)
    gate = Gate(1, pol)
    games = [SimpleNamespace(index=i) for i in range(4)]
    assert gate.acquire(0, *pol.fresh(games[0], 0), 0.0) == [0]
    for g in games[3:0:-1]:                       # acquire in reverse: order must not matter
        assert gate.acquire(g.index, *pol.fresh(g, 0), 0.0) == []
    # game 0 trims with the highest re-queue price there is; the fresh band still wins
    snap = sim2.PS.PrioritySnapshot(10, 0, 0.0, 1.0, 10)
    assert gate.handover(0, pol.reprice(snap, 0), snap, 0.0) == [1]
    assert gate.release(1.0) == [2]
    assert gate.release(2.0) == [3]
    assert gate.release(3.0) == [0]


def test_handover_can_win_its_slot_straight_back():
    gate = Gate(1, _Const(100.0))
    assert gate.acquire(0, 5, 5, 0.0) == [0]
    assert gate.acquire(1, 3, 3, 0.0) == []
    assert gate.handover(0, 4, 4, 1.0) == [0]      # still above the waiter: a repick
    assert gate.handover(0, 2, 2, 2.0) == [1]      # now below: switched out
    assert gate.queued() == [0]


def test_ties_go_to_the_earlier_arrival():
    gate = Gate(1, _Const(100.0))
    gate.acquire(0, 9, 9, 0.0)
    gate.acquire(1, 5, 5, 0.0)
    gate.acquire(2, 5, 5, 0.0)
    assert gate.handover(0, 5, 5, 1.0) == [1]


def test_refreshed_queue_reprices_waiters_at_each_pump():
    gate = Gate(1, _Const(100.0))
    gate.acquire(0, 9, 9, 0.0)
    gate.acquire(1, 5, lambda now: 5 if now < 10 else 1, 0.0)
    gate.acquire(2, 3, 3, 0.0)
    assert gate.release(20.0) == [2]               # game 1 decayed below game 2 by t = 20
    no_refresh = Gate(1, sim2.FIFO(100.0))
    no_refresh.acquire(0, 9, None, 0.0)
    no_refresh.acquire(1, 5, None, 0.0)
    no_refresh.acquire(2, 3, None, 0.0)
    assert no_refresh.release(20.0) == [1]


def test_tail_fade_uses_the_shared_clock():
    pol = sim2.Franzen(deadline=1000.0, fade=0.2)
    assert pol.tail_fraction(0.0, 0.2) == 5.0       # priority_value clips it to 1
    assert close(pol.tail_fraction(900.0, 0.2), 0.5)
    snap = sim2.PS.PrioritySnapshot(2, 10, 5000.0, 1.0, 8)
    assert pol.reprice(snap, 900.0) == sim2.PS.priority_value(
        snap, tail_fraction=0.5, **sim2.FRANZEN_CONFIG)
    assert pol.reprice(snap, 950.0) < pol.reprice(snap, 900.0)


@needs_stacked_scheduler
def test_dprime_ranks_fresh_games_by_its_formula_unless_fresh_first():
    d = sim2.DPrime(deadline=7260.0)
    six, ten = SimpleNamespace(index=5, n_levels=6), SimpleNamespace(index=0, n_levels=10)
    assert d.fresh(six, 0)[0] > d.fresh(ten, 0)[0]
    assert d.fresh(six, 0)[0] < sim2.PRIORITY_BAND
    ff = sim2.DPrime(deadline=7260.0, fresh_first=True)
    assert ff.fresh(ten, 0) == (sim2.PRIORITY_UNTRIMMED_BASE, None)


def test_ucb_matches_the_ff0630c_tests():
    lwr = sim2.level_weighted_reward
    assert lwr(0, 6) == 0.0 and lwr(6, 6) == 1.0 and lwr(9, 6) == 1.0
    assert close(lwr(2, 6), 3 / 21)
    snap = lambda n, r, rec=None: sim2.UCBSnapshot(n, r, rec)  # noqa: E731
    total = 4.5e6
    good = sim2.ucb_value(snap(180e3, lwr(4, 6)), total_tokens=total)
    stalled = sim2.ucb_value(snap(250e3, lwr(1, 6), 0.0), total_tokens=total)
    fresh = sim2.ucb_value(snap(0, 0.0), total_tokens=total)
    assert good > fresh > stalled
    assert sim2.ucb_value(snap(1, 1.0), total_tokens=1e9, c=50.0) == sim2.UCB_VALUE_CAP
    # recent rate from level-up marks: window 64k, cutoff 136k, 2/21 gained over 0.64 units
    pol = sim2.UCB(deadline=7260.0)
    g = SimpleNamespace(index=0, tokens=200_000, level=2, n_levels=6,
                        marks=[(50_000, lwr(1, 6)), (180_000, lwr(2, 6))])
    s = pol.snapshot(g)
    assert s.reward == lwr(2, 6) and close(s.recent_rate, (2 / 21) / 0.64)
    # T counts each game's tokens as recorded at its last handover
    pol.handover(g, 0.0)
    pol.handover(SimpleNamespace(index=1, tokens=50_000, level=0, n_levels=6, marks=[]), 0.0)
    assert sum(pol.recorded.values()) == 250_000


@needs_stacked_scheduler
def test_easy_first_holds_its_slot_on_a_streak():
    pol = sim2.EasyFirst(deadline=7260.0)
    cheap = SimpleNamespace(cleared_levels=2, cleared_tokens=20_000.0, spent=5_000.0)
    stuck = SimpleNamespace(cleared_levels=2, cleared_tokens=20_000.0, spent=40_000.0)
    assert pol.hold(cheap) and not pol.hold(stuck)


def test_drain_level_follows_the_retained_target():
    """Default setting: unchanged. A smaller drain keeps more history (80k at drain 38k)."""
    assert sim2.drain_target(sim2.DEFAULT_CTX, sim2.DEFAULT_DRAIN) == 59_136   # 58 Ki capped at budget/2
    assert sim2.drain_target(sim2.DEFAULT_CTX, 38 * 1024) == 79_360
    req = calib().requests
    rng = random.Random(0)
    obs = req.drain_prompt(random.Random(0), sim2.DEFAULT_DRAIN)
    assert obs == random.Random(0).choice(req.drain_prompts)
    more = req.drain_prompt(random.Random(0), 38 * 1024)
    assert close(more / obs, 79_360 / 59_136)
    # the calibration runs all drain to half the budget, so their scaling barely moves
    s12 = req.drain_prompt(rng, 50176, 112640) / random.Random(0).choice(req.drain_prompts)
    assert abs(s12 - 50176 / 59392) < 0.01


# ============================================================================ event loop

class _Req:
    """Fixed request shape: every request is `c` tokens and grows the prompt by `dp`."""

    def __init__(self, c=1000, dp=1000, first_prompt=1000.0, drain=500.0):
        self.c, self.dp, self.first_prompt, self.drain = c, dp, first_prompt, drain
        self.cached_prefix = 0.0

    def sample(self, rng, kind, prompt):
        return (self.c, self.dp, True)

    def drain_prompt(self, rng, drain_tokens, ctx_window=None):
        return self.drain


class _Gap:
    def __init__(self, gap=1.0):
        self.gap = gap

    def sample(self, rng, game, prompt, first, actions):
        return self.gap


class _Levels:
    def __init__(self, costs):
        self.costs = costs

    def sample_game(self, name, rng, replay_run=None):
        return [LevelDraw(t, 10.0, 10.0 / t if math.isfinite(t) else 0.01, "test") for t in self.costs[name]]


def _stub(costs, curve, c=1000, dp=1000, gap=1.0, first_prompt=1000.0, drain=500.0):
    meta = {"games": {n: {"levels": len(v), "baselines": [10] * len(v)} for n, v in costs.items()}}
    cal = SimpleNamespace(meta=meta, requests=_Req(c, dp, first_prompt, drain), gaps=_Gap(gap),
                          curves={"std": curve, "s12": curve, "s14": curve})
    return cal, _Levels(costs)


def _sim(cal, levels, games, **kw):
    kw.setdefault("ctx_window", 10**9)          # no trims unless asked
    kw.setdefault("arrival_jitter", 0.0)
    kw.setdefault("arrival_late_p", 0.0)
    cfg = SimConfig(games=games, first_request_s=0.0, **kw)
    s = Simulator(cfg, cal, seed=0, levels=levels)
    s.serving.prefill_overhead_s = 0.0
    s.serving.prefill_tps = 1e12
    return s


def test_single_game_timeline_is_prefill_decode_gap():
    # 1000 tokens at 100 tok/s = 10 s per request, 1 s gap: a request completes every 11 s
    cal, lv = _stub({"aa00": [2500.0, math.inf]}, {1: 100.0})
    s = _sim(cal, lv, ["aa00"], slots=1, minutes=1.0)          # 60 s
    res = s.run()
    g = res.games[0]
    assert g.tokens == 5000                    # completions at 10, 21, 32, 43, 54 s
    assert g.level == 1 and close(g.level_up_s[0], 32.0)
    assert close(g.spent, 5000 - 2500)          # overflow carries into level 2
    assert res.lost_tokens == 1000              # the 6th request (from 55 s) is cut at 60 s
    assert close(res.mean_decoding, 55.0 / 60.0)


def test_decode_is_processor_sharing_over_running_streams():
    cal, lv = _stub({"aa00": [math.inf], "bb00": [math.inf]}, {1: 100.0, 2: 50.0}, gap=1000.0)
    s = _sim(cal, lv, ["aa00", "bb00"], slots=2, minutes=1.0)
    res = s.run()
    # both decode together at 50 tok/s each: 1000 tokens in 20 s, then a 1000 s gap
    assert [g.tokens for g in res.games] == [1000, 1000]
    cal, lv = _stub({"aa00": [math.inf], "bb00": [math.inf]}, {1: 100.0, 2: 50.0}, gap=1000.0, c=2000)
    s = _sim(cal, lv, ["aa00", "bb00"], slots=2, minutes=0.5)   # 30 s: 2000 tokens need 40 s
    res = s.run()
    assert [g.tokens for g in res.games] == [0, 0] and res.lost_tokens == 4000


def test_trim_hands_over_and_waiting_counts_against_the_clock():
    # budget 2500: the prompt (1000, +1000 per request) passes it after 2 requests -> trim
    cal, lv = _stub({"aa00": [math.inf], "bb00": [math.inf]}, {1: 100.0})
    s = _sim(cal, lv, ["aa00", "bb00"], slots=1, minutes=10.0, policy="fifo",
             ctx_window=2500 + sim2.OUTPUT_RESERVE)
    res = s.run()
    a, b = res.games
    assert a.first_admit == 0.0
    # aa00: requests end at 10, 21 (prompt 3000 > 2500 after the gap at 22 s) -> hands over
    assert close(b.first_admit, 22.0)
    assert a.n_switch_outs >= 1 and a.n_trims >= 1
    assert res.trims == a.n_trims + b.n_trims
    # nobody runs past the per-game deadline, and slot time never exceeds slots x T
    assert sum(g.active_s for g in res.games) <= 600.0 + 1e-6


def test_uncached_counts_first_prompt_increments_and_reprefills():
    cal, lv = _stub({"aa00": [math.inf]}, {1: 100.0})
    s = _sim(cal, lv, ["aa00"], slots=1, minutes=1.0, ctx_window=2500 + sim2.OUTPUT_RESERVE)
    res = s.run()
    g = res.games[0]
    # first prompt 1000, then +1000 per request; a trim to 500 re-prefills 500 (no cached prefix)
    expect = 1000.0 + 1000.0 * (g.n_requests - 1 - g.n_trims) + 500.0 * g.n_trims
    assert g.n_trims >= 1 and close(g.uncached, expect) and close(res.uncached_tokens, expect)


# ============================================================================ prefill levers

def test_competition_yield_replay_is_the_identity_lever():
    """Replaying the yields at the competition's 2,048 tokens reproduces every real resumption,
    so the y2048 lever must leave the calibrated request model unchanged, and the lever table
    joins to every request pair."""
    base, same = calib().requests, calib().request_model("y2048")
    assert same.cells == base.cells
    assert same.adjusted == sum(len(v) for v in base.cells.values())


def test_dedup_lever_removes_exactly_the_tabled_tokens():
    base, ded = calib().requests, calib().request_model("dedup")
    rows = list(csv.DictReader(open(sim2.CALIB / "prefill_levers.csv")))
    tabled = -sum(int(float(r["d_dedup"])) for r in rows if r["trim"] == "0" and r["idx"] != "0"
                  and r["answered"] == "1")
    grow = lambda m: sum(dp for v in m.cells.values() for _, dp, _ in v)   # noqa: E731
    assert 0 < grow(base) - grow(ded) <= tabled
    assert grow(base) - grow(ded) > 0.95 * tabled          # only clipping at 0 may differ
    # dedup does not move invocation boundaries; a larger yield budget removes some
    first = lambda m: sum(f for v in m.cells.values() for _, _, f in v)    # noqa: E731
    assert first(ded) == first(base)
    assert first(calib().request_model("y6144")) < first(base)


def test_resume_aware_gaps_split_fresh_turns_from_resumptions():
    """A resumption's client gap is close to a mid-invocation one; a fresh turn's holds the
    env step. The pooled default keeps one turn-start term (phase-3 calibration unchanged)."""
    pooled, aware = calib().gaps, sim2.Calibration(resume_gaps=True).gaps
    assert len(pooled.beta) == 6 and not pooled.resume_aware
    fresh, resume = aware.beta[4], aware.beta[5]
    assert fresh > pooled.beta[4] > resume > 0
    x = aware.features(50_000, sim2.RESUME, 100, True)
    assert x[4:6] == [0.0, 1.0]
    assert aware.features(50_000, True, 100, True)[4:6] == [1.0, 0.0]      # a bool turn start is fresh
    assert pooled.features(50_000, sim2.RESUME, 100) == pooled.features(50_000, True, 100)
    kinds = [k for v in calib().requests.cells.values() for _, _, k in v]
    assert kinds.count(sim2.RESUME) > 300 and kinds.count(sim2.FRESH) > 900


def test_unknown_lever_is_refused():
    try:
        calib().request_model("no-such-lever")
    except ValueError as exc:
        assert "no-such-lever" in str(exc)
    else:
        raise AssertionError("an unknown lever must raise")


def test_a_finished_game_releases_its_slot():
    cal, lv = _stub({"aa00": [500.0], "bb00": [math.inf]}, {1: 100.0})
    s = _sim(cal, lv, ["aa00", "bb00"], slots=1, minutes=1.0)
    res = s.run()
    a, b = res.games
    assert a.done and a.score() == 100.0        # 10 actions on a 10-action level
    assert close(b.first_admit, 10.0)          # admitted the moment aa00 won


@needs_stacked_scheduler
def test_simulator_is_deterministic_and_uses_common_level_draws():
    games = calib().meta["runs"]["fz-arms-v4-control"]["dispatch_order"]
    a = sim2.simulate(SimConfig(games=games, policy="franzen"), calib(), 3)
    b = sim2.simulate(SimConfig(games=games, policy="franzen"), calib(), 3)
    c = sim2.simulate(SimConfig(games=games, policy="dprime"), calib(), 3)
    assert a.score == b.score and a.total_tokens == b.total_tokens
    assert [[d.tokens for d in g.draws] for g in a.games] == [[d.tokens for d in g.draws] for g in c.games]


def test_calibrated_run_reproduces_the_franzen_timeline():
    """Smoke check of the whole calibration: 10 slots, 25 games, 121 min."""
    games = calib().meta["runs"]["fz-arms-v4-control"]["dispatch_order"]
    res = [sim2.simulate(SimConfig(games=games, ledger="noledger"), calib(), s) for s in range(4)]
    for r in res:
        starts = sorted(r.starts)
        assert starts[:10] == [0.0] * 10 and 600 < starts[10] < 1500 and 1500 < starts[-1] < 2400
        assert 4.2e6 < r.total_tokens < 4.9e6
        assert 30 <= r.switch_outs <= 70


def test_arrival_model_gives_the_calibration_first_waves():
    games = calib().meta["runs"]["fz-arms-v4-control"]["dispatch_order"]
    exact = sim2.simulate(SimConfig(games=games, arrival_jitter=0.0, arrival_late_p=0.0, minutes=1.0), calib(), 0)
    assert [g.index for g in exact.games if g.first_admit == 0.0] == list(range(10))
    # the five calibration runs: 1-2 of indices 0-9 lose their first-wave slot (6 in 50), never
    # index 0-4 except r1's 5; the deepest entrant was index 11
    missing, deepest, waves = [], [], set()
    for seed in range(200):
        r = sim2.simulate(SimConfig(games=games, minutes=0.2), calib(), seed)
        wave = {g.index for g in r.games if g.first_admit == 0.0}
        assert len(wave) == 10
        missing.append(len(set(range(10)) - wave))
        deepest.append(max(wave))
        waves.add(frozenset(wave))
    assert 0.8 < st.fmean(missing) < 2.0
    assert st.median(deepest) in (10, 11, 12) and max(deepest) <= 20
    for real in (set(range(9)) | {10}, set(range(11)) - {6}, set(range(11)) - {5}, set(range(9)) | {11}):
        assert frozenset(real) in waves


def test_cli_runs():
    sim2.main(["--reps", "2", "--quiet", "--policy", "franzen", "ucb", "--n-games", "12",
               "--minutes", "10"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
