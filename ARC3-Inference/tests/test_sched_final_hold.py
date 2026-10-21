"""ARC3_SCHED_FINAL_HOLD and ARC3_SCHED_PARK_CAP_MIN on Franzen's priority gate.

Flags off the price and the gate are exactly as before. With the final hold, a game on its last
level gets a floor on B, and at a trim on one of its last levels keeps its slot when its price is
within the hold ratio of the best waiter's. With the park cap, a waiter that has cleared a level
and waited the cap is admitted before every priced waiter, never before a never-started game.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time as real_time
from types import SimpleNamespace

import pytest

from inference.agent import tool_agent as ta
from inference.agent.priority_scheduler import PrioritySnapshot, priority_value
from inference.agent.tool_agent import ToolAgent

KAGGLE_ENV = {
    "ARC3_MAX_ACTIVE_STREAMS": "14",
    "ARC3_PRIORITY_REFRESH_QUEUE": "1",
    "ARC3_PRIORITY_PACE": "0",
    "ARC3_PRIORITY_TAIL_FADE": "1",
    "ARC3_PRIORITY_TAIL_FADE_FRACTION": "0.2",
    "ARC3_PRIORITY_TAIL_LOOKUP": "remaining",
    "ARC3_PRIORITY_SCORE_NORMALIZATION": "1",
}
NEW_VARS = ("ARC3_SCHED_FINAL_HOLD", "ARC3_SCHED_FINAL_HOLD_FLOOR", "ARC3_SCHED_FINAL_HOLD_RATIO",
            "ARC3_SCHED_FINAL_HOLD_LEVELS", "ARC3_SCHED_PARK_CAP_MIN")
BASE = 10_000.0
KAGGLE_PRICE = dict(human_actions=25.0, action_scale=115.0, token_scale=62000.0, action_weight=0.25,
                    tail_base=2.0, tail_cap=4.0, normalize_score=True, tail_lookup="remaining",
                    tail_efficiency=0.8)


@pytest.fixture
def kaggle_env(monkeypatch):
    for key, value in KAGGLE_ENV.items():
        monkeypatch.setenv(key, value)
    for key in NEW_VARS:
        monkeypatch.delenv(key, raising=False)
    clock = [BASE]
    monkeypatch.setattr(ta, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    return clock


def _gate(slots=1, minutes=121.0):
    gate = ta._PriorityGate(slots)
    gate._free = 0                       # every slot held by someone outside the test
    gate.configure_clock(BASE, BASE + minutes * 60.0)
    return gate


def _queue(gate, items, clock=None):
    """Queue waiters as _enqueue_and_wait does, without blocking; token -> name."""
    names = {}
    with gate._cond:
        for name, snap in items:
            gate._seq += 1
            if snap is not None:
                gate._snapshots[gate._seq] = snap
            if ta._park_cap_minutes() > 0:
                gate._queued_at[gate._seq] = clock[0] if clock else BASE
            names[gate._seq] = name
            heapq.heappush(gate._waiting, (-(ta._PRIORITY_UNTRIMMED_BASE if snap is None else 1), gate._seq))
    return names


def _admit_one(gate, names):
    with gate._cond:
        before = set(gate._admitted)
        gate._free += 1
        gate._pump()
        return [names[k] for k in sorted(gate._admitted - before)]


SNAPS = [PrioritySnapshot(level, actions, tokens, 1.0, total)
         for level, actions, tokens, total in itertools.product(
             (1, 2, 5, 6, 7, 10), (0, 20, 140), (0.0, 30_000.0, 150_000.0), (None, 6, 7, 10))]


# ---------------------------------------------------------------- flags off

@pytest.mark.parametrize("fraction", [None, 1.0, 0.3, 0.0])
def test_zero_floor_is_the_old_price(fraction):
    for snap in SNAPS:
        assert (priority_value(snap, tail_fraction=fraction, **KAGGLE_PRICE)
                == priority_value(snap, tail_fraction=fraction, final_floor=0.0, **KAGGLE_PRICE))


def test_flags_off_leave_options_and_gate_alone(kaggle_env):
    assert "final_floor" not in ta._priority_level_options()
    assert not ta._final_hold_enabled() and ta._park_cap_minutes() == 0.0
    gate = _gate()
    snap = PrioritySnapshot(6, 10, 5_000.0, 1.0, 6)
    with gate._cond:
        assert gate._snapshot_priority(snap, BASE + 600) == priority_value(
            snap, tail_fraction=(121 * 60 - 600) / (0.2 * 121 * 60), **KAGGLE_PRICE)
    other = PrioritySnapshot(2, 5, 1_000.0, 1.0, 8)
    names = _queue(gate, [("a", other), ("b", snap)])
    assert not gate._queued_at
    kaggle_env[0] = BASE + 3600
    with gate._cond:
        prices = {"a": gate._snapshot_priority(other, kaggle_env[0]), "b": gate._snapshot_priority(snap, kaggle_env[0])}
    assert _admit_one(gate, names) == [max(prices, key=prices.get)]


# ---------------------------------------------------------------- the floor

def test_floor_only_on_the_last_level_and_after_the_fade(kaggle_env, monkeypatch):
    last = PrioritySnapshot(6, 200, 150_000.0, 1.0, 6)
    before = PrioritySnapshot(5, 200, 150_000.0, 1.0, 6)
    for fraction in (None, 1.0, 0.0):
        plain = priority_value(last, tail_fraction=fraction, **KAGGLE_PRICE)
        floored = priority_value(last, tail_fraction=fraction, final_floor=3.0, **KAGGLE_PRICE)
        assert floored > plain
        assert (priority_value(before, tail_fraction=fraction, final_floor=3.0, **KAGGLE_PRICE)
                == priority_value(before, tail_fraction=fraction, **KAGGLE_PRICE))
    # the floor is not faded: with the tail fully faded the floored price is the same
    assert (priority_value(last, tail_fraction=0.0, final_floor=3.0, **KAGGLE_PRICE)
            == priority_value(last, tail_fraction=1.0, final_floor=3.0, **KAGGLE_PRICE))
    monkeypatch.setenv("ARC3_SCHED_FINAL_HOLD", "1")
    assert ta._priority_level_options()["final_floor"] == 3.0
    monkeypatch.setenv("ARC3_SCHED_FINAL_HOLD_FLOOR", "2.5")
    gate = _gate()
    with gate._cond:
        assert gate._snapshot_priority(last, BASE + 600) == priority_value(
            last, tail_fraction=(121 * 60 - 600) / (0.2 * 121 * 60), final_floor=2.5, **KAGGLE_PRICE)


def test_tr87_style_final_level_price_rises(kaggle_env):
    """A 6-level game deep into its last level: the price that parked it, and with the floor."""
    snap = PrioritySnapshot(6, 120, 70_000.0, 1.0, 6)
    old = priority_value(snap, tail_fraction=1.0, **KAGGLE_PRICE)
    new = priority_value(snap, tail_fraction=1.0, final_floor=3.0, **KAGGLE_PRICE)
    assert old < 60 and new > 2 * old


# ---------------------------------------------------------------- the hold

def _hold_gate(monkeypatch, waiters, ratio=None):
    monkeypatch.setenv("ARC3_SCHED_FINAL_HOLD", "1")
    if ratio is not None:
        monkeypatch.setenv("ARC3_SCHED_FINAL_HOLD_RATIO", str(ratio))
    gate = _gate()
    names = _queue(gate, waiters)
    return gate, names


def test_hold_keeps_the_slot_when_close_to_the_best_waiter(kaggle_env, monkeypatch):
    rival = PrioritySnapshot(3, 10, 20_000.0, 1.0, 8)
    gate, _ = _hold_gate(monkeypatch, [("rival", rival)])
    kaggle_env[0] = BASE + 1800
    with gate._cond:
        best = gate._snapshot_priority(rival, kaggle_env[0])
    mine = PrioritySnapshot(6, 60, 40_000.0, 1.0, 6)
    waiting = list(gate._waiting)
    gate.handover(int(0.8 * best), mine)                 # 0.8 >= 0.75: kept, returns at once
    assert gate._free == 0 and gate._waiting == waiting
    with gate._cond:
        assert not gate._final_hold(int(0.7 * best), mine, kaggle_env[0])
        assert gate._final_hold(best * 2, mine, kaggle_env[0])


def test_hold_needs_one_of_the_last_levels(kaggle_env, monkeypatch):
    gate, _ = _hold_gate(monkeypatch, [("rival", PrioritySnapshot(1, 0, 0.0, 1.0, 8))])
    with gate._cond:
        assert gate._final_hold(10**6 - 1, PrioritySnapshot(5, 0, 0.0, 1.0, 6), BASE)      # last two
        assert not gate._final_hold(10**6 - 1, PrioritySnapshot(4, 0, 0.0, 1.0, 6), BASE)
        assert not gate._final_hold(10**6 - 1, PrioritySnapshot(6, 0, 0.0, 1.0, None), BASE)
    monkeypatch.setenv("ARC3_SCHED_FINAL_HOLD_LEVELS", "1")
    with gate._cond:
        assert not gate._final_hold(10**6 - 1, PrioritySnapshot(5, 0, 0.0, 1.0, 6), BASE)
        assert gate._final_hold(10**6 - 1, PrioritySnapshot(6, 0, 0.0, 1.0, 6), BASE)


def test_hold_never_beats_a_never_started_game(kaggle_env, monkeypatch):
    gate, _ = _hold_gate(monkeypatch, [("fresh", None)])
    with gate._cond:
        assert not gate._final_hold(5_000, PrioritySnapshot(6, 0, 0.0, 1.0, 6), BASE)


def test_hold_with_nobody_waiting(kaggle_env, monkeypatch):
    gate, _ = _hold_gate(monkeypatch, [])
    gate._free = 0
    gate.handover(3, PrioritySnapshot(6, 0, 0.0, 1.0, 6))
    assert gate._free == 0 and not gate._waiting


def test_handover_without_the_flag_queues(kaggle_env, monkeypatch):
    """Flag off: a final-level game hands over and the better waiter takes the slot."""
    gate = _gate()
    rival = PrioritySnapshot(3, 10, 20_000.0, 1.0, 8)
    names = _queue(gate, [("rival", rival)])
    admitted = []

    def other():
        gate.handover(1, PrioritySnapshot(6, 60, 40_000.0, 1.0, 6))
        admitted.append("mine")

    t = threading.Thread(target=other)
    t.start()
    real_time.sleep(0.05)
    assert not admitted and len(gate._waiting) == 1     # rival won the slot, mine waits
    assert [names[k] for k in gate._admitted] == ["rival"]
    with gate._cond:
        gate._admitted.clear()
        gate._free += 1
        gate._pump()
    t.join(timeout=5)
    assert admitted == ["mine"]


def test_maybe_handover_reads_the_level_count_for_the_hold(kaggle_env, monkeypatch):
    monkeypatch.delenv("ARC3_PRIORITY_SCORE_NORMALIZATION")
    monkeypatch.delenv("ARC3_PRIORITY_TAIL_LOOKUP")
    monkeypatch.setenv("ARC3_SCHED_FINAL_HOLD", "1")
    seen = []

    class FakeGate:
        def _snapshot_priority(self, snapshot, now):
            seen.append(snapshot)
            return 9

        def handover(self, estimate, snapshot):
            seen.append(("handover", estimate))

    monkeypatch.setattr(ta, "_priority_gate", lambda: FakeGate())
    agent = SimpleNamespace(
        _context_was_trimmed=True, _last_step_summary={"level": 6, "end_action_num": 40},
        _actions_at_level_start=25, _session_generated_tokens=50_000, _tokens_at_level_start=20_000,
        _progress_pace=ta.ProgressPace(), _priority_total_levels=lambda: 6,
    )
    ToolAgent._maybe_handover(agent)
    snap, handover = seen
    assert (snap.level, snap.actions, snap.tokens, snap.total_levels) == (6, 15, 30_000, 6)
    assert handover == ("handover", 9)


# ---------------------------------------------------------------- the park cap

def test_park_cap_readmits_a_long_waiter(kaggle_env, monkeypatch):
    monkeypatch.setenv("ARC3_SCHED_PARK_CAP_MIN", "10")
    gate = _gate()
    parked = PrioritySnapshot(2, 100, 120_000.0, 1.0, 8)       # cleared one level, cheap now
    keen = PrioritySnapshot(4, 5, 2_000.0, 1.0, 8)             # deeper and fresh on its level
    names = _queue(gate, [("parked", parked)], kaggle_env)
    kaggle_env[0] = BASE + 300
    names.update(_queue(gate, [("keen", keen)], kaggle_env))
    with gate._cond:
        assert gate._snapshot_priority(keen, kaggle_env[0]) > gate._snapshot_priority(parked, kaggle_env[0])
    kaggle_env[0] = BASE + 500                                   # parked has waited 8.3 min
    assert _admit_one(gate, names) == ["keen"]
    names.update(_queue(gate, [("keen2", keen)], kaggle_env))
    kaggle_env[0] = BASE + 601                                   # 10 min: the cap
    assert _admit_one(gate, names) == ["parked"]
    assert not gate._queued_at.get(1)


def test_park_cap_skips_level_one_and_fresh_games(kaggle_env, monkeypatch):
    monkeypatch.setenv("ARC3_SCHED_PARK_CAP_MIN", "10")
    gate = _gate()
    names = _queue(gate, [("l1", PrioritySnapshot(1, 100, 120_000.0, 1.0, 8)),
                          ("l3", PrioritySnapshot(3, 100, 120_000.0, 1.0, 8))], kaggle_env)
    kaggle_env[0] = BASE + 900
    names.update(_queue(gate, [("fresh", None), ("keen", PrioritySnapshot(4, 5, 2_000.0, 1.0, 8))], kaggle_env))
    kaggle_env[0] = BASE + 1200
    assert _admit_one(gate, names) == ["fresh"]
    assert _admit_one(gate, names) == ["l3"]
    assert _admit_one(gate, names) == ["keen"]
    assert _admit_one(gate, names) == ["l1"]
    assert not gate._queued_at and not gate._snapshots


# ---------------------------------------------------------------- threads

@pytest.mark.parametrize("env", [{}, {"ARC3_SCHED_FINAL_HOLD": "1"}, {"ARC3_SCHED_PARK_CAP_MIN": "0.0001"},
                                 {"ARC3_SCHED_FINAL_HOLD": "1", "ARC3_SCHED_PARK_CAP_MIN": "0.0001"}])
def test_threaded_games_all_finish(env, monkeypatch):
    """Eight game threads on three slots: acquire, hand over at five trims (the last ones on the
    final levels), release; every game finishes and the gate is clean."""
    for key, value in {**KAGGLE_ENV, **env}.items():
        monkeypatch.setenv(key, value)
    for key in NEW_VARS:
        if key not in env:
            monkeypatch.delenv(key, raising=False)
    gate = ta._PriorityGate(3)
    start = real_time.monotonic()
    done = []

    def game(i):
        gate.configure_clock(start, start + 60.0)
        gate.acquire(ta._PRIORITY_UNTRIMMED_BASE - 100 * i)
        for trim in range(5):
            real_time.sleep(0.002)
            snap = PrioritySnapshot(min(6, 2 + trim), 10 * trim, 20_000.0 * (1 + i % 3), 1.0, 6)
            gate.handover(gate._snapshot_priority(snap, real_time.monotonic()), snap)
        gate.release()
        done.append(i)

    threads = [threading.Thread(target=game, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert sorted(done) == list(range(8))
    assert gate._free == 3 and not gate._waiting and not gate._snapshots and not gate._queued_at
