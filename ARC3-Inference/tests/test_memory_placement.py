"""memory-v2 placement (ARC3_MEMORY_PLACEMENT) through the real ToolAgent.analyze().

Each config plays ``tests/memory_sim.py``'s scripted 3-level game under the
Kaggle knobs with every memory-v2 component on, in its own interpreter (some
knobs are read at import time), and checks what the model was sent:

- trim: the card only at the head after a real trim, exactly once, byte-stable
  until the next trim; no ``<memory_update>`` lines.
- delta: no head card; ``<memory_update>`` lines only in new turn messages, so
  every request extends the previous one until a trim.
- both: the head card plus deltas, which together cover the full card.
- the per-turn block sits after the valid-actions line (where the facts-stack
  lines go), the system message never changes, the per-turn hold lets only
  solved/first-death lines through, a rolled-back turn gives its delta back,
  a resumed turn gets no lines, and a level change resets what is per level.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import memory_sim  # noqa: E402

MEM_ON = {
    **memory_sim.KAGGLE_ENV,
    "ARC3_LEDGER_ON_TRIM": "1",
    "ARC3_MEM_SOLVED": "1",
    "ARC3_MEM_CARD": "1",
    "ARC3_MEM_EFFECTS": "1",
    "ARC3_MEM_SEEN": "1",
    "ARC3_MEM_LINES_AFTER_TOKENS": "0",
}
CARD_RE = re.compile(r"\A<memory>\n.*?\n</memory>", re.S)
DELTA_RE = re.compile(r"<memory_update>\n(.*?)\n</memory_update>", re.S)
BANNED = re.compile(r"\b(inert|refuted|impossible|useless|cannot)\b", re.I)


def _lower_priority() -> None:
    try:
        os.nice(10)
    except OSError:
        pass


@lru_cache(maxsize=None)
def _dump(extra: tuple[tuple[str, str], ...] = (), fail: int = -1, turns: int = 0) -> dict:
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_", "MEMSIM_"))
    }
    env.update(MEM_ON)
    env.update(dict(extra))
    env = {k: v for k, v in env.items() if v is not None and v != "<unset>"}
    proc = subprocess.run(
        [sys.executable, "tests/memory_sim.py", "dump", str(fail), str(turns)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, preexec_fn=_lower_priority,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _placement(name: str, **kw) -> dict:
    return _dump((("ARC3_MEMORY_PLACEMENT", name),), **kw)


def _openers(d: dict) -> list[tuple[int, dict]]:
    """(request index, last message) for each turn's first request."""
    out, index = [], 0
    for count in d["turn_requests"]:
        if count:
            out.append((index, d["requests"][index][-1]))
        index += count
    return out


def _extends(prev: list[dict], cur: list[dict]) -> bool:
    return [m["key"] for m in cur[: len(prev)]] == [m["key"] for m in prev]


def _card_count(request: list[dict]) -> int:
    return sum(m["text"].count("<memory>\n") for m in request[1:])


# ---------------------------------------------------------------- placements

def test_trim_shows_the_card_once_after_trims_and_no_deltas():
    d = _placement("trim")
    reqs = d["requests"]
    assert not any("<memory_update>" in m["text"] for r in reqs for m in r)
    carded = [i for i, r in enumerate(reqs) if CARD_RE.match(r[1]["text"])]
    assert len(carded) >= 3
    for i in range(1, len(reqs)):
        assert _card_count(reqs[i]) <= 1
        # E4c: without a trim the head (card included) is byte-identical
        if len(reqs[i]) > len(reqs[i - 1]) and reqs[i][1]["key"] == reqs[i - 1][1]["key"]:
            assert _extends(reqs[i - 1], reqs[i])
    # E4a: the card rides on an existing turn message; nothing is pinned
    for i in carded:
        head = reqs[i][1]
        assert head["role"] == "user" and len(head["text"]) > len(CARD_RE.match(head["text"]).group(0)) + 100


@pytest.mark.parametrize("name", ["trim", "delta", "both"])
def test_every_request_extends_the_last_between_trims(name):
    # E5: no placement edits retained history between trims
    reqs = _placement(name)["requests"]
    pairs = 0
    for prev, cur in zip(reqs, reqs[1:]):
        if len(cur) > len(prev) and cur[1]["key"] == prev[1]["key"]:
            pairs += 1
            assert _extends(prev, cur)
    assert pairs >= 10


def test_delta_has_no_head_card_and_deltas_only_in_openers():
    d = _placement("delta")
    reqs = d["requests"]
    assert not any(_card_count(r) for r in reqs)
    deltas = [o for _, o in _openers(d) if "<memory_update>" in o["text"]]
    assert len(deltas) >= 8
    # deltas live only in turn openers, as they were first sent
    opener_texts = {o["text"] for _, o in _openers(d)}
    for r in reqs:
        for m in r[1:]:
            if "<memory_update>" in m["text"]:
                assert m["text"] in opener_texts


def test_both_head_card_plus_later_deltas_cover_the_full_card():
    d = _placement("both")
    reqs = d["requests"]
    last_card_at, last_card = 0, ""
    for i, r in enumerate(reqs):
        match = CARD_RE.match(r[1]["text"])
        if match and (i == 0 or not reqs[i - 1][1]["text"].startswith(match.group(0))):
            last_card_at, last_card = i, match.group(0)
    seen = last_card + "\n".join(
        "\n".join(DELTA_RE.findall(o["text"])) for i, o in _openers(d) if i >= last_card_at
    ) + "\n".join(d["ledger"]["pending"])
    full = d["ledger"]["full_card"]
    lines = re.findall(r"<(?:solved|hazard|world_model|hypothesis[^>]*)>(.*?)</", full)
    assert lines
    for text in lines:
        assert text in seen, text


def test_lines_sit_after_the_valid_actions_line_in_one_block():
    d = _placement("both")
    found = 0
    for _, opener in _openers(d):
        text = opener["text"]
        marks = [text.find(m) for m in ("Seen before", "<memory_update>") if m in text]
        if not marks:
            continue
        found += 1
        valid = text.index("Valid actions right now:")
        tool = text.index("Only tool: `python`")
        assert all(valid < m < tool for m in marks)
        if len(marks) == 2:
            assert marks[0] < marks[1]  # seen-before first, then the deltas
    assert found >= 10


def test_delta_system_text_describes_only_the_lines_it_sends():
    # no head card: the card's <solved>/<actions>/<hazard> tags never appear,
    # so the system text describes the <memory_update> forms instead
    d = _placement("delta")
    system = d["requests"][0][0]["text"]
    sent = "\n".join(m["text"] for r in d["requests"] for m in r[1:])
    for tag in ("<solved>", "<actions", "<hazard>"):
        assert tag not in system and tag not in sent, tag
    assert "`solved L<N>:` lines" in system and "`<action>: first <effect>` lines" in system
    lines = [line for _, o in _openers(d) for b in DELTA_RE.findall(o["text"]) for line in b.splitlines()]
    assert any(line.startswith("+ solved L1:") for line in lines)
    assert any(re.match(r"\+ \S+: first ", line) for line in lines)
    assert any(re.search(r"L\d+: attempt \d+ ended in GAME OVER", line) for line in lines)
    for name in ("trim", "both"):
        system = _placement(name)["requests"][0][0]["text"]
        assert '`<actions level="N">` lines' in system and "`<solved>` lines" in system


def test_system_message_never_changes():
    for name in ("trim", "delta", "both"):
        d = _placement(name)
        assert len({r[0]["key"] for r in d["requests"]}) == 1


def test_harness_text_states_facts_not_verdicts():
    d = _placement("both")
    for _, opener in _openers(d):
        for line in opener["text"].splitlines():
            if line.startswith(("Seen before", "+ ")) and "refuted:" not in line and "world_model:" not in line:
                assert not BANNED.search(line), line
    card = d["ledger"]["full_card"]
    harness = [l for l in card.splitlines() if "<hypothesis" not in l and "<world_model>" not in l and "<level_note>" not in l]
    assert not any(BANNED.search(l) for l in harness)


def test_hold_lets_only_solved_and_first_death_lines_through():
    d = _dump((("ARC3_MEMORY_PLACEMENT", "both"), ("ARC3_MEM_LINES_AFTER_TOKENS", "<unset>")))
    texts = [o["text"] for _, o in _openers(d)]
    assert not any("Seen before" in t for t in texts)
    lines = [line for t in texts for block in DELTA_RE.findall(t) for line in block.splitlines()]
    assert any(line.startswith("+ solved L1:") for line in lines)
    assert any("attempt 1 ended in GAME OVER" in line for line in lines)
    for line in lines:
        assert line.startswith("+ solved ") or "ended in GAME OVER" in line, line
    # the second DOWN death of level 1 is not a first death: held
    assert not any("L1: attempt 2 ended" in line and "last action DOWN" in line for line in lines)


def _levels_of_openers(d: dict) -> list[tuple[int, str]]:
    """(level, opener text) per turn: a level starts at the opener that
    carries the previous level's solved line (it passes any hold)."""
    out, level = [], 1
    for _, opener in _openers(d):
        text = opener["text"]
        while f"+ solved L{level}:" in text:
            level += 1
        out.append((level, text))
    return out


def _turn_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(("Seen before", "+ "))]


def test_hold_releases_after_the_level_tokens_and_restarts_each_level():
    # 1000 generated tokens per request and a hold of 8000: the first 8 turn
    # messages of each level carry only solved and first-death lines; level 1
    # (13 turns) releases its lines, level 2 (7 turns) never does
    d = _dump((("ARC3_MEMORY_PLACEMENT", "delta"), ("MEMSIM_COMPLETION_TOKENS", "1000"),
               ("ARC3_MEM_LINES_AFTER_TOKENS", "8000")))
    by_level: dict[int, list[str]] = {}
    for level, text in _levels_of_openers(d):
        by_level.setdefault(level, []).append(text)
    assert sorted(by_level) == [1, 2, 3] and len(by_level[2]) < 8 < len(by_level[1])
    for level, texts in by_level.items():
        for text in texts[:8]:
            for line in _turn_lines(text):
                assert line.startswith("+ solved ") or " ended in GAME OVER " in line, (level, line)
    released = [line for text in by_level[1][8:] for line in _turn_lines(text)]
    assert any(line.startswith("Seen before") for line in released)
    assert any(": first " in line for line in released) and any("world_model:" in line for line in released)
    # what level 2 queued while held is not shown on level 3: hazards are
    # always the current level's, and the first-effect line goes too
    for level, texts in by_level.items():
        for text in texts:
            for line in _turn_lines(text):
                match = re.match(r"\+ L(\d+): attempt ", line)
                assert not match or int(match.group(1)) == level, (level, line)
    assert not any("UP: first edge only" in text for texts in by_level.values() for text in texts)
    unheld = [text for _, text in _levels_of_openers(_placement("delta"))]
    assert any("UP: first edge only" in text for text in unheld)  # level 2 does produce it


def test_first_level_opener_carries_the_solved_line():
    d = _placement("delta")
    for _, opener in _openers(d):
        if "progressed to a new level" in opener["text"] or "completed the previous level" in opener["text"]:
            assert "+ solved L" in opener["text"]
            break
    else:
        raise AssertionError("no level transition opener")


# ---------------------------------------------------------------- rollback (E6)

def _rollback_case(turn_pick):
    base = _placement("both")
    reqs = base["requests"]
    starts = []
    index = 0
    for count in base["turn_requests"]:
        starts.append(index)
        index += count
    # turns whose first request shows a new card and whose opener carries a delta
    candidates = [
        t for t, i in enumerate(starts)
        if 0 < i < len(reqs) and reqs[i][1]["key"] != reqs[i - 1][1]["key"]
        and CARD_RE.match(reqs[i][1]["text"]) and "<memory_update>" in reqs[i][-1]["text"]
        and t + 1 < len(starts)
    ]
    turn = turn_pick(candidates, reqs, starts, base["drops"])
    failed = _placement("both", fail=turn)
    freqs = failed["requests"]
    fstarts, index = [], 0
    for count in failed["turn_requests"]:
        fstarts.append(index)
        index += count
    bad, retry = freqs[fstarts[turn]], freqs[fstarts[turn + 1]]
    assert CARD_RE.match(bad[1]["text"])  # the failed request had trimmed and shown the card
    assert _card_count(retry) == 1
    lost = DELTA_RE.findall(bad[-1]["text"])[0].splitlines()
    retried = retry[-1]["text"]
    for line in lost:
        assert line in retried, line


def test_rollback_when_the_failed_turn_trimmed_itself():
    # the trim happened in the failed turn: the retry re-trims and re-shows the card
    def pick(candidates, reqs, starts, drops):
        return next(t for t in candidates if ["turn", starts[t]] in drops and ["persist", starts[t]] not in drops)
    _rollback_case(pick)


def test_rollback_when_the_previous_turn_trimmed_at_persist():
    # the trim came from the previous turn's persist; the rolled-back history is
    # already trimmed, so only the restored card flag brings the card back
    def pick(candidates, reqs, starts, drops):
        return next(t for t in candidates if ["persist", starts[t]] in drops)
    _rollback_case(pick)


# ---------------------------------------------------------------- level change (10)

def test_level_change_resets_the_level_and_carries_the_game():
    d = _placement("both", turns=15)  # level 2 starts at turn 13
    rec, led = d["recorder"], d["ledger"]
    assert rec["level"] == 2
    assert "solved:1" in led["items"]
    assert not any(k.startswith("hazard:1:") for k in led["items"])  # L1 hazards gone
    assert "DOWN" in rec["game_keys"] and "DOWN" not in rec["level_keys"]
    assert rec["seen_boards"] < 10
    assert led["world_model"]  # model beliefs carry
    assert led["pending_think"] == []
    actions = led["items"]["actions"]
    assert "DOWN: not tried this level | game " in actions
    assert '<actions level="2">' in led["full_card"]


# ---------------------------------------------------------------- python code tags (11)

def test_tags_in_python_code_are_absorbed_and_stripped():
    d = _placement("trim")
    assert "walls block movement" in d["ledger"]["world_model"]
    assert all("<world_model>" not in code for code in d["codes"])
    legacy = _dump((("ARC3_MEM_SOLVED", "0"), ("ARC3_MEM_CARD", "0"), ("ARC3_MEM_EFFECTS", "0"), ("ARC3_MEM_SEEN", "0")))
    assert any("<world_model>" in code for code in legacy["codes"])  # ledger-only: untouched


def test_code_absorption_runs_only_code_that_compiles(monkeypatch):
    from inference.agent.memory_ledger import MemoryLedger
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("ARC3_MEM_CARD", "1")
    agent = ToolAgent.__new__(ToolAgent)
    agent._ledger = MemoryLedger(card_mode=True)
    out = agent._mem_absorb_code({"code": "<world_model>a bare tag</world_model>\nprint(1)\n"})
    assert out["code"] == "\nprint(1)\n"  # the SyntaxError case now runs
    assert agent._ledger.world_model == ["a bare tag"]
    code = 'x = "<world_model>in a string</world_model>" + (\n'
    assert agent._mem_absorb_code({"code": code})["code"] == code  # stripped would not compile either
    monkeypatch.setenv("ARC3_MEM_CARD", "0")
    same = {"code": "# <world_model>z</world_model>\n"}
    assert agent._mem_absorb_code(same) is same


_CODE_FLAGS = ("ARC3_MEM_SOLVED", "ARC3_MEM_CARD", "ARC3_MEM_EFFECTS", "ARC3_MEM_SEEN",
               "ARC3_MEM_PATH", "ARC3_MEM_CODE_TAGS")


@pytest.mark.parametrize("env,strips", [
    ({}, False),  # LEDGER_ON_TRIM alone (main's user-combo-10x): the code runs as written
    ({"ARC3_MEM_SOLVED": "1"}, True),  # no card: the mem-solved-10x arm
    ({"ARC3_MEM_EFFECTS": "1"}, True),
    ({"ARC3_MEM_SEEN": "1"}, True),
    ({"ARC3_MEM_PATH": "1"}, True),
    ({"ARC3_MEM_CARD": "1"}, True),
    ({"ARC3_MEM_CODE_TAGS": "1"}, True),  # any ledger, on request
    ({"ARC3_MEM_SOLVED": "1", "ARC3_MEM_CODE_TAGS": "0"}, False),
])
def test_code_tags_follow_any_memory_flag(monkeypatch, env, strips):
    from inference.agent.memory_ledger import MemoryLedger
    from inference.agent.tool_agent import ToolAgent

    for flag in _CODE_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    agent = ToolAgent.__new__(ToolAgent)
    agent._ledger = MemoryLedger(card_mode="ARC3_MEM_CARD" in env)
    arguments = {"code": '<hypothesis status="refuted">UP opens the door</hypothesis>\nprint(1)\n'}
    out = agent._mem_absorb_code(arguments)
    if strips:
        assert out["code"] == "\nprint(1)\n"
        assert agent._ledger.refuted == ["UP opens the door"]
    else:
        assert out is arguments and agent._ledger.refuted == []


def test_code_tags_need_a_ledger(monkeypatch):
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("ARC3_MEM_CODE_TAGS", "1")
    agent = ToolAgent.__new__(ToolAgent)
    agent._ledger = None
    arguments = {"code": "<world_model>x</world_model>\n"}
    assert agent._mem_absorb_code(arguments) is arguments


def test_delta_placement_never_hides_the_plain_ledger(monkeypatch):
    from inference.agent.memory_ledger import MemoryLedger
    from inference.agent.prompts import LEDGER_ON_TRIM_ADDENDUM, memory_addendum
    from inference.agent.tool_agent import ToolAgent

    for flag in ("ARC3_MEM_SOLVED", "ARC3_MEM_CARD", "ARC3_MEM_EFFECTS", "ARC3_MEM_SEEN"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("ARC3_LEDGER_ON_TRIM", "1")
    monkeypatch.setenv("ARC3_MEMORY_PLACEMENT", "delta")
    agent = ToolAgent.__new__(ToolAgent)
    agent._ledger = MemoryLedger()
    agent._note_history_evicted()
    assert agent._ledger_due  # the system prompt promises this card
    monkeypatch.setenv("ARC3_MEM_SOLVED", "1")
    agent._note_history_evicted()
    assert not agent._ledger_due  # memory-v2 with deltas only: no card
    text = memory_addendum(ledger=True, card=False, solved=True, effects=False, seen=False,
                           head_card=False, deltas=True)
    assert LEDGER_ON_TRIM_ADDENDUM not in text and "<memory_update>" in text and "`solved L<N>:` lines" in text


def test_board_identity_border_follows_the_noop_guard(monkeypatch):
    from inference.agent import tool_agent as ta

    monkeypatch.delenv("ARC3_MEM_STATE_BORDER", raising=False)
    monkeypatch.delenv("ARC3_NOOP_GUARD_BORDER", raising=False)
    assert ta._mem_state_border() == 4
    monkeypatch.setenv("ARC3_NOOP_GUARD_BORDER", "2")
    assert ta._mem_state_border() == 2
    monkeypatch.setenv("ARC3_MEM_STATE_BORDER", "0")
    assert ta._mem_state_border() == 0


# ---------------------------------------------------------------- resume (6)

def test_resumed_turn_gets_no_lines_and_keeps_them_pending(monkeypatch):
    from inference.agent.memory_ledger import MemoryLedger
    from inference.agent.tool_agent import ToolAgent

    monkeypatch.setenv("ARC3_MEM_EFFECTS", "1")
    monkeypatch.setenv("ARC3_MEMORY_PLACEMENT", "both")
    monkeypatch.setenv("ARC3_MEM_LINES_AFTER_TOKENS", "0")
    agent = ToolAgent.__new__(ToolAgent)
    agent._ledger = MemoryLedger(delta_mode=True)
    agent._mem_recorder = None
    agent._session_generated_tokens = 0
    agent._tokens_at_level_start = 0
    agent._ledger.add_event("fx:UP:shift", "UP: first moved")
    agent._resume_after_yield = True
    assert agent._mem_turn_lines(["UP"]) == []
    assert agent._ledger.pending()
    agent._resume_after_yield = False
    assert agent._mem_turn_lines(["UP"]) == ["<memory_update>\n+ UP: first moved\n</memory_update>"]
