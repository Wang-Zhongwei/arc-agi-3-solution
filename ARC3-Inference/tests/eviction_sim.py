"""Does anything but a real history drop give the game's GPU slot away? (#58)

Under the Kaggle scheduler a game keeps its slot until ``_note_history_evicted``
sets ``_context_was_trimmed``; the next request's ``_maybe_handover`` then
releases the slot and re-queues the game in the bottom band. That is free only
when the prefix really broke because already-sent messages were dropped. Memory
text that edits, strips, merges or re-inserts history can reach the same flag
(Fable's journal did) or break the prefix without it (an unpriced re-prefill).

This drives the real ``ToolAgent.analyze()`` through ``memory_sim``'s scripted
game, adding the request shapes the plain script never makes: a python call that
does not act (two requests in one turn), a reply over the per-turn token budget
(a yield and a resume opener), a context-length rejection (the force-drain
retry), and a request failure after a successful one (rollback). It records

  handovers  index of the request each ``gate.handover()`` preceded (fake gate;
             ``_maybe_handover`` itself is the real code)
  drops      index of the next request whenever the trimmer, the force-reduce
             fallback or the turn-end persist returned FEWER messages than it
             was given - a real drop, measured at the method boundary so the
             floor-mode look-ahead probe does not count

and checks, comparing each request with its reference - the last accepted
request of the same turn, else the last accepted request of the last turn that
was kept (a rolled-back turn's messages are no longer history):

  1. every handover follows a real drop
  2. a request that does not extend its reference has a handover (no unpriced
     prefix break)
  3. no two consecutive accepted requests are both full re-prefills (diverging
     at message 0 or 1), except right after a rejection or a rollback, where
     the harness re-trims by design
  4. no request carries two ``<memory>`` blocks or two copies of a pinned message

The ``neg-*`` configs install known traps (from the #58 measurements) and must
each fail the assertion named in ``expect_fail``; they prove the checks can see
the trap.

Usage: python tests/eviction_sim.py <config>   (prints one JSON line)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memory_sim  # noqa: E402

# memory_sim.KAGGLE_ENV is frozen by the identity golden; these are the Kaggle
# knobs (kaggle/template.ipynb cell 4) that change scheduling or turn shape.
EVICTION_ENV = {
    **memory_sim.KAGGLE_ENV,
    "ARC3_PRIORITY_REFRESH_QUEUE": "1",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "ARC3_ACTION_ECHO": "1",
    "ARC3_MIDDLE_TRUNCATION": "1",
}
LEDGER = {"ARC3_LEDGER_ON_TRIM": "1"}
MEMORY_V2_ON = {
    **LEDGER,
    "ARC3_MEM_SOLVED": "1", "ARC3_MEM_CARD": "1", "ARC3_MEM_EFFECTS": "1", "ARC3_MEM_SEEN": "1",
    # the scripted replies report no token usage, so the default 15k-token
    # hold would keep every per-turn line back; show them all instead
    "ARC3_MEM_LINES_AFTER_TOKENS": "0",
}

def _arm_env(arm: str) -> dict[str, str]:
    """An arm's env from kaggle/arms.json without its window, drain, stream and
    sampling settings (EVICTION_ENV's small window is what makes trims here)."""
    arms_file = HERE.parents[1] / "kaggle" / "arms.json"
    if not arms_file.exists():
        return {}
    env = json.loads(arms_file.read_text())["arms"].get(arm, {}).get("env", {})
    skip = ("LOCAL_ANALYZER_", "ARC3_CONTEXT_DRAIN_TOKENS", "ARC3_MAX_ACTIVE_STREAMS")
    return {k: v for k, v in env.items() if not k.startswith(skip)}


# The combo configs' window: on this stack the system prompt and first opener
# take ~8.1k tokens (level-path's), so the plain configs' 14k window leaves
# ~4k for history, and one turn after a trim the next opener (head card,
# re-sent static tail, route line) overflowed it again: two full re-prefills
# in a row (check 3) at request 6 of combo-level-path-10x. In the arms' 96k
# window that would need one turn to add more than the 42k drain. 18k, with
# the plain configs' drain share (4,000 of 14,000), still trims five times.
COMBO_DRAIN = {"LOCAL_ANALYZER_CONTEXT_WINDOW": "18000",
               "ARC3_CONTEXT_DRAIN_TOKENS": str(round(18000 * 4000 / 14000))}

CONFIGS: dict[str, dict[str, Any]] = {
    "off": dict(env={}),
    "ledger": dict(env=LEDGER),
    "ledger-death": dict(env={**LEDGER, "ARC3_DEATH_LEDGER": "1"}),
    # floor mode's look-ahead drops a block from a copy and may throw it away
    "off-floor": dict(env={"ARC3_CONTEXT_DRAIN_FLOOR": "1"}),
    # memory-v2 placements (skipped on a branch without ARC3_MEMORY_PLACEMENT)
    "mem-trim": dict(env={**MEMORY_V2_ON, "ARC3_MEMORY_PLACEMENT": "trim"}, needs="ARC3_MEMORY_PLACEMENT"),
    "mem-delta": dict(env={**MEMORY_V2_ON, "ARC3_MEMORY_PLACEMENT": "delta"}, needs="ARC3_MEMORY_PLACEMENT"),
    "mem-both": dict(env={**MEMORY_V2_ON, "ARC3_MEMORY_PLACEMENT": "both"}, needs="ARC3_MEMORY_PLACEMENT"),
    # ARC3_MEM_PATH: the route suffix in each tool result and the route line in
    # each turn message, on memory-v2 (both placements' lines) and alone
    "mem-path": dict(env={**MEMORY_V2_ON, "ARC3_MEMORY_PLACEMENT": "both", "ARC3_MEM_PATH": "1"},
                     needs="ARC3_MEM_PATH"),
    "path-only": dict(env={"ARC3_MEM_PATH": "1"}, needs="ARC3_MEM_PATH"),
    # main's user-combo-10x harness flags (opener dedup, animation layers,
    # EXPOSE_RESET, ledger on trim) alone and with each memory-main arm's
    # flags, as kaggle/arms.json sets them, in an 18k window (see COMBO_WINDOW)
    "combo-10x": dict(env={**_arm_env("user-combo-10x"), **COMBO_DRAIN}, needs="ARC3_OPENER_DEDUP"),
    **{
        f"combo-{arm}": dict(env={**_arm_env(arm), **COMBO_DRAIN, "ARC3_MEM_LINES_AFTER_TOKENS": "0"},
                             needs="ARC3_MEM_PATH")
        for arm in ("mem-solved-10x", "mem-v2-10x", "level-path-10x")
    },
    # ARC3_MEM_EVERY_CALL: <previous_levels> after the system prompt, redrawn
    # at a level commit (with ARC3_LEVEL_COMPACT, together with the drop of
    # the cleared level's turns) and at trims; a redraw is a real prefix
    # break, so it counts as one here (see the drop recorders)
    "combo-mem-layout-10x": dict(env={**_arm_env("mem-layout-10x"), **COMBO_DRAIN},
                                 needs="ARC3_MEM_EVERY_CALL"),
    "combo-mem-layout-nocompact": dict(
        env={**{k: v for k, v in _arm_env("mem-layout-10x").items() if k != "ARC3_LEVEL_COMPACT"}, **COMBO_DRAIN},
        needs="ARC3_MEM_EVERY_CALL"),
    # negative controls
    "neg-strip-counted": dict(env=LEDGER, trap="strip_counted", expect_fail={1}),
    "neg-strip-uncounted": dict(env=LEDGER, trap="strip_uncounted", expect_fail={2}),
    "neg-pin-ledger": dict(env=LEDGER, trap="pin", expect_fail={1, 4}),
}

# script turn -> what the model does before its scripted action reply
EVENTS: dict[int, list[str]] = {
    3: ["probe"],
    7: ["yield"],
    11: ["probe", "probe"],
    20: ["probe", "fail"],
    25: ["yield"],
}
# One context-length rejection, on the first request from this turn on that was
# not itself just trimmed (a fresh trim leaves the force-drain nothing to drop,
# the retry raises, and the path under test never runs).
OVERFLOW_FROM_TURN = 15

DELTA_TAG = "<memory_update>"
PIN_MARKER = "LEVEL LOG"
PIN_TEXT = f"{PIN_MARKER} (harness-computed facts for this level)\n" + "UP: moved 3 of 4 | RIGHT: noop 2\n" * 8
PIN_REPLY = "Noted."


ROUTE_TURN_HEAD = "Shortest route on record from this level's start board"


def _key(message: dict[str, Any]) -> str:
    return json.dumps(message, sort_keys=True)


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content or "")


def _strip_delta(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        kept = [p for p in content if not (isinstance(p, dict) and str(p.get("text", "")).startswith(DELTA_TAG))]
        if len(kept) != len(content):
            return {**message, "content": kept}
    elif isinstance(content, str) and DELTA_TAG in content:
        return {**message, "content": content.split("\n\n" + DELTA_TAG)[0]}
    return message


def _first_difference(prev: list[str], cur: list[str]) -> int:
    index = 0
    while index < min(len(prev), len(cur)) and prev[index] == cur[index]:
        index += 1
    return index


def run_config(name: str) -> dict[str, Any]:
    import requests

    from inference.agent import tool_agent as ta

    cfg = CONFIGS[name]
    trap = cfg.get("trap")
    rec: dict[str, Any] = {
        "requests": [], "handovers": [], "drops": set(), "rejected": set(),
        "rolled_back_turns": set(), "turn_start": {}, "calls": {}, "turn": 0,
        "resumes": 0, "pinned": None, "delta_n": 0, "overflowed": False,
    }

    class FakeGate:
        def handover(self, priority, snapshot=None):
            rec["handovers"].append(len(rec["requests"]))

        def _snapshot_priority(self, snapshot, now):
            return 0

    gate = FakeGate()
    ta._priority_gate = lambda: gate

    def probe_reply(turn: int, usage: dict[str, Any]):
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"Let me look at the board before acting (turn {turn}). " + memory_sim.FILLER,
            "tool_calls": [{
                "id": f"probe_{turn}_{rec['calls'][turn]}",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": "print(current_frame.level)\n"})},
            }],
        }
        return ta._ChatCompletionResult(message=message, finish_reason="tool_calls", usage=usage)

    def setup(agent, requests_seen):
        rec["requests"] = requests_seen
        drops = rec["drops"]

        # Drop recorders go on first, so they sit next to the real methods and
        # a trap wrapper installed below cannot hide or fake a drop.
        real_trim = agent._trim_messages_for_context
        real_force = agent._force_reduce_messages
        real_persist = agent._persistent_history_messages

        def trim(messages, **kwargs):
            out = real_trim(messages, **kwargs)
            if len(out) < len(messages):
                drops.add(len(requests_seen))
            return out

        def force(messages, **kwargs):
            out = real_force(messages, **kwargs)
            if len(out) < len(messages):
                drops.add(len(requests_seen))
            return out

        def persist(messages, *, tools=None):
            out = real_persist(messages, tools=tools)
            if len(out) < len(messages) - 1:
                drops.add(len(requests_seen))
            return out

        agent._trim_messages_for_context = trim
        agent._force_reduce_messages = force
        agent._persistent_history_messages = persist
        real_commit = getattr(agent, "_mem_layout_commit", None)
        if real_commit is not None:
            # ARC3_MEM_EVERY_CALL: a <previous_levels> redraw changes message 1
            # of every later request, a prefix break as real as a drop
            def commit(*args, **kwargs):
                out = real_commit(*args, **kwargs)
                if out:
                    drops.add(len(requests_seen))
                return out

            agent._mem_layout_commit = commit

        scripted_chat = agent._chat_completion

        def chat(messages, **kwargs):
            turn = rec["turn"]
            index = len(requests_seen)
            if not rec["overflowed"] and turn >= OVERFLOW_FROM_TURN and index not in rec["handovers"]:
                rec["overflowed"] = True
                rec["rejected"].add(index)
                requests_seen.append(json.loads(json.dumps(ta._strip_control_keys(messages))))
                raise requests.HTTPError("This model's maximum context length is 14000 tokens.")
            plan = EVENTS.get(turn, [])
            call = rec["calls"].get(turn, 0)
            rec["calls"][turn] = call + 1
            event = plan[call] if call < len(plan) else None
            if event is None:
                return scripted_chat(messages, **kwargs)
            requests_seen.append(json.loads(json.dumps(ta._strip_control_keys(messages))))
            if event == "fail":
                rec["rejected"].add(index)
                raise requests.ConnectionError("simulated outage")
            usage = {"completion_tokens": 4096} if event == "yield" else {"completion_tokens": 64}
            return probe_reply(turn, usage)

        agent._chat_completion = chat
        real_analyze = agent.analyze

        def analyze(*args, **kwargs):
            # a failed turn's history is restored to what it was before it
            result = real_analyze(*args, **kwargs)
            if result is None or getattr(result, "retryable_failure", False):
                rec["rolled_back_turns"].add(rec["turn"])
            return result

        agent.analyze = analyze

        if trap in ("strip_counted", "strip_uncounted"):
            real_build = agent._build_user_message

            def build(user_prompt, current_frame):
                message = real_build(user_prompt, current_frame)
                rec["delta_n"] += 1
                delta = f"{DELTA_TAG}\n+ item {rec['delta_n']}: RIGHT first noop\n</memory_update>"
                content = message.get("content")
                if isinstance(content, list):
                    message["content"] = [*content, {"type": "text", "text": delta}]
                else:
                    message["content"] = f"{content}\n\n{delta}"
                return message

            agent._build_user_message = build
            recorded_persist = agent._persistent_history_messages

            if trap == "strip_uncounted":
                def strip_persist(messages, *, tools=None):
                    # older openers lose their delta before the persist compare,
                    # so the stored history silently differs from what was sent
                    return recorded_persist([messages[0], *[_strip_delta(m) for m in messages[1:]]], tools=tools)
            else:
                def strip_persist(messages, *, tools=None):
                    history = recorded_persist(messages, tools=tools)
                    stripped = [_strip_delta(m) for m in history]
                    if stripped != history:
                        agent._note_history_evicted()
                    return stripped

            agent._persistent_history_messages = strip_persist

        if trap == "pin":
            recorded_trim = agent._trim_messages_for_context

            def restore_pin(messages, **kwargs):
                # Fable's _fl_restore_pinned: re-insert the pinned pair after the
                # trimmer's own check, deduped on exact content
                out = recorded_trim(messages, **kwargs)
                pinned = rec["pinned"]
                if not pinned or any(
                    m.get("role") == "user" and m.get("content") == pinned[0]["content"] for m in out[1:]
                ):
                    return out
                return [out[0], *[dict(m) for m in pinned], *out[1:]]

            agent._trim_messages_for_context = restore_pin

    def before_turn(agent, turn):
        rec["turn"] = turn
        rec["turn_start"][turn] = len(rec["requests"])
        if getattr(agent, "_resume_after_yield", False):
            rec["resumes"] += 1
        if trap == "pin" and turn == 4:
            rec["pinned"] = [{"role": "user", "content": PIN_TEXT}, {"role": "assistant", "content": PIN_REPLY}]
            agent._history_messages = [*agent._history_messages, *[dict(m) for m in rec["pinned"]]]

    result = memory_sim.run(setup=setup, before_turn=before_turn)
    return analyze_requests(result.requests, rec)


def analyze_requests(requests_seen: list[list[dict[str, Any]]], rec: dict[str, Any]) -> dict[str, Any]:
    keys = [[_key(m) for m in request] for request in requests_seen]
    starts = sorted(rec["turn_start"].items())
    turn_of = [max(turn for turn, start in starts if start <= index) for index in range(len(keys))]
    first_of_turn = {index for index in range(len(keys)) if index == 0 or turn_of[index] != turn_of[index - 1]}
    rolled_back = {i for i, turn in enumerate(turn_of) if turn in rec["rolled_back_turns"]}
    rejected = set(rec["rejected"])
    handovers = set(rec["handovers"])
    drops = set(rec["drops"])

    # The reference is what the server has cached and history still holds: the
    # last accepted request of this turn, else the last accepted request of a
    # kept turn (a rolled-back turn's messages are gone from history).
    divergence: dict[int, int] = {}
    breaks: list[int] = []
    recovering: set[int] = set()  # follows a rejection or a rollback: re-trims by design
    committed: int | None = None
    in_turn: int | None = None
    for index in range(len(keys)):
        if index in first_of_turn and index > 0:
            if turn_of[index - 1] in rec["rolled_back_turns"]:
                recovering.add(index)
            elif in_turn is not None:
                committed = in_turn
            in_turn = None
        if index - 1 in rejected:
            recovering.add(index)
        reference = in_turn if in_turn is not None else committed
        if reference is not None:
            d = _first_difference(keys[reference], keys[index])
            divergence[index] = d
            if d < len(keys[reference]):
                breaks.append(index)
        if index not in rejected:
            in_turn = index

    full = {i for i, d in divergence.items() if d <= 1 and i not in rejected}
    cascades = [i for i in sorted(full) if i - 1 in full and i not in recovering]
    # system prompt excluded: with the ledger on it names the block in prose
    memory_blocks = [sum(_text_of(m).count("<memory>\n") for m in request[1:]) for request in requests_seen]
    pin_copies = [
        sum(1 for m in request if m.get("role") == "user" and PIN_MARKER in _text_of(m))
        for request in requests_seen
    ]
    # ARC3_MEM_PATH's text, counted once per distinct message: the route lines
    # of turn messages and the route echoes of tool results
    route_turn, route_echo = set(), set()
    for request in requests_seen:
        for message in request[1:]:
            text = _text_of(message)
            if message.get("role") == "user" and ROUTE_TURN_HEAD in text:
                route_turn.add(text)
            elif message.get("role") == "tool" and (" | route " in text or "[route] route " in text):
                route_echo.add(text)
    violations = {
        1: sorted(handovers - drops),
        2: sorted(i for i in breaks if i not in handovers),
        3: cascades,
        4: sorted(i for i in range(len(keys)) if memory_blocks[i] > 1 or pin_copies[i] > 1),
    }
    return {
        "requests": len(keys),
        "handovers": len(rec["handovers"]),
        "handover_at": sorted(handovers),
        "drops_at": sorted(drops),
        "prefix_breaks": len(breaks),
        "full_reprefills": len(full - rejected),
        "later_in_turn_requests": sum(1 for i in range(len(keys)) if i not in first_of_turn),
        "later_in_turn_handovers": sorted(i for i in handovers if i not in first_of_turn),
        "rejected": sorted(rejected),
        "rolled_back_requests": sorted(rolled_back),
        "resumes": rec["resumes"],
        "max_memory_blocks": max(memory_blocks, default=0),
        "max_pin_copies": max(pin_copies, default=0),
        "route_turn_lines": sum(text.count(ROUTE_TURN_HEAD) for text in route_turn),
        # tool results hold the sandbox output as escaped JSON: count occurrences
        "route_echo_lines": sum(text.count(" | route ") + text.count("[route] route ") for text in route_echo),
        "violations": {str(k): v for k, v in violations.items()},
    }


def config_env(name: str) -> dict[str, str]:
    """A clean environment for one config: no inherited harness knobs."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_", "MEMSIM_"))
    }
    env.update(EVICTION_ENV)
    env.update(CONFIGS[name]["env"])
    return env


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    print(json.dumps({"config": sys.argv[1], **run_config(sys.argv[1])}))
