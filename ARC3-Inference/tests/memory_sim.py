"""A scripted game and model for end-to-end memory tests of ToolAgent.analyze().

Nothing here talks to a server or runs the python sandbox: the chat completion
is replaced by a script of replies, and the python tool by a parser that finds
the reply's ``action([...])`` list and plays it on a small deterministic game.
Everything else - the opener, history, trimming, the ledger - is the real code.

Run as a script to print one scenario's fingerprints as JSON (the flags-off
identity test runs it in a subprocess, so import-time env knobs take effect):

    python tests/memory_sim.py <scenario>
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402

SIZE = 20
BORDER = 4
BG, WALL, PLAYER, GOAL, HAZARD, BAR, KEY = 5, 3, 9, 11, 8, 12, 14
BAR_CELLS = 16
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

LEVELS = [
    dict(start=(8, 8), goal=(5, 12), hazards={(11, 9)}, walls={(7, 10), (8, 10), (9, 10)}, key=(6, 8)),
    dict(start=(10, 10), goal=(5, 5), hazards={(10, 13)}, walls={(8, 6), (8, 7), (8, 8)}, key=None),
    dict(start=(6, 6), goal=(14, 14), hazards={(10, 10)}, walls=set(), key=None),
]


@dataclass
class FakeGame:
    """A block on a board: arrows move it, red kills, yellow clears the level.

    The top border row is a budget bar that loses a cell per action until it is
    empty, so early wall bumps are HUD-only changes and later ones true no-ops.
    """

    level: int = 1
    pos: tuple[int, int] = (0, 0)
    key_taken: bool = False
    ticks: int = 0
    action_count: int = 0
    won: bool = False
    history: list[HistoryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._restart()
        self.history.append(HistoryEntry(action="", frame=self.frame()))

    @property
    def spec(self) -> dict[str, Any]:
        return LEVELS[self.level - 1]

    def _restart(self) -> None:
        self.pos = self.spec["start"]
        self.key_taken = False
        self.ticks = 0

    def grid(self) -> tuple[tuple[int, ...], ...]:
        rows = [[BG] * SIZE for _ in range(SIZE)]
        for col in range(2, 2 + max(0, BAR_CELLS - self.ticks)):
            rows[1][col] = BAR
        for r in range(BORDER, SIZE - BORDER):
            rows[r][BORDER - 1] = WALL
            rows[r][SIZE - BORDER] = WALL
        for r, c in self.spec["walls"]:
            rows[r][c] = WALL
        for r, c in self.spec["hazards"]:
            rows[r][c] = HAZARD
        gr, gc = self.spec["goal"]
        rows[gr][gc] = GOAL
        if self.spec["key"] and not self.key_taken:
            kr, kc = self.spec["key"]
            rows[kr][kc] = KEY
        pr, pc = self.pos
        rows[pr][pc] = PLAYER
        return tuple(tuple(row) for row in rows)

    def frame(self) -> Frame:
        return Frame(grid=self.grid(), step=self.action_count, level=self.level)

    def valid_actions(self) -> list[str]:
        return ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]

    def _interior(self, grid):
        return [row[BORDER:SIZE - BORDER] for row in grid[BORDER:SIZE - BORDER]]

    def _apply(self, name: str) -> dict[str, Any]:
        before = self.grid()
        self.action_count += 1
        self.ticks += 1
        game_over = level_completed = run_complete = False
        if name in MOVES:
            dr, dc = MOVES[name]
            r, c = self.pos[0] + dr, self.pos[1] + dc
            inside = BORDER <= r < SIZE - BORDER and BORDER <= c < SIZE - BORDER
            if inside and (r, c) not in self.spec["walls"]:
                self.pos = (r, c)
                if self.spec["key"] and (r, c) == self.spec["key"]:
                    self.key_taken = True
                if (r, c) in self.spec["hazards"]:
                    game_over = True
                elif (r, c) == self.spec["goal"]:
                    if self.level == len(LEVELS):
                        run_complete = True
                        self.won = True
                    else:
                        level_completed = True
        after = self.grid()
        if level_completed:
            self.level += 1
            self._restart()
            after = self.grid()
        result = {
            "executed": True,
            "action_num": self.action_count,
            "level": self.level,
            "score": self.level - 1,
            "state": "WIN" if run_complete else ("GAME_OVER" if game_over else "NOT_FINISHED"),
            "valid_actions": self.valid_actions(),
            "board_changed": before != after,
            "gameplay_changed": self._interior(before) != self._interior(after),
            "no_op": self._interior(before) == self._interior(after),
            "done": run_complete,
            "level_completed": level_completed,
            "game_over": game_over,
            "run_complete": run_complete,
            "action_name": name,
            "action_data": {},
            "action_display": name,
            "automatic": False,
        }
        self.history.append(HistoryEntry(action=name, frame=self.frame(), result=dict(result)))
        return result

    def _auto_reset(self) -> None:
        self.action_count += 1
        self._restart()
        frame = self.frame()
        self.history.append(HistoryEntry(
            action="RESET", frame=frame,
            result={"executed": True, "action_name": "RESET", "action_display": "RESET",
                    "level": self.level, "automatic": True, "board_changed": True,
                    "gameplay_changed": True, "game_over": False, "level_completed": False},
        ))

    def step(self, actions: list[str]) -> list[dict[str, Any]]:
        executed: list[dict[str, Any]] = []
        for index, name in enumerate(actions, 1):
            if self.won:
                break
            payload = self._apply(str(name).upper())
            payload["batch_index"] = index
            payload["batch_size"] = len(actions)
            executed.append(payload)
            if payload["game_over"]:
                self._auto_reset()
                break
            if payload["level_completed"] or payload["run_complete"]:
                break
        return executed


def batch_payload(executed: list[dict[str, Any]], requested: list[str]) -> dict[str, Any]:
    final = dict(executed[-1])
    final["requested_count"] = len(requested)
    final["executed_count"] = len(executed)
    final["requested_actions"] = list(requested)
    final["executed_actions"] = [item["action_display"] for item in executed]
    final["board_changed"] = any(item["board_changed"] for item in executed)
    final["gameplay_changed"] = any(item["gameplay_changed"] for item in executed)
    final["stopped_early"] = len(executed) < len(requested)
    for item in executed:
        if item.get("game_over"):
            final["fatal_action"] = item["action_display"]
    return final


FILLER = (
    "Looking at the board again, the layout has not changed in any way that matters "
    "and the bar at the top keeps shrinking by one cell per action, which is fine. "
) * 6

# (actions, reasoning, content). Tags are spread out so the ledger fills slowly.
SCRIPT: list[tuple[list[str], str, str]] = [
    (["UP"], "Let me probe UP first. " + FILLER, ""),
    (["DOWN"], "<world_model>The blue block is the player and moves one cell per arrow</world_model> "
               "So DOWN should undo the UP. " + FILLER, ""),
    (["RIGHT", "RIGHT"], "Try RIGHT twice toward the yellow cell. " + FILLER, ""),
    (["DOWN", "DOWN", "DOWN", "DOWN"], '<hypothesis status="confirmed">UP moves the blue block up one cell</hypothesis> '
                                       "Going down to look under the wall. " + FILLER, ""),
    (["SPACE"], "Hmm, the red cell ended the attempt. Wait, let me test SPACE. " + FILLER, ""),
    (["RIGHT"], "So the wall at column 10 blocks RIGHT from (8,9). " + FILLER,
     '<hypothesis status="refuted">The red cell is safe to step on</hypothesis>'),
    (["DOWN", "DOWN"], "Down again to check the red cell. " + FILLER, ""),
    (["DOWN"], "One more DOWN. " + FILLER, ""),
    (["UP", "UP", "UP"], "Then the route must go over the top: UP three times reaches row 5. " + FILLER, ""),
    (["SPACE"], "SPACE again to see whether it does anything here. " + FILLER, ""),
    (["RIGHT", "RIGHT"], "Now RIGHT along row 5. " + FILLER, ""),
    (["LEFT"], "Let me check LEFT returns the block. " + FILLER, ""),
    (["RIGHT", "RIGHT", "RIGHT"], "So moving RIGHT 3 times from (5,9) reaches the yellow goal at (5,12). " + FILLER,
     "<level_note>the goal is the yellow cell</level_note>"),
    (["UP"], "New level. Ok, the goal is the yellow cell again. " + FILLER, ""),
    (["LEFT", "LEFT", "LEFT"], "<world_model>Correction: the player is the blue block and grey cells are walls</world_model> "
                               + FILLER, ""),
    (["UP"], "Try UP into the wall row. " + FILLER, ""),
    (["RIGHT", "RIGHT", "RIGHT", "RIGHT", "DOWN", "RIGHT", "RIGHT"], "Go around on the right. " + FILLER, ""),
    (["SPACE"], "That was the red cell at (10,13). " + FILLER, ""),
    (["LEFT", "LEFT", "LEFT", "LEFT", "LEFT"], "So the left side is clear: LEFT five times reaches column 5. " + FILLER, ""),
    (["UP", "UP", "UP", "UP", "UP"], "Then UP five times reaches the goal at (5,5). " + FILLER, ""),
    (["DOWN", "DOWN", "DOWN", "DOWN"], "Level 3. Down first. " + FILLER, ""),
    (["UP"], "Back up. " + FILLER, ""),
    (["DOWN"], "And down again. " + FILLER, ""),
    (["RIGHT", "RIGHT", "RIGHT", "RIGHT"], "Right into the middle. " + FILLER, ""),
    (["UP"], "Probe UP after the death. " + FILLER, ""),
    (["DOWN"], "Probe DOWN. " + FILLER, ""),
    (["UP"], "Probe UP again. " + FILLER, ""),
    (["RIGHT", "RIGHT"], "Two to the right. " + FILLER, ""),
    (["LEFT", "LEFT"], "Two back. " + FILLER, ""),
    (["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"], "Run along row 6. " + FILLER, ""),
    (["DOWN"] * 9, "So DOWN nine times from (5,14) reaches the goal at (14,14). " + FILLER, ""),
]


def _code_for(actions: list[str], turn: int) -> str:
    note = "# <world_model>walls block movement</world_model>\n" if turn == 6 else ""
    return f"{note}res = action({json.dumps(actions)})\nprint(res)\n"


def _actions_from_code(code: str) -> list[str]:
    match = re.search(r"action\((\[.*?\])\)", code, re.S)
    return [str(a) for a in ast.literal_eval(match.group(1))] if match else []


@dataclass
class SimResult:
    requests: list[list[dict[str, Any]]]
    openers: list[Any]
    agent: Any
    game: FakeGame
    codes: list[str]
    turn_requests: list[int]
    tools: list[Any] = field(default_factory=list)


def _path_on() -> bool:
    # ARC3_MEM_PATH, read the way the agent reads it; checked here by name so
    # this file also runs against a branch without the flag
    return os.environ.get("ARC3_MEM_PATH", "").strip().lower() in {"1", "true", "yes", "on"}


def run(
    *,
    script: list[tuple[list[str], str, str]] | None = None,
    fail_turns: set[int] = frozenset(),
    budget_tokens: int | None = None,
    workdir: Path | None = None,
    setup=None,
    before_turn=None,
) -> SimResult:
    """Play the script through ToolAgent.analyze() and capture every request.

    Args:
        script: turns as (actions, reasoning, content); defaults to SCRIPT.
        fail_turns: 0-based turns whose request raises a RequestException, so
            the turn is abandoned and history rolled back.
        budget_tokens: overrides the agent's context budget (forces trims).
        workdir: where the runtime state and transcripts are written.
        setup: called as setup(agent, requests_seen) once the fakes are in
            place and before the first turn, to wrap agent methods.
        before_turn: called as before_turn(agent, turn) before each analyze().
    """
    import requests

    from inference.agent import tool_agent as ta

    script = list(SCRIPT if script is None else script)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="memsim-"))
    state_path = workdir / "artifacts" / "sim_p0_tool_runtime_state.json"
    game = FakeGame()
    write_runtime_state(state_path, current_frame=game.frame(), history=game.history)

    agent = ta.ToolAgent(model="sim-model", base_url="http://127.0.0.1:9/v1", provider="vllm")
    if budget_tokens is not None:
        agent._context_budget_tokens = budget_tokens
    requests_seen: list[list[dict[str, Any]]] = []
    tools_seen: list[Any] = []
    codes: list[str] = []
    turn_requests: list[int] = []
    state = {"turn": 0}

    def fake_chat(messages, **kwargs):
        requests_seen.append(json.loads(json.dumps(ta._strip_control_keys(messages))))
        tools_seen.append(json.loads(json.dumps(kwargs.get("tools"))))
        turn = state["turn"]
        if turn in fail_turns:
            raise requests.ConnectionError("simulated outage")
        actions, reasoning, content = script[turn]
        message = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": [{
                "id": f"call_{turn}",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": _code_for(actions, turn)})},
            }],
        }
        # MEMSIM_COMPLETION_TOKENS: generated tokens reported per request, so
        # per-level token counts (the memory hold) advance; unset reports none
        tokens = int(os.environ.get("MEMSIM_COMPLETION_TOKENS", "0") or 0)
        usage = {"completion_tokens": tokens} if tokens else {}
        return ta._ChatCompletionResult(message=message, finish_reason="tool_calls", usage=usage)

    def fake_python(path, arguments):
        code = str(arguments.get("code", ""))
        codes.append(code)
        actions = _actions_from_code(code)
        route = _path_on()
        if route:
            # what the real tool's state does with ARC3_MEM_PATH: feed the
            # recorder at snippet start and after the call, then the echo
            agent._route_feed(game.history, state_path)
            rows_before = agent._route_rows_total()
        executed = game.step(actions)
        write_runtime_state(state_path, current_frame=game.frame(), history=game.history)
        if not executed:
            return ta._ToolDispatchResult(json.dumps({"tool": "python", "error": "no action"}))
        final = batch_payload(executed, actions)
        agent._last_step_summary = agent._summarize_step_sequence([final])
        agent._update_summarized_knowledge_from_step_summary()
        payload = {"tool": "python", "returncode": 0, "stdout": json.dumps(
            {k: final[k] for k in ("executed_count", "executed_actions", "level", "game_over", "level_completed")}
        )}
        if route:
            agent._route_feed(game.history, state_path)
            echo: dict[str, Any] = {}
            agent._route_echo(echo, rows_before)
            if echo.get("action_echo"):
                payload["stdout"] = echo["action_echo"] + "\n" + payload["stdout"]
        return ta._ToolDispatchResult(json.dumps(payload, indent=2), step_executed=True, made_progress=True)

    agent._chat_completion = fake_chat
    agent._run_python_tool = fake_python
    if setup is not None:
        setup(agent, requests_seen)
    openers: list[Any] = []
    for turn in range(len(script)):
        if game.won:
            break
        state["turn"] = turn
        before = len(requests_seen)
        if before_turn is not None:
            before_turn(agent, turn)
        agent.analyze(
            state_path,
            game.action_count,
            valid_actions=game.valid_actions(),
            transcript_path=workdir / "transcript.txt",
            analysis_step=turn + 1,
        )
        turn_requests.append(len(requests_seen) - before)
        if len(requests_seen) > before:
            openers.append(requests_seen[before][-1])
    return SimResult(requests_seen, openers, agent, game, codes, turn_requests, tools_seen)


def _sha(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


TRANSCRIPT_ROOT = Path(os.environ.get(
    "ARC3_MEM_REPLAY_RUNS",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs",
))
REPLAY_GAMES = ("ls20-9607627b", "ft09-0d8bbf25", "vc33-5430563c")
_SECTION_RE = re.compile(r"^\[(THINKING|ASSISTANT|USER PROMPT)\]\n(.*?)(?=^\[[A-Z][A-Z :_]+\]\n|\Z)", re.M | re.S)


def replay_ledger_blocks(run_name: str = "fz-ledger-on-trim-r2") -> dict[str, list[str]] | None:
    """Feed a real run's transcripts through MemoryLedger, hashing render() per reply.

    Returns None when the run is not on this machine.
    """
    from inference.agent.memory_ledger import MemoryLedger

    out: dict[str, list[str]] = {}
    for game in REPLAY_GAMES:
        path = TRANSCRIPT_ROOT / run_name / "transcripts" / f"{game}_p0.txt"
        if not path.exists():
            return None
        ledger = MemoryLedger()
        level = 1
        hashes: list[str] = []
        pending_reasoning = ""
        for kind, body in _SECTION_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
            if kind == "USER PROMPT":
                if "progressed to a new level" in body or "completed the previous level" in body:
                    level += 1
                    ledger.on_level_change(level)
            elif kind == "THINKING":
                pending_reasoning = body
                ledger.absorb(body)
                ledger.note_think(body)
                hashes.append(hashlib.sha1(ledger.render().encode("utf-8")).hexdigest()[:16])
            else:
                ledger.absorb(body)
                hashes.append(hashlib.sha1(ledger.render().encode("utf-8")).hexdigest()[:16])
        del pending_reasoning
        out[game] = hashes
    return out


def fingerprints() -> dict[str, Any]:
    """What the model is sent in one scenario, hashed: system prompt and requests."""
    from inference.agent import tool_agent as ta

    budget = int(os.environ.get("MEMSIM_BUDGET_TOKENS", "0") or 0) or None
    result = run(budget_tokens=budget)
    return {
        "system_prompt": hashlib.sha1(
            ta._build_system_prompt(tool_output_tokens=1024).encode("utf-8")
        ).hexdigest(),
        "requests": [_sha(messages) for messages in result.requests],
        "request_count": len(result.requests),
        "trims": sum(
            1 for prev, cur in zip(result.requests, result.requests[1:])
            if len(cur) <= len(prev) or _sha(cur[1]) != _sha(prev[1])
        ),
        "ledger_render": (
            hashlib.sha1(result.agent._ledger.render().encode("utf-8")).hexdigest()
            if getattr(result.agent, "_ledger", None) is not None else None
        ),
        # the tool schema of every request: one sha when it never changed
        "tools_sha": sorted({_sha(tools) for tools in result.tools}),
    }


# Env sets for the flags-off identity test. Import-time knobs are why each
# scenario runs in its own interpreter.
KAGGLE_ENV = {
    "MULTIMODAL_CONTEXT": "current_grid",
    "ARC3_DIFF_IMAGE": "1",
    "ARC3_GAMEOVER_DIFF_IMAGE": "1",
    "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_ACTION_INFO": "1",
    "ARC3_DEDUPE_MULTICALL_LINE": "1",
    "ARC3_FRAME_DIFF_HINT": "1",
    "ARC3_PERSISTENT_FUNCTIONS": "1",
    "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_REASONING_HISTORY_KEY": "reasoning_content",
    "ARC3_HISTORY_TURN_DRAIN": "4",
    "ARC3_HISTORY_DRAIN_COALESCE": "1",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "14000",
    "LOCAL_ANALYZER_MAX_OUTPUT": "1024",
    "ARC3_CONTEXT_DRAIN_TOKENS": "4000",
}
SCENARIOS: dict[str, dict[str, str]] = {
    "defaults": {"LOCAL_ANALYZER_CONTEXT_WINDOW": "12000", "LOCAL_ANALYZER_MAX_OUTPUT": "1024"},
    "kaggle": dict(KAGGLE_ENV),
    "kaggle-ledger": {**KAGGLE_ENV, "ARC3_LEDGER_ON_TRIM": "1"},
    "kaggle-ledger-death": {**KAGGLE_ENV, "ARC3_LEDGER_ON_TRIM": "1", "ARC3_DEATH_LEDGER": "1"},
}
# Flags added on top of memory-v2; with all of them off a memory-v2 arm must
# send exactly what arm/memory-v2 sent (tests/fixtures/memory_v2_on_golden.json).
MEMORY_V3_FLAGS = ("ARC3_MEM_PATH", "ARC3_MEM_PATH_AFTER_TOKENS", "ARC3_MEM_PATH_GAME_CHARS",
                   "ARC3_MEM_DEBRIEF", "ARC3_MEM_KEEP_EDGE")
# Every flag memory-v2 and later arms add; the flags-off identity test clears them all.
MEMORY_V2_FLAGS = (
    "ARC3_MEM_SOLVED", "ARC3_MEM_CARD", "ARC3_MEM_EFFECTS", "ARC3_MEM_SEEN",
    "ARC3_MEMORY_PLACEMENT", "ARC3_MEM_STATE_BORDER", "ARC3_MEM_CARD_CHARS",
    "ARC3_MEM_DELTA_CHARS", "ARC3_MEM_DELTA_GAME_CHARS", "ARC3_MEM_SEEN_GAME_CHARS",
    "ARC3_MEM_LINES_AFTER_TOKENS", "ARC3_MEM_REPLY_WINS", "ARC3_MEM_CODE_TAGS", *MEMORY_V3_FLAGS,
)
_MEMORY_V2_ON = {
    "ARC3_LEDGER_ON_TRIM": "1", "ARC3_MEM_SOLVED": "1", "ARC3_MEM_CARD": "1",
    "ARC3_MEM_EFFECTS": "1", "ARC3_MEM_SEEN": "1",
}
# memory-v2 arms as kaggle/arms.json runs them, with token usage so the 15k
# hold releases; golden from arm/memory-v2 in memory_v2_on_golden.json
MEMORY_V2_ON_SCENARIOS: dict[str, dict[str, str]] = {
    "kaggle-memv2-both": {**KAGGLE_ENV, **_MEMORY_V2_ON, "ARC3_MEMORY_PLACEMENT": "both",
                          "MEMSIM_COMPLETION_TOKENS": "4000"},
    "kaggle-memv2-trim": {**KAGGLE_ENV, **_MEMORY_V2_ON, "ARC3_MEMORY_PLACEMENT": "trim",
                          "MEMSIM_COMPLETION_TOKENS": "4000"},
}


def _slim(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        text = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
        images = sum(1 for p in content if isinstance(p, dict) and p.get("type") != "text")
    else:
        text, images = str(content or ""), 0
    return {"role": message.get("role"), "text": text, "images": images,
            "key": hashlib.sha1(json.dumps(message, sort_keys=True).encode("utf-8")).hexdigest()[:16]}


def dump(*, fail_turns: set[int] = frozenset(), turns: int | None = None) -> dict[str, Any]:
    """Every request as text (images counted, not kept) plus the memory state at
    the end, for the memory-v2 placement tests."""
    budget = int(os.environ.get("MEMSIM_BUDGET_TOKENS", "0") or 0) or None
    script = SCRIPT[:turns] if turns else None
    drops: list[tuple[str, int]] = []

    def setup(agent, seen):
        # where history really lost messages: ("turn", k) before request k,
        # ("persist", k) at the end of the turn whose last request was k - 1
        real_trim = agent._trim_messages_for_context
        real_persist = agent._persistent_history_messages

        def trim(messages, **kwargs):
            out = real_trim(messages, **kwargs)
            if len(out) < len(messages):
                drops.append(("turn", len(seen)))
            return out

        def persist(messages, *, tools=None):
            out = real_persist(messages, tools=tools)
            if len(out) < len(messages) - 1:
                drops.append(("persist", len(seen)))
            return out

        agent._trim_messages_for_context = trim
        agent._persistent_history_messages = persist

    result = run(budget_tokens=budget, fail_turns=fail_turns, script=script, setup=setup)
    agent = result.agent
    ledger = getattr(agent, "_ledger", None)
    recorder = getattr(agent, "_mem_recorder", None)
    out: dict[str, Any] = {
        "requests": [[_slim(m) for m in request] for request in result.requests],
        "turn_requests": result.turn_requests,
        "codes": result.codes,
        "drops": sorted(set(drops)),
        "ledger": None,
        "recorder": None,
    }
    if ledger is not None:
        out["ledger"] = {
            "world_model": list(ledger.world_model),
            "confirmed": list(ledger.confirmed),
            "refuted": list(ledger.refuted),
            "level_notes": list(ledger.level_notes),
            "pending_think": list(ledger._pending_think),
            "items": {k: v.text for k, v in ledger._items.items()},
            "pending": [e.text for e in ledger._queue],
            "full_card": ledger.render_card(),
        }
    if recorder is not None:
        out["recorder"] = {
            "level": recorder.attempts.level,
            "attempt": recorder.attempts.attempt,
            "seen_boards": len(recorder.seen.boards),
            "level_keys": sorted({k for k, _ in recorder.effects.level_counts}),
            "game_keys": sorted({k for k, _ in recorder.effects.game_counts}),
            "stats": dict(recorder.stats),
        }
    return out


def scenario_env(name: str) -> dict[str, str]:
    """A clean environment for one scenario: no ARC3_/analyzer knobs but its own."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("ARC3_", "LOCAL_ANALYZER_", "MULTIMODAL_", "EXPOSE_", "MEMSIM_"))
    }
    env.update(SCENARIOS[name] if name in SCENARIOS else MEMORY_V2_ON_SCENARIOS[name])
    return env


if __name__ == "__main__":
    name = sys.argv[1]
    if name == "replay":
        print(json.dumps(replay_ledger_blocks()))
    elif name == "dump":
        # python tests/memory_sim.py dump [fail_turn] [turns]  (-1 = none)
        import logging

        logging.disable(logging.CRITICAL)
        fail = int(sys.argv[2]) if len(sys.argv) > 2 else -1
        turns = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        print(json.dumps(dump(fail_turns={fail} if fail >= 0 else frozenset(), turns=turns or None)))
    else:
        print(json.dumps(fingerprints()))
