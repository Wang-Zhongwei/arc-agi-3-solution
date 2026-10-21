"""End-to-end run of the animation layers through the real game loop.

Everything is the real harness except the HTTP post, which goes to a fake
SGLang. Unlike the opener-dedup and reasoning-scaffold loop sims, the game side
is real too: ``_HarnessGameSession.play()`` drives ``ToolAgent.analyze()``
exactly as a Kaggle game does (auto-RESET after a death, yield resumptions,
the priority gate), over ``solver_fakes.FakeGame`` with an animated script,
and the replies' python is executed in the real sandbox (``action([...])``,
``animation_for(n)``, ``last_animation_frames``). The fake server renders every
request through the model's chat template (tests/fixtures) and reports prompt
tokens from the rendered length, so the harness's context estimate, trims
and the ledger behave as against a server.

The game (24x24, border 4): a 2x2 block (colour 9) and a level marker.
  MOUSE(r, c)  flashes one cell and reverts: hides 1 cell, board area unchanged.
  MOUSE(0, c)  recolours border cell (0, c) for good, one frame: a bar tick
               (only the ANIM_SIM_LEGIBILITY replies click row 0).
  RIGHT        stretches the block 2 cells right and settles: a motion-only
               animation (every cell changes once; nothing hidden).
  UP           slides the block up 3 cells, one per state (transient cells,
               board changed).
  SPACE        nothing at all, one frame: a real no-op.
  DOWN         flashes a bar, then game over (the loop auto-RESETs).
  LEFT         completes the level (new board, one frame).
  RESET        the level's start board.

The reply script yields on the 2,048-token budget twice; with a small
LOCAL_ANALYZER_CONTEXT_WINDOW the history is trimmed several times. After the
first trim, the next reply also reads ``animation_for`` for the oldest
action still in the ring (or prints "RINGPROBE none" without the ring).
ANIM_SIM_RESET=1 adds a model-issued RESET (for EXPOSE_RESET=on runs).
ANIM_SIM_MEMORY_TAGS=1 adds memory tags to three replies that do not act (see
memory_tags), for the memory loop checks in test_memory_main_loop.py.
ANIM_SIM_LEGIBILITY=1 ends the script, on level 2, with replies that read what
the ARC3_LEGIBILITY fixes change (LEGIBILITY_SCRIPT): a retained function's
source, a segmentation node, the runs helper, the action numbering, a not-kept
animation, a caught stale-state refusal followed by a second call, a
border-only click in a batch and a frame_diff of it.
ANIM_SIM_DEBRIEF=1 makes the first reply after the level-up open its reasoning
with a recap of level 1 (DEBRIEF_RECAP), as the model does at about a third of
clears, for ARC3_MEM_DEBRIEF. ANIM_SIM_RECAP_ASKED=1 makes it open with the
recap ARC3_MEM_RECAP_ASK asks for (ASKED_RECAP) instead.
ANIM_SIM_OBJECTS=1 adds three objects to every board (a single yellow pixel
at (16, 14) and two 2x2 sky-blue squares at rows 16-17, cols 5-6 and 9-10) and,
on level 2, a click on row 19 that moves the pixel to (16, 17) and the right
square two cells left, so the squares touch (one 4x2 segmentation node); the
replies' reasoning then also states rules with a confirmation mark
(RULE_SENTENCES), for the arm/mem-unified checks. Without it every board,
reply and request is as before.
ANIM_SIM_FAIL_ON=<text> makes the first request whose newest message holds
<text> fail once (the server returns no choices), so that turn is rolled back
and retried; the failed request is listed under "failed", not "requests".
Every run also lists "handovers": the request indexes at which the agent
handed its slot over (`_maybe_handover` with `_context_was_trimmed` set).

    python tests/anim_loop_sim.py <out.json> [--inference-root DIR]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "fixtures" / "qwen38_flash_next_chat_template.jinja"
SIZE, BORDER = 24, 4
CHARS_PER_TOKEN = 3.5
FILLER = (
    "Looking at the board again: the block, the marker, the empty interior. "
    "The animation is the part to check before deciding what the action did. "
) * 10


def _click(r: int, c: int) -> str:
    return f"{{'action': 'MOUSE', 'row': {r}, 'col': {c}}}"


def _act(*actions: str, show: str = "") -> str:
    return f"r = action([{', '.join(actions)}])\n" + (f"print({show})\n" if show else "")


# One reply per request: (python code or None for a reply without a tool call, completion tokens).
SCRIPT: list[tuple[str | None, int]] = [
    ("print(current_frame.shape)", 40),
    (_act(_click(10, 10), show="r.get('animation')"), 40),                       # 1: hides
    ("print(len(last_animation_frames))", 40),
    (_act(_click(10, 12), _click(11, 12), _click(12, 12), "{'action': 'RIGHT'}",
          show="r.get('action_trace'), r.get('animations'), r.get('animations_unlisted')"), 40),  # 2-5
    ("print('thinking')", 3000),                                                   # yields
    ("names = dir()\nprint([k for k in ('animation_for', 'animations') if k in names])", 40),
    (_act("{'action': 'SPACE'}", show="r.get('gameplay_changed')"), 40),          # 6: real no-op
    ("print(len(last_animation_frames))", 40),
    (_act("{'action': 'UP'}", show="r.get('animation')"), 40),                    # 7: slide
    ("print('before the risky move')", 40),
    (_act("{'action': 'DOWN'}"), 40),                                             # 8: death (+ auto RESET)
    ("print(len(last_animation_frames), last_animation_timeline.get('action'))", 40),
    (_act(*[_click(14, c) for c in range(6, 15)], show="r.get('executed_count')"), 40),  # 10-18
    ("print('long think')", 3000),                                                 # yields
    ("try:\n    _n = animations[-1]['action_num']\n    print('FOR', _n, animation_for(_n)['timeline']['action_num'])\n"
     "except NameError:\n    print('no ring')", 40),
    (_act("{'action': 'RIGHT'}", "{'action': 'LEFT'}", _click(9, 10), show="r.get('stop_reason')"), 40),  # level up
    ("print('level 2')", 40),
    (_act(_click(10, 10)), 40),                                                    # same mechanic, new level
    ("print('again')", 2000),
    (_act(_click(10, 11), _click(10, 13)), 40),
    ("print('more')", 2000),
    (_act("{'action': 'RIGHT'}", "{'action': 'RIGHT'}"), 40),
    ("print('still more')", 2000),
    (_act(_click(15, 15)), 40),
    ("print('last')", 40),
    (_act(_click(16, 16), show="r.get('animation')"), 40),
    (None, 40),                                                                    # no tool call
]
RESET_SCRIPT = [
    ("print('try a reset')", 40),
    (_act("{'action': 'RESET'}", show="r.get('gameplay_changed')"), 40),
    ("print(len(last_animation_frames))", 40),
]
# ANIM_SIM_LEGIBILITY=1: inserted before the last reply. The same code runs with
# the flags off and on; only what it prints may differ.
LEGIBILITY_SCRIPT = [
    ("def twice(x):\n    return 2 * x\nprint('DEF', twice(2))", 40),
    ("print('SRC', retained_source.get('twice') if 'retained_source' in dir() else None)", 40),
    ("_n = current_frame.segmentation['nodes']\n"
     "print('NODE', sorted(_n[1]), _n[1].get('bbox'))", 40),
    ("print('RUNS', current_frame.runs(8, 9, 4, 9) if hasattr(current_frame, 'runs') else None)", 40),
    ("print('IDX', current_frame.step, len(transitions), transitions[-1].action)", 40),
    ("_d = animation_for(999)\nprint('NOTKEPT', bool(_d), sorted(_d))", 40),
    ("r = action([{'action': 'SPACE'}])\n"
     "for _ in range(2):\n"
     "    try:\n"
     "        action([{'action': 'UP'}])\n"
     "        print('STALE ran')\n"
     "    except StaleStateActionError as _e:\n"
     "        print('STALE', str(_e)[-150:])\n"
     "print('STEP', current_frame.step)", 40),                                    # a real no-op
    (_act(_click(0, 1), _click(0, 2), show="r.get('action_trace')"), 40),        # a bar tick
    ("_f = frame_diff()\n"
     "print('DIFF', _f['changed_cell_count'], [sorted(e) for e in _f['appeared'] + _f['disappeared']])", 40),
    ("print('IDX', current_frame.step, len(transitions), transitions[-1].action)", 40),
]
RING_PROBE = (
    "\ntry:\n"
    "    _a = animations\n"
    "    _n = _a[0]['action_num']\n"
    "    print('RINGFRAMES', _n, len(animation_for(_n)['frames']))\n"
    "except NameError:\n"
    "    print('RINGPROBE none')\n"
)


# ANIM_SIM_MEMORY_TAGS=1: the replies also write memory tags, so the ledger has
# something to show. Reply 0 states a world-model fact; reply 2 drafts a
# hypothesis in its reasoning and states the opposite in its reply (which one
# survives depends on ARC3_MEM_REPLY_WINS and the card); the reply before the
# risky move writes a bare tag in its python code, a SyntaxError unless the
# harness strips it (ARC3_MEM_CODE_TAGS). None of these replies acts, so the
# game is the same either way.
MEMORY_CODE_NOTE = "DOWN from the start row ends the attempt"
# ANIM_SIM_DEBRIEF=1: the reasoning of the first reply on level 2
DEBRIEF_RECAP = "Level 1 solved: the goal was to slide the block LEFT onto the exit, because LEFT completes the level."
# ANIM_SIM_RECAP_ASKED=1: the recap ARC3_MEM_RECAP_ASK asks for
ASKED_RECAP_BODY = "the goal is to slide the block onto the exit; LEFT moves the block one cell and reaching the exit completes the level."
ASKED_RECAP = f"Level 1 recap: {ASKED_RECAP_BODY}"


# ANIM_SIM_OBJECTS=1: sentences in the replies' reasoning, by reply index (all
# on level 1) or by the code of a level-2 reply. Two are rules with a
# confirmation mark, one restates the first (word overlap >= 0.6, so it
# replaces it), one is a question and one opens with filler: neither is kept.
RULE_SENTENCES = {
    1: "Confirmed: RIGHT always stretches the block two cells to the right, then it settles \u2713.",
    6: "Does SPACE always do nothing at all \u2713? Hmm, the block must move with UP \u2713.",
    9: "So DOWN from the start row always ends the attempt in GAME OVER, confirmed.",
    11: "Confirmed: RIGHT always stretches the block two cells right, then it settles there \u2713.",
}
OBJECT_RULE_CODE = "print('objects')"
OBJECT_RULE = "Verified: a click on row 19 moves the yellow pixel three cells right \u2713."
# the object scene, inserted on level 2 before the last reply
OBJECT_SCRIPT = [
    (OBJECT_RULE_CODE, 40),
    (_act(_click(19, 12)), 40),                                                    # the objects move
    ("print('after the objects moved')", 40),
]
OBJECT_COLOURS = (10, 11)
OBJECTS_AT_START = {(16, 14): 11, **{(16 + dr, c + dc): 10 for c in (5, 9) for dr in (0, 1) for dc in (0, 1)}}
OBJECTS_MOVED = {(16, 17): 11, **{(16 + dr, c + dc): 10 for c in (5, 7) for dr in (0, 1) for dc in (0, 1)}}


def objects_on() -> bool:
    return os.environ.get("ANIM_SIM_OBJECTS") == "1"


def _overlay(frame, cells: dict):
    rows = [list(row) for row in frame]
    for (r, c), value in cells.items():
        if rows[r][c] == 0:
            rows[r][c] = value
    return tuple(tuple(row) for row in rows)


def _objects_of(board) -> dict:
    return {(r, c): v for r, row in enumerate(board) for c, v in enumerate(row) if v in OBJECT_COLOURS}


def object_script(action_name: str, data: dict, board):
    """``script`` with ANIM_SIM_OBJECTS=1: the objects ride along on every
    frame (the start objects after a RESET or a level up), and a click on row
    19 moves them."""
    if action_name == "ACTION6" and int(data.get("y", 0)) == 19:
        bare = tuple(tuple(0 if v in OBJECT_COLOURS else v for v in row) for row in board)
        return [_overlay(bare, OBJECTS_MOVED)], {}
    frames, flags = script(action_name, data, board)
    keep = OBJECTS_AT_START if action_name == "RESET" or flags.get("level_up") else _objects_of(board)
    return [_overlay(frame, keep) for frame in frames], flags


def rule_reasoning(index: int, code: str | None, reasoning: str) -> str:
    if code == OBJECT_RULE_CODE:
        return f"{reasoning} {OBJECT_RULE}"
    if index in RULE_SENTENCES:
        return f"{reasoning} {RULE_SENTENCES[index]}"
    return reasoning


def memory_tags(index: int, code: str | None, reasoning: str) -> tuple[str | None, str, str]:
    if index == 0:
        return code, "<world_model>The 2x2 block of colour 9 is the player</world_model>", reasoning
    if index == 2:
        draft = ' Maybe <hypothesis status="confirmed">SPACE opens the door</hypothesis>. Wait, no.'
        final = '<hypothesis status="refuted">SPACE opens the door on contact</hypothesis>'
        return code, final, reasoning + draft
    if code == "print('before the risky move')":
        return f"<level_note>{MEMORY_CODE_NOTE}</level_note>\n{code}", "", reasoning
    return code, "", reasoning


def _grid(level: int, block: tuple[int, int] | None, extra: dict | None = None):
    rows = [[0] * SIZE for _ in range(SIZE)]
    rows[5][4 + level] = 3  # the level marker
    if block is not None:
        r, c = block
        for dr in (0, 1):
            for dc in (0, 1):
                rows[r + dr][c + dc] = 9
    for (r, c), value in (extra or {}).items():
        rows[r][c] = value
    return tuple(tuple(row) for row in rows)


def _block_at(board) -> tuple[int, int] | None:
    for r, row in enumerate(board):
        for c, value in enumerate(row):
            if value == 9:
                return r, c
    return None


def _level_of(board) -> int:
    return next(c - 4 for c, value in enumerate(board[5]) if value == 3)


def script(action_name: str, data: dict, board):
    level, block = _level_of(board), _block_at(board)
    start = _grid(level, (8, 6))
    if action_name == "RESET":
        return [start], {}
    if action_name == "ACTION6" and int(data.get("y", 0)) == 0:
        tick = [list(row) for row in board]
        tick[0][int(data.get("x", 0))] = 12
        return [tuple(map(tuple, tick))], {}
    if action_name == "ACTION6":
        r, c = int(data.get("y", 0)), int(data.get("x", 0))
        flash = [list(row) for row in board]
        flash[r][c] = 8
        return [tuple(map(tuple, flash)), board], {}
    if action_name == "ACTION4" and block is not None:
        r, c = block
        wide1 = _grid(level, block, {(r, c + 2): 9, (r + 1, c + 2): 9})
        wide2 = _grid(level, block, {(r, c + 2): 9, (r + 1, c + 2): 9, (r, c + 3): 9, (r + 1, c + 3): 9})
        return [wide1, wide2, _grid(level, (r, c + 2))], {}
    if action_name == "ACTION1" and block is not None:
        r, c = block
        return [_grid(level, (r - step, c)) for step in (1, 2, 3)], {}
    if action_name == "ACTION2":
        bar = [list(row) for row in board]
        for c in range(8, 14):
            bar[18][c] = 12
        return [tuple(map(tuple, bar)), _grid(level, None, {(18, 10): 12})], {"game_over": True}
    if action_name == "ACTION3":
        return [_grid(level + 1, (8, 6))], {"level_up": True}
    return [board], {}


def render(payload: dict[str, Any]) -> str:
    """The prompt string Pennyroyal's serving_chat builds for this payload."""
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(TEMPLATE.read_text())
    messages = json.loads(json.dumps(payload["messages"]))
    for message in messages:
        if isinstance(message.get("content"), list):
            message["content"] = [p for p in message["content"] if p.get("type") in ("text", "image_url")]
        for call in message.get("tool_calls") or []:  # serving_chat parses arguments for the template
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args or "{}")
    tools = [t["function"] if "function" in t else t for t in payload.get("tools") or []] or None
    return template.render(messages=messages, tools=tools, add_generation_prompt=True,
                           **(payload.get("chat_template_kwargs") or {}))


class Response:
    status_code = 200

    def __init__(self, body: dict[str, Any]) -> None:
        self._body, self.text = body, json.dumps(body)

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._body


def main(out_path: str) -> None:
    sys.path.insert(0, str(HERE))
    from inference.agent import tool_agent as ta
    from inference.framework import solver as solver_module
    from solver_fakes import FakeGame

    class Game(FakeGame):
        number_of_levels = 3

        def __init__(self, start, script_fn):
            super().__init__(start, script_fn)
            self.game_run = SimpleNamespace(
                state="playing", history=[], final_score=None, solver_note=None,
                solver_analysis_html="", levels_completed=0, number_of_levels=3,
                actions_per_level=[], game_id="sim-0001",
            )

            self.current_state.available_actions = [0, 1, 2, 3, 4, 5, 6]  # RESET too, as the engine lists it

        def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
            state = super().execute_action(action, generated_tokens, uncached_input_tokens)
            state.available_actions = [0, 1, 2, 3, 4, 5, 6]
            self.game_run.levels_completed = state.levels_completed
            return state

        def finish_game(self):
            self.game_run.final_score = 0.0
            if self.game_run.state == "playing":
                self.game_run.state = "finished"

    workdir = Path(tempfile.mkdtemp(prefix="anim-loop-sim-"))
    script_rows = list(SCRIPT)
    if os.environ.get("ANIM_SIM_RESET") == "1":
        # after the first action of level 2: the EXPOSE_RESET guard refuses a RESET at zero
        # actions since the level started
        script_rows[18:18] = RESET_SCRIPT
    if os.environ.get("ANIM_SIM_LEGIBILITY") == "1":
        script_rows[-1:-1] = LEGIBILITY_SCRIPT
    if objects_on():
        script_rows[-1:-1] = OBJECT_SCRIPT
        game = Game(_overlay(_grid(1, (8, 6)), OBJECTS_AT_START), object_script)
    else:
        game = Game(_grid(1, (8, 6)), script)
    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    stop = threading.Event()
    solver = SimpleNamespace(
        max_runtime_s_per_game=121 * 60.0, max_actions_per_game=None, max_generated_tokens_per_game=None,
        job_dir=None, soft_time_remaining_seconds=lambda: None, _claim_warmup_slot=lambda: False,
    )
    session = solver_module._HarnessGameSession(
        solver=solver, game=game, analyzer=agent, game_index=0, pass_index=0,
        state_path=workdir / "artifacts" / "sim_p0_tool_runtime_state.json",
        transcript_path=workdir / "transcript.txt", analysis_html_relpath="",
        stop_event=stop, viewer_data_path=workdir / "viewer.json",
    )
    # a deterministic clock in the action results, so two runs compare byte for byte
    session.timing_payload = lambda: {
        "run_elapsed_seconds": float(10 * session.action_count),
        "time_remaining_seconds": 121 * 60.0 - 10 * session.action_count,
    }

    handovers: list[int] = []
    real_handover = agent._maybe_handover

    def traced_handover() -> None:
        if agent._context_was_trimmed:
            handovers.append(len(log))
        real_handover()

    agent._maybe_handover = traced_handover
    fail_on = os.environ.get("ANIM_SIM_FAIL_ON", "")
    failed: list[dict[str, Any]] = []
    evictions = {"n": 0, "probed": 0}
    real_note = agent._note_history_evicted

    def note_evicted() -> None:
        evictions["n"] += 1
        real_note()

    agent._note_history_evicted = note_evicted
    log: list[dict[str, Any]] = []
    state = {"reply": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        payload = __import__("json").loads(__import__("json").dumps(json))
        if fail_on and not failed:
            newest = payload["messages"][-1].get("content")
            text = newest if isinstance(newest, str) else "".join(
                p.get("text") or "" for p in newest or [] if isinstance(p, dict))
            if fail_on in text:
                failed.append({"before_request": len(log), "messages": payload["messages"]})
                return Response({"error": "simulated failure"})
        index = state["reply"]
        state["reply"] += 1
        if index >= len(script_rows):
            stop.set()
            code, tokens = None, 40
        else:
            code, tokens = script_rows[index]
        if code is not None and "action(" not in code and evictions["n"] > evictions["probed"]:
            evictions["probed"] = evictions["n"]
            code += RING_PROBE
        log.append({
            "reply": state["reply"], "evictions": evictions["n"], "actions": session.action_count,
            "resume": bool(getattr(agent, "_resume_after_yield", False)),
            "messages": payload["messages"], "tools": payload.get("tools"),
            "prompt": render(payload),
        })
        reasoning = f"Reply {state['reply']}. " + FILLER
        content = ""
        if objects_on():
            reasoning = rule_reasoning(index, code, reasoning)
        if os.environ.get("ANIM_SIM_MEMORY_TAGS") == "1":
            code, content, reasoning = memory_tags(index, code, reasoning)
        if os.environ.get("ANIM_SIM_DEBRIEF") == "1" and (code or "").startswith("print('level 2')"):
            reasoning = f"{DEBRIEF_RECAP} Now level 2. " + reasoning
        if os.environ.get("ANIM_SIM_RECAP_ASKED") == "1" and (code or "").startswith("print('level 2')"):
            reasoning = f"{ASKED_RECAP} Now level 2. " + reasoning
        message: dict[str, Any] = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        finish = "stop"
        if code is not None:
            message["tool_calls"] = [{
                "id": f"call_{state['reply']}", "type": "function",
                "function": {"name": "python", "arguments": __import__("json").dumps({"code": code})}}]
            finish = "tool_calls"
        return Response({
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": int(len(log[-1]["prompt"]) / CHARS_PER_TOKEN),
                      "completion_tokens": tokens},
        })

    ta.requests.post = fake_post
    analyze = agent.analyze
    invocations: list[dict[str, Any]] = []

    def traced_analyze(*args, **kwargs):
        first = len(log)
        resumed = bool(getattr(agent, "_resume_after_yield", False))
        result = analyze(*args, **kwargs)
        invocations.append({
            "first_request": first, "requests": len(log) - first, "resumed": resumed,
            "yielded": bool(result and result.yielded_control),
            "executed": bool(result and result.step_executed),
        })
        return result

    agent.analyze = traced_analyze
    session.play()
    transcript = session.transcript_path.read_text() if session.transcript_path.exists() else ""
    Path(out_path).write_text(json.dumps({
        "requests": log, "invocations": invocations, "actions": session.action_count,
        "handovers": handovers, "failed": failed,
        "levels_completed": game.current_state.levels_completed,
        "engine_actions": [a.id.name for a in game.game_run.history],
        "replies_used": state["reply"], "script_len": len(script_rows),
        "transcript": transcript, "inference_file": str(Path(solver_module.__file__).resolve()),
    }))


if __name__ == "__main__":
    args = sys.argv[1:]
    root = HERE.parent
    if "--inference-root" in args:
        i = args.index("--inference-root")
        root = Path(args[i + 1])
        del args[i:i + 2]
    sys.path.insert(0, str(root))
    main(args[0])
