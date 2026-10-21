"""End-to-end run of ToolAgent.analyze() for the prefill-v2 checks (ARC3_OPENER_DEDUP).

Everything is the real harness except two edges, as in the reasoning-scaffold loop sim: the
HTTP post goes to a fake SGLang, and the python tool plays the reply's ``action([...])`` on a
small deterministic game. The fake server renders every request through the model's own
chat template (tests/fixtures, from Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) and reports
prompt tokens from the rendered length, so the harness's context estimate and trims behave
as they would against a server.

The driver calls analyze() the way solver.py does: a yielded invocation that executed
nothing is re-invoked with the same analysis_step (a resumption), anything else starts the
next step. The replies come from one fixed script, so every configuration sees the same
model; the script yields on the token budget several times, twice in a row once, and once
after a reply with no tool call.

Run in a fresh process (import-time env knobs), with the env applied by the caller:

    python tests/opener_dedup_loop_sim.py <out.json> [--inference-root DIR]
"""
from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "fixtures" / "qwen38_flash_next_chat_template.jinja"
SIZE, BORDER = 12, 2
BG, WALL, PLAYER, GOAL = 5, 3, 9, 11
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
LEVELS = [
    dict(start=(6, 4), goal=(6, 7)),     # RIGHT x3
    dict(start=(8, 8), goal=(4, 8)),     # UP x4
    dict(start=(5, 3), goal=(5, 8)),     # RIGHT x5
]
CHARS_PER_TOKEN = 3.5
# One reply per request, consumed in order: ("inspect", tokens), ("act", actions, tokens),
# ("bare", tokens) = reasoning only, no tool call. 3,000-token replies cross the 2,048 yield.
SCRIPT = [
    ("inspect", 40), ("act", ["RIGHT"], 40),
    ("inspect", 3000), ("inspect", 3000), ("act", ["RIGHT", "RIGHT"], 40),   # two resumptions
    ("bare", 3000), ("act", ["UP"], 40),                                     # yield on a bare reply
    ("inspect", 40), ("inspect", 2500), ("act", ["UP", "UP"], 40),
    ("inspect", 40), ("act", ["UP"], 40),
    ("inspect", 3000), ("act", ["RIGHT", "RIGHT"], 40),
    ("inspect", 40), ("act", ["RIGHT"], 40),
    ("inspect", 3000), ("inspect", 40), ("act", ["RIGHT", "RIGHT"], 40),
]
FILLER = (
    "Checking the segmentation again: the player block, the goal cell, the two walls. "
    "Nothing else on the board has changed, so the plan still holds. "
) * 12


class Game:
    def __init__(self) -> None:
        from inference.agent.runtime_state import Frame, HistoryEntry

        self._frame_cls, self._entry_cls = Frame, HistoryEntry
        self.level, self.action_count, self.won = 1, 0, False
        self.pos = LEVELS[0]["start"]
        self.history = [HistoryEntry(action="", frame=self.frame())]

    def grid(self):
        rows = [[BG] * SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            rows[r][BORDER - 1] = rows[r][SIZE - BORDER] = WALL
        gr, gc = LEVELS[self.level - 1]["goal"]
        rows[gr][gc] = GOAL
        rows[self.pos[0]][self.pos[1]] = PLAYER
        return tuple(tuple(row) for row in rows)

    def frame(self):
        return self._frame_cls(grid=self.grid(), step=self.action_count, level=self.level)

    def step(self, actions: list[str]) -> list[dict[str, Any]]:
        executed = []
        for index, name in enumerate(actions, 1):
            before = self.grid()
            self.action_count += 1
            dr, dc = MOVES[name]
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)
            done = completed = False
            if self.pos == LEVELS[self.level - 1]["goal"]:
                if self.level == len(LEVELS):
                    done = self.won = True
                else:
                    completed = True
                    self.level += 1
                    self.pos = LEVELS[self.level - 1]["start"]
            after = self.grid()
            result = {
                "executed": True, "action_num": self.action_count, "level": self.level,
                "score": self.level - 1, "state": "WIN" if done else "NOT_FINISHED",
                "valid_actions": list(MOVES), "board_changed": before != after,
                "gameplay_changed": before != after, "no_op": before == after, "done": done,
                "level_completed": completed, "game_over": False, "run_complete": done,
                "action_name": name, "action_data": {}, "action_display": name, "automatic": False,
                "batch_index": index, "batch_size": len(actions),
            }
            self.history.append(self._entry_cls(action=name, frame=self.frame(), result=dict(result)))
            executed.append(result)
            if completed or done:
                break
        return executed


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
    from inference.agent import tool_agent as ta
    from inference.agent.runtime_state import write_runtime_state
    from inference.agent.vision_context import current_grid_image_part

    workdir = Path(tempfile.mkdtemp(prefix="opener-dedup-sim-"))
    state_path = workdir / "artifacts" / "sim_p0_tool_runtime_state.json"
    game = Game()
    write_runtime_state(state_path, current_frame=game.frame(), history=game.history)
    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    evictions = {"n": 0}
    real_note = agent._note_history_evicted

    def note_evicted() -> None:
        evictions["n"] += 1
        real_note()

    agent._note_history_evicted = note_evicted
    log: list[dict[str, Any]] = []
    state = {"reply": 0, "invocation": 0, "step": 0, "resumed": False, "board_image": None}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        payload = __import__("json").loads(__import__("json").dumps(json))
        spec = SCRIPT[state["reply"]]
        state["reply"] += 1
        kind = spec[0]
        tokens = spec[-1]
        reasoning = f"Reply {state['reply']} ({kind}). " + FILLER
        log.append({
            "reply": state["reply"], "invocation": state["invocation"], "step": state["step"],
            "resumed": state["resumed"], "board_image": state["board_image"],
            "evictions": evictions["n"], "messages": payload["messages"], "tools": payload.get("tools"),
            # the static tail of this invocation's opener as the prompt builder recorded it
            # (None on a checkout without ARC3_OPENER_DEDUP)
            "static_block": getattr(agent, "_opener_static_block", None),
            "prompt": render(payload),
        })
        message: dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": reasoning}
        finish = "stop"
        if kind != "bare":
            code = (f"res = action({__import__('json').dumps(spec[1])})\nprint(res)\n" if kind == "act"
                    else "print(current_frame.shape)\n")
            message["tool_calls"] = [{
                "id": f"call_{state['reply']}", "type": "function",
                "function": {"name": "python", "arguments": __import__("json").dumps({"code": code})}}]
            finish = "tool_calls"
        return Response({
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": int(len(log[-1]["prompt"]) / CHARS_PER_TOKEN),
                      "completion_tokens": tokens},
        })

    def fake_python(path, arguments):
        code = str(arguments.get("code", ""))
        match = re.search(r"action\((\[.*?\])\)", code, re.S)
        if not match:
            return ta._ToolDispatchResult(json.dumps({"tool": "python", "returncode": 0, "stdout": str((SIZE, SIZE))}))
        executed = game.step([str(a) for a in ast.literal_eval(match.group(1))])
        write_runtime_state(state_path, current_frame=game.frame(), history=game.history)
        final = dict(executed[-1])
        final.update(executed_count=len(executed), executed_actions=[e["action_display"] for e in executed])
        agent._last_step_summary = agent._summarize_step_sequence([final])
        return ta._ToolDispatchResult(
            json.dumps({"tool": "python", "returncode": 0, "stdout": json.dumps(final)}),
            step_executed=True, made_progress=True)

    ta.requests.post = fake_post
    agent._run_python_tool = fake_python
    # as solver.py does before a game's first turn: the gate's shared clock (the
    # competition's tail fade needs a deadline) and the game's initial admission,
    # so a trim's handover runs the real gate code
    gate = ta._priority_gate()
    if gate is not None:
        import time
        gate.configure_clock(time.monotonic(), time.monotonic() + 7260.0)
        gate.acquire(ta._PRIORITY_UNTRIMMED_BASE)
    transcript = workdir / "transcript.txt"
    invocations = []
    step = 1
    while state["reply"] < len(SCRIPT) and not game.won:
        state["invocation"] += 1
        state["step"] = step
        state["resumed"] = bool(getattr(agent, "_resume_after_yield", False))
        state["board_image"] = (current_grid_image_part(game.frame()) or {}).get("image_url", {}).get("url")
        result = agent.analyze(state_path, game.action_count, valid_actions=list(MOVES),
                               transcript_path=transcript, analysis_step=step)
        invocations.append({"step": step, "resumed": state["resumed"],
                            "yielded": bool(result and result.yielded_control),
                            "executed": bool(result and result.step_executed)})
        if result is None:
            raise RuntimeError("analyze() returned None")
        if not (result.yielded_control and not result.step_executed):
            step += 1
    Path(out_path).write_text(json.dumps({
        "requests": log, "invocations": invocations, "actions": game.action_count,
        "level": game.level, "won": game.won, "replies_used": state["reply"],
        "transcript": transcript.read_text() if transcript.exists() else "",
        "context_budget": agent._context_budget_tokens,
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
