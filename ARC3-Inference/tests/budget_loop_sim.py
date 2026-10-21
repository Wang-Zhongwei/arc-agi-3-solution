"""End-to-end check of ARC3_BUDGET_TRUTH inside ToolAgent.analyze().

Everything is the real harness except two edges: the HTTP post goes to a fake
SGLang, and the environment behind ``action(...)`` is a small deterministic
game. The python tool itself is REAL: each reply's code runs in the harness
sandbox, so action results reach the model exactly as they would on Kaggle
(``_compact_action_result``, the sandbox's own printing, the auto-rendered
``result``, ``last_action_call_result``). The game's ``step_env`` returns the
solver's payload shape, built with the solver's own helpers, including the
run clock (``run_elapsed_seconds``, ``time_remaining_seconds``) from a fake
clock whose values are recorded so the test can look for them.

The fake server renders every request through the model's chat template
(tests/fixtures: chat_template.jinja from Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
as Pennyroyal's serving_chat does.

Each turn's first reply only inspects; one reply reports enough generated
tokens to trigger the 2048-token yield, so a resumption turn is exercised.

Run in a fresh process (import-time env knobs), with the Kaggle notebook's
env applied by the caller:

    python tests/budget_loop_sim.py <out.json> [--inference-root DIR]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "fixtures" / "qwen38_flash_next_chat_template.jinja"
SIZE, BORDER = 12, 2
BG, WALL, PLAYER, GOAL, BAR = 5, 3, 9, 11, 14
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
LEVELS = [dict(start=(6, 4), goal=(6, 7)), dict(start=(8, 8), goal=(4, 8))]
# Fake run clock: distinctive values, advanced on every step_env call.
CLOCK_START, CLOCK_TICK, RUN_SECONDS = 1801.25, 13.5, 7260.0

# Per turn: (inspect code, act code or None). Each act snippet reaches the
# action result by a different route the model really uses.
SCRIPT = [
    # the harness renders the result itself (no stdout)
    ("def probe():\n    return current_frame.shape\nprint(probe())\n",
     "action(['RIGHT'])\n"),
    # a retained function, the persisted result, then a printed batch result
    ("print(probe())\nprint(last_action_call_result)\n",
     "res = action(['RIGHT', 'RIGHT'])\nprint(res)\n"),
    # solver-side history results, then one field and the whole dict
    ("print(transitions[-1].result)\n",
     "res = action(['UP'])\nprint(res['action_num'], res)\n"),
    # yields on tokens: a resumption turn follows
    ("print(last_action_call_result)\n", None),
    ("print(sorted(last_action_call_result))\n",
     "print(action(['UP', 'UP', 'UP']))\n"),
]
YIELD_TURN = 3


class Game:
    """The solver's side of action(): runtime state plus step_env payloads."""

    def __init__(self, state_path: Path) -> None:
        from inference.agent.runtime_state import Frame, HistoryEntry

        self._frame_cls, self._entry_cls = Frame, HistoryEntry
        self.state_path = state_path
        self.level, self.action_count, self.level_actions, self.won = 1, 0, 0, False
        self.pos = LEVELS[0]["start"]
        self.history = [HistoryEntry(action="", frame=self.frame())]
        self.clock = CLOCK_START
        self.clock_values: list[float] = []
        self.write()

    def grid(self):
        rows = [[BG] * SIZE for _ in range(SIZE)]
        for r in range(1, SIZE):
            rows[r][BORDER - 1] = rows[r][SIZE - BORDER] = WALL
        for c in range(max(0, SIZE - self.level_actions)):  # the step bar, top row
            rows[0][c] = BAR
        gr, gc = LEVELS[self.level - 1]["goal"]
        rows[gr][gc] = GOAL
        rows[self.pos[0]][self.pos[1]] = PLAYER
        return tuple(tuple(row) for row in rows)

    def frame(self):
        return self._frame_cls(grid=self.grid(), step=self.action_count, level=self.level)

    def write(self) -> None:
        from inference.agent.runtime_state import write_runtime_state

        write_runtime_state(self.state_path, current_frame=self.frame(), history=self.history)

    def timing_payload(self) -> dict[str, float]:
        """solver.timing_payload(), on the fake clock."""
        self.clock += CLOCK_TICK
        elapsed, remaining = self.clock, RUN_SECONDS - self.clock
        self.clock_values += [elapsed, remaining]
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """The solver's step_env payload for a batch (see solver.step_env)."""
        from inference.framework import solver

        if str(arguments.get("query") or "").strip() == "animation":
            return {"executed": False, "query": "animation", "record": None}
        names = [str(item.get("action")) for item in arguments.get("actions") or []]
        executed: list[dict[str, Any]] = []
        stop_reason = None
        for index, name in enumerate(names, 1):
            before = self.grid()
            self.action_count += 1
            self.level_actions += 1
            dr, dc = MOVES[name]
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)
            done = completed = False
            if self.pos == LEVELS[self.level - 1]["goal"]:
                if self.level == len(LEVELS):
                    done = self.won = True
                else:
                    completed = True
                    self.level += 1
                    self.level_actions = 0
                    self.pos = LEVELS[self.level - 1]["start"]
            after = self.grid()
            payload = {
                "executed": True, "action_num": self.action_count, "level": self.level,
                "score": self.level - 1 + int(done), "reward": 0.5 if (completed or done) else 0.0,
                "state": "WIN" if done else "NOT_FINISHED", "valid_actions": list(MOVES),
                "board_changed": before != after, "gameplay_changed": before != after,
                "no_op": before == after, "done": done, "level_completed": completed,
                "game_over": False, "run_complete": done, "action_name": name, "action_data": {},
                "action_display": name, "batch_index": index, "batch_size": len(names),
                **self.timing_payload(), "automatic": False,
            }
            result_keys = (
                "executed", "action_num", "level", "score", "state", "valid_actions",
                "board_changed", "gameplay_changed", "no_op", "done", "level_completed",
                "game_over", "run_complete", "action_name", "action_data", "action_display",
                "animation", "automatic",
            )
            self.history.append(self._entry_cls(
                action=name, frame=self.frame(),
                result={k: payload[k] for k in result_keys if k in payload}))
            self.write()
            executed.append(payload)
            if done:
                stop_reason = "run_complete"
                break
            if completed:
                stop_reason = "level_completed"
                break
        final = dict(executed[-1])
        final.update(
            reward=sum(p["reward"] for p in executed), last_reward=executed[-1]["reward"],
            batched=len(names) > 1, requested_count=len(names), executed_count=len(executed),
            requested_actions=names, executed_actions=[p["action_display"] for p in executed],
            board_changed=any(p["board_changed"] for p in executed),
            stopped_early=len(executed) < len(names),
        )
        if stop_reason is not None:
            final["stop_reason"] = stop_reason
        if final["stopped_early"]:
            final["skipped_actions"] = names[len(executed):]
        if len(names) > 1:
            final["action_trace"] = solver._build_action_trace(names, executed, None)
            final["gameplay_changed_per_action"] = [p["gameplay_changed"] for p in executed]
        echo = solver._build_action_echo(names, executed, stop_reason)
        if echo:  # ARC3_ACTION_ECHO is on in the notebook
            final["action_echo"] = echo
        return final


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
    prefix = None
    if payload.get("continue_final_message") and messages and messages[-1]["role"] == "assistant" \
            and isinstance(messages[-1].get("content"), str):
        prefix = messages.pop()["content"]
    for message in messages:
        if isinstance(message.get("content"), list):  # images: count, keep text
            message["content"] = [p for p in message["content"] if p.get("type") in ("text", "image_url")]
        for call in message.get("tool_calls") or []:  # serving_chat parses arguments for the template
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args or "{}")
    tools = [t["function"] if "function" in t else t for t in payload.get("tools") or []] or None
    prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                             **(payload.get("chat_template_kwargs") or {}))
    return prompt + (prefix or "")


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if p.get("type") == "text")
    return str(content or "")


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

    workdir = Path(tempfile.mkdtemp(prefix="budget-sim-"))
    state_path = workdir / "artifacts" / "sim_p0_tool_runtime_state.json"
    game = Game(state_path)
    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    log: list[dict[str, Any]] = []
    state = {"turn": 0, "index": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        import json as _json

        payload = _json.loads(_json.dumps(json))
        state["index"] += 1
        turn, index = state["turn"], state["index"]
        inspect_code, act_code = SCRIPT[turn]
        acting = index >= 2 and act_code is not None
        code = act_code if acting else inspect_code
        reasoning = f" turn {turn} request {index}: {'act' if acting else 'inspect'}."
        messages = payload["messages"]
        log.append({
            "turn": turn, "index": index, "prompt": render(payload),
            "system": messages[0]["content"] if messages[0]["role"] == "system" else None,
            "tools": payload.get("tools"),
            "user_texts": [text_of(m) for m in messages if m["role"] == "user"],
            "tool_texts": [text_of(m) for m in messages if m["role"] == "tool"],
            "last_message": messages[-1],
            "generated_reasoning": reasoning,
        })
        tokens = 3000 if (turn == YIELD_TURN and not acting) else 40
        return Response({
            "choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": reasoning,
                                     "tool_calls": [{"id": f"call_{turn}_{index}", "type": "function",
                                                     "function": {"name": "python",
                                                                  "arguments": _json.dumps({"code": code})}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": tokens},
        })

    ta.requests.post = fake_post
    transcript = workdir / "transcript.txt"
    turns = []
    for turn in range(len(SCRIPT)):
        state["turn"], state["index"] = turn, 0
        agent.analyze(state_path, game.action_count, valid_actions=list(MOVES),
                      step_env=game.step_env, transcript_path=transcript, analysis_step=turn + 1)
        turns.append(state["index"])
    Path(out_path).write_text(json.dumps({
        "requests": log, "requests_per_turn": turns, "actions": game.action_count,
        "level": game.level, "won": game.won, "clock_values": game.clock_values,
        "game_elapsed_seconds": agent._game_elapsed_seconds,
        "transcript": transcript.read_text() if transcript.exists() else "",
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
