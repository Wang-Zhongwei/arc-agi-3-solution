"""EXPOSE_RESET through the real solver loop and the real python sandbox.

Everything is the real harness - the solver session's play loop (automatic
reset after a game over included), ToolAgent.analyze(), the sandboxed
``action(...)`` and the solver's step_env - except two edges: the HTTP post
goes to a fake server that replies with a fixed script of python tool calls,
and the engine is a small deterministic game that follows arcengine's RESET
rule: a RESET with no action since the level started (or was last reset)
restarts the whole game, unless ONLY_RESET_LEVELS=true. The sim leaves that
variable unset, as the competition gateway may, so a RESET the harness lets
through at zero actions shows up as a full reset.

Every request is rendered through the model's chat template
(tests/fixtures/qwen38_flash_next_chat_template.jinja, from
Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) so the prefix-cache property can be
checked on the prompt string the server actually sees.

Run in a fresh process (import-time env knobs), with the Kaggle notebook's
env applied by the caller:

    python tests/reset_loop_sim.py <out.json> [--inference-root DIR]

The script and its game are the same under every flag setting, so pointing
--inference-root at another checkout (e.g. main) with EXPOSE_RESET unset gives
request bodies that must match this checkout's byte for byte.
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
SIZE = 12
BG, WALL, PLAYER, GOAL, HAZARD, BAR = 5, 3, 9, 11, 8, 12
LEVELS = [
    dict(start=(6, 3), goal=(6, 8), hazard=(5, 3)),
    dict(start=(6, 3), goal=(6, 6), hazard=None),
]
MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}

# One python tool call per request, in order. Comments give what the engine
# has seen since the level last started from scratch, and what should happen
# with EXPOSE_RESET on. With it off, RESET is neither advertised nor guarded
# (legacy handling); the same script still runs, and the game's own RESET rule
# then decides what each one does.
SCRIPT = [
    "print(current_frame.shape)",                       # inspect
    "action(['RESET'])",                                # 0 since level start: refused
    "action(['RIGHT'])",                                # 1
    "print(len(history))",                              # inspect, new turn
    "action(['RESET'])",                                # 1 since start: level reset
    "action(['RESET'])",                                # 0 since that reset: refused
    "action(['UP'])",                                   # onto the hazard: game over, auto reset
    "action(['RESET'])",                                # 0 since the auto reset: refused
    "action(['RIGHT', 'RESET'])",                       # RESET not first: refused
    "action(['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT'])",  # level 1 complete
    "action(['RESET'])",                                # 0 since the level change: refused
    "action(['RIGHT', 'RIGHT', 'RIGHT'])",              # level 2 complete: won
]


class FakeGame:
    """The taaf.game.Game surface _HarnessGameSession uses, over a tiny engine."""

    def __init__(self) -> None:
        import arcengine

        self._ae = arcengine
        self.number_of_levels = len(LEVELS)
        self.level_index = 0
        self.levels_completed = 0
        self.pos = LEVELS[0]["start"]
        self.engine_count = 0  # arcengine's _action_count: non-RESET actions since set_level
        self.raw_state = arcengine.GameState.NOT_FINISHED
        self.engine_actions: list[str] = []
        self.full_resets = 0
        self.level_resets = 0
        self.game_run = SimpleNamespace(
            game_id="sim0", history=[], final_score=None, state="playing",
            solver_note=None, solver_analysis_html=None, levels_completed=0,
            number_of_levels=len(LEVELS), actions_per_level=[0] * len(LEVELS),
        )
        self.current_state = self._state(just_won=False)

    def _grid(self) -> list[list[int]]:
        rows = [[BG] * SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            rows[r][1] = rows[r][SIZE - 2] = WALL
        for c in range(min(self.engine_count, SIZE)):
            rows[0][c] = BAR  # actions spent on this attempt, in the edge row
        level = LEVELS[self.level_index]
        rows[level["goal"][0]][level["goal"][1]] = GOAL
        if level["hazard"]:
            rows[level["hazard"][0]][level["hazard"][1]] = HAZARD
        rows[self.pos[0]][self.pos[1]] = PLAYER
        return rows

    def _state(self, *, just_won: bool) -> SimpleNamespace:
        grid = self._grid()
        won = self.raw_state == self._ae.GameState.WIN
        return SimpleNamespace(
            levels_completed=self.levels_completed, won=won, just_won_level=just_won,
            available_actions=[0, 1, 2, 3, 4], frame=SimpleNamespace(data=grid),
            raw=SimpleNamespace(state=self.raw_state, frame=[grid]),
        )

    def _set_level(self, index: int) -> None:
        self.level_index = index
        self.pos = LEVELS[index]["start"]
        self.engine_count = 0

    def execute_action(self, action, *, generated_tokens, uncached_input_tokens):
        assert self.game_run.state == "playing"
        name = action.id.name
        self.engine_actions.append(name)
        before = self.levels_completed
        self.game_run.history.append(SimpleNamespace(action=name))
        self.game_run.actions_per_level[min(before, len(LEVELS) - 1)] += 1
        if name == "RESET":
            if self.engine_count == 0 and os.environ.get("ONLY_RESET_LEVELS") != "true":
                self.full_resets += 1
                self.levels_completed = 0
                self._set_level(0)
            else:
                self.level_resets += 1
                self._set_level(self.level_index)
            self.raw_state = self._ae.GameState.NOT_FINISHED
        else:
            self.engine_count += 1
            dr, dc = MOVES[name]
            r, c = self.pos[0] + dr, self.pos[1] + dc
            if 2 <= r <= SIZE - 3 and 2 <= c <= SIZE - 3:
                self.pos = (r, c)
            level = LEVELS[self.level_index]
            if self.pos == level["hazard"]:
                self.raw_state = self._ae.GameState.GAME_OVER
            elif self.pos == level["goal"]:
                self.levels_completed += 1
                if self.levels_completed == len(LEVELS):
                    self.raw_state = self._ae.GameState.WIN
                    self.game_run.state = "won"
                else:
                    self._set_level(self.level_index + 1)
        self.game_run.levels_completed = max(self.game_run.levels_completed, self.levels_completed)
        self.current_state = self._state(just_won=self.levels_completed > before)
        return self.current_state

    def finish_game(self) -> None:
        self.game_run.final_score = float(self.game_run.levels_completed)


def render(payload: dict[str, Any]) -> str:
    """The prompt string the server's chat template builds for this payload."""
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
        if isinstance(message.get("content"), list):  # images: keep text and image parts
            message["content"] = [p for p in message["content"] if p.get("type") in ("text", "image_url")]
        for call in message.get("tool_calls") or []:  # the server parses arguments for the template
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args or "{}")
    tools = [t["function"] if "function" in t else t for t in payload.get("tools") or []] or None
    prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                             **(payload.get("chat_template_kwargs") or {}))
    return prompt + (prefix or "")


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
    from inference.framework import solver as solver_mod

    workdir = Path(tempfile.mkdtemp(prefix="reset-sim-"))
    game = FakeGame()
    agent = ta.ToolAgent(model="flashnext", base_url="http://127.0.0.1:9/v1", provider="vllm")
    harness = solver_mod.HarnessSolver()
    harness.job_dir = None  # no viewer files
    # a finite deadline, which the priority gate's tail fade requires
    harness.max_runtime_s_per_game = 3600.0
    stop_event = threading.Event()
    session = solver_mod._HarnessGameSession(
        solver=harness, game=game, analyzer=agent, game_index=0, pass_index=0,
        state_path=workdir / "artifacts" / "sim_p0_tool_runtime_state.json",
        transcript_path=workdir / "transcripts" / "sim.txt",
        analysis_html_relpath="solver_analysis/sim.html",
        stop_event=stop_event, viewer_data_path=workdir / "viewer.json",
    )
    # Fixed clock: the action results the model reads carry the elapsed time,
    # which would otherwise make two runs' request bodies differ.
    session.timing_payload = lambda: {"run_elapsed_seconds": 1.0, "time_remaining_seconds": None}

    log: list[dict[str, Any]] = []
    queue = list(SCRIPT)

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):  # noqa: A002
        payload = __import__("json").loads(__import__("json").dumps(json))
        index = len(log)
        if queue:
            code = queue.pop(0)
        else:
            # script spent without a win: end the game at the next boundary
            stop_event.set()
            code = "print('done')"
        log.append({
            "index": index, "code": code, "body": payload, "prompt": render(payload),
            "engine_actions": list(game.engine_actions),
        })
        reasoning = f" step {index}."
        return Response({
            "choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": reasoning,
                                     "tool_calls": [{"id": f"call_{index}", "type": "function",
                                                     "function": {"name": "python",
                                                                  "arguments": __import__("json").dumps({"code": code})}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 40},
        })

    ta.requests.post = fake_post
    session.play()
    Path(out_path).write_text(json.dumps({
        "requests": log,
        "engine_actions": game.engine_actions,
        "full_resets": game.full_resets,
        "level_resets": game.level_resets,
        "levels_completed": game.game_run.levels_completed,
        "won": game.raw_state.name == "WIN",
        "script_left": len(queue),
        "history_actions": [entry.action for entry in session.history_entries],
        "transcript": session.transcript_path.read_text() if session.transcript_path.exists() else "",
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
