"""Each action's animation lands on its own history entry, in every case.

A scripted TAAF game drives the real harness session, ToolAgent and sandbox
through animated and plain actions in a batch, single actions within one
snippet, a refused call, a voluntary RESET, a death and the automatic RESET
after it, a level-up with the solved board, one with a single engine frame,
and the winning action. After every call the sandbox checks each history
entry's `.animation`, its `.frames` and `.timeline`, `transitions[j].animation`,
and `last_animation_frames` / `last_animation_timeline` against chains rebuilt
from the frames the game itself returned. TAAF pads a one-frame level
transition with an all-green frame; that frame was reported as an animation
until _animation_chain dropped it.
"""

import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import arcengine
import numpy as np
import taaf.game

from inference.agent import tool_agent
from inference.framework import solver as solver_module
from inference.utils.animation import build_animation_view, collapse_frames, normalize_frames, summarize_animation

SIZE = 16
START = (8, 8)


@dataclass
class _ScriptedGame(taaf.game.Game):
    def _start_game(self, session):
        self.game_id = "scripted-0000"
        self.number_of_levels = 3
        self.grid_size = (SIZE, SIZE)
        self.level, self.pos = 0, START
        self.emitted = []  # the frames the game returned, by action number
        return self._state([self._board()])

    def _board(self, extra=None, fill=None):
        board = np.zeros((SIZE, SIZE), dtype=np.int8)
        if fill is not None:
            board[4:12, 4:12] = fill
        board[5, 4 + self.level] = 9
        board[self.pos] = 3
        for cell, colour in (extra or {}).items():
            board[cell] = colour
        return board

    def _state(self, frames, state=arcengine.GameState.NOT_FINISHED):
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id, state=state, levels_completed=self.level,
            win_levels=self.number_of_levels, available_actions=[1, 2, 3, 4, 5, 6],
        )
        raw.frame = frames
        self.emitted.append([frame.copy() for frame in frames])
        return taaf.game.GameState(raw=raw)

    def _execute_action(self, action):
        name, (row, col) = action.id.name, self.pos
        if name == "RESET":
            self.pos = START
            return self._state([self._board()])
        if name == "ACTION1":  # a row flashes and clears while the player steps up
            flash = self._board({(10, c): 15 for c in range(4, 12)})
            self.pos = (row - 1, col)
            return self._state([flash, self._board()])
        if name == "ACTION2":  # one frame
            self.pos = (row + 1, col)
            return self._state([self._board()])
        if name == "ACTION3":  # a slide: every cell changes once
            self.pos = (row, col - 1)
            return self._state([self._board({(row, col): 3}), self._board()])
        if name == "ACTION4":  # fatal: a flash, then the death board
            flash = self._board({(9, c): 12 for c in range(4, 12)})
            return self._state([flash, self._board(fill=13)], arcengine.GameState.GAME_OVER)
        # ACTION5 completes the level showing the solved board, ACTION6 in one frame
        solved = self._board({(7, c): 9 for c in range(4, 12)})
        self.level, self.pos = self.level + 1, START
        won = self.level == self.number_of_levels
        frames = [solved, self._board()] if name == "ACTION5" else [self._board()]
        return self._state(frames, arcengine.GameState.WIN if won else arcengine.GameState.NOT_FINISHED)


class _Session(solver_module._HarnessGameSession):
    def should_stop(self, *, soft=False):
        return False

    def timing_payload(self):
        return {}

    def _append_action_viewer_event(self, *args, **kwargs):
        pass

    def write_viewer_payload(self):
        pass


def _expected(session):
    """What each entry and the last_animation_* globals should hold, from the game's own frames."""
    states, emitted = session.game.game_run.intermediate_states, session.game.emitted
    entries, last = [], None
    for entry in session.history_entries:
        num = entry.result.get("action_num")
        if not num or states[num].previous_action.id.name == "RESET":
            entries.append(None)  # the first board, or a RESET: neither touches a record
            continue
        chain = collapse_frames(normalize_frames(emitted[num - 1])[-1], normalize_frames(emitted[num]))
        if len(chain) <= 2:
            entries.append(None)
            last = None
            continue
        timeline = build_animation_view({"chain": chain, "action_display": entry.action})
        last = {
            "summary": summarize_animation(chain), "action": entry.action,
            "frames": [[list(row) for row in grid] for grid in chain[1:]],
            "diffs": [diff["frame_index"] for diff in timeline["frame_diffs"]],
        }
        entries.append(last)
    return {"entries": entries, "last": last}


_CHECK = """
bad = []
for i, h in enumerate(history):
    exp, a = EXPECTED["entries"][i], h.animation
    if exp is None:
        if a is not None:
            bad.append((i, h.action, "should have no animation"))
        continue
    if a is None:
        bad.append((i, h.action, "animation missing"))
        continue
    if dict(a) != exp["summary"]:
        bad.append((i, h.action, "summary", dict(a)))
    if [f._grid for f in a.frames] != exp["frames"]:
        bad.append((i, h.action, "frames"))
    if h.frame._grid != exp["frames"][-1]:
        bad.append((i, h.action, "settled board is not the last frame"))
    t = a.timeline
    if t["action"] != exp["action"] or [d["frame_index"] for d in t["frame_diffs"]] != exp["diffs"]:
        bad.append((i, h.action, "timeline", t["action"]))
for j, t in enumerate(transitions):
    if t.animation is not history[j + 1].animation:
        bad.append(("transition", j))
last = EXPECTED["last"]
if [f._grid for f in last_animation_frames] != (last["frames"] if last else []):
    bad.append(("last_animation_frames", len(last_animation_frames)))
if last_animation_timeline["action"] != (last["action"] if last else None):
    bad.append(("last_animation_timeline", last_animation_timeline["action"]))
result = bad
"""


def test_every_animation_is_attributed_to_its_own_action(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "ANIMATION_ENABLED", True)
    monkeypatch.setenv("ARC3_ANIMATION", "1")
    monkeypatch.setenv("EXPOSE_RESET", "on")
    game = _ScriptedGame()
    game.start_game()
    session = _Session(
        solver=None, game=game, analyzer=SimpleNamespace(generated_tokens=0), game_index=0,
        pass_index=0, state_path=tmp_path / "state.json", transcript_path=tmp_path / "t.txt",
        analysis_html_relpath="a.html", stop_event=threading.Event(),
        viewer_data_path=tmp_path / "viewer.json",
    )
    session.seed_initial_history()
    session.write_runtime_state()
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    agent._step_env_callback = session.step_env
    agent._current_valid_actions = solver_module.to_model_actions(solver_module._engine_action_names(game))

    def run(code):
        payload = json.loads(agent._run_python_tool(session.state_path, {"code": code}).content)
        assert "Traceback" not in json.dumps(payload), payload
        return payload

    steps = [
        ("batch: flash, plain, slide, flash", "action(['UP', 'DOWN', 'LEFT', 'UP'])", None),
        ("plain action empties last_animation_*", "action(['DOWN'])", None),
        ("single actions in one snippet each refresh last_animation_*",
         "action(['UP'])\nfirst = len(last_animation_frames)\naction(['DOWN'])\n"
         "result = [first, len(last_animation_frames)]", [2, 0]),
        ("a refused call changes nothing",
         "try:\n    action(['UP', 'RESET'])\nexcept RuntimeError as exc:\n    result = 'refused'", "refused"),
        ("animated action before a RESET", "action(['UP'])", None),
        ("voluntary RESET keeps the record", "action(['RESET'])", None),
        ("death at the end of a batch", "action(['UP', 'RIGHT'])", None),
        ("automatic RESET after the death", None, None),
        ("level-up with the solved board", "action(['SPACE'])", None),
        ("level-up in one engine frame", "action([{'action': 'MOUSE', 'row': 3, 'col': 3}])", None),
        ("winning action", "action(['SPACE'])", None),
    ]
    for label, code, want in steps:
        if code is None:
            session._execute_auto_reset()
        else:
            got = run(code).get("result")
            assert want is None or got == want, (label, got)
        expected = _expected(session)
        bad = run(f"EXPECTED = {expected!r}\n" + _CHECK).get("result")
        assert bad == [], (label, bad)

    actions = [entry.action for entry in session.history_entries]
    animated = [entry.action for entry in session.history_entries if entry.result.get("animation")]
    assert actions == ["", "UP", "DOWN", "LEFT", "UP", "DOWN", "UP", "DOWN", "UP", "RESET", "UP", "RIGHT",
                       "RESET", "SPACE", "MOUSE(row=3, col=3)", "SPACE"], actions
    # the one-frame level-up is not an animation: TAAF's green frame is dropped
    assert animated == ["UP", "LEFT", "UP", "UP", "UP", "UP", "RIGHT", "SPACE", "SPACE"], animated
    nums = [entry.result.get("action_num") for entry in session.history_entries[1:]]
    assert nums == list(range(1, len(nums) + 1))
