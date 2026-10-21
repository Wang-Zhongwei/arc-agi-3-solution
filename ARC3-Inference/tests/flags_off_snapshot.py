"""Print what the model is shown, as JSON, for the flags-off identity test.

Run as a script in a fresh interpreter (the harness reads some settings at
import time) with the scenario's environment already applied. It covers the
system prompt, three turn openers (an animated no-op, a game over inside an
animated batch, an ordinary step) as full user messages with images hashed,
and the python sandbox's globals. tests/test_flags_off_identity.py compares
the output with a snapshot taken from the code before the animation digest
existed, so every flag-gated addition must leave it byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


def _grid(fill: int = 0, marks: dict[tuple[int, int], int] | None = None):
    rows = [[fill] * 16 for _ in range(16)]
    for (row, col), value in (marks or {}).items():
        rows[row][col] = value
    return tuple(tuple(row) for row in rows)


def _hash_images(message):
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            url = str((part.get("image_url") or {}).get("url") or "")
            parts.append({"type": "image_url", "sha1": hashlib.sha1(url.encode()).hexdigest()})
        else:
            parts.append(part)
    return {**message, "content": parts}


def _openers(agent, Frame, HistoryEntry):
    before = _grid(0, {(8, 8): 9})
    flash = _grid(0, {(8, 8): 9, (9, 9): 8})
    animation = {"frames": 2, "transient_pixels": 1, "transient_bbox": [9, 9, 9, 9]}
    chain = [before, flash, before]
    current = Frame(grid=before, step=7, level=1)
    out = {}

    agent._last_animation_chain = chain
    agent._last_action_gameplay_changed = False
    summary = {
        "start_action_num": 7, "end_action_num": 7, "executed_count": 1,
        "executed_actions": ["MOUSE(row=9, col=9)"], "level": 1,
        "level_transition": False, "run_complete": False, "game_over": False,
        "game_over_trace": [], "fatal_action": "", "board_changed": True,
        "gameplay_changed": False, "stop_reason": None, "no_op_stops": [],
        "animation": animation, "animation_action": "MOUSE(row=9, col=9)",
        "guard_stops": [],
    }
    history = [
        HistoryEntry(action="", frame=Frame(grid=before, step=0, level=1)),
        HistoryEntry(action="MOUSE(row=9, col=9)", frame=current, result={"animation": animation}),
    ]
    prompt = agent._build_user_prompt(
        7, valid_actions=["MOUSE", "UP"], current_frame=current,
        history_entries=history, previous_step_summary=summary,
    )
    out["animated_noop"] = _hash_images(agent._build_user_message(prompt, current))

    death = _grid(0, {(8, 9): 9})
    restart = Frame(grid=_grid(0, {(1, 1): 9}), step=12, level=1)
    history = [
        HistoryEntry(action="", frame=Frame(grid=before, step=0, level=1)),
        HistoryEntry(action="UP", frame=Frame(grid=before, step=10, level=1)),
        HistoryEntry(action="RIGHT", frame=Frame(grid=death, step=11, level=1),
                     result={"animation": animation, "game_over": True}),
        HistoryEntry(action="RESET", frame=restart, result={"automatic": True}),
    ]
    agent._last_animation_chain = [before, flash, death]
    agent._last_action_gameplay_changed = True
    summary = dict(
        summary, start_action_num=10, end_action_num=11, executed_count=2,
        executed_actions=["UP", "RIGHT"], game_over=True,
        game_over_trace=["1: UP -> effect", "2: RIGHT -> GAME OVER"],
        fatal_action="RIGHT", gameplay_changed=True, animation_action="RIGHT",
        stop_reason="game_over",
    )
    prompt = agent._build_user_prompt(
        12, valid_actions=["UP", "RIGHT"], current_frame=restart,
        history_entries=history, previous_step_summary=summary,
    )
    out["game_over"] = _hash_images(agent._build_user_message(prompt, restart))

    plain = Frame(grid=_grid(0, {(2, 2): 9}), step=13, level=1)
    agent._last_animation_chain = []
    agent._last_action_gameplay_changed = True
    summary = dict(
        summary, start_action_num=13, end_action_num=13, executed_count=1,
        executed_actions=["DOWN"], game_over=False, game_over_trace=[],
        fatal_action="", gameplay_changed=True, animation=None,
        animation_action="DOWN", stop_reason=None,
    )
    history.append(HistoryEntry(action="DOWN", frame=plain))
    prompt = agent._build_user_prompt(
        13, valid_actions=["UP", "DOWN"], current_frame=plain,
        history_entries=history, previous_step_summary=summary,
    )
    out["plain"] = _hash_images(agent._build_user_message(prompt, plain))
    return out


def _sandbox_globals(run_sandboxed_python):
    code = (
        "names = sorted(n for n in dir() if not n.startswith('__'))\n"
        "print(names)\n"
        "if 'last_animation_frames' in names:\n"
        "    print(sorted(a for a in dir(type(last_animation_frames)) if not a.startswith('__')))\n"
        "    print(sorted(a for a in dir(type(last_animation_timeline)) if not a.startswith('__')))\n"
    )
    results = {}
    for label, handler in (("no_handler", None), ("animation_handler", lambda request: {})):
        result = run_sandboxed_python(
            code=code,
            timeout_seconds=20,
            initial_state={"current_frame": None, "history": [], "valid_actions": []},
            action_handler=lambda actions, stale_after=None: {},
            animation_handler=handler,
        )
        results[label] = {"stdout": result.get("stdout"), "error": result.get("error")}
    return results


def main() -> None:
    import inference
    from inference.agent import tool_agent
    from inference.agent.python_tool_sandbox import run_sandboxed_python
    from inference.agent.runtime_state import Frame, HistoryEntry

    agent = tool_agent.ToolAgent(model="snapshot-model")
    with tempfile.TemporaryDirectory() as tmp:
        agent._ensure_session(Path(tmp) / "state.json")
        snapshot = {
            "system_prompt": agent._system_prompt,
            "openers": _openers(agent, Frame, HistoryEntry),
            "sandbox": _sandbox_globals(run_sandboxed_python),
        }
    print(inference.__file__, file=sys.stderr)
    json.dump(snapshot, sys.stdout, indent=1, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
