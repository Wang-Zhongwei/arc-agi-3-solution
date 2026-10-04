import ast
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.framework.kaggle import duck_kaggle_setup_command
from inference.utils.grid_utils import ARC_COLOR_CHARS


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("ARC3_HUD_DISPLAY", "true")
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "true")
    agent = tool_agent.ToolAgent(model="test-hud")
    agent._kept_functions = {"hud": "def hud(f=current_frame):\n    return {'step': f.step}"}
    return agent


def frame(step, budget=2):
    return Frame(grid=(tuple([4] * budget + [0] * (4 - budget)),), step=step, level=1)


def test_hud_defined_in_previous_call_reads_current_frame_and_dependencies(agent):
    color = repr(ARC_COLOR_CHARS[4])
    code = f"""
def budget(f=current_frame):
    return sum(n['pixels'] for n in f.segmentation['nodes'] if n['color'] == {color})
def hud(f=current_frame):
    return {{'budget': budget(f), 'step': f.step, 'before': previous_frame.step}}
"""
    previous = frame(1, 3)
    response = run_sandboxed_python(
        code=code, timeout_seconds=5, kept_functions=[],
        initial_state={"current_frame": tool_agent._ascii_frame_view_payload(previous)},
        action_handler=lambda _: pytest.fail("HUD definition must not act"),
    )
    assert response["error"] == ""
    agent._record_retained_functions(response, {})
    assert set(agent._kept_functions) == {"budget", "hud"}
    kept = dict(agent._kept_functions)

    for step, remaining in ((2, 2), (3, 1), (4, 4)):
        current = frame(step, remaining)
        history = [HistoryEntry("", previous), HistoryEntry("ACTION1", current)]
        text = agent._hud_display_context(current, history)
        reading = json.loads(text.split(": ", 1)[1])
        assert reading == {"budget": remaining, "step": step, "before": previous.step}
        assert agent._kept_functions == kept
        previous = current


@pytest.mark.parametrize("missing", ["option", "retention", "hud", "frame"])
def test_disabled_or_missing_hud_does_not_run_sandbox(agent, monkeypatch, missing):
    current = frame(2)
    if missing == "option":
        monkeypatch.setenv("ARC3_HUD_DISPLAY", "false")
    elif missing == "retention":
        monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "false")
    elif missing == "hud":
        agent._kept_functions = {}
    else:
        current = None
    sandbox = Mock(side_effect=AssertionError("Display should be skipped"))
    monkeypatch.setattr(tool_agent, "run_sandboxed_python", sandbox)
    assert agent._hud_display_context(current, []) == ""
    sandbox.assert_not_called()


@pytest.mark.parametrize("caught", [False, True])
def test_automatic_hud_cannot_act_even_if_it_catches_the_error(agent, caught):
    body = ("    try:\n        action(['LEFT'])\n    except RuntimeError:\n        pass\n"
            if caught else "    action(['LEFT'])\n")
    agent._kept_functions = {"hud": "def hud(f=current_frame):\n" + body + "    return 99"}
    game_callback = Mock(side_effect=AssertionError("Automatic display must not touch game"))
    agent._step_env_callback = game_callback
    assert "unavailable (hud() attempted an action)" in agent._hud_display_context(frame(2), [])
    game_callback.assert_not_called()


def test_hud_failure_keeps_definition_and_does_not_break_user_message(agent, monkeypatch):
    agent._kept_functions = {"hud": "def hud(f=current_frame):\n    return 1 / 0"}
    sources = dict(agent._kept_functions)
    text = agent._hud_display_context(frame(2), [])
    assert "unavailable" in text and "ZeroDivisionError" in text
    assert agent._kept_functions == sources
    monkeypatch.setattr(tool_agent, "current_grid_image_part", lambda _: None)
    message = agent._build_user_message("Next turn", frame(2), hud_context=text)
    assert "Next turn" in message["content"] and text in message["content"]


def test_hud_timeout_is_bounded(agent):
    agent._python_timeout = 1
    agent._kept_functions = {"hud": "def hud(f=current_frame):\n    while True:\n        pass"}
    assert "unavailable" in agent._hud_display_context(frame(2), [])


@pytest.mark.parametrize("source", [
    "def hud(f=current_frame):\n    return 'x' * 5000",
    "def hud(f=current_frame):\n    print('x' * 5000)",
])
def test_returned_and_printed_hud_output_are_bounded(agent, source):
    agent._kept_functions = {"hud": source}
    text = agent._hud_display_context(frame(2), [])
    assert "[truncated]" in text
    assert len(text) < 1100


def test_hud_supports_retained_import_dependencies(agent, monkeypatch):
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS_IMPORTS", "true")
    agent._kept_functions = {
        "hud": "from collections import Counter\ndef hud(f=current_frame):\n    return dict(Counter(f.ascii))",
    }
    text = agent._hud_display_context(frame(2), [])
    assert json.loads(text.split(": ", 1)[1])[ARC_COLOR_CHARS[4]] == 2


@pytest.mark.parametrize("image", [None, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}])
def test_hud_is_present_in_text_and_image_user_messages(agent, monkeypatch, image):
    monkeypatch.setattr(tool_agent, "current_grid_image_part", lambda _: image)
    text = agent._hud_display_context(frame(2), [])
    # Equivalent to turn-scoped clearing: the reading survives in the opener.
    agent._kept_functions = {}
    for opener in ("Full turn", "Short resumption", "State-only resumption"):
        message = agent._build_user_message(opener, frame(2), hud_context=text)
        content = message["content"]
        rendered = content if isinstance(content, str) else content[0]["text"]
        assert rendered.count("Current HUD") == 1
        assert '"step":2' in rendered


@pytest.mark.parametrize("mode", ["normal", "full", "short", "state_only"])
def test_analyzer_opener_uses_hud_before_turn_scoped_clearing(agent, monkeypatch, tmp_path, mode):
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS_SCOPE", "turn")
    monkeypatch.setenv("ARC3_YIELD_RESUME_PROMPT", mode if mode != "normal" else "full")
    monkeypatch.setattr(tool_agent, "current_grid_image_part", lambda _: None)
    path = tmp_path / "runtime.json"
    current = frame(2)
    write_runtime_state(path, current_frame=current, history=[HistoryEntry("", frame(1))])
    agent._ensure_session(path)
    agent._kept_functions = {"hud": "def hud(f=current_frame):\n    return {'step': f.step}"}
    agent._resume_after_yield = mode != "normal"
    monkeypatch.setattr(agent, "_maybe_append_rolling_summary", lambda *args, **kwargs: None)
    callback = Mock(side_effect=AssertionError("Prompt construction must not act"))
    transcript = tmp_path / "transcript.txt"
    agent.analyze(path, 2, valid_actions=["LEFT"], step_env=callback,
                  transcript_path=transcript, should_stop=lambda: True)
    text = transcript.read_text()
    assert text.count("Current HUD (your hud() reading)") == 1
    assert '"step":2' in text
    assert bool(agent._kept_functions) == (mode != "normal")
    callback.assert_not_called()


@pytest.mark.parametrize("option,retention", [(False, False), (True, False), (True, True)])
def test_system_guidance_matches_effective_enablement(monkeypatch, option, retention):
    monkeypatch.setenv("ARC3_HUD_DISPLAY", str(option).lower())
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", str(retention).lower())
    prompt = tool_agent._build_system_prompt(tool_output_tokens=1024)
    assert ("later user prompts show its reading" in prompt) == (option and retention)


@pytest.mark.parametrize("enabled,override", [(False, None), (True, None), (True, "false")])
def test_json_config_reaches_make_and_kaggle_setup(tmp_path, monkeypatch, enabled, override):
    config = tmp_path / "hud.json"
    config.write_text(json.dumps({"analyzer": {"persistent_functions": enabled, "hud_display": enabled}}))
    env = dict(os.environ)
    for name in ("ARC3_HUD_DISPLAY", "ARC3_PERSISTENT_FUNCTIONS"):
        env.pop(name, None)
    if override is not None:
        env["ARC3_HUD_DISPLAY"] = override
    result = subprocess.run(
        ["make", "--no-print-directory", "-s", "-f", "Makefile", "-f", "-", "hud-config", f"CONFIG_PATH={config}"],
        cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True, check=True,
        input="hud-config:\n\t@python3 -c 'import os,json; print(json.dumps({k:os.environ[k] for k in (\"ARC3_PERSISTENT_FUNCTIONS\",\"ARC3_HUD_DISPLAY\")}))'\n",
    )
    exported = json.loads(result.stdout)
    assert exported["ARC3_PERSISTENT_FUNCTIONS"] == str(enabled).lower()
    assert exported["ARC3_HUD_DISPLAY"] == (override or str(enabled).lower())
    monkeypatch.setenv("LOCAL_ANALYZER_PROVIDER", "vllm")
    for name, value in exported.items():
        monkeypatch.setenv(name, value)
    command = duck_kaggle_setup_command()
    script = "\n".join(command.splitlines()[1:-1])
    tree = ast.parse(script)
    setup = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "setup_env" for target in node.targets))
    actual = {key.value: ast.literal_eval(value) for key, value in zip(setup.keys, setup.values)
              if isinstance(key, ast.Constant) and key.value in exported}
    assert actual == exported
