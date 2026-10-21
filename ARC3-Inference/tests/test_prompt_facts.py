"""The prompt-facts flags under the prompt-facts-10x arm's env.

The arm is user-combo-10x plus ARC3_BUDGET_TRUTH, ARC3_PROMPT_HYGIENE,
ARC3_IMAGE_GEOMETRY and ARC3_PREFER_TOOL_CALLS. Each config is rendered in a
fresh interpreter (some knobs are read at import time) under the Kaggle
notebook's setup_env plus user-combo-10x's env from kaggle/arms.json, by
tests/flags_off_snapshot.py: the system prompt, three full turn openers (an
animated no-op, a game over, an ordinary step) and the sandbox globals.

Each flag alone must change exactly its own text and nothing else, and all
four together must equal the composition of the four. The loop check runs
the real analyze() loop (tests/budget_loop_sim.py) with all four on and off.
BUDGET_SIM_BASELINE_ROOT=<another checkout's ARC3-Inference> also compares
the flags-off requests under this env with that checkout's, byte for byte.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

import pytest
from PIL import Image

from inference.agent import prompts
from inference.agent import tool_agent as ta
from inference.agent.runtime_state import Frame
from inference.agent.vision_context import ARC_COLOR_MAP, frame_to_png_data_url
from notebook_env import notebook_env
from test_budget_truth_loop import run_sim
from test_flags_off_identity import _snapshot

ROOT = Path(__file__).resolve().parents[1]
ARMS = json.loads((ROOT.parent / "kaggle" / "arms.json").read_text(encoding="utf-8"))["arms"]
BASE_ARM = "user-combo-10x"
FLAGS = {
    "ARC3_BUDGET_TRUTH": "1",
    "ARC3_PROMPT_HYGIENE": "1",
    "ARC3_IMAGE_GEOMETRY": "1",
    "ARC3_PREFER_TOOL_CALLS": "1",
}
FLAGS_OFF = {key: "0" for key in FLAGS}

GEOMETRY_LINE = (
    "- The board image is 640x640 px: 64x64 cells at 10 px per cell; cell (row r, col c) "
    "spans x 10c..10c+9, y 10r..10r+9.\n"
)
TOOL_SESSION_HEADER = "\n\nTool session rules:\n"
RATION_LINE = (
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm "
    "what changed between frames and whether the last action affected gameplay state or only "
    "HUD elements such as countdown bars.\n"
)
GAME_OVER_DEFAULT = prompts.GAME_OVER_RECORD_DEFAULT
GAME_OVER_LEDGER = prompts.GAME_OVER_RECORD_LEDGER


def arm_env(**extra: str) -> dict[str, str]:
    """The notebook's setup_env with user-combo-10x's overrides, as build_notebook applies them."""
    return {**notebook_env(), **{k: str(v) for k, v in ARMS[BASE_ARM]["env"].items()}, **extra}


CONFIGS = {
    "off": {},
    "hygiene": {"ARC3_PROMPT_HYGIENE": "1"},
    "geometry": {"ARC3_IMAGE_GEOMETRY": "1"},
    "prefer": {"ARC3_PREFER_TOOL_CALLS": "1"},
    "budget": {"ARC3_BUDGET_TRUTH": "1"},
    "all": FLAGS,
}


@pytest.fixture(scope="module")
def shown():
    return {name: _snapshot(arm_env(**flags)) for name, flags in CONFIGS.items()}


# -- the intended change of each flag, applied to the flags-off rendering ----

def _replace_once(prompt: str, old: str, new: str) -> str:
    assert prompt.count(old) == 1, old
    return prompt.replace(old, new)


def with_hygiene(prompt: str) -> str:
    # the ledger section leaves the Python guidance and sits right before the
    # tool-session rules
    prompt = _replace_once(prompt, prompts.LEDGER_ON_TRIM_ADDENDUM, "")
    prompt = _replace_once(prompt, TOOL_SESSION_HEADER, prompts.LEDGER_ON_TRIM_ADDENDUM + TOOL_SESSION_HEADER)
    for old, new in prompts.PROMPT_HYGIENE_REPLACEMENTS + prompts.PROMPT_HYGIENE_PERSISTENT_REPLACEMENTS:
        prompt = _replace_once(prompt, old, new)
    return prompt


def with_geometry(prompt: str) -> str:
    return _replace_once(prompt, prompts.MULTIMODAL_CONTEXT_ADDENDUM,
                         prompts.MULTIMODAL_CONTEXT_ADDENDUM + GEOMETRY_LINE)


def with_prefer(prompt: str) -> str:
    return _replace_once(prompt, RATION_LINE, RATION_LINE + prompts.PREFER_TOOL_CALLS_LINE)


def with_budget(prompt: str) -> str:
    for old, new in prompts.BUDGET_TRUTH_BAR_REPLACEMENTS:
        prompt = _replace_once(prompt, old, new)
    # a no-op after with_hygiene, which makes the same substitution
    for old, new in prompts.BUDGET_TRUTH_PERSISTENT_REPLACEMENTS:
        prompt = prompt.replace(old, new)
    return prompt


def opener_with_hygiene(message: dict) -> dict:
    """The game-over opener with the ledger's tags named (memory sections off, ledger on)."""
    texts = [part["text"] for part in message["content"] if part.get("type") == "text"]
    for old in GAME_OVER_DEFAULT:  # each sentence once in the whole message
        assert sum(text.count(old) for text in texts) == 1, old
    parts = []
    for part in message["content"]:
        if part.get("type") == "text":
            text = part["text"]
            for old, new in zip(GAME_OVER_DEFAULT, GAME_OVER_LEDGER):
                text = text.replace(old, new)
            part = {**part, "text": text}
        parts.append(part)
    return {**message, "content": parts}


# -- each flag alone ---------------------------------------------------------

def test_base_env_has_what_the_flags_depend_on():
    env = arm_env()
    assert not set(FLAGS) & set(env), "user-combo-10x already sets one of the flags"
    assert env["ARC3_LEDGER_ON_TRIM"] == "1" and env["ARC3_MEMORY_SECTIONS"] == "off"
    assert env["ARC3_PERSISTENT_FUNCTIONS"] == "1" and env["ARC3_PERSISTENT_FUNCTIONS_SCOPE"] == "game"
    assert env["MULTIMODAL_CONTEXT"] == "current_grid" and env["MULTIMODAL_UPSCALE"] == "10"


def test_hygiene_alone(shown):
    off, on = shown["off"], shown["hygiene"]
    assert on["system_prompt"] == with_hygiene(off["system_prompt"])
    assert on["openers"]["game_over"] == opener_with_hygiene(off["openers"]["game_over"])
    for name in ("animated_noop", "plain"):
        assert on["openers"][name] == off["openers"][name]
    assert on["sandbox"] == off["sandbox"]


def test_hygiene_section_structure(shown):
    off, on = shown["off"]["system_prompt"], shown["hygiene"]["system_prompt"]
    header = "Memory that survives history trimming:\n"
    # flags off: the search and frame_diff bullets sit under the ledger header
    ledger_section = off.split(header, 1)[1].split("\n\n\n", 1)[0]
    assert "- IMPORTANT: Especially when the game is about making an agent navigate" in ledger_section
    assert "`frame_diff(before=None, after=None)` compares two frames" in ledger_section
    # on: the ledger section holds the ledger bullets and nothing else
    ledger_section = on.split(header, 1)[1].split("\n\n\n", 1)[0]
    assert header + ledger_section + "\n" == prompts.LEDGER_ON_TRIM_ADDENDUM.lstrip("\n")
    python_section = on.split("Python tool guidance:\n", 1)[1].split("\n\n\n", 1)[0]
    assert "- IMPORTANT: Especially when the game is about making an agent navigate" in python_section
    assert "`frame_diff(before=None, after=None)` compares two frames" in python_section
    assert on.index(header) < on.index("Tool session rules:")
    # the bullets that ran together, the typos and the missing marker
    assert "1x1 tokens.- Some games" in off and "1x1 tokens.\n- Some games" in on
    assert "usually be rendered" not in on and "Sometime " not in on
    assert "usually rendered" in on and "Sometimes they might" in on
    assert "\n- A common failure mode is to mistake a segmented edge bar" in on
    assert "\n Alongside the current grid image" in off
    assert "\n- Alongside the current grid image you receive a diff image:" in on
    assert "ephemeral" in off and "ephemeral" not in on


def test_geometry_alone(shown):
    off, on = shown["off"], shown["geometry"]
    assert on["system_prompt"] == with_geometry(off["system_prompt"])
    assert on["openers"] == off["openers"] and on["sandbox"] == off["sandbox"]


def test_prefer_tool_calls_alone(shown):
    off, on = shown["off"], shown["prefer"]
    assert on["system_prompt"] == with_prefer(off["system_prompt"])
    assert on["openers"] == off["openers"] and on["sandbox"] == off["sandbox"]


def test_budget_truth_alone(shown):
    off, on = shown["off"], shown["budget"]
    assert on["system_prompt"] == with_budget(off["system_prompt"])
    assert "ephemeral" not in on["system_prompt"]
    assert on["openers"] == off["openers"] and on["sandbox"] == off["sandbox"]


def test_all_four_compose(shown):
    off, on = shown["off"], shown["all"]
    expected = with_budget(with_prefer(with_geometry(with_hygiene(off["system_prompt"]))))
    assert on["system_prompt"] == expected
    assert on["openers"]["game_over"] == opener_with_hygiene(off["openers"]["game_over"])
    for name in ("animated_noop", "plain"):
        assert on["openers"][name] == off["openers"][name]
    assert on["sandbox"] == off["sandbox"]


def test_flags_explicitly_off_match_unset(shown):
    assert _snapshot(arm_env(**FLAGS_OFF)) == shown["off"]


# -- the geometry line states what the renderer draws ------------------------

def test_geometry_line_matches_the_rendered_image(monkeypatch):
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "current_grid")
    monkeypatch.setenv("MULTIMODAL_UPSCALE", arm_env()["MULTIMODAL_UPSCALE"])
    monkeypatch.delenv("MULTIMODAL_RESAMPLE", raising=False)
    row, col = 5, 7
    grid = [[0] * prompts.BOARD_CELLS for _ in range(prompts.BOARD_CELLS)]
    grid[row][col] = 9
    url = frame_to_png_data_url(Frame(grid=tuple(map(tuple, grid)), step=0, level=1))
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    assert image.size == (640, 640)
    assert prompts.image_geometry_line(10) == GEOMETRY_LINE
    for x in range(10 * col, 10 * col + 10):  # x 10c..10c+9
        for y in range(10 * row, 10 * row + 10):  # y 10r..10r+9
            assert image.getpixel((x, y)) == ARC_COLOR_MAP[9]
    for x, y in ((10 * col - 1, 10 * row), (10 * col + 10, 10 * row),
                 (10 * col, 10 * row - 1), (10 * col, 10 * row + 10)):
        assert image.getpixel((x, y)) == ARC_COLOR_MAP[0]


def test_geometry_line_follows_the_configured_scale(monkeypatch):
    for key, value in arm_env(ARC3_IMAGE_GEOMETRY="1", MULTIMODAL_UPSCALE="4").items():
        monkeypatch.setenv(key, value)
    prompt = ta._build_system_prompt(tool_output_tokens=3072)
    assert prompts.image_geometry_line(4) in prompt
    assert "256x256 px: 64x64 cells at 4 px per cell" in prompt
    assert prompts.image_geometry_line(1) == (
        "- The board image is 64x64 px: 64x64 cells at 1 px per cell; cell (row r, col c) is "
        "the pixel at x c, y r.\n")
    monkeypatch.setenv("MULTIMODAL_CONTEXT", "")  # no image, no line
    assert "The board image is" not in ta._build_system_prompt(tool_output_tokens=3072)


# -- where the game-over opener says a diagnosis goes ------------------------

@pytest.mark.parametrize("hygiene,sections,ledger,expected", [
    ("0", "off", "1", GAME_OVER_DEFAULT),
    ("1", "off", "1", GAME_OVER_LEDGER),
    ("1", "off", "0", prompts.GAME_OVER_RECORD_REASONING),
    ("1", "full", "1", GAME_OVER_DEFAULT),  # the `World model:` section exists
    ("1", "", "0", GAME_OVER_DEFAULT),
])
def test_game_over_record_wording(monkeypatch, hygiene, sections, ledger, expected):
    monkeypatch.setenv("ARC3_PROMPT_HYGIENE", hygiene)
    monkeypatch.setenv("ARC3_MEMORY_SECTIONS", sections)
    monkeypatch.setenv("ARC3_LEDGER_ON_TRIM", ledger)
    assert ta._game_over_record_wording() == expected


def test_ledger_tags_named_in_the_opener_exist():
    """The opener may only name tags the system prompt explains and the ledger parses."""
    from inference.agent.memory_ledger import MemoryLedger

    for tag in ("<world_model>", '<hypothesis status="refuted">'):
        assert any(tag in line for line in GAME_OVER_LEDGER)
        assert tag in prompts.LEDGER_ON_TRIM_ADDENDUM
    ledger = MemoryLedger()
    ledger.absorb('<world_model>red cells kill</world_model> '
                  '<hypothesis status="refuted">the left route is safe</hypothesis>')
    rendered = ledger.render()
    assert "red cells kill" in rendered and "the left route is safe" in rendered


def test_hygiene_replacements_match_the_notebook_prompt_once(monkeypatch):
    """Drift guard without the ledger too: a reworded source line would make a fix inert."""
    for key, value in notebook_env().items():
        monkeypatch.setenv(key, value)
    for attr, key in (("_DIFF_IMAGE", "ARC3_DIFF_IMAGE"), ("_AUTO_FRAME_DIFF", "ARC3_AUTO_FRAME_DIFF")):
        monkeypatch.setattr(ta, attr, ta._get_env_bool(key, False))  # read at import time
    prompt = ta._build_system_prompt(tool_output_tokens=3072)
    for old, _ in prompts.PROMPT_HYGIENE_REPLACEMENTS + prompts.PROMPT_HYGIENE_PERSISTENT_REPLACEMENTS:
        assert prompt.count(old) == 1, old
    assert prompts.PROMPT_HYGIENE_PERSISTENT_REPLACEMENTS == prompts.BUDGET_TRUTH_PERSISTENT_REPLACEMENTS
    monkeypatch.setenv("ARC3_PROMPT_HYGIENE", "1")
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "0")
    on = ta._build_system_prompt(tool_output_tokens=3072)
    assert "call it with one ephemeral `code` string" in on  # true without retention
    assert "Memory that survives history trimming" not in on  # the ledger is off here


def test_hygiene_alone_drops_ephemeral_from_the_tool_schema(monkeypatch):
    """With retained functions, "ephemeral" is false in the tool schema as well as the system line."""
    for key, value in notebook_env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("ARC3_BUDGET_TRUTH", raising=False)
    agent = ta.ToolAgent.__new__(ta.ToolAgent)
    agent._ensure_session = lambda state_path: None
    off = agent._tools(None)[0]["function"]
    assert off["description"] == ta._PYTHON_TOOL_DESCRIPTION
    assert off["parameters"]["properties"]["code"]["description"] == ta._CODE_PARAMETER_DESCRIPTION
    monkeypatch.setenv("ARC3_PROMPT_HYGIENE", "1")
    on = agent._tools(None)[0]["function"]
    assert on["description"] == ta._PYTHON_TOOL_DESCRIPTION.replace(
        ta._EPHEMERAL_TOOL_DESCRIPTION_OLD, ta._EPHEMERAL_TOOL_DESCRIPTION_NEW)
    assert on["parameters"]["properties"]["code"]["description"] == "Python code to run."
    monkeypatch.setenv("ARC3_PERSISTENT_FUNCTIONS", "0")
    assert agent._tools(None)[0]["function"] == off  # true without retention


# -- through the real analyze() loop -----------------------------------------

@pytest.fixture(scope="module")
def loop(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prompt-facts")
    base = {k: str(v) for k, v in ARMS[BASE_ARM]["env"].items()}
    return {"off": run_sim(tmp, "off", base), "on": run_sim(tmp, "on", {**base, **FLAGS})}


def test_loop_plays_through(loop):
    for run in loop.values():
        assert run["won"] and run["actions"] == 7
        assert run["requests_per_turn"] == [2, 2, 2, 1, 2]


def test_loop_system_prompt_is_static_and_is_the_rendered_one(loop, shown):
    for config, name in (("off", "off"), ("on", "all")):
        requests = loop[config]["requests"]
        assert {r["system"] for r in requests} == {shown[name]["system_prompt"]}
        assert len({json.dumps(r["tools"], sort_keys=True) for r in requests}) == 1


def test_loop_each_prompt_extends_the_previous_one(loop):
    for config in ("on", "off"):
        requests = loop[config]["requests"]
        for before, after in zip(requests, requests[1:]):
            assert after["prompt"].startswith(before["prompt"]), (config, before["turn"], before["index"])


def test_loop_turn_messages_change_only_by_the_clock(loop):
    """User turns are identical; tool results lose only the run clock (ARC3_BUDGET_TRUTH)."""
    clock = re.compile(r", '(?:run_elapsed_seconds|time_remaining_seconds)': [0-9.]+"
                       r"|'(?:run_elapsed_seconds|time_remaining_seconds)', ")
    for off, on in zip(loop["off"]["requests"], loop["on"]["requests"]):
        assert on["user_texts"] == off["user_texts"]
        assert [clock.sub("", t) for t in off["tool_texts"]] == on["tool_texts"]
    assert any("time_remaining_seconds" in t for t in loop["off"]["requests"][-1]["tool_texts"])


def test_loop_tools_drop_only_ephemeral(loop):
    off = loop["off"]["requests"][0]["tools"][0]["function"]
    on = loop["on"]["requests"][0]["tools"][0]["function"]
    assert on["description"] == off["description"].replace(
        ta._EPHEMERAL_TOOL_DESCRIPTION_OLD, ta._EPHEMERAL_TOOL_DESCRIPTION_NEW)
    assert on["parameters"]["properties"]["code"]["description"] == "Python code to run."
    assert {k: v for k, v in on.items() if k not in ("description", "parameters")} == {
        k: v for k, v in off.items() if k not in ("description", "parameters")}


def test_loop_flags_off_matches_baseline_checkout(loop, tmp_path):
    baseline = os.environ.get("BUDGET_SIM_BASELINE_ROOT", "")
    if not baseline:
        pytest.skip("set BUDGET_SIM_BASELINE_ROOT to compare with another checkout")
    base_env = {k: str(v) for k, v in ARMS[BASE_ARM]["env"].items()}
    base = run_sim(tmp_path, "baseline", base_env, root=Path(baseline))
    strip = lambda run: [{k: r[k] for k in ("prompt", "system", "tools", "last_message")}
                         for r in run["requests"]]
    assert strip(base) == strip(loop["off"])
