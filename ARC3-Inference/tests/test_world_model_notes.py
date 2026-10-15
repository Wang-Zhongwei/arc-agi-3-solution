"""ARC3_WORLD_MODEL_NOTES: `<world_model>` updates kept per game in world_model.md."""

import pytest

from inference.agent import tool_agent
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.world_model_notes import extract_world_model, render_notes
from inference.utils.grid_utils import format_grid_ascii


def test_last_complete_block_wins_and_reply_beats_thinking():
    thinking = "<world_model>\ndraft\n</world_model>\nno wait\n<world_model>\nrevised\n</world_model>"
    assert extract_world_model(thinking, "") == "revised"
    assert extract_world_model(thinking, "<world_model>\nfrom reply\n</world_model>") == "from reply"


def test_inline_block_counts():
    assert extract_world_model("so: <world_model>ACTION1 moves up</world_model> ok", "") == "ACTION1 moves up"


@pytest.mark.parametrize("text", [
    "<world_model>\ncut off by the output cap",  # unclosed
    "I should update my <world_model> later",  # mention without a block
    "use `<world_model>` and `</world_model>` tags",  # backticked mention
    "<world_model>\n   \n</world_model>",  # empty
    "<tool_call>\n<function=python>\n<parameter=code>\n<world_model>\nx\n</world_model>\n"
    "</parameter>\n</function>\n</tool_call>",  # inside a tool call
])
def test_non_blocks_are_ignored(text):
    assert extract_world_model(text, "") == ""


def test_a_mention_or_unclosed_draft_before_a_block_is_dropped():
    assert extract_world_model("my <world_model> so far\n<world_model>\nkept\n</world_model>", "") == "kept"


def test_every_line_is_tagged():
    assert render_notes([(1, 3, "a\n\n- b"), (2, 57, "c")]) == "[L1 s3] a\n[L1 s3] - b\n[L2 s57] c\n"
    assert render_notes([]) == ""


@pytest.mark.parametrize("enabled", [True, False])
def test_system_prompt_section_follows_flag(monkeypatch, enabled):
    monkeypatch.setenv("ARC3_WORLD_MODEL_NOTES", "1" if enabled else "0")
    prompt = tool_agent._build_system_prompt(tool_output_tokens=2000)
    assert ("<world_model>" in prompt) is enabled
    assert ("world_model.md" in prompt) is enabled


def _set_frame(state_path, *, level, step):
    frame = Frame(grid=((0, 0), (0, 9)), step=step, level=level)
    write_runtime_state(state_path, current_frame=frame, history=[HistoryEntry("", frame)], flush=True)


def _agent(tmp_path, monkeypatch, state_name="ar25-0c556536_p0_tool_runtime_state.json"):
    monkeypatch.setenv("ARC3_WORLD_MODEL_NOTES", "1")
    agent = tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:8001/v1", provider="vllm")
    state_path = tmp_path / state_name
    agent._ensure_session(state_path)
    return agent, state_path


def test_updates_are_tagged_deduplicated_and_written_per_game(tmp_path, monkeypatch):
    agent, state_path = _agent(tmp_path, monkeypatch)
    _set_frame(state_path, level=1, step=1)
    agent._record_world_model_note(state_path, "<world_model>\nACTION1 moves the block up\n</world_model>", "")
    agent._record_world_model_note(state_path, "no block this time", "")
    _set_frame(state_path, level=2, step=57)
    agent._record_world_model_note(state_path, "", "<world_model>\nACTION1 moves the block up\n</world_model>")
    agent._record_world_model_note(state_path, "", "<world_model>\nwalls push back\ngoal: fill the frame\n</world_model>")
    expected = "[L1 s1] ACTION1 moves the block up\n[L2 s57] walls push back\n[L2 s57] goal: fill the frame\n"
    assert render_notes(agent._world_model_notes) == expected
    assert (tmp_path / "ar25-0c556536_p0_world_model.md").read_text() == expected


def test_updates_reset_on_a_new_game(tmp_path, monkeypatch):
    agent, state_path = _agent(tmp_path, monkeypatch)
    _set_frame(state_path, level=1, step=3)
    agent._record_world_model_note(state_path, "<world_model>\nx\n</world_model>", "")
    assert agent._world_model_notes
    other = tmp_path / "next-game" / "state.json"
    other.parent.mkdir()
    agent._ensure_session(other)
    assert agent._world_model_notes == []


def _run_sandbox(code, **state):
    grid = [[0, 0], [0, 9]]
    frame = {"ascii": format_grid_ascii(grid), "step": 0, "level": 1, "shape": [2, 2], "grid": grid}
    return run_sandboxed_python(
        code=code, timeout_seconds=20,
        initial_state={"current_frame": frame, "history": [{"action": "", "frame": frame}], **state},
        action_handler=lambda *a, **k: {},
    )


NOTES = "[L1 s3] ACTION3 rotates the piece\n[L2 s57] walls push back\n[L2 s60] ACTION3 only near walls\n"


@pytest.mark.parametrize("code, expected", [
    ("result = open('world_model.md').read()", NOTES),
    ("result = [l.rstrip() for l in open('world_model.md') if l.startswith('[L2 ')]",
     ["[L2 s57] walls push back", "[L2 s60] ACTION3 only near walls"]),
    ("import re\nresult = re.findall(r'^.*ACTION3.*$', open('world_model.md').read(), re.M)",
     ["[L1 s3] ACTION3 rotates the piece", "[L2 s60] ACTION3 only near walls"]),
    ("with open('./world_model.md') as f:\n    result = len(f.readlines())", 3),
])
def test_sandbox_reads_and_filters_the_file(code, expected):
    out = _run_sandbox(code, world_model_md=NOTES)
    assert out.get("result") == expected, out


def test_other_paths_are_in_memory_scratch_not_the_real_disk():
    out = _run_sandbox("open('/etc/passwd')", world_model_md=NOTES)
    assert "No such file or directory" in str(out.get("error") or out), out


@pytest.mark.parametrize("mode", ["a", "w"])
def test_sandbox_writes_append_tagged_and_never_truncate(mode):
    code = (
        f"open('world_model.md', '{mode}').write('ACTION4 moves right\\n')\n"
        f"with open('world_model.md', '{mode}') as f:\n"
        "    print('goal: reach x=2', file=f)\n"
        "result = open('world_model.md').read()\n"
    )
    out = _run_sandbox(code, world_model_md=NOTES)
    assert out.get("result") == NOTES + "[L1 s0] ACTION4 moves right\n[L1 s0] goal: reach x=2\n", out
    texts = [text for _, _, text in out["world_model_appends"]]
    assert texts == ["ACTION4 moves right", "goal: reach x=2"], out


def test_writes_before_an_error_still_reach_the_host():
    out = _run_sandbox("open('world_model.md', 'a').write('kept')\n1/0", world_model_md=NOTES)
    assert "ZeroDivisionError" in out["error"] and out["world_model_appends"] == [[1, 0, "kept"]], out


def test_agent_logs_python_writes_tagged_where_they_ran(tmp_path, monkeypatch):
    import json

    agent, state_path = _agent(tmp_path, monkeypatch)
    _set_frame(state_path, level=2, step=10)

    def step_env(arguments):
        _set_frame(state_path, level=2, step=11)
        return {"executed": True, "executed_count": 1, "action_display": "RIGHT", "action_name": "ACTION4",
                "valid_actions": ["RIGHT"], "gameplay_changed": True, "board_changed": True,
                "level": 2, "state": "NOT_FINISHED"}

    agent._current_valid_actions = ["RIGHT"]
    agent._step_env_callback = step_env
    code = (
        "open('world_model.md', 'a').write('before the move')\n"
        "action(['RIGHT'])\n"
        "open('world_model.md', 'a').write('after the move')\n"
    )
    agent._run_python_tool(state_path, {"code": code})
    expected = "[L2 s10] before the move\n[L2 s11] after the move\n"
    assert render_notes(agent._world_model_notes) == expected
    assert (tmp_path / "ar25-0c556536_p0_world_model.md").read_text() == expected
    # the next snippet reads what the last one wrote
    result = agent._run_python_tool(state_path, {"code": "result = open('world_model.md').read()"})
    assert json.loads(result.content)["result"] == expected


def test_sandbox_has_no_open_when_off():
    out = _run_sandbox("open('world_model.md')")
    assert "open" in str(out.get("error") or out), out


# Every open() pattern the model used in ticbudja/world-model-notes v1, which
# had a read-only world_model.md and nothing else: each must now just work.
V1_PATTERNS = [
    # reads, slices, guarded reads
    ("txt=open('world_model.md').read()\nresult = len(txt)", len(NOTES)),
    ("try: print(open('world_model.md').read()[-1500:])\nexcept Exception as e: print(e)\nresult = 1", 1),
    ("lines=[l for l in open('world_model.md').read().splitlines() if l]\nresult = len(lines)", 3),
    # appends carrying the model's own tags, multi-line
    ("open('world_model.md','a').write(\"\"\"[L1-5] Mechanics: SPACE grabs the block\n[L2] walls stop it\n\"\"\")\n"
     "result = open('world_model.md').read().count('\\n')", 5),
    # empty appends
    ("open('world_model.md','a').write('')\nresult = open('world_model.md').read()", NOTES),
    # read, edit, write the whole file back: only the new line is added
    ("txt=open('world_model.md').read()\ntxt += 'L1-2: corrected rule\\n'\n"
     "open('world_model.md','w').write(txt)\nresult = open('world_model.md').read().count('\\n')", 4),
    # /dev/null as a sink
    ("f=open('/dev/null','w')\nprint('noise', file=f)\nresult = open('/dev/null').read()", ""),
    ("open('/dev/null','w')\nresult = 2", 2),
    # scratch files, one-liners and json.dump through a context manager
    ("import json\nopen('/tmp/plan.json','w').write(json.dumps({'plan':[1,2]}))\n"
     "result = json.load(open('/tmp/plan.json'))", {"plan": [1, 2]}),
    ("import json\nwith open('/tmp/masks.json','w') as f:\n    json.dump({'water':[[1,2]]}, f)\n"
     "result = json.loads(open('/tmp/masks.json').read())", {"water": [[1, 2]]}),
    ("open('wmap.txt','w').write('')\nresult = open('wmap.txt').read()", ""),
]


@pytest.mark.parametrize("code, expected", V1_PATTERNS)
def test_v1_open_patterns_work(code, expected):
    out = _run_sandbox(code, world_model_md=NOTES, scratch_files={})
    assert not out.get("error") and out.get("result") == expected, out


def test_written_back_file_adds_only_new_lines():
    code = ("txt=open('world_model.md').read()\n"
            "open('world_model.md','w').write(txt + 'new rule\\n')\n")
    out = _run_sandbox(code, world_model_md=NOTES, scratch_files={})
    assert [text for _, _, text in out["world_model_appends"]] == ["new rule"], out


def test_missing_scratch_file_and_binary_mode_raise_normal_errors():
    out = _run_sandbox("open('/tmp/none.json')", world_model_md=NOTES, scratch_files={})
    assert "FileNotFoundError" in out["error"] and "No such file or directory" in out["error"], out
    out = _run_sandbox("open('/tmp/x.bin','wb')", world_model_md=NOTES, scratch_files={})
    assert "binary mode is not supported" in out["error"], out


def test_scratch_size_is_capped():
    out = _run_sandbox("open('/tmp/big','w').write('x' * 1_000_001)", world_model_md=NOTES, scratch_files={})
    assert "No space left on device" in out["error"], out


def test_scratch_files_persist_across_tool_calls_and_reset_per_game(tmp_path, monkeypatch):
    import json

    agent, state_path = _agent(tmp_path, monkeypatch)
    _set_frame(state_path, level=1, step=2)
    agent._run_python_tool(state_path, {"code": "import json\nopen('/tmp/plan.json','w').write(json.dumps([3, 4]))"})
    result = agent._run_python_tool(state_path, {"code": "import json\nresult = json.load(open('/tmp/plan.json'))"})
    assert json.loads(result.content)["result"] == [3, 4]
    other = tmp_path / "next-game" / "state.json"
    other.parent.mkdir()
    agent._ensure_session(other)
    assert agent._scratch_files == {}



def test_prompt_points_at_re_for_filtering():
    from inference.agent.prompts import WORLD_MODEL_NOTES_ADDENDUM

    assert "with `re`." in WORLD_MODEL_NOTES_ADDENDUM
    assert "open('world_model.md', 'a').write(...)" in WORLD_MODEL_NOTES_ADDENDUM
