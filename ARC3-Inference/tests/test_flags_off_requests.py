"""With the throughput-stack flags off, every request body matches arm/stack-v2's (d55df75).

tests/request_snapshot.py runs one scripted analyzer turn with HTTP stubbed:
reasoning and content with blank lines before a tool call, a reasoning-only
reply cut off at the output limit, a content-only reply, and tool-call
arguments with blank lines. It runs in a fresh process because tool_agent reads
settings at import time. EXPECTED holds the hashes it printed against the
arm/stack-v2 tree under the same environments; a deliberate change to the
flags-off requests updates them the same way.

The merged branches change the harness only behind ARC3_VERBATIM_HISTORY and
ARC3_PREFIX_REUSE_DIAG (fix-3); gate11 and stub-dead-reasoning only add arms.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name("request_snapshot.py")
EXPECTED = {
    "defaults": "49abde9aedf67b797f8dfca276a13bf224a31b96ef0648e098688818f5446349",
    "kaggle": "cd616ac38775963b0eeb0f5eaf53d5fe7cd20b5821b7c95c4e21538ee82a2851",
    "stack-v2": "4ae2f633d65fd8ff9466849a0af654a866d3f3b6ea2924354ed0e8723ebe4097",
}


def _snapshot(name, out=None):
    command = [sys.executable, str(HELPER), name] + ([str(out)] if out else [])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    count, digest, module = result.stdout.strip().splitlines()[-1].split(" ", 2)
    assert Path(module).resolve().is_relative_to(ROOT), f"imported the harness from {module}"
    return int(count), digest


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_flags_off_requests_match_stack_v2(name):
    count, digest = _snapshot(name)
    assert count == 6
    assert digest == EXPECTED[name], (
        "the flags-off requests differ from arm/stack-v2's; if that is deliberate, "
        "recompute EXPECTED with tests/request_snapshot.py against the base tree"
    )


def test_tput_safe_changes_only_the_stored_replies(tmp_path):
    """The tput-safe arm's flags change history text and stub a dead reply; nothing else."""
    _snapshot("stack-v2", tmp_path / "off.json")
    _snapshot("tput-safe", tmp_path / "on.json")
    off = json.loads((tmp_path / "off.json").read_text())
    on = json.loads((tmp_path / "on.json").read_text())
    assert len(on) == len(off) == 6
    for before, after in zip(off, on):
        assert {k: v for k, v in before.items() if k != "messages"} == {
            k: v for k, v in after.items() if k != "messages"}
        assert [m for m in before["messages"] if m["role"] != "assistant"] == [
            m for m in after["messages"] if m["role"] != "assistant"]
    replies_off = [m for m in off[-1]["messages"] if m["role"] == "assistant"]
    replies_on = [m for m in on[-1]["messages"] if m["role"] == "assistant"]
    assert len(replies_on) == len(replies_off) == 5
    # blank lines survive in history (ARC3_VERBATIM_HISTORY)
    assert "\n\n" not in replies_off[0]["reasoning_content"]
    assert replies_on[0]["reasoning_content"] == (
        "I look at the board first.\n\nThe player is at (3, 4).\n   \nSo I probe UP.")
    assert replies_on[0]["content"] == "Plan:\n\n1. probe\n\n2. act"
    # the cut-off reasoning-only reply is replaced by a stub (ARC3_STUB_DEAD_REASONING)
    assert "reasoning_content" in replies_off[1]
    assert set(replies_on[1]) == {"role", "content"} and "reasoning_content" not in replies_on[1]
    # tool-call arguments were already stored as sent
    assert [m.get("tool_calls") for m in replies_on] == [m.get("tool_calls") for m in replies_off]


@pytest.mark.parametrize("base", ["kaggle", "stack-v2"])
def test_prompt_facts_flags_off_send_the_same_requests(base):
    """ARC3_BUDGET_TRUTH, ARC3_PROMPT_HYGIENE, ARC3_IMAGE_GEOMETRY and
    ARC3_PREFER_TOOL_CALLS set to "0" send exactly the unset requests; set to
    "1" they do not (so the check can fail)."""
    count, digest = _snapshot(f"{base}-facts-off")
    assert (count, digest) == (6, EXPECTED[base])
    count, digest = _snapshot(f"{base}-facts-on")
    assert count == 6 and digest != EXPECTED[base]
