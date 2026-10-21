"""Hash every request body one scripted analyzer turn sends, under a named environment.

Used by test_flags_off_requests.py, which runs it in a fresh process because
tool_agent reads several settings (temperature, top_p, ...) at import time.
Run from ARC3-Inference:

    python tests/request_snapshot.py <defaults|kaggle|stack-v2|tput-safe> [out.json]

It prints the number of requests, the sha256 of their bodies and the path of
the tool_agent module it imported; with ``out.json`` it also writes the bodies. The scripted replies cover what the throughput flags touch: reasoning and
content with blank lines before a tool call, a reasoning-only reply cut off at
the output limit, a content-only reply, and tool-call arguments with blank
lines. HTTP is stubbed at ``requests.post``, so each body is exactly what would
reach SGLang; images are replaced by a marker.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

CLEARED = ("ARC3_", "MULTIMODAL_", "EXPOSE_", "LOCAL_ANALYZER_")
# The model-facing analyzer settings of kaggle/template.ipynb (setup_env, with
# priority scheduling on). Server URLs and model names are left at the test's.
KAGGLE_ENV = {
    "ARC3_REASONING_HISTORY_KEY": "reasoning_content", "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30", "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_YIELD_TOKENS": "2048", "LOCAL_ANALYZER_TEMPERATURE": "0.7",
    "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str((116 + 12) * 1024), "LOCAL_ANALYZER_MAX_OUTPUT": str(12 * 1024),
    "ARC3_CONTEXT_DRAIN_TOKENS": str(58 * 1024), "ARC3_HISTORY_ASSISTANT_TURNS": "150",
    "ARC3_HISTORY_TURN_DRAIN": "30", "ARC3_HISTORY_DRAIN_COALESCE": "1",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072", "ARC3_MIDDLE_TRUNCATION": "1", "ARC3_ACTION_ECHO": "1",
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1", "ARC3_NEW_CHANGED_PROMPTS": "1", "ARC3_LEVEL_TRANSFER_GUIDANCE": "1",
    "ARC3_ACTION_INFO": "1", "ARC3_DEDUPE_MULTICALL_LINE": "1", "ARC3_FRAME_DIFF_HINT": "1",
    "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "10", "ARC3_DIFF_IMAGE": "1",
    "ARC3_GAMEOVER_DIFF_IMAGE": "1", "ARC3_ANIMATION": "1", "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_BATCH_NOOP_BLOCK": "1", "ARC3_STALE_STATE_BLOCK": "1", "ARC3_MEMORY_SECTIONS": "off",
    "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
    "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1", "EXPOSE_UNDO": "on",
    "ARC3_MAX_ACTIVE_STREAMS": "10", "ARC3_PRIORITY_REFRESH_QUEUE": "1", "ARC3_PRIORITY_PACE": "0",
    "ARC3_PRIORITY_TAIL_FADE": "1", "ARC3_PRIORITY_TAIL_FADE_FRACTION": "0.2",
    "ARC3_PRIORITY_TAIL_LOOKUP": "remaining", "ARC3_PRIORITY_SCORE_NORMALIZATION": "1",
}
ENVS = {
    "defaults": {},
    "kaggle": KAGGLE_ENV,
    # the stack-v2 arm (kaggle/arms.json)
    "stack-v2": {**KAGGLE_ENV, "ARC3_LEDGER_ON_TRIM": "1"},
}
# the tput-safe arm: stack-v2 plus the throughput flags
ENVS["tput-safe"] = {**ENVS["stack-v2"], "ARC3_MAX_ACTIVE_STREAMS": "11", "ARC3_STUB_DEAD_REASONING": "1",
                     "ARC3_VERBATIM_HISTORY": "1", "ARC3_PREFIX_REUSE_DIAG": "1"}
# the prompt-facts flags (the prompt-facts-10x arm) set to "0", which must
# send the same requests as leaving them unset, and set to "1"
PROMPT_FACTS_FLAGS = ("ARC3_BUDGET_TRUTH", "ARC3_PROMPT_HYGIENE", "ARC3_IMAGE_GEOMETRY", "ARC3_PREFER_TOOL_CALLS")
for _base in ("kaggle", "stack-v2"):
    ENVS[f"{_base}-facts-off"] = {**ENVS[_base], **{flag: "0" for flag in PROMPT_FACTS_FLAGS}}
    ENVS[f"{_base}-facts-on"] = {**ENVS[_base], **{flag: "1" for flag in PROMPT_FACTS_FLAGS}}

REASONING = "\nI look at the board first.\n\nThe player is at (3, 4).\n   \nSo I probe UP.\n"
CONTENT = "Plan:\n\n1. probe\n\n2. act\n"


def _scrub(value):
    if isinstance(value, dict):
        return {k: ("<image>" if k == "url" and str(v).startswith("data:") else _scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def snapshot(name: str) -> list[dict]:
    """The request bodies of one scripted analyzer turn under ENVS[name].

    Sets the environment before importing the harness, so call it once per process.
    """
    for key in list(os.environ):
        if key.startswith(CLEARED):
            del os.environ[key]
    os.environ.update(ENVS[name])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from inference.agent import tool_agent
    from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state

    def reply(reasoning, content="", code=None, finish=None, completion=50):
        message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
        if code is not None:
            message["tool_calls"] = [{
                "id": "call_" + hashlib.sha1(code.encode()).hexdigest()[:6], "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": code})},
            }]
        return {"choices": [{"message": message, "finish_reason": finish or ("tool_calls" if code else "stop")}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": completion}}

    script = [
        reply(REASONING, CONTENT, code="print(current_frame.step)\n\n"),
        reply(REASONING + "\n\nStill thinking about the walls", "", finish="length", completion=300),
        reply("Short.\n\nThen act.", "", code="x = 1\n\nprint(x)"),
        reply("", "I will not act yet.\n\nWaiting."),
        reply(REASONING, "", code="print('again')"),
    ]
    sent: list[dict] = []

    class Response:
        status_code = 200
        text = ""

        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def fake_post(url, **kwargs):
        sent.append(_scrub(json.loads(json.dumps(kwargs["json"], default=str))))
        if len(sent) > 20:
            raise RuntimeError("the scripted turn did not end")
        return Response(script.pop(0) if script else reply("Done.", "", code="print('fallback')"))

    tool_agent.requests.post = fake_post
    tool_agent.run_sandboxed_python = lambda **kwargs: {
        "stdout": "ok\n\n", "result": None, "error": "", "action_results": [], "keepable_functions": []}
    state = Path(tempfile.mkdtemp()) / "artifacts" / "g_p0_tool_runtime_state.json"
    frame = Frame(grid=tuple(tuple((r * 7 + c) % 4 for c in range(16)) for r in range(16)), step=0, level=1)
    write_runtime_state(state, current_frame=frame, history=[HistoryEntry(action="", frame=frame)])
    agent = tool_agent.ToolAgent(model="test", base_url="http://127.0.0.1:9/v1", provider="vllm")
    agent._tool_steps = 6
    agent.analyze(state, action_num=1, valid_actions=["UP", "DOWN"], step_env=lambda arguments: {})
    return sent


def digest(requests: list[dict]) -> str:
    return hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    bodies = snapshot(sys.argv[1])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(bodies, indent=1, sort_keys=True))
    # the harness module actually used, so a caller can check it came from this tree
    print(f"{len(bodies)} {digest(bodies)} {sys.modules['inference.agent.tool_agent'].__file__}")
