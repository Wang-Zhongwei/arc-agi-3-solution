"""ARC3_TOKEN_EXACT_HISTORY: replies go back as their sampled ids, through the harness and the server splice."""
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from inference.agent import tool_agent
from inference.agent import token_history as history
from inference.framework import lordhansolo
from inference.framework.lordhansolo import kaggle as lord
from inference.utils.openai_compat import assemble_streamed_chat_response

TOKEN_EXACT = Path(lordhansolo.__file__).resolve().parent / "token_exact"
_spec = importlib.util.spec_from_file_location("arc3_token_splice", TOKEN_EXACT / "arc3_token_splice.py")
splice = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(splice)

THINK_END, TURN_END, IMAGE_PAD = 900, 901, 902


class Tokenizer:
    """One token per character, with the specials the splice looks up."""

    SPECIAL = {"</think>": THINK_END, "<|im_end|>": TURN_END, "<|image_pad|>": IMAGE_PAD,
               "<|video_pad|>": 903, "<|audio_pad|>": 904}

    def convert_tokens_to_ids(self, token):
        return self.SPECIAL[token]

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        out, i = [], 0
        while i < len(text):
            special = next((s for s in self.SPECIAL if text.startswith(s, i)), None)
            out.append(self.SPECIAL[special] if special else ord(text[i]))
            i += len(special) if special else 1
        return out


def marked(ids):
    return f"{splice.MARKER_OPEN}{','.join(map(str, ids))}{splice.MARKER_CLOSE}\n</think>\n\n<|im_end|>"


def test_markers_match_between_harness_and_server():
    assert (history.MARKER_OPEN, history.MARKER_CLOSE) == (splice.MARKER_OPEN, splice.MARKER_CLOSE)


def test_prompt_without_marker_is_untouched():
    assert splice.splice_reply_tokens("plain", Tokenizer()) == "plain"
    assert splice.splice_reply_tokens([1, 2], Tokenizer()) == [1, 2]


def test_ids_replace_the_rendered_reply_and_the_text_around_it_tokenizes():
    tok = Tokenizer()
    prompt = "a" + marked([7, 8, THINK_END, 9, TURN_END]) + "\nb"
    assert splice.splice_reply_tokens(prompt, tok) == [ord("a"), 7, 8, THINK_END, 9, TURN_END, ord("\n"), ord("b")]


def test_reply_cut_off_in_thinking_is_closed_like_the_template():
    tok = Tokenizer()
    assert splice.splice_reply_tokens(marked([7, 8]), tok) == [7, 8, 10, THINK_END, 10, 10, TURN_END]


def test_reply_without_a_turn_end_gets_one():
    assert splice.splice_reply_tokens(marked([THINK_END, 9]), Tokenizer()) == [THINK_END, 9, TURN_END]


def test_sampled_image_token_is_dropped():
    assert splice.splice_reply_tokens(marked([IMAGE_PAD, THINK_END, TURN_END]), Tokenizer()) == [THINK_END, TURN_END]


def test_marker_that_did_not_render_as_expected_fails_rather_than_reach_the_model():
    with pytest.raises(ValueError):
        splice.splice_reply_tokens(f"{splice.MARKER_OPEN}1,2{splice.MARKER_CLOSE}", Tokenizer())


def reply(tokens="1,2,3"):
    message = {"role": "assistant", "reasoning_content": "think", "content": "", "tool_calls": [{"id": "x"}],
               history.REPLY_TOKENS_KEY: tokens}
    history.seal_reply_tokens(message)
    return message


def test_unchanged_reply_goes_out_as_its_ids_only():
    wire, last, stale = history.render_reply_tokens([{"role": "user", "content": "u"}, reply()], ["reasoning_content"])
    assert wire[1] == {"role": "assistant", "content": "",
                       "reasoning_content": f"{history.MARKER_OPEN}1,2,3{history.MARKER_CLOSE}"}
    assert (last, stale) == ("1,2,3", 0)


@pytest.mark.parametrize("field,value", [("reasoning_content", "edited"), ("content", "edited"), ("tool_calls", [])])
def test_edited_reply_goes_out_as_text(field, value):
    message = reply()
    message[field] = value
    wire, last, stale = history.render_reply_tokens([message], ["reasoning_content"])
    assert not set(history.REPLY_KEYS) & set(wire[0])
    assert wire[0][field] == value
    assert (last, stale) == ("", 1)


def test_prompt_holds_reply_needs_the_ids_in_order_and_adjacent():
    assert history.prompt_holds_reply([5, 1, 2, 3, 6], "1,2,3")
    assert not history.prompt_holds_reply([5, 1, 3, 2, 6], "1,2,3")
    assert not history.prompt_holds_reply([1, 2, 9, 3], "1,2,3")


def test_stream_assembly_keeps_the_ids():
    chunks = [
        {"prompt_token_ids": [4, 5], "choices": [{"delta": {"content": "a"}, "token_ids": [1]}]},
        {"choices": [{"delta": {"content": "b"}, "token_ids": [2], "finish_reason": "stop"}]},
    ]
    out = assemble_streamed_chat_response(f"data: {json.dumps(c)}" for c in chunks)
    assert out["prompt_token_ids"] == [4, 5]
    assert out["choices"][0]["token_ids"] == [1, 2]


class Response:
    status_code = 200
    text = ""

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("ARC3_TOKEN_EXACT_HISTORY", "1")
    monkeypatch.setenv("ARC3_REASONING_HISTORY_KEY", "reasoning_content")
    monkeypatch.setattr(tool_agent, "_LOCAL_ANALYZER_ENABLE_THINKING", True)
    return tool_agent.ToolAgent(model="test-model", base_url="http://127.0.0.1:1/v1")


def serve(monkeypatch, bodies):
    sent = []

    def post(url, json=None, **kwargs):
        sent.append(json)
        return Response(bodies.pop(0))

    monkeypatch.setattr(tool_agent.requests, "post", post)
    return sent


def body(prompt, ids):
    return {"prompt_token_ids": prompt, "choices": [
        {"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop", "token_ids": ids}]}


def test_request_asks_for_ids_and_returns_the_reply_ids(monkeypatch, agent):
    sent = serve(monkeypatch, [body([4], [1, 2])])
    result = agent._chat_completion([{"role": "user", "content": "u"}], tools=None)
    assert sent[0]["return_token_ids"] is True
    assert result.reply_token_ids == [1, 2]


def test_thinking_off_reply_is_not_kept_as_ids(monkeypatch, agent):
    serve(monkeypatch, [body([4], [1, 2])])
    result = agent._chat_completion([{"role": "user", "content": "u"}], tools=None, enable_thinking=False)
    assert result.reply_token_ids is None


def test_estimate_ignores_stored_ids(agent):
    plain = reply()
    stored = dict(plain, **{history.REPLY_TOKENS_KEY: ",".join(["123456"] * 5000)})
    without = {k: v for k, v in plain.items() if k not in history.REPLY_KEYS}
    assert agent._estimate_request_input_tokens([stored]) == agent._estimate_request_input_tokens([without])


def test_server_without_the_splice_switches_the_game_back_to_text(monkeypatch, agent):
    message = reply()
    sent = serve(monkeypatch, [body([70, 71], [8]), body([70, 71], [9])])
    first = agent._chat_completion([{"role": "user", "content": "u"}, message], tools=None)
    assert first.reply_token_ids is None
    agent._chat_completion([{"role": "user", "content": "u"}, message], tools=None)
    assert history.MARKER_OPEN in json.dumps(sent[0], ensure_ascii=False)
    assert history.MARKER_OPEN not in json.dumps(sent[1], ensure_ascii=False)
    assert sent[1]["messages"][1]["reasoning_content"] == "think"
    assert "return_token_ids" not in sent[1]


def install_namespace(root, enabled=True):
    script = lord._DUCK_VLLM_SETUP_SCRIPT
    start = script.index("TOKEN_EXACT_DIR = ")
    end = script.index("def tiny_png_data_url")
    source = script[start:end].replace("__PROJECT_DIR_NAME__", "'project'")
    namespace = {"Path": Path, "json": json, "os": os, "shutil": shutil, "IMAGE_RUNTIME_ROOT": root,
                 "VLLM_TOKEN_EXACT_HISTORY": enabled}
    exec(source, namespace)
    namespace["TOKEN_EXACT_DIR"] = TOKEN_EXACT
    return namespace


HF = """        prompt = parse_dec_only_prompt(prompt_raw)

        # When `prompt_embeds` is mixed with other modality data,
        prompt = parse_dec_only_prompt(prompt_raw)

        # See `render_messages` for the rationale.
"""


def runtime(tmp_path, source=HF):
    (tmp_path / "vllm" / "renderers").mkdir(parents=True)
    (tmp_path / "vllm" / "renderers" / "hf.py").write_text(source)
    return tmp_path / "vllm"


def test_install_patches_both_render_paths(tmp_path):
    vllm = runtime(tmp_path)
    install_namespace(tmp_path)["install_token_exact_history"]({"dist_packages": "."})
    patched = (vllm / "renderers" / "hf.py").read_text()
    assert patched.count("prompt_raw = splice_reply_tokens(prompt_raw, tokenizer)") == 2
    assert (vllm / "arc3_token_splice.py").is_file()


def test_install_is_off_by_default_and_fails_on_a_missing_anchor(tmp_path):
    vllm = runtime(tmp_path, "other source")
    install_namespace(tmp_path, enabled=False)["install_token_exact_history"]({"dist_packages": "."})
    assert not (vllm / "arc3_token_splice.py").exists()
    with pytest.raises(RuntimeError, match="anchor"):
        install_namespace(tmp_path)["install_token_exact_history"]({"dist_packages": "."})


def test_setup_script_compiles_with_the_new_functions():
    script = re.sub(r"__[A-Z0-9_]+__", "None", lord._DUCK_VLLM_SETUP_SCRIPT)
    compile(script, "setup", "exec")
    assert "install_token_exact_history(manifest)" in script and "run_token_exact_probe()" in script
