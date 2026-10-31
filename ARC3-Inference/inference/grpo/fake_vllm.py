"""A stand-in for the vLLM server in GRPO mode, for local tests of the harness and the pipeline (no GPU, no model).

Renders each request with the model's own chat template and tokenizer, as vLLM does: images become the processor's
placeholder count (`<|image_pad|>` x (H/32)(W/32) after Qwen's smart_resize at the server's max_pixels), assistant
tool-call arguments are parsed back into dicts, `preserve_thinking` is on by default. It answers in the model's
format: thinking, `</think>`, one python tool call that takes the first valid action, `<|im_end|>`, and returns
`prompt_token_ids`, the reply's `token_ids` and a logprob per token when asked (`return_token_ids`, `logprobs`).

- Every `--force-every`-th reply thinks past the request's `thinking_token_budget`: like vLLM's sampler, its
  thinking is cut at the budget and `</think>` is forced (logprob 0), so the re-rendered reply differs from it.
- `max_tokens` cuts a reply (finish reason "length", no tool call).

    python -m inference.grpo.fake_vllm --processor DIR [--port 8911 --thinking 300 --force-every 7]
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_PIXELS = 589_824  # the server's --mm-processor-kwargs max_pixels (768 x 768)
MIN_PIXELS = 65_536  # preprocessor_config shortest_edge
FACTOR = 32  # patch 16 x merge 2
CODE = "a = valid_actions[0]\nprint('acting', a)\naction([a])"


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL's resize: both sides to multiples of `factor`, pixels within [min_pixels, max_pixels]."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def image_tokens(data: bytes, max_pixels: int = MAX_PIXELS) -> int:
    from PIL import Image

    width, height = Image.open(io.BytesIO(data)).size
    h, w = smart_resize(height, width, max_pixels=max_pixels)
    return (h // FACTOR) * (w // FACTOR)


def to_template_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[bytes]]:
    """The request's messages in the template's form, and its images in order."""
    out, images = [], []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if p.get("type") == "image_url":
                    images.append(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))
                    parts.append({"type": "image"})
                else:
                    parts.append(p)
            m["content"] = parts
        if m.get("role") == "assistant":
            m["content"] = m.get("content") or ""
            calls = []
            for c in m.get("tool_calls") or []:
                fn = dict(c.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out, images


class FakeVLLM:
    def __init__(self, processor: str, thinking: int = 300, force_every: int = 7, max_pixels: int = MAX_PIXELS):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(processor)
        self.image_pad = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.thinking, self.force_every, self.max_pixels = thinking, force_every, max_pixels
        self.counter = itertools.count()
        self.lock = threading.Lock()

    def prompt_ids(self, payload: dict[str, Any]) -> list[int]:
        messages, images = to_template_messages(payload["messages"])
        kwargs = {"preserve_thinking": True, **(payload.get("chat_template_kwargs") or {})}
        text = self.tok.apply_chat_template(messages, tools=payload.get("tools"), add_generation_prompt=True,
                                            tokenize=False, **kwargs)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        out, used = [], 0
        for t in ids:
            if t == self.image_pad:
                out += [self.image_pad] * image_tokens(images[used], self.max_pixels)
                used += 1
            else:
                out.append(t)
        if used != len(images):
            raise ValueError(f"{used} image placeholders for {len(images)} images")
        return out

    def reply(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            n = next(self.counter)
        prompt = self.prompt_ids(payload)
        filler = " ".join(f"Step {n}.{i}: the board shows the same shapes, so I try the next action." for i in range(400))
        thinking_ids = self.tok(filler, add_special_tokens=False)["input_ids"]
        budget = payload.get("thinking_token_budget")
        forced = bool(budget) and self.force_every > 0 and n % self.force_every == self.force_every - 1
        if forced:
            thinking_ids = thinking_ids[:int(budget)]  # cut mid-thought; </think> is forced next
            ids = thinking_ids + [self.think_end]
            logprobs = [-0.05] * len(thinking_ids) + [0.0]
            rest = self.tok("\n\n<tool_call>\n<function=python>\n<parameter=code>\n" + CODE +
                            "\n</parameter>\n</function>\n</tool_call>", add_special_tokens=False)["input_ids"]
        else:
            thinking_ids = thinking_ids[:self.thinking]
            reasoning = self.tok.decode(thinking_ids).strip()
            text = (reasoning + "\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n" + CODE +
                    "\n</parameter>\n</function>\n</tool_call>")
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            logprobs = [-0.05] * len(ids)
            rest = []
        ids = ids + rest + [self.im_end]
        logprobs = logprobs + [-0.05] * (len(rest) + 1)
        finish = "tool_calls"
        cap = payload.get("max_tokens")
        if cap is not None and len(ids) > int(cap):
            ids, logprobs, finish = ids[:int(cap)], logprobs[:int(cap)], "length"
        text = self.tok.decode(ids, skip_special_tokens=False)
        reasoning_text, _, after = text.partition("</think>")
        message: dict[str, Any] = {"role": "assistant", "reasoning_content": reasoning_text, "content": None}
        if finish == "tool_calls" and "<parameter=code>\n" in after:
            code = after.split("<parameter=code>\n", 1)[1].split("\n</parameter>", 1)[0]
            message["tool_calls"] = [{"id": f"call_{n}", "type": "function",
                                      "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
        out = {"id": f"chatcmpl-{n}", "object": "chat.completion", "model": payload.get("model", "flashnext"),
               "choices": [{"index": 0, "finish_reason": finish, "message": message}],
               "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids),
                         "total_tokens": len(prompt) + len(ids)}}
        if payload.get("return_token_ids"):
            out["prompt_token_ids"] = prompt
            out["choices"][0]["token_ids"] = ids
        if payload.get("logprobs"):
            out["choices"][0]["logprobs"] = {"content": [{"token": f"token_id:{t}", "logprob": lp, "bytes": None,
                                                          "top_logprobs": []} for t, lp in zip(ids, logprobs)]}
        return out


def serve(fake: FakeVLLM, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": "flashnext", "object": "model",
                                                              "owned_by": "vllm", "max_model_len": 147072}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path.rstrip("/").endswith("/chat/completions"):
                try:
                    self._send(200, fake.reply(payload))
                except Exception as exc:  # noqa: BLE001 - reported to the client like a server error
                    self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            else:
                self._send(200, {})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processor", required=True, help="directory with tokenizer.json and chat_template.jinja")
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--thinking", type=int, default=300, help="thinking tokens per reply")
    parser.add_argument("--force-every", type=int, default=7, help="every n-th reply runs into the thinking budget")
    args = parser.parse_args()
    server = serve(FakeVLLM(args.processor, args.thinking, args.force_every), args.port)
    print(f"fake vLLM on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
