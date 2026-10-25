"""Rebuild the messages of every model request from a game's transcript.

The transcript records each analyzer turn's new messages (user prompt,
thinking, tool calls, tool results) but not which older messages were still in
the window. The window is rebuilt from the `history_messages` count the
harness logs at the end of each turn: history is always a suffix of the
message stream, so the count fixes where it starts.

Known approximations, measured by validate_rebuild.py against request logs:
- tool results are logged as display text; the JSON payload is rebuilt from it
  and loses `function_retention` notes and `returncode`;
- trims inside a turn are not logged; they are placed by a token estimate,
  and requests near the budget are flagged `uncertain`;
- a turn that ends with exactly the history count it started with is read as
  rolled back, so a trim that happens to land on that count is missed;
- harness-inserted messages that the transcript omits (the trimmed-history
  note of some versions) are not rebuilt;
- images are placeholders (`url` None), to be rendered from the event boards;
  the token estimate prices each one as a 64x64 board at the run's
  MULTIMODAL_UPSCALE (run_tokens_per_image), as the harness counts images.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STEP_HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| [^|\n]* \| [\w-]+ ---$", re.M)
SECTION = re.compile(
    r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|ASSISTANT|MODEL RESPONSE META|ANALYZER STATUS|COMPRESSED TRACE"
    r"|TOOL CALL: [^\]\n]+|TOOL RESULT: [^\]\n]+)\]\n",
    re.M,
)
IMAGE = "[image attached]"
TOOL_CALL_MARKUP = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)
# Prompt tokens per text char and fixed overhead, fitted on the request-logged runs in
# docs/reasoning/data/level_up_context/reconstruction_validation.json (all drawn at MULTIMODAL_UPSCALE 10).
TOKENS_PER_CHAR, TOKENS_FIXED = 0.333, 520.0
# Vision tokens of one image, counted as the harness counts them (tool_agent._image_part_tokens): one per
# 32x32-pixel merged patch (16-pixel patches, 2x2 merge) plus 2 sentinels. Images are 64x64 boards drawn at
# the run's MULTIMODAL_UPSCALE: 402 tokens at upscale 10, 578 at 12. A flat 402 undercounted upscale-12 runs
# by up to 17% of the context (vLLM prompt counts of 1,445 replayed requests, 2026-10-25).
VISION_PATCH_PIXELS, VISION_SENTINEL_TOKENS, ARC_FRAME_CELLS, DEFAULT_UPSCALE = 32, 2, 64, 10
# Above this share of the budget the estimate cannot tell whether the harness's
# own estimate (JSON chars / calibrated chars per token) trimmed before a request.
UNCERTAIN_SHARE = 0.85
ERROR_PREFIXES = ("Traceback (most recent call last)", "Python syntax error", "python requires", "Tool ", "Unknown tool")


@dataclass
class Request:
    """One model call: the messages sent and the reply."""
    step: int
    index: int  # 1-based request index within the analyzer turn
    messages: list[dict[str, Any]]
    reply: dict[str, Any]
    finish_reason: str | None
    history_start: int  # stream index of the first retained history message
    stream_end: int  # stream index one past the last message sent
    level: int | None = None
    uncertain: bool = False  # in a trimmed turn, close enough to the budget that the trim may have come earlier


@dataclass
class Game:
    system: str
    stream: list[dict[str, Any]] = field(default_factory=list)
    requests: list[Request] = field(default_factory=list)


def split_steps(transcript: str) -> list[tuple[int, int, str]]:
    """(analysis_step, action, body) for each analyzer invocation."""
    heads = list(STEP_HEADER.finditer(transcript))
    return [(int(h[1]), int(h[2]), transcript[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(transcript)])
            for i, h in enumerate(heads)]


def split_sections(body: str) -> list[tuple[str, str]]:
    parts = SECTION.split(body)
    return [(label, text.rstrip("\n")) for label, text in zip(parts[1::2], parts[2::2])]


def user_message(display: str, **extra: Any) -> dict[str, Any]:
    """Content parts from the transcript rendering: text pieces joined by newlines, images as markers."""
    content: list[dict[str, Any]] = []
    for i, piece in enumerate(display.split(IMAGE)):
        if i:
            content.append({"type": "image_url", "image_url": {"url": None}})
        if piece.strip():
            content.append({"type": "text", "text": piece.strip()})
    return {"role": "user", "content": content, **extra}


def parse_meta(text: str) -> tuple[str | None, list[dict[str, Any]] | None]:
    finish = re.search(r"^finish_reason: (\S+)", text, re.M)
    calls = None
    marker = text.find("raw_tool_calls:\n")
    if marker >= 0:
        try:
            calls = json.loads(text[marker + len("raw_tool_calls:\n"):])
        except json.JSONDecodeError:
            calls = None  # truncated at the meta limit: rebuilt from the TOOL CALL markup
    return (finish[1] if finish else None), calls


def markup_arguments(markup: str) -> str:
    return json.dumps({name: value for name, value in TOOL_CALL_MARKUP.findall(markup)}, ensure_ascii=False)


def tool_payload(name: str, display: str) -> str:
    """Approximate JSON tool content from its display rendering."""
    body = display
    payload: dict[str, Any] = {"tool": name}
    if body.startswith(ERROR_PREFIXES):
        payload["error"] = body
    else:
        stdout, _, error = body.partition("\n\nerror:\n")
        payload["returncode"] = 0
        payload["stdout"] = stdout
        if error:
            payload["error"] = error
    return json.dumps(payload, indent=2, ensure_ascii=False)


def message_chars(message: dict[str, Any]) -> tuple[int, int]:
    """(text chars, images) of a message, as the context estimate counts them."""
    chars, images = 0, 0
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "image_url":
                images += 1
            else:
                chars += len(part.get("text") or "")
    elif content:
        chars += len(content)
    chars += len(message.get("reasoning") or "")
    for call in message.get("tool_calls") or []:
        chars += len(call["function"].get("arguments") or "")
    return chars, images


def image_tokens(width: int, height: int) -> int:
    return -(-width // VISION_PATCH_PIXELS) * -(-height // VISION_PATCH_PIXELS) + VISION_SENTINEL_TOKENS


def tokens_per_image(upscale: int) -> int:
    side = ARC_FRAME_CELLS * upscale
    return image_tokens(side, side)


TOKENS_PER_IMAGE = tokens_per_image(DEFAULT_UPSCALE)


def run_tokens_per_image(run_dir: Path) -> int:
    """Vision tokens per image of a run, from MULTIMODAL_UPSCALE in its taaf_setup_env.json (default 10)."""
    env_path = Path(run_dir) / "taaf_setup_env.json"
    upscale = DEFAULT_UPSCALE
    if env_path.is_file():
        raw = str(json.loads(env_path.read_text(encoding="utf-8")).get("MULTIMODAL_UPSCALE") or "").strip()
        if raw:
            upscale = max(1, int(raw))
    return tokens_per_image(upscale)


def estimate_tokens(messages: list[dict[str, Any]], tokens_per_image: float = TOKENS_PER_IMAGE) -> float:
    """Prompt tokens of these messages; pass the run's run_tokens_per_image (the default is upscale 10)."""
    chars = images = 0
    for message in messages:
        c, i = message_chars(message)
        chars, images = chars + c, images + i
    return TOKENS_PER_CHAR * chars + tokens_per_image * images + TOKENS_FIXED


def rebuild(transcript: str, *, trim_threshold: float = 1.0, tokens_per_image: float = TOKENS_PER_IMAGE) -> Game:
    """Every request of one game, with its rebuilt messages.

    A turn that ends with fewer history messages than it started with plus
    what it added was trimmed. The trim is placed before the first request
    whose estimated input exceeds `trim_threshold` times the context budget,
    otherwise after the turn (turn caps and drains apply there)."""
    game: Game | None = None
    trace: dict[str, Any] | None = None
    history: list[int] = []  # stream indices currently retained
    for step, _action, body in split_steps(transcript):
        turn: list[int] = []
        sections = split_sections(body)
        kept = budget = None
        for label, text in sections:
            if label == "ANALYZER STATUS":
                if (m := re.search(r"^history_messages: (\d+)", text, re.M)):
                    kept = int(m[1])
                if (m := re.search(r"^context_budget_tokens: (\d+)", text, re.M)):
                    budget = int(m[1])
        pending_calls: list[dict[str, Any]] = []
        step_requests: list[Request] = []
        for label, text in sections:
            if label == "SYSTEM PROMPT":
                if game is None:
                    game = Game(system=text)
                game.system = text
                continue
            if game is None:
                continue
            stream = game.stream
            if label == "COMPRESSED TRACE":
                trace = user_message(text)
            elif label == "ANALYZER STATUS" and text.startswith("vc_sampling: replaced reply") and step_requests:
                # the reply and its tool results were swapped for another
                # sampled candidate, logged next without a meta section
                request = step_requests[-1]
                start = next(i for i in range(len(stream) - 1, -1, -1) if stream[i] is request.reply)
                del stream[start:]
                turn = [i for i in turn if i < start]
                request.reply = {"role": "assistant", "tool_calls": []}
                stream.append(request.reply)
                turn.append(len(stream) - 1)
                pending_calls = []
            elif label == "USER PROMPT":
                extra = {"_control": "followup"} if turn else {}
                stream.append(user_message(text, **extra))
                turn.append(len(stream) - 1)
            elif label == "MODEL RESPONSE META":
                finish, calls = parse_meta(text)
                head = [{"role": "system", "content": game.system}] + ([trace] if trace else [])
                sent = [stream[i] for i in history + turn]
                reply: dict[str, Any] = {"role": "assistant", "tool_calls": calls or []}
                stream.append(reply)
                turn.append(len(stream) - 1)
                pending_calls = list(calls or [])
                step_requests.append(Request(step, len(step_requests) + 1, head + sent, reply, finish,
                                             history[0] if history else (turn[0] if turn else len(stream) - 1),
                                             len(stream) - 1))
            elif label == "THINKING" and step_requests:
                step_requests[-1].reply["reasoning"] = text
            elif label == "ASSISTANT" and step_requests:
                step_requests[-1].reply["content"] = text
            elif label.startswith("TOOL CALL: ") and step_requests:
                reply = step_requests[-1].reply
                name = label[len("TOOL CALL: "):]
                if not pending_calls:
                    # meta was truncated: rebuild this call from the markup
                    call = {"id": f"rebuilt-{step}-{len(stream)}", "type": "function",
                            "function": {"name": name, "arguments": markup_arguments(text)}}
                    reply["tool_calls"].append(call)
                    pending_calls.append(call)
            elif label.startswith("TOOL RESULT: ") and game.stream:
                name = label[len("TOOL RESULT: "):]
                call = pending_calls.pop(0) if pending_calls else {"id": None}
                stream.append({"role": "tool", "tool_call_id": call["id"], "content": tool_payload(name, text)})
                turn.append(len(stream) - 1)
        if game is None:
            continue
        candidate = history + turn
        if kept is not None and turn and kept == len(history):
            # the turn was rolled back (a yielded or failed turn whose
            # exchange the harness discarded): its requests still ran on
            # the history they were built from
            pass
        elif kept is not None and kept < len(candidate):
            trimmed = candidate[len(candidate) - kept:]
            # a trim inside the turn: from the first request whose untrimmed
            # input would exceed the budget, requests ran on the trimmed history
            estimates = [estimate_tokens(r.messages, tokens_per_image) for r in step_requests]
            over = [bool(budget) and e > trim_threshold * budget for e in estimates]
            first = over.index(True) if True in over else len(step_requests)
            for request, estimate in zip(step_requests[:first], estimates):
                request.uncertain = bool(budget) and estimate > UNCERTAIN_SHARE * budget
            head_len = 1 + (1 if trace else 0)
            for request in step_requests[first:]:
                retained = [i for i in trimmed if i < request.stream_end]
                request.messages = request.messages[:head_len] + [game.stream[i] for i in retained]
                request.history_start = retained[0] if retained else request.stream_end
            history = trimmed
        else:
            history = candidate
        game.requests.extend(step_requests)
    return game or Game(system="")


def rebuild_file(path: Path) -> Game:
    """Rebuild <run>/transcripts/<game>.txt, pricing images at that run's upscale."""
    return rebuild(path.read_text(encoding="utf-8"), tokens_per_image=run_tokens_per_image(path.parent.parent))
