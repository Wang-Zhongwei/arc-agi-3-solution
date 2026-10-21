"""Serving benchmark: one SGLang launch per config, a fixed load, decode speed.

A bench notebook (``kaggle/build_notebook.py --bench <name>``) plays no games.
For each config of the bench in kaggle/bench.json it:

1. launches the server through the notebook's own launcher
   (serving/kaggle/launch_sglang.py), with that config's CFG, server-env,
   launch-argument and runtime-patch overrides, and waits until it is healthy;
2. drives the same synthetic load at it: closed-loop async clients replaying
   real harness conversations (kaggle/bench_corpus.json.gz), with thinking on,
   the harness's sampling and tools, prompts that share prefixes, and fixed seeds;
3. measures client-side aggregate tok/s, TTFT and per-stream decode tok/s, and
   parses SGLang's decode log windows (gen throughput, accept len, #running-req);
4. kills the server and moves to the next config;
5. writes bench_results.json and bench_results.md to the working directory
   after every phase, so a run cut short still reports what it measured.

The module is plain stdlib so it runs in the notebook kernel. The builder
imports it too, to validate a bench before anything is pushed.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import gzip
import json
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import types
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Plan: kaggle/bench.json -> one self-contained bench plan
# ----------------------------------------------------------------------------

OVERRIDE_KEYS = ("cfg", "env", "args_set", "args_add", "args_remove", "runtime_patches")
PHASE_DEFAULTS = {
    "concurrency": 10,
    "duration_s": 240,
    "max_tokens": 2048,
    "min_tokens": 0,
    "stream": True,
    "start_depth": 0,
    "game_offset": 0,
    "cold_fraction": 0.0,
    "requests_per_stream": None,
    "report": True,
    "profile": None,
}
BENCH_DEFAULTS = {
    "budget_min": 150,
    "drop_order": [],
    "est_startup_s": 600,
    "est_phase_overhead_s": 20,
    "startup_timeout_s": 1800,
    "request_timeout_s": 900,
    "decode_log_interval": 40,
    "thresholds": [800, 900, 1000],
    "seed": 0,
    "select_by": {"phase": "clean10", "metric": "tps_clean_median"},
}


def _merge_overrides(*parts: dict) -> dict:
    """Later parts win for mappings; lists are concatenated without repeats."""
    out: dict = {"cfg": {}, "env": {}, "args_set": {}, "args_add": [], "args_remove": [], "runtime_patches": []}
    for part in parts:
        for key in ("cfg", "env", "args_set"):
            out[key].update(part.get(key) or {})
        for key in ("args_add", "args_remove", "runtime_patches"):
            for item in part.get(key) or []:
                if item not in out[key]:
                    out[key].append(item)
    out["env"] = {str(k): str(v) for k, v in out["env"].items()}
    out["args_set"] = {str(k): [str(x) for x in (v if isinstance(v, list) else [v])]
                       for k, v in out["args_set"].items()}
    out["args_add"] = [str(x) for x in out["args_add"]]
    return out


def resolve_plan(spec: dict, bench: str) -> dict:
    """Expand one bench of kaggle/bench.json into the plan the notebook runs.

    Validates everything that can be checked offline: phases exist, compose
    groups name configs that run earlier, overrides use known keys. Runtime
    patch files are listed in plan["patches_needed"]; the builder embeds them."""
    benches = spec.get("benches", {})
    if bench not in benches:
        raise ValueError(f"unknown bench {bench!r}; defined: {', '.join(benches)}")
    settings = {**BENCH_DEFAULTS, **spec.get("defaults", {}), **benches[bench]}
    request = {k: v for k, v in spec.get("request", {}).items() if not k.startswith("_")}
    phases_spec = spec.get("phases", {})
    configs_spec = spec.get("configs", {})
    default_phases = settings.get("default_phases") or []
    plan_configs, seen, needed = [], set(), []
    for name in settings.get("configs", []):
        if name not in configs_spec:
            raise ValueError(f"bench {bench!r}: unknown config {name!r}")
        raw = configs_spec[name]
        unknown = set(raw) - set(OVERRIDE_KEYS) - {"description", "phases", "compose", "enabled", "disabled_reason"}
        if unknown:
            raise ValueError(f"config {name!r}: unknown keys {sorted(unknown)}")
        phase_names = settings.get("phases_override") or raw.get("phases") or default_phases
        if not phase_names:
            raise ValueError(f"config {name!r}: no phases")
        phases = []
        for pname in phase_names:
            if pname not in phases_spec:
                raise ValueError(f"config {name!r}: unknown phase {pname!r}")
            unknown = set(phases_spec[pname]) - set(PHASE_DEFAULTS) - {"description"}
            if unknown:
                raise ValueError(f"phase {pname!r}: unknown keys {sorted(unknown)}")
            phases.append({"name": pname, **PHASE_DEFAULTS, **phases_spec[pname]})
        compose = raw.get("compose") or []
        for group in compose:
            for ref in group:
                if ref not in seen:
                    raise ValueError(f"config {name!r}: compose names {ref!r}, which does not run before it")
        enabled = raw.get("enabled", True)
        if enabled:
            for patch in raw.get("runtime_patches") or []:
                if patch not in needed:
                    needed.append(patch)
        plan_configs.append({
            "name": name,
            "description": raw.get("description", ""),
            "enabled": enabled,
            "disabled_reason": raw.get("disabled_reason", ""),
            "compose": compose,
            "overrides": _merge_overrides(raw),
            "phases": phases,
        })
        seen.add(name)
    if not plan_configs:
        raise ValueError(f"bench {bench!r}: no configs")
    for name in settings.get("drop_order", []):
        if name not in seen:
            raise ValueError(f"bench {bench!r}: drop_order names {name!r}, which is not in the bench")
    keys = ("budget_min", "drop_order", "est_startup_s", "est_phase_overhead_s", "startup_timeout_s",
            "request_timeout_s", "decode_log_interval", "thresholds", "seed", "select_by")
    return {
        "bench": bench,
        "description": settings.get("description", ""),
        **{k: settings[k] for k in keys},
        "request": request,
        "configs": plan_configs,
        "patches_needed": needed,
    }


def config_estimate_s(config: dict, plan: dict) -> float:
    """Expected wall time of one config: startup plus every phase and its drain."""
    phases = sum(p["duration_s"] + plan["est_phase_overhead_s"] for p in config["phases"])
    return plan["est_startup_s"] + phases


def budget_drops(pending: list[dict], plan: dict, remaining_s: float) -> list[str]:
    """Names to drop, in plan["drop_order"], until the pending configs fit."""
    need = sum(config_estimate_s(c, plan) for c in pending)
    names = {c["name"] for c in pending}
    drops = []
    for name in plan["drop_order"]:
        if need <= remaining_s:
            break
        if name in names:
            drops.append(name)
            need -= config_estimate_s(next(c for c in pending if c["name"] == name), plan)
    return drops


def metric_of(result: dict, select_by: dict) -> float | None:
    """The selection metric of one finished config, or None when it has none."""
    if result.get("status") != "ok":
        return None
    for phase in result.get("phases", []):
        if phase["name"] == select_by["phase"]:
            value = (phase.get("log") or {}).get(select_by["metric"])
            return float(value) if isinstance(value, (int, float)) else None
    return None


def compose_overrides(config: dict, results: dict[str, dict], select_by: dict) -> tuple[dict, list[str]]:
    """Overrides of a config whose "compose" groups pick the best earlier config.

    Each group contributes the resolved overrides of its member with the
    highest select_by metric; a group with no measured member contributes
    nothing. The config's own overrides apply last."""
    parts, picked = [], []
    for group in config.get("compose") or []:
        scored = [(metric_of(results[n], select_by), n) for n in group if n in results]
        scored = [(v, n) for v, n in scored if v is not None]
        if not scored:
            continue
        best = max(scored, key=lambda vn: vn[0])[1]  # first wins ties: max keeps the first
        picked.append(best)
        parts.append(results[best]["overrides"])
    return _merge_overrides(*parts, config["overrides"]), picked


def describe_overrides(overrides: dict) -> str:
    bits = []
    if overrides.get("cfg"):
        bits.append("CFG " + ", ".join(f"{k}={v}" for k, v in sorted(overrides["cfg"].items())))
    if overrides.get("env"):
        bits.append("env " + ", ".join(f"{k}={v}" for k, v in sorted(overrides["env"].items())))
    if overrides.get("args_set"):
        bits.append("args " + ", ".join(f"{k} {' '.join(v)}" for k, v in sorted(overrides["args_set"].items())))
    if overrides.get("args_add"):
        bits.append("+args " + " ".join(overrides["args_add"]))
    if overrides.get("args_remove"):
        bits.append("-args " + " ".join(overrides["args_remove"]))
    if overrides.get("runtime_patches"):
        bits.append("patches " + ", ".join(overrides["runtime_patches"]))
    return "; ".join(bits) or "baseline (no overrides)"


def apply_arg_overrides(args: list, args_set: dict, args_add: list, args_remove: list) -> list:
    """Edit a launch argv: replace a flag's values, drop flags, append args.

    A flag's values are the tokens after it up to the next "--" token."""
    args = [str(a) for a in args]

    def span(flag):
        if flag not in args:
            return None
        i = args.index(flag)
        j = i + 1
        while j < len(args) and not args[j].startswith("--"):
            j += 1
        return i, j

    for flag in args_remove:
        while (s := span(flag)) is not None:
            del args[s[0]:s[1]]
    for flag, values in args_set.items():
        s = span(flag)
        if s is None:
            args += [flag, *values]
        else:
            args[s[0]:s[1]] = [flag, *values]
    return args + list(args_add)


# ----------------------------------------------------------------------------
# Load: requests built from real harness conversations
# ----------------------------------------------------------------------------

# The harness's only tool, verbatim (ARC3-Inference tool_agent.py _tools()).
PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run one ephemeral Python snippet against preloaded ASCII game state. Available globals: "
            "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
            "`valid_actions`, `last_action_call_result`, "
            "and `action(actions)` for executing one or more real environment actions. "
            "`current_frame` and each `history[*].frame` expose only `.ascii`, `.segmentation`, `.step`, "
            "`.level`, and `.shape` (a `(rows, cols)` tuple); "
            "`history[-1].frame` is the current post-action frame, not the previous frame. "
            "For before/after diffs, compare `previous_frame` to `current_frame` or use "
            "`last_transition.before_frame` and `.after_frame`. "
            "For MOUSE, pass `row` and `col` integer fields; legacy x/y fields are rejected. "
            "The raw numeric grid is not available. Use `.segmentation` as the primary view; use `.ascii` "
            "only to read a small, specific region. "
            "Use `print(...)` for compact output or assign final data to `result`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. The snippet is ephemeral and is not saved across tool calls.",
                },
            },
            "required": ["code"],
        },
    },
}

COLORS = "WwgGcBMPRbSYOrNp"


def synthetic_corpus(seed: int = 0, games: int = 4, turns: int = 6) -> dict:
    """A small procedural stand-in for the corpus (tests, or a missing corpus).

    Same shape as kaggle/bench_corpus.json.gz: a system prompt shared by every
    game, then per game user / assistant(tool call) / tool messages."""
    rng = random.Random(seed)
    system = (
        "You are a coding agent solving a grid-based puzzle game.\n"
        "Boards are 64 x 64 color grids rendered with ARC color symbols: " + COLORS + ".\n"
        "Call `action(actions)` from inside the `python` tool to act.\n"
    ) * 8
    out = []
    for g in range(games):
        messages = []
        for t in range(turns):
            rows = ["".join(rng.choice(COLORS[:4]) for _ in range(64)) for _ in range(12)]
            if t == 0:
                messages.append({"role": "user", "content": f"Game g{g}. Current state: step 1, level 1.\n"
                                 "Valid actions right now: UP, DOWN, LEFT, RIGHT.\n" + "\n".join(rows)})
            call_id = f"call_{g}_{t}"
            code = f"print(current_frame.segmentation['nodes'][:{rng.randint(3, 9)}])"
            messages.append({
                "role": "assistant", "content": "",
                "reasoning_content": f"Turn {t}: inspect the objects near row {rng.randint(0, 63)} first.",
                "tool_calls": [{"id": call_id, "type": "function",
                                "function": {"name": "python", "arguments": json.dumps({"code": code})}}],
            })
            messages.append({"role": "tool", "tool_call_id": call_id, "content": "\n".join(rows[:6])})
        out.append({"name": f"g{g}", "messages": messages})
    return {"version": 1, "source": "synthetic", "system": system, "games": out}


def load_corpus(data: bytes | None, seed: int = 0) -> dict:
    """The embedded corpus (gzipped JSON), or the synthetic one when absent."""
    if not data:
        print("bench: no corpus embedded; using the synthetic one", flush=True)
        return synthetic_corpus(seed, games=12, turns=12)
    return json.loads(gzip.decompress(data))


def cut_points(messages: list[dict]) -> list[int]:
    """Prefix lengths at which the model speaks next (after a user or tool turn)."""
    cuts = []
    for i, m in enumerate(messages):
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if m["role"] in ("user", "tool") and (nxt is None or nxt["role"] == "assistant"):
            cuts.append(i + 1)
    return cuts


def _rng(*parts) -> random.Random:
    return random.Random("|".join(map(str, parts)))


def build_request(corpus: dict, phase: dict, request: dict, *, stream_idx: int, turn: int,
                  seed: int, model: str) -> tuple[dict, dict]:
    """The chat request for one stream's turn in a phase, and what it is.

    Deterministic in (corpus, phase, request, stream, turn, seed): every config
    sees the same requests in the same order. Stream s replays game
    (s + game_offset) mod #games; turn t sends the conversation up to its
    (start_depth + t)-th model turn, so each request extends the previous one
    and the prefix cache serves all but the new suffix. The system prompt is
    shared by every stream. A cold request (cold_fraction) puts a nonce first,
    so nothing of it is cached."""
    games = corpus["games"]
    game = games[(stream_idx + phase.get("game_offset", 0)) % len(games)]
    cuts = cut_points(game["messages"])
    depth = (phase.get("start_depth", 0) + turn) % len(cuts)
    cold = _rng(seed, phase["name"], stream_idx, turn, "cold").random() < phase.get("cold_fraction", 0.0)
    system = corpus["system"]
    if cold:
        nonce = _rng(seed, phase["name"], stream_idx, turn, "nonce").getrandbits(64)
        system = f"[session {nonce:016x}]\n{system}"
    messages = [{"role": "system", "content": system}] + copy.deepcopy(game["messages"][:cuts[depth]])
    payload = {
        "model": model,
        "messages": messages,
        "stream": bool(phase["stream"]),
        "max_tokens": int(phase["max_tokens"]),
    }
    if phase.get("min_tokens"):
        payload["min_tokens"] = int(phase["min_tokens"])
    for key in ("temperature", "top_p", "top_k"):
        if request.get(key) is not None:
            payload[key] = request[key]
    if request.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = dict(request["chat_template_kwargs"])
    if request.get("tools", "python") == "python":
        payload["tools"] = [copy.deepcopy(PYTHON_TOOL)]
        if request.get("tool_choice"):
            payload["tool_choice"] = request["tool_choice"]
    if request.get("send_seed"):
        payload["seed"] = _rng(seed, phase["name"], stream_idx, turn, "seed").getrandbits(31)
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}
    info = {"stream": stream_idx, "turn": turn, "game": game["name"], "depth": depth, "cold": cold}
    return payload, info


# ----------------------------------------------------------------------------
# Async HTTP/1.1 client (stdlib only)
# ----------------------------------------------------------------------------


class SSEDecoder:
    """Incremental server-sent-events decoder: bytes in, "data:" payloads out."""

    def __init__(self):
        self._buf = b""

    def feed(self, data: bytes) -> list[str]:
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        out = []
        for line in lines:
            line = line.rstrip(b"\r")
            if line.startswith(b"data:"):
                out.append(line[5:].strip().decode("utf-8", "replace"))
        return out


async def _iter_body(reader: asyncio.StreamReader, headers: dict):
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await reader.readline()
            if not size_line:
                return
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                    pass
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)
            yield data
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining > 0:
            data = await reader.read(min(remaining, 1 << 16))
            if not data:
                return
            remaining -= len(data)
            yield data
    else:
        while data := await reader.read(1 << 16):
            yield data


async def http_request(host: str, port: int, method: str, path: str, payload=None, *, on_data=None):
    """One HTTP/1.1 request on its own connection; returns (status, body bytes).

    With on_data, body bytes are handed over as they arrive and not kept."""
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=30)
    try:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        head = (f"{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\n"
                f"Accept: application/json, text/event-stream\r\nContent-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n")
        writer.write(head.encode("ascii") + body)
        await writer.drain()
        raw = await reader.readuntil(b"\r\n\r\n")
        lines = raw.decode("latin-1").split("\r\n")
        status = int(lines[0].split()[1])
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        kept = []
        async for data in _iter_body(reader, headers):
            if on_data is not None and status == 200:
                on_data(data)
            else:
                kept.append(data)
        return status, b"".join(kept)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def new_record(info: dict) -> dict:
    return {**info, "t_send": time.time(), "t_first": None, "t_last": None, "t_end": None, "chunks": 0,
            "chunk_times": [], "status": "pending", "http": None, "completion_tokens": None,
            "prompt_tokens": None, "cached_tokens": None, "finish_reason": None, "error": None}


def _take_usage(rec: dict, usage) -> None:
    if not isinstance(usage, dict):
        return
    rec["completion_tokens"] = usage.get("completion_tokens", rec["completion_tokens"])
    rec["prompt_tokens"] = usage.get("prompt_tokens", rec["prompt_tokens"])
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        rec["cached_tokens"] = details["cached_tokens"]


def handle_stream_event(rec: dict, payload: str, now: float) -> None:
    """Fold one SSE "data:" payload of a chat-completion stream into a record."""
    if payload == "[DONE]":
        return
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return
    if not isinstance(obj, dict):
        return
    if obj.get("error"):
        rec["error"] = str(obj["error"])[:500]
        return
    _take_usage(rec, obj.get("usage"))
    for choice in obj.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            if rec["t_first"] is None:
                rec["t_first"] = now
            rec["t_last"] = now
            rec["chunks"] += 1
            rec["chunk_times"].append(now)
        if choice.get("finish_reason"):
            rec["finish_reason"] = choice["finish_reason"]


def handle_response_body(rec: dict, body: bytes) -> None:
    """Fold a non-streaming chat-completion response into a record."""
    obj = json.loads(body)
    _take_usage(rec, obj.get("usage"))
    choices = obj.get("choices") or []
    if choices:
        rec["finish_reason"] = choices[0].get("finish_reason")


async def chat(host: str, port: int, payload: dict, rec: dict) -> dict:
    """Send one chat completion and fill its record (status ok/error)."""
    sse = SSEDecoder()

    def on_data(data: bytes) -> None:
        now = time.time()
        for event in sse.feed(data):
            handle_stream_event(rec, event, now)

    rec["t_send"] = time.time()
    status, body = await http_request(host, port, "POST", "/v1/chat/completions", payload,
                                      on_data=on_data if payload.get("stream") else None)
    rec["t_end"] = time.time()
    rec["http"] = status
    if status != 200:
        rec["status"], rec["error"] = "error", f"HTTP {status}: {body[:500].decode('utf-8', 'replace')}"
        return rec
    if not payload.get("stream"):
        handle_response_body(rec, body)
    rec["status"] = "error" if rec["error"] else "ok"
    return rec


async def run_phase(host: str, port: int, model: str, corpus: dict, phase: dict, plan: dict,
                    *, log_offset=lambda: 0, profile_dir: str | None = None) -> dict:
    """Closed-loop load for one phase; returns its raw records and timings.

    Each of `concurrency` streams sends its next request as soon as the last
    returns. At duration_s the phase ends: requests in flight are cancelled
    (closing the connection makes SGLang abort them), so every config's phase
    covers the same wall time."""
    records: list[dict] = []
    t0 = time.time()
    deadline = t0 + phase["duration_s"]
    offsets = {"start": log_offset()}
    profile = {}

    async def worker(s: int) -> None:
        turn, errors = 0, 0
        while time.time() < deadline:
            if phase.get("requests_per_stream") is not None and turn >= phase["requests_per_stream"]:
                return
            payload, info = build_request(corpus, phase, plan["request"], stream_idx=s, turn=turn,
                                          seed=plan["seed"], model=model)
            rec = new_record(info)
            records.append(rec)
            try:
                await asyncio.wait_for(chat(host, port, payload, rec), timeout=plan["request_timeout_s"])
            except asyncio.CancelledError:
                rec["status"], rec["t_end"] = "cancelled", time.time()
                raise
            except Exception as exc:  # timeouts, refused connections, malformed bodies
                rec["status"], rec["t_end"], rec["error"] = "error", time.time(), repr(exc)[:500]
            if rec["status"] == "error":
                errors += 1
                if errors >= 5:
                    print(f"bench: stream {s} stopped after 5 errors; last: {rec['error']}", flush=True)
                    return
                await asyncio.sleep(1.0)
            turn += 1

    async def profiler() -> None:
        cfg = phase["profile"]
        await asyncio.sleep(cfg.get("start_after_s", 30))
        body = {"output_dir": profile_dir, "num_steps": cfg.get("num_steps", 20),
                "activities": cfg.get("activities", ["CPU", "GPU"]),
                "with_stack": cfg.get("with_stack", False), "record_shapes": cfg.get("record_shapes", False)}
        try:
            status, text = await http_request(host, port, "POST", "/start_profile", body)
            profile.update(status=status, response=text[:300].decode("utf-8", "replace"), request=body)
        except Exception as exc:
            profile.update(status=None, response=repr(exc)[:300], request=body)

    tasks = [asyncio.create_task(worker(s)) for s in range(phase["concurrency"])]
    extra = [asyncio.create_task(profiler())] if phase.get("profile") and profile_dir else []
    await asyncio.wait(tasks, timeout=max(0.0, deadline - time.time()))
    t1 = time.time()
    offsets["end"] = log_offset()
    for task in tasks + extra:
        task.cancel()
    await asyncio.gather(*tasks, *extra, return_exceptions=True)
    return {"records": records, "t0": t0, "t1": t1, "offsets": offsets, "profile": profile}


def run_coro(coro):
    """Run a coroutine to completion, also from inside Jupyter's running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def target():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, name="bench-loop")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def _pct(values, q: float):
    """Nearest-rank q-quantile (0..1), or None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def summarize_client(records: list[dict], t0: float, t1: float) -> dict:
    """Client-side numbers for one phase.

    tok_s is tokens generated inside [t0, t1] over its length. Streamed
    requests count the share of their chunks that arrived by t1; a request
    cancelled at t1 has no usage, so its tokens are its chunks times the mean
    tokens per chunk of completed requests (non-streamed: elapsed time times
    the median per-request rate)."""
    span = max(1e-9, t1 - t0)
    ok = [r for r in records if r["status"] == "ok" and r["completion_tokens"] is not None]
    cancelled = [r for r in records if r["status"] == "cancelled"]
    errors = [r for r in records if r["status"] == "error"]
    ok_chunks = sum(r["chunks"] for r in ok)
    per_chunk = sum(r["completion_tokens"] for r in ok if r["chunks"]) / ok_chunks if ok_chunks else None
    rates = [r["completion_tokens"] / (r["t_end"] - r["t_send"]) for r in ok if r["t_end"] > r["t_send"]]
    tokens = 0.0
    for r in ok:
        if r["chunk_times"]:
            tokens += r["completion_tokens"] * sum(t <= t1 for t in r["chunk_times"]) / len(r["chunk_times"])
        elif r["t_end"] <= t1:
            tokens += r["completion_tokens"]
        else:
            tokens += r["completion_tokens"] * max(0.0, t1 - r["t_send"]) / (r["t_end"] - r["t_send"])
    for r in cancelled:
        if r["chunk_times"] and per_chunk:
            tokens += per_chunk * sum(t <= t1 for t in r["chunk_times"])
        elif rates and not r["chunk_times"]:
            tokens += statistics.median(rates) * max(0.0, min(r["t_end"] or t1, t1) - r["t_send"])
    ttft = [r["t_first"] - r["t_send"] for r in ok + cancelled if r["t_first"] is not None]
    decode = [(r["completion_tokens"] - 1) / (r["t_last"] - r["t_first"]) for r in ok
              if r["t_first"] is not None and r["t_last"] > r["t_first"] and r["completion_tokens"] >= 32]
    e2e = [r["t_end"] - r["t_send"] for r in ok]
    reasons: dict[str, int] = {}
    for r in ok:
        reasons[str(r["finish_reason"])] = reasons.get(str(r["finish_reason"]), 0) + 1
    prompt = [r["prompt_tokens"] for r in ok if r["prompt_tokens"] is not None]
    hits = [(r["cached_tokens"], r["prompt_tokens"]) for r in ok
            if r["cached_tokens"] is not None and r["prompt_tokens"]]
    return {
        "requests_ok": len(ok), "requests_cancelled": len(cancelled), "requests_error": len(errors),
        "first_error": errors[0]["error"] if errors else None,
        "cold_requests": sum(1 for r in records if r.get("cold")),
        "tok_s": _r(tokens / span),
        "completion_tokens_mean": _r(statistics.mean(r["completion_tokens"] for r in ok)) if ok else None,
        "ttft_p50_s": _r(_pct(ttft, 0.5), 3), "ttft_p90_s": _r(_pct(ttft, 0.9), 3), "ttft_max_s": _r(_pct(ttft, 1.0), 3),
        "stream_tok_s_p10": _r(_pct(decode, 0.1)), "stream_tok_s_p50": _r(_pct(decode, 0.5)),
        "stream_tok_s_p90": _r(_pct(decode, 0.9)),
        "e2e_p50_s": _r(_pct(e2e, 0.5), 2), "e2e_p90_s": _r(_pct(e2e, 0.9), 2),
        "prompt_tokens_mean": _r(statistics.mean(prompt), 0) if prompt else None,
        "prompt_tokens_max": max(prompt) if prompt else None,
        "client_cache_hit": _r(sum(c for c, _ in hits) / sum(p for _, p in hits), 3) if hits else None,
        "finish_reasons": reasons,
        "length_rate": _r(reasons.get("length", 0) / len(ok), 3) if ok else None,
    }


_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] ")
_KV = re.compile(r"([#A-Za-z][\w #()/.-]*?): ([^,]+)")
_REQ = re.compile(r"ReqTimeStats\((.*?)\): (.*)$")
_EQ = re.compile(r"(\w+)=([^,\s]+)")


def _num(value: str):
    value = value.strip()
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value) if any(c in value for c in ".eE") else int(value)
    except ValueError:
        return value.rstrip("ms") if value.endswith("ms") else value


def parse_log(text: str) -> list[dict]:
    """SGLang serve.log lines -> decode / prefill / request events, in order.

    Decode and prefill lines are read as "key: value" pairs, so a fork that
    adds or drops a field still parses. Keys keep SGLang's spelling, e.g.
    "#running-req", "accept len", "gen throughput (token/s)"."""
    events = []
    for line in text.splitlines():
        m = _TS.match(line)
        ts = m[1] if m else None
        if "Decode batch, " in line or "Prefill batch, " in line:
            kind = "decode" if "Decode batch, " in line else "prefill"
            body = line.split("Decode batch, " if kind == "decode" else "Prefill batch, ", 1)[1]
            fields = {k.strip(): _num(v) for k, v in _KV.findall(body)}
            events.append({"kind": kind, "ts": ts, **fields})
        elif (m := _REQ.search(line)):
            fields = {k: _num(v.rstrip("ms") if v.endswith("ms") else v) for k, v in _EQ.findall(m[1] + ", " + m[2])}
            events.append({"kind": "req", "ts": ts, **fields})
    return events


def summarize_log(events: list[dict], *, target_bs: int, interval: int = 40,
                  thresholds=(800, 900, 1000)) -> dict:
    """Decode-window numbers for one phase's slice of serve.log.

    SGLang logs a decode line every `interval` decode steps; its gen throughput
    is the tokens generated since the last line over the time since then, so a
    prefill run in between shows up as a slow window. A window is clean when the
    event before it in the slice is also a decode line (no prefill between) and
    #running-req is target_bs. step ms = 1000 * bs * accept len / gen throughput.
    stall_share is the share of decode-window time lost against the clean
    median at the same batch size."""
    decode = [e for e in events if e["kind"] == "decode" and "gen throughput (token/s)" in e]
    clean, all_tps, by_bs = [], [], {}
    prev = None
    for e in events:
        if e["kind"] == "decode" and "gen throughput (token/s)" in e:
            bs = int(e.get("#running-req", 0))
            tps = float(e["gen throughput (token/s)"])
            all_tps.append(tps)
            is_clean = prev is not None and prev["kind"] == "decode"
            slot = by_bs.setdefault(bs, {"windows": 0, "clean": []})
            slot["windows"] += 1
            if is_clean:
                slot["clean"].append(e)
            if is_clean and bs == target_bs:
                clean.append(e)
        if e["kind"] in ("decode", "prefill"):
            prev = e

    def step_ms(e):
        tps = float(e["gen throughput (token/s)"])
        return 1000.0 * int(e.get("#running-req", 0)) * float(e.get("accept len", 1.0)) / tps if tps > 0 else None

    tps_clean = [float(e["gen throughput (token/s)"]) for e in clean]
    out = {
        "target_bs": target_bs,
        "windows": len(decode),
        "windows_clean": len(clean),
        "tps_clean_median": _r(_pct(tps_clean, 0.5)), "tps_clean_p90": _r(_pct(tps_clean, 0.9)),
        "tps_clean_max": _r(_pct(tps_clean, 1.0)),
        "tps_all_median": _r(_pct(all_tps, 0.5)), "tps_all_p90": _r(_pct(all_tps, 0.9)),
        "tps_all_p99": _r(_pct(all_tps, 0.99)), "tps_all_max": _r(_pct(all_tps, 1.0)),
        "accept_clean_mean": _r(statistics.mean(float(e.get("accept len", 0)) for e in clean), 3) if clean else None,
        "accept_all_mean": _r(statistics.mean(float(e.get("accept len", 0)) for e in decode), 3) if decode else None,
        "step_ms_clean_median": _r(_pct([s for e in clean if (s := step_ms(e))], 0.5), 2),
        "bs_mean": _r(statistics.mean(int(e.get("#running-req", 0)) for e in decode), 2) if decode else None,
        "kv_usage_max": max((float(e.get("full token usage", 0)) for e in decode), default=None),
        "queue_max": max((int(e.get("#queue-req", 0)) for e in decode), default=None),
        "cuda_graph_share": _r(sum(bool(e.get("cuda graph")) for e in decode) / len(decode), 3) if decode else None,
    }
    for t in thresholds:
        out[f"clean_ge_{t}"] = _r(sum(x >= t for x in tps_clean) / len(tps_clean), 3) if tps_clean else None
        out[f"all_ge_{t}"] = _r(sum(x >= t for x in all_tps) / len(all_tps), 3) if all_tps else None
    out["by_bs"] = {
        str(bs): {
            "windows": v["windows"], "clean": len(v["clean"]),
            "tps_clean_median": _r(_pct([float(e["gen throughput (token/s)"]) for e in v["clean"]], 0.5)),
            "accept_clean_mean": _r(statistics.mean(float(e.get("accept len", 0)) for e in v["clean"]), 3)
            if v["clean"] else None,
            "step_ms_clean_median": _r(_pct([s for e in v["clean"] if (s := step_ms(e))], 0.5), 2),
        }
        for bs, v in sorted(by_bs.items())
    }
    lost = total = 0.0
    for e in decode:
        bs, tps = int(e.get("#running-req", 0)), float(e["gen throughput (token/s)"])
        ref = out["by_bs"].get(str(bs), {}).get("tps_clean_median")
        if tps > 0 and ref:
            tokens = interval * bs * float(e.get("accept len", 1.0))
            total += tokens / tps
            lost += max(0.0, tokens / tps - tokens / ref)
    out["stall_share"] = _r(lost / total, 3) if total else None
    prefill = [e for e in events if e["kind"] == "prefill"]
    new = sum(int(e.get("#new-token", 0)) for e in prefill)
    cached = sum(int(e.get("#cached-token", 0)) for e in prefill)
    out.update({
        "prefill_batches": len(prefill),
        "prefill_cold_batches": sum(1 for e in prefill if int(e.get("#cached-token", 0)) == 0),
        "prefill_new_tokens": new,
        "prefill_cache_hit": _r(cached / (cached + new), 3) if cached + new else None,
    })
    reqs = [e for e in events if e["kind"] == "req"]
    if reqs:
        queue = [float(e.get("queue_duration", 0)) for e in reqs]
        first = [float(e.get("queue_duration", 0)) + float(e.get("initial_prefill_elapsed", 0)) for e in reqs]
        out.update({"server_requests": len(reqs), "queue_p50_ms": _r(_pct(queue, 0.5)),
                    "queue_p99_ms": _r(_pct(queue, 0.99)), "server_ttft_p50_ms": _r(_pct(first, 0.5)),
                    "server_ttft_p90_ms": _r(_pct(first, 0.9))})
    return out


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def _fmt(value, spec="{:.0f}"):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return spec.format(value)


def render_markdown(results: dict) -> str:
    """bench_results.md: one row per config, then one row per measured phase."""
    plan = results["plan"]
    lines = [f"# Serving bench `{plan['bench']}`", ""]
    if plan.get("description"):
        lines += [plan["description"], ""]
    lines += [f"Started {results['started_at']}; last written {results['updated_at']}. "
              f"Sampling: {json.dumps({k: v for k, v in plan['request'].items() if k != 'tools'})}.", ""]
    lines += ["## Configs", "",
              "| config | status | change | startup s | note |", "|---|---|---|---:|---|"]
    for c in results["configs"]:
        note = c.get("reason") or ""
        if c.get("composed_from"):
            note = (note + "; " if note else "") + "composed from " + ", ".join(c["composed_from"])
        lines.append(f"| {c['name']} | {c['status']} | {describe_overrides(c.get('overrides') or {})} | "
                     f"{_fmt(c.get('startup_s'))} | {note.replace('|', '/')} |")
    th = plan["thresholds"]
    mid = th[len(th) // 2] if th else 900
    lines += ["", f"## Decode by phase (gen throughput windows from serve.log; clean = full batch, no prefill)", "",
              f"| config | phase | conc | bs | clean/all win | clean median | clean p90 | max (any) | clean >={mid} "
              f"| accept | step ms | client tok/s | TTFT p50 s | stream tok/s p50 | stall | length |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in results["configs"]:
        for p in c.get("phases", []):
            if not p.get("report", True):
                continue
            lg, cl = p.get("log") or {}, p.get("client") or {}
            lines.append(
                f"| {c['name']} | {p['name']} | {p['concurrency']} | {lg.get('target_bs', '-')} | "
                f"{lg.get('windows_clean', '-')}/{lg.get('windows', '-')} | {_fmt(lg.get('tps_clean_median'))} | "
                f"{_fmt(lg.get('tps_clean_p90'))} | {_fmt(lg.get('tps_all_max'))} | "
                f"{_fmt(lg.get(f'clean_ge_{mid}'), '{:.0%}')} | {_fmt(lg.get('accept_clean_mean'), '{:.2f}')} | "
                f"{_fmt(lg.get('step_ms_clean_median'), '{:.1f}')} | {_fmt(cl.get('tok_s'))} | "
                f"{_fmt(cl.get('ttft_p50_s'), '{:.2f}')} | {_fmt(cl.get('stream_tok_s_p50'))} | "
                f"{_fmt(lg.get('stall_share'), '{:.0%}')} | {_fmt(cl.get('length_rate'), '{:.0%}')} |")
    lines += ["", "Per batch size, prefill and request numbers are in bench_results.json.", ""]
    return "\n".join(lines)


def write_results(results: dict, working_dir: Path) -> None:
    results["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (working_dir / "bench_results.json").write_text(json.dumps(results, indent=1) + "\n")
    (working_dir / "bench_results.md").write_text(render_markdown(results))


# ----------------------------------------------------------------------------
# Server control (runs in the notebook)
# ----------------------------------------------------------------------------


def prepare_site_packages(venv: str, prefix: str) -> None:
    """Give each config the installed SGLang exactly as shipped.

    The first call snapshots site-packages/sglang; later calls restore it and
    clear the launcher's applied-patch markers, so the launcher re-applies the
    base runtime patches plus only this config's. Bench patches must therefore
    edit files under site-packages/sglang only."""
    site = sorted(Path(venv).glob("lib/python*/site-packages"))
    if len(site) != 1:
        raise RuntimeError(f"Expected one site-packages under {venv}, found {site}")
    live = site[0] / "sglang"
    pristine = Path(prefix) / "bench-pristine" / "sglang"
    markers = Path(prefix) / "runtime-patches"
    if not pristine.exists():
        if markers.is_dir() and any(markers.glob("*.applied")):
            raise RuntimeError("runtime patches were applied before the bench snapshot; SGLang is not pristine")
        pristine.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(live, pristine, symlinks=True)
        print(f"bench: snapshot of the installed SGLang at {pristine}", flush=True)
    else:
        shutil.rmtree(live)
        shutil.copytree(pristine, live, symlinks=True)
        print("bench: restored the installed SGLang from the snapshot", flush=True)
    for marker in markers.glob("*.applied") if markers.is_dir() else []:
        marker.unlink()


def _get(url: str, timeout: float = 5.0) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as exc:
        return getattr(exc, "code", None), repr(exc)


def _pauses(first: float = 0.1, cap: float = 2.0):
    """Polling pauses: quick at first, then every `cap` seconds."""
    pause = first
    while True:
        yield pause
        pause = min(cap, pause * 2)


def wait_healthy(port: int, proc, timeout_s: float) -> bool:
    end = time.time() + timeout_s
    for pause in _pauses():
        if proc is not None and proc.poll() is not None:
            return False
        if _get(f"http://127.0.0.1:{port}/health")[0] == 200:
            return True
        if time.time() >= end:
            return False
        time.sleep(pause)
    return False


def wait_idle(port: int, timeout_s: float = 60.0) -> bool:
    """Wait until /metrics shows no running or queued requests."""
    end = time.time() + timeout_s
    for pause in _pauses(0.05, 1.0):
        if time.time() >= end:
            return False
        status, text = _get(f"http://127.0.0.1:{port}/metrics")
        if status != 200:
            time.sleep(5)
            return False
        busy = 0.0
        for line in text.splitlines():
            if line.startswith(("sglang:num_running_reqs", "sglang:num_queue_reqs")):
                try:
                    busy += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        if busy == 0:
            return True
        time.sleep(pause)
    return False


def stop_server(proc, port: int) -> None:
    """Kill the server's process group and wait for its port and GPU memory."""
    if proc is not None:
        for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
            if proc.poll() is not None:
                break
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                pass
        try:  # workers can outlive the leader
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    end = time.time() + 120
    while time.time() < end:
        with socket.socket() as probe:
            port_free = probe.connect_ex(("127.0.0.1", int(port))) != 0
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()
            gpu_free = all(float(u) < 2048 for u in used)
        except Exception:
            gpu_free = True
        if port_free and gpu_free:
            return
        time.sleep(3)
    print("bench: server port or GPU memory still busy after 120 s", flush=True)


def _log_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _read_slice(path: Path, start: int, end: int) -> str:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(max(0, end - start)).decode("utf-8", "replace")


def preflight(port: int, model: str, corpus: dict, plan: dict, *, min_tokens: int = 8) -> dict:
    """One short streamed request shaped like the load (min_tokens included).

    A shape the server rejects then fails the config in seconds rather than
    after every phase has logged errors."""
    phase = {**PHASE_DEFAULTS, "name": "preflight", "max_tokens": 16, "min_tokens": min_tokens}
    payload, info = build_request(corpus, phase, plan["request"], stream_idx=0, turn=0, seed=plan["seed"],
                                  model=model)
    rec = new_record(info)
    try:
        rec = run_coro(asyncio.wait_for(chat("127.0.0.1", port, payload, rec), timeout=plan["request_timeout_s"]))
    except Exception as exc:
        rec.update(status="error", error=repr(exc)[:500], t_end=time.time())
    out = {k: rec[k] for k in ("status", "http", "error", "prompt_tokens", "completion_tokens", "finish_reason")}
    return {**out, "min_tokens": min_tokens, "seconds": _r(rec["t_end"] - rec["t_send"], 2)}


def abort_all(port: int) -> None:
    """Ask SGLang to abort whatever still runs (best effort; older servers lack it)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/abort_request", method="POST",
                                 data=json.dumps({"rid": "", "abort_all": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def launch(launcher_src: str, nb: dict, overrides: dict, out_dir: Path, plan: dict, ns: dict) -> dict:
    """Run the notebook's launcher cell for one config in namespace `ns`.

    The builder edited the launcher to read BENCH_* names: CFG overrides,
    server env, a site-packages reset before runtime patches, extra patches,
    and a hook over the final argv. `ns` is the caller's, so a launch that
    fails after starting the server still leaves `proc` there to kill."""
    ns.update({k: nb[k] for k in ("MODEL_DIR", "DRAFT_MODEL_DIR", "WHEELHOUSE_DIR", "SERVED_MODEL_NAME",
                                  "SERVED_MODEL_PORT")})
    precache = nb.get("precache_model_thread")
    ns.update({
        "__name__": "bench_launcher",
        "WORKING_DIR": str(out_dir),
        "NOTEBOOK_START_TIME": time.time(),
        "SERVER_STARTUP_TIMEOUT": plan["startup_timeout_s"],
        # A thread starts once; later launches find the page cache already warm.
        "precache_model_thread": types.SimpleNamespace(
            start=lambda: precache.start() if precache is not None and precache.ident is None else None),
        "BENCH_CFG": dict(overrides["cfg"]),
        "BENCH_ENV": dict(overrides["env"]),
        "BENCH_RUNTIME_PATCHES": {f"bench-{name}": plan["patches"][name] for name in overrides["runtime_patches"]},
        "BENCH_PREPARE_SITE_PACKAGES": prepare_site_packages,
        "BENCH_ARGS": lambda args: apply_arg_overrides(args, overrides["args_set"], overrides["args_add"],
                                                       overrides["args_remove"]),
    })
    exec(compile(launcher_src, "serving/kaggle/launch_sglang.py", "exec"), ns)
    return ns


def run_bench(plan: dict, *, launcher: str, corpus: bytes | None, nb: dict, working_dir) -> dict:
    """Run every config of the plan; returns the results also written to disk."""
    working_dir = Path(working_dir)
    root = working_dir / "bench"
    root.mkdir(parents=True, exist_ok=True)
    port, model = int(nb["SERVED_MODEL_PORT"]), str(nb["SERVED_MODEL_NAME"])
    corpus_obj = load_corpus(corpus, plan["seed"])
    deadline = float(nb.get("NOTEBOOK_START_TIME", time.time())) + 60 * plan["budget_min"]
    results = {"plan": {k: v for k, v in plan.items() if k != "patches"},
               "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "corpus": {"source": corpus_obj.get("source"), "games": [g["name"] for g in corpus_obj["games"]]},
               "configs": []}
    by_name: dict[str, dict] = {}
    pending = list(plan["configs"])
    while pending:
        config = pending.pop(0)
        entry = {"name": config["name"], "description": config["description"], "status": "pending",
                 "overrides": config["overrides"], "phases": []}
        results["configs"].append(entry)
        by_name[config["name"]] = entry
        if not config["enabled"]:
            entry.update(status="disabled", reason=config["disabled_reason"])
            write_results(results, working_dir)
            continue
        runnable = [config] + [c for c in pending if c["enabled"]]
        drops = budget_drops(runnable, plan, deadline - time.time())
        for c in runnable[1:]:
            if c["name"] in drops:
                results["configs"].append({"name": c["name"], "description": c["description"], "status": "dropped",
                                           "reason": "session budget", "overrides": c["overrides"], "phases": []})
        pending = [c for c in pending if c["name"] not in drops]
        if config["name"] in drops:
            entry.update(status="dropped", reason="session budget")
            write_results(results, working_dir)
            continue
        if config_estimate_s(config, plan) > deadline - time.time():
            entry.update(status="skipped", reason="session budget")
            write_results(results, working_dir)
            continue
        overrides, picked = compose_overrides(config, by_name, plan["select_by"])
        entry.update(overrides=overrides, composed_from=picked)
        out_dir = root / config["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== bench config {config['name']}: {describe_overrides(overrides)} =====", flush=True)
        t_start = time.time()
        ns: dict = {}
        try:
            launch(launcher, nb, overrides, out_dir, plan, ns)
            if not wait_healthy(port, ns.get("proc"), max(0.0, plan["startup_timeout_s"] - (time.time() - t_start))):
                raise RuntimeError("server did not become healthy")
            entry["startup_s"] = _r(time.time() - t_start)
            cfg = ns.get("CFG", {})
            entry["effective_cfg"] = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))}
            entry["launch_args"] = [str(a) for a in ns.get("args", [])]
            entry["preflight"] = preflight(port, model, corpus_obj, plan)
            strip_min_tokens = False
            if entry["preflight"]["status"] != "ok":
                # A server that rejects min_tokens can still be measured, with natural lengths.
                retry = preflight(port, model, corpus_obj, plan, min_tokens=0)
                if retry["status"] != "ok":
                    raise RuntimeError(f"preflight request failed: {entry['preflight']}")
                strip_min_tokens = True
                entry["warning"] = f"min_tokens rejected ({entry['preflight']['error']}); phases ran without it"
                entry["preflight"] = retry
                print(f"bench: {entry['warning']}", flush=True)
            log_path = Path(ns["LOG"])
            max_running = int(cfg.get("MAXREQ", 10))
            for phase in config["phases"]:
                if strip_min_tokens:
                    phase = {**phase, "min_tokens": 0}
                print(f"bench {config['name']}/{phase['name']}: {phase['concurrency']} streams, "
                      f"{phase['duration_s']} s", flush=True)
                raw = run_coro(run_phase("127.0.0.1", port, model, corpus_obj, phase, plan,
                                         log_offset=lambda: _log_size(log_path),
                                         profile_dir=str(out_dir / "profile")))
                abort_all(port)
                idle = wait_idle(port, timeout_s=120)
                text = _read_slice(log_path, raw["offsets"]["start"], raw["offsets"]["end"])
                target = min(phase["concurrency"], max_running)
                record = {
                    "name": phase["name"], "concurrency": phase["concurrency"], "stream": phase["stream"],
                    "max_tokens": phase["max_tokens"], "min_tokens": phase["min_tokens"],
                    "duration_s": _r(raw["t1"] - raw["t0"]), "report": phase["report"], "idle_after": idle,
                    "log_bytes": [raw["offsets"]["start"], raw["offsets"]["end"]],
                    "log": summarize_log(parse_log(text), target_bs=target, interval=plan["decode_log_interval"],
                                         thresholds=plan["thresholds"]),
                    "client": summarize_client(raw["records"], raw["t0"], raw["t1"]),
                }
                if raw["profile"]:
                    record["profile"] = raw["profile"]
                entry["phases"].append(record)
                lg, cl = record["log"], record["client"]
                print(f"  clean median {lg['tps_clean_median']} p90 {lg['tps_clean_p90']} max {lg['tps_all_max']} "
                      f"({lg['windows_clean']}/{lg['windows']} windows), accept {lg['accept_clean_mean']}, "
                      f"client {cl['tok_s']} tok/s, ttft p50 {cl['ttft_p50_s']} s, "
                      f"ok/cancel/err {cl['requests_ok']}/{cl['requests_cancelled']}/{cl['requests_error']}",
                      flush=True)
                write_results(results, working_dir)
            entry["status"] = "ok"
        except Exception as exc:
            entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}"[:800])
            print(f"bench {config['name']} failed: {entry['reason']}", flush=True)
        finally:
            stop_server(ns.get("proc"), port)
            entry["wall_s"] = _r(time.time() - t_start)
            write_results(results, working_dir)
    print("\n" + render_markdown(results), flush=True)
    return results


def describe(plan: dict) -> None:
    print(f"Serving bench {plan['bench']!r}: budget {plan['budget_min']} min, drop order {plan['drop_order']}")
    for c in plan["configs"]:
        state = "" if c["enabled"] else f"  [disabled: {c['disabled_reason']}]"
        compose = f" + best of {c['compose']}" if c["compose"] else ""
        print(f"  {c['name']}: {describe_overrides(c['overrides'])}{compose}; "
              f"phases {', '.join(p['name'] for p in c['phases'])}; ~{config_estimate_s(c, plan) / 60:.0f} min{state}")


def encode_blob(obj: dict) -> str:
    """gzip + base64 in 120-char lines, as the builder embeds payloads."""
    encoded = base64.b64encode(gzip.compress(json.dumps(obj).encode("utf-8"), 9, mtime=0)).decode("ascii")
    return "\n".join(encoded[i:i + 120] for i in range(0, len(encoded), 120))
