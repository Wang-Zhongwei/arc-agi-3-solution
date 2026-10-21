"""Tests for the serving benchmark: plan, requests, log parsing, client, build.

Pure Python and quick: the client and the bench driver run against a tiny fake
OpenAI-compatible server (a subprocess) instead of SGLang.

Run from ARC3-Inference:  python -m pytest -q ../kaggle/tests/test_bench.py
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load("serving_bench", REPO / "serving" / "kaggle" / "bench.py")
SPEC = json.loads((REPO / "kaggle" / "bench.json").read_text())


# ---------------------------------------------------------------- plan


def test_levers_plan_matches_the_throughput_plan():
    plan = bench.resolve_plan(SPEC, "levers")
    configs = {c["name"]: c for c in plan["configs"]}
    assert [c["name"] for c in plan["configs"]] == ["C0", "C7", "C1", "C2", "C3", "C4", "C5", "C6"]
    # config 0 is Franzen's launcher untouched
    assert bench.describe_overrides(configs["C0"]["overrides"]) == "baseline (no overrides)"
    assert configs["C1"]["overrides"]["cfg"] == {"SPEC_ACCEPT_SINGLE": 0.5, "SPEC_ACCEPT_ACC": 0.5}
    assert configs["C2"]["overrides"]["cfg"] == {"SPEC_ACCEPT_SINGLE": 0.3, "SPEC_ACCEPT_ACC": 0.3}
    assert configs["C3"]["overrides"]["runtime_patches"] == ["m64.py", "nosplitk.py"]
    assert not configs["C4"]["enabled"]
    assert plan["patches_needed"] == ["m64.py", "nosplitk.py"]  # the disabled C4 needs none
    assert configs["C6"]["overrides"]["args_set"] == {"--cuda-graph-bs-decode": [str(i) for i in range(1, 11)]}
    assert configs["C7"]["overrides"]["cfg"]["MAXREQ"] == 14
    assert plan["drop_order"] == ["C6", "C2"]
    # the load sends what the harness sends
    assert plan["request"] == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "tools": "python", "tool_choice": "auto", "send_seed": False,
    }
    clean = next(p for p in configs["C1"]["phases"] if p["name"] == "clean10")
    assert (clean["concurrency"], clean["min_tokens"], clean["max_tokens"]) == (10, 1536, 2048)


def test_smoke_plan_overrides_every_configs_phases():
    plan = bench.resolve_plan(SPEC, "smoke")
    assert [p["name"] for p in plan["configs"][0]["phases"]] == ["smoke-warmup", "smoke-clean", "smoke-mixed"]


@pytest.mark.parametrize("edit, message", [
    (lambda s: s["configs"]["C1"].update(phases=["nope"]), "unknown phase"),
    (lambda s: s["configs"]["C1"].update(colour="red"), "unknown keys"),
    (lambda s: s["phases"]["clean10"].update(conc=3), "unknown keys"),
    (lambda s: s["configs"]["C0"].update(compose=[["C1"]]), "does not run before it"),
    (lambda s: s["benches"]["levers"].update(drop_order=["C9"]), "not in the bench"),
])
def test_resolve_plan_rejects_mistakes(edit, message):
    spec = json.loads(json.dumps(SPEC))
    edit(spec)
    with pytest.raises(ValueError, match=message):
        bench.resolve_plan(spec, "levers")


def _result(name, tps, status="ok", **overrides):
    return {"name": name, "status": status, "overrides": bench._merge_overrides(overrides),
            "phases": [{"name": "clean10", "log": {"tps_clean_median": tps}}]}


def test_compose_takes_the_fastest_measured_member_of_each_group():
    select = {"phase": "clean10", "metric": "tps_clean_median"}
    results = {
        "C0": _result("C0", 790.0),
        "C1": _result("C1", 840.0, cfg={"SPEC_ACCEPT_SINGLE": 0.5, "SPEC_ACCEPT_ACC": 0.5}),
        "C2": _result("C2", 880.0, status="failed", cfg={"SPEC_ACCEPT_SINGLE": 0.3}),
        "C3": _result("C3", 805.0, runtime_patches=["m64.py"], env={"X": 1}),
        "C4": {"name": "C4", "status": "disabled", "overrides": bench._merge_overrides({}), "phases": []},
    }
    config = {"compose": [["C0", "C3", "C4"], ["C0", "C1", "C2"]],
              "overrides": bench._merge_overrides({"env": {"X": "2"}})}
    overrides, picked = bench.compose_overrides(config, results, select)
    assert picked == ["C3", "C1"]
    assert overrides["runtime_patches"] == ["m64.py"]
    assert overrides["cfg"] == {"SPEC_ACCEPT_SINGLE": 0.5, "SPEC_ACCEPT_ACC": 0.5}
    assert overrides["env"] == {"X": "2"}  # the config's own overrides apply last


def test_budget_drops_in_drop_order_until_the_rest_fits():
    plan = {"est_startup_s": 600, "est_phase_overhead_s": 0, "drop_order": ["C6", "C2"]}
    pending = [{"name": n, "phases": [{"duration_s": 600}]} for n in ("C2", "C5", "C6")]  # 20 min each
    assert bench.budget_drops(pending, plan, 3 * 1200) == []
    assert bench.budget_drops(pending, plan, 2 * 1200) == ["C6"]
    assert bench.budget_drops(pending, plan, 1200) == ["C6", "C2"]


def test_arg_overrides_replace_drop_and_append():
    args = ["sglang", "serve", "--cuda-graph-bs-decode", "1", "2", "4", "--schedule-policy", "lpm", "--x"]
    out = bench.apply_arg_overrides(args, {"--cuda-graph-bs-decode": ["1", "2", "3"], "--new": ["5"]},
                                    ["--flag"], ["--x"])
    assert out == ["sglang", "serve", "--cuda-graph-bs-decode", "1", "2", "3", "--schedule-policy", "lpm",
                   "--new", "5", "--flag"]


# ---------------------------------------------------------------- requests


@pytest.fixture(scope="module")
def corpus():
    return json.loads(gzip.decompress((REPO / "kaggle" / "bench_corpus.json.gz").read_bytes()))


def _phase(**kw):
    return {**bench.PHASE_DEFAULTS, "name": "clean10", **kw}


REQUEST = SPEC["request"]


def test_corpus_is_real_harness_conversations(corpus):
    assert len(corpus["games"]) >= 16  # one game per stream at 16 clients
    assert corpus["system"].startswith("You are a coding agent solving a grid-based puzzle game.")
    for game in corpus["games"]:
        assert game["messages"][0]["role"] == "user"
        assert len(bench.cut_points(game["messages"])) >= 8
        calls = {c["id"] for m in game["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])}
        assert all(m["tool_call_id"] in calls for m in game["messages"] if m["role"] == "tool")


def test_cut_points_end_where_the_model_speaks():
    msgs = [{"role": r} for r in ("user", "assistant", "tool", "assistant", "tool", "user", "assistant")]
    # a tool result followed by a new user turn was never sent on its own
    assert bench.cut_points(msgs) == [1, 3, 6]


def test_request_carries_the_harness_shape(corpus):
    payload, info = bench.build_request(corpus, _phase(min_tokens=1536), REQUEST, stream_idx=3, turn=0,
                                        seed=1, model="flashnext")
    assert payload["model"] == "flashnext" and payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert (payload["max_tokens"], payload["min_tokens"]) == (2048, 1536)
    assert (payload["temperature"], payload["top_p"], payload["top_k"]) == (0.7, 0.95, 20)
    assert payload["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert payload["tools"][0]["function"]["name"] == "python" and payload["tool_choice"] == "auto"
    assert "seed" not in payload
    assert payload["messages"][0] == {"role": "system", "content": corpus["system"]}
    assert payload["messages"][-1]["role"] in ("user", "tool")
    assert info == {"stream": 3, "turn": 0, "game": corpus["games"][3]["name"], "depth": 0, "cold": False}


def test_requests_are_deterministic_and_extend_each_other(corpus):
    phase = _phase()
    a, _ = bench.build_request(corpus, phase, REQUEST, stream_idx=2, turn=4, seed=7, model="m")
    b, _ = bench.build_request(corpus, phase, REQUEST, stream_idx=2, turn=4, seed=7, model="m")
    assert a == b
    nxt, _ = bench.build_request(corpus, phase, REQUEST, stream_idx=2, turn=5, seed=7, model="m")
    assert nxt["messages"][:len(a["messages"])] == a["messages"]  # the prefix cache serves the earlier turn
    other, _ = bench.build_request(corpus, phase, REQUEST, stream_idx=5, turn=4, seed=7, model="m")
    assert other["messages"][0] == a["messages"][0]  # every stream shares the system prompt
    assert other["messages"][1] != a["messages"][1]  # but plays its own game


def test_depth_starts_at_start_depth_and_wraps(corpus):
    n = len(bench.cut_points(corpus["games"][0]["messages"]))
    _, info = bench.build_request(corpus, _phase(start_depth=6), REQUEST, stream_idx=0, turn=1, seed=0, model="m")
    assert info["depth"] == 7
    _, info = bench.build_request(corpus, _phase(), REQUEST, stream_idx=0, turn=n, seed=0, model="m")
    assert info["depth"] == 0


def test_cold_requests_put_a_nonce_first(corpus):
    phase = _phase(name="mixed10", cold_fraction=0.25, stream=False)
    infos = [bench.build_request(corpus, phase, REQUEST, stream_idx=s, turn=t, seed=3, model="m")
             for s in range(10) for t in range(20)]
    cold = [(p, i) for p, i in infos if i["cold"]]
    assert 0.15 < len(cold) / len(infos) < 0.35
    payload, _ = cold[0]
    assert payload["messages"][0]["content"].startswith("[session ")
    assert payload["messages"][0]["content"].endswith(corpus["system"])
    assert "stream_options" not in payload
    assert len({p["messages"][0]["content"] for p, _ in cold}) == len(cold)  # every cold request is new


def test_send_seed_is_opt_in_and_fixed(corpus):
    req = {**REQUEST, "send_seed": True}
    a, _ = bench.build_request(corpus, _phase(), req, stream_idx=0, turn=0, seed=5, model="m")
    b, _ = bench.build_request(corpus, _phase(), req, stream_idx=0, turn=0, seed=5, model="m")
    assert a["seed"] == b["seed"]


def test_synthetic_corpus_has_the_corpus_shape():
    corpus = bench.synthetic_corpus(seed=1, games=3, turns=4)
    payload, _ = bench.build_request(corpus, _phase(), REQUEST, stream_idx=1, turn=2, seed=0, model="m")
    assert payload["messages"][-1]["role"] == "tool"


# ---------------------------------------------------------------- log parsing

D = ("[2026-10-03 01:27:{s:02d}] Decode batch, #running-req: {bs}, #full token: 29760, full token usage: {kv}, "
     "mamba num: 40, mamba usage: 0.67, accept len: {acc}, accept rate: 0.58, cuda graph: True, "
     "gen throughput (token/s): {tps}, #queue-req: {q}")
P = ("[2026-10-03 01:27:{s:02d}] Prefill batch, #new-seq: 1, #new-token: {new}, #cached-token: {cached}, "
     "full token usage: 0.00, mamba usage: 0.05, #running-req: 9, #queue-req: 0, #pending-token: 0, "
     "cuda graph: False, input throughput (token/s): 7.33")
R = ("[2026-10-03 01:26:42] ReqTimeStats(rid=35dc19f5e0b749f2a11b7b41279552d7, input_len=10, cached_input_len=0, "
     "output_len=35, attempts=0, type=unified): queue_duration=1.39ms, initial_prefill_elapsed=8102.75ms, "
     "post_prefill_elapsed=164.51ms, forward_duration=8267.25ms, entry_time=1790990794.588")


def _d(s, bs=10, tps=800.0, acc=2.5, kv=0.03, q=0):
    return D.format(s=s, bs=bs, tps=f"{tps:.2f}", acc=f"{acc:.2f}", kv=f"{kv:.2f}", q=q)


def test_parse_log_reads_sglang_lines():
    events = bench.parse_log("\n".join([_d(5), P.format(s=6, new=512, cached=0), R, "noise line"]))
    assert [e["kind"] for e in events] == ["decode", "prefill", "req"]
    d, p, r = events
    assert d["#running-req"] == 10 and d["accept len"] == 2.5 and d["gen throughput (token/s)"] == 800.0
    assert d["cuda graph"] is True and d["#queue-req"] == 0 and d["ts"] == "2026-10-03 01:27:05"
    assert p["#new-token"] == 512 and p["#cached-token"] == 0
    assert r["input_len"] == 10 and r["queue_duration"] == 1.39 and r["initial_prefill_elapsed"] == 8102.75


def test_summarize_log_counts_only_full_batch_windows_without_prefill():
    lines = [
        _d(1, tps=500.0),                      # first in the slice: not clean
        _d(2, tps=820.0, acc=2.6),             # clean
        P.format(s=3, new=4096, cached=0),     # a cold prefill
        _d(4, tps=400.0),                      # stalled by it: not clean
        _d(5, tps=900.0, acc=2.8, kv=0.5),     # clean
        _d(6, bs=9, tps=700.0),                # clean, but bs 9
        P.format(s=7, new=256, cached=7936),
        _d(8, tps=880.0, acc=2.7, q=2),        # after a prefill: not clean
        _d(9, tps=960.0, acc=3.0),             # clean
        R,
    ]
    out = bench.summarize_log(bench.parse_log("\n".join(lines)), target_bs=10, thresholds=(900,))
    assert (out["windows"], out["windows_clean"]) == (7, 3)
    assert (out["tps_clean_median"], out["tps_clean_max"], out["tps_all_max"]) == (900.0, 960.0, 960.0)
    assert out["clean_ge_900"] == round(2 / 3, 3)
    assert out["accept_clean_mean"] == round((2.6 + 2.8 + 3.0) / 3, 3)
    # step ms of the clean windows: 31.7, 31.1, 31.25
    assert out["step_ms_clean_median"] == round(1000 * 10 * 3.0 / 960.0, 2)
    assert out["by_bs"]["9"]["clean"] == 1 and out["by_bs"]["10"]["windows"] == 6
    assert out["kv_usage_max"] == 0.5 and out["queue_max"] == 2
    assert out["prefill_batches"] == 2 and out["prefill_cold_batches"] == 1
    assert out["prefill_cache_hit"] == round(7936 / (7936 + 4096 + 256), 3)
    assert out["server_requests"] == 1 and out["server_ttft_p50_ms"] == round(1.39 + 8102.75, 1)
    assert 0 < out["stall_share"] < 1


def test_summarize_log_of_an_empty_slice():
    out = bench.summarize_log([], target_bs=10)
    assert out["windows"] == 0 and out["tps_clean_median"] is None and out["stall_share"] is None


# ---------------------------------------------------------------- client


def test_sse_decoder_handles_split_events():
    sse = bench.SSEDecoder()
    assert sse.feed(b'data: {"a"') == []
    assert sse.feed(b': 1}\r\n\r\ndata: [DO') == ['{"a": 1}']
    assert sse.feed(b"NE]\n\n: keep-alive\n") == ["[DONE]"]


def test_stream_events_set_ttft_usage_and_finish_reason():
    rec = bench.new_record({"stream": 0})
    bench.handle_stream_event(rec, json.dumps({"choices": [{"delta": {"role": "assistant"}}]}), 1.0)
    assert rec["t_first"] is None  # a role-only chunk carries no token
    bench.handle_stream_event(rec, json.dumps({"choices": [{"delta": {"reasoning_content": "hm"}}]}), 2.0)
    bench.handle_stream_event(rec, json.dumps({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}), 3.0)
    bench.handle_stream_event(rec, json.dumps({"choices": [], "usage": {
        "prompt_tokens": 900, "completion_tokens": 64, "prompt_tokens_details": {"cached_tokens": 800}}}), 3.1)
    bench.handle_stream_event(rec, "[DONE]", 3.2)
    assert (rec["t_first"], rec["t_last"], rec["chunks"]) == (2.0, 3.0, 2)
    assert (rec["prompt_tokens"], rec["completion_tokens"], rec["cached_tokens"]) == (900, 64, 800)
    assert rec["finish_reason"] == "length"


def test_summarize_client_prorates_tokens_to_the_phase():
    def rec(status, t_send, chunks, tokens=None, reason="length"):
        times = [t_send + 1 + i * 0.1 for i in range(chunks)]
        return {**bench.new_record({"stream": 0, "cold": False}), "status": status, "t_send": t_send,
                "t_first": times[0], "t_last": times[-1], "t_end": times[-1], "chunks": chunks, "chunk_times": times,
                "completion_tokens": tokens, "prompt_tokens": 1000, "cached_tokens": 900,
                "finish_reason": reason if status == "ok" else None}
    records = [rec("ok", 0.0, 50, 101), rec("ok", 0.0, 50, 101, "stop"), rec("cancelled", 5.0, 20)]
    out = bench.summarize_client(records, 0.0, 10.0)
    per_chunk = 202 / 100
    assert out["tok_s"] == round((202 + 20 * per_chunk) / 10.0, 1)
    assert (out["requests_ok"], out["requests_cancelled"]) == (2, 1)
    assert out["ttft_p50_s"] == 1.0
    assert out["stream_tok_s_p50"] == round(100 / 4.9, 1)
    assert out["length_rate"] == 0.5 and out["client_cache_hit"] == 0.9


FAKE_SERVER = r'''
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, log = int(sys.argv[1]), sys.argv[2]
speed = float(os.environ.get("FAKE_SPEED", "1"))
running, lock = [0], threading.Lock()

def logline(text):
    with open(log, "a") as f:
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + text + "\n")

def ticker():
    while True:
        time.sleep(0.05)
        with lock:
            bs = running[0]
        if bs:
            logline(f"Decode batch, #running-req: {bs}, #full token: 100, full token usage: 0.01, accept len: 2.50, "
                    f"accept rate: 0.50, cuda graph: True, gen throughput (token/s): {100.0 * bs * speed:.2f}, "
                    f"#queue-req: 0")

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def send_body(self, code, body, ctype="application/json"):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            self.send_body(200, "ok")
        elif self.path == "/metrics":
            self.send_body(200, 'sglang:num_running_reqs{m="x"} %d.0\nsglang:num_queue_reqs{m="x"} 0.0\n' % running[0])
        else:
            self.send_body(404, "{}")
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/start_profile":
            return self.send_body(200, "{}")
        if req["model"] != "flashnext" or req["messages"][0]["role"] != "system":
            return self.send_body(400, '{"error": "bad request"}')
        if os.environ.get("FAKE_REJECT_MIN_TOKENS") and req.get("min_tokens"):
            return self.send_body(400, '{"error": "min_tokens unsupported"}')
        logline("Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 128, full token usage: 0.01, "
                "#running-req: 1, #queue-req: 0, cuda graph: False, input throughput (token/s): 1000.00")
        n = max(req.get("min_tokens", 0), min(req["max_tokens"], 40))
        with lock:
            running[0] += 1
        try:
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                def chunk(obj):
                    data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                for _ in range(n // 2):
                    time.sleep(0.004)
                    chunk({"choices": [{"index": 0, "delta": {"reasoning_content": "ab"}, "finish_reason": None}]})
                chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
                chunk({"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": n,
                                                "prompt_tokens_details": {"cached_tokens": 400}}})
                chunk("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                time.sleep(0.004 * n / 2)
                self.send_body(200, json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                                                "usage": {"prompt_tokens": 500, "completion_tokens": n}}))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with lock:
                running[0] -= 1

threading.Thread(target=ticker, daemon=True).start()
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int) -> None:
    for _ in range(100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise RuntimeError("fake server did not start")


@pytest.fixture()
def fake_server(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    port, log = _free_port(), tmp_path / "serve.log"
    proc = subprocess.Popen([sys.executable, str(script), str(port), str(log)])
    try:
        _wait_port(port)
        yield port, log
    finally:
        proc.kill()
        proc.wait()


def _plan(**kw):
    return {**bench.BENCH_DEFAULTS, "request": dict(REQUEST), "seed": 1, "request_timeout_s": 30, **kw}


def test_run_phase_streams_against_a_server(fake_server):
    port, log = fake_server
    corpus = bench.synthetic_corpus(games=3, turns=4)
    phase = _phase(concurrency=3, duration_s=1.0, max_tokens=40)
    raw = bench.run_coro(bench.run_phase("127.0.0.1", port, "flashnext", corpus, phase, _plan(),
                                         log_offset=lambda: log.stat().st_size if log.exists() else 0))
    out = bench.summarize_client(raw["records"], raw["t0"], raw["t1"])
    assert out["requests_ok"] >= 3 and out["requests_error"] == 0
    assert out["ttft_p50_s"] is not None and out["stream_tok_s_p50"] > 0 and out["tok_s"] > 0
    assert out["finish_reasons"] == {"length": out["requests_ok"]} and out["client_cache_hit"] == 0.8
    assert raw["offsets"]["end"] > raw["offsets"]["start"]
    assert {r["stream"] for r in raw["records"]} == {0, 1, 2}
    turns = sorted(r["turn"] for r in raw["records"] if r["stream"] == 0)
    assert turns == list(range(len(turns)))


def test_run_phase_without_streaming_and_with_server_errors(fake_server):
    port, _ = fake_server
    corpus = bench.synthetic_corpus(games=2, turns=3)
    phase = _phase(concurrency=2, duration_s=0.5, max_tokens=20, stream=False)
    raw = bench.run_coro(bench.run_phase("127.0.0.1", port, "flashnext", corpus, phase, _plan()))
    out = bench.summarize_client(raw["records"], raw["t0"], raw["t1"])
    assert out["requests_ok"] >= 2 and out["finish_reasons"] == {"stop": out["requests_ok"]}
    assert out["ttft_p50_s"] is None  # not measurable without streaming
    raw = bench.run_coro(bench.run_phase("127.0.0.1", port, "wrong-model", corpus, phase, _plan()))
    assert raw["records"] and all(r["status"] == "error" and r["http"] == 400 for r in raw["records"])


# ---------------------------------------------------------------- driver, end to end

FAKE_LAUNCHER = r'''
import os, subprocess, sys
from pathlib import Path
PREFIX = @PREFIX@
LOG = str(Path(WORKING_DIR) / "serve.log")
CFG = dict(MAXREQ=10, SPEC_ACCEPT_SINGLE=1.0, SPEC_ACCEPT_ACC=1.0)
assert set(BENCH_CFG) <= set(CFG), set(BENCH_CFG) - set(CFG)
CFG.update(BENCH_CFG)
VENV = f"{PREFIX}/venv"
SGLANG_RUNTIME_PATCHES = {"01-base.py": "import sys, pathlib; pathlib.Path(sys.argv[1], 'sglang', 'base.txt').write_text('1')"}
BENCH_PREPARE_SITE_PACKAGES(VENV, PREFIX)
SGLANG_RUNTIME_PATCHES = {**SGLANG_RUNTIME_PATCHES, **BENCH_RUNTIME_PATCHES}
site = next(Path(VENV).glob("lib/python*/site-packages"))
before = sorted(p.name for p in (site / "sglang").iterdir())
for name, text in sorted(SGLANG_RUNTIME_PATCHES.items()):
    path = Path(PREFIX) / name
    path.write_text(text)
    subprocess.run([sys.executable, str(path), str(site)], check=True)
after = sorted(p.name for p in (site / "sglang").iterdir())
Path(WORKING_DIR, "sglang_files.json").write_text(__import__("json").dumps({"before": before, "after": after}))
env = dict(os.environ)
env.update(BENCH_ENV)
args = [sys.executable, @SERVER@, str(SERVED_MODEL_PORT), LOG]
args = BENCH_ARGS(args)
precache_model_thread.start()
proc = subprocess.Popen(args, env=env, start_new_session=True)
'''


def test_run_bench_end_to_end(tmp_path):
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_SERVER)
    prefix = tmp_path / "prefix"
    sglang = prefix / "venv" / "lib" / "python3.12" / "site-packages" / "sglang"
    sglang.mkdir(parents=True)
    (sglang / "__init__.py").write_text("")
    launcher = FAKE_LAUNCHER.replace("@PREFIX@", repr(str(prefix))).replace("@SERVER@", repr(str(server)))
    touch = "import sys, pathlib; pathlib.Path(sys.argv[1], 'sglang', 'patched.txt').write_text('1')"
    phases = {
        "warm": {"concurrency": 2, "duration_s": 0.2, "max_tokens": 20, "report": False},
        # long enough requests that most 50 ms decode windows hold no prefill
        "clean10": {"concurrency": 3, "duration_s": 0.8, "max_tokens": 200, "min_tokens": 200},
        "mixed": {"concurrency": 2, "duration_s": 0.3, "max_tokens": 20, "stream": False, "cold_fraction": 0.5},
    }
    spec = {
        "defaults": {"default_phases": ["warm", "clean10", "mixed"], "est_startup_s": 1, "seed": 1,
                     "request_timeout_s": 30, "startup_timeout_s": 30},
        "request": REQUEST,
        "phases": phases,
        "configs": {
            "A": {},
            "B": {"cfg": {"SPEC_ACCEPT_SINGLE": 0.5}, "env": {"FAKE_SPEED": "2"}, "runtime_patches": ["touch.py"]},
            "C": {"compose": [["A", "B"]], "args_add": ["--extra"]},
            "D": {"enabled": False, "disabled_reason": "not written"},
            "E": {"cfg": {"NO_SUCH_KEY": 1}},
            "F": {"env": {"FAKE_REJECT_MIN_TOKENS": "1"}, "phases": ["clean10"]},
        },
        "benches": {"t": {"configs": ["A", "B", "C", "D", "E", "F"], "budget_min": 30}},
    }
    plan = bench.resolve_plan(spec, "t")
    plan.pop("patches_needed")
    plan["patches"] = {"touch.py": touch}
    nb = {"MODEL_DIR": "m", "DRAFT_MODEL_DIR": "d", "WHEELHOUSE_DIR": "w", "SERVED_MODEL_NAME": "flashnext",
          "SERVED_MODEL_PORT": _free_port(), "NOTEBOOK_START_TIME": time.time()}
    results = bench.run_bench(plan, launcher=launcher, corpus=None, nb=nb, working_dir=tmp_path)

    status = {c["name"]: c["status"] for c in results["configs"]}
    assert status == {"A": "ok", "B": "ok", "C": "ok", "D": "disabled", "E": "failed", "F": "ok"}
    by = {c["name"]: c for c in results["configs"]}
    assert "unknown" in by["E"]["reason"] or "NO_SUCH_KEY" in by["E"]["reason"]
    # C composes the faster of A and B, then adds its own argument
    assert by["C"]["composed_from"] == ["B"]
    assert by["C"]["effective_cfg"]["SPEC_ACCEPT_SINGLE"] == 0.5 and by["C"]["launch_args"][-1] == "--extra"
    # each config starts from the pristine install: base patch always, touch.py only where configured
    files = {n: json.loads((tmp_path / "bench" / n / "sglang_files.json").read_text()) for n in "ABC"}
    assert all(f["before"] == ["__init__.py"] for f in files.values())
    assert files["A"]["after"] == ["__init__.py", "base.txt"]
    assert files["B"]["after"] == files["C"]["after"] == ["__init__.py", "base.txt", "patched.txt"]
    # measured phases, written to disk
    a_clean = next(p for p in by["A"]["phases"] if p["name"] == "clean10")
    assert a_clean["log"]["windows"] > 0 and a_clean["client"]["requests_ok"] > 0
    assert by["A"]["preflight"]["status"] == "ok"
    saved = json.loads((tmp_path / "bench_results.json").read_text())
    assert [c["name"] for c in saved["configs"]] == ["A", "B", "C", "D", "E", "F"]
    # a server that rejects min_tokens is still measured, without it
    assert "min_tokens rejected" in by["F"]["warning"] and by["F"]["preflight"]["min_tokens"] == 0
    f_clean = by["F"]["phases"][0]
    assert f_clean["min_tokens"] == 0 and f_clean["client"]["requests_ok"] > 0
    md = (tmp_path / "bench_results.md").read_text()
    assert "| B | clean10 | 3 |" in md and "| warm |" not in md and "| D | disabled |" in md


def test_run_bench_skips_and_drops_on_budget(tmp_path):
    spec = {
        "defaults": {"default_phases": ["p"], "est_startup_s": 600},
        "phases": {"p": {"duration_s": 600}},
        "configs": {"A": {}, "B": {}, "C": {}},
        "benches": {"t": {"configs": ["A", "B", "C"], "budget_min": 15, "drop_order": ["B"]}},
    }
    plan = bench.resolve_plan(spec, "t")
    plan["patches"] = {}
    nb = {"SERVED_MODEL_NAME": "x", "SERVED_MODEL_PORT": _free_port(), "NOTEBOOK_START_TIME": time.time()}
    results = bench.run_bench(plan, launcher="", corpus=None, nb=nb, working_dir=tmp_path)
    # nothing fits in 15 min (each needs ~20): B is dropped first, the others skipped
    assert {c["name"]: c["status"] for c in results["configs"]} == {"A": "skipped", "B": "dropped", "C": "skipped"}


# ---------------------------------------------------------------- build


builder = _load("build_notebook", REPO / "kaggle" / "build_notebook.py")


@pytest.fixture(scope="module")
def template():
    return json.loads((REPO / "kaggle" / "template.ipynb").read_text())


def _arm_notebook(template):
    return builder.build_notebook(template, "diff --git a/x b/x\n", env={}, demo_games="franzen", demo_minutes=None)


def test_dry_build_of_the_levers_bench(template):
    plan = builder.bench_plan("levers")
    assert set(plan["patches"]) == {"m64.py", "nosplitk.py"}
    nb = builder.apply_bench(_arm_notebook(template), plan, builder.BENCH_CORPUS.read_bytes())
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    assert not any("await bm.run(" in c for c in code)  # no games
    assert sum("bench.run_bench(" in c for c in code) == 1
    source = json.dumps(nb, indent=1) + "\n"
    assert len(source.encode()) < builder.KERNEL_SOURCE_LIMIT
    setup = next(c for c in code if "serving_bench" in c)
    ns: dict = {}
    exec(setup, ns)  # decodes the blob, loads the module, prints the plan
    launcher = ns["BENCH_LAUNCHER"]
    compile(launcher, "launcher", "exec")
    hooks = ["CFG.update(BENCH_CFG)", "install_marker =", "BENCH_PREPARE_SITE_PACKAGES(VENV, PREFIX)",
             "if SGLANG_RUNTIME_PATCHES:", "env.update(BENCH_ENV)", "args = BENCH_ARGS(args)",
             "precache_model_thread.start()"]
    positions = [launcher.index(h) for h in hooks]
    assert positions == sorted(positions)
    assert ns["BENCH_PLAN"]["configs"][0]["name"] == "C0"
    assert json.loads(gzip.decompress(ns["BENCH_CORPUS"]))["games"]


def test_bench_build_leaves_arm_notebooks_alone(template):
    nb = _arm_notebook(template)
    assert not any("BENCH_" in c["source"] for c in nb["cells"] if c["cell_type"] == "code")


def test_bench_build_rejects_unknown_cfg_keys(template):
    plan = builder.bench_plan("smoke")
    plan["configs"][0]["overrides"]["cfg"] = {"MAXREQS": 12}
    with pytest.raises(SystemExit, match="MAXREQS"):
        builder.apply_bench(_arm_notebook(template), plan, b"")


def test_benches_are_unique_but_outside_arm_batches():
    runs = [{"arm": "control", "fingerprint": "f1", "code": "c1", "commit": "abc", "batch_open": True}]
    builder._check_unique("bench-levers", "f2", "c2", runs, new_batch=False, bench=True)  # other code: fine
    with pytest.raises(SystemExit, match="open batch"):
        builder._check_unique("panel-x", "f2", "c2", runs, new_batch=False)
    runs.append({"arm": "bench-levers", "fingerprint": "f2", "code": "c2", "commit": "abc", "batch_open": False})
    with pytest.raises(SystemExit, match="duplicate"):
        builder._check_unique("bench-levers", "f2", "c2", runs, new_batch=False, bench=True)


# ---------------------------------------------------------------- corpus maker


def test_corpus_maker_rebuilds_harness_messages():
    maker = _load("make_bench_corpus", REPO / "kaggle" / "make_bench_corpus.py")
    call = {"id": "call_1", "index": 0, "type": "function",
            "function": {"name": "python", "arguments": "{\"code\": \"print(1)\"}"}}
    text = "\n".join([
        "--- analysis_step=1 | action=1 | 01:58:04 | tool-agent ---",
        "[SYSTEM PROMPT]", "You are a coding agent.", "",
        "[USER PROMPT]", "Current state: step 1.", "", "Current grid image:", "", "[image attached]", "",
        "[MODEL RESPONSE META]", "finish_reason: tool_calls", "raw_tool_calls:", json.dumps([call], indent=2), "",
        "[THINKING]", "Let me look.", "",
        "[TOOL CALL: python]", "<tool_call>...</tool_call>", "",
        "[TOOL RESULT: python]", "1", "",
        "[MODEL RESPONSE META]", "finish_reason: stop", "raw_tool_calls:", "[]", "",
        "[THINKING]", "Done.",
    ])
    system, messages = maker.conversation(text)
    assert system == "You are a coding agent."
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[0]["content"] == "Current state: step 1.\n\nCurrent grid image:"
    assert messages[1]["reasoning_content"] == "Let me look."
    assert messages[1]["tool_calls"] == [{"id": "call_1", "type": "function", "function": call["function"]}]
    assert messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "1"}
    assert "tool_calls" not in messages[3]
    assert maker.truncate(messages, 10_000) == messages[:3]  # ends where the model speaks next
