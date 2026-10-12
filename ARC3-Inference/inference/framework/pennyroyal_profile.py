"""Decode profile of a running Pennyroyal server: fixed fake workload, kernel times.

For each stream count, the launcher (kaggle.profile) starts that many identical-
length requests, waits until all are decoding, measures unprofiled throughput for
a fixed window, then records `num_steps` scheduler steps with SGLang's built-in
torch profiler (`/start_profile`). Every trace is summarized into kernel time per
step, grouped by name and by rough category, in `profile-summary.json`. Traces
stay in `profile/` for a closer look in Perfetto.
"""
import gzip
import json
import re
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# Rough buckets for a first read of the receipt; unmatched kernels stay "other".
CATEGORIES = (
    ("experts", r"moe|marlin|cutlass.*(fp4|group)|grouped|expert|topk|router"),
    ("act_quant", r"quant|per_token|scaled_fp4|fp4_quant|cvt_fp16_to_fp4"),
    ("dense_gemm", r"gemm|gemv|nvjet|cublas|cutlass|sm120_lowm|scaled_mm|matmul|splitk"),
    ("attention", r"attn|attention|flash|qsa|mla|paged|decode_kernel"),
    ("linear_attn", r"gdn|delta|recurrent|chunk|conv1d|causal_conv|mamba|ssm"),
    ("norm_act", r"norm|silu|gelu|act_and_mul|swiglu|rope|rotary|softmax"),
    ("sampling", r"sample|argmax|top_p|top_k|logit|vocab"),
    ("memory", r"memcpy|memset|copy|cat|index|gather|scatter|fill"),
)
PASSAGE = (
    "The grid is a small world of colored cells. Each move changes something, and the "
    "goal is to find out what. Careful play means noticing which cells moved, which "
    "stayed, and what rule could explain both. "
)


class _Stream(threading.Thread):
    """One streaming /generate request; tracks generated tokens until stopped."""

    def __init__(self, url: str, index: int, prompt_tokens: int, max_new_tokens: int):
        super().__init__(daemon=True)
        prefix = f"Request {index}. Notes for game {index * 7919 % 1000}: "
        text = prefix + PASSAGE * (prompt_tokens * 4 // len(PASSAGE) + 1)
        self.payload = json.dumps({
            "text": text[: prompt_tokens * 4],
            "stream": True,
            "sampling_params": {"max_new_tokens": max_new_tokens, "ignore_eos": True, "temperature": 0.7},
        }).encode()
        self.url = url
        self.tokens = 0
        self.first_token_at = None
        self.error = None
        self.stop = threading.Event()

    def run(self) -> None:
        req = urllib.request.Request(self.url, data=self.payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                for raw in resp:
                    if self.stop.is_set():
                        break  # closing the connection aborts the request server-side
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    meta = json.loads(line[5:]).get("meta_info", {})
                    self.tokens = int(meta.get("completion_tokens", self.tokens))
                    if self.first_token_at is None:
                        self.first_token_at = time.time()
        except Exception as exc:  # noqa: BLE001 - recorded in the summary
            if not self.stop.is_set():
                self.error = repr(exc)


def summarize_trace(path: Path, num_steps: int) -> dict:
    """Kernel time per step from a torch chrome trace, by kernel name and category."""
    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as f:
        events = json.load(f).get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    graph_launches = sum(1 for e in events if e.get("name") == "cudaGraphLaunch")
    if not kernels:
        return {"kernels": 0, "graph_launches": graph_launches}
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e.get("dur", 0) for e in kernels)
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    by_cat: dict[str, float] = defaultdict(float)
    for e in kernels:
        name, dur = e.get("name", "?"), float(e.get("dur", 0))
        by_name[name][0] += dur
        by_name[name][1] += 1
        low = name.lower()
        cat = "memory" if e.get("cat") != "kernel" else next((c for c, p in CATEGORIES if re.search(p, low)), "other")
        by_cat[cat] += dur
    busy = sum(v[0] for v in by_name.values())
    per_step = lambda us: round(us / max(num_steps, 1) / 1000, 3)  # noqa: E731 - ms per step
    top = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:60]
    return {
        "kernels": len(kernels),
        "graph_launches": graph_launches,
        "span_ms_per_step": per_step(end - start),
        "gpu_busy_ms_per_step": per_step(busy),
        "idle_fraction": round(1 - busy / max(end - start, 1), 3),
        "category_ms_per_step": {c: per_step(v) for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "top_kernels": [{"name": n[:200], "ms_per_step": per_step(t), "calls_per_step": round(c / max(num_steps, 1), 2)}
                        for n, (t, c) in top],
    }


def _post(url: str, payload: dict, timeout: float = 60) -> str:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def run_profile(base: str, out_dir: Path, *, streams=(1, 4, 10, 16), prompt_tokens=8192, num_steps=40,
                measure_seconds=10.0, max_new_tokens=16384, first_token_timeout=900.0) -> dict:
    """Profile each stream count in turn; returns (and writes) the summary."""
    trace_dir = out_dir / "profile"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary = {"settings": {"streams": list(streams), "prompt_tokens": prompt_tokens, "num_steps": num_steps,
                            "measure_seconds": measure_seconds}, "runs": []}
    for count in streams:
        row = {"streams": count, "started_at": time.time()}
        workers = [_Stream(f"{base}/generate", i, prompt_tokens, max_new_tokens) for i in range(count)]
        for w in workers:
            w.start()
        deadline = time.time() + first_token_timeout
        while time.time() < deadline and not all(w.first_token_at or w.error for w in workers):
            time.sleep(0.5)
        time.sleep(3)  # let the last prefill chunk clear
        t0, n0 = time.time(), sum(w.tokens for w in workers)
        time.sleep(measure_seconds)
        t1, n1 = time.time(), sum(w.tokens for w in workers)
        row["decode_tok_s"] = round((n1 - n0) / (t1 - t0), 1)
        row["decode_tok_s_per_stream"] = round(row["decode_tok_s"] / count, 1)
        before = set(trace_dir.glob("*.trace.json*"))
        prefix = f"streams{count:02d}-"
        row["profile_window_start"] = time.time()
        try:
            _post(f"{base}/start_profile", {"output_dir": str(trace_dir), "num_steps": num_steps,
                                            "activities": ["CPU", "GPU"], "profile_prefix": prefix})
        except Exception as exc:  # noqa: BLE001
            row["profile_error"] = repr(exc)
        trace = None
        wait_until = time.time() + 300
        while trace is None and time.time() < wait_until and "profile_error" not in row:
            new = sorted(set(trace_dir.glob("*.trace.json*")) - before)
            trace = new[0] if new else None
            time.sleep(1)
        row["profile_window_end"] = time.time()
        for w in workers:
            w.stop.set()
        row["errors"] = [w.error for w in workers if w.error]
        if trace is not None:
            time.sleep(2)  # the exporter may still be flushing
            row["trace"] = trace.name
            try:
                row.update(summarize_trace(trace, num_steps))
            except Exception as exc:  # noqa: BLE001
                row["summary_error"] = repr(exc)
        summary["runs"].append(row)
        (out_dir / "profile-summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[profile] streams={count} decode {row['decode_tok_s']} tok/s "
              f"step {row.get('span_ms_per_step')} ms busy {row.get('gpu_busy_ms_per_step')} ms", flush=True)
        for w in workers:
            w.join(timeout=10)
        time.sleep(5)  # aborted requests drain before the next stream count
    return summary
