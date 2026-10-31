"""Steady-state rollout throughput of one vLLM server at a fixed concurrency C, from recorded GRPO episodes.

C slots each replay an episode request after request, as GRPO sends them: token prompts asking for exactly the
recorded reply length (times --gen-scale) with GRPO sampling and logprobs on, where each prompt is the previous
prompt + the ids the server just generated + the recorded observation that followed, so only the observation is
new to the prefix cache. A slot whose episode ends (or would pass the context limit) starts the next one. Every
episode starts at its first request with >= --start-tokens of context, so decoding runs at GRPO lengths. Slots
beyond the number of episodes reuse them with one token changed after the first prompt (they share the system
prompt and nothing more, like another pass of the game). With --dp N, slot i always goes to data-parallel engine
i % N (X-data-parallel-rank), so an episode keeps its prefix cache, as a multi-URL loop would. Tool and game time
are left out: the server is the only clock. The recordings come from waves with a 120 s request timeout, so
replies are shorter than GRPO's will be with 900 s; --gen-scale lengthens them.

The window opens once every slot has finished its first request (the opening prefill) and lasts --window
seconds; tokens/s come from the server's /metrics counters over the window. Writes OUT/requests.jsonl,
OUT/metrics.jsonl and OUT/summary.json, then exits, which drops the requests still in flight.

    python3 bench.py --workload workload.jsonl --concurrency 80 --out DIR [--url http://127.0.0.1:8000/v1]

Copied from Frida bench_b200/bench.py (GRPO rollout bench, 10-28). Changes for the drafter comparison: no
logprobs (the harness requests none), raw /metrics saved at the window edges, and accept_per_pos in the
summary (accepted tokens at each draft position / drafts).
"""

import argparse
import json
import os
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path

SAMPLE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
COUNTERS = ["generation_tokens_total", "prompt_tokens_total", "num_preemptions_total",
            "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total", "prefix_cache_queries_total", "prefix_cache_hits_total"]


def scrape(root):
    """{metric: {"sum": total over label sets, "max": largest, "n": label sets}} from /metrics."""
    out = {}
    with urllib.request.urlopen(root + "/metrics", timeout=20) as response:
        for line in response.read().decode().splitlines():
            m = SAMPLE.match(line)
            if not m:
                continue
            name, value = m.group(1)[5:], float(m.group(3))
            if name.endswith("_created"):
                continue
            entry = out.setdefault(name, {"sum": 0.0, "max": float("-inf"), "n": 0})
            entry["sum"] += value
            entry["max"] = max(entry["max"], value)
            entry["n"] += 1
    return out


POS = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)$')


def per_position(raw0, raw1, drafts):
    """Acceptance rate at each draft position over the window (accepted at position i / drafts)."""
    def read(raw):
        values = {}
        for line in raw.splitlines():
            m = POS.match(line)
            if m:
                values[int(m.group(1))] = values.get(int(m.group(1)), 0.0) + float(m.group(2))
        return values
    a, b = read(raw0), read(raw1)
    return [round((b[i] - a.get(i, 0.0)) / drafts, 4) for i in sorted(b)] if drafts else None


def get(sample, name, key="sum"):
    return (sample.get(name) or {}).get(key, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--start-tokens", type=int, default=40000)
    ap.add_argument("--min-after", type=int, default=20000)
    ap.add_argument("--window", type=float, default=300)
    ap.add_argument("--warmup-max", type=float, default=900)
    ap.add_argument("--label", default="")
    ap.add_argument("--dp", type=int, default=0, help="sticky routing over N data-parallel engines")
    ap.add_argument("--gen-scale", type=float, default=1.0)
    ap.add_argument("--max-tokens-cap", type=int, default=14336)  # the harness's max_output
    ap.add_argument("--max-context", type=int, default=147072)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = args.url[:-3] if args.url.rstrip("/").endswith("/v1") else args.url.rstrip("/")

    episodes = []
    for line in open(args.workload):
        e = json.loads(line)
        start = next((k for k, (plen, _) in enumerate(e["schedule"]) if plen >= args.start_tokens), None)
        if start is not None and len(e["ids"]) - e["schedule"][start][0] >= args.min_after:
            e["start"] = start
            episodes.append(e)
    if not episodes:
        raise SystemExit(f"no episode reaches {args.start_tokens} tokens with {args.min_after} after it")

    lock = threading.Lock()
    rows, first_done = [], set()
    stop = threading.Event()
    log = open(out / "requests.jsonl", "w")

    def prompt(e, copy, length):
        ids = e["ids"][:length]
        if copy:  # diverge right after the first prompt: shares the system prompt only
            at = min(e["schedule"][0][0] - 1, length - 1)
            ids = list(ids)
            ids[at] = e["ids"][(at + 997 * copy) % len(e["ids"])]
            if ids[at] == e["ids"][at]:
                ids[at] = e["ids"][(at + 1) % len(e["ids"])]
        return ids

    def slot(index):
        n = index
        headers = {"Content-Type": "application/json"}
        if args.dp:
            headers["X-data-parallel-rank"] = str(index % args.dp)
        while not stop.is_set():
            e, copy = episodes[n % len(episodes)], n // len(episodes)
            schedule, ids = e["schedule"], None
            for k in range(e["start"], len(schedule)):
                if stop.is_set():
                    return
                length, recorded = schedule[k]
                if ids is None:
                    ids = prompt(e, copy, length)
                gen = min(max(1, round(recorded * args.gen_scale)), args.max_tokens_cap)
                if len(ids) + gen > args.max_context:
                    break
                body = {"model": args.model, "prompt": ids, "max_tokens": gen, "min_tokens": gen,
                        "ignore_eos": True, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
                        "return_token_ids": True}
                request = urllib.request.Request(args.url + "/completions", data=json.dumps(body).encode(),
                                                 headers=headers)
                t0 = time.time()
                try:
                    with urllib.request.urlopen(request, timeout=3600) as response:
                        out = json.loads(response.read())
                    usage, error = out.get("usage") or {}, None
                    generated = out["choices"][0].get("token_ids") or []
                except Exception as exc:  # noqa: BLE001 - recorded, the slot goes on
                    usage, error, generated = {}, repr(exc)[:300], []
                row = {"slot": index, "episode": e["id"], "copy": copy, "k": k, "t0": t0, "t1": time.time(),
                       "prompt": len(ids), "gen": usage.get("completion_tokens"),
                       "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "error": error}
                with lock:
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    first_done.add(index)
                if error or not generated:
                    time.sleep(5)
                    ids = None  # start over from the recording at the next request
                    continue
                if k + 1 < len(schedule):  # the next prompt: this one + the generated ids + the next observation
                    observed = e["ids"][length + recorded:schedule[k + 1][0]] \
                        if schedule[k + 1][0] >= length + recorded else None
                    ids = ids + generated + observed if observed is not None else None
            n += args.concurrency

    samples = []

    def sampler():
        with open(out / "metrics.jsonl", "w") as f:
            while not stop.is_set():
                try:
                    s = scrape(root)
                    s["t"] = time.time()
                    samples.append(s)
                    f.write(json.dumps({"t": s["t"], **{k: v["sum"] for k, v in s.items() if k != "t"},
                                        "kv_max": get(s, "kv_cache_usage_perc", "max")
                                        or get(s, "gpu_cache_usage_perc", "max")}) + "\n")
                    f.flush()
                except Exception:  # noqa: BLE001
                    pass
                stop.wait(5)

    began = time.time()
    threading.Thread(target=sampler, daemon=True).start()
    for i in range(args.concurrency):
        threading.Thread(target=slot, args=(i,), daemon=True).start()
    while len(first_done) < args.concurrency and time.time() - began < args.warmup_max:
        time.sleep(2)
    opened = time.time()
    s0 = scrape(root)
    raw0 = urllib.request.urlopen(root + "/metrics", timeout=20).read().decode()
    time.sleep(args.window)
    s1 = scrape(root)
    raw1 = urllib.request.urlopen(root + "/metrics", timeout=20).read().decode()
    (out / "metrics_window_start.txt").write_text(raw0)
    (out / "metrics_window_end.txt").write_text(raw1)
    closed = time.time()
    stop.set()

    span = closed - opened
    d = {c: get(s1, c) - get(s0, c) for c in COUNTERS}
    window = [s for s in samples if opened <= s["t"] <= closed]
    running = [get(s, "num_requests_running") for s in window]
    waiting = [get(s, "num_requests_waiting") for s in window]
    kv_name = "kv_cache_usage_perc" if "kv_cache_usage_perc" in s1 else "gpu_cache_usage_perc"
    kv = [get(s, kv_name, "max") for s in window]
    done = [r for r in rows if r["gen"] and opened <= r["t0"] and r["t1"] <= closed]
    per_request = [r["gen"] / (r["t1"] - r["t0"]) for r in done if r["t1"] > r["t0"]]
    contexts = [r["prompt"] for r in rows if opened <= r["t0"] <= closed]
    summary = {
        "label": args.label, "concurrency": args.concurrency, "episodes": len(episodes), "dp": args.dp,
        "gen_scale": args.gen_scale,
        "cached_share": round(sum(r["cached"] or 0 for r in done) / max(1, sum(r["prompt"] for r in done)), 3),
        "start_tokens": args.start_tokens, "warmup_s": round(opened - began, 1), "window_s": round(span, 1),
        "warmup_complete": len(first_done) >= args.concurrency,
        "gen_tok_s": round(d["generation_tokens_total"] / span, 1),
        "prompt_tok_s": round(d["prompt_tokens_total"] / span, 1),
        "gen_tok_s_per_slot": round(d["generation_tokens_total"] / span / args.concurrency, 2),
        "running_mean": round(statistics.mean(running), 1) if running else None,
        "waiting_max": max(waiting) if waiting else None,
        "kv_usage_max_engine": round(max(kv), 4) if kv else None,
        "kv_usage_end": round(kv[-1], 4) if kv else None,
        "preemptions": d["num_preemptions_total"],
        "mtp_tokens_per_step": round(1 + d["spec_decode_num_accepted_tokens_total"]
                                     / d["spec_decode_num_drafts_total"], 3) if d["spec_decode_num_drafts_total"] else None,
        "prefix_hit_rate": round(d["prefix_cache_hits_total"] / d["prefix_cache_queries_total"], 3)
        if d["prefix_cache_queries_total"] else None,
        "requests_in_window": len(done), "request_tok_s_median": round(statistics.median(per_request), 1)
        if per_request else None,
        "context_median": int(statistics.median(contexts)) if contexts else None,
        "errors": sum(1 for r in rows if r["error"]),
        "accept_per_pos": per_position(raw0, raw1, d["spec_decode_num_drafts_total"]),
        "engines": int((s1.get("num_requests_running") or {}).get("n", 0)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    log.close()
    os._exit(0)  # drop the requests still in flight: the server aborts them when the connections close


if __name__ == "__main__":
    main()
