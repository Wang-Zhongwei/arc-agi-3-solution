"""One GRPO rollout step against a vLLM server: C replay games at once, each to its end.

Every game of the step starts together (as a GRPO step does) and sends its requests one after another:
the untrimmed context of the recorded game, asking for exactly the recorded reply length (min_tokens =
max_tokens), with the harness's sampling, thinking budget and per-token logprobs (GRPO needs them).
Tool and game time are left out: the server is the only clock. The step ends when the last game ends.

Writes OUT/requests.jsonl (one row per request), OUT/metrics.jsonl (server /metrics every 10 s) and
OUT/summary.json. The games come from build_replay.py: --games picks the games, --copies how many
recorded attempts of each (different runs, so no two rollouts share more than the system prompt).

    python bench_rollout.py --replay replay.jsonl --out DIR --games cn04,sc25 --copies 8 [--url ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import aiohttp

METRICS = re.compile(r"^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc"
                     r"|num_preemptions_total|prompt_tokens_total|generation_tokens_total"
                     r"|spec_decode_num_drafts_total|spec_decode_num_draft_tokens_total"
                     r"|spec_decode_num_accepted_tokens_total|prefix_cache_queries_total|prefix_cache_hits_total)"
                     r"(\{[^}]*\})? (\S+)$")


def pick(replay: Path, games: list[str], copies: int, seed: int) -> list[dict]:
    by_game = defaultdict(list)
    for line in open(replay):
        g = json.loads(line)
        if g["game"] in games:
            by_game[g["game"]].append(g)
    rng = random.Random(seed)
    chosen = []
    for game in games:
        pool = sorted(by_game[game], key=lambda g: g["id"])
        if len(pool) < copies:
            raise SystemExit(f"{game}: {len(pool)} recorded attempts, {copies} asked")
        chosen += rng.sample(pool, copies)
    return chosen


def messages_for(game: dict, end: int) -> list[dict]:
    out = [{"role": "system", "content": game["system"]}]
    for m in game["stream"][:end]:
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    sha = p["image_url"]["url"].removeprefix("img:")
                    p = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{game['images'][sha]}"}}
                parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out


async def play(session, args, slot: int, game: dict, rows: list, t0: float) -> None:
    for k, (end, max_tokens, est_context) in enumerate(game["requests"]):
        max_tokens = min(max_tokens, args.max_reply)
        payload = {
            "model": args.model, "messages": messages_for(game, end), "tools": game["tools"],
            "max_tokens": max_tokens, "min_tokens": max_tokens,
            "temperature": 0.7, "top_p": 0.95, "top_k": 20,
            "thinking_token_budget": args.thinking_budget,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if args.logprobs:
            payload["logprobs"] = True
        start = time.monotonic()
        error = None
        usage = {}
        for attempt in range(3):
            try:
                async with session.post(f"{args.url}/chat/completions", json=payload) as r:
                    body = await r.json(content_type=None)
                    if r.status != 200:
                        raise RuntimeError(f"HTTP {r.status}: {str(body)[:300]}")
                usage = body.get("usage") or {}
                error = None
                break
            except Exception as e:  # noqa: BLE001 - recorded, retried, then the game goes on
                error = repr(e)[:300]
                await asyncio.sleep(2)
        rows.append({"slot": slot, "game": game["game"], "id": game["id"], "k": k, "est_context": est_context,
                     "asked": max_tokens, "start": start - t0, "end": time.monotonic() - t0,
                     "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                     "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "error": error})


async def sample_metrics(session, args, out: Path, t0: float, done: asyncio.Event) -> None:
    """Server /metrics every 10 s, plus one sample after the step so the counters span it."""
    base = args.url.rsplit("/v1", 1)[0]
    with open(out, "w") as f:
        while True:
            finished = done.is_set()
            try:
                async with session.get(f"{base}/metrics") as r:
                    text = await r.text()
                values = defaultdict(float)
                for line in text.splitlines():
                    if (m := METRICS.match(line)):
                        values[m[1]] += float(m[3])
                f.write(json.dumps({"t": round(time.monotonic() - t0, 1), **values}) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"t": round(time.monotonic() - t0, 1), "error": repr(e)[:200]}) + "\n")
            if finished:
                return
            try:
                await asyncio.wait_for(done.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass


def summarize(rows: list[dict], metrics_path: Path, wall: float, label: str, c: int) -> dict:
    finished = defaultdict(float)
    for r in rows:
        finished[r["slot"]] = max(finished[r["slot"]], r["end"])
    ends = sorted(finished.values())
    generated = sum(r["completion_tokens"] or 0 for r in rows)
    prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    cached = sum(r["cached_tokens"] or 0 for r in rows)
    metrics = [json.loads(line) for line in open(metrics_path)]
    metrics = [m for m in metrics if "error" not in m]
    spec = {}
    if len(metrics) >= 2 and metrics[-1].get("spec_decode_num_drafts_total"):
        a, b = metrics[0], metrics[-1]
        drafts = b["spec_decode_num_drafts_total"] - a.get("spec_decode_num_drafts_total", 0)
        accepted = b["spec_decode_num_accepted_tokens_total"] - a.get("spec_decode_num_accepted_tokens_total", 0)
        spec = {"tokens_per_step": round(1 + accepted / max(drafts, 1), 3)}
    kv = [m.get("kv_cache_usage_perc", m.get("gpu_cache_usage_perc", 0)) for m in metrics]
    half = ends[len(ends) // 2] if ends else 0
    gen_by_half = sum(r["completion_tokens"] or 0 for r in rows if r["end"] <= half)
    return {
        "label": label, "concurrency": c, "games": len(finished), "requests": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        # The replay sends each turn's recorded reply, not the one just generated, so the cache misses from
        # that reply on: uncached prompt tokens overstate GRPO's prefill by about the generated tokens.
        "wall_s": round(wall, 1), "generated": generated, "prompt": prompt, "cached": cached,
        "uncached_prompt": prompt - cached,
        "tok_s": round(generated / wall, 1) if wall else None,
        "tok_s_until_half_done": round(gen_by_half / half, 1) if half else None,
        "per_game_tok_s_median": round(sorted((sum(r["completion_tokens"] or 0 for r in rows if r["slot"] == s) / finished[s])
                                              for s in finished)[len(finished) // 2], 1) if finished else None,
        "game_end_s": {"p10": ends[len(ends) // 10] if ends else None, "p50": half, "max": ends[-1] if ends else None},
        "kv_usage_max": max(kv) if kv else None, "running_max": max((m.get("num_requests_running", 0) for m in metrics), default=None),
        "waiting_max": max((m.get("num_requests_waiting", 0) for m in metrics), default=None),
        "preemptions": (metrics[-1].get("num_preemptions_total", 0) - metrics[0].get("num_preemptions_total", 0)) if metrics else None,
        **spec,
    }


async def main_async(args) -> dict:
    games = pick(Path(args.replay), args.games.split(","), args.copies, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        done = asyncio.Event()
        t0 = time.monotonic()
        sampler = asyncio.create_task(sample_metrics(session, args, out / "metrics.jsonl", t0, done))
        plays = [asyncio.create_task(play(session, args, i, g, rows, t0)) for i, g in enumerate(games)]
        try:
            await asyncio.wait_for(asyncio.gather(*plays), timeout=args.step_timeout)
        except asyncio.TimeoutError:
            for p in plays:
                p.cancel()
        wall = time.monotonic() - t0
        done.set()
        await sampler
    with open(out / "requests.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: (r["slot"], r["k"])):
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows, out / "metrics.jsonl", wall, args.label, len(games))
    summary["planned_generated"] = sum(min(x[1], args.max_reply) for g in games for x in g["requests"])
    summary["complete"] = len(rows) == sum(len(g["requests"]) for g in games)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--label", default="")
    ap.add_argument("--thinking-budget", type=int, default=10240)
    ap.add_argument("--max-reply", type=int, default=12288)  # LOCAL_ANALYZER_MAX_OUTPUT
    ap.add_argument("--logprobs", type=int, default=1)
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--step-timeout", type=float, default=2700)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
