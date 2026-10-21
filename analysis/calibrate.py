"""Calibration dataset for the v2 scheduling simulator, from real Kaggle run outputs.

Reads each run directory under the kaggle-outputs root and writes small CSV/JSON
tables to analysis/calibration/ (see calibration/README.md for every column):

  runs.csv             run metadata, score check, startup and deadline timing
  levels.csv           (a) one row per (run, game, attempted level)
  levels_pooled.csv    (a) per (game, level) pooled over Franzen-scheduler full runs
  games.csv            (b) per (run, game) timeline summary (first token, active time)
  intervals.csv        (b) per (run, game) slot-holding intervals
  requests.csv         (c) per request, runs whose requests.jsonl attributes requests to games
  reqstats.csv         (c) every SGLang ReqTimeStats line (all runs with serve.log)
  serve_decode.csv     (c) decode throughput by #running-req, per run
  serve_minutes.csv    (c) per-minute serving time series, per run
  segments.csv         (d) context segments between trims, per (run, game)
  summary.json         key medians/percentiles (the data card) and validation checks

Three timeline methods, best available per run:
  requests    per-game requests.jsonl matched to serve.log ReqTimeStats on
              (input_len, cached_input_len) and nearest output_len: exact
              per-request entry/prefill/decode times.
  transcript  per-turn start times (1 s) from transcripts/ or artifacts events;
              trims from the history_messages drop; a trim turn whose duration
              exceeds its estimated work by >= SWITCH_WAIT_S is a switch-out.
  history     action timestamps only; an action gap exceeding its token work by
              >= HISTORY_WAIT_S is a switch-out (no trim information).

Usage (stdlib only):
    nice -n 10 python3 analysis/calibrate.py [--root KAGGLE_OUTPUTS] [--out DIR]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_ROOT = Path("/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs")
HERE = Path(__file__).resolve().parent

# A trim request (or trim turn) that waited at least this long beyond its own work lost the
# slot (switch-out). In fz-ledger-on-trim-r2 the exact trim gaps split at 23 s / 114 s.
SWITCH_WAIT_S = 60.0
# History-only runs have no trim marker, so a looser bar on the action gap.
HISTORY_WAIT_S = 120.0
# Generated tokens per transcript character (reasoning + content + tool-call JSON), measured
# on fz-ledger-on-trim-r2 where both exist: 0.32 (p10 0.27, p90 0.38). Only used to split an
# action's exact token count across the turns that produced it.
TOKENS_PER_CHAR = 0.32
# Fixed per-request client overhead used in work estimates (median non-trim gap, r2).
REQUEST_OVERHEAD_S = 2.0
# A game re-admitted less than this before the deadline counts as never re-admitted.
TERMINAL_SLACK_S = 120.0
# A trim is due after a request when prompt + completion reach this share of the budget
# (the next prompt also adds a tool result <= 3072 tokens and a new frame).
TRIM_DUE_FRACTION = 0.92
# Re-prefill of a trimmed ~55k-token prompt (median prefill of trim requests, r2).
REPREFILL_S = 5.0


@dataclass
class Run:
    name: str
    kind: str            # "full" (25 games, 121 min) or "panel" (10 games, 20-25 min)
    scheduler: str       # franzen | ucb | dprime | easy-first
    ledger: int          # ARC3_LEDGER_ON_TRIM (fz) / stack ledger (v4)
    slots: int           # ARC3_MAX_ACTIVE_STREAMS
    ctx_window: int      # LOCAL_ANALYZER_CONTEXT_WINDOW
    drain: int           # ARC3_CONTEXT_DRAIN_TOKENS
    minutes: float       # max_runtime_s_per_game / 60
    games: str           # game panel: all | balanced | franzen
    notes: str = ""


DEFAULT_CTX, DEFAULT_DRAIN = 128 * 1024, 58 * 1024
RUNS = [
    Run("fz-arms-v4-control", "full", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all"),
    Run("fz-arms-v5-temp06", "full", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all",
        "LOCAL_ANALYZER_TEMPERATURE=0.6"),
    Run("fz-arms-v6-audit60k", "full", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all",
        "ARC3_STRATEGY_AUDIT_TOKENS=60000"),
    Run("fz-experts16-full", "full", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all",
        "num_experts_per_tok=16: slower serving"),
    Run("fz-ledger-on-trim", "full", "franzen", 1, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all"),
    Run("fz-ledger-on-trim-r2", "full", "franzen", 1, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all"),
    Run("fz-ledger-streams12", "full", "franzen", 1, 12, 112640, 50176, 121, "all",
        "MAXREQ 12, server ctx 120832"),
    Run("fz-streams14", "full", "franzen", 0, 14, 98304, 43008, 121, "all",
        "MAXREQ 14, server ctx 106496"),
    Run("fz-ucb", "full", "ucb", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all", "ARC3_UCB_C=0.1"),
    Run("v4-stack-v1", "full", "dprime", 1, 10, DEFAULT_CTX, DEFAULT_DRAIN, 121, "all",
        "D' (fade 0.4) + ledger; no serve.log or notebook log; turns from artifacts events"),
    Run("v4-panel-control", "panel", "franzen", 0, 4, DEFAULT_CTX, DEFAULT_DRAIN, 20, "balanced"),
    Run("v4-panel-easy-first", "panel", "easy-first", 0, 4, DEFAULT_CTX, DEFAULT_DRAIN, 20, "balanced"),
    Run("v4-panel-dprime", "panel", "dprime", 0, 4, DEFAULT_CTX, DEFAULT_DRAIN, 20, "balanced",
        "fade 0.4"),
    Run("fz-tput-base", "panel", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 20, "balanced",
        "10 games on 10 slots: gate never binds"),
    Run("fz-experts8", "panel", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 20, "balanced",
        "num_experts_per_tok=8; 10 games on 10 slots"),
    Run("fz-experts16", "panel", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 25, "franzen",
        "num_experts_per_tok=16; 10 games on 10 slots"),
    Run("fz-spec-adaptive", "panel", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 20, "balanced",
        "adaptive speculative steps; 10 games on 10 slots"),
    Run("fz-tput-all-v7", "panel", "franzen", 0, 10, DEFAULT_CTX, DEFAULT_DRAIN, 25, "franzen",
        "temperature 0, MXFP8; 10 games on 10 slots"),
]


# ----------------------------------------------------------------------------- helpers

def q(values, p):
    """Linear-interpolated quantile of a list (None when empty)."""
    v = sorted(x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def qs(values, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return {f"p{int(p * 100)}": r(q(values, p)) for p in ps} | {"n": len(values)}


def r(x, nd=2):
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return round(x, nd)
    return x


def utc_epoch(naive_iso: str) -> float:
    return datetime.fromisoformat(naive_iso).replace(tzinfo=timezone.utc).timestamp()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else r(row.get(k), 3)) for k in fields})


def level_score(baseline: float, actions: int) -> float:
    """Per-level score of the harness scorer (taaf GameRun._compute_final_score)."""
    if actions <= 0:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def game_score(levels_total: int, completed: int, actions_per_level, baselines) -> float:
    """taaf GameRun._compute_final_score: level-weighted, capped at the scored weight share."""
    total = tw = mw = 0.0
    for i in range(levels_total):
        w = i + 1
        tw += w
        a = actions_per_level[i] if i < len(actions_per_level) else 0
        s = level_score(baselines[i], a) if i < completed and a > 0 else 0.0
        if s > 0:
            mw += w
        total += s * w
    return 0.0 if tw == 0 else min(total / tw, mw / tw * 100.0)


# ----------------------------------------------------------------------------- parsers

def load_benchmark(run_dir: Path) -> dict:
    return json.loads((run_dir / "benchmark.json").read_text())


def parse_notebook_log(run_dir: Path) -> dict:
    """Notebook-relative times (s since kernel start) of the startup milestones."""
    logs = [p for p in run_dir.glob("*.log") if p.name != "serve.log"]
    out = {}
    if not logs:
        return out
    try:
        entries = json.loads(logs[0].read_text())
    except json.JSONDecodeError:
        return out
    seen_timeouts = set()
    for e in entries:
        d, t = e.get("data", ""), float(e.get("time", 0.0))
        if "READY after" in d and "nb_ready_s" not in out:
            out["nb_ready_s"] = t
            m = re.search(r"READY after (\d+)s", d)
            out["server_ready_after_launch_s"] = float(m.group(1)) if m else None
        elif d.startswith("Starting benchmark") and "nb_bench_start_s" not in out:
            out["nb_bench_start_s"] = t
        elif "pid " in d and "serve.log" in d and "nb_server_launch_s" not in out:
            out["nb_server_launch_s"] = t
        elif "tail fade phase after" in d and "tail_fade_at_min" not in out:
            m = re.search(r"after ([\d.]+) minutes: ([\d.]+) minutes remaining", d)
            if m:
                out["tail_fade_at_min"] = float(m.group(1))
                out["tail_fade_remaining_min"] = float(m.group(2))
        elif "analyzer request timed out" in d and d not in seen_timeouts:
            # the notebook log carries each warning twice (stream handler + lastResort)
            seen_timeouts.add(d)
            out["n_request_timeouts"] = out.get("n_request_timeouts", 0) + 1
            out.setdefault("nb_first_timeout_s", t)
    return out


RTS_RE = re.compile(
    r"ReqTimeStats\(rid=(\S+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+), "
    r"attempts=(\d+), type=\w+\): queue_duration=([\d.]+)ms, initial_prefill_elapsed=([\d.]+)ms, "
    r"post_prefill_elapsed=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)"
)
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
DEC_RE = re.compile(
    r"Decode batch, #running-req: (\d+), #full token: (\d+).*?accept len: ([\d.]+).*?"
    r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
PRE_RE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?"
    r"#running-req: (\d+), #queue-req: (\d+)"
)


def parse_serve_log(path: Path, bench_epoch: float) -> dict:
    """ReqTimeStats, Decode batch and Prefill batch lines, times relative to bench start."""
    rts, dec, pre = [], [], []
    with path.open(errors="replace") as fh:
        for line in fh:
            if "ReqTimeStats" in line:
                m = RTS_RE.search(line)
                if not m:
                    continue
                g = m.groups()
                if g[0].startswith("HEALTH_CHECK"):
                    continue
                entry = float(g[9]) - bench_epoch
                rts.append({
                    "t_entry_s": entry, "input": int(g[1]), "cached": int(g[2]), "output": int(g[3]),
                    "queue_ms": float(g[5]), "prefill_s": float(g[6]) / 1e3,
                    "decode_s": float(g[7]) / 1e3, "forward_s": float(g[8]) / 1e3,
                })
                continue
            if "Decode batch" in line or "Prefill batch" in line:
                ts = TS_RE.match(line)
                if not ts:
                    continue
                t = utc_epoch(ts.group(1).replace(" ", "T")) - bench_epoch
                if "Decode batch" in line:
                    m = DEC_RE.search(line)
                    if m:
                        dec.append((t, int(m.group(1)), float(m.group(4)), float(m.group(3)),
                                    int(m.group(5)), int(m.group(2))))
                else:
                    m = PRE_RE.search(line)
                    if m:
                        pre.append((t, int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
    rts.sort(key=lambda x: x["t_entry_s"])
    dec.sort()
    pre.sort()
    # mean #running-req while each request decoded (decode lines are ~1.5 s apart)
    dts = [d[0] for d in dec]
    for x in rts:
        a = x["t_entry_s"] + x["prefill_s"]
        b = x["t_entry_s"] + x["forward_s"]
        i, j = bisect.bisect_left(dts, a), bisect.bisect_right(dts, b + 1.0)
        vals = [dec[k][1] for k in range(i, j)]
        if not vals and dec:
            k = min(max(0, i), len(dec) - 1)
            vals = [dec[k][1]]
        x["running_mean"] = statistics.fmean(vals) if vals else None
        x["stream_tps"] = x["output"] / x["decode_s"] if x["decode_s"] > 0.05 else None
    return {"rts": rts, "dec": dec, "pre": pre}


HDR_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d) \|[^\n]*---$", re.M)
SECTION_RE = re.compile(r"^\[([A-Z][A-Z _]*)(?::[^\]\n]*)?\]$", re.M)


def parse_turns(text: str, bench_start: datetime) -> list[dict]:
    """One dict per analyzer turn: start time, request count, output chars, history size."""
    heads = list(HDR_RE.finditer(text))
    turns = []
    day0 = bench_start.replace(hour=0, minute=0, second=0, microsecond=0)
    for i, m in enumerate(heads):
        body = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        t = day0 + timedelta(hours=int(m.group(3)), minutes=int(m.group(4)), seconds=int(m.group(5)))
        rel = (t - bench_start).total_seconds()
        if rel < -3600:
            rel += 86400
        secs = list(SECTION_RE.finditer(body))
        n_req = chars = 0
        hist_msgs = None
        message = ""
        for k, s in enumerate(secs):
            name = s.group(1)
            chunk = body[s.end(): secs[k + 1].start() if k + 1 < len(secs) else len(body)]
            if name == "MODEL RESPONSE META":
                n_req += 1
                cc = re.search(r"content_chars: (\d+)", chunk)
                rc = re.search(r"reasoning_chars: (\d+)", chunk)
                tc = chunk.find("raw_tool_calls:")
                chars += (int(cc.group(1)) if cc else 0) + (int(rc.group(1)) if rc else 0)
                chars += len(chunk) - tc if tc >= 0 else 0
            elif name == "ANALYZER STATUS":
                hm = re.search(r"^history_messages: (\d+)", chunk, re.M)
                if hm:
                    hist_msgs = int(hm.group(1))
                mm = re.search(r"^message: (.*)$", chunk, re.M)
                if mm:
                    message = mm.group(1).strip()
        turns.append({
            "step": int(m.group(1)), "action": int(m.group(2)), "t_s": rel,
            "n_req": n_req, "chars": chars, "hist_msgs": hist_msgs, "message": message,
        })
    return turns


def load_turns(run_dir: Path, game_id: str, bench_start: datetime) -> list[dict] | None:
    tr = run_dir / "transcripts" / f"{game_id}_p0.txt"
    if tr.exists() and tr.stat().st_size > 0:
        return parse_turns(tr.read_text(errors="replace"), bench_start)
    ev = run_dir / "artifacts" / f"{game_id}_p0_events.jsonl"
    if ev.exists() and ev.stat().st_size > 0:
        parts = []
        with ev.open("rb") as fh:
            for line in fh:
                if b'"type": "analysis"' not in line and b'"type":"analysis"' not in line:
                    continue
                row = json.loads(line)
                if row.get("type") == "analysis" and row.get("transcript"):
                    parts.append(row["transcript"])
        if parts:
            return parse_turns("\n".join(parts), bench_start)
    if tr.exists() or ev.exists():
        return []
    return None


def load_requests(run_dir: Path, game_id: str) -> list[dict] | None:
    """Response records of one game's requests.jsonl (usage + turn indices + message count)."""
    path = run_dir / f"{game_id}_p0_requests.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return None
    out = []
    with path.open("rb") as fh:
        for line in fh:
            i = line.rfind(b'"chat_template_kwargs"')
            if i < 0:
                continue
            tail = json.loads(b"{" + line[i:])
            if tail.get("event") != "response":
                continue
            u = tail.get("usage") or {}
            out.append({
                "step": tail.get("analysis_step"), "action": tail.get("action"),
                "req_in_turn": tail.get("request_index_within_turn"),
                "prompt": int(u.get("prompt_tokens") or 0),
                "cached": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "completion": int(u.get("completion_tokens") or 0),
                "reasoning": int(u.get("reasoning_tokens") or 0),
                "image_tokens": int((u.get("prompt_tokens_details") or {}).get("image_tokens") or 0),
                "finish": tail.get("finish_reason"),
                "n_msgs": line[:i].count(b'"role": "'),
            })
    return out


def match_requests(games_reqs: dict[str, list[dict]], rts: list[dict]) -> int:
    """Attach serve-side timing to each request; returns the number left unmatched."""
    by_key = defaultdict(list)
    for idx, x in enumerate(rts):
        by_key[(x["input"], x["cached"])].append(idx)
    used = set()
    unmatched = 0
    for reqs in games_reqs.values():
        last = -1e9
        for rq in reqs:
            cands = [i for i in by_key.get((rq["prompt"], rq["cached"]), []) if i not in used]
            cands = [i for i in cands if abs(rts[i]["output"] - rq["completion"]) <= 64] or cands
            if not cands:
                unmatched += 1
                continue
            later = [i for i in cands if rts[i]["t_entry_s"] >= last - 1.0] or cands
            i = min(later, key=lambda k: (abs(rts[k]["output"] - rq["completion"]), rts[k]["t_entry_s"]))
            used.add(i)
            rq["rts"] = rts[i]
            last = rts[i]["t_entry_s"]
    return unmatched


# ----------------------------------------------------------------------------- per game

def stream_rate_fn(rts: list[dict], fallback: float):
    """Per-stream decode tok/s around time t (median of requests entering within +-5 min)."""
    pts = sorted((x["t_entry_s"], x["stream_tps"]) for x in rts
                 if x.get("stream_tps") and x["output"] >= 256 and x["input"] >= 2000)
    ts = [p[0] for p in pts]

    def rate(t: float) -> float:
        i, j = bisect.bisect_left(ts, t - 300), bisect.bisect_right(ts, t + 300)
        vals = [pts[k][1] for k in range(i, j)]
        return statistics.median(vals) if len(vals) >= 5 else fallback
    return rate


def level_bounds(game: dict) -> list[tuple[int, int]]:
    """(first, last) history index per attempted level, from actions_per_level."""
    per = game.get("actions_per_level") or []
    n = int(game["number_of_levels"])
    done = int(game["levels_completed"])
    attempted = done + (1 if done < n else 0)
    out, cur = [], 0
    for lv in range(attempted):
        a = int(per[lv]) if lv < len(per) else 0
        out.append((cur, cur + a - 1))
        cur += a
    return out


def timeline_from_requests(reqs: list[dict], end_s: float, won: bool, budget: int) -> dict:
    """Exact slot-holding intervals: a trim request preceded by a >= SWITCH_WAIT_S gap.

    A game whose last request left a trim due (prompt + completion near the context budget)
    and that then sent nothing for >= SWITCH_WAIT_S before the deadline was switched out at
    that trim and never re-admitted; that terminal wait is recorded too.
    """
    timed = [x for x in reqs if x.get("rts")]
    if not timed:
        return {}
    intervals, waits, trims = [], [], []
    start = timed[0]["rts"]["t_entry_s"]
    prev = None
    for x in timed:
        t0 = x["rts"]["t_entry_s"]
        if prev is not None:
            prev_end = prev["rts"]["t_entry_s"] + prev["rts"]["forward_s"]
            gap = t0 - prev_end
            x["gap_s"] = gap
            trim = x["prompt"] < 0.8 * prev["prompt"] and x["prompt"] > 10_000
            x["trim"] = trim
            if trim:
                trims.append(t0)
                x["switched_out"] = gap >= SWITCH_WAIT_S
                if gap >= SWITCH_WAIT_S:
                    intervals.append((start, prev_end, "switch_out"))
                    waits.append((prev_end, t0, gap))
                    start = t0
        prev = x
    last_end = prev["rts"]["t_entry_s"] + prev["rts"]["forward_s"]
    if waits and not won and start >= end_s - TERMINAL_SLACK_S:
        # re-admitted only as the deadline stopped everyone else: never re-admitted in time
        w0 = waits[-1][0]
        waits[-1] = (w0, end_s, end_s - w0)
        return {"first_active_s": timed[0]["rts"]["t_entry_s"], "intervals": intervals,
                "waits": waits, "trims": trims, "terminal_wait": True}
    trim_due = prev["prompt"] + prev["completion"] >= TRIM_DUE_FRACTION * budget
    if not won and trim_due and end_s - last_end >= SWITCH_WAIT_S:
        intervals.append((start, last_end, "switch_out"))
        waits.append((last_end, end_s, end_s - last_end))
        trims.append(last_end)
        return {"first_active_s": timed[0]["rts"]["t_entry_s"], "intervals": intervals,
                "waits": waits, "trims": trims, "terminal_wait": True}
    intervals.append((start, max(min(end_s, last_end), start), "game_end"))
    return {"first_active_s": timed[0]["rts"]["t_entry_s"], "intervals": intervals,
            "waits": waits, "trims": trims}


def turn_tokens(turns: list[dict], history: list[dict], total_tokens: int) -> None:
    """Split each action's exact generated tokens over the turns that produced it, by chars."""
    groups = defaultdict(list)
    for t in turns:
        groups[t["action"]].append(t)
    hist_sum = sum(int(s.get("generated_tokens") or 0) for s in history)
    for a, ts in groups.items():
        if 1 <= a <= len(history):
            tok = int(history[a - 1].get("generated_tokens") or 0)
        elif a > len(history):
            tok = max(0, total_tokens - hist_sum)    # reasoning after the last action
        else:
            tok = sum(x["chars"] for x in ts) * TOKENS_PER_CHAR
        ch = sum(x["chars"] for x in ts)
        for x in ts:
            x["tokens_est"] = tok * (x["chars"] / ch) if ch else tok / len(ts)


def timeline_from_turns(turns: list[dict], end_s: float, rate, won: bool = True) -> dict:
    """Intervals from turn start times; a trim followed by a turn far longer than its work
    is a switch-out.

    history_messages (reported at the end of a turn) drops in the turn whose trim happened.
    The trimmer runs before each request, so the handover wait falls either in that same
    turn (trim before one of its requests) or in the next one (trim after its last request,
    the handover then happens at the next turn's first request). Both are checked and the
    larger excess over the estimated work is taken.
    """
    if not turns:
        return {}
    last_h = None
    for k, t in enumerate(turns):
        nxt = turns[k + 1]["t_s"] if k + 1 < len(turns) else end_s
        t["dur_s"] = max(0.0, nxt - t["t_s"])
        tok = t.get("tokens_est") or 0.0
        t["work_s"] = tok / max(1.0, rate(t["t_s"])) + REQUEST_OVERHEAD_S * max(1, t["n_req"])
        t["excess_s"] = t["dur_s"] - t["work_s"] - REPREFILL_S
        h = t["hist_msgs"]
        t["trim"] = h is not None and last_h is not None and h < last_h
        t["wait_s"] = 0.0
        if h is not None:
            last_h = h
    intervals, waits, trims = [], [], []
    start = turns[0]["t_s"]
    for k, t in enumerate(turns):
        if not t["trim"]:
            continue
        trims.append(t["t_s"])
        cands = [k] + ([k + 1] if k + 1 < len(turns) and not turns[k + 1]["trim"] else [])
        j = max(cands, key=lambda i: turns[i]["excess_s"])
        wait = turns[j]["excess_s"]
        t["wait_est_s"] = max(0.0, wait)
        if wait >= SWITCH_WAIT_S:
            tj = turns[j]["t_s"]
            turns[j]["wait_s"] = wait
            intervals.append((start, tj, "switch_out"))
            waits.append((tj, tj + wait, wait))
            start = tj + wait
    out = {"first_active_s": turns[0]["t_s"], "intervals": intervals, "waits": waits, "trims": trims}
    if waits and not won and start >= end_s - TERMINAL_SLACK_S:
        # switched out and not re-admitted before the deadline. A game still queued when the
        # deadline stops everyone else may be admitted for a moment (its last turn then ends
        # with "stop_requested"); that sliver is folded into the wait.
        w0 = waits[-1][0]
        waits[-1] = (w0, end_s, end_s - w0)
        for t in turns:
            if t["t_s"] >= w0:
                t["wait_s"] = t["dur_s"]
        out["terminal_wait"] = True
        return out
    intervals.append((start, max(start, end_s), "game_end"))
    return out


def timeline_from_history(history: list[dict], end_s: float, rate, gated: bool,
                          won: bool = True, tail_tokens: int = 0) -> dict:
    """Intervals from action gaps alone (no trim marker). The stretch from the last action
    to the deadline counts as a terminal wait when it exceeds the tail tokens' work."""
    acts = [(float(s["wallclock_seconds"]), int(s.get("generated_tokens") or 0)) for s in history]
    acts = [a for a in acts if a[0] > 0]
    if not acts:
        return {}

    def work(t, tok):
        return tok / max(1.0, rate(t)) + REQUEST_OVERHEAD_S * max(1, math.ceil(tok / 1500))
    first_tok = next(((t, tok) for t, tok in acts if tok > 0), acts[0])
    first = max(0.0, first_tok[0] - work(*first_tok))
    intervals, waits = [], []
    start, prev_t = first, first_tok[0]
    for t, tok in acts:
        if t <= first_tok[0]:
            continue
        wait = (t - prev_t) - work(t, tok)
        if gated and wait >= HISTORY_WAIT_S:
            intervals.append((start, prev_t, "switch_out"))
            waits.append((prev_t, prev_t + wait, wait))
            start = prev_t + wait
        prev_t = t
    tail_wait = (end_s - prev_t) - work(end_s, tail_tokens)
    if gated and not won and tail_wait >= HISTORY_WAIT_S:
        intervals.append((start, end_s - tail_wait, "switch_out"))
        waits.append((end_s - tail_wait, end_s, tail_wait))
        return {"first_active_s": first, "intervals": intervals, "waits": waits, "trims": [],
                "terminal_wait": True}
    intervals.append((start, max(start, end_s), "game_end"))
    return {"first_active_s": first, "intervals": intervals, "waits": waits, "trims": []}


def inactive_overlap(waits, a: float, b: float) -> float:
    return sum(max(0.0, min(b, w1) - max(a, w0)) for w0, w1, _ in waits)


# ----------------------------------------------------------------------------- main build

def build(root: Path, out: Path, only: set[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = defaultdict(list)
    summary: dict = {"validation": {}}
    all_rts_rates = []

    for run in RUNS:
        if only and run.name not in only:
            continue
        run_dir = root / run.name
        if not (run_dir / "benchmark.json").exists():
            print(f"skip {run.name}: no benchmark.json")
            continue
        bench = load_benchmark(run_dir)
        bench_start = datetime.fromisoformat(bench["start_time"])
        bench_epoch = utc_epoch(bench["start_time"])
        deadline_s = run.minutes * 60.0
        nb = parse_notebook_log(run_dir)
        serve = parse_serve_log(run_dir / "serve.log", bench_epoch) if (run_dir / "serve.log").exists() else None
        rts = serve["rts"] if serve else []
        big = [x for x in rts if x["input"] >= 2000]
        run_rate = statistics.median([x["stream_tps"] for x in big if x.get("stream_tps") and x["output"] >= 256]) \
            if big else None
        if run_rate:
            all_rts_rates.append(run_rate)
        fallback = run_rate or (statistics.median(all_rts_rates) if all_rts_rates else 70.0)
        rate = stream_rate_fn(rts, fallback)
        gated = len(bench["game_runs"]) > run.slots
        budget = run.ctx_window - 12 * 1024    # LOCAL_ANALYZER_MAX_OUTPUT reserve
        st = [utc_epoch(g["started_at"]) for g in bench["game_runs"] if g.get("started_at")]
        starts_equal = bool(st) and max(st) - min(st) < 5.0

        # ---- per game: requests, turns, timeline
        games = bench["game_runs"]
        game_reqs, game_turns = {}, {}
        for g in games:
            gid = g["game_id"]
            rq = load_requests(run_dir, gid)
            if rq is not None:
                game_reqs[gid] = rq
            tn = load_turns(run_dir, gid, bench_start)
            if tn is not None:
                game_turns[gid] = tn
        unmatched = match_requests(game_reqs, rts) if (game_reqs and rts) else None

        run_scores, run_rows_games = [], []
        method_counts = defaultdict(int)
        for g in games:
            gid, short = g["game_id"], g["game_id"][:4]
            hist = g["history"]
            n_lv, done = int(g["number_of_levels"]), int(g["levels_completed"])
            per = [int(a) for a in (g.get("actions_per_level") or [])]
            base = [float(b) for b in (g.get("base_actions_per_level") or [])]
            m = re.search(r"tokens=(\d+)", g.get("solver_note") or "")
            hist_tokens = sum(int(s.get("generated_tokens") or 0) for s in hist)
            total_tokens = int(m.group(1)) if m else hist_tokens
            score = game_score(n_lv, done, per, base)
            run_scores.append(score)
            won = done == n_lv
            final_wall = float(g.get("final_wallclock_seconds") or 0.0)
            last_action = float(hist[-1]["wallclock_seconds"]) if hist else None
            end_s = (last_action if won and last_action else min(final_wall or deadline_s, deadline_s))

            reqs = game_reqs.get(gid)
            turns = game_turns.get(gid)
            if turns:
                turn_tokens(turns, hist, total_tokens)
            tl, method = {}, "none"
            if reqs and any(x.get("rts") for x in reqs):
                tl, method = timeline_from_requests(reqs, end_s, won, budget), "requests"
            elif turns:
                tl, method = timeline_from_turns(turns, end_s, rate, won), "transcript"
            elif hist:
                tl, method = timeline_from_history(hist, end_s, rate, gated, won,
                                                   max(0, total_tokens - hist_tokens)), "history"
            if not gated and tl:
                # with slots >= games the gate never makes anyone wait
                tl["intervals"] = [(tl["first_active_s"], tl["intervals"][-1][1], "game_end")]
                tl["waits"] = []
            method_counts[method] += 1
            # validation side-channel: transcript estimate alongside the exact request timeline
            if method == "requests" and turns and gated:
                alt = timeline_from_turns(turns, end_s, rate, won)
                summary["validation"].setdefault(run.name, {"exact": [], "transcript": [], "history": []})
                summary["validation"][run.name]["exact"] += [w[2] for w in tl.get("waits", [])]
                summary["validation"][run.name]["transcript"] += [w[2] for w in alt.get("waits", [])]
                alt_h = timeline_from_history(hist, end_s, rate, gated, won, max(0, total_tokens - hist_tokens))
                summary["validation"][run.name]["history"] += [w[2] for w in alt_h.get("waits", [])]

            waits = tl.get("waits", [])
            intervals = tl.get("intervals", [])
            first_active = tl.get("first_active_s")
            active_s = sum(b - a for a, b, _ in intervals)
            for k, (a, b, why) in enumerate(intervals):
                tok_in = sum(int(s.get("generated_tokens") or 0) for s in hist
                             if a - 1 <= float(s["wallclock_seconds"]) <= b + 1)
                act_in = sum(1 for s in hist if a - 1 <= float(s["wallclock_seconds"]) <= b + 1)
                rows["intervals"].append({
                    "run": run.name, "game": short, "idx": k, "start_s": a, "end_s": b,
                    "dur_s": b - a, "end_reason": why if k < len(intervals) - 1 else (
                        "won" if won else "switch_out_not_readmitted" if tl.get("terminal_wait")
                        else "deadline"),
                    "wait_after_s": waits[k][2] if k < len(waits) else None,
                    "tokens_hist": tok_in, "actions": act_in, "method": method,
                })

            # ---- (a) levels
            bounds = level_bounds(g)
            charged = 0
            for lv, (i0, i1) in enumerate(bounds, start=1):
                chunk = hist[i0:i1 + 1] if i1 >= i0 else []
                tok = sum(int(s.get("generated_tokens") or 0) for s in chunk)
                charged += tok
                cleared = lv <= done
                if not cleared:
                    tok = max(0, total_tokens - (charged - tok))
                lv_start = (float(hist[i0 - 1]["wallclock_seconds"]) if i0 > 0 and i0 - 1 < len(hist)
                            else first_active)
                lv_end = (float(hist[i1]["wallclock_seconds"]) if cleared and i1 < len(hist) and i1 >= 0
                          else end_s)
                if lv_start is None and hist:
                    lv_start = float(hist[0]["wallclock_seconds"])
                wall = (lv_end - lv_start) if (lv_start is not None and lv_end is not None) else None
                inactive = inactive_overlap(waits, lv_start, lv_end) if wall is not None else None
                n_trims = sum(1 for t in tl.get("trims", []) if lv_start <= t <= lv_end) \
                    if (wall is not None and method != "history") else None
                n_sw = sum(1 for w in waits if lv_start <= w[0] <= lv_end) if wall is not None else None
                rows["levels"].append({
                    "run": run.name, "kind": run.kind, "scheduler": run.scheduler, "ledger": run.ledger,
                    "slots": run.slots, "game": short, "level": lv, "levels_total": n_lv,
                    "human_actions": base[lv - 1] if lv - 1 < len(base) else None,
                    "tokens": tok, "actions": len(chunk), "cleared": int(cleared),
                    "censored": int(not cleared),
                    "level_score": level_score(base[lv - 1], len(chunk)) if cleared and lv - 1 < len(base) else 0.0,
                    "start_s": lv_start, "end_s": lv_end, "wall_s": wall,
                    "active_s": (wall - inactive) if wall is not None else None,
                    "n_trims": n_trims, "n_switch_outs": n_sw, "timeline": method,
                })

            # ---- (d) context segments
            if reqs:
                seg_rows = context_segments_from_requests(run, short, reqs)
                rows["segments"] += seg_rows
            elif turns:
                rows["segments"] += context_segments_from_turns(run, short, turns)

            # ---- (c) per-request rows
            if reqs:
                for x in reqs:
                    t = x.get("rts") or {}
                    rows["requests"].append({
                        "run": run.name, "game": short, "step": x["step"], "action": x["action"],
                        "req_in_turn": x["req_in_turn"], "t_entry_s": t.get("t_entry_s"),
                        "prompt_tokens": x["prompt"], "cached_tokens": x["cached"],
                        "uncached_tokens": x["prompt"] - x["cached"], "completion_tokens": x["completion"],
                        "reasoning_tokens": x["reasoning"], "image_tokens": x["image_tokens"],
                        "n_msgs": x["n_msgs"], "finish": x["finish"],
                        "queue_ms": t.get("queue_ms"), "prefill_s": t.get("prefill_s"),
                        "decode_s": t.get("decode_s"), "running_mean": t.get("running_mean"),
                        "stream_tps": t.get("stream_tps"), "gap_before_s": x.get("gap_s"),
                        "trim": int(bool(x.get("trim"))), "switched_out": int(bool(x.get("switched_out"))),
                    })

            n_turns = len(turns) if turns else None
            run_rows_games.append({
                "run": run.name, "kind": run.kind, "scheduler": run.scheduler, "ledger": run.ledger,
                "slots": run.slots, "game": short, "levels_total": n_lv, "levels_completed": done,
                "state": g.get("state"), "score": score, "final_score_harness": g.get("final_score"),
                "first_active_s": first_active,
                "first_action_s": next((float(s["wallclock_seconds"]) for s in hist
                                        if int(s.get("generated_tokens") or 0) > 0), None),
                "last_action_s": last_action, "end_s": end_s, "total_tokens": total_tokens,
                "hist_tokens": hist_tokens, "actions": len(hist), "n_turns": n_turns,
                "n_requests": len(reqs) if reqs else None,
                "n_intervals": len(intervals), "active_s": active_s,
                "n_trims": len(tl.get("trims", [])) if method != "history" else None,
                "n_switch_outs": len(waits), "wait_total_s": sum(w[2] for w in waits),
                "timeline": method,
            })

        # start rank
        ranked = sorted((x for x in run_rows_games if x["first_active_s"] is not None),
                        key=lambda x: x["first_active_s"])
        for k, x in enumerate(ranked, start=1):
            x["start_rank"] = k
        rows["games"] += run_rows_games

        # ---- serving tables
        if serve:
            serving_tables(run, serve, rows, deadline_s)
            for x in rts:
                rows["reqstats"].append({"run": run.name, **x})

        mean_score = statistics.fmean(run_scores) if run_scores else None
        first_req = next((x["t_entry_s"] for x in rts if x["input"] >= 2000), None)
        run_row = {
            "run": run.name, "kind": run.kind, "scheduler": run.scheduler, "ledger": run.ledger,
            "slots": run.slots, "ctx_window": run.ctx_window, "drain_tokens": run.drain,
            "minutes": run.minutes, "deadline_s": deadline_s, "games": len(games), "panel": run.games,
            "score_mean": mean_score,
            "score_harness": statistics.fmean(float(g.get("final_score") or 0.0) for g in games),
            "total_tokens": sum(x["total_tokens"] for x in run_rows_games),
            "total_actions": sum(x["actions"] for x in run_rows_games),
            "bench_start_utc": bench["start_time"], "bench_end_utc": bench.get("end_time"),
            "all_sessions_same_start": int(starts_equal),
            "nb_server_launch_s": nb.get("nb_server_launch_s"),
            "server_ready_after_launch_s": nb.get("server_ready_after_launch_s"),
            "nb_ready_s": nb.get("nb_ready_s"), "nb_bench_start_s": nb.get("nb_bench_start_s"),
            "first_request_after_bench_s": first_req,
            "nb_first_request_s": (nb["nb_bench_start_s"] + first_req)
            if (first_req is not None and nb.get("nb_bench_start_s") is not None) else None,
            "tail_fade_at_min": nb.get("tail_fade_at_min"),
            "n_request_timeouts_at_end": nb.get("n_request_timeouts"),
            "stream_tps_median": run_rate, "timeline_methods": ";".join(f"{k}:{v}" for k, v in sorted(method_counts.items())),
            "requests_unmatched": unmatched, "notes": run.notes,
        }
        rows["runs"].append(run_row)
        print(f"{run.name:24s} score={mean_score:6.2f} harness={run_row['score_harness']:6.2f} "
              f"timeline={run_row['timeline_methods']} unmatched={unmatched}")

    write_all(out, rows)
    summary |= data_card(rows, summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def context_segments_from_requests(run: Run, game: str, reqs: list[dict]) -> list[dict]:
    segs, cur = [], []

    def close(seg, nxt):
        timed = [x for x in seg if x.get("rts")]
        row = {
            "run": run.name, "game": game, "ledger": run.ledger, "ctx_window": run.ctx_window,
            "drain_tokens": run.drain, "method": "requests", "idx": len(segs),
            "prompt_start": seg[0]["prompt"], "prompt_peak": max(x["prompt"] for x in seg),
            "prompt_growth": max(x["prompt"] for x in seg) - seg[0]["prompt"],
            "gen_tokens": sum(x["completion"] for x in seg), "n_requests": len(seg),
            "n_turns": len({(x["step"], x["action"]) for x in seg if x["req_in_turn"] == 1}) or 1,
            "n_actions": (max(x["action"] for x in seg) - min(x["action"] for x in seg)),
            "start_s": timed[0]["rts"]["t_entry_s"] if timed else None,
            "end_s": (timed[-1]["rts"]["t_entry_s"] + timed[-1]["rts"]["forward_s"]) if timed else None,
            "ended_by_trim": int(nxt is not None),
        }
        if nxt is not None:
            t = nxt.get("rts") or {}
            row |= {
                "next_prompt": nxt["prompt"], "next_cached": nxt["cached"],
                "reprefill_uncached": nxt["prompt"] - nxt["cached"], "reprefill_s": t.get("prefill_s"),
                "wait_after_s": nxt.get("gap_s"), "switched_out_after": int(bool(nxt.get("switched_out"))),
            }
        # switch-outs only happen at trims, i.e. before a segment's first request
        row["busy_s"] = (row["end_s"] - row["start_s"]) if timed else None
        segs.append(row)

    for x in reqs:
        if cur and x["prompt"] < 0.8 * cur[-1]["prompt"] and x["prompt"] > 10_000:
            close(cur, x)
            cur = []
        cur.append(x)
    if cur:
        close(cur, None)
    return segs


def context_segments_from_turns(run: Run, game: str, turns: list[dict]) -> list[dict]:
    segs, cur = [], []

    def close(seg, nxt):
        row = {
            "run": run.name, "game": game, "ledger": run.ledger, "ctx_window": run.ctx_window,
            "drain_tokens": run.drain, "method": "transcript", "idx": len(segs),
            "hist_msgs_start": seg[0]["hist_msgs"], "hist_msgs_peak": max((x["hist_msgs"] or 0) for x in seg),
            "gen_tokens": sum(x.get("tokens_est") or 0 for x in seg),
            "n_requests": sum(x["n_req"] for x in seg), "n_turns": len(seg),
            "n_actions": max(x["action"] for x in seg) - min(x["action"] for x in seg),
            "start_s": seg[0]["t_s"], "end_s": seg[-1]["t_s"] + seg[-1].get("dur_s", 0.0),
            "ended_by_trim": int(nxt is not None),
        }
        if nxt is not None:
            row["wait_after_s"] = nxt.get("wait_est_s")
            row["switched_out_after"] = int((nxt.get("wait_est_s") or 0) >= SWITCH_WAIT_S)
        row["busy_s"] = sum(x.get("dur_s", 0.0) - x.get("wait_s", 0.0) for x in seg)
        segs.append(row)

    for t in turns:
        if cur and t.get("trim"):
            close(cur, t)
            cur = []
        cur.append(t)
    if cur:
        close(cur, None)
    return segs


def serving_tables(run: Run, serve: dict, rows, deadline_s: float) -> None:
    dec = [d for d in serve["dec"] if 0 <= d[0] <= deadline_s + 120]
    by_n = defaultdict(list)
    for t, n, tps, acc, qr, ft in dec:
        by_n[n].append((tps, acc))
    for n in sorted(by_n):
        v = by_n[n]
        agg = [x[0] for x in v]
        rows["serve_decode"].append({
            "run": run.name, "slots": run.slots, "running_req": n, "n_lines": len(v),
            "agg_tps_p10": q(agg, 0.1), "agg_tps_median": q(agg, 0.5), "agg_tps_p90": q(agg, 0.9),
            "agg_tps_mean": statistics.fmean(agg),
            "per_stream_tps_median": (q(agg, 0.5) / n) if n else None,
            "accept_len_median": q([x[1] for x in v], 0.5),
        })
    # per-minute series: decode lines (mean running, aggregate tok/s), prefill new tokens,
    # and requests entering/finished
    mins = defaultdict(lambda: {"run_sum": 0.0, "tps_sum": 0.0, "n": 0, "queue_sum": 0.0,
                                "prefill_tokens": 0, "prefill_cached": 0, "req_entered": 0,
                                "gen_tokens_finished": 0})
    for t, n, tps, acc, qr, ft in dec:
        b = mins[int(t // 60)]
        b["run_sum"] += n
        b["tps_sum"] += tps
        b["queue_sum"] += qr
        b["n"] += 1
    for t, new, cached, running, qr in serve["pre"]:
        if 0 <= t <= deadline_s + 120:
            mins[int(t // 60)]["prefill_tokens"] += new
            mins[int(t // 60)]["prefill_cached"] += cached
    for x in serve["rts"]:
        if 0 <= x["t_entry_s"] <= deadline_s + 120 and x["input"] >= 2000:
            mins[int(x["t_entry_s"] // 60)]["req_entered"] += 1
            fin = x["t_entry_s"] + x["forward_s"]
            mins[int(fin // 60)]["gen_tokens_finished"] += x["output"]
    for mnt in sorted(k for k in mins if k >= 0):
        b = mins[mnt]
        rows["serve_minutes"].append({
            "run": run.name, "minute": mnt,
            "running_mean": b["run_sum"] / b["n"] if b["n"] else None,
            "agg_tps_mean": b["tps_sum"] / b["n"] if b["n"] else None,
            "per_stream_tps": (b["tps_sum"] / b["run_sum"]) if b["run_sum"] else None,
            "queue_req_mean": b["queue_sum"] / b["n"] if b["n"] else None,
            "prefill_new_tokens": b["prefill_tokens"], "prefill_cached_tokens": b["prefill_cached"],
            "requests_entered": b["req_entered"], "gen_tokens_finished": b["gen_tokens_finished"],
        })


FIELDS = {
    "runs": ["run", "kind", "scheduler", "ledger", "slots", "ctx_window", "drain_tokens", "minutes",
             "deadline_s", "games", "panel", "score_mean", "score_harness", "total_tokens", "total_actions",
             "bench_start_utc", "bench_end_utc", "all_sessions_same_start", "nb_server_launch_s",
             "server_ready_after_launch_s", "nb_ready_s", "nb_bench_start_s", "first_request_after_bench_s",
             "nb_first_request_s", "tail_fade_at_min", "n_request_timeouts_at_end", "stream_tps_median",
             "timeline_methods", "requests_unmatched", "notes"],
    "levels": ["run", "kind", "scheduler", "ledger", "slots", "game", "level", "levels_total",
               "human_actions", "tokens", "actions", "cleared", "censored", "level_score", "start_s",
               "end_s", "wall_s", "active_s", "n_trims", "n_switch_outs", "timeline"],
    "games": ["run", "kind", "scheduler", "ledger", "slots", "game", "levels_total", "levels_completed",
              "state", "score", "final_score_harness", "start_rank", "first_active_s", "first_action_s",
              "last_action_s", "end_s", "total_tokens", "hist_tokens", "actions", "n_turns", "n_requests",
              "n_intervals", "active_s", "n_trims", "n_switch_outs", "wait_total_s", "timeline"],
    "intervals": ["run", "game", "idx", "start_s", "end_s", "dur_s", "end_reason", "wait_after_s",
                  "tokens_hist", "actions", "method"],
    "requests": ["run", "game", "step", "action", "req_in_turn", "t_entry_s", "prompt_tokens",
                 "cached_tokens", "uncached_tokens", "completion_tokens", "reasoning_tokens", "image_tokens",
                 "n_msgs", "finish", "queue_ms", "prefill_s", "decode_s", "running_mean", "stream_tps",
                 "gap_before_s", "trim", "switched_out"],
    "reqstats": ["run", "t_entry_s", "input", "cached", "output", "queue_ms", "prefill_s", "decode_s",
                 "running_mean"],
    "serve_decode": ["run", "slots", "running_req", "n_lines", "agg_tps_p10", "agg_tps_median",
                     "agg_tps_p90", "agg_tps_mean", "per_stream_tps_median", "accept_len_median"],
    "serve_minutes": ["run", "minute", "running_mean", "agg_tps_mean", "per_stream_tps", "queue_req_mean",
                      "prefill_new_tokens", "prefill_cached_tokens", "requests_entered", "gen_tokens_finished"],
    "segments": ["run", "game", "ledger", "ctx_window", "drain_tokens", "method", "idx", "start_s", "end_s",
                 "busy_s", "n_turns", "n_requests", "n_actions", "gen_tokens", "prompt_start", "prompt_peak",
                 "prompt_growth", "hist_msgs_start", "hist_msgs_peak", "ended_by_trim", "next_prompt",
                 "next_cached", "reprefill_uncached", "reprefill_s", "wait_after_s", "switched_out_after"],
}
FRANZEN_FULL = {r.name for r in RUNS if r.kind == "full" and r.scheduler == "franzen"}


def pooled_levels(levels: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for x in levels:
        if x["run"] not in FRANZEN_FULL:
            continue
        for grp in ("franzen_full", "franzen_ledger" if x["ledger"] else "franzen_noledger"):
            groups[(x["game"], x["level"], grp)].append(x)
    out = []
    for (game, lv, grp), xs in sorted(groups.items()):
        cl = [x for x in xs if x["cleared"]]
        ce = [x for x in xs if x["censored"]]
        out.append({
            "game": game, "level": lv, "group": grp, "levels_total": xs[0]["levels_total"],
            "human_actions": xs[0]["human_actions"], "n_attempts": len(xs), "n_cleared": len(cl),
            "n_censored": len(ce),
            "tokens_cleared_median": q([x["tokens"] for x in cl], 0.5),
            "tokens_cleared_mean": statistics.fmean([x["tokens"] for x in cl]) if cl else None,
            "actions_cleared_median": q([x["actions"] for x in cl], 0.5),
            "wall_s_cleared_median": q([x["wall_s"] for x in cl], 0.5),
            "active_s_cleared_median": q([x["active_s"] for x in cl], 0.5),
            "level_score_cleared_median": q([x["level_score"] for x in cl], 0.5),
            "tokens_cleared": ";".join(str(int(x["tokens"])) for x in sorted(cl, key=lambda x: x["tokens"])),
            "tokens_censored": ";".join(str(int(x["tokens"])) for x in sorted(ce, key=lambda x: x["tokens"])),
            "actions_cleared": ";".join(str(x["actions"]) for x in sorted(cl, key=lambda x: x["tokens"])),
            "runs": ";".join(x["run"] for x in xs),
        })
    return out


POOLED_FIELDS = ["game", "level", "group", "levels_total", "human_actions", "n_attempts", "n_cleared",
                 "n_censored", "tokens_cleared_median", "tokens_cleared_mean", "actions_cleared_median",
                 "wall_s_cleared_median", "active_s_cleared_median", "level_score_cleared_median",
                 "tokens_cleared", "tokens_censored", "actions_cleared", "runs"]


def write_all(out: Path, rows) -> None:
    for name, fields in FIELDS.items():
        write_csv(out / f"{name}.csv", rows[name], fields)
    write_csv(out / "levels_pooled.csv", pooled_levels(rows["levels"]), POOLED_FIELDS)


def data_card(rows, summary) -> dict:
    """Key medians/percentiles over the extracted tables."""
    card: dict = {}
    full = [x for x in rows["runs"] if x["kind"] == "full"]
    card["startup"] = {
        "nb_bench_start_s": qs([x["nb_bench_start_s"] for x in rows["runs"] if x["nb_bench_start_s"]]),
        "nb_ready_s": qs([x["nb_ready_s"] for x in rows["runs"] if x["nb_ready_s"]]),
        "first_request_after_bench_s": qs([x["first_request_after_bench_s"] for x in rows["runs"]
                                           if x["first_request_after_bench_s"] is not None]),
        "all_sessions_same_start": all(x["all_sessions_same_start"] for x in rows["runs"]),
    }
    lv = [x for x in rows["levels"] if x["run"] in FRANZEN_FULL]
    cl = [x for x in lv if x["cleared"]]
    card["levels_franzen_full"] = {
        "attempts": len(lv), "cleared": len(cl), "censored": len(lv) - len(cl),
        "tokens_cleared": qs([x["tokens"] for x in cl]),
        "actions_cleared": qs([x["actions"] for x in cl]),
        "wall_s_cleared": qs([x["wall_s"] for x in cl]),
        "active_s_cleared": qs([x["active_s"] for x in cl]),
        "tokens_censored": qs([x["tokens"] for x in lv if x["censored"]]),
        "tokens_per_active_s_cleared": qs([x["tokens"] / x["active_s"] for x in cl if x["active_s"] and x["active_s"] > 30]),
    }
    g_full = [x for x in rows["games"] if x["kind"] == "full"]
    card["games_full"] = {}
    for run in sorted({x["run"] for x in g_full}):
        xs = [x for x in g_full if x["run"] == run]
        starts = sorted(x["first_active_s"] for x in xs if x["first_active_s"] is not None)
        card["games_full"][run] = {
            "first_active_s_sorted": [round(s) for s in starts],
            "started_after_30min": sum(1 for s in starts if s > 1800),
            "never_started": sum(1 for x in xs if x["first_active_s"] is None),
            "active_s_median": r(q([x["active_s"] for x in xs], 0.5)),
            "switch_outs_total": sum(x["n_switch_outs"] for x in xs),
            "trims_total": sum(x["n_trims"] or 0 for x in xs),
            "never_readmitted": sum(1 for v in rows["intervals"] if v["run"] == run
                                    and v["end_reason"] == "switch_out_not_readmitted"),
            "tokens_per_active_s": r(sum(x["total_tokens"] for x in xs) / max(1.0, sum(x["active_s"] for x in xs))),
        } | occupancy(rows["intervals"], run, next(x for x in rows["runs"] if x["run"] == run))
    rq = [x for x in rows["requests"] if x["t_entry_s"] is not None]
    nontrim = [x for x in rq if x["gap_before_s"] is not None and not x["trim"]]
    trims = [x for x in rq if x["trim"]]
    card["requests"] = {
        "n": len(rq), "runs": sorted({x["run"] for x in rq}),
        "completion_tokens": qs([x["completion_tokens"] for x in rq]),
        "requests_per_turn": qs(list(_per_turn_counts(rq).values())),
        "client_gap_s_nontrim": qs([x["gap_before_s"] for x in nontrim], (0.1, 0.5, 0.9, 0.99)),
        "client_gap_s_by_action_bin": {
            b: qs([x["gap_before_s"] for x in nontrim if lo <= x["action"] < hi], (0.5, 0.9))
            for b, lo, hi in (("0-49", 0, 50), ("50-99", 50, 100), ("100-199", 100, 200),
                              ("200-299", 200, 300), ("300+", 300, 10 ** 6))},
        "client_gap_s_by_prompt_bin": {
            b: qs([x["gap_before_s"] for x in nontrim if lo <= x["prompt_tokens"] < hi], (0.5, 0.9))
            for b, lo, hi in (("<30k", 0, 30_000), ("30-60k", 30_000, 60_000), ("60-90k", 60_000, 90_000),
                              ("90k+", 90_000, 10 ** 7))},
        "client_gap_s_by_game_median": {
            g: r(q([x["gap_before_s"] for x in nontrim if x["game"] == g], 0.5))
            for g in sorted({x["game"] for x in nontrim})},
        "trim_gap_s": qs([x["gap_before_s"] for x in trims if x["gap_before_s"] is not None], (0.1, 0.25, 0.5, 0.75, 0.9)),
        "trim_switch_out_share": r(sum(x["switched_out"] for x in trims) / len(trims)) if trims else None,
        "trim_reprefill_uncached": qs([x["uncached_tokens"] for x in trims]),
        "trim_reprefill_s": qs([x["prefill_s"] for x in trims]),
    }
    rs = rows["reqstats"]
    big = [x for x in rs if x["input"] >= 2000]
    card["serving"] = {
        "queue_ms": qs([x["queue_ms"] for x in big], (0.5, 0.9, 0.99)),
        "prefill_tps_by_uncached": {
            b: qs([(x["input"] - x["cached"]) / x["prefill_s"] for x in big
                   if lo <= x["input"] - x["cached"] < hi and x["prefill_s"] > 0], (0.25, 0.5, 0.75))
            for b, lo, hi in (("<1k", 0, 1000), ("1-4k", 1000, 4000), ("4-16k", 4000, 16000),
                              ("16-48k", 16000, 48000), ("48k+", 48000, 10 ** 7))},
        "prefill_s_by_uncached": {
            b: qs([x["prefill_s"] for x in big if lo <= x["input"] - x["cached"] < hi], (0.5, 0.9))
            for b, lo, hi in (("<1k", 0, 1000), ("1-4k", 1000, 4000), ("4-16k", 4000, 16000),
                              ("16-48k", 16000, 48000), ("48k+", 48000, 10 ** 7))},
        "request_stream_tps_by_running": {},
        "decode_lines_by_running": {},
    }
    for n in range(1, 15):
        v = [x["output"] / x["decode_s"] for x in big if x["output"] >= 256 and x["decode_s"] > 0
             and x["running_mean"] is not None and n - 0.5 <= x["running_mean"] < n + 0.5]
        if len(v) >= 20:
            card["serving"]["request_stream_tps_by_running"][n] = qs(v, (0.25, 0.5, 0.75))
    for x in rows["serve_decode"]:
        if x["run"] in FRANZEN_FULL and x["n_lines"] >= 30:
            card["serving"]["decode_lines_by_running"].setdefault(x["run"], {})[x["running_req"]] = {
                "agg_median": r(x["agg_tps_median"]), "per_stream_median": r(x["per_stream_tps_median"]),
                "accept": r(x["accept_len_median"]), "n": x["n_lines"]}
    std10 = {x.name for x in RUNS if x.kind == "full" and x.slots == 10 and x.name != "fz-experts16-full"}
    pooled = defaultdict(lambda: [0, 0.0, 0.0, []])
    for x in rows["serve_decode"]:
        if x["run"] in std10:
            p = pooled[x["running_req"]]
            p[0] += x["n_lines"]
            p[1] += x["agg_tps_mean"] * x["n_lines"]
            p[3].append(x["accept_len_median"])
    card["serving"]["decode_lines_pooled_std10"] = {
        n: {"lines": v[0], "agg_tps_mean": r(v[1] / v[0]), "per_stream_tps_mean": r(v[1] / v[0] / n),
            "accept_len_median_of_runs": r(q(v[3], 0.5))}
        for n, v in sorted(pooled.items()) if v[0] >= 50}
    card["serving"]["mean_running_req_by_run"] = {}
    for run in sorted({x["run"] for x in rows["serve_minutes"]}):
        mins = [x for x in rows["serve_minutes"] if x["run"] == run and x["running_mean"] is not None]
        if mins:
            card["serving"]["mean_running_req_by_run"][run] = {
                "mean": r(statistics.fmean(x["running_mean"] for x in mins)),
                "agg_tps_mean": r(statistics.fmean(x["agg_tps_mean"] for x in mins)),
                "gen_tokens_per_s_finished": r(sum(x["gen_tokens_finished"] for x in mins) / (60 * len(mins)))}
    sg = rows["segments"]
    card["context"] = {}
    card["context"]["by_run_steady_gen_tokens_p50"] = {
        run: {"p50": r(q([x["gen_tokens"] for x in sg if x["run"] == run and x["ended_by_trim"] and x["idx"] > 0], 0.5)),
              "busy_s_p50": r(q([x["busy_s"] for x in sg if x["run"] == run and x["ended_by_trim"] and x["idx"] > 0], 0.5)),
              "n": sum(1 for x in sg if x["run"] == run and x["ended_by_trim"] and x["idx"] > 0)}
        for run in sorted({x["run"] for x in sg})}
    for method in ("requests", "transcript"):
        xs = [x for x in sg if x["method"] == method and x["ended_by_trim"] and x["idx"] > 0]
        first = [x for x in sg if x["method"] == method and x["ended_by_trim"] and x["idx"] == 0]
        card["context"][method] = {
            "steady_segments": len(xs),
            "gen_tokens_per_segment": qs([x["gen_tokens"] for x in xs]),
            "turns_per_segment": qs([x["n_turns"] for x in xs]),
            "requests_per_segment": qs([x["n_requests"] for x in xs]),
            "actions_per_segment": qs([x["n_actions"] for x in xs]),
            "busy_s_per_segment": qs([x["busy_s"] for x in xs]),
            "first_segment_gen_tokens": qs([x["gen_tokens"] for x in first]),
            "first_segment_requests": qs([x["n_requests"] for x in first]),
        }
        if method == "requests":
            card["context"][method] |= {
                "prompt_start": qs([x["prompt_start"] for x in xs]),
                "prompt_peak": qs([x["prompt_peak"] for x in xs]),
                "prompt_growth": qs([x["prompt_growth"] for x in xs]),
                "growth_per_gen_token": qs([x["prompt_growth"] / x["gen_tokens"] for x in xs if x["gen_tokens"]]),
            }
    val = summary.get("validation", {})
    card["validation"] = {
        run: {
            "switch_outs_exact": len(v["exact"]), "switch_outs_transcript": len(v["transcript"]),
            "switch_outs_history": len(v["history"]),
            "wait_s_exact": qs(v["exact"], (0.25, 0.5, 0.75)),
            "wait_s_transcript": qs(v["transcript"], (0.25, 0.5, 0.75)),
            "wait_s_history": qs(v["history"], (0.25, 0.5, 0.75)),
            "wait_total_s": {"exact": r(sum(v["exact"])), "transcript": r(sum(v["transcript"])),
                             "history": r(sum(v["history"]))},
        } for run, v in val.items()}
    card["score_check"] = {x["run"]: {"recomputed": r(x["score_mean"]), "harness": r(x["score_harness"])}
                           for x in rows["runs"]}
    return card


def occupancy(intervals: list[dict], run: str, run_row: dict) -> dict:
    """Slot occupancy implied by the reconstructed intervals (a check on the timeline):
    sum of holding time over slots x deadline, and concurrent holders sampled every 10 s."""
    xs = [v for v in intervals if v["run"] == run]
    horizon = run_row["deadline_s"]
    held = sum(min(v["end_s"], horizon) - v["start_s"] for v in xs if v["start_s"] < horizon)
    counts = []
    for t in range(0, int(horizon), 10):
        counts.append(sum(1 for v in xs if v["start_s"] <= t < v["end_s"]))
    return {"slot_occupancy": r(held / (run_row["slots"] * horizon), 3),
            "concurrent_p50": q(counts, 0.5), "concurrent_p95": q(counts, 0.95),
            "concurrent_max": max(counts) if counts else None,
            "share_time_over_slots": r(sum(1 for c in counts if c > run_row["slots"]) / max(1, len(counts)), 3)}


def _per_turn_counts(rq):
    """Requests per turn; rows are in per-game order and a turn starts at req_in_turn == 1."""
    turn = defaultdict(int)
    out = defaultdict(int)
    for x in rq:
        k = (x["run"], x["game"])
        if x["req_in_turn"] == 1:
            turn[k] += 1
        out[(k, turn[k])] += 1
    return out


def write_games_meta(root: Path, out: Path, only: set[str] | None = None) -> dict:
    """games_meta.json: per game its level count and human baselines, and per run the
    dispatch order (benchmark.json game_runs order = solver game_index) and final scores.

    The simulator needs baselines for levels no run reached, which levels.csv lacks.
    Baselines are game constants; a run that disagrees with the first one raises.
    """
    games: dict = {}
    runs: dict = {}
    for run in RUNS:
        if only and run.name not in only:
            continue
        path = root / run.name / "benchmark.json"
        if not path.exists():
            continue
        bench = load_benchmark(root / run.name)
        order = []
        for g in bench["game_runs"]:
            short = g["game_id"][:4]
            order.append(short)
            meta = {"game_id": g["game_id"], "levels": int(g["number_of_levels"]),
                    "baselines": [int(b) for b in g["base_actions_per_level"]]}
            if short in games and games[short] != meta:
                raise ValueError(f"{run.name}: {short} metadata differs from an earlier run")
            games[short] = meta
        runs[run.name] = {
            "dispatch_order": order,
            "final_score": {g["game_id"][:4]: g.get("final_score") for g in bench["game_runs"]},
            "levels_completed": {g["game_id"][:4]: int(g["levels_completed"]) for g in bench["game_runs"]},
        }
    meta = {"games": dict(sorted(games.items())), "runs": runs}
    (out / "games_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=HERE / "calibration")
    ap.add_argument("--runs", nargs="*", help="only these runs (for testing)")
    a = ap.parse_args()
    build(a.root, a.out, set(a.runs) if a.runs else None)
    write_games_meta(a.root, a.out, set(a.runs) if a.runs else None)


if __name__ == "__main__":
    main()
