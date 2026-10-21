"""Measure, from real request logs, how many prompt tokens each prefill-v2 lever removes.

Reads every `<game>_p0_requests.jsonl[.xz]` of a run. Each line is a request snapshot (the
full message list as sent) or a response (usage). For every request it works out which
messages are new since the game's previous request (trims drop from the front, so the new
messages are what is left after aligning the previous history), classifies the new user
message as a fresh turn opener, a yield resumption opener (same analysis_step as the request
before it) or a mid-turn nudge, and counts tokens in them.

Levers measured (see research/archive/reports/prefill-v2.md):
- static:  `ARC3_OPENER_DEDUP=static`. The opener's static tail (the lines from "Only tool:"
           to the end of the prompt text, which the system prompt already carries) is sent
           only when no retained message carries the identical block. Carriers are tracked on
           the run's own retained windows, so a carrier dropped by a trim makes the next
           opener carry the block again.
- image:   `ARC3_OPENER_DEDUP=resume_image`. A yield resumption omits its current-grid image
           when the most recent retained image-bearing message belongs to the same analysis
           step (nothing executed since, so the board is identical) and adds a one-line
           caption instead.
- yield B: `LOCAL_ANALYZER_YIELD_TOKENS=B`. The yield points of every logical turn are
           replayed from the observed completion tokens (a turn yields at the first loop top
           where its invocation has generated >= B tokens); the model's behaviour is held
           fixed. Removing a resumption removes its opener, but a resumption that followed a
           reply with no tool call is replaced by the no-tool-call nudge. New resumptions are
           charged the game's median resumption opener.
- yield S s: `LOCAL_ANALYZER_YIELD_SECONDS=S` (tokens off), replayed on the requests'
           server entry times (requests.csv; fz-ledger-on-trim-r2 only).

Output: a per-run summary (`--json`) and, with `--table`, one row per request of the run
for the simulator (`sim2.py --lever`): the prompt increment the request would carry under
each lever relative to the run, and whether it would open an invocation.

Token counts use the Qwen3.8-Flash-Next tokenizer when `--tokenizer` points at its
tokenizer.json and the `tokenizers` package is importable; otherwise 3.9 characters per token
(the tokenizer's ratio on these openers). An image is 400 tokens (usage image_tokens per
attached 640x640 image).

Usage (from the repo root):
    nice -n 15 python3 analysis/prefill_levers.py runs/kaggle/fz-ledger-on-trim-r2/v1 \\
        --tokenizer ~/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/snapshots/*/tokenizer.json \\
        --requests-csv analysis/calibration/requests.csv --table analysis/calibration/prefill_levers.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import lzma
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_TOKENS = 400
CHARS_PER_TOKEN = 3.9
STATIC_START = "Only tool: `python`."
# the caption the resume_image lever puts where the image was (tool_agent.py
# _RESUME_IMAGE_OMITTED_CAPTION); counted against the saving
RESUME_IMAGE_CAPTION = (
    "Current grid image: unchanged since the last grid image above "
    "(nothing has executed), so it is not repeated."
)
# the no-tool-call nudge ("You have not acted yet. ...", memory sections off), tokenized;
# used where a game logged none (bare replies are rare: 31 of 2,245 replies in r2)
NUDGE_TOKENS = 300
YIELD_BUDGETS = (2048, 4096, 6144, 8192, 12288)
YIELD_SECONDS = (60, 90, 120, 150)


class Tok:
    def __init__(self, path: str | None):
        self.tk = None
        if path:
            try:
                from tokenizers import Tokenizer
                self.tk = Tokenizer.from_file(path)
            except Exception as exc:  # noqa: BLE001
                print(f"tokenizer unavailable ({exc}); using {CHARS_PER_TOKEN} chars/token",
                      file=sys.stderr)
        self.cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        if self.tk is None:
            return round(len(text) / CHARS_PER_TOKEN)
        n = self.cache.get(text)
        if n is None:
            n = len(self.tk.encode(text, add_special_tokens=False).ids)
            if len(self.cache) < 200_000:
                self.cache[text] = n
        return n


def message_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if not c:
        return ""
    return "\n".join(p.get("text") or "" for p in c if isinstance(p, dict) and p.get("type") == "text")


def message_images(m: dict) -> int:
    c = m.get("content")
    if not isinstance(c, list):
        return 0
    return sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")


def static_block(text: str) -> str | None:
    """The opener's static tail as the harness builds it: from "Only tool:" to the end of the
    prompt text, which ends at the first blank line (retained functions or image caption)."""
    i = text.find(STATIC_START)
    if i < 0:
        return None
    j = text.find("\n\n", i)
    return text[i:] if j < 0 else text[i:j]


@dataclass
class Msg:
    id: int
    role: str
    step: int | None
    images: int
    block: str | None = None
    carrier: bool = False          # carries the static block in the dedup world


@dataclass
class Req:
    game: str
    step: int
    action: int
    req_idx: int
    n_msgs: int
    new_roles: list
    kind: str = ""                 # fresh | resume | nudge | "" (continuation)
    trim: bool = False
    prompt: int = 0
    cached: int = 0
    completion: int = 0
    finish: str = ""
    tool_call: bool = False
    opener_tokens: int = 0         # text + images of the new user message
    opener_images: int = 0
    static_tokens: int = 0         # static block in this opener (0 if none)
    static_saved: int = 0          # dedup world: tokens the static lever removes here
    image_saved: int = 0           # dedup world: tokens the resume_image lever removes here
    nudge_tokens: int = 0
    dp: int | None = None          # prompt increment since the previous request (non-trim)


def read_events(path: str):
    op = lzma.open if path.endswith(".xz") else open
    with op(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def scan_game(path: str, tok: Tok) -> list[Req]:
    game = Path(path).name.split("-")[0]
    prev_hashes: list[str] = []
    prev_ids: list[int] = []
    msgs: dict[int, Msg] = {}
    next_id = 0
    reqs: list[Req] = []
    last_step = None
    for ev in read_events(path):
        if ev.get("event") == "response":
            if reqs and reqs[-1].prompt == 0:
                r = reqs[-1]
                u = ev.get("usage") or {}
                r.prompt = int(u.get("prompt_tokens") or 0)
                r.cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                r.completion = int(u.get("completion_tokens") or 0)
                r.finish = str(ev.get("finish_reason") or "")
                r.tool_call = r.finish == "tool_calls"
            continue
        if ev.get("event") != "request":
            continue
        messages = ev["messages"]
        hist = messages[1:]
        hashes = [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest() for m in hist]
        # align: the new history starts with a suffix of the previous one (front trims)
        keep = 0
        start = None
        if hashes:
            for j in range(len(prev_hashes)):
                if prev_hashes[j] == hashes[0] and prev_hashes[j:] == hashes[:len(prev_hashes) - j]:
                    start = j
                    break
        if start is not None:
            keep = len(prev_hashes) - start
            ids = prev_ids[start:]
        else:
            ids = []
        trim = start is None or start > 0
        new = hist[keep:]
        step = int(ev.get("analysis_step") or 0)
        req_idx = int(ev.get("request_index_within_turn") or 0)
        r = Req(game=game, step=step, action=int(ev.get("action") or 0), req_idx=req_idx,
                n_msgs=len(messages), new_roles=[m.get("role") for m in new], trim=trim and bool(reqs))
        new_ids = []
        for m in new:
            mm = Msg(id=next_id, role=str(m.get("role")), step=step, images=message_images(m))
            next_id += 1
            msgs[mm.id] = mm
            new_ids.append(mm.id)
        retained = ids
        if new and new[-1].get("role") == "user":
            m = new[-1]
            mm = msgs[new_ids[-1]]
            text = message_text(m)
            if req_idx == 1:
                r.kind = "resume" if (last_step == step and reqs) else "fresh"
                r.opener_images = mm.images
                r.opener_tokens = tok(text) + IMAGE_TOKENS * mm.images
                blk = static_block(text)
                if blk:
                    mm.block = blk
                    r.static_tokens = tok("\n" + blk)
                    carried = any(msgs[i].carrier and msgs[i].block == blk for i in retained)
                    if carried:
                        r.static_saved = r.static_tokens
                    else:
                        mm.carrier = True
                if r.kind == "resume" and mm.images:
                    last_img = next((msgs[i] for i in reversed(retained) if msgs[i].images), None)
                    if last_img is not None and last_img.step == step:
                        r.image_saved = IMAGE_TOKENS - tok("\n\n" + RESUME_IMAGE_CAPTION) + tok("\n\nCurrent grid image:")
            else:
                r.kind = "nudge"
                r.nudge_tokens = tok(text)
        prev_hashes = hashes
        prev_ids = ids + new_ids
        last_step = step
        reqs.append(r)
    for a, b in zip(reqs, reqs[1:]):
        if not b.trim and b.prompt and a.prompt:
            b.dp = b.prompt - a.prompt
    return reqs


# ---------------------------------------------------------------------------- yield replay

def logical_turns(reqs: list[Req]) -> list[list[int]]:
    """Indices of consecutive requests sharing an analysis_step (one logical turn)."""
    turns, cur = [], []
    for i, r in enumerate(reqs):
        if cur and reqs[cur[-1]].step != r.step:
            turns.append(cur)
            cur = []
        cur.append(i)
    if cur:
        turns.append(cur)
    return turns


def replay_yields(reqs: list[Req], budget: int | None = None, seconds: float | None = None,
                  t_entry: list[float | None] | None = None) -> list[bool]:
    """resume[i]: under this yield setting, request i opens a resumed invocation."""
    out = [False] * len(reqs)
    for turn in logical_turns(reqs):
        acc = 0
        inv_start = t_entry[turn[0]] if t_entry else None
        for a, b in zip(turn, turn[1:]):
            acc += reqs[a].completion
            y = False
            if budget is not None and acc >= budget:
                y = True
            if seconds is not None and t_entry is not None:
                tb = t_entry[b]
                if tb is not None and inv_start is not None and tb - inv_start >= seconds:
                    y = True
            if y:
                out[b] = True
                acc = 0
                if t_entry is not None:
                    inv_start = t_entry[b]
    return out


# ---------------------------------------------------------------------------- levers

def lever_deltas(reqs: list[Req], resume_cf: list[bool], dedup_static: bool, dedup_image: bool,
                 median_resume: dict) -> list[int]:
    """Change in each request's prompt increment (new uncached tokens) under a lever setting."""
    out = []
    for i, r in enumerate(reqs):
        d = 0
        if r.kind == "fresh":
            if dedup_static:
                d -= r.static_saved
        actual_resume = r.kind == "resume"
        cf = resume_cf[i]
        prev_bare = i > 0 and not reqs[i - 1].tool_call
        if actual_resume and cf:
            if dedup_static:
                d -= r.static_saved
            if dedup_image:
                d -= r.image_saved
        elif actual_resume and not cf:
            d -= r.opener_tokens
            if prev_bare:
                d += median_resume["nudge"]
        elif cf and not actual_resume:
            d += median_resume[(dedup_static, dedup_image)]
            if r.kind == "nudge":
                d -= r.nudge_tokens
        out.append(d)
    return out


def medians(reqs: list[Req]) -> dict:
    res = [r for r in reqs if r.kind == "resume"]
    nud = [r.nudge_tokens for r in reqs if r.kind == "nudge"] or [NUDGE_TOKENS]
    out = {"nudge": round(st.median(nud))}
    for s in (False, True):
        for im in (False, True):
            vals = [r.opener_tokens - (r.static_saved if s else 0) - (r.image_saved if im else 0)
                    for r in res] or [0]
            out[(s, im)] = round(st.median(vals))
    return out


def load_t_entry(requests_csv: str, run: str) -> dict:
    """(game, step, action, req_in_turn, occurrence) -> server entry s, from calibration."""
    out, seen = {}, defaultdict(int)
    with open(requests_csv) as fh:
        for row in csv.DictReader(fh):
            if row["run"] != run:
                continue
            k = (row["game"], int(row["step"]), int(row["action"]), int(row["req_in_turn"]))
            occ = seen[k]
            seen[k] += 1
            out[k + (occ,)] = float(row["t_entry_s"]) if row["t_entry_s"] else None
    return out


def analyze_run(run_dir: str, tok: Tok, requests_csv: str | None = None):
    run_dir, _, name = run_dir.partition("=")
    run_dir = run_dir.rstrip("/")
    run = name or Path(run_dir).parent.name
    files = sorted(glob.glob(f"{run_dir}/*_p0_requests.jsonl*"))
    t_map = load_t_entry(requests_csv, run) if requests_csv else {}
    games = {}
    for f in files:
        reqs = scan_game(f, tok)
        if reqs:
            games[reqs[0].game] = reqs
    summary = {"run": run, "games": len(games)}
    tot = defaultdict(float)
    rows = []
    variants = {}
    for game, reqs in games.items():
        # requests.csv holds responses only, so count occurrences over answered requests
        seen = defaultdict(int)
        t_entry = []
        for r in reqs:
            k = (game, r.step, r.action, r.req_idx)
            if r.prompt:
                t_entry.append(t_map.get(k + (seen[k],)))
                seen[k] += 1
            else:
                t_entry.append(None)
        timed = sum(t is not None for t in t_entry) >= 0.9 * sum(1 for r in reqs if r.prompt)
        med = medians(reqs)
        actual = [r.kind == "resume" for r in reqs]
        cf = {f"y{b}": replay_yields(reqs, budget=b) for b in YIELD_BUDGETS}
        if timed:
            for s in YIELD_SECONDS:
                cf[f"t{s}"] = replay_yields(reqs, seconds=s, t_entry=t_entry)
        cf["actual"] = actual
        # lever settings: (yield key, static, image)
        settings = {
            "static": ("actual", True, False),
            "image": ("actual", False, True),
            "dedup": ("actual", True, True),
        }
        for k in cf:
            if k.startswith(("y", "t")):
                settings[k] = (k, False, False)
                settings[f"{k}+dedup"] = (k, True, True)
        deltas = {name: lever_deltas(reqs, cf[yk], s, im, med) for name, (yk, s, im) in settings.items()}
        variants.update({name: None for name in settings})
        for i, r in enumerate(reqs):
            tot["requests"] += 1
            tot["prompt"] += r.prompt
            tot["uncached"] += r.prompt - r.cached
            tot["completion"] += r.completion
            tot["trims"] += r.trim
            tot["growth"] += r.dp or 0
            tot[f"kind_{r.kind or 'cont'}"] += 1
            if r.kind in ("fresh", "resume"):
                tot[f"{r.kind}_opener_tokens"] += r.opener_tokens
                tot[f"{r.kind}_static_tokens"] += r.static_tokens
                tot[f"{r.kind}_images"] += r.opener_images
            tot["static_saved"] += r.static_saved
            tot["image_saved"] += r.image_saved
            tot["nudge_tokens"] += r.nudge_tokens
            for name, d in deltas.items():
                tot[f"delta_{name}"] += d[i]
            for k, flags in cf.items():
                tot[f"resumes_{k}"] += flags[i]
            rows.append({"run": run, "game": game, "idx": i, "step": r.step, "action": r.action,
                         "req_in_turn": r.req_idx, "answered": int(bool(r.prompt)),
                         "kind": r.kind, "trim": int(r.trim),
                         "t_entry_s": "" if t_entry[i] is None else t_entry[i],
                         "prompt": r.prompt, "completion": r.completion,
                         "opener_tokens": r.opener_tokens, "static_saved": r.static_saved,
                         "image_saved": r.image_saved,
                         **{f"d_{name}": d[i] for name, d in deltas.items()},
                         **{f"open_{k}": int(flags[i] or r.kind == "fresh") for k, flags in cf.items()
                            if k != "actual"}})
    summary.update({k: round(v, 1) for k, v in sorted(tot.items())})
    summary["variants"] = sorted(variants)
    return summary, rows


def report(s: dict) -> str:
    out = []
    u = s["uncached"]
    g = s["growth"]
    out.append(f"## {s['run']} ({s['games']} games, {int(s['requests'])} requests, "
               f"{s['trims']:.0f} trims)")
    out.append(f"prompt {s['prompt']/1e6:.1f}M, uncached {u/1e6:.2f}M, completion "
               f"{s['completion']/1e6:.2f}M, non-trim prompt growth {g/1e6:.2f}M")
    out.append(f"fresh openers {s.get('kind_fresh', 0):.0f} (mean {s.get('fresh_opener_tokens', 0)/max(1, s.get('kind_fresh', 1)):.0f} tok, "
               f"static {s.get('fresh_static_tokens', 0)/max(1, s.get('kind_fresh', 1)):.0f}); resumes "
               f"{s.get('kind_resume', 0):.0f} (mean {s.get('resume_opener_tokens', 0)/max(1, s.get('kind_resume', 1)):.0f} tok, "
               f"{s.get('resume_images', 0):.0f} images); nudges {s.get('kind_nudge', 0):.0f}")
    out.append("| lever | resumes | prompt tokens removed | % of uncached | % of growth |")
    out.append("|---|---|---|---|---|")
    for name in s["variants"]:
        d = -s[f"delta_{name}"]
        yk = name.split("+")[0]
        res = s.get(f"resumes_{yk}", s.get("resumes_actual"))
        out.append(f"| {name} | {res:.0f} | {d/1e6:.3f}M | {100*d/u:.1f}% | {100*d/g:.1f}% |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run output dirs; DIR=NAME names the run "
                    "(default: the parent directory's name)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--requests-csv", default=None,
                    help="calibration requests.csv, for the time-based yield replay (entry times)")
    ap.add_argument("--table", default=None, help="write the per-request lever table here")
    ap.add_argument("--json", default=None, help="write the summaries here")
    a = ap.parse_args(argv)
    tok = Tok(glob.glob(a.tokenizer)[0] if a.tokenizer and glob.glob(a.tokenizer) else a.tokenizer)
    summaries, all_rows = [], []
    for d in a.run_dirs:
        s, rows = analyze_run(d, tok, a.requests_csv)
        summaries.append(s)
        all_rows += rows
        print(report(s))
        print()
    if a.table:
        with open(a.table, "w", newline="") as fh:
            names = list(dict.fromkeys(k for row in all_rows for k in row))
            w = csv.DictWriter(fh, fieldnames=names, restval="")
            w.writeheader()
            w.writerows(all_rows)
    if a.json:
        Path(a.json).write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
