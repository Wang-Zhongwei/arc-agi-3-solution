#!/usr/bin/env python3
"""Build the inputs for labelling run v06's pinned rules: its real lines, verifier batches and reader sections.

Lines come from the last copy of the section in each v06 transcript, so they are what the run actually showed.
Each line is mapped back to its thinking to get the next sentence and the v7-tidied text that rule v9 filters.
Writes to <v06 run>/rules_audit: claims_v06.json, batches_v06.json, batch_ids_v06.json, cold_args_v06.json, cross_runs_v06.json.
"""
import collections as C
import glob
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[4]
RUNS = REPO / "ARC3-Inference/runs/kaggle"
RUN = pathlib.Path(os.path.realpath(RUNS / "jonathanwang2022/main-tuned-drafter/v06"))
OUT = RUN / "rules_audit"
HEADER = "## Confirmed and ruled-out rules from trimmed turns"
BLOCK = re.compile(r"^(\[(THINKING|TOOL CALL: \w+|TOOL RESULT: \w+|ASSISTANT|ANALYZER STATUS|USER PROMPT|SYSTEM PROMPT|MODEL RESPONSE META)\]|--- analysis_step=(\d+).*---)\s*$")
STATE = re.compile(r"Current state: step (\d+), level (\d+)\.")


def code_at(commit):
    """v7's extraction, read from Git so later edits to confirmed_rules.py do not change this."""
    src = subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:ARC3-Inference/inference/agent/confirmed_rules.py"])
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
    tmp.write(src)
    tmp.close()
    spec = importlib.util.spec_from_file_location("confirmed_rules_v7", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V7 = code_at("a71991a2")
squash = lambda t: re.sub(r"[\s\-*#>•]+", "", t)


def read_blocks(path):
    blocks, sec, buf, start, level, step, top = [], None, [], 0, 1, 0, 1
    for n, line in enumerate(open(path, encoding="utf-8", errors="replace").read().split("\n"), 1):
        m = BLOCK.match(line)
        if not m:
            buf.append(line)
            continue
        text = "\n".join(buf)
        if sec == "USER PROMPT" and STATE.findall(text):
            step, level = map(int, STATE.findall(text)[-1])
            top = max(top, level)
        blocks.append((sec, start, text, step))
        sec, buf, start = m.group(2), [], n
    blocks.append((sec, start, "\n".join(buf), step))
    return blocks, top


def section_lines(blocks):
    last = next((b[2][b[2].rfind(HEADER):] for b in reversed(blocks) if HEADER in b[2]), None)
    items, level, kind = [], None, None
    for ln in (last or "").split("\n"):
        m = re.match(r"Level (\d+):", ln)
        if m:
            level, kind = int(m.group(1)), None
        elif ln.startswith("Confirmed:"):
            kind = "confirmed"
        elif ln.startswith("Ruled out:"):
            kind = "refuted"
        elif ln.startswith("- ") and kind:
            items.append((level, kind, ln[2:]))
    return items


claims = []
for path in sorted(glob.glob(str(RUN / "transcripts/*.txt"))):
    game = os.path.basename(path).split("-")[0]
    blocks, top = read_blocks(path)
    segs = [(b, p) for b in blocks if b[0] == "THINKING" for p in V7._segments(b[2], True)]
    keys = [squash(p) for _, p in segs]
    used = set()
    for level, kind, line in section_lines(blocks):
        key = squash(line.rstrip("…"))[:45]
        tail = squash(" ".join(line.split()[1:]))[:40]
        hits = [i for i, k in enumerate(keys) if key in k] or [i for i, k in enumerate(keys) if tail and tail in k]
        if not hits:
            raise SystemExit(f"{game}: line not found in thinking: {line[:80]}")
        i = next((h for h in hits if h not in used), hits[0])
        used.add(i)
        block, piece = segs[i]
        after = [segs[i + k][1].strip() for k in (1, 2) if i + k < len(segs) and segs[i + k][0] is block]
        tidied = [s for k, s in V7.classified_sentences(piece, confirmed=True, refuted=True) if k == kind]
        claims.append(dict(game=game, level=level, step=block[3], kind=kind, in_section=True, shown=line, v7_text=tidied[0] if tidied else None,
                           transcript=path, thinking_line=block[1], raw=piece.strip()[:900], next1=after[0][:600] if after else None,
                           next2=after[1][:600] if len(after) > 1 else None, max_level_seen=top))
for n, c in enumerate(claims):
    c["id"] = f"{c['game']}-{n:03d}"

# other runs of each game, for the cross-run verifier
index = C.defaultdict(list)
for bench in glob.glob(str(RUNS / "*/*/v*/benchmark.json")):
    run = pathlib.Path(bench).parent
    if os.path.realpath(run) == str(RUN):
        continue
    try:
        games = json.load(open(bench))["game_runs"]
    except (OSError, ValueError, KeyError):
        continue
    for g in games:
        gid = g["game_id"]
        tr = run / "transcripts" / f"{gid}_p0.txt"
        if tr.exists():
            index[gid.split("-")[0]].append(dict(run=str(run.relative_to(RUNS)), levels_completed=g.get("levels_completed") or 0, transcript=str(tr), events=str(run / "artifacts" / f"{gid}_p0_events.jsonl")))
for runs in index.values():
    runs.sort(key=lambda r: -r["levels_completed"])

OUT.mkdir(exist_ok=True)
json.dump(claims, open(OUT / "claims_v06.json", "w"), indent=1, ensure_ascii=False)
json.dump(index, open(OUT / "cross_runs_v06.json", "w"))
bench = {g["game_id"].split("-")[0]: g for g in json.load(open(RUN / "benchmark.json"))["game_runs"]}
by_game = C.defaultdict(list)
for c in claims:
    by_game[c["game"]].append(c)
batches, cold = [], []
keep = ["id", "kind", "in_section", "level", "step", "thinking_line", "shown", "raw", "next1", "next2"]
for game, cs in sorted(by_game.items()):
    b = bench[game]
    runs = index.get(game, [])
    n_levels = int(b.get("number_of_levels") or 0)
    ctx = dict(game=game, game_id=b["game_id"], this_run_transcript=cs[0]["transcript"], this_run_events=str(RUN / "artifacts" / f"{b['game_id']}_p0_events.jsonl"),
               this_run_levels_completed=b.get("levels_completed"), number_of_levels=n_levels, this_run_actions_per_level=b.get("actions_per_level"),
               other_runs_total=len(runs), other_runs_clearing_level={L: sum(r["levels_completed"] >= L for r in runs) for L in range(1, n_levels + 1)},
               best_other_runs=runs[:6], all_other_runs_index=str(OUT / "cross_runs_v06.json"))
    for i in range(0, len(cs), 10):
        batches.append(dict(batch_id=f"{game}-{i // 10}", context=ctx, claims=[{k: c.get(k) for k in keep} for c in cs[i:i + 10]]))
    text = [HEADER]
    for level in sorted({c["level"] for c in cs}):
        text.append(f"Level {level}:")
        for kind, title in (("confirmed", "Confirmed"), ("refuted", "Ruled out")):
            lines = [c for c in cs if c["level"] == level and c["kind"] == kind]
            if lines:
                text.append(f"  {title}:")
                text += [f"  - [{c['id']}] {c['shown']}" for c in lines]
    cold.append(dict(game=game, ids=[c["id"] for c in cs], section="\n".join(text)))
json.dump(batches, open(OUT / "batches_v06.json", "w"))
json.dump([dict(batch_id=b["batch_id"], ids=[c["id"] for c in b["claims"]]) for b in batches], open(OUT / "batch_ids_v06.json", "w"))
json.dump(cold, open(OUT / "cold_args_v06.json", "w"), ensure_ascii=False)
print("claims", len(claims), dict(C.Counter(c["kind"] for c in claims)), "games", len(by_game), "batches", len(batches), "other runs per game", sorted(len(v) for v in index.values())[:3], "...")
