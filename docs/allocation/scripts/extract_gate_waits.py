"""Gate waits and grant sizes for one run, plus vLLM throughput.

Waits are inferred from transcript step timestamps: a gap longer than
--gap seconds between consecutive analysis steps counts as time spent waiting
for an admission slot (a long single generation can also produce such a gap).
Tokens per grant split the game's generated tokens in proportion to the
reasoning + content characters of the steps between waits.

Writes data/gate_waits_<run>.csv and data/vllm_throughput_<run>.json.

Usage:
  python docs/allocation/scripts/extract_gate_waits.py \
      --run-dir ARC3-Inference/runs/kaggle/jonathanwang2022/bp35-lordhansolo/v07
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
HEADER = re.compile(r"^--- analysis_step=\d+ \| action=\d+ \| (\d\d:\d\d:\d\d) \|")
CHARS = re.compile(r"^(reasoning|content)_chars: (\d+)")
VLLM = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s")
HIT = re.compile(r"Prefix cache hit rate: ([\d.]+)%")


def game_rows(run_dir: Path, gap: float) -> list[dict]:
    ev = json.loads((run_dir / "evaluation.json").read_text())
    rows = []
    for path in sorted((run_dir / "transcripts").glob("*_p0.txt")):
        game = next(g for g in ev["games"] if g.startswith(path.name[:4]))
        tokens = ev["games"][game]["generated_tokens_mean"]
        stamps, chars, cur = [], [], 0
        for line in path.open(errors="ignore"):
            m = HEADER.match(line)
            if m:
                t = datetime.strptime(m.group(1), "%H:%M:%S")
                if stamps and t < stamps[-1] - timedelta(hours=1):
                    t += timedelta(days=1)
                stamps.append(t); chars.append(cur); cur = 0
            m = CHARS.match(line)
            if m:
                cur += int(m.group(2))
        chars = chars[1:] + [cur]
        total = sum(chars) or 1
        gaps = [(stamps[i] - stamps[i - 1]).total_seconds() for i in range(1, len(stamps))]
        waits = [i for i, x in enumerate(gaps) if x > gap]
        cuts = [0] + [i + 1 for i in waits] + [len(stamps)]
        grants = [sum(chars[a:b]) / total * tokens for a, b in zip(cuts[:-1], cuts[1:])]
        rows.append({
            "game": game, "levels": ev["games"][game]["levels_completed_mean"],
            "total_levels": ev["games"][game]["total_levels"], "score": ev["games"][game]["score"],
            "generated_tokens": int(tokens), "active_minutes": round((stamps[-1] - stamps[0]).total_seconds() / 60, 1),
            "waits": len(waits), "waited_minutes": round(sum(gaps[i] for i in waits) / 60, 1),
            "grant_tokens": " ".join(str(round(g)) for g in grants),
        })
    return rows


def throughput(run_dir: Path) -> dict:
    prompt, gen, hit = [], [], []
    for line in (run_dir / "vllm-openai-server.log").open(errors="ignore"):
        m = VLLM.search(line)
        if m:
            prompt.append(float(m.group(1))); gen.append(float(m.group(2)))
        m = HIT.search(line)
        if m:
            hit.append(float(m.group(1)))
    busy = [p for p in prompt if p > 0]
    return {
        "intervals": len(gen),
        "generation_tokens_per_s_median": st.median(gen),
        "prompt_tokens_per_s_median_when_busy": st.median(busy),
        "prompt_tokens_per_s_max": max(busy),
        "prefix_cache_hit_rate_pct_median": st.median(hit) if hit else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--gap", type=float, default=150.0, help="seconds between steps that count as a wait")
    args = ap.parse_args()
    tag = args.run_dir.name
    rows = game_rows(args.run_dir, args.gap)
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / f"gate_waits_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (DATA / f"vllm_throughput_{tag}.json").write_text(json.dumps(throughput(args.run_dir), indent=1))
    grants = [float(x) for r in rows for x in r["grant_tokens"].split()]
    waited = sum(r["waited_minutes"] for r in rows); active = sum(r["active_minutes"] for r in rows)
    print(f"{len(rows)} games; waited {waited:.0f} of {active:.0f} game-minutes; "
          f"{len(grants)} grants, median {st.median(grants):.0f} tokens")


if __name__ == "__main__":
    main()
