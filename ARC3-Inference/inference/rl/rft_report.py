"""What an rft_data dataset covers, as Markdown: per game, per level, per run, and the validation levels.

    python -m inference.rl.rft_report DATA_DIR [--check CHECK_JSONL] > inference/rl/data/<name>.md

CHECK_JSONL is check_rft_data's --out (exact tokens per sample); without it token columns are estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    m = json.loads((args.data / "manifest.json").read_text())
    samples = [json.loads(line) for line in open(args.data / "train.jsonl")]
    exact = {}
    if args.check:
        exact = {r["id"]: r for r in map(json.loads, open(args.check))}
    run_of = lambda path: path.split("runs/", 1)[-1]
    tokens, trained, n_samples = Counter(), Counter(), Counter()
    by_run_samples = Counter()
    for s in samples:
        key = (s["game"], s["level"])
        n_samples[key] += 1
        by_run_samples[s["run"]] += 1
        r = exact.get(s["id"])
        tokens[key] += r["tokens"] if r else s["est_tokens"]
        trained[key] += r["trained"] if r else 0
    kept = [l for l in m["levels"] if l.get("kept")]
    a = m["args"]
    out = [f"# Dataset `{args.data.name}`", "",
           f"Built by [rft_data.py](../rft_data.py): levels cleared in < {float(a['max_pass_rate']):.0%} of attempts; per level, "
           f"clears within the best {float(a['top']):.0%} by actions whose generated tokens are at most the level's "
           f"{float(a['token_quantile']):.0%} quantile (thresholds over all runs' clears). Token columns are "
           + ("exact ([check_rft_data.py](../check_rft_data.py))." if exact else "estimates."), ""]
    traj = len(m["train"])
    out += [f"Train: {len(samples)} samples from {traj} trajectories (one clear of one level in one run), "
            f"{len(kept)} levels, {len({l['game'] for l in kept})} games, {len({t['run'] for t in m['train']})} runs; "
            f"{sum(tokens.values()) / 1e6:.1f}M tokens" + (f", {sum(trained.values()) / 1e6:.2f}M trained." if exact else "."), ""]

    games = defaultdict(lambda: [0, 0, 0])
    for l in kept:
        g = games[l["game"]]
        g[0] += 1
        g[1] += l["kept"]
        g[2] += n_samples[(l["game"], l["level"])]
    out += ["## Per game", "", "| Game | Levels | Trajectories | Samples |", "|---|---|---|---|"]
    out += [f"| {g} | {v[0]} | {v[1]} | {v[2]} |" for g, v in sorted(games.items(), key=lambda kv: -kv[1][2])]
    none = sorted({l["game"] for l in m["levels"]} - set(games))
    out += ["", f"No level selected in {len(none)} games ({', '.join(none)}): all their levels were cleared in "
            f"≥ {float(a['max_pass_rate']):.0%} of attempts, or no qualifying clear was in a usable run.", ""]

    out += ["## Per level", "",
            "Clears: all runs / usable runs. Caps: the action and generated-token limits a clear had to meet.", "",
            "| Level | Pass rate | Attempts | Clears | Caps (actions / tokens) | Trajectories | Samples | Tokens (trained) |",
            "|---|---|---|---|---|---|---|---|"]
    for l in kept:
        key = (l["game"], l["level"])
        tok = f"{tokens[key] / 1e6:.2f}M" + (f" ({trained[key] / 1e6:.2f}M)" if exact else "")
        out.append(f"| {l['game']} L{l['level']} | {l['pass_rate']:.2f} | {l['attempts']} | {l['clears']}/{l['eligible_clears']} | "
                   f"{l['max_actions']} / {l['max_generated_tokens'] / 1000:.0f}K | {l['kept']} | {n_samples[key]} | {tok} |")
    empty = [l for l in m["levels"] if l["clears"] and l["pass_rate"] < float(a["max_pass_rate"]) and not l.get("kept")]
    if empty:
        out += ["", "Hard levels with nothing selected: " + "; ".join(
            f"{l['game']} L{l['level']} (pass {l['pass_rate']:.2f}, {l.get('eligible_clears', 0)} clears in usable runs)"
            for l in empty) + ". With usable clears, every one used more generated tokens than the cap."]

    traj_by_run = Counter(run_of(t["run"]) for t in m["train"])
    out += ["", "## Per run", "", "| Run | Trajectories | Samples |", "|---|---|---|"]
    out += [f"| {r} | {n} | {by_run_samples[r]} |" for r, n in sorted(traj_by_run.items(), key=lambda kv: (-kv[1], kv[0]))]
    usable_unused = sorted(run_of(r) for r, p in m["runs"].items() if p == "eligible" and run_of(r) not in traj_by_run)
    if usable_unused:
        out += ["", "Usable runs with nothing selected: " + ", ".join(usable_unused) + "."]
    excluded = Counter(p for p in m["runs"].values() if p != "eligible")
    out += ["", f"Runs left out ({sum(excluded.values())} of {len(m['runs'])}): "
            + "; ".join(f"{reason} ({n})" for reason, n in excluded.most_common()) + "."]
    out += ["", "Requests dropped while building samples: " + "; ".join(f"{k} ({v})" for k, v in
                                                                         sorted(m["drops"].items(), key=lambda kv: -kv[1])) + "."]

    val = [json.loads(line) for line in open(args.data / "val.jsonl")] if (args.data / "val.jsonl").exists() else []
    vlevels = sorted({(t["game"], t["level"]) for t in m["val"]})
    out += ["", "## Validation", "",
            f"{len(m['val'])} trajectories, {len(val)} samples: for each level, the next-best clear by actions that was not "
            "selected: " + ", ".join(f"{g} L{lv}" for g, lv in vlevels) + "."]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
