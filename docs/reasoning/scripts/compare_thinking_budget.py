"""Paired across-game score tests at a token cap or using final scores."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import ttest_rel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=150_000)
    parser.add_argument("--final", action="store_true", help="Compare final scores without a token cap")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    root = Path(__file__).resolve().parents[3]
    runs = root / "ARC3-Inference/runs/kaggle"
    paths = {
        "v5": runs / "dangthuygiang/arc3-v207-danibase-base-dfranzen/v05/evaluation.json",
        "v2": runs / "ticbudja/thinking-budget-and-user-prompt-trim/v02/evaluation.json",
        "v3": runs / "ticbudja/thinking-budget-and-user-prompt-trim/v03/evaluation.json",
    }
    evaluations = {key: json.loads(path.read_text()) for key, path in paths.items()}
    games = sorted(evaluations["v5"]["games"])
    assert all(sorted(d["games"]) == games for d in evaluations.values())
    scores, rows, summaries = {}, [], {}
    for name, evaluation in evaluations.items():
        values, ended = [], []
        for game in games:
            lanes = evaluation["progression_lanes"]["tokens"][game]
            trial_scores = evaluation["games"][game]["trial_scores"]
            assert lanes.keys() == trial_scores.keys(), (name, game)
            assert len(lanes) == 1, "These comparisons expect one pass per game"
            trial, points = next(iter(lanes.items()))
            assert points and points[0][0] == 0, (name, game)
            assert all(np.isfinite(point).all() for point in points)
            assert all(a[0] <= b[0] for a, b in zip(points, points[1:]))
            assert np.isclose(points[-1][1], trial_scores[trial]), (name, game)
            eligible = points if args.final else [point for point in points if point[0] <= args.tokens]
            score = trial_scores[trial] if args.final else eligible[-1][1]
            state = evaluation["games"][game]["states"][trial]
            early = not args.final and points[-1][0] < args.tokens
            if early:
                ended.append({"game": game, "state": state, "end_tokens": points[-1][0]})
            values.append(score)
            rows.append({"run": name, "game": game, "trial": trial,
                         "score": score, "last_score_knot_tokens": eligible[-1][0],
                         "end_tokens": points[-1][0], "state": state, "ended_before_cap": early})
        scores[name] = np.asarray(values)
        summaries[name] = {
            "evaluation": str(paths[name].relative_to(root)),
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "ended_before_cap": ended,
        }
        summary_path = paths[name].with_name("summary.txt")
        summary_fields = dict(line.split(":", 1) for line in summary_path.read_text().splitlines()
                              if ":" in line and not line.startswith(" "))
        summaries[name]["duration_from_summary"] = summary_fields["duration"].strip()
        summaries[name]["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if args.final:
            assert abs(float(summary_fields["mean score"]) - np.mean(values)) <= 0.0051
            assert abs(float(summary_fields["median score"]) - np.median(values)) <= 0.0051
    comparisons = []
    for candidate, baseline in [("v2", "v5"), ("v3", "v5"), ("v3", "v2")]:
        x, y = scores[candidate], scores[baseline]
        result = ttest_rel(x, y, alternative="two-sided")
        interval = result.confidence_interval()
        # Independently verify the statistic from the paired differences.
        delta = x - y
        assert np.isclose(result.statistic, delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta))))
        comparisons.append({
            "candidate": candidate, "baseline": baseline, "n": len(games),
            "mean_difference": float(delta.mean()), "t": float(result.statistic),
            "df": int(result.df), "p_two_sided": float(result.pvalue),
            "p_one_sided_improvement": float(ttest_rel(x, y, alternative="greater").pvalue),
            "ci95": [float(interval.low), float(interval.high)],
            "wins": int(sum(x > y)), "ties": int(sum(x == y)), "losses": int(sum(x < y)),
        })
    output = root / "docs/reasoning/data"
    output.mkdir(exist_ok=True)
    prefix = output / ("thinking_budget_ttest_final" if args.final else f"thinking_budget_ttest_{args.tokens}")
    method = ("Final per-game scores, checked against summary.txt means and medians; pair exact game IDs"
              if args.final else "Last score knot at or below cap; carry terminal scores forward; pair exact game IDs")
    payload = {"token_cap_per_game": None if args.final else args.tokens, "scipy_version": scipy.__version__,
               "method": method,
               "runs": summaries, "comparisons": comparisons}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summaries": {name: {k: d[k] for k in ("mean", "median")} for name, d in summaries.items()},
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
