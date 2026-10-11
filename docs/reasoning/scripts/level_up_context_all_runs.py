"""Scatter every saved level-up, for every game, around level lines.

Defaults to both adjacent run collections; override with ARC3_RUNS_DIRS
(colon-separated roots). Runs streamed from Kaggle by recover_level_up_metrics.py
are read from compact digests. Group duplicate snapshots by run/trial timestamp.
Benchmarks provide level completions and scores, events provide clearing
analysis steps, request logs provide prompt-token counts and retained prior
turn starts. Logged calibration notes are a fallback context sample within the
clear turn. Where neither exists, the clearing call's history is rebuilt from
the analysis transcripts and the per-turn `history_messages` count; its prompt
tokens are estimated from the rebuilt text, thinking and image counts with a
linear model fitted on the request-logged runs (leave-one-run-out error saved).
Estimates are drawn hollow; missing metrics sit in an unavailable column, never
at configured limits.
"""

import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parents[1]
DATA = REPORT / "data" / "level_up_context"
FIGURES = REPORT / "figures" / "level_up_context"
ROOTS = [Path(p).resolve() for p in os.environ.get(
    "ARC3_RUNS_DIRS", f"{REPO / 'ARC3-Inference/runs'}:{REPO.parent / 'duck-harness/ARC3-Inference/runs'}"
).split(":")]
FILE = re.compile(r"([^_]+)_p(\d+)_(events|requests)\.jsonl$")
PRUNE = {"src", "kernel", ".git", ".venv", "node_modules", "thumbnails", "__pycache__", "source-dataset", "_source_bundles"}
MEASURED = {"context_tokens": {"last_response_prompt_tokens", "clearing_turn_calibration_sample"},
            "history_steps": {"retained_original_turn_starts"}}
ESTIMATED = {"context_tokens": "transcript_estimate", "history_steps": "transcript_rebuild"}
SECTION = re.compile(r"^\[([A-Z][A-Z _\-]*(?::[^\]\n]*)?)\]$", re.M)
CALIBRATION = re.compile(r"text_token_calibration:[^\n]*\[prompt_tokens (\d+)")
FEATURES = ("text_chars", "thinking_chars", "images")
# Prompt tokens per text char, thinking char and image, plus fixed overhead, as
# fitted on the request-logged runs (reconstruction_validation.json); used only
# to decide whether a rebuilt call would have exceeded the run's budget.
TOKEN_COSTS = (0.333, 0.309, 402.0, 520.0)
BUDGET_SLACK = 1.05
# A run's rebuilt history is trusted only if at most this share of its last
# calls would exceed the run's logged budget (summary, image-window and other
# harness variants the replay does not model fail this).
MAX_OVER_BUDGET = .02


def signature(content):
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def request_metrics(path):
    with path.open() as stream:
        return request_metrics_lines(stream, str(path))


def request_metrics_lines(stream, source):
    starts, signature_steps, measured = {}, defaultdict(set), {}
    for line in stream:
        if line:
            r = json.loads(line)
            step = r.get("analysis_step")
            if step is None:
                continue
            messages = r.get("messages") or []
            if step not in starts:
                users = [m for m in messages if m.get("role") == "user"]
                if users:
                    sig = signature(users[-1].get("content"))
                    starts[step] = sig
                    signature_steps[sig].add(step)
            if r.get("event") != "response":
                continue
            prompt = (r.get("usage") or {}).get("prompt_tokens")
            history, ambiguous = set(), False
            for m in messages:
                if m.get("role") == "user":
                    known = {s for s in signature_steps.get(signature(m.get("content")), ()) if s < step}
                    ambiguous |= len(known) > 1
                    history.update(known)
            if prompt is not None:
                measured[step] = dict(context_tokens=int(prompt), context_source="last_response_prompt_tokens",
                                      history_steps=None if ambiguous else len(history),
                                      history_source="retained_original_turn_starts" if not ambiguous else "ambiguous_turn_starts",
                                      telemetry_file=source)
    return measured


def transcript_messages(transcript):
    """System prompt size, the messages one analysis invocation added in order,
    and its calibration samples as (messages before the calibrated call, prompt tokens)."""
    parts, messages, calibrations, system = SECTION.split(transcript), [], [], 0
    message = lambda role, text=0, images=0: dict(role=role, text=text, thinking=0, images=images)
    for tag, body in zip(parts[1::2], parts[2::2]):
        head, images = tag.split(":")[0], body.count("[image attached]")
        if head == "SYSTEM PROMPT":
            system = len(body)
        elif head == "USER PROMPT":
            messages.append(message("start", len(body), images))
        elif head == "ANALYZER STATUS":
            calibrations.extend((len(messages), int(t)) for t in CALIBRATION.findall(body))
        elif head == "MODEL RESPONSE META":
            messages.append(message("assistant"))
        elif head == "THINKING" and messages:
            messages[-1]["thinking"] += len(body)
        elif head in ("TOOL CALL", "ASSISTANT") and messages:
            messages[-1]["text"] += len(body)
        elif head in ("TOOL RESULT", "USER") and messages:
            messages.append(message("tool" if head == "TOOL RESULT" else "user", len(body), images))
    return system, messages, calibrations


def call_features(system, step, prior, current):
    seen = [m for _, m in prior] + current
    return dict(history_steps=len({s for s, m in prior if m["role"] == "start" and s != step}),
                text_chars=system + sum(m["text"] for m in seen),
                thinking_chars=sum(m["thinking"] for m in seen),
                images=sum(m["images"] for m in seen))


def token_estimate(features, coefficients=TOKEN_COSTS):
    return sum(c * v for c, v in zip(coefficients, [features[f] for f in FEATURES] + [1.0]))


def digest_events(lines):
    """One pass over an events file: level wins, calibration notes, rebuilt history.

    The rebuild replays each invocation's messages into a queue and trims it to
    the logged post-turn `history_messages` count, keeping a slot for the trim
    notice. `rebuilt` describes the input of each step's last call: history
    entering that invocation plus its own messages before the reply. A request
    cannot exceed `context_budget_tokens`, so when that input would and the turn
    trimmed, the call is rebuilt on the trimmed history instead. `checks` pair
    every logged calibration (true prompt tokens of one call) with the rebuilt
    input of that call; `budget` counts last calls still over budget.
    """
    wins, notes, rebuilt, checks, previous_level = {}, {}, {}, [], 1
    queue, rebuild_ok, budget = deque(), True, dict(calls=0, over=0, limit=None)
    for line in lines:
        if not line:
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "analysis":
            transcript, step = event.get("transcript", ""), event.get("analysis_step")
            system, messages, calibrations = transcript_messages(transcript)
            if calibrations:
                notes[step] = calibrations[-1][1]
            if not rebuild_ok:
                continue
            counts = re.findall(r"^history_messages: (\d+)", transcript, re.M)
            limits = re.findall(r"^context_budget_tokens: (\d+)", transcript, re.M)
            if not counts or not messages or messages[0]["role"] != "start":
                rebuild_ok = False
                continue
            prior, limit, kept = list(queue), int(limits[-1]) if limits else None, int(counts[-1])
            queue.extend((step, m) for m in messages)
            trimmed = kept < len(queue)
            if trimmed:
                while len(queue) > max(kept - 1, 0):
                    queue.popleft()
                queue.appendleft(("notice", dict(role="user", text=0, thinking=0, images=0)))
            trimmed_prior = [(s, m) for s, m in queue if s != step]

            def features(current):
                value = call_features(system, step, prior, current)
                if trimmed and limit and token_estimate(value) > limit:
                    value = call_features(system, step, trimmed_prior, current)
                return value
            for position, tokens in calibrations:
                checks.append(dict(step=step, prompt_tokens=tokens, **features(messages[:position])))
            rebuilt[step] = features(messages[:-1] if messages[-1]["role"] == "assistant" else messages)
            if limit:
                budget["limit"] = limit
                budget["calls"] += 1
                budget["over"] += token_estimate(rebuilt[step]) > limit * BUDGET_SLACK
        elif kind == "initial":
            previous_level = int(event.get("level") or 1)
        elif kind == "action":
            if event.get("level_completed") or event.get("run_complete"):
                wins.setdefault(previous_level, dict(step=event.get("analysis_step"), action_num=event.get("action_num")))
            previous_level = int(event.get("level") or previous_level)
    return dict(wins=wins, notes=notes, rebuilt=rebuilt, checks=checks, budget=budget)


def compact_digest(digest, metrics=None):
    """Keep only clearing steps (plus calibration checks), for persisted remote digests."""
    steps = {w["step"] for w in digest["wins"].values()}
    keep = lambda values: {str(s): v for s, v in values.items() if s in steps}
    return dict(wins={str(k): v for k, v in digest["wins"].items()}, notes=keep(digest["notes"]),
                rebuilt=keep(digest["rebuilt"]), metrics=keep(metrics or {}), checks=digest["checks"], budget=digest["budget"])


def expand_digest(digest):
    return dict(wins={int(k): v for k, v in digest["wins"].items()},
                notes={int(k): v for k, v in digest["notes"].items()},
                rebuilt={int(k): v for k, v in digest["rebuilt"].items()},
                metrics={int(k): v for k, v in digest.get("metrics", {}).items()},
                checks=digest.get("checks"), budget=digest.get("budget"))


def benchmark_summary(benchmark):
    return dict(start_time=benchmark.get("start_time"), game_runs=[dict(
        game_id=g.get("game_id", ""), levels_completed=g.get("levels_completed"), actions_per_level=g.get("actions_per_level", []),
        final_score=g.get("final_score"), started_at=g.get("started_at"), history_length=len(g.get("history", [])))
        for g in benchmark.get("game_runs", [])])


def inventory():
    directories = defaultdict(lambda: {"benchmarks": [], "events": defaultdict(list), "requests": defaultdict(list)})
    for root in ROOTS:
        if not root.exists():
            continue
        for base, children, names in os.walk(root):
            children[:] = [d for d in children if d not in PRUNE]
            base = Path(base)
            for name in names:
                match = FILE.fullmatch(name)
                if name == "benchmark.json":
                    directories[str(base.relative_to(root))]["benchmarks"].append(base / name)
                elif match:
                    game, pass_index, kind = match.groups()
                    directory = base.parent if base.name == "artifacts" else base
                    directories[str(directory.relative_to(root))][kind][(game, int(pass_index))].append(base / name)
    return directories


def local_digest(files, key):
    event_paths = files["events"].get(key, [])
    if not event_paths:
        return None
    event_path = max(event_paths, key=lambda p: p.stat().st_size)
    with event_path.open() as stream:
        digest = digest_events(stream)
    digest["metrics"] = {}
    for request in files["requests"].get(key, []):
        for step, value in request_metrics(request).items():
            digest["metrics"].setdefault(step, value)
    digest["event_file"] = str(event_path)
    return digest


def run_sources():
    """Yield (run, benchmark summary, digest loader, files) for local runs, then Kaggle-streamed ones."""
    path = DATA / "remote_runs.json"
    remote = {r["run"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    for relative, files in sorted(inventory().items()):
        if relative in remote and not files["events"]:
            continue
        benchmark = None
        for path in files["benchmarks"]:
            value = benchmark_summary(json.loads(path.read_text()))
            if benchmark is None or sum(g["history_length"] for g in value["game_runs"]) > sum(g["history_length"] for g in benchmark["game_runs"]):
                benchmark = value
        yield relative, benchmark, (lambda key, files=files: local_digest(files, key)), files
    for run, item in sorted(remote.items()):
        trials = item["trials"]
        load = lambda key, trials=trials: expand_digest(trials[f"{key[0]}|{key[1]}"]) if f"{key[0]}|{key[1]}" in trials else None
        yield run, item["benchmark"], load, None


def fit_context_model(pairs):
    """Least squares prompt_tokens ~ text + thinking + images + 1, with leave-one-run-out error."""
    if len({p[0] for p in pairs}) < 2:
        return None
    runs = np.array([p[0] for p in pairs])
    x = np.array([[p[1][f] for f in FEATURES] + [1.0] for p in pairs], float)
    y = np.array([p[2] for p in pairs], float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    held_out = {}
    for run in sorted(set(runs)):
        train = runs != run
        c = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
        error = np.abs(x[~train] @ c - y[~train]) / y[~train]
        held_out[run] = dict(calls=int((~train).sum()), median_abs_pct=round(float(np.median(error)) * 100, 1),
                             p90_abs_pct=round(float(np.percentile(error, 90)) * 100, 1))
    errors = np.abs(x @ coefficients - y) / y
    pooled = [v for run in held_out for v in [held_out[run]["median_abs_pct"]]]
    return dict(features=list(FEATURES) + ["intercept"], coefficients=[float(c) for c in coefficients], calls=len(pairs),
                in_sample_median_abs_pct=round(float(np.median(errors)) * 100, 1),
                leave_one_run_out_median_of_run_medians_pct=round(statistics.median(pooled), 1), leave_one_run_out=held_out)


def rebuild_gate(run_budget, run_checks, model):
    """Accept a run's rebuild when its last calls stay within the logged budget.

    Calibration errors are reported, not gated on: calibrations are logged right
    after trims, the calls the replay handles worst, so they overstate error.
    """
    gate = {}
    for run, budget in run_budget.items():
        share = budget["over"] / budget["calls"] if budget["calls"] else None
        errors = sorted(abs(token_estimate(c, model["coefficients"] if model else TOKEN_COSTS) - c["prompt_tokens"]) / c["prompt_tokens"] * 100
                        for c in run_checks[run] if c["history_steps"] > 0)
        gate[run] = dict(accepted=share is not None and share <= MAX_OVER_BUDGET, last_calls=budget["calls"],
                         over_budget_share=None if share is None else round(share, 4), calibration_checks=len(errors),
                         calibration_median_abs_pct=round(errors[len(errors) // 2], 1) if errors else None)
    return gate


def history_validation(pairs):
    by_run = defaultdict(list)
    for run, rebuilt, measured in pairs:
        by_run[run].append(rebuilt - measured)
    summary = lambda d: dict(calls=len(d), exact=round(sum(v == 0 for v in d) / len(d), 3),
                             within_one=round(sum(abs(v) <= 1 for v in d) / len(d), 3),
                             mean_abs_error=round(statistics.mean(abs(v) for v in d), 2))
    if not by_run:
        return None
    return dict(overall=summary([v for d in by_run.values() for v in d]), by_run={r: summary(d) for r, d in by_run.items()})


def extract():
    candidates, coverage, trials = defaultdict(list), [], defaultdict(list)
    recovered = {}
    path = DATA / "recovered_request_metrics.json"
    if path.exists():
        for metric in json.loads(path.read_text()):
            recovered[(metric["source_run"], metric["game"], metric["pass_index"], metric["step"])] = metric
    context_pairs, history_pairs = [], []
    run_budget, run_checks = defaultdict(lambda: dict(calls=0, over=0)), defaultdict(list)
    for relative, benchmark, load, files in run_sources():
        print(f"Scanning {relative}", flush=True)
        if benchmark is None:
            coverage.append(dict(run=relative, status="no benchmark", event_trials=len(files["events"])))
            continue
        passes = Counter()
        for game in benchmark["game_runs"]:
            game_id = game["game_id"]
            key = (game_id, passes[game_id])
            passes[game_id] += 1
            identity = (benchmark.get("start_time") or relative, key[0], key[1], game.get("started_at") or "")
            trials[identity].append(dict(source_run=relative, game=game_id.split("-")[0], pass_index=key[1],
                                         final_score=game.get("final_score"), levels_completed=game.get("levels_completed"),
                                         history_length=game["history_length"], context_budget=None))
            completed = int(game.get("levels_completed") or 0)
            digest = load(key) or dict(wins={}, notes={}, rebuilt={}, metrics={})
            metrics, rebuilt = digest["metrics"], digest["rebuilt"]
            limit = (digest.get("budget") or {}).get("limit")
            for field in ("calls", "over"):
                run_budget[relative][field] += (digest.get("budget") or {}).get(field, 0)
            run_checks[relative].extend(digest.get("checks") or [])
            for step, measured in metrics.items():
                if step in rebuilt:
                    context_pairs.append((relative, rebuilt[step], measured["context_tokens"]))
                    if measured["history_steps"] is not None:
                        history_pairs.append((relative, rebuilt[step]["history_steps"], measured["history_steps"]))
            boundaries, n = {}, 0
            for level, count in enumerate(game.get("actions_per_level", []), 1):
                n += int(count)
                if level <= completed:
                    boundaries[level] = n
            rows = []
            for level in range(1, completed + 1):
                win = digest["wins"].get(level)
                step = win.get("step") if win else None
                metric = {**recovered.get((relative, key[0].split("-")[0], key[1], step), {}), **metrics.get(step, {})}
                context = metric.get("context_tokens", digest["notes"].get(step))
                rows.append(dict(
                    source_run=relative, game=key[0].split("-")[0], pass_index=key[1], level=level, step=step,
                    action_num=win.get("action_num") if win else boundaries.get(level),
                    context_tokens=context, history_steps=metric.get("history_steps"),
                    context_source=metric.get("context_source", "clearing_turn_calibration_sample" if context is not None else "unavailable"),
                    history_source=metric.get("history_source", "unavailable"),
                    telemetry_file=metric.get("telemetry_file", digest.get("event_file", "") if context is not None else ""),
                    benchmark_start=benchmark.get("start_time"), trial_start=game.get("started_at"),
                    rebuilt=rebuilt.get(step),
                ))
            trials[identity][-1]["context_budget"] = limit
            for row in rows:
                row["rebuild_run"], row["context_budget"] = relative, limit
            candidates[identity].append((game["history_length"], rows, relative))
            coverage.append(dict(run=relative, game=key[0], pass_index=key[1], completed_levels=completed,
                                 event_level_ups=len(digest["wins"]), request_log=bool(metrics),
                                 context_measured=sum(r["context_tokens"] is not None for r in rows),
                                 history_measured=sum(r["history_steps"] is not None for r in rows),
                                 rebuilt=sum(r["rebuilt"] is not None for r in rows)))
    rows, duplicates = [], []
    for variants in candidates.values():
        variants.sort(key=lambda x: (x[0], len(x[1]), sum(r["context_tokens"] is not None for r in x[1]),
                                     sum(r["rebuilt"] is not None for r in x[1])), reverse=True)
        _, chosen, source = variants[0]
        for _, other, other_source in variants[1:]:
            duplicates.append(dict(kept=source, duplicate=other_source))
            by_level = {r["level"]: r for r in other}
            for row in chosen:
                alternative = by_level.get(row["level"], {})
                for metric in ("context", "history"):
                    field = metric + ("_tokens" if metric == "context" else "_steps")
                    if row[field] is None and alternative.get(field) is not None:
                        row[field] = alternative[field]
                        row[metric + "_source"] = alternative[metric + "_source"]
                if row["rebuilt"] is None and alternative.get("rebuilt") is not None:
                    row["rebuilt"], row["rebuild_run"] = alternative["rebuilt"], alternative["rebuild_run"]
        rows.extend(chosen)
    model = fit_context_model(context_pairs)
    gate = rebuild_gate(run_budget, run_checks, model)
    for row in rows:
        rebuilt, run = row.pop("rebuilt"), row.pop("rebuild_run")
        if rebuilt is None:
            continue
        if not gate[run]["accepted"]:
            for metric in ("context", "history"):
                if row[metric + ("_tokens" if metric == "context" else "_steps")] is None:
                    row[metric + "_source"] = "rebuild_rejected"
            continue
        if row["history_steps"] is None:
            row["history_steps"], row["history_source"] = rebuilt["history_steps"], ESTIMATED["history_steps"]
        if row["context_tokens"] is None and model:
            row["context_tokens"] = round(sum(c * v for c, v in zip(model["coefficients"], [rebuilt[f] for f in FEATURES] + [1.0])))
            row["context_source"] = ESTIMATED["context_tokens"]
    scores = []
    for identity, variants in trials.items():
        best = max(variants, key=lambda v: v["history_length"])
        scores.append(dict(best, benchmark_start=identity[0], trial_start=identity[3]))
    validation = dict(context_model=model, history_rebuild=history_validation(history_pairs),
                      run_gate=dict(max_over_budget=MAX_OVER_BUDGET, budget_slack=BUDGET_SLACK, runs=gate))
    return rows, coverage, duplicates, len(candidates), scores, validation


def game_scores(scores):
    by_game = defaultdict(list)
    for s in scores:
        if s["final_score"] is not None:
            by_game[s["game"]].append(float(s["final_score"]))
    return {g: (statistics.mean(v), len(v)) for g, v in by_game.items()}


def split(rows, metric):
    measured = [r for r in rows if r[metric] is not None and r[metric.split("_")[0] + "_source"] in MEASURED[metric]]
    estimated = [r for r in rows if r[metric] is not None and r[metric.split("_")[0] + "_source"] == ESTIMATED[metric]]
    missing = [r for r in rows if r[metric] is None]
    return measured, estimated, missing


def counts(rows, metric):
    return dict(zip(("measured", "estimated", "unavailable"), map(len, split(rows, metric))))


def x_formatter(metric, missing_x=None):
    def label(value, _):
        if missing_x is not None and abs(value - missing_x) < .001:
            return "unavailable"
        return f"{value / 1024:.0f}k" if metric == "context_tokens" else f"{value:.0f}"
    return FuncFormatter(label)


def draw_panel(ax, subset, metric, color, upper, rng, point=14):
    """Level lines with measured (filled), estimated (hollow) and unavailable (grey column) level-ups."""
    measured, estimated, missing = split(subset, metric)
    top_level = max(r["level"] for r in subset)
    missing_x = -upper * .13
    ax.axvspan(missing_x - .055 * upper, missing_x + .045 * upper, color="#f0f0f0", zorder=0)
    for level in range(1, top_level + 1):
        ax.axhline(level, color="#dddddd", lw=.8, zorder=0)
    jitter = lambda rows: [r["level"] + rng.uniform(-.23, .23) for r in rows]
    ax.scatter([r[metric] for r in estimated], jitter(estimated), s=point * .8, facecolors="none",
               edgecolors=color, alpha=.45, linewidths=.7, rasterized=True)
    ax.scatter([r[metric] for r in measured], jitter(measured), s=point, color=color, alpha=.75,
               edgecolors="white", linewidths=.35, rasterized=True, zorder=3)
    ax.scatter([missing_x + rng.uniform(-.026, .026) * upper for _ in missing], jitter(missing),
               s=point * .4, color="#999999", alpha=.14, linewidths=0, rasterized=True)
    ax.set_yticks(range(1, top_level + 1), [f"L{i}" for i in range(1, top_level + 1)])
    ax.set_ylim(.45, top_level + .55)
    ax.set_xlim(missing_x - upper * .06, upper * 1.02)
    ticks = [v for v in MaxNLocator(nbins=5, integer=True).tick_values(0, upper) if 0 <= v <= upper]
    ax.set_xticks([missing_x] + ticks)
    ax.xaxis.set_major_formatter(x_formatter(metric, missing_x))
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


LABELS = {"context_tokens": "Current context size (prompt tokens)", "history_steps": "Completed steps retained in history"}
NOTES = {"context_tokens": "Context = prompt tokens on the clearing call. Filled: measured (request log or logged calibration sample). "
                           "Hollow: estimated from the rebuilt transcript.",
         "history_steps": "History steps = earlier agent turns whose start message remains in the clearing call's input. "
                          "Filled: measured (request log). Hollow: rebuilt from transcripts."}


def legend_handles(color="#555555"):
    return [Line2D([], [], marker="o", ls="", color=color, label="measured"),
            Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=color, label="estimated"),
            Line2D([], [], marker="o", ls="", color="#bbbbbb", label="unavailable")]


def upper_bound(rows, metric):
    return max((r[metric] for r in rows if r[metric] is not None), default=1) * 1.07


def score_label(game, scores):
    mean, trials = scores.get(game, (float("nan"), 0))
    return f"avg score {mean:.1f} over {trials} trials"


def panel_title(game, subset, metric, scores, with_game=True):
    c = counts(subset, metric)
    clearing = len({(r["source_run"], r["pass_index"], r["trial_start"]) for r in subset})
    head = f"{game} · {score_label(game, scores)}\n" if with_game else ""
    return head + f"{len(subset):,} level-ups, {clearing} clearing trials · {c['measured']:,} measured + {c['estimated']:,} est."


def plot_overview(rows, metric, filename, scores, folder, scope):
    """One panel per game, highest average score first, shared x scale."""
    games = sorted({r["game"] for r in rows}, key=lambda g: -scores.get(g, (0, 0))[0])
    columns = 5
    lines = math.ceil(len(games) / columns)
    fig, axes = plt.subplots(lines, columns, figsize=(25, 4.2 * lines), squeeze=False)
    rng, upper = random.Random(20261011), upper_bound(rows, metric)
    for index, (ax, game) in enumerate(zip(axes.flat, games)):
        subset = [r for r in rows if r["game"] == game]
        draw_panel(ax, subset, metric, plt.get_cmap("tab10")(index % 10), upper, rng)
        ax.set_title(panel_title(game, subset, metric, scores), loc="left", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel(LABELS[metric], fontsize=9)
    for ax in list(axes.flat)[len(games):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="upper right", ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(.99, .995))
    fig.suptitle(f"Saved runs, {scope} · {len(games)} games, highest average score first · "
                 + ("current context size" if metric == "context_tokens" else "steps retained in history"),
                 x=.03, ha="left", fontsize=20, fontweight="bold")
    fig.text(.03, .008, "One dot per cleared level; vertical jitter only. " + NOTES[metric], fontsize=11, color="#555555")
    fig.tight_layout(rect=(0, .018, 1, .975))
    save(fig, folder / filename)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rows, coverage, duplicates, trial_count, scores, validation = extract()
    write_csv(DATA / "level_ups.csv", rows)
    write_csv(DATA / "trial_scores.csv", scores)
    per_game = game_scores(scores)
    summary = dict(run_roots=[str(p) for p in ROOTS], unique_trials=trial_count,
                   clearing_runs=len({r["source_run"] for r in rows}), level_ups=len(rows),
                   context_tokens=counts(rows, "context_tokens"), history_steps=counts(rows, "history_steps"),
                   level_ups_by_context_budget={str(k): v for k, v in Counter(r["context_budget"] for r in rows).most_common()},
                   games={g: dict(avg_score=round(mean, 2), trials=n, level_ups=sum(r["game"] == g for r in rows),
                                  context_tokens=counts([r for r in rows if r["game"] == g], "context_tokens"),
                                  history_steps=counts([r for r in rows if r["game"] == g], "history_steps"))
                          for g, (mean, n) in sorted(per_game.items())},
                   duplicate_trial_snapshots=len(duplicates),
                   runs_without_benchmark=sorted(c["run"] for c in coverage if c.get("status") == "no benchmark"))
    (DATA / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    (DATA / "reconstruction_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    # Figures keep the most common context budget, so one x scale means one trimming regime.
    current = Counter(r["context_budget"] for r in rows if r["context_budget"]).most_common(1)[0][0]
    scope_rows = [r for r in rows if r["context_budget"] == current]
    scope_scores = game_scores([s for s in scores if s["context_budget"] == current])
    for metric, name in (("context_tokens", "context"), ("history_steps", "history_steps")):
        plot_overview(scope_rows, metric, f"all_games_level_up_{name}", scope_scores, FIGURES, f"{current:,}-token context budget")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"games", "runs_without_benchmark"}}, indent=2), flush=True)
    print(json.dumps((validation["history_rebuild"] or {}).get("overall")),
          json.dumps({k: v for k, v in (validation["context_model"] or {}).items() if k != "leave_one_run_out"}), flush=True)


if __name__ == "__main__":
    main()
