# Simulator calibration data

These tables calibrate the v2 scheduling simulator. They are extracted from real Kaggle runs by
`analysis/calibrate.py`, which uses only the standard library and runs in about 6 s:

    nice -n 10 python3 analysis/calibrate.py \
        [--root /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs]

Times are in seconds from the benchmark start (`benchmark.json` `start_time`). Every game session
is constructed at that instant, and all `started_at` values agree within 1 s in every run. Token
counts are generated (completion) tokens unless a column says otherwise.

## Sources

| Run | Kind | Scheduler | Ledger | Slots | Timeline | Notes |
|---|---|---|---|---|---|---|
| fz-arms-v4-control | full | franzen | 0 | 10 | transcript | |
| fz-arms-v5-temp06 | full | franzen | 0 | 10 | transcript | temperature 0.6 |
| fz-arms-v6-audit60k | full | franzen | 0 | 10 | transcript | strategy audit 60k |
| fz-experts16-full | full | franzen | 0 | 10 | history | 16 experts per token, slower serving |
| fz-ledger-on-trim | full | franzen | 1 | 10 | transcript | |
| fz-ledger-on-trim-r2 | full | franzen | 1 | 10 | **requests** | the only full run with requests.jsonl |
| fz-ledger-streams12 | full | franzen | 1 | 12 | transcript | context 112640, drain 50176 |
| fz-streams14 | full | franzen | 0 | 14 | transcript | context 98304, drain 43008 |
| fz-ucb | full | ucb | 0 | 10 | transcript | `ARC3_UCB_C=0.1` |
| v4-stack-v1 | full | dprime | 1 | 10 | transcript | turns from `artifacts/*_events.jsonl`; no serve.log or notebook log |
| v4-panel-control / -easy-first / -dprime | panel | franzen / easy-first / dprime | 0 | 4 | requests | 10 games, 20 min |
| fz-experts8 | panel | franzen | 0 | 10 | requests | 10 games on 10 slots |
| fz-tput-base, fz-experts16, fz-spec-adaptive, fz-tput-all-v7 | panel | franzen | 0 | 10 | history / transcript | 10 games on 10 slots |

- **Full runs:** 25 games for 121 min (`max_runtime_s_per_game` = 7260 s).
- **Panels:** 10 games for 20–25 min.
- **Context:** the default window is 131072 with a 12288-token output reserve (budget 118272).
  Trims drain the prompt to about 58k (`ARC3_CONTEXT_DRAIN_TOKENS` 59392).
- **Pooled "Franzen full" set:** the eight full runs with `scheduler == franzen`.
- **Scores:** recomputed with the harness formula (taaf `GameRun._compute_final_score`, which
  `inference.tools.eval` reports). They match `final_score` in every run (`summary.json`
  `score_check`).

## Timeline methods

A game holds a slot from admission until it hands the slot over at a context trim
(`_maybe_handover`). When it hands over, it re-queues at once and either wins the slot straight
back (a *repick*) or waits (a *switch-out*). Switch-outs are reconstructed in one of three ways:

- **requests (exact).** Each response in `<game>_p0_requests.jsonl` is matched to an SGLang
  `ReqTimeStats` line.
  - The match is on `(input_len, cached_input_len)` and the nearest `output_len`. The server
    counts 0–5 more output tokens than `usage`.
  - Every request matched (0 unmatched).
  - A trim request is one whose prompt is below 0.8 × the previous prompt. It is a switch-out
    when the gap before it is ≥ 60 s.
  - In r2 the trim gaps split cleanly: 40 are ≤ 23 s (repicks) and 44 are ≥ 114 s.
- **transcript.** Turn start times come from the `--- analysis_step=… | HH:MM:SS ---` headers.
  - A trim is a drop in `history_messages`.
  - The handover wait falls in the trim turn or in the next one. The wait is the turn's
    duration minus its estimated work:
    - tokens ÷ the local per-stream rate,
    - plus 2 s per request,
    - plus 5 s to re-prefill.
  - Turn tokens come from the action's exact token count, split across its turns by output
    characters (0.32 tokens per char).
- **history.** Action timestamps only. An action gap that exceeds its token work by ≥ 120 s is a
  wait. There is no trim information.

Two more rules apply to every method:
- **Not re-admitted.** A switch-out that is not re-admitted at least 120 s before the deadline
  counts as never re-admitted (`end_reason = switch_out_not_readmitted`). At the deadline,
  queued games get a slot for a moment, which shows up as a last turn ending in
  `stop_requested`.
- **Ungated runs.** When slots ≥ games the gate never binds, and no waits are recorded.

**Validation on the runs with exact timing** (`summary.json` `validation`):

| Run | Switch-outs (exact / transcript / history) | Total wait s (exact / transcript / history) |
|---|---|---|
| fz-ledger-on-trim-r2 | 45 / 44 / 43 | 66 764 / 66 215 / 66 322 |
| v4-panel-control | 4 / 4 / 4 | 2 215 / 2 212 / 2 107 |
| v4-panel-easy-first | 3 / 3 / 3 | 1 096 / 1 116 / 1 050 |
| v4-panel-dprime | 3 / 3 / 3 | 1 671 / 1 671 / 1 592 |

The reconstructed intervals fill the slots:
- **Occupancy.** Σ holding time ÷ (slots × 7260) is 0.99–1.01 in every full run.
- **Concurrency.** Holders exceed the slot count 0–4% of the time in transcript runs (0% in r2).
- **History-only run.** fz-experts16-full exceeds the slot count 17% of the time. The history
  method cannot tell where the wait falls inside an action gap. Its counts and totals are good
  (see the table), but its interval edges are approximate.

## Files

| File | Rows | Content |
|---|---|---|
| `runs.csv` | 18 | Run settings and recomputed score. Startup milestones (notebook-relative s), first request, tail-fade time, request timeouts at the deadline, median per-stream decode tok/s, timeline methods. |
| `levels.csv` | 1 487 | **(a)** One row per (run, game, attempted level), columns listed below. |
| `levels_pooled.csv` | 443 | **(a)** Per (game, level, group), with groups `franzen_full`, `franzen_ledger` and `franzen_noledger`. Attempts, clears, censored count, median tokens, actions, wall and active s, level score. Semicolon lists hold every cleared and censored token cost. |
| `games.csv` | 330 | **(b)** Per (run, game): start rank, first active s, first and last action s, end s, total and history tokens, actions, turns, requests, intervals, active s, trims, switch-outs, total wait. |
| `intervals.csv` | 724 | **(b)** Slot-holding intervals: start, end, end reason (`switch_out`, `won`, `deadline`, `switch_out_not_readmitted`), the wait that followed, tokens and actions inside. |
| `requests.csv` | 3 518 | **(c)** Per request, for r2, the v4 panels and fz-experts8 (3 428 timed), plus v4-stack-v1 ar25 (untimed). Turn indices; prompt, cached, uncached, completion, reasoning and image tokens; message count. Server queue, prefill and decode s; mean #running-req while decoding; per-stream tok/s; client gap since the game's previous request ended; trim and switch-out flags. |
| `reqstats.csv` | 24 870 | **(c)** Every `ReqTimeStats` line from every serve.log (not attributed to games): entry s, input, cached and output tokens, queue ms, prefill s, decode s, mean #running-req. |
| `serve_decode.csv` | 146 | **(c)** Per run × #running-req: count of "Decode batch" lines, aggregate gen tok/s (p10, median, p90, mean), per-stream tok/s, median accept length. |
| `serve_minutes.csv` | 1 277 | **(c)** Per run × minute: mean #running-req, aggregate and per-stream tok/s, #queue-req, prefill new and cached tokens, requests entered, generated tokens finished. |
| `segments.csv` | 1 209 | **(d)** Context segments between trims. Holds turns, requests, actions, generated tokens, busy s and, for requests runs, prompt at start, peak and growth. Also the re-prefill at the next trim (uncached tokens and s), the wait after, and whether the game was switched out. |
| `summary.json` | | The data card below, the per-run start lists, validation, and the score check. |
| `games_meta.json` | 25 games, 18 runs | Per game: level count and human baselines for every level (`base_actions_per_level`, including levels no run reached). Per run: dispatch order (`benchmark.json` `game_runs` order, the solver's `game_index`), final scores and levels completed. |

**`levels.csv` columns:**
- **Run settings:** run, kind, scheduler, ledger, slots.
- **Level:** game, level, levels_total, human_actions.
- **Cost:** tokens, actions, cleared, censored, level_score. Censored means the level was still
  being played when the run ended. Its tokens include reasoning after the last action.
- **Time:** start_s, end_s and wall_s. A level starts at the action that cleared the previous
  one, or at the game's first activity. It ends at its own clearing action, or at the game's
  end.
- **Slot time:** active_s is wall_s minus the time spent waiting at the gate. n_trims and
  n_switch_outs count those events inside the level. timeline names the method.

## Data card

### (a) Levels (995 attempts in the 8 Franzen full runs: 849 cleared, 146 censored)

| | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| Tokens, cleared | 5.6k | 10.5k | 21.0k | 41.7k | 62.4k |
| Actions, cleared | 13 | 18 | 30 | 51 | 75 |
| Wall s, cleared | 90 | 181 | 368 | 837 | 2 852 |
| Active s, cleared | 89 | 176 | 350 | 692 | 1 076 |
| Tokens per active s, cleared | 51 | 56 | 61 | 65 | 68 |
| Tokens, censored | 16k | 46k | 90k | 113k | 131k |

- **Ledger runs:** the median cleared level costs 19.9k tokens and 29 actions (333 clears).
- **No-ledger runs:** 22.1k tokens and 31 actions (516 clears).

### (b) Game timelines

- **Franzen starts.** At 10 slots the 25 games start in three waves:
  - 10 at t ≈ 0;
  - 10 at 13–21 min, as the first wave trims;
  - the last 5 at 30–34 min.
  - The latest start is 1 872–2 030 s. UCB and the 12- and 14-slot runs follow the same
    pattern, with fewer waves.
- **D′ starts.** v4-stack-v1 starts 10 games after 30 min, the last at 4 296 s (72 min).
- **Switch-outs.** 35–63 per full run (406 in the 8 Franzen full runs).
  - The wait after one has median 1 156 s (p25 614, p75 1 948).
  - 48% of trims end in a switch-out; the rest are repicks.
  - 5–13 games per run are switched out and never re-admitted before the deadline.
- **Tokens per second of slot holding** (game-level average):

  | Setup | tok/s |
  |---|---|
  | 10 slots | 62–64 |
  | 12 slots | 55.8 |
  | 14 slots | 51.7 |
  | 16 experts | 53.8 |

### (c) Requests and serving

- **Per-stream decode tok/s** by mean #running-req (`ReqTimeStats`; requests with ≥ 256 output
  tokens; medians):

  | #running-req | 4 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | tok/s | 111 | 76 | 76 | 72 | 71 | 72 | 64 | 64 | 58 | 61 |

- **Aggregate decode** in the standard 10-slot runs (from "Decode batch" lines):

  | #running-req | 4 | 6 | 8 | 9 | 10 |
  |---|---|---|---|---|---|
  | aggregate tok/s | 463 | 562 | 686 | 693 | 705 |

  - Accept length is about 2.65.
  - Mean #running-req over a 10-slot run is 8.7–9.1, because games spend time in client gaps
    and prefill.
  - Finished generated tokens per second over the run: about 615–634 at 10 slots, 662 at 12,
    720 at 14, and 524 with 16 experts.
- **Prefill:**

  | Uncached tokens | < 1k | 1–4k | 4–16k | 16–48k | ≥ 48k |
  |---|---|---|---|---|---|
  | tok/s (median) | 2.8k | 7.4k | 10.3k | 11.8k | 11.6k |
  | s (median) | 0.13 | 0.26 | 0.59 | 3.5 | 4.8 |

  - A re-prefill after a trim is about 53.6k uncached tokens and takes 4.7 s (p10 4.2, p90 5.6).
  - Server queue time is about 1.5 ms (p99 about 5 s), because max-running equals the slot
    count.
  - The real queue is the harness gate, covered in (b).
- **Request shape:**
  - Completion tokens per request: median 1 196 (p10 269, p90 4 511).
  - Requests per turn: median 1 (p90 3). A turn yields after 2 048 generated tokens or ends
    when an action executes.
- **Client gap** (the end of a game's request to its next request; python tool, env step,
  prompt build): median 1.6 s (p90 6.1, p99 21.5) outside trims. It grows with history length:

  | Actions so far | 0–49 | 50–99 | 100–199 | 200–299 | 300+ |
  |---|---|---|---|---|---|
  | Median gap s | 0.95 | 2.1 | 3.1 | 4.5 | 6.5 |

  - It also grows with prompt size: 0.4 s below 30k tokens, 2.7 s above 90k.
  - Per-game medians range 1.1–4.2 s, with wa30 and ka59 slowest.

### (d) Context dynamics

| | Exact (r2 + panels) | Transcript runs at 10 slots |
|---|---|---|
| First segment, generated tokens (5.7k → ~116k prompt) | 66k | ~60k |
| First segment, requests | 37 | 35 |
| Steady segment, generated tokens | 40.4k (p10 31.8k, p90 49.8k) | 38–40k |
| Steady segment, requests | 18 | |
| Steady segment, turns | 9 | |
| Steady segment, actions | 40 | |
| Steady segment, busy s | 646 | 610–655 |

- **Prompt range.** A steady segment starts at a 56.3k prompt and peaks at 115.9k. The prompt
  grows 1.47 tokens per generated token: outputs plus tool results and frames.
- **Smaller windows** give shorter steady segments: 34k generated tokens at 12 slots
  (ctx 112640) and 29k at 14 slots (ctx 98304).
- **Ledger runs** have the same segment sizes.

### (e) Startup and deadline

- **Startup** (notebook-relative s, 17 runs with a notebook log):
  - Server launch is at 50–70 s.
  - SGLang is READY 472–637 s after launch.
  - The benchmark starts about 2 s after READY: median 598 s (p10 548, p90 682).
  - The first LLM request is about 1 s after the benchmark start.
  - In fz-spec-adaptive the harness started at 730 s, before READY at 879 s
    (`SERVER_STARTUP_TIMEOUT`). Its first request came 128 s into the game clock.
- **Deadline semantics** (solver.py, tool_agent.py, the notebook):
  - Each session's clock starts at construction, which is the benchmark start.
  - A game stops at `started_at + max_runtime_s_per_game`: 121 min offline and 532 min in the
    competition rerun.
  - Waiting at the gate counts against the game's own clock.
  - The gate's fade clock uses the earliest session start and deadline.
  - The Franzen tail fade (fraction 0.2) was announced at 96.8–98.7 min.
  - At the deadline, in-flight requests are cut, 8–12 per full run ("analyzer request timed
    out"). Their tokens are lost: the server still finishes them (unmatched `ReqTimeStats`
    lines after 7 176 s), but they never reach the history.
  - Offline runs also have a soft end at notebook start + 32 400 − 600 s, which never binds at
    121 min.

## What could not be extracted

- **Exact per-request timing exists only for fz-ledger-on-trim-r2** among the full runs, plus
  the v4 panels and fz-experts8.
  - The other full runs have turn-level times (1 s resolution) from transcripts.
  - fz-experts16-full has action times only.
  - Serve.log `ReqTimeStats` lines are not attributed to games in those runs. The rid is not
    logged client-side, so `reqstats.csv` is unattributed.
- **v4-stack-v1 is missing most logs.**
  - It has no serve.log and no notebook log, so it has no serving data and no startup times.
  - Two of its request files are empty and 22 are missing; only ar25 has requests.
  - Its timeline comes from the transcripts inside `artifacts/*_events.jsonl`, using the median
    per-stream rate of the other runs.
- **Gate decisions are not logged.** There is no record of who was queued, priority values,
  or who won a freed slot. Waits are inferred as the gap between losing the slot at a trim and
  the next request.
- **Prompt sizes come only from requests runs.** Transcript-only runs give `history_messages`
  counts and generated tokens per segment (estimated per turn), not prompt token counts.
  Prompt growth comes from r2 and the panels only.
- **Turn tokens are estimated in transcript runs.** The action's token count is exact; its
  split across turns uses output characters (±20% per turn).
- **The client gap cannot be decomposed.** Python tool time, env step, animation and prompt
  build are one number. `uncached_input_tokens` in `benchmark.json` is always 0.
- **Tokens lost at the deadline are not attributable to games.**
- **Unattempted levels have no cost.** A level the run never reached has no cost observation.
  Censored attempts give only a lower bound.
- **There is no competition-scale run.** No hidden-set (110 games, 532 min) data exists, so
  ratios beyond 25 games on 10 slots are extrapolation.
- **Some runs are excluded.** b1-thinking-real and final-c1-ledgerfix use an older harness;
  fz-tput-all v1–v6 have no final scores; fz-ledger-fitted has only a partial 40-min log; and
  fz-arms-v3-smoke is a 5-min smoke test.
