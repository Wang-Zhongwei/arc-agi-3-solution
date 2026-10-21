# sim-v2-calibrate-validate-explore: calibrate

Workflow run `wf_15abaf47-5b6`, phase Calibrate. Calibrate a continuous-time scheduler simulator from real runs, validate it against held-out real policies (UCB, D-prime), then explore annealed round-robin to D-prime/UCB schedulers

---

PHASE 1 DATA CARD — calibration dataset for sim v2

Phase 1 is done and committed in two commits on branch `arm/sim-v2` (worktree `.../scratchpad/wt/sim-v2`). Nothing was pushed, and the `daniel-franzen-sol` working tree was not touched.
- `70920e7` adds `analysis/calibrate.py`.
- `a34d5c3` adds `analysis/calibration/`: 2.6 MB, with `README.md` documenting every column.

Running the script again gives the same files. It uses only the standard library and takes about 6 s.

Checks on the extracted data:
- **Scores:** recomputed with the harness formula (`GameRun._compute_final_score`, the one `inference.tools.eval` reports), they match `final_score` in all 18 runs.
- **Levels:** all 869 rows of the old `levels.csv` are reproduced exactly.
- **Tokens:** per-level tokens add up to each game's total.
- **Slot use:** the reconstructed slot-holding intervals fill the slots, at 0.98–1.01 of slots × 7260 s in every full run.

Timeline sources:
- **Exact (per request):** each request in `requests.jsonl` is matched to the server's timing line on (input, cached) tokens and the nearest output count; 0 were left unmatched. This covers r2, the three v4 panels and fz-experts8.
- **Transcript:** turn start times from the transcript headers, with trims found by a drop in `history_messages`.
- **History only:** action timestamps, used for fz-experts16-full.

How the three methods agree where all exist:

| Run | Switch-outs (exact / transcript / history) | Total wait s (exact / transcript / history) |
|---|---|---|
| r2 | 45 / 44 / 43 | 66.8k / 66.2k / 66.3k |
| v4 panels | 4/4/4, 3/3/3, 3/3/3 | within 6% |

In r2, the gap before the request after a trim is bimodal: 23 s or less means the game won its slot straight back, 114 s or more means it lost the slot.

**(a) Levels** (8 Franzen full runs: 995 attempts, 849 cleared, 146 censored)

| | p10 | p50 | p90 |
|---|---|---|---|
| Tokens, cleared | 5.6k | 21.0k | 62.4k |
| Actions, cleared | 13 | 30 | 75 |
| Wall s, cleared | 90 | 368 | 2852 |
| Active s, cleared | 89 | 350 | 1076 |
| Tokens per active s | 51 | 61 | 68 |
| Tokens, censored | 16k | 90k | 131k |

- Median cleared level is 19.9k tokens / 29 actions with the ledger, and 22.1k / 31 without.
- `levels_pooled.csv` groups the levels three ways: all 8 Franzen full runs, ledger only, and no ledger.

**(b) Timelines**
- **Franzen starts:** games start in three waves at 10 slots: 10 at t≈0, 10 at 13–21 min, and 5 at 30–34 min. The last start is at 31–34 min.
- **D′ starts:** v4-stack-v1 starts 10 games after 30 min, the last at 72 min.
- **Switch-outs:** 35–63 per full run. 48% of trims end in one; the rest win the slot straight back. The wait has median 1156 s (p25 614, p75 1948).
- **Never re-admitted:** 5–13 games per run are switched out and not let back in before the deadline.
- **Tokens per second while holding a slot:** 62–64 at 10 slots, 55.8 at 12, 51.7 at 14, 53.8 with 16 experts.

**(c) Requests and serving**
- **Per-stream decode by #running-req** (median tok/s): 4→111, 8→72, 9→71, 10→72, 11–12→64, 13–14→58–61.
- **Aggregate decode at 10 slots:** 705 tok/s with 10 running requests, accept length about 2.65.
- **Mean #running-req:** about 8.8 at 10 slots, because games spend time between requests and in prefill.
- **Prefill:** about 11.6k tok/s for large prompts. The re-prefill after a trim is about 53.6k tokens and takes 4.7 s.
- **Server queue:** about 1.5 ms. The real queue is the harness gate.
- **Request shape:**
  - Completion tokens per request: median 1196.
  - Requests per turn: median 1 (p90 3).
- **Client gap** (end of a game's request to its next request, outside trims): median 1.6 s, p90 6.1, p99 21.5. It grows with history:

  | Actions so far | <50 | 300+ |
  |---|---|---|
  | Median gap s | 0.95 | 6.5 |

  By prompt size it is 0.4 s below 30k and 2.7 s above 90k. Per-game medians run 1.1–4.2 s, with wa30 and ka59 slowest.

**(d) Context**
- **First segment:** 66k generated tokens and 37 requests, from a 5.7k prompt to about 116k.
- **Steady segment:** prompt 56k → 116k; 40.4k generated tokens (p10 31.8k, p90 49.8k), 18 requests, 9 turns, 40 actions and about 646 s of holding time.
- **Prompt growth:** 1.47 prompt tokens per generated token.
- **Smaller windows:** 34k per segment at 12 slots and 29k at 14.
- **Ledger:** no change in segment size.

**(e) Startup and deadline**
- **Startup:** the benchmark starts a median 598 s after notebook start (p10 548, p90 682): server launch at 50–70 s, then 472–637 s to load. The first request comes about 1 s after the benchmark starts.
- **Exception:** in fz-spec-adaptive the harness started before the server was ready, losing 128 s of game clock.
- **Deadline:** every game's clock starts at the benchmark start and it stops at that + 7260 s (532 min in the competition). Time spent waiting at the gate counts against the game's own clock. 8–12 in-flight requests are cut at the deadline and their tokens are lost.
- **Tail fade:** announced at 96.8–98.7 min.

**What could not be extracted**
- **Exact per-request timing for other full runs:** only r2 has it. The rest have turn times at 1 s resolution, or action times for fz-experts16-full. Server timing lines can't be tied to games there, so `reqstats.csv` isn't attributed.
- **v4-stack-v1 logs:** no `serve.log` and no notebook log. Only ar25 has requests: bp35 and cd82 are empty and the other 22 files are missing. Its timeline comes from the transcripts inside `artifacts/*_events.jsonl`.
- **Gate decisions:** the queue contents, priority values and who won a freed slot aren't logged. Waits are inferred.
- **Prompt sizes and growth:** these come only from r2 and the panels. Transcript runs give message counts, not prompt tokens.
- **Per-turn tokens in transcript runs:** estimated by splitting each action's exact count by output characters, about ±20% per turn.
- **Client gap breakdown:** python tool, env step and prompt build can't be separated. `uncached_input_tokens` in `benchmark.json` is always 0.
- **Unreached levels:** no cost data. Censored levels give a lower bound only.
- **Competition scale:** there are no 110-game / 532-min runs.
- **Excluded runs:** b1-thinking-real and final-c1-ledgerfix (older harness), fz-tput-all v1–v6 (no scores), fz-ledger-fitted (only a partial 40-min log) and fz-arms-v3-smoke.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/sim-v2/analysis/`:
- `calibrate.py`
- `calibration/README.md`
- `calibration/summary.json` — the data card, start lists and validation
- `calibration/runs.csv`
- `calibration/levels.csv`
- `calibration/levels_pooled.csv`
- `calibration/games.csv`
- `calibration/intervals.csv`
- `calibration/requests.csv`
- `calibration/reqstats.csv`
- `calibration/serve_decode.csv`
- `calibration/serve_minutes.csv`
- `calibration/segments.csv`
