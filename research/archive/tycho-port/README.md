# Tycho port (experiment, 2026-10-08)

[Tycho](https://github.com/NIMI-research/Tycho) (NIMI research, Apache-2.0) is a self-directed
ARC-AGI-3 harness built around executable world models. Its paper reports RHAE 79–100 on the 25
public games with Claude Opus and GPT. We ported it to Kaggle to see how it does on our model and
compute: the W4A16 Qwen3.8-Flash-Next served by Franzen's Pennyroyal SGLang build (MTP drafter +
FR-Spec, the fastest engine we have).

`patches/` is the port as a patch series on Tycho's `Initial release` commit (`f68912a`). Apply
with `git am` on a clone of NIMI-research/Tycho. The working copy lives outside this repo in
`arc-agi-v4/tycho-port/`.

| Patch | Change |
|---|---|
| 0001 | The Chat Completions transport reads SGLang's `reasoning_content` and replays it in history (`LLM_REPLAY_REASONING`), and sends extra sampling fields (`LLM_SAMPLING_JSON`). |
| 0002 | `TYCHO_ENV_DIR`: the environment files directory (Kaggle mounts the games read-only). |
| 0003 | Enforces `TYCHO_PER_GAME_WALLCLOCK_BUDGET_SECONDS`, which Tycho registered but never read. The game stops cleanly at the deadline and keeps its score. |
| 0004–0005 | `kaggle/build_notebook.py`: builds `mellamoa/v4-tycho-port` from our Franzen builder's serving cells plus the Tycho tree embedded as a tarball. |
| 0006 | `TYCHO_ALLOW_HOST_SANDBOX`: lets benchmarks run model-written Python on the host. Kaggle has no container runtime; the VM is disposable and offline (user-approved). |
| 0007–0009 | Fix the manifest's throughput block (issue #118). It summed only committed trace steps, so the calls of the turn the wall-clock deadline cut off were dropped. Each game record now carries the agent's cumulative counters (`llm_usage`), and the aggregate takes them, falling back to the status events for older records. Also fixes `length_stops` and `max_tokens_out` reading 0 when the per-call log is on. Analysis only; play is unchanged. 0007 is a pure move. |

## Result (`runs/kaggle/v4-tycho-port/v1`)

Balanced 10-game panel, 25 min per game, all games at once: the F001 v2 setup.

| | Tycho (orchestrator mode) | Our harness without the Fable layer (`v4-fable-f001-off`) |
|---|---|---|
| Mean score | **3.53** | 29.22 |
| Levels cleared | 8 | 27 |
| Actions, all games | 280 | about 820 |
| Generated tok/s per job wall-clock | 478 (server log: 501) | 603 |
| Engine decode tok/s, median (`kaggle/serve_report.py`) | 734 (741 at full batch) | 731 (778 at full batch) |

Games ran 14:20:04–14:45:10 UTC (1507.8 s), all ten stopped by the 25-minute deadline.

Why: 6.53M fresh prompt tokens against 720k generated, and a 64% prefix-cache hit rate (Tycho
rewrites old turns every action). Each call took 22.5 s and about 1,140 output tokens, giving
only 633 completed calls in total. The engine itself decoded as fast as for our harness (median
734 vs 731 tok/s, batch mean 9.5 vs 9.0, accept length 2.66 vs 2.64), so the 21% lower
generated-tokens-per-wall-clock rate is time the server spent not decoding: more prefill (4.3% vs
1.8% of GPU time, queue waits 5.9 vs 0.7 min) and gaps between calls. The world-model builder
never fired. Tycho is built for long frontier-model runs (up to
3,500 calls per game), so it barely acts in a 25-minute budget. See `research/notes/arms.md`,
"Didn't work → Other harnesses".

The first write-up quoted the manifest's throughput block: 367 tok/s, 553k generated, 6.15M fresh,
57% cache hits, 546 calls, 20.5 s and about 1,000 tokens per call, and called the 367 a decode
speed, a 39% gap. Both rates are generated tokens per wall-clock second, not engine decode speed. That block
dropped each game's last, deadline-cut turn (patch 0009 fixes it). The figures above sum the 633
`llm_finished` events in `tycho-results/status/*/events.jsonl`, which equal the per-game
`inference_budget.usage` in `manifest.json`, over the manifest's `wall_clock_s`. The server's own
request log (`serve.log.xz`) agrees: 643 requests (the extra 10 are calls cut off in flight),
756k generated, 6.68M fresh, 64.4% cache hits. Details are in `summary.txt`.
