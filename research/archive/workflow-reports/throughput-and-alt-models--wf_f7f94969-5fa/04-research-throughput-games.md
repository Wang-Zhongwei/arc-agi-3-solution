# throughput-and-alt-models: research:throughput-games

Workflow run `wf_f7f94969-5fa`, phase Research. Win 2+ more games via throughput (sim-v2 sweep + a throughput stack of built-but-unrun fixes) and serve an alternative model (GLM-5.3-Flash EXL3, ISTA Qwen3.8-Flash-Next GSQ-RCO GGUF) by refactoring the RTX Pro 6000 SGLang launcher into per-model profiles

---

**Throughput → games won: simulator sweep, near-win analysis and throughput lever inventory**

> **Correction (2026-10-08, issue #103).** Every "×" speed below is sim2's `speed_scale`, which
> scales only the decode curve; a request's own prefill and the client gaps are not scaled. In
> wall-clock tok/s (generated tokens / run clock, the unit of serve_report's harness tok/s and of
> the lever gains in section 3) the sim's gains are about 80% of the speed gain: ×1.1 is +8.0%,
> ×1.27 is about +21%, ×1.3 is +23%. Restated in wall-clock tok/s (offline, ledger costs):
> +1 game won needs about +10% and +2 games +21% (no-ledger +13% / +38%; hidden set +2 of 110
> needs +7–8%); the slopes are about +0.34 points and +0.10 games (+0.05 ever-won) offline per
> +1%, and +0.28 games of 110 per +1% at the competition setting. The lever table's
> whole-run gains are wall-clock gains but were looked up on the speed axis, so its points and
> games are about 20% low: the safe set's +6% is about +2.1 points and +0.6 games offline
> (+0.3 ever-won) and +1.7 games of 110; +12% is about +4.0 points and +1.2 games. Same caveats
> (upper bounds; the slope is unconfirmed by a real run). `analysis/throughput_sweep.py` on
> `arm/sim-v2` now reports wall-clock tok/s next to every speed and `report --wall-gains`
> does the lookup. The text below is left as written.

**Bottom line**
- The simulator says the extra throughput needed for **+2 games won offline** is **×1.27** with ledger costs and **×1.51** without. On the hidden set (110 games, 532 min), +2 games out of 110 needs ×1.09–1.10. The same share of games as +2 of 25 (+8.8) needs ×1.34 with ledger costs and ×1.74 without.
- **Over half of the ledger gain is on games no real run has ever won.** Those wins come from the simulator's default level costs for levels with few attempts, not from data. Counting only the 12 games some real run has won, ledger gains level off at about +1.2 games (offline). So **+2 offline games from throughput alone is out of reach** on the conservative reading.
- Score moves about **+2.2 to +2.8 points per +10% throughput**, in both settings, up to ×1.2.
- **Startup savings are worth almost nothing:** 150 s saved gives +0.16 points and +0.13 games of 110 at the competition setting. Offline, startup never counts against the game clock.
- **Every lever that is built and doesn't change model outputs, stacked, is plausibly worth +6% (range +3 to +9%):** about +1.7 points and +0.5 games offline (+0.25 counting only ever-won games).
  - Adding relaxed speculative acceptance at 0.5/0.5 makes it about +11%.
  - Reaching ×1.27 needs acceptance 0.3/0.3 plus the FP8 dense-weight port, which isn't written yet. Both change outputs.
- **Caveat:** the simulator's throughput-to-score slope has never been confirmed in a real run that raised throughput.
  - 14 streams: about +2 predicted, +0.13 real, though that run also cut context.
  - 16 experts per token (speed ×0.854): −3.3 predicted, −7.6 real, though routing also changed.
  - Treat these predictions as an upper bound, and differences under 2 points as unresolved.

## 1. Simulator tables

Setup: Franzen scheduler, 10 slots, default fit on the five Franzen runs. Results are paired with common random numbers; sd is the simulator's run-to-run spread, about 2× too wide. Startup seconds saved are added to the game clock. That only matters at the competition setting, and only if the 32,400 s notebook cap cuts the tail (today about 3 min: 667 s startup + 31,920 s > 32,400 s) or the deadline is re-anchored to notebook start. **Ledger costs** means level costs fitted on the runs that used the memory ledger; **no-ledger** means the runs without it.

**Speed needed for more games won (no startup saving)**

| Setting | Base games won | +1 game | +2 games | +2 × n/25 | Games won per +10% speed |
|---|---|---|---|---|---|
| Offline (25 games / 121 min), ledger costs | 7.76 | ×1.13 | ×1.27 | ×1.27 | +0.72 |
| Offline, no-ledger costs | 6.16 | ×1.17 | ×1.51 | ×1.51 | +0.50 |
| Competition (110 games / 532 min), ledger costs | 32.08 | ×1.04 | ×1.09 | ×1.34 (+8.8) | +2.62 |
| Competition, no-ledger costs | 25.46 | ×1.05 | ×1.10 | ×1.74 (+8.8) | +1.61 |
| *Check: offline, ledger, fit on every run* | 8.29 | ×1.16 | ×1.49 | – | +0.52 |
| *Check: offline, no-ledger, fit on every run* | 5.99 | ×1.21 | ×1.55 | – | +0.46 |

**Change in games won, split by whether any of the 10 real full runs won that game**

| Setting | ×1.1 | ×1.2 | ×1.3 | ×1.5 | ×2.0 |
|---|---|---|---|---|---|
| Offline ledger: ever-won games / never-won games | +0.43 / +0.37 | +0.74 / +0.84 | +0.89 / +1.27 | +1.04 / +2.05 | +1.15 / +3.43 |
| Offline no-ledger | +0.51 / +0.12 | +0.91 / +0.27 | +1.14 / +0.37 | +1.44 / +0.55 | +1.97 / +1.00 |
| Competition ledger (80 reps) | +0.82 / +1.26 | +1.68 / +3.12 | +2.31 / +5.40 | +2.96 / +9.68 | +3.46 / +15.78 |
| Competition no-ledger (80 reps) | +1.10 / +0.57 | +2.40 / +1.34 | +2.89 / +1.68 | +3.29 / +2.73 | +5.17 / +4.96 |
| Offline ledger, fit on every run | +0.55 / +0.12 | +0.96 / +0.25 | +1.16 / +0.41 | +1.32 / +0.69 | +1.43 / +1.40 |

Several level estimates are optimistic under ledger costs. For example, ka59 level 6 has 0 clears in 9 attempts across all runs, but the ledger fit sees only 2 attempts, so it falls back to a pooled estimate that sometimes clears it.

**Offline, ledger costs (400 reps)**

Speed changes are at no startup saving; the last two columns stack a startup saving on top of the same speed.

| Speed | Score (sd) | Δ score (se) | Won (sd) | Δ won (se) | Levels (sd) | Δ levels (se) | +150 s: Δscore / Δwon | +300 s: Δscore / Δwon |
|---|---|---|---|---|---|---|---|---|
| ×0.90 | 47.84 (4.66) | −3.21 (0.08) | 6.94 (1.61) | −0.82 (0.04) | 107.1 (6.9) | −5.2 (0.1) | −2.45 / −0.64 | −1.80 / −0.48 |
| ×1.00 | 51.05 (4.89) | 0 | 7.76 (1.66) | 0 | 112.4 (7.4) | 0 | +0.77 / +0.23 | +1.45 / +0.42 |
| ×1.10 | 53.82 (5.20) | +2.76 (0.08) | 8.56 (1.87) | +0.80 (0.05) | 116.8 (7.9) | +4.4 (0.1) | +3.52 / +1.02 | +4.16 / +1.23 |
| ×1.20 | 56.22 (5.43) | +5.17 (0.12) | 9.34 (1.90) | +1.57 (0.06) | 120.5 (8.5) | +8.1 (0.2) | +5.75 / +1.74 | +6.29 / +1.93 |
| ×1.30 | 58.02 (5.65) | +6.97 (0.14) | 9.92 (1.95) | +2.16 (0.07) | 123.2 (8.9) | +10.8 (0.2) | +7.48 / +2.32 | +7.97 / +2.50 |
| ×1.50 | 60.73 (5.90) | +9.68 (0.18) | 10.85 (2.04) | +3.09 (0.08) | 127.1 (9.4) | +14.7 (0.3) | +10.13 / +3.22 | +10.52 / +3.38 |
| ×2.00 | 64.34 (6.41) | +13.29 (0.23) | 12.34 (2.16) | +4.58 (0.09) | 132.0 (10.5) | +19.6 (0.3) | | |
| ×3.00 | 65.86 (6.55) | +14.81 (0.25) | 13.31 (2.08) | +5.55 (0.10) | 134.1 (10.8) | +21.7 (0.4) | | |

**Offline, no-ledger costs (400 reps)**

| Speed | Score (sd) | Δ score (se) | Won (sd) | Δ won (se) | Levels (sd) | Δ levels (se) | +150 s: Δscore / Δwon | +300 s: Δscore / Δwon |
|---|---|---|---|---|---|---|---|---|
| ×0.90 | 42.96 (3.67) | −2.81 (0.08) | 5.43 (1.33) | −0.73 (0.04) | 99.7 (5.5) | −4.6 (0.1) | −2.20 / −0.59 | −1.57 / −0.43 |
| ×1.00 | 45.78 (3.97) | 0 | 6.16 (1.44) | 0 | 104.3 (6.1) | 0 | +0.56 / +0.15 | +1.07 / +0.32 |
| ×1.10 | 47.97 (4.21) | +2.19 (0.07) | 6.79 (1.53) | +0.63 (0.04) | 107.7 (6.5) | +3.4 (0.1) | +2.64 / +0.78 | +3.08 / +0.95 |
| ×1.20 | 49.55 (4.34) | +3.78 (0.10) | 7.34 (1.60) | +1.18 (0.05) | 110.0 (6.9) | +5.7 (0.1) | +4.15 / +1.31 | +4.43 / +1.38 |
| ×1.30 | 50.56 (4.39) | +4.78 (0.11) | 7.67 (1.64) | +1.51 (0.05) | 111.5 (7.0) | +7.2 (0.2) | +5.05 / +1.61 | +5.28 / +1.70 |
| ×1.50 | 51.68 (4.56) | +5.91 (0.13) | 8.15 (1.67) | +1.99 (0.06) | 113.2 (7.5) | +8.9 (0.2) | +6.03 / +2.04 | +6.15 / +2.12 |
| ×2.00 | 53.11 (5.00) | +7.33 (0.16) | 9.13 (1.76) | +2.96 (0.07) | 115.4 (8.2) | +11.1 (0.2) | | |
| ×3.00 | 53.90 (5.31) | +8.12 (0.18) | 9.69 (1.88) | +3.53 (0.08) | 116.5 (8.7) | +12.2 (0.3) | | |

**Competition, ledger costs (150 reps)**

| Speed | Score (sd) | Δ score (se) | Won (sd) | Δ won (se) | Levels (sd) | Δ levels (se) | +150 s: Δscore / Δwon | +300 s: Δscore / Δwon |
|---|---|---|---|---|---|---|---|---|
| ×0.90 | 43.01 (2.28) | −3.09 (0.07) | 29.43 (3.51) | −2.65 (0.14) | 421.0 (14.0) | −26.0 (0.4) | −2.93 / −2.55 | −2.77 / −2.38 |
| ×1.00 | 46.10 (2.28) | 0 | 32.08 (3.63) | 0 | 447.0 (14.4) | 0 | +0.16 / +0.13 | +0.32 / +0.29 |
| ×1.10 | 48.91 (2.40) | +2.81 (0.06) | 34.35 (3.84) | +2.27 (0.12) | 470.8 (15.1) | +23.8 (0.4) | +2.99 / +2.47 | +3.16 / +2.62 |
| ×1.20 | 51.53 (2.45) | +5.43 (0.08) | 37.11 (4.01) | +5.03 (0.17) | 491.1 (15.9) | +44.1 (0.6) | +5.62 / +5.27 | +5.79 / +5.39 |
| ×1.30 | 54.16 (2.54) | +8.06 (0.10) | 39.93 (4.12) | +7.85 (0.20) | 510.7 (16.7) | +63.7 (0.7) | +8.25 / +8.01 | +8.42 / +8.21 |
| ×1.50 | 58.23 (2.70) | +12.13 (0.13) | 44.90 (4.25) | +12.82 (0.25) | 539.6 (18.5) | +92.6 (0.9) | +12.28 / +13.09 | +12.42 / +13.27 |
| ×2.00 | 62.15 (3.09) | +16.05 (0.18) | 51.77 (4.47) | +19.69 (0.27) | 565.1 (22.3) | +118.1 (1.4) | | |
| ×3.00 | 63.61 (3.27) | +17.51 (0.21) | 55.71 (4.66) | +23.63 (0.32) | 573.1 (23.8) | +126.1 (1.6) | | |

**Competition, no-ledger costs (150 reps)**

| Speed | Score (sd) | Δ score (se) | Won (sd) | Δ won (se) | Levels (sd) | Δ levels (se) | +150 s: Δscore / Δwon | +300 s: Δscore / Δwon |
|---|---|---|---|---|---|---|---|---|
| ×0.90 | 39.25 (1.70) | −2.67 (0.06) | 23.01 (2.94) | −2.45 (0.12) | 392.6 (10.3) | −23.0 (0.4) | −2.57 / −2.39 | −2.44 / −2.30 |
| ×1.00 | 41.92 (1.76) | 0 | 25.46 (3.21) | 0 | 415.5 (10.5) | 0 | +0.10 / +0.03 | +0.23 / +0.18 |
| ×1.10 | 44.21 (1.87) | +2.29 (0.06) | 27.38 (3.23) | +1.92 (0.12) | 435.0 (12.0) | +19.4 (0.4) | +2.40 / +2.04 | +2.56 / +2.23 |
| ×1.20 | 46.00 (1.92) | +4.07 (0.09) | 29.29 (3.28) | +3.83 (0.14) | 448.9 (13.0) | +33.3 (0.6) | +4.18 / +3.95 | +4.26 / +4.08 |
| ×1.30 | 46.92 (2.14) | +5.00 (0.11) | 30.30 (3.42) | +4.84 (0.17) | 456.7 (14.9) | +41.2 (0.8) | +5.05 / +4.90 | +5.10 / +4.98 |
| ×1.50 | 48.02 (2.29) | +6.09 (0.13) | 32.01 (3.69) | +6.55 (0.19) | 464.7 (16.1) | +49.1 (0.9) | +6.14 / +6.71 | +6.17 / +6.77 |
| ×2.00 | 49.13 (2.45) | +7.21 (0.15) | 36.04 (3.82) | +10.58 (0.25) | 472.5 (17.4) | +56.9 (1.1) | | |
| ×3.00 | 49.57 (2.50) | +7.65 (0.16) | 37.63 (3.84) | +12.17 (0.26) | 476.0 (17.8) | +60.4 (1.2) | | |

**Games whose win probability moves most at ×1.3 (offline)**
- No-ledger costs: cd82 0.36→0.77, r11l 0.45→0.68, tr87 0.32→0.55.
- Ledger costs: ka59, sp80, cn04 (all never won in a real run), then tu93 0.28→0.47, tn36 and m0r0.

**gate11 run directly in the simulator** (11 holders on the 10-stream curve): +7% tokens. That is +2.27 points / +0.67 games offline with ledger costs, +1.65 / +0.46 without, and +1.94 games of 110 at the competition setting. It is an **upper bound**: the simulator's per-stream rate is flat at 71.6 tok/s for 8–10 streams, while the real serve logs show combined decode nearly flat (683, 697, 710 tok/s at 8, 9, 10 streams).

## 2. Near-wins in the real runs

Across the 10 full runs, 5 games are near-certain wins (lp85, sb26, ar25, r11l, ft09), 7 swing (cd82 6/10, tr87 6/10, vc33 5/10, sc25 2/10, tn36, m0r0 and tu93 1/10 each), and **13 have never been won**.

Games that were not won but whose remaining levels other runs have cleared:

| Run | Game | Levels left | Tokens still needed (vs others' median clear) | Chance the remaining levels clear | What it lacked |
|---|---|---|---|---|---|
| control | vc33 | 3 | 57k (32% of its tokens) | 0.83 | Waited 76 min, re-admitted at 104 min |
| control | cd82 | 3 | 35k (22%) | ~0.38 | Re-admitted at 91 min |
| control | tu93 | 1 | 77k (29%) | 0.14 (level 9 cleared once, at 97k) | Re-admitted at 114 min |
| ledger r1 | m0r0 | 1 | 12k (5%) | 0.25 | Level 6 cleared once, at 53k |
| ledger r1 | sc25 | 3 | 111k (84%) | – | – |
| ledger r2 | tr87 | 1 | 18k (23%), about 4 min of a slot | 0.75 | Only 83k tokens and 25 min active; waited 66 min, re-admitted at 112 min |
| ledger r2 | sc25 | 2 | 53k | 0.27 | Switched out at 95 min, never re-admitted |

- **What they lacked was slot time at the end of the run.** All of them sat 35–76 min in the queue, and most were re-admitted in the last 7–35 min.
- If each got its full deficit, the expected extra wins are about 1.35, 0.55 and 1.0 per run (about 1 on average). That matches the simulator's +0.9 to +1.2 on ever-won games at ×1.3.
- **The rest are walls throughput can't fix.** In every run about 7 unfinished games are stuck on a level no full run has ever cleared: cn04 L6, ka59 L6, dc22 L5, re86 L7, s5i5 L7, ls20 L6, g50t L6, sp80 L5–6. These need new capability (perception, memory), not compute.

## 3. Lever inventory, ranked

Base throughput: about 611 tok/s for the whole run (control), 789 at a full batch of 10, 826 with no prefill in the window. Prefill stalls take 21.4% of wall time, so a decode-only gain of k becomes 1/(0.214 + 0.786/k) overall. Score and games come from the sweep slopes: about +0.28 points and +0.08 games (+0.043 ever-won) offline per +1%, and +0.23 games of 110 per +1% at the competition setting.

| # | Lever | Status | Expected whole-run gain | Offline: points / games | Competition: games of 110 | Safe on stack-v2? |
|---|---|---|---|---|---|---|
| 1 | Speculative acceptance 0.5/0.5 (serving-bench C1) | Built (config only), never run on GPU | +5% (3–7%), about +30 tok/s; acceptance 2.66→~2.85 | +1.4 / +0.4 (+0.2 ever-won) | +1.1 | **No**: changes outputs at temperature 0.7; needs its own score A/B |
| 1b | Same at 0.3/0.3 (C2) | Same | +12% (8–18%) | +3.2 / +0.95 | +2.6 | No, more lossy |
| 2 | gate11, #38 (11 gate slots on the server's 10) | Built (env only) | +2.2% (1.5–4.4%), about +13 tok/s | +0.6 / +0.18 | +0.5 | Yes; watch KV (peaks at 0.99) |
| 3 | Prefix reuse, #53 (`ARC3_VERBATIM_HISTORY`, 60 mamba slots) | Built | +2.5% (1.5–3.5%): recovers 1.3–2.6M re-prefilled tokens, 100–275 s of stalls | +0.7 / +0.2 | +0.6 | Yes, after a one-block merge fix; history keeps blank lines. The 84-slot variant costs about 1.3 GB of KV: not safe |
| 4 | Low-M GEMM widened to M=64, no split-K (#12; serving-bench C3) | Built as bench patches, never run on GPU | +1% (0–3%) | +0.3 / +0.08 | +0.2 | Only after a bench: a 64-row block may be slower |
| 5 | Persisted JIT cache, #54 (fix-6) | Built; needs one cache-saving Kaggle run plus a published dataset | Startup −145 s; nothing offline | 0 | +0.12 (+0.15 points) | Yes; falls back to a cold start on mismatch |
| 6 | Stub dead reasoning, #39 | Built (env only) | +0.2–0.5% (fewer trims); mainly cleaner context | ~+0.1 / ~0 | ~0 | Yes |
| 7 | CUDA graphs for batch sizes 3, 5, 6 | Launcher setting | ≤+0.3% | ~0 | ~0 | Yes (about 70 MB each) |
| 8 | Chunk 16384 / max prefill | Launcher setting | Max prefill does nothing (no batch over 8,192); chunk 0 to +1%, untested | – | – | Bench first |
| 9 | Schedule policy lpm → fcfs | Launcher setting | About 0 (queue non-empty in about 2% of decode logs) | 0 | 0 | Keep lpm |
| 10 | Request-log flags | On | <0.5% | 0 | 0 | Keep: they are our only diagnostics |
| 11 | `OMP_NUM_THREADS=1` / overlap plan stream | Not built | ≤1–2%, weak prior | – | – | Skip overlap plan stream (broke native MTP in a community setup) |
| 12 | History-state performance fix, #43 | Built | +1–3% (shorter client gaps), unmeasured | – | – | No: it was in stack-v1, which lost; stack-v2 leaves it out deliberately |
| 13 | NVFP4 checkpoint, #47 | Built on stack-v1 (D′) | −20% to +10%: Franzen found similar throughput and less KV; its native draft head measured −22.9% at 4 streams | – | – | No: model swap, and its build script conflicts with fix-6 |
| 14 | `reasoning_effort` low | **Not built**; only the after-truncation ladder exists | 0 tok/s; changes tokens per action, which the simulator can't price | – | – | Needs a 20-min smoke run (gate: −25% tokens per request) |
| 15 | 14 streams (C7) | Configurable | +15% combined throughput, but needs a context cut | Real run: +0.13 | – | No |
| 16 | FP8 dense-weight port (Gabriel 0005/0006) | **Not written** | +4–11%; costs about 3.3 GB of KV | +1.1–3.0 / +0.3–0.9 | – | n/a |

**Combining on arm/stack-v2**
- **Safe set (lossless or near):** gate11 + prefix reuse (60 slots) + stub dead reasoning + CUDA graphs 3/5/6 + low-M GEMM (if the bench passes) + JIT cache.
  - About ×1.06 (range ×1.03 to ×1.09).
  - Offline: about +1.7 points (0.8–2.5) and +0.5 games (+0.25 ever-won).
  - Competition: about +1.4 games of 110, plus about 0.1 from the JIT cache.
- With acceptance 0.5/0.5: about ×1.12 → +3.2 points, +0.9 games.
- With acceptance 0.3/0.3 plus the FP8 port: about ×1.27, the level the default fit needs for +2 offline games. Under the ever-won-only reading +2 is still out of reach.

**What merging takes**
- stack-v2 has no editable SGLang launcher. Merging arm/serving-bench (which includes custom-serving) is clean.
- The low-M GEMM patches sit in `serving/bench_patches/`, so they must move to `serving/runtime_patches/` and be listed in the launcher's runtime-patch setting.
- `kaggle/arms.json` conflicts on every branch, but only because each adds its own arm.
- fix-3 has one real conflict, in `tool_agent.py` around line 7082: keep both the ledger lines and `reply_message = assistant_message`.
- fix-6 merges cleanly with serving-bench. nvfp4 conflicts with fix-6 in `kaggle/build_notebook.py`.

## Files and commits

Three commits on **arm/sim-v2**; nothing pushed, no Kaggle runs (out of quota). sim2 already had a throughput knob (`speed_scale`), so I added only a sweep driver and its tests:
- `360377e` throughput_sweep.py: paired speed and startup sweep with games won
- `99cbaa3` throughput_sweep.py: split games won by real-run wins; --calib option
- `0200c0b` README: document throughput_sweep.py

Simulator CPU used was about 23 min, run one job at a time under `nice`.

Files are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad:
- wt/sim-v2/analysis/throughput_sweep.py
- wt/sim-v2/analysis/test_throughput_sweep.py
- wt/sim-v2/analysis/README.md
- tput/tput.jsonl (main sweep)
- tput/tput.md (full report)
- tput/condensed.md
- tput/comp_split.jsonl
- tput/calib_all.jsonl
- tput/follow.log (includes the gate11 results)
- tput/gate11.py
- tput/nearwins.py
