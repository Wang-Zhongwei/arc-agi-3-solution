# Research archive (up to 2026-10-14)

> Archived on 2026-10-14, when the research folder was restarted. Everything that was in `research/` up to
> that day is here unchanged, and its internal links still work. The current state is in [`../README.md`](../README.md).

This folder holds the research behind this fork: what we tried, what we measured, and why. The code for each idea lives on its own `arm/<name>` branch with its own PR. The raw Kaggle run logs live next to it in [`runs/`](../../runs), compressed with xz.

Snapshot refreshed 2026-10-08. The living copies of `notes/` sit outside the repo and change daily; this folder is refreshed by commits to `main`.

## Start here

| File | What it is |
|---|---|
| [`notes/arms.md`](notes/arms.md) | Every idea ("arm") with its status: worked, didn't work, in progress, backlog. It also covers the target (56–60 offline on the 25 public games, 121-min run) and the noise levels to read results against. |
| [`notes/research-sweep.md`](notes/research-sweep.md) | 39 improvement ideas, each checked against Franzen's code, Kaggle limits and our own results, ranked into tiers. |
| [`notes/scratchpad.md`](notes/scratchpad.md) | The owner's own hunches: what might be an alpha and what probably isn't. Opinions, not results. |
| [`reports/m1-to-m2-trajectory.md`](reports/m1-to-m2-trajectory.md) | What changed from Milestone 1 (Tufa's Duck harness) to Milestone 2 (Franzen and other placers), fact-checked. |
| [`reports/fable-vs-ours.md`](reports/fable-vs-ours.md) | Comparison with leoprovorov's "Fable & Astra" handbook notebook and its Fable layer. |
| [`reports/failure-taxonomy.md`](reports/failure-taxonomy.md) | Where tokens, actions and wall time go in 7 full runs (levels never cleared, idle python calls, cut-offs, errors, deaths), which arm covers each cause, and three new arms for the uncovered ones. Tool: `analysis/failure_taxonomy.py` (PR #77). |
| [`reports/score-timeline.md`](reports/score-timeline.md) | How the score grows over wall-clock in 11 full runs and 12 panels: half the final score comes after minute 50, minute 25 barely predicts the final ranking (r = 0.21), and ledger-on-trim's lead appears only after minute 50. Charts and CSVs in [`data/score-timeline/`](data/score-timeline/); tool: `analysis/score_timeline.py` (PR #78). |
| [`reports/tycho-world-model-port-design.md`](reports/tycho-world-model-port-design.md) | Porting Tycho's executable world model into Franzen's harness: evidence that it hasn't helped at our scale (DF-WM 29.3 ≈ Franzen; Makarov: ignored), a cheap adoption screen (`panel-dfwm`) first, then verify/plan helpers + an auto-check line. |
| [`reports/animation-combo-design.md`](reports/animation-combo-design.md) | Franzen + Wang animation: rebase PR #40 into layers. Strongest: `ARC3_ANIM_TRUTH`, as 20% of NO-OP echoes were on animated actions that changed and restored cells. |
| [`reports/sandbox-v3-design.md`](reports/sandbox-v3-design.md) | The owner's Wang-era sandbox work merged on top of #76: teaching errors, an exact per-action change line, a few helpers, a game-long store, hardening. With usage counts from real runs. |
| [`reports/inspiration/`](reports/inspiration/) | Inspiration research (2026-10-08): game-solving AI, automated science and experiment design, what reasoning is (with talk timestamps), quant methods for scheduling. Each ends with features ranked by evidence; most evidence is frontier-model (tier 4), so treat as ideas for our local Qwen, not results. |
| [`reports/harness-ideas-harvest.md`](reports/harness-ideas-harvest.md) | Safe ideas from other harnesses (Tufa, Lord Han Solo, rellik13, Fable, Tycho, DF-WM, Wang, ...) with evidence tiers for our local Qwen; top 10 ranked ADOPT/TRY/SKIP. Main finding: our model ignores per-turn world-model prose but uses harness facts, one-off asks at defining moments, and tool parameters. |
| [`reports/prompts-side-by-side.md`](reports/prompts-side-by-side.md) | Every harness's effective system prompt and python tool description as run, verbatim, plus a section-by-section comparison and each harness's world-model-like elements. Ours is byte-identical to a real v4-panel-control request. |
| [`reports/prefill-v2.md`](reports/prefill-v2.md) | Prompt tokens the server re-processes: yield openers, the opener's static block, duplicate resumption images, and the drain level, measured from request logs and sized in sim2 at matched wall-clock. The yield waste is about 4–5% of prefill, not 31.7%; every lever lands within ±0.7 points, and draining to 80k loses. Tools: `analysis/prefill_levers.py`, `analysis/prefill_sizing.py` (on `arm/prefill-v2`). |
| [`reports/convergence-plan.md`](reports/convergence-plan.md) | How to stop the model dropping correct ideas (2026-10-11): ranked prompt versions and harness systems (trim-time harvest of its own confirmed sentences, row-run annotations, RESET confirm/vote, scoped no-op wording, plan-vs-code check, ...), the state-lab extensions, and a round plan: OpenRouter screens, Kaggle lab notebooks on the real W4A16 model (~15 GPU-h), then paired confirmation runs. |
| [`reports/alpha-table.md`](reports/alpha-table.md) | Every full run's score against its reference with a noise model (pooled run sd 2.47; placebo among identical runs reaches ±6). Real: ledger-on-trim (+5.2, both runs above all controls) and the stack vs Franzen controls (+7.3); the stack's gain over the same-era control (+2.3) is not proven. Real losses: hats, 18 streams/64k, 16 streams/80k, throughput levers, 16 experts. Script `analysis/alpha_table.py`. |
| [`reports/next-moves.md`](reports/next-moves.md) | What to do with three weeks left (2026-10-10), after throughput and action efficiency ran out. Of the 47 missing points per run, 45 are on levels never cleared. The plan: a "victory lap" that replays the shortest route after a win (the scorer keeps the best play, about +0.8 on the best stack); a final-pick protocol (two copies of one frozen config); decision rules at the stack's own noise (sd 3.3–3.9: +3 needs 15–21 runs per arm); a 16-run factorial of the built flags; pods for power; and a week-by-week GPU schedule. mem-v2-10x scored 43.81. Scripts in [`data/next-moves/`](data/next-moves/). |
| [`reports/vision-object-model.md`](reports/vision-object-model.md) | Learned object models benchmarked on exact engine ground truth (2026-10-10): MobileSAM +8.2 F1 but 2–3 s/frame on CPU; a tiny engine-trained CNN-ViT fails on held-out games; lexicon-cc +3.5 F1 in 4 ms. Oracle segments don't improve player identification, so the next step is a behavioural entity tracker (revised spec in the review). |
| [`reports/near-miss-games.md`](reports/near-miss-games.md) | Why re86, su15, ka59 and dc22 are never won (2026-10-10, nine v4 runs): the clock and parks, verified rules lost at trims, the selected object lost after SPACE, unprobed novel objects. Per-lever win-probability estimates; the scheduler levers measured small (final-level hold +0.04–0.10 full clears per run, PR #155; park cap rejected). |
| [`reports/memv2-reasoning-audit.md`](reports/memv2-reasoning-audit.md) | Turn-by-turn audit of the mem-v2-10x run (43.81; 2026-10-10), checked against frame-exact engine replays. The model dropped a correct idea 143 times in 25 games (0 of them wrong), on the final level in 18 of the 19 unfinished games. Most reversals came from misreading its own output, losing rules at trims, or trusting its own broken searches, not from pure doubt. It also covers code vs reasoning vs actions (279 mismatches), what memory-v2 actually saves (paths and colour tallies, no rules; its prompt wording cut model notes from 34–58 per run to 10), and nine ranked harness fixes. Data in [`data/memv2-reasoning-audit/`](data/memv2-reasoning-audit/). |
| [`reports/state-lab-round1.md`](reports/state-lab-round1.md) | Re-sampling the audit's worst moments with and without harness facts and guards, on OpenRouter Qwen3.8 Flash, with each reply executed on the exact engine (2026-10-11). The round stopped at 22 of ~290 samples (no OpenRouter credit), so no variant can be ranked: 2 vs 2 per cell, smallest possible p 0.33. The loop guard is mechanically solid (stops tr87's 102-action loop at 13 and avoids the GAME OVER) and goes into stack-3h as `ARC3_REPEAT_STATE_GUARD`. The untried-RESET guard was re-confirmed 2/2. The quoted-rule fact was the only change that produced the target move (1/2). win_snapshot was never sampled. Tool: `analysis/state_lab/analyze.py` (on `tooling/state-lab`). |
| [`reports/leaderboard-35-45.md`](reports/leaderboard-35-45.md) | How teams reach 35–45 on the public LB (2026-10-08): mostly best-of-N luck plus private throughput/KV/quantization work; top public notebooks diffed against Franzen. |
| [`tycho-port/`](tycho-port/README.md) | NIMI's Tycho harness ported to Kaggle on our model (patch series and result: 3.53 vs 29.22 on the balanced panel). |

## Layout

- `notes/`: living notes, copied from the project folder. `project-CLAUDE.md` holds the working rules the agents follow (branching, PRs, docs, laptop load, Kaggle model).
- `reports/`: finished write-ups.
  - `engine-model-research.md`: alternative engines and models.
  - `throughput-900.md`: the plan for 900+ tok/s.
  - `ling-vs-qwen-plan.md`: a staged go/no-go plan to compare Ling-3.0-flash-int4 with Qwen3.8-Flash-Next: OpenRouter first (models only, about $6 per panel pair), then Kaggle (upstream sglang 0.5.19 wheelhouse, 77 GB staging, sm120 bench, text-only panel pair) only if Ling leads.
  - `definitely-fixes.md`: fixes for the DEFINITELY bottleneck issues.
  - `memory-v2-measurement.md`, `memory-v2-design.md`: how memory fails in real transcripts, and the harness-memory design built from that.
  - `memory-main.md`: memory-v2 and level-path merged onto the user-combo-10x stack (`arm/memory-main`): what was ported, the fixes, the three arms and their keep rule.
  - `memory-layout.md`: the user's memory layout on `arm/memory-main` (`ARC3_MEM_EVERY_CALL`, `ARC3_MEM_LEGIBLE`, `ARC3_LEVEL_COMPACT`, the recap precision fix): design, cache cost replayed on logged runs, before/after samples, the `mem-layout-10x` arm.
  - `fable-handbook-digest.md`: a page-cited digest of the 90-page Fable handbook.
  - `gb10-overnight-results.md`: the GB10 arm tests from the old harness.
  - `harness-feature-matrix.md`: features across competitor harnesses.
  - `failure-taxonomy.md`: where tokens and actions are wasted, per cause and per run.
  - `score-timeline.md`: score, levels and tokens over wall-clock, with arms compared at matched minutes.
- `workflow-reports/`: every multi-agent research and build run, with one file per agent report in run order. [`INDEX.md`](workflow-reports/INDEX.md) lists them. The largest are:
  - `arc3-idea-sweep`: the research sweep.
  - `arc3-bottleneck-audit`: the source of GitHub issues #2–#33.
  - `scheduling-policy-search`: scheduler candidates in the replay simulator.
  - `sim-v2-calibrate-validate-explore`: the calibrated simulator.
  - `m1-vs-m2-trajectory`.
- `old-harness/`: studies from the earlier Wang/Duck-based notebook.
  - `run-atlas/`: every old run with charts and findings.
  - `world-model-study/`, `context-window-study/`.
  - `wang-notes/`.
  - Franzen level-cost analyses.
- `data/`: small measurement outputs and the scripts that made them. `memory-measure/` holds the transcript measurements (ledger tag rates, revisits, deaths). `sched-results/` holds the scheduler simulation grids. `failure-taxonomy/` holds the failure-taxonomy tables and per-response rows. `score-timeline/` holds the per-run score-over-time CSVs and the HTML overlays (`full-runs.html`, `panels-*.html`). `next-moves/` holds the scripts and tables behind `reports/next-moves.md` (loss decomposition, victory-lap replays, power tables, the factorial design). `memv2-reasoning-audit/` holds the per-game analyses, the synthesis and verifier reports, and the tag-count and replay scripts behind `reports/memv2-reasoning-audit.md`.

## Where results come from

- **Full runs:** 25 public games, 121 minutes, one RTX Pro 6000, W4A16 Qwen3.8-Flash-Next. Score with `python -m inference.tools.eval <run_dir>` from `ARC3-Inference/`, and compare two runs with `make significance`. Per-game swings are 50–80 points and run-to-run noise on the mean is about ±2.7, so one run is a screen, not a verdict.
- **Run outputs:** every Kaggle kernel version on the account is in [`runs/`](../../runs). Its README explains the layout and how to unpack the `.xz` files.
- **Simulators:** `analysis/` on `arm/sched-policies` (the replay simulator) and `arm/sim-v2` (the calibrated event-driven simulator).
