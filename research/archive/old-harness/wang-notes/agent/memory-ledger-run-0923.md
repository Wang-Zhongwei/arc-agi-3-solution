# Memory ledgers: the wang-intel-memory-fixed run

2026-09-23. Kernel `mellamoa/taaf-wang-intel-memory-fixed`, harness commit
`c6de0ac`, config `configs/inference.flash-next-intel-w4a16.json`.
**Mean RHAE 13.65 on the 25 public games, 49/183 levels, and the first two
games any run of this harness has ever completed.** Baseline for comparison is
`taaf-wang-intel-w4a16-0922` (9.86), whose submission scored 2.40 publicly.

This is currently the best-measured configuration and the control that later
arms are judged against. Read [allocation.md](../../ARC3-Inference/docs/allocation.md)
for the scheduler and [../serving/](../serving/) for the serving stack.

## What it runs

| | |
| --- | --- |
| checkpoint | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` via `mellamoa/flashnext-w4a16-serving` |
| allocation | `ucb`, `ucb_c` 2e-4, quantum 300 s, **exploration_fraction 0.25** |
| budget | 132 min public / 524 min submission, 8 concurrent, 1 pass, 25 games |
| serving | `gpu_memory_utilization` **0.96**, MTP 3, prefix caching on, `max_num_seqs` 12 |
| perception | images on, upscale 4, animation 8 frames |

## What it adds over the stock duck harness

Three changes, all on top of upstream's agent loop. None of them touch the
solver's game logic; they change what the model is told and how time is split.

### 1. Memory ledgers (`inference/agent/memory_ledger.py`)

The analyzer already parsed `World model:` lines out of its own prose — and then
never sent them back. `ToolAgent._summarized_knowledge_lines` had **no caller
anywhere in the repo**, so every turn re-derived the game from nothing. That is
the bug this run fixes.

Three bounded ledgers are now carried turn to turn and rendered into the prompt
as the same Qwen3-style XML the model writes:

    <world_model>a ship rises when the column above it is clear</world_model>
    <hypothesis status="confirmed">MOUSE removes one green block</hypothesis>
    <hypothesis status="refuted">SPACE rotates the ship</hypothesis>
    <level_note>level 4: drill the ceiling gap first</level_note>

Splitting confirmed from refuted is the point: a refuted hypothesis is what
stops the agent spending actions on a dead idea twice. Tags are extracted from
anywhere in the reply and the surrounding prose is left alone, so the response
stays fluid and only what the model wants remembered becomes structured.
Level notes are distilled from the model's own `<think>` trace by keeping
sentences that state a finding — no second model call, since traces average
~1000 tokens and reach 39k.

Bounds: 6 world-model lines, 8 per hypothesis status, 6 level notes, 200 chars
each; re-stating a line refreshes rather than duplicates it; empty ledgers
render nothing, so turn one pays no tokens.

**Stale-block fix (`c6de0ac`).** The block is re-rendered every turn, so leaving
past copies in history handed the model one contradicting snapshot of its own
memory per turn — four blocks in a single request by turn four, each an earlier
and smaller version of the current one. They are now stripped from persisted
history the same way transient images are. A side effect worth naming: because
the live copy is rebuilt from the ledger each turn, **memory now outlives
context trimming**. It is state, not transcript.

### 2. Exploration floor (`inference/framework/allocation.py`)

`ExtensionLedger.choose_ucb` never consulted `exploration_fraction`. With
`ucb_c` small, a game that has never scored sits at priority ~0 and is never
picked again, so it is abandoned while budget remains. In the 0922 baseline
**all 158 admits had reason `ucb`**, and the eight zero-score games got 2.0
slices and 10.8 minutes each against 8.4 and 47.2 for scoring games.

`choose_ucb` now honours the exploration share, except when an unplayed game
(score `inf`) would win on score anyway. At 0.25 this produced 41 exploration
admits and cut zero-score games from 8 to 3.

### 3. `gpu_memory_utilization` 0.90 → 0.96

A larger KV pool. Free throughput: 199 → 303 generated tokens/sec, no OOM, the
serving watchdog never fired.

## Results

Mean 13.65 (baseline 9.86), 49 levels (37), 3 zero-score games (8), 2,833
actions (1,589), 303 tok/s (199), and **2 games won where no previous run won
any**.

| game | baseline | memory-fixed | delta | levels |
|---|---|---|---|---|
| `ft09` | 47.62 | 100.00 **won** | +52.38 | 4→6/6 |
| `lp85` | 58.33 | 95.21 **won** | +36.87 | 6→8/8 |
| `cd82` | 0.00 | 13.66 | +13.66 | 0→2/6 |
| `re86` | 16.67 | 27.78 | +11.11 | 3→4/8 |
| `sb26` | 2.78 | 12.74 | +9.96 | 1→4/8 |
| `sc25` | 3.50 | 13.30 | +9.80 | 1→3/6 |
| `s5i5` | 1.09 | 8.33 | +7.25 | 1→2/8 |
| `cn04` | 0.00 | 4.76 | +4.76 | 0→1/6 |
| `ka59` | 0.00 | 3.57 | +3.57 | 0→1/7 |
| `dc22` | 0.00 | 2.95 | +2.95 | 0→1/6 |
| `wa30` | 0.00 | 2.22 | +2.22 | 0→1/9 |
| *(9 games unchanged)* | | | 0.00 | |
| `tn36` | 3.57 | 2.82 | −0.75 | 1→1/7 |
| `m0r0` | 14.29 | 4.76 | −9.52 | 2→1/6 |
| `vc33` | 21.43 | 3.86 | −17.57 | 3→2/7 |
| `ar25` | 41.67 | 8.33 | −33.33 | 5→2/8 |

Eleven games improved, four regressed, ten were flat. The regressions are the
exploration floor working as designed: time moved off games that were already
progressing. `ar25` lost three levels, `vc33` one.

## Evidence the feature actually ran

Verified from the run's own artifacts, not inferred from the score:

- **1,209** `<memory>` blocks reached prompts across 25 transcripts.
- Maximum **1** memory block per user prompt (the staleness fix holding).
- **CORRECTED 2026-09-26.** This section originally read "Tags written by the
  model: 1,794 `<level_note>`, 1,065 confirmed, 935 refuted, 989
  `<world_model>`". Those are occurrence counts over whole transcripts, and the
  ledger is **re-rendered into the prompt every turn** -- so one tag is counted
  again on every subsequent turn of its game. They measured the feature's own
  output, not the model's. Counting only distinct entries, and only in regions
  the model actually wrote:

  | | this run | b1-thinking |
  | --- | --- | --- |
  | tags the model wrote in its `<think>` traces | 0 | 1 |
  | tags in assistant text | 8 | 16 |
  | **distinct entries that reached the ledger** | **25** | **48** |
  | games carrying any ledger content | 12/25 | 16/25 |

  Of b1's 48, thirty-four are `level_note` -- the one type `note_think()`
  **auto-distills from the trace with no tag required**. So the model
  contributed roughly 14 tags across 25 games and 950 turns, and nine games
  carried no memory at all. The ledger's gain is real, but it came mostly from
  code reading the trace, not from the model complying with the tag
  instruction. See [prompt-compliance-0926.md](prompt-compliance-0926.md).
- A real ledger from `bp35`, carrying mechanics across a level transition:

      <world_model>a yellow/blue ship in a vertical tower of sky chambers
        separated by rock bands. Only LEFT/RIGHT and MOUSE...</world_model>
      <hypothesis status="confirmed">MOUSE on a green block removes only that
        block (21 cells of N)</hypothesis>
      <level_note>level 2: each room's ceiling gap was plugged by 3 green
        blocks; drill them, slide under the gap, ship rises</level_note>

This check matters more than it sounds: two later arms (`a3-thinking`,
`a4-scratchpad`) shipped with their features silently inactive because an env
key never reached the notebook, and were only caught by grepping transcripts for
the prompt text. Always verify a feature from the output, never from the config.

## What pays off

1. **Memory ledgers** — the largest single gain measured, and the only feature
   with a visible mechanism in the transcripts. It also produced the first
   completed games.
2. **Exploration floor** — reliably converts zero-score games (8 → 3). It is a
   redistribution, not free: expect deep games to give something up.
3. **GPU utilisation 0.96** — +45% tokens/sec for nothing.

## What does not

- **`spend-actions` prompt** (telling the agent to stop conserving actions):
  **RESOLVED 2026-09-26, rejected.** Rerun as `b4-scoring-guide` at this run's
  settings: 11.12 against 13.65, 41 levels against 49, on 2,775 turns against
  2,833 -- so 1.48 levels per 100 turns against 1.73. The turn-matched control is
  negative in every group (-2.80 starved, -2.38 matched, -2.37 fed) and the sign
  test is 6 up / 14 down. The earlier "best on robust metrics" reading does not
  survive matching on turns. Delivery was verified: the text reached all 1,022
  prompts and the line it contradicts was correctly removed (0 occurrences
  against 935 in this run), so this is a clean negative, not a dead arm. Note
  the model did *not* spend more actions either -- same turns, fewer levels.
- **Wang's v5 stack** (NVFP4, MTP off, prefix caching off, `max_num_seqs` 28):
  6.48 on these same games. His public 6.18 came from serving discipline, not a
  better agent.
- **`ucb_marginal` / `ucb_knapsack`**: **the 1.2 figure is void.** Both were
  missing from the tuple that routes a run into `_run_allocated_games`, so they
  fell through to the plain one-task-per-game path and played as **uniform**,
  with `_selection_priority`'s marginal and knapsack branches dead code. The
  1.2 measured uniform at a 30-minute budget. Fixed 2026-09-26 (`0a4adaf`): the
  routing set is a named constant with a test asserting every extension policy
  is in it. Both are still genuinely untested.
- **Scratchpad**: **retested 2026-09-26 as `g1-scratchpad2h`** with the flag
  verified reaching the notebook, 132 min, everything else identical: 7.21
  against 13.65, and the model wrote **1** `<scratchpad>` tag across 997 turns
  that all carried the instruction. A negative now, not a null -- and the reason
  is compliance, not the idea.

## Caveats

- **n=1, 25 games.** Two identical configurations (`a3` vs `a4`, both running
  plain base config after a flag failure) scored **3.00 vs 1.45** with 13 vs 8
  games scoring. That is the noise floor: treat any gap under roughly 2× as
  unresolved. This run's edge survives it mainly because "2 games won vs 0" is
  categorical.
- **Offline does not predict the leaderboard.** 0922 scored 9.86 here and 2.40
  publicly; the `diff-perception` build scored 5.23 here and 4.37 publicly —
  the two points are *anti*-correlated. The scored rerun plays ~110 hidden
  games (25 public / 55 semi-private / 55 private per the ARC-AGI-3 technical
  report), and the public 25 are described as deliberately easier.
- **R0 introspection was unintentionally on.** `LOCAL_ANALYZER_TOP_LOGPROBS`
  was meant to be 0 but its env key was missing, so per-token logprobs were
  requested at 5. The 13.65 includes whatever effect that had.
- Scoring is RHAE: `min(115, (human_baseline/our_actions)^2 * 100)` per level,
  weighted by level number. On completed levels this agent runs at ~0.72x the
  human action count with 29 of 49 levels at the 115 cap, so **~85-90% of all
  lost score is levels never finished, not actions wasted**. Optimise coverage
  and depth; never action economy.
