# scheduling-policy-search: research:competitors

Workflow run `wf_f55d767d-e13`, phase Research. Find new slot-scheduling policies (competitors, theory, prior work), simulate them against D-prime, verify the best

---

# Slot/compute-allocation policies used by other ARC-AGI-3 teams

C = `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors`. W = `/Users/advaith/Documents/studying-stuff/experiments/wang-duck-harness`, a local copy of Wang-Zhongwei's repo that is not under C.

Coverage gaps:
- Kaggle discussion pages (744792, 745062, 717133) and lordhansolo's Kaggle writeup are JS-rendered and returned no text, so I used only local copies.
- WebSearch found no public scheduling discussion beyond Franzen's writeup.
- github.com/Wang-Zhongwei/duck-harness returns 404, so W is the only copy.
- The code of the scorers above 31 LB is not public.

## 1. Fixed per-game wall cap with FIFO waves (stock Tufa/Duck solver)
- **Logic:** `asyncio.Semaphore(concurrency)` around `_play_one`. Each game holds its slot until its own `max_runtime_s_per_game` expires or it ends. There is no ranking and no re-queue (`C/duck-harness/ARC3-Inference/inference/framework/solver.py:898-904`).
- **Duck-era notebooks:** 28 concurrent games x 7,920 s, 4 waves. These are `arc3-b32-reap-…:717-719`, `taaf-flashnext-sheetu12b-0922:2983-2986` (its text says "c16's schedule untouched") and the `duck-qwen3-8-*` notebooks.
- **lordhansolo (LB 23.84):** 14 workers x 65.3 min x 8 waves (`C/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/Makefile:544-548`), plus a 150 s per-turn yield (`configs/inference.json:82`). It also adds a strategy-audit prompt once a level has used 25% of the game's time (`framework/solver.py:239-249`). Scheduling is otherwise none.
- **Others:** polyphony uses a pull queue with 1,800 s per game and shuffled order (`C/polyphony-arc-3/arc_hs/swarm.py:99-131`). dream-team uses FIFO queue slots (`C/dream-team/arc_agi_3/multi_runner.py:299`).
- **Score:** Duck-era notebooks scored about 5-6 LB.

## 2. Franzen's priority gate and the variants he rejected
- **Baseline (you know it):** P=(A+B)·C with a 10-stream gate, trim-time handover and a 20% tail fade. LB 27.89.
- **Rejected in his own tests** (`C/tong-m2-comparison/WRITEUP.md:305`): pace adjustment, and an empirical continuation lookup with both A and B normalized. Neither showed an advantage over a constant tail (B=5 earlier).
- **Existing code paths:** `TAIL_LOOKUP_80K` table in `priority_scheduler.py`, and `max_generated_tokens_per_game` (a per-game token cap that exists but is off).
- **Variance:** unchanged Franzen copies averaged 25.77 with SD about 3.9 (`research-sweep.md:15`, author-reported, discussion 745062). The same code scored 27.80 then 21.01 (`C/kaggle/arc-agi-3-27-80-lb-100-trajectory-audit`).

## 3. Sovereign-agent: Franzen's priority with different env values
- **Settings:** `TAIL_FADE_FRACTION=0.05` instead of 0.2, so deep games are kept until the last 5%. Also `LOCAL_ANALYZER_YIELD_SECONDS=90`, `max_generated_tokens_per_game=750000`, and `ARC3_MAX_ACTIVE_STREAMS=10` (same as Franzen).
- **Source:** `C/kaggle/arc-prize-2026-sovereign-agent`, the "5. DYNAMIC PRIORITY SCHEDULER" block in the customization cell.
- **Score:** none in the notebook. The 750k cap is non-binding, since Franzen's runs average roughly 175k tokens per game.
- **Unchanged copies** (patch byte-identical, same priority env): dfranzen-fork, arc-ak47-3-agi, kinetic-learning-v8. No scheduling change.

## 4. Affectify form v1 and D' (shiiin9, Osaka Metropolitan Univ.)
Source: `C/kaggle/affectify-arc-dfranzen-form-d1`, the `_FP_SOURCE` string in the second-to-last code cell.
- **Form v1** (`CHOSEN`):
  - priority = e^{0.2(ℓ-1)}·norm(N) · 2^{-a/100} · pace · 2^{-(t/124000)²} · (1+bonus/8).
  - pace = clip((30000/mean tokens per level)^0.25, 0.25, 4).
  - bonus is 4.0, 3.5 or 2.5 when 3+, 2 or 1 levels remain.
- **D'** = A·M·C + B·φ, as in your notes, with one detail you should check against your port. The notebook also removes "unstarted games first" (the 2M priority band) and prices fresh games by the formula at ℓ=1, a=t=0. This replaces `_PriorityGate.acquire` for unstarted games. Its README lists this as change 2.
- **Score:** D' 31.54 LB, one submission, within noise of 27.62-31.47 for unchanged copies. The notebook says sim gain over v1 is about +1.5 and keeps the coefficient derivation private. Form v1 has no LB number in the files.

## 5. Affectify lottery (Duck plus ours, Duck-era, ~5.7 LB)
Source: `C/kaggle/duck-plus-ours` (cell 17, base64 module `dpo_blob2/3` and plain cell code).
- **Policy:** all 110 games stay alive. A slot (`OURS_SCHED_WIDTH=8`) is re-drawn per analyzer turn, with probability proportional to tickets (`AttentionScheduler.draw`).
- **Tickets:** with `OURS_STALL_OUTSIDE=1`, tickets = max(1e-4, (base + depth·alive)·decay).
  - depth = (ℓ/(N(N+1)/2) · 55)^pow.
  - `alive` = distinct board hashes among the last 20 actions on the current level, divided by 20. It is computed from board history alone, with no model query.
  - decay = 1 for stall ≤ 80 actions, else max(floor, 1/(1+(stall-80)/40)). The knee is from a measured clear rate that falls from 61% to 14% at 80 actions. Floor is 0.1 (version "6") or 0.03 (version "7").
  - Anneal with u = wall-clock progress: pow = 0.4+1.5·u³ and base = 1.0·(1-u³).
- **Score evidence:**
  - Version 6 scored 5.68 LB (rank 7 on 2026-09-09). Best public was 5.75.
  - A variant with the stall knee scaled by level (slope=20) scored 4.00 LB, a 2.8σ drop. Rehearsal showed 7.697 vs 10.647 for the base version. Loosening the knee was the wrong direction.
  - An earlier ticket form, 1 + level·alive, was run at 110 clones for 2 h. Its attention went 70% to level-0 games.

## 6. rellik13 / sirikilohit (LB 22.53, #3)
Source: `C/rellik13-milestone2/cells`.
- **M41/M43, two-tier permit scheduler** (`cells/21…py:362-620`; the notebook's `_N=28` setting is not what produced 22.53):
  - Permit: a game gets an admission worth X=20 actions (the median level-1 clear cost over 923 clears), then parks. It also parks immediately on a level-up. "Park, never kill" keeps the game's thread and state.
  - Rank: fairness first, then (tier, ℓ/T/a²), where tier = 1 if ≥1 level cleared and T=N(N+1)/2.
  - Throttle: past 60 actions on a level, a grant shrinks to 4 actions, about one turn.
  - Fit: points per action fall as actions^-1.73. Measured on 2,443 attempts: 0.495 pts/action at 1-10 actions, 0.035 at 40-60, 0.006 at 60-100.
  - Earlier versions M37 ("depth stop": stop games that have cleared nothing) and M39/40 (stall threshold plus token allowance) were superseded.
  - Evidence: 336 park events, mean 18.7 actions per grant. It is a documented dead path in the final submission. `run/m43_summary.json` shows 0 grants because M72 switched it off.
- **M72, one-pool UCB with long slices** (`cells/44…py`, ported from Wang's `allocation.py`, see section 7):
  - Warm-up: every game gets one 2,400 s slice (300 s originally; raised because resume re-prefill cost about 10% of the GPU).
  - Extension: each free slot goes to the playable game with the highest UCB.
  - UCB = 0.5·(lifetime + last-32k-token weighted-levels-per-token) + 0.0002·sqrt(ln(1+minutes)/actions), with weights L(L+1)/2. Unvisited games go first. Ties go to least service.
  - Slice ends at the first request boundary. The shared pool is about 31k s, with 16 concurrent games.
  - Motivation in the file: local ≈2x board score (12.04 vs 5.02/5.50) under per-game caps.
  - Scheduler evidence: Wang's team reports no isolated gain from the scheduler. rellik13 credits 14.49→22.53 to FP8 KV plus longer history, with no prompt change. His 20-minute save run had a 1,199 s budget and ran 25 warm-ups with 0 extension slices, so UCB was never exercised there (`run/m72_allocation_events.jsonl`).
- **Held back** (`WRITEUP.md:128-132`): a priority scheduler, because late-stage games stagnate and moving time between games "would mostly move stagnant time".

## 7. Wang-Zhongwei duck-harness: four selectable policies
Source: `W/ARC3-Inference/inference/framework/allocation.py` and `solver.py:1195-1500`; docs in `W/ARC3-Inference/docs/allocation.md` and `W/notes/allocation/*.md`.
- **uniform:** same as section 1.
- **rolling_extension:**
  - Every game first gets 5/6 of its per-game cap.
  - Then 300 s quanta go to the highest priority, where priority = 0.5·W/lifetime_tokens + 0.5·recent_weighted_gain/min(tokens, 32k) and W = L(L+1)/2.
  - `exploration_fraction` sets a share of extension time given by least-service fairness instead. The adaptive-reasoning v3 notebook used 0.5.
  - Cap of 8 resumed games per endpoint.
- **ucb:** the warm-up and UCB rule from section 6, with `ucb_c` default 1e-4 and 2e-4 in the FlashNext config. The FlashNext profile uses 25 concurrent games and a 524 min submission budget.
- **stall** (adapted from Matija Ludvig's `spn-rrw` bundle):
  - Games run continuously with a base clock of 5/6 of the cap.
  - The first cleared level earns the full guarantee.
  - Extra 300 s quanta are granted only if a level was cleared within `stall_timeout` (5,400 s), `min_levels` ≥ 1, and a full guarantee remains reserved for every queued game.
  - Time granted never shrinks.
- **Score evidence:**
  - Wang's notes say "No model score improvement was measured".
  - adaptive-reasoning-ucb v5 scored 6.18 public, but its submission ended at about 6 h from a disk-full bug (`W/notes/serving/ucb-v05-disk-and-retry-loop.md`).
  - v04 (rolling) ran the full 9 h.
  - Your own `arms.md:87,185` records UCB at 43.84 vs control 46.25 on the Franzen stack, and "UCB failed twice".

## 8. Other mentions (design only, or opaque)
- `C/solve-arc-agi-3/docs/PLAN.md:170` plans a global deadline scheduler. It sets per-environment allowances from remaining wall time, active concurrency and unfinished games, with initial guards of 12 turns and 4,096 tokens per environment. Degradation order: shorten output, stop code promotion, ledger-only, deterministic fallback, stop. No code or score exists for it.
- `C/kaggle/arc-agi-3-milestone-2-submission` (richardcsaky "arena hero"): "fixed winning … scheduler settings", 110 cyclic replicas. The logic sits in an unavailable tarball, so it is unknown.
- arcagibook.com field manual: recommends a bounded inference queue with per-game fairness so one hard game cannot monopolize the model. No formula.

## What is new relative to your tested list
- **Not yet tested:**
  - Action-quantum permits with a 1/a² rank and a hard cliff throttle (6, M43).
  - Retire-and-guarantee stall clocks (7).
  - Lottery draws per turn with an annealed exponent (5).
  - A board-novelty "alive" signal as a stall detector, as an alternative to a pure action/token stall count. It needs only the harness's own board history.
- **Already covered by you:** UCB (Wang/rellik form; your 43.84 vs 46.25 used C=0.1, which is not Wang's 1e-4/2e-4 form) and round-robin.
- **Not run in a clean A/B anywhere:** none of these has a clean A/B score against a Franzen baseline except D' (31.54, noise-level) and lottery version 6 vs its slope variant (5.68 vs 4.00).
