# m1-vs-m2-trajectory: research:ms2-franzen

Workflow run `wf_903996f8-246`, phase Research. Research what changed from ARC-AGI-3 Milestone 1 (Tufa Duck) to Milestone 2 solutions, then verify the claims

---

## Franzen (MS2 #1) vs Duck (MS1): what changed

**Dates and score**
- MS2 deadline was 30 Sep 2026 (MS1 was 30 Jun 2026) per https://arcprize.org/competitions/2026/arc-agi-3. A third-party snippet (internshala) gives winners' announcement as 4 Dec 2026; I could not confirm that on arcprize.org.
- Franzen's last commits are dated 2026-09-30 21:27-21:35 +0200 (`git log 7652836..10882e3`). The inventory says "Audited on 29 September 2026" (CONFIGURATION.md:5).
- His public-LB score is **27.89** (WRITEUP.md:5). He briefly held #1 on 6 Sep 2026 before Tufa passed him (WRITEUP.md:29).
- Diff size: his work is 9 commits on top of Duck `7652836` (2026-07-01). The cumulative diff, excluding deleted `example-run/`, is ~+8.9k/-0.3k lines across 19 harness/viewer/Makefile files, plus 4 serving files. `tool_agent.py` alone is +5482 (`git diff --stat 7652836 10882e3`).
- `main` also holds the user's later arm/replay commits (2026-10-04); I excluded those.
- **Evidence caveat:** he says results are "practical experiment results rather than a complete controlled ablation study" (WRITEUP.md:293). No per-feature score deltas are reported, only direction words ("substantial", "large", "strongest").
- **Submitted config:** the single best record of what was enabled is `kaggle/template.ipynb` cell 4 at commit af3b23e (Franzen's notebook as the user's arm template). Most citations below point to it.

**Duck baseline (for contrast)**
- Qwen3.6-27B-FP8 on vLLM (Duck notebook cell 6, `DATASET_SOURCES`).
- Default context 65,536 (Makefile:102 @7652836).
- Characters-per-token estimate of `(len+2)//3` (tool_agent.py:462-467 @7652836).
- History capped at 30 assistant turns (tool_agent.py:151 @7652836).
- 16 concurrent games, 75 min per game (Duck ARC3-Inference/README.md:254 @7652836).
- Time-based yield, 60 s (Makefile:148 @7652836).
- No priority scheduler, `frame_diff` or animation code (greps of Duck `tool_agent.py` and sandbox return nothing).

### 1. Model and serving
- Qwen3.6-27B → Qwen3.8-27B → **Qwen3.8-Flash-Next**. "Each switch brought a substantial improvement"; the last got him briefly to #1 (WRITEUP.md:15, 29).
- Final weights are Intel W4A16-AutoRound. A separate Albucino MTP draft model drives speculative decoding with 3 steps. The BF16 PLE table is kept in host RAM (WRITEUP.md:31-33, 59).
- Serving stack is Pennyroyal's SGLang fork v2.5.3 (WRITEUP.md:35), swapped in for vLLM. Patches: low-M BF16 GEMM, a speculative-state memory-budget fix, a Marlin dtype fix, his prefix-cache patch and his checkpoint-prefetch patch (WRITEUP.md:37-41).
- Model resharded into smaller shards plus a background prefetcher, so startup is reliable (WRITEUP.md:43).
- Settings: FP8 KV cache, 60 Mamba slots, memory fraction 0.96, chunked prefill 8,192, temperature 0.7 / top-p 0.95 / top-k 20 (WRITEUP.md:47-60).
- **Caveat:** he ran out of Kaggle compute before a full demo run with the final config (WRITEUP.md:62).

### 2. Context and memory
- **Context 64Ki → 128Ki:** window (116+12)·1024, 12Ki max output (template cell 4). Evidence is qualitative: "helped especially with the more difficult games and levels" (WRITEUP.md:67).
- **Better size estimate:** a calibrated text token ratio plus image tokens from PNG dimensions, instead of base64 counted as text (WRITEUP.md:73-75; CONFIGURATION.md:191-202).
- **Hysteresis trimming:** drain ~58Ki in one block so the prefix stays cacheable (WRITEUP.md:93-97; `ARC3_CONTEXT_DRAIN_TOKENS`=58·1024).
- **Assistant-turn cap 30 → 150** with a 30-turn drain and coalescing (template cell 4). The write-up never mentions this change.
- **Cache evidence (earlier demo run, slightly different config; not the final one):**
  - 93.43% of prompt tokens reused, across 5,132 requests.
  - 1 of 3,248 prompts longer than 62k had under 10k cached.
  - P90 prefill 10.5k tok/s, P90 decode 837 tok/s (WRITEUP.md:101-112).
- **Mamba cache patch:** keep only the final-prefill checkpoint and refresh its LRU position (WRITEUP.md:87-89).
- **Retained Python functions:** a new `retained_functions.py`, `scope=game`, with imports retained (template cell 4; WRITEUP.md:255-257). Evidence is bundled: undo + action info + level-transfer + persistent functions gave "a large improvement" on the competition set, with no isolated delta (WRITEUP.md:266).

### 3. Compute allocation and scheduling
- **Admission gate:** all games start at once, but only 10 streams are admitted; the rest are frozen (WRITEUP.md:123-126; `ARC3_MAX_ACTIVE_STREAMS`=10). This replaces Duck's 16-concurrent / fixed-time-per-game setup.
- **Priority** `P=(A+B)·C` (WRITEUP.md:128-142):
  - A is the level gain with an efficiency penalty (human-action constant 25).
  - B is a lookup by levels remaining: 8/7/5/0.
  - C is a Gaussian-shaped decay in actions (scale 115) and tokens (scale 62,000).
  - A is normalized by level count; B is not.
- **C calibration:** 725 observed level attempts (557 completed, 168 unfinished) from four earlier runs. A Gaussian fit better than an exponential (WRITEUP.md:157).
- **Tail fade:** B fades linearly over the last 20% of runtime (WRITEUP.md:164; `FADE_FRACTION`=0.2).
- **Slot retention:** a game keeps its slot across tool calls and only re-queues when context eviction forces a re-prefill (WRITEUP.md:174).
- The Duck-era 2/4 and constant-tail defaults still exist in code (CONFIGURATION.md:311-314).

### 4. Perception
- **Animations:** intermediate frames and a timeline exposed as REPL variables (not just printed). The hint fires only when "transient pixels" exist (WRITEUP.md:181-185). Called the "strongest benefit" of the perception changes (WRITEUP.md:181); tn36 is cited as essentially unsolvable without it.
- **`frame_diff()`:** object-level changes (moved, rotated, appeared, disappeared, recolored, resized) from 4-connected components (WRITEUP.md:190-192; `frame_diff.py` is new).
- **Images:** upscale 4× → **10×**, described as the "biggest score improvement" among image changes (WRITEUP.md:203). New diff images and a fatal-frame image after game over (WRITEUP.md:199-201).
- **Unresolved discrepancy:** Duck's source default is 16 (vision_context.py:42-49 and Makefile:158 @7652836; CONFIGURATION.md:96). The "4×" in WRITEUP.md:203 comes from Duck's notebook setting, which I did not find.

### 5. Tools and prompts
- **UNDO:** Duck advertised `ACTION7` but could not execute it (action_names.py @7652836, Franzen's WRITEUP.md:262). He exposes it as `UNDO` with `EXPOSE_UNDO=on`. It was only clearly useful on su15 among demo games (WRITEUP.md:264).
- **Game-over prompt:** replaced Duck's one-line "The game is over." with a long diagnosis and "do NOT replay the fatal sequence" text (WRITEUP.md:216-233).
- **Level-up / level-transfer prompt:** carry mechanics forward, check new elements, and treat the new frame as a fresh start (WRITEUP.md:238-250).
- **Opener and results:** clearer repeated-opener wording (WRITEUP.md:210-212). Per-action results kept, plus `last_action_call_result` (replacing `last_action_result`). Tool output raised from 1024 to 3072 tokens, with middle truncation and action echo (WRITEUP.md:271-275; template cell 4; Duck tool_agent.py:142).
- **Action-info prompt, frame-diff hint, deduped multicall line** are all on (template cell 4).
- **Yield:** time-based 60 s → 2,048 generated tokens, so it no longer depends on throughput (WRITEUP.md:299).

### 6. Guards
- Three are enabled (WRITEUP.md:282-288):
  - **Batch no-op guard:** ignores a 4-cell border so a shrinking HUD bar does not count as progress.
  - **Stale-state guard:** blocks a later action call after an ineffective one. The first ineffective action returns normally.
  - **Terminal-state guard:** blocks actions after a game over or level completion.
- The stale-state guard starts from level 2 (`ARC3_GUARDS_FROM_LEVEL=2`). Guard exceptions derive from `BaseException`, so a model's `except Exception` cannot swallow them (CONFIGURATION.md:142).
- No ablation number is given.

### 7. Removed or turned off
- **Structured world-model memory is off** (`ARC3_MEMORY_SECTIONS=off`). He calls this the largest improvement in that area, because the larger rolling context already holds the evidence (WRITEUP.md:295).
- Summaries (inserted or replacing history): no clear gain, so default 0 / off (WRITEUP.md:297; CONFIGURATION.md:234).
- Composite animation images and automatic timeline insertion: no gain, and hurt some games (WRITEUP.md:301).
- Resume-prompt variants, stronger step-verification hint, repeat-state / remembered-no-op / death-ledger guards: no clear gain (WRITEUP.md:299, 303).
- Pace adjustment and a normalized-B empirical lookup: no clear gain over the simple lookup (WRITEUP.md:305).
- Duck's fixed per-game time budget and 16-way game concurrency (see section 3).
- Thinking retention is *kept*; removing it "leads to a large regression" (WRITEUP.md:69).

### Post-submission context (not Franzen's work)
- A third party (shiiin9) changed only his slot-priority formula ("D′") and reports 31.5 on the LB from one submission (CONFIGURATION.md:350). The scheduler, not perception, was the cheap lever. This text sits in the user's own `main` and was not in Franzen's repo as of 10882e3.

**Unknown:**
- Duck's exact Kaggle score.
- Per-feature ablation deltas.
- Final-config demo-run results.
- The 4× vs 16 upscale provenance.

**Sources:** `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/` (`WRITEUP.md`, `README.md`, `ARC3-Inference/CONFIGURATION.md`, `kaggle/template.ipynb`, and git `7652836`..`10882e3`).
