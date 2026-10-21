# m1-vs-m2-trajectory: verify

Workflow run `wf_903996f8-246`, phase Verify. Research what changed from ARC-AGI-3 Milestone 1 (Tufa Duck) to Milestone 2 solutions, then verify the claims

---

# ARC-AGI-3 MS1 (Duck) vs MS2 (Franzen and other placers): learning trajectory (fact-checked)

**Source keys.** Paths are under `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/`.
- [WU] is Tufa's Kaggle write-up, d/717133, opened in a browser (no line numbers). [BLOG] is tufalabs.ai/research/duck-harness/. [APB] is arcprize.org/blog/arc-prize-2026-milestone-1.
- [ARC-X1] is x.com/arcprize/status/2074557069215313931, the MS1 winners post, 2026-07-07 18:12 UTC (date decoded from the tweet ID). [ARC-X2] is x.com/arcprize/status/2105737436450201734, the MS2 winners post, 2026-10-01 19:11 UTC.
- d/<id> means kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/<id>.
- [DUCK] is `competitors/duck-harness/ARC3-Inference/` @7652836, dated 2026-07-01 16:59 +0200. That is the GitHub release one day after the 06-30 Kaggle post.
- [FW] is `daniel-franzen-sol/WRITEUP.md`. [FC] is `daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md`, a user-fork document with post-Franzen additions.
- [FT] is `daniel-franzen-sol/kaggle/template.ipynb` cell 4 @af3b23e. That commit is the user's own 2026-10-04 copy of Franzen's notebook, and its values match [CMP]:42-43.
- [CMP] is `competitors/tong-m2-comparison/kaggle/COMPARISON.md`, the source of d/744792. [RW] is `competitors/rellik13-milestone2/WRITEUP.md`.
- [LHS] is kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2. [LH] is `competitors/tong-m2-comparison/kaggle/lordhansolo/`.
- [LB] is a Kaggle leaderboard CSV I pulled at 2026-10-05T01:38 UTC (2026-10-04 18:38 PDT).
- **Global caveat:** no placer ran controlled ablations ([FW]:293, [LHS], [RW]:136-138). "X helped" is the author's claim unless a number is given.
- **Variance is large.**
  - Duck scored 0.77 to 1.21 on Kaggle for the same submission, and an initial 1.30 was retracted [WU].
  - Hundreds of copies of Franzen's notebook scored mean 25.77, SD 3.93, median 26.19, in a range of roughly 20 to 34 (d/745062). The poster assumes the copies were unchanged.
  - shiiin9's README reports Franzen's unchanged code at 27.62 to 31.47 across their runs (`competitors/kaggle/affectify-arc-dfranzen-form-d1/*.ipynb`, cell 0).
  - Small deltas are not trustworthy.

## 1. Timeline and score table
- **Competition dates** (arcprize.org/competitions/2026, fetched):
  - Start 2026-03-25.
  - MS1 deadline 2026-06-30.
  - MS2 deadline 2026-09-30.
  - Final submissions 2026-11-02.
  - Results 2026-12-04.
- **2026-06-30 21:10 UTC:** Tufa posts "Code for milestone 1st place open sourced" (d/716696).
- **2026-07-01:**
  - The write-up goes up at 14:47 UTC (d/717133).
  - Tufa's tweet reports 1.21% (x.com/tufalabs/status/2072336849465417747).
- **2026-07-07:** ARC Prize announces the MS1 winners [ARC-X1].
- **2026-08-11:** rellik13 starts at 0.99 ([RW]:26-35).
- **2026-08-21:** a community user (not Tufa) posts d/736578.
  - They ran the Duck harness with Qwen 3.8 and got about 2.1 locally on the 25 public games and about 1.4 on the LB.
  - They note a public notebook showing 2.2.
- **2026-08-26:** Qwen3.8-Flash-Next is released (gigazine.net/gsc_news/en/20260827-qwen3-8-flash-next/, third-party).
- **2026-09-02:** rellik13 reaches 3.81 ([RW]:28-29).
- **2026-09-06:** Franzen is briefly #1 on the LB, then passed by Tufa ([FW]:29).
- **2026-09-23:** Tufa announces it will skip the MS2 prize (d/742801).
- **2026-09-24:** gedouluhui posts d/743060, a Duck + Qwen3.8-27B write-up.
- **2026-09-30:**
  - MS2 deadline.
  - Franzen's last scored LB submission is 08:58 UTC [LB].
  - His last repo commits are 21:27:30 to 21:35:18 +0200 (git log 7652836..10882e3).
- **2026-10-01 19:11 UTC:** ARC Prize announces the MS2 winners [ARC-X2].

| Stage | Entry | Score | Model / serving | Source |
|---|---|---|---|---|
| MS1 #1 | Tufa "Duck" | 1.21 (1.30 retracted; same submission as low as 0.77) | Qwen3.6-27B-FP8, vLLM | [WU]; [ARC-X1] |
| MS1 #2 | Reki | 0.867 | Gemma-4-31B, "pick a JSON action" | [ARC-X1]; [APB] |
| MS1 #3 | forge (Md Boktiar Mahbub Murad) | 0.864 (d/716719 title says 0.86) | Gemma-4-31B | [ARC-X1]; [APB]; d/716719 |
| Post-MS1, not Tufa's runs | Duck + Qwen3.8 | 1.12 (model swap only), about 1.4, 1.43 (with 2 added patches) | Qwen3.8-27B-FP8, vLLM | d/743060 §1, §7; d/736578 |
| MS2 #1 | Franzen | 27.89 | Flash-Next W4A16 + MTP draft, SGLang 2.5.3 | [FW]:5; [CMP]:12 |
| MS2 #2 | lordhansolo | 23.84 | Flash-Next mixed NVFP4+FP8, patched vLLM | [CMP]:12,17 |
| MS2 #3 | "Lohit Siriki" (team rellik13, Kaggle user sirikilohit) | 22.53 | Flash-Next W4A16, SGLang 2.5.0 | [CMP]:12; [RW]; [LB] |

- **Post-MS1 row.** d/743060 (gedouluhui's own runs) gives 1.12 for Qwen3.8-27B with the harness otherwise unmodified. Its 1.43 adds perception hints and a prompt patch, so it is not "same harness". The "about 1.4" is the d/736578 poster's number.
- **Derived arithmetic.** 27.89 / 1.21 is about 23x. Both are public-LB percentages per the ARC tweets. The notes do not say whether the public LB game subset was the same for both milestones.
- **Franzen's own score vs his copies.** 27.89 is about 0.5 SD above the copies' mean of 25.77.
- **Tufa's own Duck run** on the 25 public games was 1.6002 +/- 0.4475, 20 tries per game, Qwen3.6-27B-FP8 [WU][BLOG].
  - It matches `competitors/duck-harness/example-run/score.json` (1.600203...).
  - That run was on 2x B200 under Slurm at 45 min per game (`example-run/evaluation.json` `runtime_budget`). It was not the Kaggle RTX Pro 6000 setup.
- **Fork status.**
  - Franzen is built on the Duck harness ([FW]:7).
  - lordhansolo is "built on the Tufa Labs duck harness" ([LHS]), a fork of ARC3-Inference ([CMP]:168).
  - rellik13 uses Duck plus Wang's design ([RW]:5).
- **Prizes.**
  - MS1 total is $37.5K [APB]. The split was $25K / $7.5K / $5K for Tufa / Reki / forge [ARC-X1].
  - MS2 was $25K / $7.5K / $5K for Franzen / lordhansolo / Lohit [ARC-X2].
  - The competition-page schedule ($25K / $10K / $2.5K) is also repeated by AlphaSignal for MS1. The ARC tweets contradict that for MS1 as well as MS2.
  - postcutoff.com (secondary) says the MS2 split differs from the page and gives no reason.
- **Tufa skipped MS2** because it feared "the best of 4000 copies" (d/742801). It says it still plans to release its final solution when the competition ends.
- **Leaderboard now** [LB]:
  - Tufa Labs leads at 55.89 (154 submissions, last 2026-10-03 07:17 UTC).
  - Franzen's 27.89 is rank 333 of 3,720 teams.
  - Lord Han Solo is now 28.32 (rank 288).
  - rellik13 is still 22.53 (rank 684).
  - shiiin9's team AFF AI CLUB is 31.54 (rank 56).
  - Which of the teams above Franzen are copies of his notebook is not established.

## 2. The MS1 baseline (Duck as shipped)
- **Model and serving:** Qwen3.6-27B-FP8 on one vLLM server, TP=1, `--max-model-len 65536`, prefix caching on ([DUCK] `inference/framework/kaggle.py:10-15` for the constants, `:323-335` for the launch args). Sampling is thinking on, T=0.6, top_p 0.95, top_k 20 (`configs/inference.json`).
- **Loop:**
  - Each assistant message carries a `python` tool call ([WU]).
  - The analyzer loops reply to python call with no step cap (`tool_steps` 0 = unlimited in `configs/inference.json`; `tool_agent.py:932,1783`).
  - It stops when an `action(...)` has run or a time yield fires (`tool_agent.py:1777`).
- **REPL:**
  - Fresh state per call, 30 s timeout (`tool_agent.py:933`).
  - Output capped at 1024 tokens (`configs/inference.json`), which [WU] states as 4096 characters (`tool_agent.py:939` is tokens x 4 chars).
  - 14-module import allowlist (`python_tool_sandbox.py:42-57`, enforced at `:245-249`).
- **Context:**
  - The budget is about 31.7k tokens: 32,768 window, minus 512 reply reserve, minus 512 safety margin (`tool_agent.py:937,945-948`).
  - The token estimate is len/3 of the JSON dump (`:462-467`).
  - History is capped at the last 30 assistant turns (`:151`).
  - [WU] says it keeps 32k of input and caps context at 64k, and that prefix caching is "not optimally" used.
- **Memory:**
  - 7 labeled text blocks: World model, Goal model, Action model, Recent findings, Open questions, Plan, Cross-level notes.
  - The first six are wiped on level change or game over; Cross-level notes survives (`tool_agent.py:263-296,1113-1126`).
  - rellik13 measured Cross-level notes as filled in 0 of 989 and 0 of 1,189 level-2+ prompts in its test runs (`rellik13-milestone2/cells/42_...py:3-8`).
- **Perception:**
  - One 4x-upscaled PNG of the current frame, plus an ASCII grid and a segmentation listing (`vision_context.py:52-82`, `prompts.py:41-44`).
  - 4x comes from `configs/inference.json` (`multimodal.upscale: 4`) and `kaggle.py:120`. That is 256 px, which is about 66 image tokens. The Makefile's 16 (`Makefile:158`) is only a fallback.
  - [WU] says Tufa tried injecting more frames or video but small models struggled with it.
  - The shipped harness shows no animation. [WU] names sb26 and tn36 as games where animation feedback "can be crucial".
- **Actions:**
  - UP/DOWN/LEFT/RIGHT/SPACE/MOUSE/RESET.
  - UNDO is withheld because the model "undoes big batches of actions" ([WU]; `action_names.py:8-15`).
- **Prompt:**
  - It tells the model not to get distracted by the timer or remaining-steps bar, nor treat it as core gameplay without concrete evidence, and not to click bar segments (`prompts.py:28-29`).
  - [WU] says the model is told minimizing the bar is not the goal.
  - [WU] admits heavy prompt tuning: no bar-as-goal, no hallucinated sprites, no dumping whole boards.
- **Scheduling:**
  - None. Games run in list order through a semaphore (`solver.py:896-926`).
  - A turn yields after 60 s, time-based (`kaggle.py:114`, `Makefile:148`).
- **Tufa's own limits** [WU]:
  - Not ablated.
  - Context management could use compaction or curated memory.
  - Perception could improve.
  - Variance is huge.
  - Hand-crafted tools "did not help" ([WU]). [APB] paraphrases this more strongly ("actually hurt"). The stated drivers were better base models and multimodality.

## 3. What changed, grouped by driver

### A. More useful tokens per hardware budget (same 1x RTX Pro 6000, 9 h, 110 games [WU]; [FW]:49)
- **Model:** Qwen3.6-27B, then 3.8-27B, then Flash-Next. Franzen says each switch was "a substantial improvement"; per-switch deltas are unknown ([FW]:15,29). rellik13's first Flash-Next score went 0.99 to 3.81 ([RW]:28-29).
- **Quant and speculation:**
  - Franzen runs W4A16 AutoRound plus an MTP draft with 3 steps, and keeps the PLE table in host RAM ([FW]:31-33,59).
  - lordhansolo runs NVFP4 with a built-in MTP head ([CMP]:18-19).
- **Server patches:**
  - Franzen on SGLang 2.5.3 with 5 patches ([FW]:35-41).
  - lordhansolo on vLLM e975732 with a 32-file overlay ([CMP]:148,175). See section 7 for a provenance caveat on which runtime scored 23.84.
  - rellik13 on SGLang 2.5.0 with 4 boot patches ([CMP]:183-187).
- **FP8 KV cache** is used by all three ([CMP]:94).
- **rellik13's last jump,** 14.49 to 22.53 (+8.04, derived):
  - It was FP8 KV, longer history and more context, with "changed no prompts" ([RW]:8,96-101).
  - Games at once also went 15 to 16.
  - The final 20-minute Save & Run "couldn't show the history effect" ([RW]:122-126).
  - The attribution is his own claim from a single submission.
- **Cache reuse:** 93.43% of prompt tokens reused over 5,132 requests, from an earlier demo run, not the final config ([FW]:101-112).
- **Yield:** Duck yields after 60 s. Franzen yields after 2,048 generated tokens, so it no longer depends on throughput ([FW]:299; [FT]).
- **Thesis:** lordhansolo says "the score follows the number of turns each game gets" ([LHS]). Peak decode tok/s was 946 / 1,135 / 1,159 for 10 / 14 / 16 slots ([CMP]:206,231-232).

### B. Remember more, cache-friendly
- **Context size:**
  - Duck keeps about 32k of input and caps at 64k.
  - Franzen uses a 131,072 harness window and a 139,264 server limit ([FW]:51,53), and says it "helped especially with the more difficult games and levels" ([FW]:67, qualitative).
  - lordhansolo uses a 147,072 server and trims about 126k down to 81,536 ([LH] `dataset-taaf-kaggle-source/src/ARC3-Inference/configs/inference.json:43,51,71`).
  - rellik13 went from 49,152 to 69,632 ([RW]:96-101).
  - Counter-evidence: d/743060 §5.1 reports Duck 32k to 64k on vLLM scoring 1.20 vs 1.43, with throughput falling from about 283 to about 195 tok/s. It is a single run, and the same post puts run noise at about 0.8-0.9. This matches [FW]:67 that more context only helps if the serving stack reuses it.
- **History cap:**
  - Duck 30 assistant turns.
  - Franzen raised it to 150 with a 30-turn drain ([FT]; the write-up never mentions it).
  - lordhansolo's cap is off whenever a target context is set ([CMP]:43,59-60).
  - rellik13 lifted it (`cells/35_...py:22-23`).
- **Block trimming for the prefix cache:**
  - Franzen's trim triggers at the budget of 118,272 tokens (131,072 - 12,288 - 512).
  - The configured drain is 58 Ki = 59,392, clamped to half the budget, so about 59,136 is dropped and about 59,136 (57.75 Ki) is retained (`tool_agent.py` @10882e3 `:3506-3509,6469-6471`; [FW]:54,97).
  - lordhansolo trims about 126k down to 81,536.
  - rellik13 trims 57,344 down to 45,056 (`cells/35_...py:30-31`).
- **Token estimate:** Franzen calibrates a text ratio against backend usage and computes image tokens from PNG-header dimensions ([FW]:73-75; `tool_agent.py:2980-3022`). Duck uses len/3 and counts base64 as text (`tool_agent.py:462-467`).
- **Thinking retention** was kept; removing it "leads to a large regression" ([FW]:69).

### C. Allocate compute across games
- **Duck:** list order, one per-game time cap, no priority. The `make kaggle-duck` target sets 132 min per game and 28 concurrent (`Makefile:400-405`); see section 7.
- **Franzen:** all games start but only 10 streams are admitted ([FW]:123-126). Priority is P=(A+B)·C ([FW]:128-142):
  - A is level gain with an efficiency penalty (`human_actions=25`, `priority_scheduler.py` @10882e3 `:82`).
  - B is a lookup by levels remaining, 8/7/5/0.
  - C is a Gaussian decay in actions (scale 115) and tokens (scale 62,000), weighted 0.25 and 0.75 ([FW]:139), fit on 725 level attempts (557 done, 168 not). A Gaussian beat an exponential ([FW]:157).
  - B fades over the last 20% of runtime ([FW]:164). A game keeps its slot until context eviction forces a re-prefill ([FW]:174).
- **rellik13:** UCB with 2,400 s slices on a pool of about 31k s, 16 concurrent ([CMP]:73-74; d/744792). A basic priority scheduler existed but was held back ([RW]:128-132).
- **lordhansolo:** 14 games live, a 3,918 s cap per game sized as 8 waves of 14 over 110 games, and a freed slot goes to the next game at once (d/744792 "Budget per game"; [LHS] "in waves"). No priority.
- **Post-submission, not Franzen's work:**
  - shiiin9 changed only the priority formula and scored 31.54 in one submission (shiiin9 README; [FC]:350).
  - That README also reports Franzen's unchanged code at 27.62 to 31.47 across their runs. 31.54 is therefore indistinguishable from the unchanged notebook's best, and this is not evidence that the scheduler is a cheap lever.

### D. Better perception and truthful feedback
- **Animation:**
  - Franzen exposes intermediate frames and a timeline as REPL variables, the "strongest benefit" of the perception changes, with tn36 essentially unsolvable without it ([FW]:181-185).
  - lordhansolo caps `animation_frames` at 16 ([LH] `.../agent/animation.py:37`).
  - rellik13 added M76 mid-action frames.
  - Duck's shipped harness had none ([WU] says Tufa tried and dropped it).
- **`frame_diff()`:** object-level moved/rotated/appeared/recolored/resized ([FW]:190-192).
- **Image scale:**
  - Franzen went 4x to 10x, the "biggest score improvement" among his image changes ([FW]:203). He also added diff images and a fatal-frame image ([FW]:199-201).
  - lordhansolo and rellik13 kept Duck's 256 px, 66-token boards ([CMP]:47-48).
- **UNDO exposed** by all three (`lordhansolo/.../agent/action.py:20`; `rellik13-milestone2/cells/40_...py:288`; [FW]:262). Franzen's TL;DR says it "helped substantially in my competition runs" ([FW]:18), but [FW]:266 says it came bundled with action info, level-transfer guidance and persistent functions, so UNDO's own effect is not isolated. Among demo games only su15 clearly used it ([FW]:264).
- **Game-over text:** Franzen wrote a long diagnosis plus "Do NOT re-submit this same series of moves" ([FW]:216-233). rellik13 added an accurate "Level restarted." ([RW]:80-85).
- **Edge bar:**
  - lordhansolo's prompt says it is a per-level action budget ([LH] `prompts.py:58`).
  - rellik13 deleted the "ignore the bar" line ([RW]:80-85; d/744792 comment).
  - Franzen kept Duck's line (his `prompts.py:84`) but added budget-bar semantics to the flag-semantics and game-over prompts ([FW]:225,229; `prompts.py:226`).
- **Guards (Franzen):** batch no-op (ignores a 4-cell border), stale-state, terminal-state ([FW]:282-288). No ablation number.

### E. Persistent, verified artifacts across levels
- **Franzen:** retained functions (`retained_functions.py` per [CMP]:157), scope=game ([FW]:255-257; [FT]). The bundle of undo, action info, level-transfer and persistent functions gave "a large improvement"; no isolated delta ([FW]:266).
- **lordhansolo:** `save_module`; a save that breaks older replayed tests is rejected ([LHS]; [LH] `prompts.py:122`). Also `game_overs` and a per-action plan.
- **rellik13:**
  - M85 pins the winning rule and last 30 actions per cleared level in the system prompt (`cells/42_...py`; newest 3 levels keep actions plus reasoning, older ones keep the rule only).
  - It was added after transcripts showed 60-100+ minutes re-deriving.
  - Dated step 7.14 to 9.96 (+2.82, derived).
- **Duck:** only the text "Cross-level notes" (`tool_agent.py:263-296`). Tufa lists missing curated memory as a limitation [WU].

### F. Removing things that hurt
- Franzen turned the structured world-model sections off (`ARC3_MEMORY_SECTIONS=off` in [FT]), the "largest improvement in that area" ([FW]:295). His explanation, not tested, is that the bigger rolling context already holds the evidence. He also notes the original parsing often missed model updates.
- rellik13: always-on text 6.42 to 4.16, and a level-start advice block 13.40 to 9.98 ([RW]:59). The cause (instructions displacing recent turns at a ~37k cap) is his unproven hypothesis.
- Duck's fixed per-game time cap and fixed game-level concurrency were replaced by the admission gate ([FW]:123-126).

## 4. Converged vs diverged
**Converged (all three top placers):**
- Same model (Flash-Next), FP8 KV, 3-step MTP, one GPU ([CMP]:3-4,19,94).
- The 30-turn history cap was raised (Franzen 150), turned off (lordhansolo) or lifted (rellik13) ([CMP]:43). All trimmed in large blocks to preserve the prefix cache.
- Animation frames exposed, UNDO exposed. A cross-level carry-over mechanism, built differently by each.
- All three tell the model the edge bar can be a budget. Only two (lordhansolo, rellik13) removed or rewrote Duck's "don't get distracted" line.
- Throughput patches to the server stack. Franzen and rellik13 used Claude, ChatGPT or Claude Code; lordhansolo unknown ([FW]:316; [RW]:14).

| | Franzen 27.89 | lordhansolo 23.84 | Lohit 22.53 |
|---|---|---|---|
| Server | SGLang 2.5.3 | vLLM nightly | SGLang 2.5.0 |
| Slots / prompt budget | 10 / ~118k | 14 / ~126k | 16 / 65.5k |
| Trim | to ~59k | to 81.5k | 57k to 45k |
| Board image | 640 px, ~402 tok | 256 px, 66 tok | 256 px, 66 tok |
| Scheduling | priority gate | rolling 14 slots, 3,918 s cap | UCB |
| Prompting | added transfer/game-over text; memory off | rewrite + strategy audit | added text hurt; deleted lines |
| Cross-level memory | retained functions | save_module | pinned solved-level rule |

Source: [CMP]:10-19,36-50,67-76; prompting row from d/744792 and [LHS].
- Tong's inference: "The top score belongs to the build with the fewest server slots and the longest per-slot context" ([CMP]:31-32). Three entries, not a controlled result. lordhansolo's [LHS] says higher concurrency raised total throughput.
- Demo means are not comparable: Franzen 36.56 on 10 games, lordhansolo 5.28 and Lohit 6.89 on 25 ([CMP]:219,238).
- Image resolution diverges: Franzen invested in 10x. rellik13 lists "higher-resolution board images" among things that didn't work ([RW]:147; resolution unspecified).

## 5. Tried and did NOT help (or not shown to help)
- **Franzen** ([FW]:295-305):
  - Summaries.
  - Composite animation images (hurt some games) and auto timeline insertion (no clear gain).
  - Resume-prompt variants.
  - A stronger step-verification hint.
  - Repeat-state, remembered-no-op and death-ledger guards.
  - Pace adjustment and normalized-B lookup.
- **rellik13** ([RW]:140-148, 154-155): advice or summaries, fresh-context restarts, temperature 1.0, a proven/refuted ledger, expert pruning, higher-res images, Swift 1.5, a LoRA trial. Also the 6.42 to 4.16 and 13.40 to 9.98 text regressions ([RW]:59).
- **lordhansolo:** "I cannot confirm that any of these changes improved the score" ([LHS]). A later vLLM runtime v3 had no run that beat the score (`competitors/kaggle/built-on-tufa-labs-duck-harness-milestone-2/*.txt`, cell 0).
- **MS1 Tufa:** hand-crafted tools "did not help" ([WU]); [APB] says "actually hurt".
- **Community (d/743060):** temperature 0.3 (0.95) and an archetype playbook (1.54) were rejected on Duck + Qwen3.8-27B, single runs against a noise floor of about 0.8-0.9.
- **Untested:** Franzen ran out of Kaggle compute before a full demo run with his final config ([FW]:62).

## 6. Generating lessons (my synthesis; the sources give the evidence, not these framings)
1. **Same compute, more thinking per game.** The big jumps came from model, quantization, speculative decoding and FP8 KV, which raise tokens per second. rellik13's last change (+8.04, single submission) and lordhansolo's thesis point the same way. Model and serving cannot be separated, since nobody ablated them.
2. **Raw long context beats curated prose notes, but only with cache-friendly serving.**
   - Franzen turned the notes off.
   - rellik13's added text hurt, yet its pinned solved-level memory helped.
   - d/743060 saw no gain from 64k on a cache-unfriendly vLLM setup.
   - Cost control came from block trimming that keeps the prefix cache valid.
3. **Persist verified artifacts, not summaries.** Summaries and world-model text showed no gain. Retained functions, `save_module` and pinned rules were claimed to help, but only as bundles, so this is an interpretation of confounded claims.
4. **Make feedback truthful and visible.** Animation, frame diffs, UNDO, fatal-frame and game-over messaging, and an accurate edge-bar rule all fix cases where the model could not see something or was told the wrong thing.
5. **Spend compute where marginal progress is.** Franzen's calibrated priority gate and rellik13's UCB reallocate budget across games. The shiiin9 result (31.54) is within the 27.62 to 31.47 range of Franzen's unchanged code, so this lever is unproven. Evidence is the thinnest of the five.
6. **Read scores skeptically.** The same notebook varied 5.02 to 6.42 for rellik13 ([RW]:17-20) and 27.62 to 31.47 for Franzen's code; copies had SD 3.93. Steps smaller than about that range cannot be attributed.

## 7. Uncertainties and conflicts in the notes
- **Duck's Kaggle run shape.**
  - `make kaggle-duck` is 132 min per game, 540 min total, 28 concurrent (`Makefile:400-405`).
  - d/743060 §3 independently says the official Kaggle path uses a 132 min per-game budget and 9 h total.
  - [DUCK] `README.md:254` describes a 16-game default with 16 concurrent, 75 min and a 90 min runtime.
  - The Tufa notebook loads a pickled benchmark (`taaf-duck-harness-kaggle-share.ipynb` cell 10), so the exact shape of the 1.21 run is unverified.
- **Duck's upscale: resolved.** It is 4x (256 px) via `configs/inference.json` and `kaggle.py:120`. Makefile's 16 is a fallback.
- **Duck's history cap in the 1.21 run is unknown.**
  - `example-run/git_status.txt:4` shows commit dd50b55 on branch `user/periodic-chat-clearing`, "Make history turn cap configurable, default to unlimited", flagged DIRTY.
  - The run is labelled "0-history-turns", and it is a B200 Slurm run, not the Kaggle submission.
  - The shipped snapshot hardcodes 30 (`tool_agent.py:151`), and d/743060 §3 says "history 30 turns" for the official Kaggle path.
- **Franzen's trim target: resolved.** 57 Ki, 58 Ki and ~59k are the same quantity (59,136 tokens retained; configured drain 59,392; see 3B).
- **Duck 1.03** appears only in AlphaSignal, which also lists wrong MS1 prize splits. [ARC-X1] and Tufa's tweet give 1.21.
- **Tufa 45.33% on Sep 29** comes from postcutoff.com. Kaggle comments in d/742801 are consistent with it (27.29 about 8 days before 2026-10-04, then 45.33 about 6 days before) but do not name Tufa. The LB pull's "2026-10-05" stamp is UTC. The local date was 2026-10-04 PDT, so there is no discrepancy.
- **lordhansolo provenance.**
  - [CMP] documents `vllm-main-e975732-arc3` (runtime v3) and `taaf-kaggle-source` v305 (`kernel-metadata.json`; [CMP]:243-246).
  - lordhansolo's own notebook note says the MS2 scoring version is v293, using source v297 and `arc3-vllm-e975732-gdn-recoverssm` v2.
  - It calls v3 "faster" and says no run with it has beaten the earlier score.
  - The 32-file overlay, the CUDA-graph 44/52 patch and the throughput numbers may therefore not be the 23.84 config.
- **Not found:** per-feature and per-model-switch deltas, lordhansolo's score history, and whether Tufa's `taaf-kaggle-source-share` dataset matches the 7652836 snapshot.
- **Tension, not resolved:** Tufa says hand-crafted tools did not help, yet Franzen added `frame_diff` and retained functions. Possibly "perception helpers" differ from "hand-crafted solvers"; the notes do not say.

## Corrections
1. **MS1 #2 Reki "unknown" became 0.867, and forge 0.86 became 0.864.** Source: [ARC-X1], fetched.
2. **MS1 prize split "unknown" became $25K / $7.5K / $5K.** Source: [ARC-X1]. MS2 uses the same split [ARC-X2]. The $25K / $10K / $2.5K schedule is the page's or AlphaSignal's, not what was announced.
3. **"Hand-crafted tools did not help [WU][APB]" was split.** [WU] says "did not help"; [APB] says "actually hurt".
4. **"Duck + Qwen3.8-27B about 1.4, same harness" was rewritten.** d/736578 (2026-08-21) is a community user's run, not Tufa's. The 1.43 in d/743060 (2026-09-24) includes perception hints and a prompt patch; its model-swap-only score is 1.12.
5. **Duck variance "0.77 to 1.30" now notes that 1.30 was retracted.** Source: [WU].
6. **Franzen copies' stats** now include the median (26.19) and the caveat that the copies being unchanged is the poster's assumption. Source: d/745062.
7. **"Mostly passed by copies" was removed** for Franzen's rank 333 of 3,720. The leaderboard shows no copy provenance. Rank, team count and Tufa's 55.89 are confirmed from [LB].
8. **The shiiin9 31.5 claim was reversed.** The README says Franzen's unchanged code scored 27.62 to 31.47, so 31.54 is not distinguishable from it. The cite moved from [FC]:350 (user-fork doc) to the notebook's cell 0.
9. **The Duck loop description was fixed.** It is not one python call per turn; steps are unlimited (`tool_steps` 0) until an action runs or the yield fires. Source: `tool_agent.py:932,1777-1783`; `configs/inference.json`.
10. **The Duck bar prompt was reworded.** It says "do not get distracted" and do not click bar segments, not "ignore". Source: `prompts.py:28-29`; [WU].
11. **Franzen did not reverse Duck's bar line.** He kept it (his `prompts.py:84`) and added budget-bar text ([FW]:225,229). So all three flag the bar as a budget, but only two removed or rewrote the line. Source: d/744792 (rellik13 comment); [RW]:80-85; [LH] `prompts.py:58`.
12. **"All three lifted or dropped the 30-turn cap" was fixed.** Franzen raised it to 150 with a 30-turn drain. Source: [FT]; [CMP]:43.
13. **Franzen's trim numbers were reconciled.** The budget is 118,272 and the drain is clamped to half of it, giving about 59,136 retained (57.75 Ki). Source: `tool_agent.py` @10882e3 `:3506-3509,6469-6471`; [FT].
14. **Duck's upscale is resolved at 4x** (`configs/inference.json`, `kaggle.py:120`; d/743060 says 256 px). lordhansolo and rellik13 kept it.
15. **Franzen's UNDO claim was completed.** The TL;DR says "helped substantially in my competition runs" ([FW]:18); the su15 note is only about demo games.
16. **"All three say they are forks" was re-cited.** d/744792 is Tong's statement. Citations now go to [FW]:7, [LHS] and [CMP]:168, and [RW]:5.
17. **lordhansolo's scheduling was fixed.** It is a rolling slot pool with a 3,918 s cap, "sized as 8 waves". Source: d/744792; [LHS].
18. **Citation fixes:**
    - FP8 KV is [CMP]:94, not :3-4.
    - Peak decode is [CMP]:206,231-232, not :207.
    - NVFP4 + MTP head is [CMP]:18-19, not `PATCH_README.md:9-28`.
    - The yield default is `kaggle.py:114`, not :111.
    - The vLLM args are at `kaggle.py:323-335`, not :168-198.
    - The "default to unlimited" note is `git_status.txt`, not `git_info.txt`.
    - The sandbox allowlist is `python_tool_sandbox.py:42-57`.
    - The tool timeout is `tool_agent.py:933`.
19. **Timeline items were added or corrected.**
    - The 2026-10-01 tweet was fetched and its time is 19:11 UTC.
    - d/742801 is dated 2026-09-23 and says Tufa will release its final solution when the competition ends.
    - Tufa's open-source post is 06-30 21:10 UTC.
    - The GitHub snapshot is dated 07-01.
    - The "2026-10-05" LB stamp is explained as UTC.
20. **lordhansolo version provenance flagged.** [CMP] uses runtime v3 and source v305; the scoring notebook is v293 with source v297 and runtime v2. Source: `built-on-tufa-labs-duck-harness-milestone-2.txt` cell 0; `kernel-metadata.json`.
21. **Context-length evidence added.** d/743060 §5.1 reports a Duck 64k run scoring below baseline (1.20 vs 1.43) on a cache-unfriendly setup, which qualifies the "long context wins" lesson.
22. **rellik13 caveats added.** The 14.49 to 22.53 attribution rests on one submission, with the history effect untested ([RW]:122-126). Duck's Cross-level notes was empty in his test runs (`cells/42_...py:3-8`). M85 keeps full actions for only the newest 3 levels.
23. **Verbatim quote fixes.** "do NOT replay the fatal sequence" became the actual wording from [FW]:233, and the quote marks around "maintain 32k, cap 64k" were removed.
24. **Tufa's 1.6002 run context added.** It was 2x B200 Slurm at 45 min per game, not the Kaggle config. Source: `example-run/evaluation.json`.
