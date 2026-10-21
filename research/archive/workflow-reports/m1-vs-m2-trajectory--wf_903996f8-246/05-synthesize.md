# m1-vs-m2-trajectory: synthesize

Workflow run `wf_903996f8-246`, phase Synthesize. Research what changed from ARC-AGI-3 Milestone 1 (Tufa Duck) to Milestone 2 solutions, then verify the claims

---

# ARC-AGI-3 MS1 (Duck) vs MS2 (Franzen and other placers): learning trajectory

**Source keys.** Paths are under `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/`.
- [WU] is Tufa's Kaggle write-up, d/717133 (no line numbers). [BLOG] is tufalabs.ai/research/duck-harness/. [APB] is arcprize.org/blog/arc-prize-2026-milestone-1.
- d/<id> means kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/<id>.
- [DUCK] is `competitors/duck-harness/ARC3-Inference/` @7652836.
- [FW] is `daniel-franzen-sol/WRITEUP.md`. [FC] is `daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md`. [FT] is `daniel-franzen-sol/kaggle/template.ipynb` cell 4 @af3b23e.
- [CMP] is `competitors/tong-m2-comparison/kaggle/COMPARISON.md`. [RW] is `competitors/rellik13-milestone2/WRITEUP.md`.
- [LHS] is kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2. [LH] is `competitors/tong-m2-comparison/kaggle/lordhansolo/`.
- **Global caveat:** no team ran controlled ablations ([FW]:293, [LHS], [RW]). "X helped" is the author's claim unless a number is given.
- Same-submission variance is large. Duck scored 0.77 to 1.30 [WU]. Unchanged copies of Franzen's notebook had mean 25.77, SD 3.93 (d/745062). Small deltas are not trustworthy.

## 1. Timeline and score table
- 2026-03-25 competition start. MS1 deadline 2026-06-30, MS2 2026-09-30, final submissions 2026-11-02, results 2026-12-04 (arcprize.org/competitions/2026).
- 2026-06-30 Tufa open-sources the Duck (d/716696). 2026-07-01 write-up (d/717133).
- 2026-08-11 rellik13 starts at 0.99 ([RW]:26-35). 2026-08-21 Duck with Qwen3.8-27B is about 1.4 (d/736578, d/743060 at 1.43).
- 2026-08-26 Qwen3.8-Flash-Next is released (gigazine.net/gsc_news/en/20260827-qwen3-8-flash-next/, third-party). 2026-09-02 rellik13 reaches 3.81 ([RW]:28-29).
- 2026-09-06 Franzen is briefly #1 on the LB, then passed by Tufa ([FW]:29). 2026-09-30 MS2 deadline; Franzen's last commits are 21:27-21:35 +0200 (git log 7652836..10882e3).
- 2026-10-01 winners tweet (x.com/arcprize/status/2105737436450201734, not fetched by the research).

| Stage | Entry | Score | Model / serving | Source |
|---|---|---|---|---|
| MS1 #1 | Tufa "Duck" | 1.21 (1.30 retracted; same submission as low as 0.77) | Qwen3.6-27B-FP8, vLLM | [WU] |
| MS1 #2 | Reki | unknown | Gemma-4-31B, "pick a JSON action" | [APB] |
| MS1 #3 | forge | 0.86 | Gemma-4-31B | d/716719 |
| Post-MS1 | Duck + Qwen3.8-27B | about 1.4 | same harness, newer model | d/736578, d/743060 |
| MS2 #1 | Franzen | 27.89 | Flash-Next W4A16 + MTP draft, SGLang 2.5.3 | [FW]:5; [CMP]:12 |
| MS2 #2 | lordhansolo | 23.84 | Flash-Next mixed NVFP4+FP8, patched vLLM | [CMP]:12,17 |
| MS2 #3 | LohitSiriki (= rellik13) | 22.53 | Flash-Next W4A16, SGLang 2.5.0 | [CMP]:12; [RW] |

- Derived arithmetic: 27.89 / 1.21 is about 23x. This assumes both milestones use the same LB metric, which the notes do not state.
- Tufa's own Duck run on the 25 public games was 1.6002 +/- 0.4475 [WU][BLOG].
- Franzen, lordhansolo and rellik13 all say their entries are forks of the Duck harness ([FW]:7; d/744792).
- Prizes: MS1 pool $37.5K, per-place split unknown [APB]. MS2 paid $25K / $7.5K / $5K per the ARC tweet. postcutoff.com says this differs from the page's $25K / $10K / $2.5K and gives no reason.
- Tufa skipped MS2, fearing "the best of 4000 copies" (d/742801), and leads the public LB at 55.89 (Kaggle CLI, pulled 2026-10-05 per the timeline notes). Franzen's 27.89 is now rank 333 of 3,720 teams, mostly passed by copies.

## 2. The MS1 baseline (Duck as shipped)
- **Model and serving:** Qwen3.6-27B-FP8 on one vLLM server, TP=1, `--max-model-len 65536`, prefix caching on ([DUCK] `framework/kaggle.py:10-15,168-198`). Sampling is thinking on, T=0.6, top_p 0.95, top_k 20 (`configs/inference.json`).
- **Loop:** each turn the model makes one `python` tool call. The turn ends once an `action(...)` has run ([DUCK] `tool_agent.py:1706+`, `solver.py:263-338`).
- **REPL:** fresh subprocess per call, 30 s timeout, output capped at 1024 tokens, 14-module import allowlist (`tool_agent.py:939`, `python_tool_sandbox.py:245-264`).
- **Context:** the budget is about 31.7k tokens (32768 minus 512 minus 512, `tool_agent.py:944-948`). Estimate is len/3 (`:462-467`). History is capped at the last 30 assistant turns (`:151`). [WU] says "maintain 32k, cap 64k" and that prefix caching is used "not optimally".
- **Memory:** 7 labeled text blocks (World model, Goal model, Action model, Recent findings, Open questions, Plan, Cross-level notes). All except Cross-level notes are wiped on level change or game over (`tool_agent.py:263-296,1113-1126`).
- **Perception:** one 4x-upscaled PNG of the current frame (per [WU]), plus an ASCII grid and a segmentation listing (`vision_context.py:52-82`, `prompts.py:41-44`). There is no animation or frame stack, so tn36 and sb26 suffer [WU].
- **Actions:** UP/DOWN/LEFT/RIGHT/SPACE/MOUSE/RESET. UNDO is withheld because the model misuses it ([WU]; `action_names.py:8-15`).
- **Prompt:** tells the model to ignore the energy/edge bar (`prompts.py:28-29`). [WU] admits heavy prompt tuning: no bar-as-goal, no hallucinated sprites, no dumping whole boards.
- **Scheduling:** none. Games run in list order through a semaphore (`solver.py:896-926`). A turn yields to the solver after 60 s, time-based (`kaggle.py:111`).
- **Tufa's own limits** [WU]: no ablations, no compaction or curated memory, weak perception, huge variance. Hand-crafted tools "did not help" ([WU][APB]); the stated drivers were better base models and multimodality.

## 3. What changed, grouped by driver

### A. More useful tokens per hardware budget (same 1x RTX Pro 6000, 9 h, 110 games [WU]; [FW]:49)
- **Model:** Qwen3.6-27B, then 3.8-27B, then Flash-Next. Franzen says each switch was "a substantial improvement"; per-switch deltas are unknown ([FW]:15,29). rellik13's first Flash-Next score went 0.99 to 3.81 ([RW]:28-29).
- **Quant and speculation:** Franzen runs W4A16 AutoRound plus an MTP draft with 3 steps, and keeps the PLE table in host RAM ([FW]:31-33,59). lordhansolo runs NVFP4 with its built-in MTP head ([LH] `PATCH_README.md:9-28`).
- **Server patches:** Franzen on SGLang 2.5.3 with 5 patches ([FW]:35-41). lordhansolo on vLLM e975732 with a 32-file overlay ([CMP]:149-151). rellik13 on SGLang 2.5.0 with 4 boot patches ([CMP]:183-187).
- **FP8 KV cache** is used by all three ([CMP]:3-4). rellik13's last jump, 14.49 to 22.53 (+8.04, derived), was FP8 KV, longer history and more context, "changed no prompts" ([RW]:8,96-101).
- **Cache reuse:** 93.43% of prompt tokens reused over 5,132 requests, from an earlier demo run, not the final config ([FW]:101-112).
- **Yield:** Duck yields after 60 s ([DUCK] `Makefile:148`). Franzen yields after 2,048 generated tokens, so it no longer depends on throughput ([FW]:299).
- **Thesis:** lordhansolo says "the score follows the number of turns each game gets" ([LHS]). Peak decode tok/s was 946 / 1,135 / 1,159 for 10 / 14 / 16 slots ([CMP]:207).

### B. Remember more, cache-friendly (raw context beats curated notes)
- **Context size:** Duck keeps about 32k and caps at 64k. Franzen uses a 131,072 harness window and a 139,264 server (timeline notes citing [FW]:51-54), "helped especially with the more difficult games and levels" ([FW]:67, qualitative). lordhansolo uses a 147,072 server and trims 126k to 81,536 ([LH] `inference.json:43,51,71`). rellik13 went 49k to 69.6k ([RW]:96-101).
- **History cap:** Duck 30 assistant turns. Franzen 150 with a 30-turn drain ([FT]; the write-up never mentions it). lordhansolo's cap is off whenever a target context is set ([CMP]:59).
- **Block trimming for the prefix cache:** Franzen drains about 58Ki at once ([FW]:93-97). lordhansolo trims 126k to 81.5k. rellik13 trims 57,344 to 45,056 (`cells/35...py:30-31`).
- **Token estimate:** Franzen uses a calibrated ratio plus image tokens from PNG size ([FW]:73-75). Duck uses len/3 and counts base64 as text ([DUCK] `tool_agent.py:462-467`).
- **Thinking retention** was kept; removing it "leads to a large regression" ([FW]:69).

### C. Allocate compute across games
- **Duck:** list order, fixed per-game cap, no priority (see section 7 for conflicting settings).
- **Franzen:** all games start but only 10 streams are admitted ([FW]:123-126). Priority is P=(A+B)·C ([FW]:128-142):
  - A is level gain with an efficiency penalty (human-action constant 25).
  - B is a lookup by levels remaining, 8/7/5/0.
  - C is a Gaussian decay in actions (scale 115) and tokens (scale 62,000), fit on 725 level attempts (557 done, 168 not). A Gaussian beat an exponential ([FW]:157).
  - B fades over the last 20% of runtime ([FW]:164). A game keeps its slot until context eviction forces a re-prefill ([FW]:174).
- **rellik13:** UCB with 2,400 s slices on a pool of about 31k s, 16 concurrent. A priority scheduler was written but held back ([CMP]:71-74; [RW]:128-132).
- **lordhansolo:** waves of 14, 3,918 s per game, no priority ([CMP]:73-76).
- **Post-submission, not Franzen's work:** shiiin9 changed only the priority formula and reports 31.5 LB from one submission ([FC]:350). Suggestive that the scheduler was a cheap lever; one sample.

### D. Better perception and truthful feedback
- **Animation:** Franzen exposes intermediate frames and a timeline as REPL variables, "strongest benefit" of the perception changes, tn36 essentially unsolvable without it ([FW]:181-185). lordhansolo caps `animation_frames` at 16 ([LH] `animation.py:37`). rellik13 added M76 mid-action frames. Duck had none.
- **`frame_diff()`:** object-level moved/rotated/appeared/recolored/resized ([FW]:190-192).
- **Image scale:** 4x to 10x, the "biggest score improvement" among image changes ([FW]:203). Also diff images and a fatal-frame image ([FW]:199-201).
- **UNDO exposed** by all three, but only clearly useful on su15 among demo games ([FW]:262-264).
- **Game-over text:** Franzen wrote a long "do NOT replay the fatal sequence" diagnosis ([FW]:216-233). rellik13 added an accurate "Level restarted." ([RW]:80-85).
- **Edge bar:** Duck says ignore it ([DUCK] `prompts.py:28-29`). lordhansolo says it is a per-level action budget ([LH] `prompts.py:58`). rellik13 deleted the ignore-bar line ([RW]:80-85).
- **Guards (Franzen):** batch no-op (ignores a 4-cell border), stale-state, terminal-state ([FW]:282-288). No ablation number.

### E. Persistent, verified artifacts across levels
- **Franzen:** `retained_functions.py`, scope=game ([FW]:255-257). The bundle of undo, action info, level-transfer and persistent functions gave "a large improvement"; no isolated delta ([FW]:266).
- **lordhansolo:** `save_module`; a save that breaks older replayed tests is rejected ([LHS]; [LH] `prompts.py:122`). Also `game_overs` and a per-action `plan` field.
- **rellik13:** M85 pins the winning rule and last 30 actions of each cleared level in the system prompt, added after transcripts showed 60-100+ minutes re-deriving ([RW]; `cells/42...py:1-30`). Dated step 7.14 to 9.96 (+2.82, derived).
- **Duck:** only the text "Cross-level notes" ([DUCK] `tool_agent.py:263-296`). Tufa lists missing "curated memory" as a limitation [WU].

### F. Removing things that hurt
- Franzen turned the structured world-model sections off (`ARC3_MEMORY_SECTIONS=off`), "largest improvement in that area", because the bigger rolling context already holds the evidence ([FW]:295).
- rellik13: always-on text 6.42 to 4.16, and a level-start advice block 13.40 to 9.98 ([RW]:59). The cause (instructions displacing recent turns at a ~37k cap) is his unproven hypothesis.
- Duck's fixed per-game time and 16-way concurrency were replaced by the admission gate ([FW]:123-126).

## 4. Converged vs diverged
**Converged (all three top placers):**
- Same model (Flash-Next), FP8 KV, 3-step MTP, one GPU ([CMP]:3-4,19,94).
- Lifted or dropped the 30-turn history cap ([CMP]:43). Trimmed in large blocks to preserve the prefix cache.
- Animation frames exposed, UNDO exposed. A cross-level carry-over mechanism, built differently by each.
- Two of three reversed Duck's "ignore the bar" (lordhansolo, rellik13).
- Throughput patches to the server stack. Franzen and rellik13 used Claude, ChatGPT or Claude Code; lordhansolo unknown ([FW]:316; [RW]:14).

| | Franzen 27.89 | lordhansolo 23.84 | Lohit 22.53 |
|---|---|---|---|
| Server | SGLang 2.5.3 | vLLM nightly | SGLang 2.5.0 |
| Slots / prompt budget | 10 / ~118k | 14 / ~126k | 16 / 65.5k |
| Trim | to ~59k | to 81.5k | 57k to 45k |
| Board image | 640 px, ~402 tok | 256 px, 66 tok | 256 px, 66 tok |
| Scheduling | priority gate | waves | UCB |
| Prompting | added transfer/game-over text; memory off | rewrite + strategy audit | added text hurt; deleted lines |
| Cross-level memory | retained functions | save_module | pinned solved-level rule |
Source: [CMP]:10-19,36-50,67-76; prompting row from the other-placers notes.
- Tong's inference: "The top score belongs to the build with the fewest server slots and the longest per-slot context" ([CMP]:31). Three entries, not a controlled result.
- Demo means are not comparable: Franzen 36.56 on 10 games, lordhansolo 5.28 and Lohit 6.89 on 25 ([CMP]:219,238).
- Image resolution is a real divergence: Franzen invested in 10x; Lohit found higher-res did not help ([RW]:140-148).

## 5. Tried and did NOT help (or not shown to help)
- **Franzen** ([FW]:295-305, "no clear gain"): summaries; composite animation images and auto timeline insertion (hurt some games); resume-prompt variants; stronger step-verification hint; repeat-state, remembered-no-op and death-ledger guards; pace adjustment; normalized-B lookup.
- **rellik13** ([RW]:140-148): advice or summaries, fresh-context restarts, temperature 1.0, proven/refuted ledger, expert pruning, higher-res images, Swift 1.5, a LoRA trial. Also the 6.42 to 4.16 and 13.40 to 9.98 text regressions ([RW]:59).
- **lordhansolo:** "I cannot confirm that any of these changes improved the score" ([LHS]). A later vLLM runtime v3 had no run that beat the score (`competitors/kaggle/built-on-tufa-labs-duck-harness-milestone-2/*.txt`, cell 0).
- **MS1 Tufa:** hand-crafted tools "did not help" ([WU][APB]).
- **Untested:** Franzen ran out of Kaggle compute before a full demo run with his final config ([FW]:62).

## 6. Generating lessons (my synthesis; the sources give the evidence, not these framings)
1. **Same compute, more thinking per game.** Hardware and rules were identical across milestones. The big jumps came from model, quantization, speculative decoding and FP8 KV, which raise tokens per second. rellik13's final change (+8.04) and lordhansolo's thesis point the same way. Hard to separate model from serving, since no one ablated them.
2. **Raw long context beats curated prose notes.** Both teams removed or ignored the 32k, 30-turn, text-notes design. Franzen turned the notes off; rellik13's extra text hurt. Cost control came from block trimming that keeps the prefix cache valid.
3. **Persist verified artifacts, not summaries.** Summaries and world-model text showed no gain, while retained functions, `save_module` and pinned solved-level rules were claimed to help. This is an interpretation of two groups of claims that are each confounded.
4. **Make feedback truthful and visible.** Animation, frame diffs, UNDO, fatal-frame and game-over messaging, and an accurate edge-bar rule all fix cases where the model could not see or was told the wrong thing. Tufa's own "ignore the bar" guidance was reversed by two teams.
5. **Spend compute where marginal progress is.** Franzen's calibrated priority gate and rellik13's UCB reallocate budget across games. The shiiin9 formula tweak (31.5, one run) hints this lever was not exhausted. Evidence here is the thinnest of the five.
6. **Read scores skeptically.** Same notebook varied 5.02 to 6.42 for rellik13 ([RW]:17-20); Franzen copies SD 3.93. Steps smaller than about that range cannot be attributed.

## 7. Uncertainties and conflicts in the notes
- **Duck's Kaggle settings conflict.** MS1 notes say `make kaggle-duck` is 132 min per game, 540 min total, 28 concurrent (`Makefile:400-405`). Franzen notes say 16 concurrent, 75 min per game ([DUCK] `README.md:254`). Unresolved; may be README default vs Makefile target.
- **Duck's upscale:** source default is 16 (`vision_context.py:42-49`, `Makefile:158`), but [WU] and [FW]:203 say 4x. Provenance of 4x unresolved.
- **Duck's history cap in the 1.21 run is unknown.** A dev-branch commit says "default to unlimited" and the run is labelled "0-history-turns" (example-run `git_info.txt`).
- **Franzen's trim target** appears as about 57k, 58Ki and about 59k across sources.
- **Duck 1.03** appears in an AlphaSignal article (alphasignal.ai); I use Tufa's 1.21.
- **Tufa 45.33% on Sep 29** comes only from postcutoff.com (secondary, unchecked). The LB pull is stamped 2026-10-05, after the session date of 2026-10-04; likely a timezone effect, unverified.
- **Not found:** Reki's MS1 score, per-feature and per-model-switch deltas, lordhansolo's score history, and whether Tufa's `taaf-kaggle-source-share` dataset matches the 7652836 snapshot.
- **Tension, not resolved:** Tufa says hand-crafted tools did not help, yet Franzen added `frame_diff` and retained functions. Possibly "perception helpers" differ from "hand-crafted solvers"; the notes do not say.
