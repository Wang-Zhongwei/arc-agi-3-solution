# m1-vs-m2-trajectory: research:ms2-others

Workflow run `wf_903996f8-246`, phase Research. Research what changed from ARC-AGI-3 Milestone 1 (Tufa Duck) to Milestone 2 solutions, then verify the claims

---

# MS2 placers other than Franzen: what each changed vs Duck

Paths below are relative to `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/` (`C/`).

## 0. Official placings and identity notes
- **Official placings:** 1. Daniel Franzen 27.9% ($25K), 2. "Lord Han Solo" 23.8% ($7.5K), 3. Lohit Siriki 22.5% ($5K). Source: ARC Prize post https://x.com/arcprize/status/2105737436450201734. Kaggle LB values are 27.89 / 23.84 / 22.53 (`C/tong-m2-comparison/kaggle/COMPARISON.md:12`).
- **Prize split (secondary source):** https://postcutoff.com/e/2026-10-01-arc-prize-2026-milestone-2-winners/ says the split differs from the competition page's $25K/$10K/$2.5K and that ARC Prize gave no reason.
- **Tufa Labs (secondary source):** the same page says Tufa Labs led the live board at 45.33% on Sep 29 but won no MS2 prize. I have not checked this against a primary source.
- **rellik13 and LohitSiriki are the same entry.**
  - The Kaggle write-up at https://www.kaggle.com/writeups/sirikilohit/arc-prize-2026-arc-agi-3-3-milestone-2 is authored "rellik13" under the handle sirikilohit.
  - The local `C/rellik13-milestone2/README.md:1-6` names kernel `sirikilohit/arc-agi-3-duck-18-1gc-submit` and the same 22.53.
  - `COMPARISON.md:258` cites `LohitSiriki/arc-agi-3-milestone2-solution/WRITEUP.md`, which has the same text.
- **`C/ARC3-solution` is not LohitSiriki's.** It is Tufa Labs' "StochasticGoose", a CNN action-learning agent from the preview competition (`C/ARC3-solution/README.md:1-2`).

## 1. lordhansolo (#2, 23.84): serving throughput first, harness rewrite second
Sources: his write-up https://www.kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2 (cited as LHS-W) and his bundle under `C/tong-m2-comparison/kaggle/lordhansolo/` (cited as LH/).
- **Thesis (LHS-W):** "the score follows the number of turns each game gets", so most work went into serving more tokens per game.
- **Stack:**
  - Model `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` with its built-in NVFP4 MTP head.
  - vLLM main commit e975732, patched, rather than SGLang.
  - Only entry on vLLM (`COMPARISON.md:17`).
- **Serving settings:**
  - 14 slots, FP8 KV, and a 147,072-token server context (`LH/dataset-taaf-kaggle-source/src/ARC3-Inference/configs/inference.json:43,51`).
  - 1,417,100-token KV pool, from his vLLM Save & Run log (`COMPARISON.md:213`).
  - Trim from ~126k down to 81,536 (`inference.json:71`; `COMPARISON.md:44`).
  - The 30-assistant-turn history cap in Duck (`C/duck-harness/ARC3-Inference/inference/agent/tool_agent.py:151`) is off whenever a target context is set (`COMPARISON.md:59`).
  - Reasoning effort "xhigh" with `preserve_thinking` (`inference.json:66,74`). Duck's config has no reasoning-effort key (grep of `C/duck-harness/ARC3-Inference/configs/inference.json`).
- **Serving patches:**
  - A 32-file hash-pinned overlay on the vLLM image, with 9 vLLM PRs ported and 16 custom fixes (`COMPARISON.md:149-151`, `175-182`).
  - Mamba state retention so prefix caching survives tool gaps, SM120 small-batch GEMMs, and exact CUDA graphs for 14 sequences with MTP3 (`LH/dataset-vllm-main-e975732-arc3/PATCH_README.md:9-28`).
  - Game state kept in memory instead of on disk, to avoid an idle engine between turns (LHS-W).
- **Scheduling:** no priority scheduling. Games run in waves of 14 with a 3,918 s per-game cap, and a freed slot goes to the next game (`COMPARISON.md:73-76`).
- **Harness (a rewrite of tool_agent.py, 1,745 lines vs Duck's 2,063; `COMPARISON.md:146`):
  - `save_module` stores tested Python modules for the whole game. New tests replay against frozen observations, and a save that breaks older tests is rejected (LHS-W; `LH/.../agent/prompts.py:122`).
  - Animation frames are exposed in `current_frame.animation_frames`, capped at 16 (`LH/.../agent/animation.py:37`).
  - `game_overs` and a per-action `plan` field let the model recall what it tested after those turns leave context (`prompts.py`).
  - A "strategy audit" prompt appears once a level stalls (`prompts.py:11`, `tool_agent.py:712`).
  - ACTION7 is renamed UNDO (`LH/.../agent/action.py:20`).
  - Image budget is small: 256 px boards, ~66 tokens each (`COMPARISON.md:47-48`).
- **Prompt change that reverses Duck:**
  - Duck says to ignore the edge bar (`C/duck-harness/.../prompts.py:28-29`).
  - He says it is a per-level action budget, that every action counts against it, and that exhausting it ends the level (`LH/.../prompts.py:58`).
  - He also adds "Every game is solvable" and tells the model that later levels weigh more (`prompts.py:51,57`).
- **Admitted limits:** "I cannot confirm that any of these changes improved the score" (LHS-W). A later vLLM runtime v3 was meant to be faster, but "no run with it has beaten that score" (`C/kaggle/built-on-tufa-labs-duck-harness-milestone-2/*.txt`, cell 0). The rc2 runtime was tested on an RTX PRO 4000 laptop and never benchmarked on the RTX PRO 6000 before upload (`COMPARISON.md:181-182`).

## 2. LohitSiriki / rellik13 (#3, 22.53): memory, not instructions
Sources: `C/rellik13-milestone2/WRITEUP.md` and `README.md`, plus the Kaggle write-up above.
- **Score history (WRITEUP.md:26-35):**
  - Aug 11: 0.99.
  - Sep 2: 3.81 (Qwen3.8-Flash-Next).
  - Sep 18: 6.42 (SGLang, more concurrency).
  - Sep 25: 7.14 (49k context).
  - Sep 26: 9.96 (solved-level memory).
  - Sep 27: 13.40 (Wang harness).
  - Sep 29: 14.49 (harness fixes).
  - Sep 30: 22.53 (FP8 KV, longer history).
- **Final jump:** "changed no prompts" (WRITEUP.md:8). BF16 to FP8 KV, trim 36,864→26,624 became 57,344→45,056 (confirmed in `cells/35...py:30-31`), context 49,152→69,632, 15→16 games (table at WRITEUP.md:96-101). Elsewhere he says prompts were unchanged but his first-pass prompt edits are listed as failures.
- **Added text hurt:** 6.42→4.16 for always-on text, and 13.40→9.98 for a level-start advice block (WRITEUP.md:59). His explanation is unproven: with a ~37k cap, instructions displace the agent's own recent turns (WRITEUP.md:59-64).
- **Harness additions (patch cells, `COMPARISON.md:188-196`):**
  - M85 solved-level memory pins the winning rule and the last 30 actions of each cleared level in the system prompt (`cells/42...py:1-30`). It was added after transcripts showed 60-100+ minutes spent re-deriving the previous level's rule.
  - M90 ports "Wang/Ludvig" harness commit 6420079: stricter sandbox, object segmentation, and a 5-line turn message (`cells/40...py:1-4`). Wang's repo later went private (WRITEUP.md:169).
  - UNDO label for ACTION7, accurate "Level restarted." after game over, and the "ignore the bar" line deleted (WRITEUP.md:80-85).
  - M76 mid-action animation frames.
  - M79 exact-token context guard.
- **Scheduler:** one pool of ~31k s, 2,400 s UCB slices, 16 concurrent, 28 games live (`COMPARISON.md:71-74`). A priority scheduler was written but held back (WRITEUP.md:128-132).
- **Serving:**
  - SGLang Pennyroyal 2.5.0 with Intel W4A16 plus RadixArk FP8 PLE, and an NVFP4 MTP draft.
  - A 48 GB host KV/Mamba tier (`COMPARISON.md:105-106`).
  - Four boot-time source patches (`COMPARISON.md:183-187`).
- **Self-reported failures:** advice or summaries, fresh-context restarts, temperature 1.0, a proven/refuted ledger, expert pruning, higher-res images, Swift 1.5, and a LoRA trial (WRITEUP.md:140-148).
- **Variance:** one notebook scored 5.02, 5.50 and 6.42 across three submissions. A notebook that scored 37 on his practice run scored 9.98 on the LB (WRITEUP.md:17-20). The final notebook was only checked by a 20-minute Save & Run (6.89, per his Kaggle write-up), so the history effect was never directly shown (WRITEUP.md:122-126).

## 3. Side-by-side (`C/tong-m2-comparison/kaggle/COMPARISON.md`, the Kaggle forum comparison linked from `tong-m2-comparison/README.md:7`)
| | Franzen 27.89 | lordhansolo 23.84 | LohitSiriki 22.53 |
|---|---|---|---|
| Server | SGLang 2.5.3 | vLLM nightly | SGLang 2.5.0 |
| Slots / prompt budget | 10 / ~118k | 14 / ~126k | 16 / 65.5k |
| Trim | to ~59k | to 81.5k | 57k→45k |
| Board image | 640 px, ~402 tok | 256 px, 66 tok | 256 px, 66 tok |
| Scheduling | priority gate | waves | UCB |
Source for the table: `COMPARISON.md:10-19, 36-50, 67-76`.
- **Tong's inference** (`COMPARISON.md:31`): "The top score belongs to the build with the fewest server slots and the longest per-slot context." This is a pattern across three entries, not a controlled result.
- **Demo mean scores are not comparable** (Franzen 36.56 on 10 games; lordhansolo 5.28 and Lohit 6.89 on 25; `COMPARISON.md:219,238`).

## 4. Common threads across MS2
- **Same model and a fast-decode stack.** All three use Qwen3.8-Flash-Next with FP8 KV and 3-step MTP on one RTX PRO 6000 (`COMPARISON.md:3-4,94,19`). Franzen's TL;DR says each model switch gave a "substantial improvement" (`C/tong-m2-comparison/WRITEUP.md:12`). Lohit's first clear jump was the same switch.
- **Keep more history, and keep it cache-friendly.** All three dropped Duck's 30-turn cap or lifted it (`COMPARISON.md:43`). All three trim in big blocks to preserve the prefix cache (Franzen's hysteresis, WRITEUP §2; Lohit's watermarks; lordhansolo's 126k→81.5k).
- **Cross-level carry-over.** Franzen retains Python functions for the whole game (WRITEUP:255). lordhansolo has `save_module`. Lohit pins solved-level rules. Each solves "don't rediscover on every level" differently.
- **Fix what misleads the model.**
  - All three expose animation frames (Franzen's WRITEUP:179; lordhansolo's `animation.py`; Lohit's M76).
  - All three expose UNDO.
  - Lohit and lordhansolo both reversed Duck's "ignore the edge bar" guidance.
- **Concurrency and throughput were first-class.** Peak decode tok/s was 946 / 1,135 / 1,159 for 10 / 14 / 16 slots (`COMPARISON.md:207`).
- **Variance warnings and no controlled ablations.** Lohit and lordhansolo say so explicitly; Franzen's section 6 is "practical experiment results rather than a complete controlled ablation study" (WRITEUP:293).
- **AI-assisted development.** Franzen used Claude and ChatGPT (WRITEUP:316), Lohit used Claude Code (WRITEUP:14), lordhansolo: unknown.

## 5. Key differences
- **Prompt philosophy:**
  - Lohit found added prompt text harmful.
  - lordhansolo rewrote the system prompt and added an audit prompt.
  - Franzen added level-transfer guidance and action addenda (`COMPARISON.md` Prompts row of `SUMMARY.md`), but disabled the world-model sections.
- **Perception:** Franzen invested in 10x images and diff images (WRITEUP TL;DR). The other two kept 256 px images, and Lohit found higher-res images did not help.
- **Context vs concurrency tradeoff:** Franzen and lordhansolo bet on very long context, 128k+. Lohit stayed at 69.6k with more slots. Lohit's own data shows the context increase paid off from the 37k baseline, but that does not show 69k beats 128k.
- **Scheduler:** Franzen's progress-based priority (WRITEUP §3) and Lohit's UCB reallocate compute; lordhansolo's plain waves do not.
- **Engineering depth:** lordhansolo shipped the heaviest serving fork (vLLM overlay). Franzen and Lohit shipped SGLang patches (5 and 4 respectively).

## 6. Gaps in this report
- **Franzen's prize and diff** are covered by another task, so I only cite his write-up for context here.
- **lordhansolo's score history** is not published; I cannot show his learning trajectory.
- **Other local notebooks:** `C/kaggle/` also holds several Franzen forks and a reproduction (`dfranzen-fork`, `arc-ak47-3-agi`, `arc-prize-2026-sovereign-agent`, `arc-agi-3-27-80-lb-100-trajectory-audit`). I did not analyze them.
