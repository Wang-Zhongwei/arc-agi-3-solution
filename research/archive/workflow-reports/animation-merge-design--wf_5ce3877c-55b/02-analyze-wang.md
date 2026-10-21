# animation-merge-design: analyze:wang

Workflow run `wf_5ce3877c-55b`, phase Analyze. Compare Wang vs Franzen animation-frame mechanisms with evidence, then design a merged mechanism

---

## Wang's animation-frames mechanism (read-only analysis)

Paths are in `/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/wang-duck-harness`, under `ARC3-Inference/`. `main:` is Wang's work and `my-main:` is yours (36 commits, `main..my-main`).

### 0. Provenance and which notebook
- **Wang's commits on `main`, all by Zhongwei Wang:**
  - `127970d` (2026-09-02, "prepend intermediate frames before user message"): the whole mechanism, 8 files and +376 lines, including `tests/test_action_animation_frames.py`.
  - `fc954eb` (09-10, "keep only last image").
  - `f1ec2e5` (09-10): level-winning animations are kept.
  - `e2d3bf5` (09-12): keep every board image in history, strip only the animation.
- **Your additions, not Wang's:**
  - `4b174e7` (09-22) adds `MULTIMODAL_ANIMATION_UPSCALE`, default `min(board upscale, 4)` (`my-main:vision_context.py:135-137,290-309`).
  - `9fceefd` (09-23) puts `animation` (numpy list) and `animation_steps()` in the sandbox, and captures frames even with images off (`my-main:prompts.py:293-300`, `vision_context.py:161-170`).
  - `05fdabb` adds the changed-cells diff image.
  - `cb64b83`, `86a4910` and `b178e88` on `gb10/frame-diff` add the text diff (section 5).
- **Wang's notebook carries no animation code.** `taaf-duck-harness-kaggle-share.ipynb` has 17 cells with 0 hits for "anim" or "MULTIMODAL". `experiments/wang-duck-harness` has only an older snapshot of the same code.
- **The Kaggle notebooks under `competitors/kaggle` don't contain it either.** `duck-qwen3-8-*` are infra-only, and no file there contains `MULTIMODAL_ANIMATION`.
- **Wang's recipe is set in config and env.** `configs/inference.flash-next.json:121-126` sets `context: current_grid`, `upscale: 4`, `animation: true`, `animation_max_frames: 8`. `gb10/runner:scripts/gb10/kaggle_run_env.json` exports `MULTIMODAL_ANIMATION "true"`, `MAX_FRAMES "8"`, `UPSCALE "4"`.

### 1. Capture
- **Source.** arcengine returns one frame per simulation step, and `state.raw.frame` is that list. Only the last entry becomes `current_frame` (docstring, `main:solver.py:122-141`).
- **Hook.** `self.animation_frames = _action_animation_frames(...)` after every `execute_action` (`solver.py:894`). It is overwritten on every action.
- **Skipped cases.**
  - Feature off (`solver.py:~142`).
  - RESET (`:144`).
  - Final state WIN or GAME_OVER (`:146`).
  - An action with `<= 1` raw frame (`:149`).
- **Level-winning actions are kept** (`f1ec2e5`: "they don't have unnecessary green zoom in and out").
- **Batches keep only the last action's frames.** `step_env` loops `_execute_action` (`solver.py:789-858`) and each action overwrites the list. The loop stops at `level_completed`, so a winning action's frames are the ones kept.
- **The model gets confused by this.** In `final-c1-ledgerfix/transcripts/g50t-5849a774_p0.txt:3951` it writes: "the intermediate frames are from the last action of the previous batch? ... Ambiguous".
- **Persistence.** The frames go into the runtime-state JSON as `"animation"`, rewritten per action so none lingers (`runtime_state.py:140-157`). `analyze()` reloads them at turn start (`tool_agent.py:1732`).

### 2. Compression and deduplication
- **Drop the final frame**, since it equals `current_frame` (`solver.py:156`).
- **Collapse only adjacent identical grids** (`:158`), done before the cap. A test covers a held pose not eating the budget.
- **Cap at 8 frames** (`MULTIMODAL_ANIMATION_MAX_FRAMES`, `vision_context.py:96-109`).
  - Sampling is an even subsample with `round(i*(n-1)/(limit-1))`, always keeping the first and last (`solver.py:163-167`).
  - Wang's docstring (`vision_context.py:99-101`) gives measured raw lengths: "2 frames in sk48, 15 in cd82, up to 111 in sb26".
  - There is no diffing, no transient-pixel detection, and the pre-action board is not in the chain.

### 3. Rendering, timing and placement
- **Format.** Python `Frame(grid, step, level)` objects become one PNG each (`frame_to_png_data_url`, `vision_context.py:54-73`).
  - Plain 16-colour ARC palette, nearest-neighbour upscale, no labels or coordinates.
  - Separate images, not a composite.
- **Gating.** Animation needs `MULTIMODAL_CONTEXT=current_grid`, so the ASCII-only arm gets none (`vision_context.py:79-93`).
- **When shown.** Once, at the start of the next turn after any action that animated. It is not shown during a turn: the python sandbox state has no animation (`tool_agent.py:~1501-1526`), and the turn ends when a python call executes an action (`:1982-2003`).
- **Placement.** `_build_user_message` (`tool_agent.py:1186-1216`) builds `[header text, N frame images, user_prompt + "\n\nCurrent grid image:", current board image]`. Animation comes first, then the unchanged prompt, then the live board.
- **History.** `_drop_animation_parts` (`:456-489`, applied at `:1664-1673`) removes the header and every image but the last from persisted messages. Past board images stay and animation never accumulates (test `:127-175`).

What the model sees, verbatim. System prompt, `prompts.py:78-83`:
```
Intermediate frames:
- When an action changes the board over several steps, those in-between frames are attached at the top of the turn, oldest first, ending at the current grid image.
- Use them to see how the board reached its current state: what moved, in which direction, and what it passed through or collided with.
- Only the current grid image is the live board; the earlier frames are already past.
```
The user turn is `"Intermediate frames from the last action (3, oldest first):"`, then three 256x256 PNGs, then `<prompt>\n\nCurrent grid image:` and the current board PNG.

### 4. Token cost
- **Per frame:** 256x256 is 8x8 visual tokens plus 2 markers = **66 tokens** at upscale 4. I checked this against `smart_resize` (`vision_context.py:160-180,201-222`). Your docstring in `my-main:vision_context.py` agrees: 66 at 4x and 1026 at 16x.
- **Per turn:** at most 8 frames + 1 board = 594 image tokens, transient (not carried into history).
- **Upscale default:** `Makefile:261` defaults upscale to 16, which is 1026 per frame and about 8.2k for 8 frames. Config and `kaggle.py:199` pin it to 4, and `4b174e7` caps animation frames at 4 regardless.
- **Wall-clock:** `gb10/frame-diff:notes/gb10-arms/frame-diff.md` reports 0.145 s per generated token versus 0.002 s per uncached prompt token. So 528 image tokens cost about 1 s of prefill.
- **Trimmer:** it prices images exactly (`tool_agent.py:492-520`).

### 5. Evidence of benefit or harm
- **No on/off score ablation of Wang's mechanism exists in the repos or `run-atlas`.**
  - Wang's job-76469 note says "startup validation, not a completed game-score" (`my-main:notes/serving/flash-next-h100-2node.md:224`).
  - `duck-arm-animation-meta` scored 5.56 against 6.98 for `duck-baseline-control` (run-atlas `runs.csv`, 2026-09-07), but it has no description and I can't tie it to Wang's code.
- **The model reads the frames by eye.** In `final-c1-ledgerfix` transcripts, tn36 ended at 10.71 (2/7 levels).
  - tn36 line 567: "The intermediate frames show a blue highlight sweeping across the 5 tokens".
  - tn36 line 1284: "Frame 1: blue box at leftmost (cols ~20-22); frame 2: at col 26 ... So left→right". Positions are estimated, not computed.
  - su15 line 4537: "I can inspect intermediate frames? The intermediate frames are attached as images, and history only stores post-action frames". The model cannot compute on frames or revisit them next turn.
- **tn36 across eras is a weak signal.** Per-game tn36 scores are above 0 in 13 of 29 Wang-era runs (images and animation on), never above 2/7 levels. Franzen-era runs reached 3/7 once (`fz-streams14`, 20.32). Budgets differ, so don't treat it as an ablation.
- **GB10 frame-diff arm** (`GB10-OVERNIGHT-RESULTS.md:41,123,134`): the arm turned animation images off and used the text diff. The control kept animation images on.
  - ar25 level 2 was cleared in 2 of 2 arm runs versus 2 of 8 controls. Fisher p = 0.13 per the arm note, not significant.
  - Mechanism: the diff line "S+W 63x3 moved ... dc +3" told the model the wall moves, which its own python had filtered out.
  - Wang's images fired rarely in the controls: frame-diff note, "intermediate-frame images sent ... control b4: 2 requests" on ar25 and 6 on sk48.
  - Losing the images showed no visible harm there. The change is confounded (images and text changed together).
- **The diff block is cheap and exact.** Mean 459 chars (v1) and 216 chars (v2) on ar25, with a 1000-char cap (about 330 tokens). It is stripped from history.
  - Example: `last action: 96 cells changed; M+W+g+w 6x12 moved r36,c11 -> r30,c11 (dr -6, dc +0) ... its animation, 2 steps:`.
  - Remaining gap, the same as Wang's: it covers only the last `action()` call. The model writes loops of single calls, and the batch "net of all N actions" line fired in 1 of 23 blocks.
- **Harm that applies to images in general, including animation:**
  - A vLLM multimodal-cache 400→500 retry loop (6,165 "Multi-modal cache miss" errors; tn36 stuck for 4,187 cycles) is described in `my-main:notes/serving/ucb-v05-disk-and-retry-loop.md:~60-100`. The note's own words are "(grid plus animation images)". The proposed `--mm-processor-cache-gb 0` fix is unverified.
  - Stripping animation from history changes the prefix and hurts prefix-cache reuse (`docs/context-history.md:9`; `notes/serving/edge-out-performance-0917.md:38`).
  - The system prompt still says "earlier turns are text only" (`prompts.py:73`), but past boards stay as images (`GB10-OVERNIGHT-RESULTS.md:91`).

### 6. Franzen comparison hooks (`WRITEUP.md`)
- Line 181: exposing animation was the "strongest benefit", and tn36 is "essentially unsolvable" without it.
- Line 301: "composite animation images did not improve results and hurt some games"; auto-inserting the timeline had "no clear gain".
- Wang's frames are separate per-frame images, not composites, so Franzen's negative result may not transfer one-to-one. I did not verify what his composite looked like.

### 7. What to keep from Wang and what to fix
- **Keep:**
  - Full-chain harness-side capture.
  - Adjacent-dedup before sampling, and the cap that preserves the first and last frames.
  - Transient per-turn attachment with history stripping.
  - Exact image pricing, and the cheap 66-token frame.
- **Mitigate:**
  - Eyeballed-only frames. Use your `animation` / `animation_steps()` (`9fceefd`) or frame-diff text.
  - Last action only. Net-diff plus per-call accumulation across a python execution.
  - Always auto-inserted. Franzen found no gain from auto-insertion, so gate on transient pixels.
  - Frames lost after one turn. Keep them readable from the REPL.
  - Multimodal-cache loop. Test `--mm-processor-cache-gb 0`.
