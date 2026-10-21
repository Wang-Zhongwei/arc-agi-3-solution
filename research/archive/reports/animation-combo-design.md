# Animation combo: Franzen's animation access plus Wang's proactive frames

Design doc, 2026-10-08. Read-only study; no code was changed. Owner (2026-10-08): "dan fran's scores were through turning on animation frames. i think combining his algorithm and wang's animation strategy is very useful."

**Bottom line.**
- Build the combination as one arm branch with **three default-off layers**, each tested on its own:
  1. **truth**: stop calling animated actions "NO-OP".
  2. **ring**: keep the last 8 animations readable from Python.
  3. **digest**: Wang-style proactive text in the next opener.
- Real keyframe images come last and stay off unless the digest pays.
- Do it by **rebasing PR #40 onto `main`** (no conflicts; same code base) and splitting its single master flag. Don't rebuild it.
- Measured stakes:
  - **20% of the harness's own NO-OP echoes (284 of 1,426, 5 runs) are actions that animated and hid something.** The "not a no-op" correction reaches the model only a turn later, and only for the last action.
  - About 10 times per run the model says, in its thinking, that an animation it wanted was overwritten.
  - Both cluster in 6–10 low-scoring games.
  - The text cost of fixing this is small: under 1% of prompt tokens.
- Wang's own evidence for proactive frames is weak:
  - His frames were images that fired rarely.
  - The best result is the GB10 text frame-diff arm: ar25 L2 in 2/2 runs vs 2/8 controls, Fisher p=0.13, one game, not on Kaggle.
  - Franzen found composite images hurt some games and auto-inserted timelines gave no clear gain.
- No public harness has an isolated measurement of animation presentation either (§2.3).
  - lordhansolo puts a typed text summary (blink, tween or path) in the tool result.
  - rellik13 built the same 8-animation ring (M111) and found extra per-turn text hurt.
  - The VLM literature consistently favours text over images for grids and fine change detection.
- So the proactive layer is the speculative one and goes last, short and typed.

Contents: 1. What each side does · 2. Evidence · 3. What it costs today (failure data) · 4. PR #40 review · 5. Recommended arm · 6. Rebase or rebuild · 7. Test plan · 8. Eval plan · 9. Open questions.

---

## 1. What each side does

### 1.1 Franzen (`main`, `ARC3-Inference/`)

Kaggle config (`kaggle/template.ipynb` cell 4): `ARC3_ANIMATION=1`, `ARC3_DIFF_IMAGE=1`, `ARC3_GAMEOVER_DIFF_IMAGE=1`, `ARC3_FRAME_DIFF_HINT=1`. Left at defaults: `ARC3_ANIMATION_TIMELINE` on, `ARC3_ANIMATION_IMAGE=off`, `ARC3_ANIMATION_DEATH_IMAGE=off`, `ARC3_AUTO_FRAME_DIFF=off`.

| Piece | Where | What it does |
|---|---|---|
| Capture | `inference/framework/solver.py:336-351` `_animation_chain`; `utils/animation.py` `normalize_frames`/`collapse_frames` | Prepends the pre-action board to `state.raw.frame` and collapses consecutive duplicates. The prepend is what lets a cell that changes X→Y→X register as transient. |
| Record | `solver.py:1271-1293` | **One** `animation_record` per session. Every executed non-RESET action replaces it, and a non-animated action clears it to `None`. The auto-RESET after a death skips it, so the fatal animation survives. |
| Summary | `utils/animation.py` `summarize_animation` | `{frames, transient_pixels, transient_bbox}` in the per-action payload under `animation`. Absent for single-frame actions, so they cost no tokens. |
| Query | `solver.py:995-1002` | `step_env({"query":"animation"})` returns the record without spending an action. |
| REPL | `python_tool_sandbox.py:592-640` | Lazy `last_animation_frames` (FrameViews with `.ascii/.shape/.segmentation`) and `last_animation_timeline` (per-step cell diffs; at most 24 cells per step and 80 in total, then a colour census). They are invalidated after every action (`:508-511`). |
| Opener hint | `tool_agent.py:4869-4881` → `utils/animation.py` `describe_animation` | One line, only when transient pixels > 0. If the board area did not change: "...was not a no-op: N cells changed and changed back... read `last_animation_timeline` before concluding it failed." There is a fatal-action variant at `:4994-5009`. |
| Prompt | `prompts.py:163-188` `ANIMATION_ADDENDUM(_TIMELINE)` | Static system text. It says `last_animation_frames` is "the frames of the last executed action that animated". In fact the record is cleared by the next non-animated action (§4, defect 5). |
| `frame_diff()` | `python_tool_sandbox.py:365-381` → `utils/frame_diff.py:82` | Object-level diff, called by the model. Issue **#41**: shape-hash matching runs before the colour check, so a highlight sweeping across identical tokens is reported as a "move". |
| Diff images | `tool_agent.py:3955-4060` (`ARC3_DIFF_IMAGE`), `:4321+` (game over) | Final-board diffs, not animation. |
| Images of the animation | `vision_context.py:134-148`, `tool_agent.py:4479-4580` | `peak`/`composite`. **Off** in his config: WRITEUP §6 says composite images "did not improve results and hurt some games where access to animation information had previously helped". |
| Auto text diff | `tool_agent.py:291-292, 4040-4060` (`ARC3_AUTO_FRAME_DIFF`) | Off. Diffs board to board across a whole turn, not the animation. |

WRITEUP:179-185:
- Animation access was the "strongest benefit" of all perception changes, and tn36 is "essentially unsolvable" without it.
- Frames are REPL variables "so the model can inspect, compare, and compute over them".
- The hint fires only on transient pixels.

WRITEUP:301: "Automatically inserting the timeline also showed no clear gain."

**Gaps:**
- Only the last action is kept, so earlier animations in a batch are lost and any later action overwrites the record.
- The same-call echo says `NO-OP` for an action that animated.
- The batch no-op stop and the stale-state latch treat an animated action as a no-op.
- `frames[k-1]` = timeline step k is never stated.
- `frame_diff` reports fake moves (#41).

### 1.2 Wang (`arc-agi-3-v3/wang-duck-harness`, branch `my-main`)
- **Origin.** `127970d` "prepend intermediate frames before user message" and `fc954eb` "keep only last image".
- **Capture.** `solver.py:269-351` `_action_animation_frames`:
  - Keeps every frame except the last, deduplicated.
  - Subsamples evenly to `MULTIMODAL_ANIMATION_MAX_FRAMES=8`, keeping the first and last.
  - Returns nothing for RESET, WIN or GAME_OVER, so a death animation is not shown.
  - Overwritten every action, so a batch shows only its last action.
- **Shown proactively, as images.** `tool_agent.py:2309-2358`:
  - Order: header "Intermediate frames from the last action (N, oldest first):", plain palette renders at `min(MULTIMODAL_UPSCALE,4)` px per cell (256×256, about 66 tokens per frame, about 530 for 8; `vision_context.py:125-213`), then the prompt text and the board image.
  - `_drop_animation_parts` (`tool_agent.py:1025`, applied at `:3161`) **strips them from history afterwards**. That rewrites earlier messages, which **breaks the prefix cache**.
- **No text diff** on `my-main` (TODO at `solver.py:307-319`).
- **On in the Wang-era Kaggle runs.** `kaggle_run_env.json:38` has `MULTIMODAL_ANIMATION=true`. In practice it fired rarely: GB10 control b4 sent frames in only 2 ar25 requests and 6 sk48 requests.

**`gb10/frame-diff`** (the user's arm on Wang): `cb64b83` v1, `b178e88` v2.
- `inference/agent/frame_diff.py` `describe_action`/`describe_batch`, capped at 1,000 characters:
  - Diffs the full chain (board before, every frame, final board).
  - Matches 4-connected same-colour objects by exact shape and merges parts that move together.
  - Collapses steady motion into one line and blinking into one "alternating" line.
  - v2 adds a "net of all N actions" line and never matches ambiguous 1×1 dots.
- Placed in the opener after "Executed actions:", shown only while current, and **stripped from history**.
- Size: v2 averaged 216 chars on ar25 and 328 on sk48 (about 55–80 tokens).
- **Confound.** The arm env also turned the animation images off.

### 1.3 Other designs (from the feature matrix, `research/reports/harness-feature-matrix.md:71-73`)
- lordhansolo: up to 16 frames and a tween/path/blink classifier.
- sirikilohit: up to 4 mid-animation images (M76) plus sandbox frames (M111).
- Tufa's shipped Duck: no animation.
- §2.3 checks these against their code and writeups.

---

## 2. Evidence: what worked and what hurt

| Mechanism | Evidence | Verdict |
|---|---|---|
| Frames and timeline as REPL variables, with a transient hint (Franzen) | WRITEUP:181: "strongest benefit"; tn36 unsolvable without it. Our data (§3): tn36 scored 0 in both runs where the model never triggered an animated no-op, and 10.7–100 where it did (correlation). | **Worked.** Keep it and extend it. |
| `frame_diff()` helper (Franzen) | Used in 140–266 calls per run (§3). No isolated score. Known fake moves (#41). | Useful. Fix #41 first. |
| Composite animation images (Franzen) | WRITEUP:301: "did not improve results and hurt some games where access to animation information had previously helped". | **Hurt.** Don't use. |
| Auto-inserted timeline (Franzen) | WRITEUP:301: "no clear gain". | Neutral. This is the closest prior to a digest, so the digest needs its own A/B. |
| Wang's proactive frame images | Never isolated. Fired in about 2–6 requests per game on GB10. Rewrites history (cache-unsafe). | Unknown. Copy the "proactive" idea, not the images or the stripping. |
| GB10 auto text frame-diff | ar25 L2 2/2 vs 2/8 controls (p=0.13). The model read the block on most acting turns and quoted its numbers. Mechanism: an exact "wall moved +3" line its own python had filtered out. sk48 0 levels everywhere. Wang harness, about 7 Kaggle-minutes per game. | **Promising, small n.** Supports a short exact text block, as distinct from Franzen's long timeline. |
| A3 `animation_meta` (per-action summary dict, about 26 tokens) | Kaggle 5.56 vs 6.14 (inside the 2.85 noise floor). | Neutral. Metadata alone is not enough. |
| numpy-perception `<perception>` block | ar25 2/2 by the same "moved" fact. | Same signal as the frame-diff arm. |

### 2.3 External evidence (online and competitor code, 2026-10-08)
[V] means checked against the source or code, [I] means inferred. Local copies are under `arc-agi-v4/competitors/`.

**Other harnesses.** None of them has an isolated, repeated measurement of how animation is presented.

| Harness | Mechanism | Placement and cost | Measured effect |
|---|---|---|---|
| **lordhansolo** (M2 2nd open-source, LB 23.84) | Collapses repeats and thins to 16 frames, keeping first and settled. The thinning steps past repeats so a blink can't alias away (`competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/.../agent/animation.py:37, 121-145`). A **typed text summary** (`_summarize`, l.163): `blink` (+cycles), `tween` (sprite hash, colour, from→to), `path` (≤8 points), or `other`, plus `frame_count` and `bbox`. Frames are REPL-only (`current_frame.animation_frames`) for the 16 newest history entries. | The dict is printed in the **action tool result** (`solver.py:1018`). The opener adds one line only when the net change was 0 but the action animated (`tool_agent.py:677`). Prompt: "If an action reports `changed_pixels` of 0, check its animation before concluding it did nothing" (`prompts.py:167`). About 40–80 tokens per animated action [I]. | None claimed: "I cannot confirm that any of these changes improved the score… The score alone mostly shows the dice." [V] |
| **sirikilohit / rellik13** (LB 22.53) | **M76:** at most 4 real mid-action frames as images, only when "information was actually lost" (some frame differs from the pre-action board in more cells than the final board does). Ranked by delta, with ≥90%-of-board flashes last (`cells/29_..._m76_mid_action_frames.py`). **M111:** `transition.frames`, at most 32 frames for the **last 8 animated actions, hex-encoded**, and only an `animation_frames` count in `action()` results: "Nothing is added to any turn message" (`cells/40_...:88-205`). | M76: 66 tokens per image at 4×, about 264 per message, shown once, after the board image. It fired on 138 of 1,302 actions (11%). | No isolated A/B; bundled in "Harness fixes". An earlier scalar version (M66: frame counts and changed cells) scored 11.55 vs 12.04: "a number it cannot form a hypothesis from". "What didn't work" lists "per-action 'what changed' notes and a level-start comparison block", and "More prompt text… dropped the score from 6.42 to 4.16". [V] |
| **Jakob Brüggen** (taaf-anim, LB 1.61, the origin of the transient-pixel test) | 13 of 24 games return multiple frames. Type 1 (ft09, sb26): the first frame equals the last, so "all information… lives only in between". Type 2 (r11l, sk48): pure interpolation. Per-action metadata plus an `animation()` diff call (8 steps × 24 cells). The hint fires only after ≥6 turns without progress. | A whole 64×64 frame is about 1,400–2,000 tokens as text, the reason he used diffs. | High variance and below 1 on test, per a commenter. The author: "very dependent on the models capability to decide whether to use the animation tool or not." [V] |
| **Franzen** (public WRITEUP) | As in §1.1. | | No numbers; qualitative "strongest benefit". [V] |
| **Wang** | No public repo ("made private", per rellik13). Only our local copy (§1.2). | | |
| **Tufa Duck** | Consumed only `raw.frame[-1]`. No stated reason. | | |

- The M2 comparison thread (discussion 744792) is about serving and says nothing on animation [V].
- ARC Prize docs: "Each response contains one or more 2D frame arrays"; multi-frame effects come from delaying `complete_action()` [V].

**Research on showing temporal information to VLMs** (primary sources):
- **Text beats images for grids and structured state.**
  - IsoBench (arXiv 2404.01266): every model does better on text than on isomorphic images.
  - SpatialEval (2406.14852): VLMs often trail their own LLM backbone.
  - BALROG (2411.13543): adding game-state images made agents worse.
  - ARC: object-based text nearly doubled GPT-4's solved tasks (2305.18354), and GPT-4V trails text-only GPT-4 on ConceptARC (2311.09247).
- **VLMs are weak at fine-grained change detection across images.** VDiff-Bench (2609.06245): 8.7–33% for 7–8B open MLLMs on low-level changes. Also Img-Diff (2408.04594), BlindTest (2407.06581), MotionBench (2501.02955), Mementos (2401.10529) and MMIU (2408.02718).
- **When text and image disagree, VLMs follow the text** ("blind faith in text", 2503.02199).
- **Frame selection.** Relevance-plus-coverage keyframe sampling beats uniform sampling (2502.21271, +5.0 points with Qwen2-VL). Caption-then-reason pipelines beat end-to-end video models on long video (LLoVi 2312.17235, LangRepo 2403.14622).
- **Qwen-VL geometry.** Qwen3-VL uses 16-px patches with a 2×2 merge, so one visual token covers 32×32 px. At 4 px per cell that is **8×8 board cells per token**, so a 1-cell transient is 1/64 of a token [I]. Qwen's *video* path merges consecutive frame pairs with depth-2 3D convolutions (2409.12191), so keyframes must go in as separate images, never as video [I].

**What this changes in the design:**
- (a) Text stays the primary channel, and images are corroboration at most. This matches Franzen's negative image result.
- (b) rellik13's M111 independently converged on PR #40's exact ring shape (last 8 animated actions, at most 32 frames, hex, nothing in the turn message), which supports L2 as the core.
- (c) LHS puts a **typed** summary (blink/tween/path with sprite and from→to) in the tool result. That is the kind of actionable content the GB10 "moved +3" line had, and it is cheaper than trails, so L3's per-action header should use that classification.
- (d) Untargeted per-turn text has lost elsewhere (rellik13 M66 and the "what changed" notes; Franzen's timeline). So L3 must stay short (target a median of about 100 tokens) and gated on information loss.
- (e) If keyframes are ever tried, choose them by information lost (M76's test, flashes last), not by even subsampling. Generate them from the same frames as the text so the two never disagree.

**Synthesis.**
- Pull access with exact data worked.
- Images, especially composites, did not; one configuration hurt.
- Push of **long** text (the full timeline) was neutral.
- Push of **short, exact, decision-relevant** text (one "wall moved +3" line) has the only positive signal, and it is thin.
- Hence the ordering below:
  1. Make the existing signals truthful (no false NO-OP).
  2. Keep more data pullable.
  3. Only then push a short, gated digest.
  4. Test images last, if ever.

---

## 3. What animation confusion costs today

Data source:
- 5 full Franzen-base runs (control 46.25, temp06, streams14, ledger-r1 52.56, ledger-r2 49.14): 23.4M generated tokens.
- Plus `research/reports/failure-taxonomy.md`.
- The taxonomy has no animation bucket. These numbers were measured from the transcripts for this doc.

- **False NO-OP verdicts.**
  - 45 / 75 / 53 / 51 / 60 of 281 / 291 / 298 / 247 / 309 NO-OP echoes were on actions whose animation hid cells. **Pooled 284 / 1,426 = 20%.**
  - This is a lower bound: it sees only the last action of a call, and only actions with transient cells.
  - The echo `[action] X -> NO-OP (executed, no change inside the board area)` comes from `_action_verdict` (`solver.py:187-207`), which looks only at the final board.
- **The correction arrives late and is half-used.**
  - The next opener's "not a no-op" line fires 50–77 times per run.
  - The model reads the animation in python in that step only **153 / 296 times (52%)** (16% after a plain notice).
- **Guards fire on animated actions.**
  - 12 of 79 batch no-op stops (15%) halted on an action that animated.
  - The stale-state latch (`python_tool_sandbox.py:546-567`) arms on any `gameplay_changed=False`, animated or not: 269 `StaleStateActionError`s in 7 runs. The animated share was not split out.
- **Lost animations.**
  - 51 thinking passages in 5 runs (about 10 per run) say the animation was overwritten or no longer available: sp80 10, sc25 7, bp35 6, g50t 4.
  - Batch-internal animations are never surfaced at all, so their loss can't be counted from transcripts.
- **UNDO to re-read is rare.**
  - There are 85 X→UNDO→X triples, but 63 are sk48 keyboard probes.
  - Only one explicit re-read was found (control su15).
  - PR #40's motivating su15 story is real but marginal.
- **Animation API errors.** `KeyError 'frames'` ×3; `AttributeError` on `_LazyAnimation.items` (in the sandbox-robust bucket).
- **Deaths.**
  - 16–23 per run; died attempts hold 14.5% of actions.
  - 79% are in attempts of 40 or more actions, which looks like the step bar running out, not animation.
  - The fatal-animation line already fires on about 60% of deaths (tu93 41, bp35 11, su15 9).
  - Little animation-specific upside here.
- **Where it concentrates.** Pooled over 5 runs (mean score; animated NO-OPs / all NO-OPs; share of game tokens in steps that read the animation):

  | game | score | animated / all NO-OPs | tokens in animation-reading steps |
  |---|---|---|---|
  | tn36 | 26.2 | 65 / 150 | 52% |
  | sp80 | 17.1 | 26 / 31 | 69% |
  | su15 | 23.6 | 35 / 66 | 23% |
  | lf52 | 6.0 | 32 / 130 | n/a |
  | s5i5 | 33.9 | 28 / 61 | 8% |
  | dc22 | 38.6 | 24 / 162 | n/a |
  | sc25 | 42.9 | 22 / 119 | 31% |
  | cd82 | 69.1 | 17 / 151 | n/a |
  | ls20 | 28.1 | 15 / 43 | n/a |
  | bp35 | 2.6 | 5 / 22 | 50% |

  No "not a no-op" notices at all: ar25, lp85, re86, tr87, m0r0, g50t, tu93, vc33, sk48. These are the **null games**; the arm should leave them unchanged.
- **Current text sizes.**
  - Animation opener line: median about 44 tokens. Over 400–520 lines that is about 20–25k tokens per run.
  - Timeline and frame reads: 125–190 tokens median per call, about 25–45k per run.
  - Both are small next to about 10.5M uncached prefill per run.

**Sizing.**
- About 57 false NO-OPs and 10 lost-animation moments per run, concentrated in 10 games averaging about 29 points.
- The ceiling is a few levels in those games. A realistic effect is **+0 to +3 offline**, which is at the noise floor. Hence the repeat rule and a pre-registered animation-game subset (§8).

---

## 4. PR #40 (`arm/animation-digest`, 16 commits, +2,785/−29) review

**What it is.** A single master flag `ARC3_ANIM_DIGEST` (needs `ARC3_ANIMATION`) turns on:
- (a) a solver ring of the last `ARC3_ANIM_RING=8` animated actions, stored as hex rows with at most 32 boards per action (`solver.py` `_animation_ring_record`, `_ring_lookup`);
- (b) `animation_for(n)`, `animations`, and `.items()/.values()` in the sandbox;
- (c) a per-batch `animations` list in the action result, at most 8 entries, placed last so truncation cuts it first;
- (d) a static `ANIMATION_DIGEST_ADDENDUM` in the system prompt;
- (e) an opener **digest**:
  - Gate `novel`: a signature (action name without coordinates plus per-step change counts and colour census) not yet shown on this level; repeats become one-line pointers.
  - Shows at most 3 actions (first, peak-transient, last), each with at most 3 "trails" (same-colour blobs not in the settled board, positions over time) of at most 8 points.
  - Capped at 900 characters per digest and 12k per game.

Also: `ARC3_ANIM_KEYFRAMES=N` (real intermediate boards at 4×, about 66 tokens each, never stripped) and a separate `ARC3_ANIM_NOOP_EXEMPT` (the batch no-op stop lets transient actions through).

**Sound?** Mostly yes.
- **Cache safety holds by construction.**
  - New text goes only into the static system prompt (constant per flag set), the new tail opener, or the new tool result. Nothing earlier is rewritten. This is the opposite of Wang's stripping.
  - Verified in the code: the turn ends at the acting python call (`tool_agent.py:7093-7105`), so exactly one opener follows each acting call.
  - The digest lives in the non-stale branch of `_build_user_prompt` (`:4826-4881`), so a **yield resumption** (`_resume_after_yield`, stale summary) carries no digest and no repeat.
  - `_anim_digest_cache` keyed on the summary object means a rolled-back turn rebuilds the same text without charging the game cap again.
- **Host cost is small.**
  - Measured niced on this laptop with PR #40's module: summarize, cap+hex and signature at 5 / 32 / 111 states (synthetic 64×64 chains with 3 moving blobs) take about 2.6 / 14 / 29 ms; the digest build takes 2.4 / 12 / 12 ms.
  - The PR's own fuzzing gives 1–221 ms per digest.
  - That is negligible against a 30 s tool timeout and multi-second requests.
  - Memory: 8 records × 32 hex boards is about 1–2 MB per game.
- **Tests.** 59 tests: a flags-off snapshot against the pre-digest tree, ring eviction, sandbox API, opener placement, and 3k fuzz chains. They use a real sandbox and a real `_HarnessGameSession` driven by `tests/solver_fakes.FakeGame`. **None go through `ToolAgent.analyze()`.** The flags-off fixture was taken against `arm/kaggle-builder`, not `main`.

**Defects and design problems to fix in the rebase.**
1. **One master flag bundles 4 mechanisms.** The ring (pull) and the digest (push) can't be A/B'd separately. Franzen's history says push is the risky half. Split it (§5).
2. **The tool-result NO-OP lie is not addressed.** The same call still echoes `NO-OP` for a hiding animation (`solver.py:187-207, 227`). This is the largest measured cost (20% of NO-OP echoes).
3. **The stale-state latch is not exempted.** Only the batch stop is (`ARC3_ANIM_NOOP_EXEMPT`).
4. **The per-batch `animations` list is noisy on tween games.** It lists every animated action, including pure motion with no transient cells. On games where every move animates, every batch carries up to 8 entries of about 25–30 tokens, up to about 240 tokens of tool result per call. Keep only entries that hid something or left the board unchanged, and give a count for the rest.
5. **`last_animation_frames` semantics vs the prompt.** `main` clears the record on a non-animated action, but the prompt says "of the last executed action that animated". With the ring on, make the `last_animation_*` views read the newest ring entry, so the existing prompt sentence becomes true.
   - The timeline header already carries `action`; also add `action_num`.
   - The opener's "not a no-op" pointer becomes correct in every case.
   - Under flags-off, `main` stays as is: byte identity.
6. **The digest's trails are a third motion heuristic** next to `frame_diff` and the timeline. Unlike #41's matcher, they claim no identity ("seen at"), so they are safe to ship.
   - But the GB10 positive came from **object motion with a delta** ("moved dc +3"). The PR leaves that as "lens B, not implemented".
   - Lens B needs the #41 fix first.
7. **Keyframe images are never stripped.** Each costs about 66 tokens of history until the next trim (4 per digest is about 264). The one image result we have (Franzen's composite) is negative. Keep them off by default and test last.
8. **The flags-off fixture is against `arm/kaggle-builder`** (code-identical to `main`, see §6). Regenerate it against `main` and add an `analyze()`-level byte comparison.
9. **Document the animation record's keys.** `frames`, `transient_pixels`, `transient_bbox`, and the ring entry keys go in the addendum; 3 `KeyError 'frames'` were seen. This goes in the flag-on addendum only.

---

## 5. Recommended combined arm

One branch, `arm/anim-combo`, from PR #40 rebased onto `main`. Each layer has its own default-off flag. Every layer needs `ARC3_ANIMATION=1` (already on in the Kaggle config). Ranked by expected score per unit of compute, most to least:

### L1. `ARC3_ANIM_TRUTH=1`: truthful verdicts (new; about half a day)
- **`_action_verdict`.** For `gameplay_changed is False` and `animation.transient_pixels > 0`, return `ANIMATED (board area unchanged; N frames, T cells changed and changed back - see animation_for(<n>))` instead of `NO-OP ...`.
  - Without the ring, point at `last_animation_timeline`.
  - It is shared with the per-action trace, so batches get the same word.
  - Cost: about +12 tokens on about 57 actions per run, about 0.7k tokens per run.
  - Optional, cheap: append LHS's type word (`blink`, `tween`, `path`, `other`; §2.3), computed from the same chain. This is still under 20 tokens.
- **Stale latch.** Don't arm it on such an action (`python_tool_sandbox.py:563`). That needs the `animation` key in the action result, which is already there.
- **Batch no-op stop.** Fold PR #40's `ARC3_ANIM_NOOP_EXEMPT` in here: an action that hid something is not a no-op.
- **Still a no-op:** an action whose animation had **no** transient cells and left the board unchanged.
- Opener and system prompt are unchanged, apart from one static clause naming the `ANIMATED` verdict, added only under the flag.
- **Why first.** It is the biggest measured error (20% of NO-OP echoes), it costs almost nothing, it is cache-neutral, and it makes Franzen's own "not a no-op" line consistent with the tool result.
- **Risk.** It could keep a batch running on an animated non-effect. The batch stop then costs at most the rest of the batch; the old behaviour stopped there.

### L2. `ARC3_ANIM_RING=8`: keep recent animations pullable (PR #40 a, b, d-part, plus fixes 4, 5, 9)
- Solver ring (8 records, 32 boards each, hex).
- Sandbox `animation_for(n)` and `animations`.
- The `last_animation_*` views read ring[-1] (fix 5).
- The per-batch `animations` result list holds only hiding or board-unchanged entries (fix 4).
- Static addendum (about 150 tokens): S0/step-k indexing, `animation_for`, "re-reading costs no action", record keys.
- **No opener text.**
- Cost per turn: 0 opener tokens, and tool-result tokens only on batches with ≥2 hiding animations, capped at 8 entries, ≤240 tokens.
- **Why second.** It extends exactly the mechanism Franzen found strongest (pull) and fixes the about 10 "overwritten" moments per run and the batch-internal loss. The ring is host-side, so it **survives history trims**: after a trim the model can still pull the last 8 animations, which no history-based scheme can do. rellik13's M111 arrived at the same shape on its own (last 8 animated actions, ≤32 frames, hex, nothing in the turn message), after seeing the model ask for frames it no longer had (§2.3).

### L3. `ARC3_ANIM_DIGEST=novel`: Wang-style proactive digest (PR #40 e)
- Needs L2. Gate `novel` (or `transient`).
- At most 900 characters (about 225 tokens) per digest and 12k (about 3k tokens) per game, so at most about 75k tokens per run (0.7% of prefill), usually much less.
  - Run it at `ARC3_ANIM_DIGEST_CHARS=600` with a median target of about 100 tokens. Per-turn text has lost elsewhere (rellik13: 6.42 → 4.16 from more prompt text; per-action "what changed" notes failed; §2.3).
- Placed right after Franzen's animation line, in the new opener only.
- Make each action's header the LHS-style **typed** line (blink with cycles; tween with sprite hash, colour and from→to; path with ≤8 points), ahead of the trail lines. It is the actionable fact type and the cheapest line to keep when the cap cuts trails.
- **Add lens B before the arm runs**, or run the digest arm without it and add lens B as `ARC3_ANIM_DIGEST_MOTION=1`. Lens B is one exact "object moved dr,dc" line from `frame_diff` over S0→S_peak and S_peak→S_n, after the #41 fix. It is the GB10-proven fact type.
- **Why third.** It is the only part with a negative prior (Franzen: auto timeline, no gain). The positive GB10 signal is thin, from one game on another harness. It must beat L1+L2, not the control.

### L4. `ARC3_ANIM_KEYFRAMES=0`: real keyframe images (PR #40, off)
- Up to N real intermediate boards at 4× (about 66 tokens each) next to a digest.
- **Run only if L3 is at least neutral**, and as a separate arm.
- The prior is negative:
  - Franzen's composite images hurt, and Wang's and rellik13's M76 images were never isolated.
  - VLM change detection is weak (§2.3). At 4×, one Qwen token spans 8×8 cells.
- If it does run:
  - Pick frames by **information lost** (M76: a frame that differs from the pre-action board more than the final board does; whole-board flashes last), replacing PR #40's even subsample plus peak.
  - Send separate images, never Qwen video input.
  - Build them from the same kept frames as the digest text, so the two can't disagree (VLMs side with text).

### Separate prerequisite PR: fix #41 (`frame_diff` recolour-as-move)
- Accept a `moved` entry only if its vacated cells became background. The prototype rejected 8/8 false moves and kept 3/3 true ones.
- This changes `frame_diff()` output for the model under **current** flags, so it is a behaviour change. Put it behind `ARC3_FRAME_DIFF_STRICT_MOVES=1`, in its own small arm (or ride with L2).
- Needed before lens B.

### Cross-cutting requirements
- **Flags off:** system prompt, every request body, sandbox globals and tool results are byte-identical to `main` (§7).
- **Prefix-cache safety:**
  - New text is appended once, in the newest user or tool message, and never edited later.
  - Each layer's system addendum is a constant string chosen at startup.
  - Nothing is stripped from history. That rules out Wang's `_drop_animation_parts`, which rewrote earlier messages every turn; keep that out.
- **Static system prompt:** no per-game or per-level text in the system prompt.
- **Token cost per turn:** at most +12 (L1) per false NO-OP, +0 to 240 (L2 tool result), +0 to 225 (L3 opener), averaged over turns well under 1% of prompt tokens. Measure it: digest chars and `animations` entries per call get logged (`ANALYZER STATUS` line or viewer event).
- **Yields:** digests appear only in non-stale openers. A yield resumption (`ARC3_YIELD_RESUME_PROMPT=full` default) rebuilds the opener with a stale summary and carries no digest; the original opener with the digest stays in history. If yield-6k (failure-taxonomy proposal) lands, nothing changes: digests are per acting call, not per request.
- **Trims:**
  - Digests and `animations` lists live in messages and are trimmed with them.
  - The ring and per-game caps live on the host and the agent, and are unaffected.
  - The digest game cap is not reset by a trim, which is deliberate: it bounds total cost.
  - After a trim the ring is the only way back to a pre-trim animation. Mention `animation_for` in the L2 addendum, not in a post-trim message, to keep the cache shape.
- **Ledger-on-trim and memory-v2:**
  - The ledger is shown only right after a trim, and the arm adds nothing to it.
  - If memory-v2's action-effect table stacks, it must classify via the same verdict as L1 (`ANIMATED`, not `no-op`), or the two will disagree. Put the shared verdict in `solver._action_verdict` so both read it.
  - Death ledger and known-no-op guard (`ARC3_NOOP_REPEAT_GUARD`, `ARC3_DEATH_LEDGER`; both off in the Kaggle config): if either is turned on, an L1-animated action must not be filed as a known no-op. Check `_noop_repeat_guard` input.
- **Interaction with `ARC3_DIFF_IMAGE`:** unchanged. The diff image covers final boards and the animation layers cover intermediate ones.

---

## 6. Rebase PR #40 or rebuild?

**Rebase. Don't rebuild.**
- PR #40's base is `4d63980` (`arm/kaggle-builder`, merged into `main` as #34). Between `4d63980` and `origin/main`, **no file under `ARC3-Inference/`, `kaggle/` or `serving/` changed**: only `README.md`, `tools/phone_viewer.py`, `runs/` and `research/`.
- So `git rebase origin/main` (or retargeting the PR base to `main`) has **0 conflicts** (checked with `git merge-tree`). The code is current.
- Stacking on the submission candidate (`arm/throughput-stack`, the `tput-safe` arms) gives **3 small conflict hunks**, checked the same way. `arm/stack-v2` and `arm/memory-v2` also give 3, `arm/sandbox-robust` 4, and `arm/facts-stack` 5. The 3 are:
  - `python_tool_sandbox.py` `_animation_fetch`: throughput-stack added `_forward_stdout(stdout)` at its top, so keep that line in PR #40's `_animation_fetch(kind, **extra)`.
  - `tool_agent.py` near `_repeat_state_seen.clear()`: keep both the `WALL_CAP_FACTOR` import and the `_anim_call_*` resets.
  - `kaggle/arms.json`: keep both sets of arms.
- Rebuild cost would be 2–3 days to redo tested code (ring, sandbox API, digest, snapshot tests). The rebase plus changes is about 1.5 days.

**Steps**, one logical change per commit, imperative messages:
1. `git checkout -b arm/anim-combo origin/arm/animation-digest && git rebase origin/main` (or keep the `arm/animation-digest` name and retarget PR #40 to `main`).
2. "Split the animation digest flag into ring and digest": `ARC3_ANIM_RING` alone gives the ring, the sandbox API and its addendum; `ARC3_ANIM_DIGEST` needs it. Split `ANIMATION_DIGEST_ADDENDUM` into ring and digest constants. Refactor only; with `ARC3_ANIM_DIGEST=1 ARC3_ANIM_RING=8` the bytes are the same as today.
3. "Read the newest ring entry from last_animation views" (fix 5).
4. "List only hiding animations in the action result" (fix 4).
5. "Report animated no-ops as ANIMATED in the echo and trace" (L1, `ARC3_ANIM_TRUTH`), folding in `ARC3_ANIM_NOOP_EXEMPT`.
6. "Leave the stale-state latch unarmed after a hiding animation" (L1).
7. "Document animation record keys in the ring addendum" (fix 9).
8. "Pin flags-off requests to main through analyze()" (test, §7).
9. "Add the animation loop simulation" (test, §7).
10. Docs: `CONFIGURATION.md` rows for every flag, `kaggle/arms.json` arms, `README.md` row, `kaggle/README.md` if a new panel is added.
- #41 goes in its own branch, `arm/frame-diff-strict`, off `main`, then merges into anim-combo before lens B.

---

## 7. Test plan

Every heavy step runs niced and one at a time (laptop rule). The model is never needed.

### 7.1 Unit tests with the real sandbox (`tests/`, carry PR #40's 59 tests and add these)
- **L1:**
  - A hiding animation echoes `ANIMATED (...)` with the right N and T. A tween with no transient cells still echoes `NO-OP`.
  - A batch trace position uses the same word.
  - The stale latch is not armed after `ANIMATED`, and is armed after a real no-op (`StaleStateActionError` on re-issue).
  - The batch stop passes `ANIMATED` and stops on a real no-op.
  - With `ARC3_NOOP_REPEAT_GUARD=1` (off in the Kaggle config), the known-no-op guard doesn't file `ANIMATED` actions.
- **L2:**
  - `animation_for(n)` for an action from **two python calls ago**, and after 9 animated actions (ring eviction: the oldest gives an error naming the kept numbers).
  - `last_animation_frames` after an animated then a non-animated action returns the animated one's frames (flag on) and `[]` (flag off).
  - The `animations` result list holds only hiding entries; a tween-only batch adds no key.
  - Indexing: `frames[k-1]` equals timeline step k on a known chain.
  - Death: the fatal animation is readable after the auto-RESET via both `last_animation_*` and `animation_for`.
- **L3:**
  - The PR #40 digest tests: gate, caps, pointer validity, level-change reset, subsampled pointer.
  - Lens B (if built): a rigid slide gives `moved dr,dc`; a tn36-style sweep gives no move (#41).
- **#41:** the issue's sweep (0 moves) and rigid-slide (3/3 kept, correct dr/dc) tests.
- **Host cost:** a pytest-benchmark-style assertion that summarize, ring record and digest at 111 states stay under 100 ms. Measured today: about 41 ms.

### 7.2 Loop test through `ToolAgent.analyze()` (`tests/anim_loop_sim.py` + `tests/test_anim_loop.py`)
Modelled on `arm/reasoning-scaffold`'s `tests/scaffold_loop_sim.py`: a fake SGLang renders every request through `tests/fixtures/qwen38_flash_next_chat_template.jinja`, under the Kaggle notebook's `setup_env` (exec'd from `kaggle/template.ipynb`, priority scheduling on). **Two differences, both deliberate:**
1. **Real game session.** `analyze()` gets `step_env=session.step_env` from a real `_HarnessGameSession` over `solver_fakes.FakeGame` (PR #40), with a script that returns multi-frame chains:
   - (a) a click whose highlight flashes and reverts (hiding, board unchanged);
   - (b) a 4-frame tween move (no transient cells);
   - (c) a batch of 3 hiding clicks plus a move;
   - (d) a fatal action with an animation, then auto-RESET;
   - (e) a level-up in mid-batch;
   - (f) more than 8 animated actions over the game.
2. **Real python tool.** Don't monkeypatch `_run_python_tool`. Scripted replies send python code that is really executed in the sandbox: `action([...])`, `print(animation_for(n)['timeline'])`, `last_animation_frames[k-1].ascii`.

One reply reports 3,000 completion tokens to force a 2,048-token **yield** and a resumption. Set `LOCAL_ANALYZER_CONTEXT_WINDOW` and `ARC3_CONTEXT_DRAIN_TOKENS` small in one config to force a **trim**, and in another enable `ARC3_LEDGER_ON_TRIM=1` when run on a stack base.

Assertions per config (`off`, `truth`, `ring`, `digest`, `digest+keyframes`):
- **Prefix property:** every request's rendered prompt starts with the previous request's full prompt plus its generated reasoning, across turns and the yield resumption, except the first request after a trim (and the ledger message after it). This is the cache-safety proof.
- **The system prompt is identical** across all requests of a game.
- **The digest appears exactly once** per qualifying acting turn, never in a resumption opener, and never in a tool message. The per-game cap holds.
- **Echo:** the tool result for (a) contains `ANIMATED`, not `NO-OP`, under `truth`; `NO-OP` under `off`.
- **The ring survives the trim:** a post-trim reply calling `animation_for(n)` for a pre-trim action gets frames.
- **The game plays through** under every config: same final level, same action count, where the script allows.
- **Flags-off vs `main`, byte for byte:** with `ANIM_SIM_BASELINE_ROOT=<main checkout>/ARC3-Inference`, the `off` run's rendered prompts, request JSON bodies (images hashed), tool results and sandbox globals equal `main`'s. Repeat with the competition env and with `ARC3_ANIM_*` knobs set while the layer's master flag is 0. On a stack base, compare against that base's tip instead.

### 7.3 Mutation checks (each must make at least one test fail; record the killing test in the PR)

| # | Mutant | Expected killer |
|---|---|---|
| M1 | Append the digest to the system prompt instead of the opener | Prefix property, static system prompt |
| M2 | Strip digests and keyframes from earlier history (Wang's `_drop_animation_parts`) | Prefix property |
| M3 | Drop the `_anim_digest_enabled()` / `ARC3_ANIM_TRUTH` guard (layer always on) | Flags-off byte identity |
| M4 | Build the digest from a stale summary too | "never in a resumption opener" |
| M5 | `ANIM_RING_SIZE = 1` | `animation_for` two calls ago, ring eviction |
| M6 | Revert `_action_verdict` to always `NO-OP` | L1 echo test |
| M7 | Arm the stale latch after `ANIMATED` | L1 latch test |
| M8 | Clear `last_animation_*` on a non-animated action with the ring on | fix-5 test |
| M9 | Remove #41's vacated-cells filter | sweep test |
| M10 | Don't charge `_anim_game_chars` | per-game cap |
| M11 | Off-by-one: `frames[k]` = step k | indexing test |

Run them by hand from a scratch worktree, one at a time and niced, with a small `scripts/mutate.sh` that applies a `.patch` per mutant.

### 7.4 Pre-push checks
- `python kaggle/build_notebook.py --arm <arm>` builds locally.
- The embedded patch equals the branch diff, and every arm env var is in `setup_env`.
- `ruff`, the full `ARC3-Inference` suite and `kaggle/tests`, niced, one at a time.

---

## 8. Eval plan (judged at matched wall-clock)

**Owner's rule:** compare at **wall-clock**, not runtime clock (scratchpad, 2026-10-08). Every comparison here uses the same 121-minute Kaggle wall budget on the 25 offline games. Any extra tokens or host time an arm spends is already charged in its score, because both arms get the same minutes.

**Arms** (`kaggle/arms.json`, all on the same commit so the control differs only by env):
- `control`: the flags-off build of `arm/anim-combo`. It is byte-identical to `main`, so an existing same-config control may be reused per the owner's 2026-10-08 note. Prefer a same-batch control if quota allows.
- `anim-truth`: L1.
- `anim-ring`: L1+L2. **This is the primary candidate.**
- `anim-digest`: L1+L2+L3 (gate `novel`).
- `anim-digest-kf`: plus `ARC3_ANIM_KEYFRAMES=4`, only if `anim-digest` is not worse.

**Order and gates:**
1. **Smoke** (20 min, not a verdict): `panel-anim-ring` and `panel-anim-digest` on a pre-registered **animation panel** of the 10 games with notices (tn36, sp80, su15, lf52, s5i5, dc22, sc25, cd82, ls20, bp35), 4 streams. Brüggen's "type 1" games (ft09, sb26: first frame equals last, so everything happens in between) are the obvious swap-ins if a slot frees up.
   - Checks: no errors; digest and echo fire; `animation_for` gets called; serve_report tok/s and prefix-cache hit rate within noise of control. A drop in cache hits means a cache-safety bug, so stop.
2. **Full:** `anim-ring` vs control, **2 paired runs each** (`push_queue.sh anim-ring anim-ring@2`, staggered).
   - `make significance BASELINE_SCORE=<control score json> CANDIDATE_SCORE=<arm score json>` per pair and pooled. `significance.py` checks that `max_runtime_minutes_per_game` matches, which enforces equal budgets.
3. Only if `anim-ring` holds up in both: `anim-digest` vs `anim-ring`, 2 paired runs. The digest must beat the ring, not the control.
4. If it is positive, stack the winner on `tput-safe` (throughput-stack) and compare against `tput-safe` at matched wall-clock.

**Wall-clock reporting** for each run, from `levels.csv`, `benchmark.json` and the transcripts. Write it into `arms.md`:
- Score at t = 30, 60, 90 and 121 minutes (the cumulative level-clear curve), so a slower-but-better arm is visible. The 121-minute point decides.
- Seconds and tokens per cleared level; generated and prefill tokens per run; serve_report tok/s; prefix-cache hit rate.
- Mechanism counters:
  - false-NO-OP echoes (target 0 under L1);
  - "overwritten / no longer available" thinking mentions (target about 0 under L2);
  - `animation_for` calls;
  - digests fired and their mean chars;
  - `StaleStateActionError` and batch stops on animated actions;
  - UNDO→re-click.

**Decision rules** (pre-registered):
- **Promote** an arm only if:
  - both repeats beat their paired control on the 25-game mean;
  - the pooled gap is ≥ +3 (the noise floor) or `make significance` is confident;
  - **and** the gain sits mainly in the animation subset while the null games (ar25, lp85, re86, tr87, m0r0, g50t, tu93, vc33, sk48) are within noise. A gain on the null games is luck, not mechanism.
- **Drop** if either repeat is below control by ≥3, or the null games lose (the "hurt some games" pattern Franzen saw with images).
- A single run, or a 20-minute panel, only decides whether to spend the repeats (owner's rule).

**Budget.**
- Smoke: 2 × 20 min.
- Full: 4 × 121 min for ring vs control, plus 4 × 121 for digest vs ring.
- That is about 16.7 GPU-hours if every gate passes, at 1 run at a time and 2 sessions maximum.

---

## 9. Open questions
- **tn36's zero runs** never triggered an animated no-op at all. The model wrote "no 'animation' key → no animation". None of L1–L3 helps if the model never takes the revealing action. A `gate=all` digest would surface the tween animations it did trigger. Whether that steers exploration is untested; leave it as a later variant.
- **Is L3's object-motion line (lens B) the active ingredient** of the GB10 result rather than "proactive" per se? If lens B ships, consider `ARC3_ANIM_DIGEST_MOTION` as its own arm.
- **Wang-style subsampling** (8 frames) vs PR #40's 32-board ring cap. Long animations (sb26, up to 111 raw frames) are subsampled in the ring but kept whole in `last_animation_*`. Check that sb26 timelines still read correctly via `animation_for`.
