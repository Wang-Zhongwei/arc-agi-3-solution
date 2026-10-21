# animation-merge-design: design

Workflow run `wf_5ce3877c-55b`, phase Design. Compare Wang vs Franzen animation-frame mechanisms with evidence, then design a merged mechanism

---

# Merged animation mechanism for Franzen's harness: "Animation Digest"

All paths are under `daniel-franzen-sol/ARC3-Inference/inference/` unless noted. I re-read the current working tree, so the `tool_agent.py` line numbers below run about 60 higher than in the input analysis. For example, the opener is at 4932, not 4869.

## 0. Corrections and new findings
- **Commits are unverifiable for Franzen.** `daniel-franzen-sol` is not a git repo, so `af58714` and the other Franzen commits come from the input analysis only. I spot-checked Wang's `main` code (solver.py ~120-170, `_action_animation_frames`; vision_context.py ~75-125), and it matches the input analysis.
- **Provenance.** `WRITEUP.md:183` credits Jakob Brüggen's notebook, not Wang's. Wang's notebook has no animation code, and the mechanism is Wang's `main` solver and harness (commit `127970d`).
- **A multi-cell slide DOES trigger Franzen's hint.**
  - `animation.py:10-16` says pure motion gets no hint. A 2x2 block sliding 3 cells (synthetic) gave `{'frames': 3, 'transient_pixels': 2, 'transient_bbox': [20, 12, 21, 12]}`, so the hint fires.
  - The rule (`animation.py:73-93`) counts any cell crossed twice.
  - Only slides of at most one object width stay silent.
- **`frame_diff` mislabels recolors among identical objects as moves.**
  - Synthetic tn36-like sweep: a 3x3 `b` highlight stepping over five 3x3 `G` tokens.
  - `compute_frame_diff` reported "G object moved (30,26) > (30,20) > ... net dc +12". That is false: the true change is two in-place recolors per step.
  - Cause: hash matching runs before the `changed_color` phase (`frame_diff.py:149-175` vs 198-212). This is the ambiguity `prompts.py:233` warns about.
  - A cheap soundness filter fixes it: accept a `moved` entry only if the cells it vacated (`from` box minus `to` box) became the most common colour (background) in the next frame. The filter rejected 8 of 8 false moves on the sweep and accepted 3 of 3 true moves on the slide.
- **Digest size.** For the sweep, the existing timeline `repr` is 1,281 chars. The digest below is 298 chars.
- **Segmentation timing.** Cost was 40 ms for 4 grids and 96 ms for 7 grids, on synthetic near-empty boards. I did not measure real boards.
- Prototype scripts: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/proto*.py`. They run on synthetic boards, not real tn36 data.

## 1. What each side contributes
| Keep from Wang | Keep from Franzen |
|---|---|
| Proactive delivery at the next turn. The model need not know to ask, which addresses "info only on demand". | Pre-action board prepended (`animation.py:50-70`), so X to Y to X cells register. |
| Adjacent dedupe before the cap, first/last kept (Wang solver.py ~156-167). | Transient-pixel detection as the gate (`animation.py:73-119`). |
| Cap of 8 frames, because raw lengths run "2 frames in sk48, 15 in cd82, up to 111 in sb26" (Wang vision_context.py ~99-101). | Exact cell text with caps of 24 per step and 80 total (`animation.py:37-38`, 311-331). |
| Exact image pricing: 66 tokens at 256x256. Franzen's trimmer agrees: 32px patches plus 2 sentinels, 402 at 10x (`dfranzen-fork.txt:4031-4033`). | REPL-computable frames and timeline, free to read (`python_tool_sandbox.py:570-640`, `prompts.py:179-185`). |
| Separate real frames, not fabricated composites. | Death animation preserved (`solver.py:1279-1290`, `tool_agent.py:5057-5065`). Wang drops WIN/GAME_OVER animations. |

| Negatives and the fix |
|---|
| **Only the last record is kept** (`solver.py:495`, 1291). Fix: a ring of 8 plus per-action summaries for the batch. |
| **Only the last action of a batch** (`tool_agent.py:3837`). Fix: `animations` list of all animated actions in the snippet. |
| **No hint for tiny motion, and the hint is a pointer only.** Fix: auto digest, gated. |
| **Auto-timeline gave "no clear gain"; composites "hurt"** (`WRITEUP.md:301`). Fix: show once per novel animation, never a fabricated composite, hard char caps. |
| **Frame/step off-by-one** (`animation.py:249-276`, frames[i] = timeline step i+1). Fix: state the mapping in the prompt and the digest footer. |
| **Wang eyeballs frames and mutates history** (Wang `tool_agent.py:456-489`, 1664-1673, per input analysis). Fix: text only, appended once and never rewritten. |

## 2. Flags (all default off; unset means byte-identical to the control)
- `ARC3_ANIM_DIGEST=0|1` is the master flag. It needs `ARC3_ANIMATION=1` (`solver.py:115`, `tool_agent.py:2425`).
- `ARC3_ANIM_DIGEST_GATE=novel|transient|all`, default `novel`.
  - `transient` = `transient_pixels>0` or `gameplay_changed is False`.
  - `novel` = `transient` AND an unseen signature.
  - `all` = any animated action, with the object-move lens enabled.
- `ARC3_ANIM_DIGEST_CHARS=900` per digest. `ARC3_ANIM_DIGEST_GAME_CHARS=12000` per game.
- `ARC3_ANIM_RING=1` is the default (today's single record). The arm sets 8.
- `ARC3_ANIM_NOOP_EXEMPT=0|1` is a separate arm.
- `ARC3_ANIM_STRIP=off|keyframes` is a separate, later arm. It needs `MULTIMODAL_CONTEXT=current_grid`.
  - Companions: `ARC3_ANIM_STRIP_FRAMES=4`, `ARC3_ANIM_STRIP_UPSCALE=4`.
  - The upscale default follows Wang `4b174e7`, `min(board upscale, 4)`.

## 3. Implementation map
1. **`utils/animation.py`** gets new functions that reuse `collapse_frames`, `summarize_animation`, `_transient_cells`, `_diff_cells`, `_color_char`, `_bbox_phrase` and the caps.
   - `animation_signature(chain)`: sha1 of the action name without coordinates, plus per-step `(n_changed, sorted colour-transition census)`. Coordinates are excluded so the same mechanic at a new cell still matches.
   - `select_keyframes(chain, k=4)`: dedupe is already done by `collapse_frames`. Take an even subsample with first/last kept, as in Wang solver.py ~163-167, and add the peak transient step.
   - `overlay_trails(chain, border)` is lens A, for blinks, sweeps and recolors.
     - Per step, take the components of cells that differ from the FINAL frame and are not background.
     - Mask the border with `NOOP_GUARD_BORDER_WIDTH` (`solver.py:~177`), as `vision_context.py:189-192` does.
     - Chain components of equal colour and size into a trail.
   - `object_moves(chain, bg)` is lens B, for rigid motion.
     - It calls `frame_diff.compute_frame_diff` on keyframe pairs, with `segment_layer` host-side exactly as at `tool_agent.py:4119-4124`.
     - It applies the vacated-cells-are-background filter and chains moves by hash into a trail with net dr/dc.
     - It drops `resized` entries covering more than 25% of the grid (the background component showed up in the slide run).
   - `build_animation_digest(records, budget_chars, border, seen)` returns `(text, new_signatures)`.
     - Router: lens B only for `gate=all`, or when lens B accepts at least one move. Otherwise lens A.
     - It lists at most 3 actions: the first animated, the one with the most transient cells, and the last. Each gets at most 3 trail lines and 8 positions, with a "(+k more animated actions; `animation_for(n)`)" tail.
2. **`framework/solver.py`**
   - At 1279-1298, also push `{action_num, action_display, summary, signature, level, frames}` onto `self.animation_ring = deque(maxlen=ARC3_ANIM_RING)`.
     - `summary`/`signature` are computed on the full chain.
     - Only the stored `frames` are capped at 32 with the first/last-preserving subsample, as `research-sweep.md:17` proposes, and stored as hex rows.
     - `animation_record` stays equal to the last record.
   - At 1002-1008, `query:"animation"` accepts `action_num` or `ring:true`. It still spends no action.
   - Around `final_payload` (~1165), add `payload["animations"] = [{action_num, action_display, animation}, ...]` for every executed payload with `animation`.
   - At 1135-1141, with `ARC3_ANIM_NOOP_EXEMPT=1`, skip the no-op stop when `payload["animation"]["transient_pixels"]>0`. This addresses `research-sweep.md:18`, where the opener says "not a no-op" but the guard stops the batch.
3. **`agent/tool_agent.py`**
   - `_build_system_prompt` 2717-2720: append `ANIMATION_DIGEST_ADDENDUM` (new, in `prompts.py` after 188) only when the flag is on.
   - Around 3837: add `"animations"` to the summary next to the existing `"animation"`.
   - Opener at 4932-4945: after `animation_line`, call `build_animation_digest` and append the block to `lines`. Pull `seen` from `self._anim_seen_sigs`, reset on level change (the same condition as `_level_transition_diff_lines`, ~4056). Charge `self._anim_game_chars`.
   - Fatal case at 5057-5065: same builder with `game_over=True`.
   - `_handle_animation` 5831-5840: kinds `"for"` and `"ring"`.
4. **`agent/python_tool_sandbox.py`** at ~570-640:
   - Add `animation_for(n)` returning `{frames, timeline}` via the same `_animation_fetch`.
   - Add `animations`, a lazy list of `{action_num, action, frames, transient_pixels}` with no grids.
   - Add `.items()` and `.values()` to `_LazyAnimation` (gap noted at `research-sweep.md:27`).
   - Keep invalidation at 510.
5. **`agent/vision_context.py`** (strip arm only): add `animation_keyframes_parts(chain, k, upscale)`, modelled on `animation_peak_part` 240-290. Plain 16-colour real frames, no dimming.
6. **Tests:** add `tests/test_animation_digest.py` covering dedupe before cap, first/last preserved, signature stability across coordinates, the char caps, and `_build_system_prompt()` byte-equal with flags off.

## 4. What the model sees
Static system-prompt addendum, only with the flag on:
```
- When an animation is novel or hid something, the opener ends with an "Animation digest". S0 is the board before the action and S<n> is current_frame; timeline step k equals `last_animation_frames[k-1]`. Trails list positions in time order. Recent animated actions stay readable for free via `animation_for(action_num)` and `animations`; re-reading costs no game action, so do not spend UNDO or repeat a click just to see it again.
```
Worked example 1: tn36-like sweep, synthetic. The board is unchanged afterward, and the existing opener is 306 chars (`describe_animation`). The digest follows it in the same opener:
```
Animation digest (novel): `MOUSE(row=31, col=21)`: 6 states (S0 = board before; S6 = current board). Board area unchanged. 45 cells changed and changed back.
- b 3x3 seen at (30,20) > (30,26) > (30,32) > (30,38) > (30,44); steps S1-S5
Exact cells: last_animation_timeline; step k = last_animation_frames[k-1].
```
This is 298 chars as prototyped, with the prototype's own header wording.

Worked example 2: rigid slide, synthetic, lens B:
```
Animation digest (novel): `RIGHT`: 3 states. 2 cells changed and changed back.
- Y object (4px) moved (20,10) > (20,11) > (20,12) > (20,13) (net dr +0, dc +3; steps S1-S3)
```
Worked example 3, a repeat of example 1 on the same level (about 100 chars):
```
Animation digest: `MOUSE(row=31, col=33)` repeats the animation of action 212 (same signature); `animation_for(212)`.
```
Digests are persisted in history as plain text.

## 5. Budget and prompt-prefix placement
- **Per digest:** at most 900 chars. At the trimmer's default 3 chars per token (`dfranzen-fork.txt:4036-4045`), that is about 300 tokens at most. The same comment records 4.3 chars/token measured on prose, so the true cost is probably lower.
- **Per game:** 12,000 chars, about 4k tokens. After that only the existing opener line remains.
- **Compute:** at most 4 keyframe pairs, so at most 5 segmentation calls per digest. `research-sweep.md:18` cites 160 ms mean per `compute_frame_diff` (unverified by me), so worst case is about 0.6 s on a rare turn.
- **Cache:** the control prefix is unchanged. The treatment prefix is constant for the whole game, with the addendum fixed text. The digest only ever lands in the new tail user message (`_build_user_message`, 4723-4757), after all cached history, and is never rewritten afterward.
  - This avoids Wang's pattern of mutating the previous user message each turn.
  - It also avoids the prefix change that `ARC3_HISTORY_IMAGE_KEEP` causes (`CONFIGURATION.md:94-96`).
- **Images** stay off by default. The strip arm attaches at most 4 real keyframes at 66 tokens each (264 tokens), placed after the board image like the existing diff image. It is persisted immutably, never stripped. Run it only after the text arm, and verify the vLLM multimodal-cache 400/500 retry loop (Wang `notes/serving/ucb-v05-disk-and-retry-loop.md`) is not present on the serving stack.

## 6. How to measure
- **Arms**
  - A0: control, the competition config (`dfranzen-fork.txt:10215-10228`).
  - A1: `ARC3_ANIM_DIGEST=1`, `ARC3_ANIM_RING=8`, gate `novel`.
  - A2: A1 plus `ARC3_ANIM_NOOP_EXEMPT=1`.
  - A3: A1 plus `ARC3_ANIM_STRIP=keyframes`.
  - Change one thing per arm.
- **Games expected to benefit**
  - tn36 (`WRITEUP.md:181`). Floor: a no-regression check. Wang-era runs reached 2/7 levels (10.71) and Franzen-era 3/7 once (20.32, `fz-streams14`), per the input analysis.
  - ar25: the frame-diff arm cleared level 2 in 2 of 2 runs vs 2 of 8 controls, Fisher p=0.13 (not significant).
  - sp80, su15, sc25, s5i5: animation-heavy per `research-sweep.md:17`.
  - sb26 (up to 111 raw frames) as a cap and cost stress test.
  - sk48 (2 frames) and cd82 (15) as cost controls that should show near-zero digest volume.
- **Metrics**
  - Levels cleared and actions to first clear.
  - Wasted-action rate: UNDO or the same MOUSE click within 3 turns of an animated action. `research-sweep.md:17` claims su15 does this; I could not verify it.
  - Digests emitted per animated action, chars and tokens added per game.
  - Cached-prompt-token fraction, which must stay within 1 point of A0. The reference is the 93.43% reuse Franzen cites in an earlier demo (arms.md).
  - Median seconds per turn, and error counts.
- **Statistics:** per-game variance is large (ar25 went 2/8 to 2/2), so use at least 8 runs per game per arm, with matched seeds where possible.

## 7. Risks and mitigations
- **A coherent digest invites stopping at it.** `tool_agent.py:1852-1857` records a confident, wrong mechanic from a timeline. Mitigations: the digest says trails are heuristic, ends with an explicit pointer to the exact `last_animation_*` views, never states causes, and lens B is filtered.
- **Wrong object-level claims on identical objects.** Mitigated by the vacated-cells-background filter and the lens A fallback. Add a unit test on the sweep case.
- **Auto-insert showed "no clear gain" (`WRITEUP.md:301`).** Mitigated by gating on `novel`, repeats shrinking to one line, and a per-game cap. `ARC3_ANIM_DIGEST_GATE=transient` is the fallback arm.
- **Prefix cache regression.** Mitigated by constant addendum text, tail-only digest, and no history mutation. Measured in section 6.
- **Token blow-up on long animations (sb26).** Mitigated by char caps, the 32-frame stored cap, and 3 actions per digest.
- **Batch confusion** (Wang `g50t` transcript: "Ambiguous"). The digest names action number and display for each entry.
- **BATCH_NOOP_BLOCK conflict.** Isolated in its own arm (A2) so it cannot confound A1.
- **Images and mm-cache loop.** Isolated in A3, off by default.
- **Memory.** The ring is 8 records of 32 hex-row frames, about 1 MB per game by my estimate.
- **Unmeasured costs.** Real-board segmentation time and Qwen token counts for digests are estimates. Measure both in a dry run before the arms.
