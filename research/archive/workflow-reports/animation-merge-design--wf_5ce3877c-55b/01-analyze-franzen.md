# animation-merge-design: analyze:franzen

Workflow run `wf_5ce3877c-55b`, phase Analyze. Compare Wang vs Franzen animation-frame mechanisms with evidence, then design a merged mechanism

---

## Franzen's animation mechanism (main @ 10882e3)

All paths are under `ARC3-Inference/inference/` unless noted. Animation code arrived in `af58714` ("harness changes", 2026-09-30). `utils/animation.py` was added there. `CONFIGURATION.md` came with `55997ab` and `WRITEUP.md` with `6236d32`, going by `git log --diff-filter=A`.

### Competition config
- The competition notebook sets only `'ARC3_ANIMATION': '1'` (`competitors/kaggle/dfranzen-fork/dfranzen-fork.txt:10224`). `ARC3_ANIMATION_TIMELINE`, `_IMAGE` and `_DEATH_IMAGE` are not set.
- Defaults then apply: timeline ON (`agent/tool_agent.py:1861`), both image modes `"off"` (`agent/vision_context.py:122`, `CONFIGURATION.md:106-107`).
- Perception knobs on alongside it: `MULTIMODAL_UPSCALE=10`, `ARC3_DIFF_IMAGE=1`, `ARC3_GAMEOVER_DIFF_IMAGE=1`, `ARC3_FRAME_DIFF_HINT=1` (`dfranzen-fork.txt:10217-10223`).
- Guards also on: `ARC3_BATCH_NOOP_BLOCK=1` (`:10228`). Tool output cap is `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=3072` (`:10207`), against a source default of 1024 (`:1489`).

### Pipeline
1. **Capture** (`framework/solver.py:1271-1294`). After each executed non-RESET action:
   - `_animation_chain` (`:336-351`) builds `collapse_frames(before, engine_frames)`.
   - `collapse_frames` (`utils/animation.py:50-70`) prepends the pre-action board and drops adjacent duplicates. Non-adjacent repeats stay.
   - The docstring at `:57-61` says the "before" board is required, or an X→Y→X cell would register as one change.
2. **Summary** (`animation.py:96-119`). It returns `None` if `len(chain) <= 2`, so ordinary actions cost no tokens. Otherwise it returns `{frames, transient_pixels, transient_bbox}`.
   - "Transient" means a cell changed at least twice along the chain (`:73-93`).
   - The summary goes into the action result (`solver.py:1324-1325`).
   - It is passed through compaction (`tool_agent.py:5695`).
   - It is stored in each history entry's `.result` (`solver.py:1332-1341`). The summary persists per transition; the frames do not.
3. **Hint** (`animation.py:141-193`, called at `tool_agent.py:4869-4881`). `describe_animation` returns `""` when `transient_pixels <= 0` (`:155-159`). Pure motion toward the final state gets no hint (docstring `:10-16`).
4. **Death.** The auto-RESET skips the record (`solver.py:1272-1277`) so the fatal action's animation survives. A separate death hint is added after the BAR RULE (`tool_agent.py:4994-5009`).
5. **REPL** (`agent/python_tool_sandbox.py:570-640`). `last_animation_frames` and `last_animation_timeline` are lazy proxies.
   - They fetch through `step_env({"query":"animation"})` (`solver.py:995-1002`, `tool_agent.py:5765-5775`), which costs no game action.
   - They are invalidated after every `action()` call, including guard refusals (`sandbox:508-511`).

### What the model sees by default
- **System prompt:** `ANIMATION_ADDENDUM` plus `ANIMATION_ADDENDUM_TIMELINE`, 1,358 chars in total as I measured (`agent/prompts.py:163-188`, wired at `tool_agent.py:2668-2671`). It explains that transient cells "appear in no frame you can otherwise reach". It also says frames are `.ascii/.shape/.segmentation` views, that index 0 is the first animation frame, and "printing a whole 64x64 frame will not fit the tool budget".
- **Action result** (any animated action, including pure motion): `last_action_call_result['animation']` = `{'frames': 3, 'transient_pixels': 2, 'transient_bbox': [10, 11, 20, 30]}` (about 72 chars).
- **Opener line** (only when `transient_pixels > 0`). I ran `animation.py` on a synthetic board to get these (176-305 chars):
  - If the board area was unchanged: "The last executed action (`MOUSE(row=20, col=30)`) animated over 3 frames and left the board area unchanged, but it was not a no-op: 2 cells changed and changed back around rows 10-20, cols 11-30. Whatever this action did is visible only there - read `last_animation_timeline` before concluding it failed."
  - Otherwise: "...animated over 3 frames. 2 cells changed and changed back around rows 10-20, cols 11-30. `last_animation_timeline` shows them."
  - On death: "That fatal action animated over 3 frames, and inside it 2 cells changed... Those cells are in neither `history[-3].frame` nor `history[-2].frame`".
- **Timeline** (`build_animation_view`, `animation.py:279-333`; synthetic repr, 366 chars): `{'action': 'MOUSE(row=20, col=30)', 'frames': 3, 'steps': [{'step': 1, 'changed': 2, 'bbox': 'rows 10-20, cols 11-30', 'changes': ['B>G @ (10,11)', 'B>R @ (20,30)']}, ...]}`
- **Frames** (`build_animation_frames`, `:249-276`): one view per distinct state, excluding the pre-action board.
  - Each carries `ascii` (4,159 chars per 64x64 frame by my measurement; the code comment at `:28-29` says about 4,100) plus `grid`.
  - There is no cap on frame count.
- **No animation image** and **no auto-pasted timeline** by default.

### Token costs
- Fixed: about 1.36k chars of system prompt, constant, so prefix-cacheable.
- Per animated action: about 72 chars in the result dict. The opener adds about 180-300 chars only when transient pixels exist.
- Timeline: at most 24 enumerated cells per step and 80 in total (`animation.py:37-38`, `:311-331`). Steps over 24 cells, or after the budget is spent, get only a colour-transition census with no coordinates.
- The code comment at `:33-36` records a past misreading: an "omitted steps" count was read as "no changes". The step limit was removed for that reason.
- Frames cost about 4.2k chars each if printed whole. I did not measure Qwen token counts, so I give no token figure.
- Image modes (`vision_context.py:151-284`, `tool_agent.py:4529-4580`) add an image and are off by default. A composite image replaces the diff image on that turn (`tool_agent.py:4673-4677`).

### Evidence
- `WRITEUP.md:181`: "Of all perception changes, exposing animations to the model showed the strongest benefit in my tests"; tn36 "might be essentially unsolvable without reading these intermediate frames". That is the only game named.
- `WRITEUP.md:183`: the design was inspired by Jakob Brüggen's notebook. His notebook has "a function" to print frames. Franzen exposes variables instead and "observed the model" computing over them.
- `WRITEUP.md:185`: the hint fires only for transient pixels.
- `WRITEUP.md:301`: composite images "did not improve results and hurt some games where access to animation information had previously helped". Auto-inserted timeline gave "no clear gain".
- `WRITEUP.md:293`: "practical experiment results rather than a complete controlled ablation study". There are no per-game numbers for animation in either doc.
- `tool_agent.py:1852-1857` (author comment): a coherent timeline "invites stopping at it". On one level it "produced a confident and wrong mechanic" that the model corrected only after reading the frames. The comment says "Only an experiment settles which costs more."

### Known limitations
- **Only one record is kept** (`solver.py:490-493`). Every executed non-RESET action overwrites it, or sets it to `None` if it did not animate (`:1284-1294`).
  - In a batch, only the last executed action's animation survives (`:1158`, `tool_agent.py:3783-3788`).
  - `research-sweep.md:17` says "Control transcripts show m0r0, sp80 and su15 hitting the limit, and su15 spends real actions on UNDO plus re-click to re-read an animation". I could not find those transcripts locally, so this is unverified by me.
  - The sweep's line numbers (`solver.py` ~1812-1830, L1490) do not match main (994-1001, 1271-1294).
- **Pure-motion animations get no hint** (`animation.py:157-159`), though the result dict still shows `transient_pixels: 0`. I confirmed that in a synthetic run.
- **BATCH_NOOP_BLOCK** (on in the competition config) stops a batch when `gameplay_changed` is false (`solver.py:1127-1134`), even if the action animated. The opener says "not a no-op", while the guard stop text says "produced no change inside the board area". `research-sweep.md:18` flags this blind spot.
- **Off-by-one between views:** `frames[i]` is timeline `step i+1`. The prompt never says so, despite the `animation.py:249-254` docstring implying they align.
- **Proxy gaps:** `_LazyAnimation` has no `.items()` or `.values()` (`sandbox:599-635`; `research-sweep.md:27`). Each snippet re-fetches the full frame payload.
- **Transient detection:** it is a "changed at least twice" rule over raw cells, with no border masking (`animation.py:73-93`). Only the composite image masks the border (`vision_context.py:189-192`).
