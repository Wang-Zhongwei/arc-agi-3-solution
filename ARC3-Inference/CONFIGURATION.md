# Duck harness: cumulative change inventory

This inventory describes the current harness changes relative to **Tufa Labs’ Duck harness, commit `7652836056c59e044f093e3c13ed7438c814169e` (`7652836`)**. Full credit belongs to Jeroen Cottaar and Tufa Labs for the original solver and harness.

**Scope:** the current cumulative source changes in `ARC3-Inference`, including six new production modules/data files and changes to thirteen existing production, build, and viewer files. README edits, analysis scripts, notebooks, model weights, and SGLang server patches are outside this inventory. Audited on 29 September 2026. Since then the `user-combo` merge into `main` (#135) added two production modules, `inference/agent/memory_ledger.py` and `inference/agent/prefix_reuse.py`, and changed `prompts.py`, `python_tool_sandbox.py`, `runtime_state.py`, `tool_agent.py`, `vision_context.py`, `framework/solver.py` and `utils/animation.py`; their flags are documented in the sections below.

This is an implementation inventory, not a claim that every option improved score or was enabled in the final submission. It includes optional experiments still present in the code. Defaults below are the effective harness defaults when the environment variable is absent; a notebook can override them. Set configuration before importing/starting the harness, since some settings are read at import time.

## 1. Action exposure and action explanations

### Undo

Added a complete model-facing route from `UNDO` to engine `ACTION7`, covering action lists, validation, execution, and prompts. The model can recover from mistakes in games that provide this action.

| Setting | Default | Behavior |
|---|---|---|
| `EXPOSE_UNDO` | `tufa` | `tufa` preserves the original inconsistent wiring: ACTION7 can be listed but cannot be executed. `off` hides ACTION7/UNDO and rejects them. `on` lists UNDO and accepts either UNDO or ACTION7 for execution. |

This setting is an enum: **use `on`, not `1`**. Invalid values fall back to `tufa`. Exposure does not make undo available in a game/state that does not support it.

### Model-initiated reset

Added explicit RESET advertising, guidance, and snippet-level restrictions. The intended use is recovery from an unrecoverable position without spending actions merely to exhaust the bar.

| Setting | Default | Behavior |
|---|---|---|
| `EXPOSE_RESET` | `off` | Enables explicit RESET presentation and the first-action restriction. Accepts `on`, `1`, `true`, or `yes`. Off preserves legacy RESET handling; it is not a blanket ban on every legacy RESET path. |

When exposed, RESET must be the first action of the Python snippet. A batch that reaches RESET after another action is stopped before executing it. Subsequent actions after a successful initial RESET remain subject to the normal guards. RESET counts as an action. Guidance says it usually restores the current level, including its bar, to the starting state.

When exposed, a RESET is also refused when no engine action has been executed since the level started or was last reset (stop reason `reset_no_actions`; nothing executes, no action is spent, and the snippet ends as for `reset_not_first`). The ARC API treats a RESET with no action since the last reset or level change as a restart of the **whole game**. Locally `taaf` sets `ONLY_RESET_LEVELS=true`, which hides this, but the competition gateway need not honour it. The count is read from the solver's history (`runtime_state.actions_since_reset`), which is never trimmed: a model RESET, the automatic RESET after a game over, the warmup RESET, a completed level and the seeded initial frame all set it to zero, and every other action, UNDO included, adds one. Context trims, slot handovers and snippet boundaries do not affect it.

With `ARC3_ACTION_INFO` on as well, the RESET line of the system prompt states that refusal, and the UNDO line (when `EXPOSE_UNDO=on`) gains "UNDO itself counts as one action and does not refund the action it reverses." With `EXPOSE_RESET` off, the system prompt, the requests and the guards are byte-identical to `main` (`tests/test_expose_reset.py`, with `RESET_SIM_BASELINE_ROOT`). Kaggle arm: `expose-reset`.

A model-requested reset receives reset-specific follow-up information rather than being represented as a death. The previous action’s animation is preserved across reset, including automatic reset after death. This lets the model inspect what happened before recovery.

### Action information in the system prompt

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_ACTION_INFO` | Off | Adds compact descriptions of directional controls, SPACE, and MOUSE; includes UNDO when undo exposure is on and RESET when reset exposure is on. |

The descriptions clarify that action semantics depend on the game. MOUSE uses integer `row` and `col` fields: rows increase downward, columns increase rightward, measured from the top-left. The normal model interface rejects legacy `x`/`y` fields. The explanation is in the system prompt rather than repeatedly appended to user prompts.

## 2. Prompt changes and optional behavioral guidance

| Setting | Default | Change |
|---|---|---|
| `ARC3_LEVEL_TRANSFER_GUIDANCE` | Off | Revises system and level-up user guidance: carry forward useful mechanics from the preceding level; inspect new elements and test their role; expect new mechanics; reconsider how the goal must be reached. The level-up message identifies the returned frame/image as the next level’s starting state. |
| `ARC3_FRAME_DIFF_HINT` | Off | Explains how to call `frame_diff`, its defaults and returned fields. Describes same-color four-connected components and warns that matching is heuristic, so a reported movement does not establish persistent object identity. It is informational rather than a requirement to use the helper. |
| `ARC3_STEP_VERIFICATION_HINT` | Off | Replaces the player-specific verification hint with general guidance to compare observations with expected effects during multi-step code. If search fails, the game is still solvable; reconsider mechanics, goal, search implementation, search limits, and interactions with new elements. |
| `ARC3_NEW_CHANGED_PROMPTS` | Off | Uses more precise language about no change in the board’s gameplay area, rather than equating every HUD-only change with useful progress. Also changes related guard feedback. |
| `ARC3_REPORT_GAMEPLAY_CHANGED` | Off | Exposes `gameplay_changed` and `no_op` in model-facing action results. The harness can calculate these internally even when reporting is off. |
| `ARC3_EXPLAIN_GAMEPLAY_CHANGED` | Inherits `ARC3_REPORT_GAMEPLAY_CHANGED` | Adds the explanation distinguishing full-board changes from changes inside the gameplay area. Can be explicitly enabled independently. |
| `ARC3_OPENER_COMMIT_HINT` | Off | Adds encouragement to commit to an action sequence when the mechanics are sufficiently understood. |
| `ARC3_PREFER_TOOL_CALLS` | Off | Adds one tool-session bullet after the "Do not ration tool calls" line: reading and computing spend no actions and only `action(...)` spends the level budget; when unsure, prefer another tool call over more reasoning, since the code answers what the reasoning would only guess at and runs faster. Static text, so the cached prefix is unaffected. |
| `ARC3_DEDUPE_MULTICALL_LINE` | Off | Removes duplicated prompt wording about multiple action calls. This is text deduplication, not deduplication of tool execution. |
| `ARC3_BUDGET_TRUTH` | Off | Tells the model the truth about its budget. Action results the model sees (the `action(...)` return value, `last_action_call_result`, tool output) leave out the run clock (`run_elapsed_seconds`, `time_remaining_seconds`); the solver's raw payload keeps it, and it still drives the endgame phase, request timeouts and the viewer. The visual-game guidance describes the edge bar as a step bar that usually counts down a per-level action budget and shrinks with actions, not wall-clock time, instead of a timer the model should not be distracted by; the next sentence classifies a segmented edge strip as "HUD state" instead of "HUD/timer state", and "DON'T DO THIS!" and "such as countdown bars" are dropped. Other model-visible wording still mentions a timer or a step/time budget and is unchanged: the `board_changed`/`gameplay_changed` line, the `RESET` line ("the remaining-action/time bar", shown when `EXPOSE_RESET` is on), the python-tail line about "only a timer, progress bar, or remaining-step bar", the flag semantics ("step/time budget depleting"), the GAME OVER opener ("step/time budget running out"), the game-over stop detail ("depleted step/time budget"), the opener note for an edge-only change ("usually a timer or remaining-steps bar") and the solver's no-op stop detail (`ARC3_TURN_BASED_FACT` rewords these; see the legibility table below). The model also loses the clock for deadline urgency near the end of the run. With `ARC3_PERSISTENT_FUNCTIONS` on, "ephemeral" is also dropped from the system line and the python tool's descriptions, where retained functions make it false. Fixed for the run, so the cached prefix is unaffected. |
| `ARC3_PROMPT_HYGIENE` | Off | Fixes wording defects without changing what the prompt says. (a) The system prompt's memory section moves from between the world-model line and the Python guidance tail to after the last Python guidance bullet: "Memory that survives history trimming:" with `ARC3_LEDGER_ON_TRIM` on, or, with the §10 memory flags, the memory-v2 section (`ARC3_MEM_SOLVED`, `ARC3_MEM_CARD`, ...), the memory layout's "Memory kept by the harness:" (`ARC3_MEM_EVERY_CALL`) or the `ARC3_MEM_PATH` line, moved whole and unchanged (after `ARC3_FRAME_DIFF_HINT` and `ARC3_STEP_VERIFICATION_HINT` text), just before "Tool session rules:"; before, the search, frame-diff and `frame_diff()` bullets sat under the memory header. (b) The entity bullet gets its missing newline (it ran into the next bullet as "...1x1 tokens.- Some games") and "usually be rendered"/"Sometime" become "usually rendered"/"Sometimes". (c) The segmented-bar line gets its "- " marker. (d) With `ARC3_PERSISTENT_FUNCTIONS` on, "call it with one ephemeral `code` string" loses "ephemeral" (retained functions survive; the bullet two lines up and the tool-session rules state how long, per `ARC3_PERSISTENT_FUNCTIONS_SCOPE`), and the python tool's two descriptions lose it too, as under `ARC3_BUDGET_TRUTH`; the same substitutions as `ARC3_BUDGET_TRUTH`'s, so the two compose. (e) The game-over opener's "Record the diagnosed cause in your world model" and "record the correction in your world model instead of retesting it" name the memory sections, which `ARC3_MEMORY_SECTIONS=off` removes: with sections off they name the ledger's `<world_model>` and `<hypothesis status="refuted">` tags when `ARC3_LEDGER_ON_TRIM` is on, or say to state the cause in the reasoning when it is off; with sections on they are unchanged. (f) The `ARC3_DIFF_IMAGE` line ("Alongside the current grid image you receive a diff image: ...") gets its missing "- " marker when it starts a line, which it does unless `ARC3_AUTO_FRAME_DIFF` is on. The system prompt is fixed for the run. |
| `ARC3_IMAGE_GEOMETRY` | Off | With `MULTIMODAL_CONTEXT=current_grid`, adds one "Multimodal context:" bullet stating the board image's pixel geometry, formatted from `MULTIMODAL_UPSCALE` (the factor `frame_to_png_data_url` resizes the 64x64 one-pixel-per-cell frame by). At the Kaggle setting of 10: "The board image is 640x640 px: 64x64 cells at 10 px per cell; cell (row r, col c) spans x 10c..10c+9, y 10r..10r+9." The diff and game-over diff images use the same scale; animation keyframes (`ARC3_ANIM_KEYFRAMES_UPSCALE`) are not described. Static for the run. |
| `ARC3_SYSTEM_PROMPT_PREFIX` | Empty | Prepends a custom system-prompt prefix for experiments. |

With level-transfer guidance enabled, the old advice about re-grounding after a score increase is replaced: the model is explicitly informed of level transitions rather than being asked to infer them from score. Transition/terminal behavior is explained alongside the corresponding runtime information.

Several consistency fixes are unconditional: documentation now includes frame `.shape`; hashes are not described as unique object identities; and normal prompt guidance describes automatic reset after death as occurring before the next analyzer turn.

### Legibility: harness text restated as facts (`ARC3_LEGIBILITY`)

Harness text the model misread in runs B (`v4-pf-keep-10x`), A (`v4-pf-solved-10x`) and PF (`v4-prompt-facts-10x`), restated as facts; evidence and counts are in the pass-2 readability report (`memstruct/readability/readability.md`, item numbers below). Each fix is its own flag, off by default; flags off, every request is byte-identical to arm/pf-mem under the pf-keep-10x env (`tests/fixtures/legibility_pf_keep_golden.json`). The umbrella `ARC3_LEGIBILITY=1` turns on the six fixes marked "bundle"; a sub-flag set explicitly wins over the umbrella either way (`ARC3_LEGIBILITY=1 ARC3_RETAINED_SOURCE=0`, `ARC3_LEGIBILITY=1 ARC3_DIFF_SAME_CELLS=1`). System-prompt text is applied once per run as exact (old, new) rows after `ARC3_BUDGET_TRUTH` (`prompts.LEGIBILITY_SYSTEM_ROWS`), so the prompt stays static; a row whose old text is not found exactly once is skipped with a warning, and the tests assert every row applies under the pf-keep-10x env. Everything else is new tool output or new text in the newest turn message, so no fix edits text already sent. Code: `inference/agent/legibility.py` (switchboard), `prompts.py`, the sandbox, the solver, `utils/segmentation.py`, `utils/frame_diff.py`, `utils/animation.py`.

| Setting | Default | Bundle | Change |
|---|---|---|---|
| `ARC3_LEGIBILITY` | Off | — | Turns on the six bundle fixes below. Explicit sub-flags override it. |
| `ARC3_SEG_BBOX` (item 3) | Off | Yes | Every segmentation node in the sandbox ends with `bbox`: `[r0, c0, r1, c1]`, the first and last row and column the object covers, inclusive (the min/max of `boundary`, the box `frame_diff` uses). The node bullet names `bbox` and says `boundary` is "the `[row, col]` cells where the outline turns, clockwise: 4 for a filled rectangle, 2 for a one-cell-wide line, 1 for a single cell; it is not a bounding box"; the `frame_diff` bullet says its `bbox` is inclusive. Host-side segmentation is unchanged. |
| `ARC3_ANIM_SCHEMA` (item 4) | Off | Yes | `animation_for(n)` of an action whose animation is not kept returns every documented key (`action` None, no frames, a timeline with `action_num`, `action`, `frames` 0 and `steps` []) plus `error`, as a dict that tests false (its timeline too). The timeline bullet says `changed` is "an int: how many cells changed at that step"; the ring bullet describes the not-kept return. |
| `ARC3_RETAINED_SOURCE` (item 6) | Off | Yes | With `ARC3_PERSISTENT_FUNCTIONS` on, every python call has `retained_source`, a dict from each restored retained function's name to the source it was restored from, and the tool session rules add "- `retained_source` maps the name of each retained function to its source text." after the retention bullet. Inert without retention. |
| `ARC3_STALE_TRUTH` (item 7) | Off | Yes | The stale-state refusal (`solver._stale_detail`) ends "(Catching StaleStateActionError does not lift this halt: every later action() call in this python call is refused the same way, separate action() calls included, and a refused action is not executed and costs nothing. Your next python call can act again. <action> itself was executed and counted as one action.)" instead of saying the error "can be caught". Behaviour is unchanged. |
| `ARC3_RESET_BAR_FACT` (item 8) | Off | Yes | The `RESET` line (shown with `EXPOSE_RESET=on`) begins "When available, `RESET` restores the current level to the board it started with, step bar included, and keeps completed levels." and says "RESET adds one action to the level's action count, which the score uses, as every action already spent does; it does not shorten the restored bar."; the flag-semantics line and the game-over opener say the auto-reset includes the step bar. |
| `ARC3_TURN_BASED_FACT` (item 9) | Off | Yes | Appends to the edge-bar bullet (`ARC3_BUDGET_TRUTH`'s, or Franzen's when that is off): "The game is turn-based: the board, bars included, changes only when an action executes; time spent reasoning or in tool calls changes nothing. One action can remove no bar cell, one, or several, depending on the game and the action." Timer and "step/time" leftovers become step-bar wording in the system prompt (RESET line, unless `ARC3_RESET_BAR_FACT` rewrote it; flag semantics; the `gameplay_changed` bullet; the after-every-action bullet) and in the game-over stop detail, the game-over opener, the edge-only opener line and the solver's no-op stop detail. |
| `ARC3_DIFF_SAME_CELLS` (item 10) | Off | No | The sandbox's `frame_diff()` gives each `appeared`/`disappeared` entry `same_colour_cells` (how many of its cells hold the same colour in the other frame) and each `moved` entry `same_hash_candidates` (how many before-frame components with that hash touched the change), and its bullet says what equal-to-`pixels` and above-1 mean. |
| `ARC3_DIGEST_COORD_FACT` (item 14) | Off | No | The animation digest's closing line adds "Positions are the (row, col) of a blob's top-left cell and sizes are rows x cols, both taken from stored states that can skip steps, so a tween's d() or a path's extreme point can differ from the settled move." (swapped in after the digest is fitted to its budget); the digest bullet says trail sizes are rows x cols. |
| `ARC3_EDGE_NOOP_FACT` (item 16) | Off | No | When an action changes only the edge band, its verdict in the per-action trace and echo ends "; N cells changed in the 4-cell border, rows a-b, cols c-d" (NO-OP and ANIMATED alike), the next opener's no-op guard note carries the same clause, and the animated-no-op opener line adds how many border cells changed and stayed changed. Tool and turn text only: `<this_level>` words are memory-v2's. |
| `ARC3_ACTION_INDEX_FACT` (item 19) | Off | No | Adds "These were actions 157-160 of the game; 160 actions have been taken." after `Executed actions:` (`Executed actions:` and `Current state:` are unchanged; extractors parse them), and the `current_frame.step` bullet states how the numberings relate (without `animation_for` when the animation ring is off). |
| `ARC3_LEGEND_CASE_PAIRS` (item 17) | Off | No (held out) | Appends to the colour legend: "Letter case changes the colour: B black / b blue, W white / w light gray, G dark gray / g gray, R red / r dark red, P pink / p purple." |
| `ARC3_FRAME_RUNS` (pass 1; not verified in pass 2) | Off | No | Every sandbox frame view gains `.runs(r0, r1, c0, c1)`: one line per row with each run of one letter and its first and last column, inclusive (`28: c38-40 Y41 B42-43 Y44 c45`), clipped to the board; the runtime-state bullets and the python tool's description list it. `.ascii` is unchanged. |

Why this bundle: items 3, 4, 6, 7, 8 and 9 waste tool rounds or plant wrong beliefs in every run. Items 10, 14, 16 and 19 each cost a few short detours per run with no traced wrong action, and two of them add text to every diff or turn, so they stay flag-only and can be stacked later as their own factor. Cost under the pf-keep-10x env, counted with the Flash-Next tokenizer: the bundle adds 213 static system-prompt tokens (5,101 → 5,314; every flag on: +479); its other text appears only when its event happens (a `bbox` key per printed node, a refusal or opener sentence, `retained_source` when printed). Kaggle arm: `pf-keep-legible-10x` (pf-keep-10x + `ARC3_LEGIBILITY=1`); `pf-keep-v2-all-10x` runs the bundle together with memory v2 (§10, "Memory v2 and legibility together").

## 3. Segmentation and structured frame differences

### Richer shape metadata — unconditional

Extended the existing segmentation representation with rotation-aware shape information. Components expose canonical hashes and orientation-related metadata, including `shape_hash`, `pose_hash`, `rotation`, and `rotational_symmetry`. Canonicalization accounts for rotations, and color-independent orientation is calculated consistently across differently colored copies of a shape.

Segmentation still uses same-color **four-connectivity**. Diagonal contacts, multicolor objects, and touching objects can therefore produce components that differ from the game’s conceptual objects. Hashes describe shapes/appearances; they are not unique object IDs.

### `frame_diff` — unconditional Python helper

Added `frame_diff(before=None, after=None)` to the sandbox. Its defaults compare `previous_frame` with `current_frame`. It returns a compact structured account of changed cells and candidate object changes: movement, rotation, appearance, disappearance, recoloring, and resizing. Different-sized frames return an explicit full-redraw/shape-mismatch result.

Matching uses component information and assignment heuristics. Multiple similar objects, merges, splits, and segmentation changes can make the correspondence ambiguous. Automatic prompt rendering bounds expensive matching for large groups of identical components; direct Python inspection remains available independently of the automatic hint.

| Setting | Default | Change |
|---|---|---|
| `ARC3_AUTO_FRAME_DIFF` | Off | Inserts a compact structured/textual frame-difference summary in the user prompt. Separate from availability of the Python helper. |
| `ARC3_AUTO_FRAME_DIFF_BUDGET` | `300` | Approximate token budget for the automatic text diff; rendering falls back from individual changes to grouped changes and then counts. |
| `ARC3_LEVEL_INVENTORY` | Off | Compares object inventories at the starts of successive levels, highlighting newly appearing types. Independent of ordinary per-action diffs and diff images. |
| `ARC3_GAMEOVER_DIFF` | Off | Adds a textual comparison involving the fatal frame after game over. |

## 4. Images and animations

### Diff images

Added images that show changed cells in their true palette colors while dimming unchanged cells to an off-palette dark navy. This avoids confusing the background marker with a real board color. Invalid comparisons, including incompatible frame sizes, do not produce a misleading diff image.

| Setting | Default | Change |
|---|---|---|
| `ARC3_DIFF_IMAGE` | Off | Attaches ordinary board-difference images. |
| `ARC3_GAMEOVER_DIFF_IMAGE` | Off | Attaches death-related comparison images in the first post-death opener. |
| `ARC3_RESUME_DIFF_IMAGE` | Off | Reattaches the current turn’s diff image to a resumed turn, useful if earlier images have been hidden. |
| `ARC3_RESUME_PROMPT_IMAGES` | On | Allows images on yield/resumption messages. Turning it off removes those repeated attachments. |
| `MULTIMODAL_RESAMPLE` | `nearest` | Selects the image resize filter: nearest, bilinear, bicubic, or lanczos. |
| `ARC3_HISTORY_IMAGE_KEEP` | Unset: unlimited | Optionally retains images only in the latest N image-bearing messages sent to the model. Zero strips all such images. This filters outgoing context and its estimate rather than deleting stored history. |

The original `MULTIMODAL_CONTEXT=current_grid` and `MULTIMODAL_UPSCALE` settings remain the basic image controls. The latter’s source default is 16; running at 10× is a notebook configuration choice, not a new fixed default. Filtering historical images changes request prefixes and can affect prefix reuse.

### Intermediate animation frames

Added capture of intermediate frames returned by an action, consecutive-frame deduplication, transient-pixel detection, and compact animation metadata. This recovers information that disappears before the final board is returned.

| Setting | Default | Change |
|---|---|---|
| `ARC3_ANIMATION` | Off | Enables animation capture, summaries, and model inspection support. |
| `ARC3_ANIMATION_TIMELINE` | On, under animation support | Provides the compressed timeline accessible through Python. Does not automatically paste the full timeline into the user prompt. |
| `ARC3_ANIMATION_IMAGE` | `off` | `peak` shows one real frame with many transient cells; `composite` combines transient information across time. Ordinary animation imagery is attached for transient activity when the gameplay area did not change. |
| `ARC3_ANIMATION_DEATH_IMAGE` | `off` | Separate `peak`/`composite` option for the fatal action’s animation. |

The image-mode variables are enums, not booleans: an invalid value such as `1` raises a configuration error. Image attachments require the multimodal image path to be enabled.

The model can inspect `last_animation_frames` and `last_animation_timeline`. These are loaded lazily through a read-only environment query, without spending a game action. The prompt points out transient activity so that the model knows intermediate information is available.

A composite is explicitly not a real single board state. It retains the last non-final transient colors, with brightness encoding duration. A peak image is an actual intermediate frame, which avoids combining moments but may omit important parts of a longer animation.

Animation correctness fixes preserve fatal-action information through automatic RESET, preserve preceding animation through explicit RESET, and invalidate cached Python views after actions, including aliases and guard refusals. A subsequent ordinary action updates or clears the animation record instead of leaving an unrelated old animation visible.

### Animation layers: truth, ring and digest

Three default-off layers that combine Franzen's animation access with Wang's proactive frames (design: `research/archive/reports/animation-combo-design.md`). Each needs `ARC3_ANIMATION`; the digest also needs the ring. With all of them unset (or `0`), the system prompt, openers, action results and sandbox globals are byte-identical to the control: `tests/test_flags_off_identity.py` pins them against the pre-digest tree, and `tests/test_anim_loop.py` compares every request of a real game loop with `arm/throughput-stack`.

| Setting | Default | Change |
|---|---|---|
| `ARC3_ANIM_TRUTH` | Off | L1, truthful verdicts. An action that left the board area unchanged but whose animation changed cells and changed them back is reported as `ANIMATED (<type>; board area unchanged; N frames, T cells changed and changed back - see animation_for(n))` instead of `NO-OP` in the action echo and in every `action_trace` position. The pointer is `animation_for(n)` with the ring on, `last_animation_timeline` (or `_frames` with the timeline off) for the last action without it, and absent for an earlier batch position without it. `<type>` is `blink`, `tween`, `path` or `other` (`classify_animation`, a port of lordhansolo's classifier, HUD border masked). Such an action also passes the batch no-op stop (`ARC3_ANIM_NOOP_EXEMPT` folded in), does not arm the stale-state latch, and is never filed by the known-no-op guard (`ARC3_NOOP_REPEAT_GUARD`). An action with no transient cells is still a `NO-OP`; `gameplay_changed`, `no_op` and `gameplay_changed_per_action` keep their meaning. Adds one constant system-prompt clause naming the verdict. |
| `ARC3_ANIM_RING` | `0` (off) | L2, the number of recent animated actions kept readable from Python (the arms use 8). Turns on `animation_for(n)` (`{action_num, action, frames, timeline}`, `note` when subsampled) and `animations` (one entry per kept action, no grids: `action_num`, `action`, `step_count`, `transient_pixels`, `level`, the level the action ran on), and `.items()`/`.values()` on the lazy views. `last_animation_frames`/`_timeline` then hold the newest action that animated (a non-animated action no longer clears them), with its full chain, and a timeline carries `action_num`. Action results gain a per-batch `animations` list (`action_num`, `action_display`, `animation`) holding only the actions whose animation hid something (transient cells or a board area left unchanged), newest 8, with `animations_omitted` for older ones and `animations_unlisted` counting motion-only animations; a motion-only batch adds no key, and a list that would only repeat `animation` is left out. These keys go last in the result: under `ARC3_MIDDLE_TRUNCATION=1` a print that ends with the result keeps them. Stored chains are capped at 32 boards (`S0` plus 31 states, first and last kept); summaries and signatures come from the full chain. Reading costs no game action. Adds a constant system-prompt addendum stating the indexing (`frames[k-1]` is step k) and the record keys. |
| `ARC3_ANIM_DIGEST` | Off | L3, Wang-style proactive text. Needs `ARC3_ANIM_RING > 0`; set without it, `ToolAgent` refuses to start. The next turn opener carries a short "Animation digest" right after the animation line (or the death line), for the animated actions that were new or hid something. Each action shown in full leads with its type: `blink xN`, `tween C HxW (r,c)>(r,c) d(dr,dc)` (one tracked sprite) or `path C HxW (r,c)>...` (at most 8 points), then the state count and transient cells, then at most three trail lines. Adds a constant system-prompt addendum. |
| `ARC3_ANIM_DIGEST_GATE` | `novel` | `transient` shows an action whose animation changed cells and changed them back, or left the board area unchanged. `novel` additionally needs a signature not yet shown on this level (action name without coordinates plus per-step change counts and colour census), and reduces a repeat to a one-line pointer. `all` shows every animated action. An invalid value fails at startup. |
| `ARC3_ANIM_DIGEST_CHARS` | `600` | Character cap for one digest (about 150 tokens). Trail lines are dropped first, then pointers, then later actions; the typed header goes last. |
| `ARC3_ANIM_DIGEST_GAME_CHARS` | `12000` | Character cap for all digests of one game. After it, only the existing animation line remains. |
| `ARC3_ANIM_KEYFRAMES` | `0` | L4, off in every arm (Franzen measured composite animation images hurting). When a digest fires and `MULTIMODAL_CONTEXT=current_grid`, attaches up to N real intermediate boards of the digest's most revealing action, after the board image. Never stripped or rewritten later; an `ARC3_HISTORY_IMAGE_KEEP` window strips them like any other history image. |
| `ARC3_ANIM_KEYFRAMES_UPSCALE` | `min(MULTIMODAL_UPSCALE, 4)` | Upscale of the keyframe images. At 4x a 64x64 board is 256x256, about 66 vision tokens. |
| `ARC3_ANIM_NOOP_EXEMPT` | Off | The batch no-op stop alone of L1: an action whose animation changed cells and changed them back passes `ARC3_BATCH_NOOP_BLOCK`. `ARC3_ANIM_TRUTH` implies it. |

Digest details:
- **Placement.** Only in the new turn opener of an acting turn; never in a yield resumption's opener, a tool result or the system prompt. Nothing already sent is rewritten, so the prefix cache is untouched. A rebuilt opener for the same step summary repeats the same text without charging the game cap again.
- **Trails (lens A)** are blobs that differ from the settled board, split by colour and four-connectivity, outer `ARC3_NOOP_GUARD_BORDER` cells masked, chained by colour and bounding-box size, listed as top-left `(row,col)` in time order. They claim no identity.
- **Numbering.** `S0` is the board before the action. Step `k` is `animation_for(n)['frames'][k-1]`, and `last_animation_frames[k-1]` for the last action unless its chain was subsampled; a subsampled entry points at `animation_for(n)`.
- **Lifetime.** Seen signatures reset on a level change and per game, each filed under the level its action ran on. The ring lives on the host, so it survives history trims: after a trim `animation_for(n)` still returns a pre-trim action.
- **Not built:** object motion from `frame_diff` (lens B), which needs the #41 fix; `ARC3_FRAME_DIFF_STRICT_MOVES` does not exist. The tween header gives a `d(dr,dc)` for a single tracked sprite without `frame_diff`.

## 5. Action guards and failure prevention

### Optional guards

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_BATCH_NOOP_BLOCK` | Off | Stops the remaining actions of a batch after an executed action fails to change the interior gameplay area. |
| `ARC3_STALE_STATE_BLOCK` | Off | Blocks a later `action(...)` call within the same snippet when a preceding action was ineffective, preventing continued execution from an unchecked assumption. |
| `ARC3_NOOP_REPEAT_GUARD` | Off | Recognizes previously ineffective state/action pairs and refuses a repeat. A deliberate immediate reissue can override the refusal. |
| `ARC3_DEATH_REPEAT_GUARD` | Off | Recognizes recorded fatal state/action continuations and refuses a repeat, with a deliberate reissue override. |
| `ARC3_DEATH_GUARD_LOOKAHEAD` | Off | Extends fatal-continuation checking to a proposed batch before earlier actions in that batch are spent. Used with the death guard. |
| `ARC3_REPEAT_STATE_GUARD` | Off | Stops an action repeated from the same state within one snippet, catching cycles caused by faulty loops. A new snippet starts a fresh check. |
| `ARC3_DEATH_LEDGER` | Off | Adds advisory information about previously fatal routes and exposes the structured ledger for inspection. |
| `ARC3_GUARDS_FROM_LEVEL` | `1` | Starts the per-action stale/repeat/death guards at this level. Does not defer the batch no-op guard or terminal latch. |
| `ARC3_NOOP_GUARD_BORDER` | `4` | Excludes the outer border when deciding whether gameplay changed, so HUD/bar changes do not count as progress. Zero disables the border-dependent batch no-op behavior. |
| `ARC3_STATE_IDENTITY_BORDER` | `0` | Separately controls the border ignored when identifying states for repeat/fatal checks. |

Death recording is enabled whenever either the ledger display or the death guard is enabled. The guard works independently of the advisory display.

Mouse action signatures translate engine coordinates into the same row/column convention used by guard comparisons, fixing missed matches between equivalent clicks.

### Unconditional terminal and result handling

A terminal latch prevents further actions within the same snippet after a death, level completion, or finished run. This protects against executing a plan on a replaced board. Guard refusals return clear stop metadata, including what actually executed and what was skipped; a refusal with zero executed actions does not invent a transition.

Known-no-op, known-death, repeated-state, and terminal-stop exceptions derive from `BaseException`, so a model-written broad `except Exception` cannot silently swallow them and continue a broken loop.

### Optional output guard

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_BOARD_DUMP_GUARD` | Off | Compresses long full-width ASCII board dumps in tool output and adds a reminder to print useful computed facts or small regions. Short crops remain available. |

## 6. More faithful Python state and action results

These changes are unconditional unless a setting is shown.

- **Per-action history metadata:** executed actions retain their individual compact result metadata in history and `transitions[i].result`, rather than assigning the batch result only to the final transition and leaving older transitions empty.
- **Separate batch result:** the global **`last_action_call_result` replaces `last_action_result`**. It describes the most recent model `action(...)` call, including a batch or refusal. This is an intentional model-facing API rename, without an old-name compatibility alias.
- **Stable inspection:** `last_action_call_result` survives intervening inspection-only snippets and analyzer turns. It is empty before the first applicable call.
- **Correct death/reset distinction:** automatic RESET has its own transition metadata marked automatic. It does not overwrite the model’s fatal action-call result with a successful reset result.
- **Correct before/after access:** prompts and tool metadata explain that `history[-1].frame` is current state. Use `previous_frame` or the transition’s before/after frames for differences.
- **Compact results:** retained per-action metadata excludes raw boards and full animation chains.
- **REPL-style last expressions:** the final expression of a snippet is evaluated and printed when non-None; it can also populate the result when no explicit result was supplied.
- **Sandbox compatibility:** added `difflib` to the allowed standard-library modules and more ordinary exception types to the available builtins, avoiding unnecessary snippet failures on harmless imports or exception handlers.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_ACTION_ECHO` | Off | Prints a compact account of action calls/results into tool output. Avoids returning the same echo again as duplicated result data. |
| `ARC3_MIDDLE_TRUNCATION` | Off | When tool output is too long, retains its beginning and ending around a visible omission marker, helping preserve exceptions or conclusions at the end. |

The original `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS` controls output length. Increasing that budget is a configuration choice; middle truncation changes which parts survive. Output truncation uses a stable character-based approximation rather than fluctuating with the context estimator’s learned token ratio.

### Python tool deadline and output of a killed call

These are unconditional; the wall-clock cap factor is the only setting.

- **The deadline counts the snippet's own time:** `LOCAL_ANALYZER_TOOL_TIMEOUT` (at most 30 s) no longer runs while the harness serves the snippet's `action(...)` and animation requests (environment step, state rebuild, serialization). That time pushes the deadline back. Without this, a loop of single-action calls late in a game timed out after 20 to 40 calls. A separate wall-clock cap, `ARC3_TOOL_WALL_CAP_FACTOR` times the timeout, still ends a call whose actions alone run long. When the cap stops a call, the model is told `Tool timed out after 90s in total, Ns of it executing actions`. The system prompt's "hard time limit of 30 seconds" wording is unchanged.
- **Output survives a timeout or crash:** the sandbox forwards print output to the harness while the snippet runs. Each completed line is forwarded at once, up to 5 lines per 0.25 s. Other output is forwarded every 0.25 s and before each `action(...)` or animation request. If a call is killed or its process dies, the call returns the output received so far as `stdout`, in the same result shape. The error also says how many actions had already executed, for example `Tool timed out after 30s (37 actions already executed)`; with no executed actions the error text is unchanged. A killed or crashed call still carries no retention result. A call that finishes reports exactly what it did before.
- **What a kill can still lose:** output from the last 0.25 s before the kill that is not a completed line, or that came faster than 5 lines per 0.25 s. The 0.25 s forwarding cannot run while the snippet is inside one long call that holds the interpreter, such as a catastrophic regular expression or a huge integer power. A completed line printed just before such a call is still forwarded if it is within the 5-line allowance.
- **A killed print flood is bounded:** past its first 64K characters, the call keeps only the first 64K and the latest 64K, joined by `... [N chars omitted] ...`. The usual tool-output truncation then applies, so its `[truncated N chars]` count covers only the kept text.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_TOOL_WALL_CAP_FACTOR` | `3` | Wall-clock cap on one python call, including the harness's action time, as a multiple of the tool timeout. `1` restores the previous wall-clock billing, except that a call with executed actions that runs out is reported as `Tool timed out after 30s in total, Ns of it executing actions`. The partial output and executed-action count stay. Values below 1 count as 1. |

## 7. Retaining model-written functions

Added optional retention of eligible Python function definitions between otherwise ephemeral tool processes. This is source-based function restoration into fresh runtime globals, not a persistent Python kernel: ordinary variables, mutable simulation state, and arbitrary objects are not automatically carried over.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PERSISTENT_FUNCTIONS` | Off | Enables function collection, validation, restoration, and feedback. |
| `ARC3_PERSISTENT_FUNCTIONS_SCOPE` | `levelup` | `turn`: retain within an analyzer turn; `levelup`: retain until a level change; `game`: retain throughout the game. |
| `ARC3_PERSISTENT_FUNCTIONS_IMPORTS` | Off | Also retains supported explicit import dependencies needed by retained definitions. |
| `ARC3_PERSISTENT_FUNCTIONS_REPAIR_HINTS` | Off | Enables extended instructions for repairing rejected function retention. |

The retention mechanism checks top-level definitions and their dependencies, including dependencies on other retained functions. Unsupported definitions or unavailable globals are rejected rather than silently restored in an invalid form. Imports inside a function remain a way to make the function self-contained without enabling import retention.

Prompt wording explicitly calls these functions **the model’s own retained definitions**, avoiding the impression that they are built-in harness helpers. With game scope, guidance reminds the model to recheck level-specific assumptions. When retention is disabled, the ordinary ephemeral-session instructions remain in effect.

Robustness fixes retain valid functions even when a snippet ends with a stale-state or other handled guard exception, rather than treating every such interruption as loss of definitions. A call the sandbox never answers (a timeout kill, a crash, an invalid or unknown reply, or a failed start) leaves the previous pool unchanged and adds no retention notes; any definitions or redefinitions made by that call are lost, so a function it redefined silently reverts to its earlier retained version. Unexpected retention bookkeeping errors degrade to dropping functions and giving feedback; they must not replace the original output/error, replay actions, or halt the game.

Feedback deduplicates rejection notices within a result and avoids repeatedly reporting unchanged successful definitions as new. Import-retention instructions are gated with the import feature. Extended repair instructions are separately gated; the short baseline advice about passing snippet-specific data as arguments or redefining it remains.

## 8. Context estimation, rolling history, and pruning

### More realistic context estimates

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_TEXT_TOKEN_CHARS` | `3` | Initial/fallback characters-per-token estimate. |
| `ARC3_CALIBRATE_TEXT_TOKENS` | On | Calibrates the text estimate from backend prompt-usage reports, accounting separately for images. |
| `ARC3_TEXT_TOKEN_CHARS_MIN` | `1.0` | Lower bound on calibrated characters per token. |
| `ARC3_TEXT_TOKEN_CHARS_MAX` | `3.3` | Upper bound on calibrated characters per token. |
| `ARC3_IMAGE_TOKEN_ESTIMATE` | On | Estimates image cost from decoded PNG dimensions rather than charging for the base64 string as text. |
| `ARC3_IMAGE_TOKENS_FLAT` | `0` | A positive value overrides dimension-based estimation with a fixed cost per image. |

The dimension estimate uses 32-pixel patches plus two image marker tokens. For example, a 256×256 image estimates 64 patch tokens plus two markers. It is an estimator, not a guarantee that every backend/vision encoder uses exactly this tokenization.

### Hysteresis and configurable history limits

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_CONTEXT_DRAIN_TOKENS` | `0` | After reaching the context limit, removes additional old history in blocks, leaving headroom and reducing the frequency of prefix-changing trims. The drain is capped at half the budget (window − max output − 512), and a trim keeps `budget − drain`: the notebook's 58 Ki exceeds the 59,136 cap, so it keeps 59,136 tokens. A smaller drain keeps more history and trims more often. |
| `ARC3_CONTEXT_DRAIN_FLOOR` | Off | Stops optional extra draining before crossing its target when the mandatory context limit has already been satisfied. |
| `ARC3_HISTORY_ASSISTANT_TURNS` | `30` | Configures the separate assistant-message history cap. Zero or negative disables it. Counts assistant messages, not game actions or whole analyzer turns. |
| `ARC3_HISTORY_TURN_DRAIN` | `0` | Adds hysteresis to the assistant-message cap. |
| `ARC3_HISTORY_DRAIN_COALESCE` | Off | Coordinates token- and message-count drains to avoid closely spaced independent trims. |

The original `LOCAL_ANALYZER_CONTEXT_WINDOW` and `LOCAL_ANALYZER_MAX_OUTPUT` still determine the ordinary prompt budget, reserving generation space plus a safety margin. A large context setting does not override a smaller assistant-message cap or a smaller server context limit.

### Optional control-message pruning

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PRUNE_CONTROL_CONTEXT` | Off | Master switch for pruning supported old harness-control messages. |
| `ARC3_PRUNE_NUDGES` | Off | Independently prunes obsolete nudge messages. |
| `ARC3_PRUNE_RESUME_PROMPTS` | Off | Independently prunes old resumption messages. |
| `ARC3_PRUNE_DEAD_STUBS` | Off | Independently prunes obsolete dead-reasoning placeholders. |
| `ARC3_STUB_DEAD_REASONING` | Off | Replaces unproductive reasoning-only replies with compact placeholders in retained context. |

Internal tags distinguish control messages without leaking those tags into requests. Pruning repairs the trailing user-message structure and records that the context has changed, so scheduling/prefix-state bookkeeping remains consistent.

**Why turn on `ARC3_STUB_DEAD_REASONING` (issue #9).** In the `fz` control, about 1.1% of replies hit the 12k output cap with no tool call. They carry about 7% of generated tokens, and the cut-off reasoning (about 12k tokens) stays in context until the next trim.

- The stub drops that dead trace from context only; the transcript keeps it.
- Replies that did produce a tool call are untouched, so the retained-thinking behaviour that WRITEUP says matters is kept.
- The truncated replies are long, non-repeating reasoning, so a repetition guard would not catch them.

## 9. Rolling summaries

Added periodic model-written summaries as a separate memory mechanism from the structured world-model sections.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_SUMMARY_INTERVAL_TOKENS` | `0`: disabled | Requests another summary after this much newly appended context has accumulated since the preceding attempt. |
| `ARC3_SUMMARY_REPLACES_HISTORY` | Off | Off appends summaries in the stream; on replaces older history after a successful summary while retaining the summary context. |
| `ARC3_DRAIN_STOP_AT_SUMMARIES` | Off | Allows draining to stop at a useful summary boundary once the mandatory context constraint is met. |
| `ARC3_HIDE_FOLLOWUP_SUMMARIES` | Off | Filters completed summary exchanges from outgoing context according to eviction state, without deleting stored history. |
| `ARC3_SUMMARY_TURN_CONTEXT` | Off | Adds recent action/current-state information to the summary-writing request. |
| `ARC3_SUMMARY_ENABLE_THINKING` | Off | Controls thinking for the summary request separately from ordinary gameplay requests. |
| `ARC3_SUMMARY_MAX_GEN` | Normal positive output cap, otherwise `8192` | Bounds summary generation even when normal generation is configured as unlimited. |

### Summary correctness fixes

- The interval measures cumulative **new context**, including generated text, inserted prompts, tool outputs, and estimated image costs. It is neither generated-token-only accounting nor the size of the surviving rolling window. Re-sent history, static system/tool-schema text, and the summary exchange itself do not advance this interval.
- Every summary attempt resets the interval baseline, successful or not. A failed summary can be skipped and tried again after another interval; trimming cannot leave an old absolute threshold permanently unreachable.
- The visibility filter preserves a live request to write a summary. It only hides completed summary exchanges, so it cannot silently remove the request before the model answers it.
- Successful history replacement sets both the context-trim flag and the history-eviction flag. Scheduling, visibility filtering, and memory-rebuild behavior therefore see the same event.
- Summary calls receive the available timeout. Empty/failed responses are skipped; useful truncated summaries are marked incomplete. Tool calls produced during summary writing are not executed.
- Summary wording retains useful current facts and uncertainty/alternative hypotheses rather than demanding a complicated new memory taxonomy.

Summary output length is a separate request limit; it does not change the normal gameplay generation reservation. Normal context-overflow recovery trims and retries ordinary requests. A failed summary request can instead be skipped until its next interval.

## 10. Structured memory / world-model options

The original structured memory mechanism remains, with configurable subsets, lifecycle behavior, and more robust extraction.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEMORY_SECTIONS` | All sections | Selects the carried memory sections. `slim` keeps world model and plan; `medium` also keeps the action model; `off` disables this structured-memory path. |
| `ARC3_WM_SECTION_EXPLANATION` | `tufa` | `full` expands the section descriptions, update semantics, and examples; default keeps the original style. |
| `ARC3_WM_WIPE_ON_LEVEL` | `1` | Clears appropriate memory at level changes; can be disabled to retain it. |
| `ARC3_WM_WIPE_ON_GAME_OVER` | `1` | Clears memory after death; can be disabled to preserve learned mechanics. |
| `ARC3_WM_REVISE_NUDGES` | Off | Requests revision when memory is retained across a level transition or death. |
| `ARC3_WM_REBUILD_NUDGE` | Off | Requests rebuilding memory after it has been cleared. |
| `ARC3_WM_NUDGE_TURNS` | `0`: disabled | Nudges after this many analysis steps without a memory update. |
| `ARC3_TOLERANT_SECTION_HEADERS` | Off | Recognizes decorated or slightly rephrased headers, including Markdown, “revised,” “updated,” and short annotations. |
| `ARC3_ALIAS_RESPECTS_STORED` | On | Prevents an alias-derived section from unintentionally replacing an existing canonical section. |
| `ARC3_MEMORY_SECTION_MAX_CHARS` | `8000` | Caps individual stored sections; zero disables the cap. |
| `ARC3_DEGENERACY_RATIO` | `0.01` | Detects extremely repetitive text by compression ratio; zero disables the check. |
| `ARC3_DEGENERACY_MIN_CHARS` | `500` | Minimum text length for the repetition check. |

Section updates accompanying tool calls are processed rather than silently lost. Oversized or degenerate memory writes cannot repeatedly inject an enormous corrupted block into later prompts. Degenerate assistant output also has a recovery path instead of being treated as useful context.

Disabling structured memory removes its storage/parsing and most associated prompt scaffolding; this is separate from rolling summaries and retained Python functions. It is not a guarantee that no legacy prompt sentence ever uses the words “world model.”


### Memory ledger shown after a trim

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_LEDGER_ON_TRIM` | Off | Keeps a ledger of what the model records in tags. It is shown once, at the head of the retained history, right after a context trim. |

The model records findings by wrapping single lines in tags, in its reply or its reasoning:
- `<world_model>`
- `<hypothesis status="confirmed|refuted">`
- `<level_note>`

When a trim drops old turns, the oldest remaining message starts with a `<memory>` block holding the ledger. The trim had already invalidated the cached prefix from that point, so the block costs no cache hits, and it stays byte-stable until the next trim. That is unlike the Wang fork, which re-rendered its ledger every turn and broke prefix caching.

Evidence: ported from the `fz` harness (commit `98cc560`). There it scored 52.56 and 49.14 offline vs control 46.25 (25 games, 121 min), above the control-like runs of that time (44.1–46.4), but neither run passes `make significance` (P = 0.896 and 0.651 against 0.90). The later `v4-control` v1 at `519c4f5`, without the ledger, scored 50.62, above 49.14. The port has never run alone: it ran inside `tput-safe` v1 (42.09) and every `user-combo` run, including the recommended `user-combo-10x` (56.94, 52.70, 49.09).

### memory-v2: harness facts in the memory (`ARC3_MEM_*`)

The largest measured loss is evidence the model already had and did not use, not trim forgetting: on the r2 ledger run, 9.0% of actions repeat an exact (board, action) pair (97.8% of them still in the context window), 27% sit on net-zero loops, and 15% of tokens go to attempts that die. The model-written tags are rare (2.7-3.9 per 100 calls). So the harness records facts from every executed action (batch members included) and shows them; the model's tags become one section of the card.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEM_SOLVED` | Off | (A) A `<solved>` line per cleared level, never evicted: the winning attempt's path with loops and no-change actions cut out, the actions spent on the level, its deaths, then (as room allows within 165 chars) the model's `<level_note>` for it and one whole sentence of the clearing turn's reasoning (`how`). `how` needs an action and a number or finding word, and skips plans, commands and hedges ("Just try UP now", "Execute: DOWN, DOWN", "One more UP should push it in"): the clearing turn mostly plans its last moves, and an imperative could be read as advice on the next level. Without `ARC3_MEM_CARD` (the `solved-levels` arm) stack-v2's level notes are kept as they were: each level's think-trace findings and open hypotheses are still folded into a `<level_note>` when the level ends, and only a model-written `<level_note>` goes on the solved line. (Before this was fixed, `ARC3_MEM_SOLVED` alone dropped those notes too; no Kaggle run used it without the card.) |
| `ARC3_MEM_CARD` | Off | (B) The card's rules for the model's own notes: entries up to 400 (world model), 250 (hypothesis) and 200 (level note) chars; a new entry replaces an older one in the same list at word overlap >= 0.6, and a confirmed/refuted pair at >= 0.6 keeps only the newer; `Correction:`/`Fix:`/`Actually,` replaces the best match (>= 0.35); think-trace fragments are no longer turned into level notes. Tags inside python code are handled by `ARC3_MEM_CODE_TAGS`, which this flag turns on. Its system-prompt text replaces the ledger's. |
| `ARC3_MEM_EFFECTS` | Off | (C) An `<actions level="N">` section: what each key (or a click on each colour, `MOUSE@<letter>`) did on level N and in the whole game, as counts (moved, changed, edge only, no change, anim only, death, clear), and `<hazard>` lines for the newest 3 deaths of the current level. Both are card sections; the delta lines add `<key>: first <effect>` the first time in the game a key has an effect, and each GAME OVER line. |
| `ARC3_MEM_SEEN` | Off | (D) A `Seen before` line when the current board's interior matches a board visited earlier on this level: when it was visited and what was tried from it. Each tried action is worded when the line is shown: no board change, GAME OVER, or the board it reached, named by one of that board's visits (`the board after action #10`), since a board that was new when the action was tried has been seen since. |
| `ARC3_MEMORY_PLACEMENT` | `trim` | (E) `trim`: the card only at the head of the retained history after a trim (as `ARC3_LEDGER_ON_TRIM`). `delta`: no card; a `<memory_update>` block in each new turn message with the lines added since they were last shown. Those lines are plain text (`+ solved L1: ...`, `+ UP: first moved`, `+ L2: attempt 2 ended in GAME OVER ...`), and the system text describes these forms rather than the card's tags. The per-key count table of `ARC3_MEM_EFFECTS` exists only on the card, so under `delta` component C is reduced to its first-effect and GAME OVER lines: a `delta` vs `trim`/`both` comparison is not placement alone. `both`: the card after each trim plus the deltas. `Seen before` lines ignore this setting. It applies only with a memory-v2 flag on: with only `ARC3_LEDGER_ON_TRIM`, which has no deltas, the ledger is still shown after each trim. |
| `ARC3_MEM_LINES_AFTER_TOKENS` | `15000` | The per-turn lines (`Seen before` and deltas) wait until the current level has used this many generated tokens. Solved lines, a key's first death, clears and the trim card are not held. `0` shows them from the level's first turn. Hazard and first-effect lines still waiting when the level ends are dropped, not shown on the next level. |
| `ARC3_MEM_STATE_BORDER` | `ARC3_NOOP_GUARD_BORDER` (4) | Edge cells cropped before two boards count as the same board, so a ticking bar does not make every board new. By default the no-op guard's border, so the memory and the guard agree on what changed the board. |
| `ARC3_MEM_CARD_CHARS` | `3900` | Hard cap on the card (about 1.3k tokens). Over it: the oldest non-refuted notes go first, then older solved lines lose `how` and `note`, then their paths (`L1-L4: 13/22/9/31 acts, spent ..., deaths ...`), then action rows. |
| `ARC3_MEM_DELTA_CHARS` | `480` | Per-message cap on deltas; the rest is carried, and the third message in a row that carries doubles the cap. |
| `ARC3_MEM_DELTA_GAME_CHARS` | `12000` | Deltas stop for the game after this many chars; head cards continue. |
| `ARC3_MEM_SEEN_GAME_CHARS` | `8000` | `Seen before` lines stop for the game after this many chars. |
| `ARC3_MEM_REPLY_WINS` | Off | Absorb an assistant message's reasoning before its reply, the order they were generated in. Off, the reply is absorbed first and the reasoning second, so with `ARC3_MEM_CARD` a tag drafted in the reasoning is the "newer" entry: it replaces an overlapping tag in the reply, and a drafted `confirmed` deletes the reply's `refuted` twin (any mode: a scratchpad in the reasoning replaces the reply's). memory-v2-both and panel-memory-v2-both ran with it off; it is a separate flag so those results keep their meaning. Turn it on for every card arm in the next comparison. |
| `ARC3_MEM_CODE_TAGS` | On with any memory flag | Memory tags written inside python code (`<world_model>`, `<hypothesis>`, `<level_note>`, bare or in comments) are recorded in the ledger and stripped before the code runs, when the stripped code still compiles; the stored assistant message keeps them. A bare tag in code is otherwise a SyntaxError and the call is lost. Until `arm/memory-main` this came only with `ARC3_MEM_CARD`; unset, it is now on whenever `ARC3_MEM_SOLVED`, `ARC3_MEM_CARD`, `ARC3_MEM_EFFECTS`, `ARC3_MEM_SEEN` or `ARC3_MEM_PATH` is on. `1` turns it on for any ledger, `ARC3_LEDGER_ON_TRIM` alone included (main's `user-combo-10x` runs the code as written); `0` turns it off. |
| `ARC3_MEM_DEBRIEF` | Off | (Audit fix, 2026-10-10; needs `ARC3_MEM_SOLVED`, meant with `ARC3_MEM_CARD`.) A solved line drops `how` and the level note and carries `debrief:`, the model's own recap of the level: after a level change the first two reasonings are searched (`memory_facts.extract_debrief`) for "Level N solved/complete...", "previous level" or a "Key mechanics:"-style label, and the text from that sentence up to where the reasoning turns to the new level ("Now level 2", "Let me inspect") is kept, whole sentences, at most 300 chars, only if it names a goal or rule word. Since the PR #148 review it is also kept only when it is about the cleared level: the first level number it names must be that level (or the new level's number followed by it, "Level 2 solved level 1"), at least 40 chars with a goal or rule word must remain without its opening celebration ("Level 4 completed! Wait -"; an arrow alone is a move trace), and a recap cut mid-sentence ends at its last whole sentence; a rejected reasoning leaves the second try open. Before that fix 14 of 83 captures on the three audited runs named another level and 4 were no recap; after it 22, 26 and 25 recaps on 96, 100 and 113 level-ups (17-19 of 25 games), none naming another level. On the audited runs the first version found a recap at 24-31 of 96-113 level-ups, in 19-20 of 25 games; `how` described the level's route in 0 of 25 cases. A recap found after the solved line was written is added to it in place (the card is drawn only at the next trim). World-model and hypothesis notes get an `L<N>: ` label. The card order becomes solved lines, model notes, hazards, action rows; over `ARC3_MEM_CARD_CHARS` it cuts action rows, then the paths of solved lines older than the newest 3, then all paths, then all but the newest hazard, then the oldest notes (refuted last), and debriefs last. The system prompt describes the head card as "holding what you recorded and what the harness recorded, so findings from dropped turns are not lost", adds "After a level is cleared, the harness copies your recap of it from your first reasoning on the next level ("Level N solved: ...") onto that level's `<solved>` line as `debrief:`." and closes the tag examples with the ledger's line about refuted hypotheses instead of the scope line. |
| `ARC3_MEM_KEEP_EDGE` | Off | (Audit fix, 2026-10-10.) The solved path keeps actions that changed only the edge band (`edge only`) or played an animation (`anim only`), and no loop is cut back across one: they can change state the cropped interior does not show (ar25's selection marker). Off, they are dropped, and ar25 L3's path in level-path-10x did not clear when replayed; on, 104 of 104 and 106 of 106 solved paths of level-path-10x and mem-v2-10x clear in exactly their length in the offline engine, at +1.1 and +2.1 actions per path. Routes (`ARC3_MEM_PATH`) are unchanged. |
| `ARC3_MEM_EVERY_CALL` | Off | (The user's layout, 2026-10-10; needs `ARC3_MEM_SOLVED`, does nothing without it.) The memory goes into every request, in two places, and the trim-only `<memory>` card, the per-turn `Seen before` line and the `<memory_update>` lines are no longer sent. (1) `<previous_levels>`: its own user message right after the system prompt, so it is never glued to an old turn message next to a stale `Current state` line. One entry per cleared level: actions taken and spent, deaths, the winning route, the model's recap (`ARC3_MEM_DEBRIEF`; with `arm/level-debrief` merged, its pinned GOAL and MECHANICS as `Goal:` and `Rules:`) and up to 4 of the model's notes on the level, frozen when the level is committed. A cleared level is committed at the first turn start after its recap settles (found, or the two reasonings checked), and every pending level at a trim; only then is the message redrawn. A redraw breaks the cached prefix from message 1, so it is priced like a trim (`_context_was_trimmed`: the slot is handed over and the next request re-prefills). Over `ARC3_MEM_HEAD_CHARS` the oldest entries lose notes, then routes, then all but their head line. The trimmer reserves the message's tokens. (2) `<this_level>`: the last text of each full turn message (not of resumption messages): a summary line, what each action and each click colour has done on this level, actions not tried, the newest two GAME OVERs (action and board), the Seen-before line, and the model's notes on this level, newest first, capped at `ARC3_MEM_LEVEL_CHARS`. It sits in the newest message, so it never breaks the prefix. Notes: open hypotheses (`<hypothesis>` without a status) are kept per level until confirmed or refuted, and every note is labelled with the level it names ("... completes level 1" written on level 2 goes to level 1) unless that level is already committed. The system prompt states where both blocks are and where notes appear. Replayed on user-combo-10x v1's logged requests and replies through the Qwen3.8 chat template: head 359 tokens on average (max 1,041), `<this_level>` 152 (max 344); without compaction +41.5% prefill (one full re-prefill per clear). |
| `ARC3_MEM_LEGIBLE` | Off | Every memory line in plain words (`memory_layout.py`): labels, full action names, `x` counts, clicks as `click (row, col)`, colour names, one fact per line. `<solved>L1: 17 acts [U L D11 L4], spent 25, 0 deaths</solved>` becomes `Level 1: cleared in 17 actions (25 spent, no deaths). Route: UP, LEFT, DOWN x11, LEFT x4.`; `UP: moved 3, changed 4 \| game moved 4, changed 4, no change 1` becomes `UP (7 presses on this level): moved cells 3 times, changed cell colours 4 times. In the whole game: ...`. Applies to the card's solved, action and hazard lines, the delta lines, the `Seen before` line and both `ARC3_MEM_EVERY_CALL` blocks; the system prompt's fact lines drop the letter legend. Does nothing without a memory flag. |
| `ARC3_LEVEL_COMPACT` | Off | (Needs `ARC3_MEM_EVERY_CALL`.) A level clear is a compaction point: when a cleared level is committed to `<previous_levels>`, the turns built on it (read from their `Current state ... level N` line) are dropped except the last `ARC3_LEVEL_COMPACT_KEEP`, in the same step as the redraw, and if what is left is above the drain target (`ARC3_CONTEXT_DRAIN_TOKENS` below the budget) it is drained there too, so a clear is one re-prefill and no ordinary trim follows a few requests later. The system prompt states the drop. Replayed on user-combo-10x v1: 8.3 prefix breaks per game against 5.7 (every extra one a commit; 18 commits fell on a logged trim and 29 logged trims became unnecessary), prefill +3.2% (+15.9k tokens, about 1.3 s per game at 12k tok/s), mean prompt 47.8k tokens against 54.9k. |
| `ARC3_LEVEL_COMPACT_KEEP` | `2` | Turns of a cleared level kept by `ARC3_LEVEL_COMPACT` (at least 1). |
| `ARC3_MEM_RECAP_ASK` | Off | (Needs `ARC3_MEM_DEBRIEF` and `ARC3_MEM_SOLVED`; the ask is sent only with `ARC3_MEM_EVERY_CALL`.) The level-start message ends with a request for `Level N recap:` (one or two sentences on the goal of the level just cleared and the rules confirmed on it), and the recap is read by `memory_facts.extract_recap`: a labelled recap of the cleared level anywhere in the first two reasonings on the new level (`Level N recap:`, `Level N summary ...:`, `Level N mechanics ...:`; `Level N+1 recap:` too), else the precise `ARC3_MEM_DEBRIEF` recap, else the reasoning's opening when it starts on the clear (either number) and states a rule. The system prompt's recap line says the next turn message asks for it. On mem-layout-10x's reasonings (before any ask) it keeps 44 of 106 recaps against 24. |
| `ARC3_LEVEL_COMPACT_NEEDS_RECAP` | Off | With `ARC3_LEVEL_COMPACT`: a cleared level's turns are dropped at its commit only when the level has the model's recap; a level without one keeps its turns until an ordinary trim. The system prompt's compaction line says so. |
| `ARC3_MEM_HEAD_CHARS` | `6000` | Cap on `<previous_levels>` (about 1.7k tokens). |
| `ARC3_MEM_LEVEL_CHARS` | `1100` | Cap on `<this_level>` (about 300 tokens). Over it: older GAME OVERs, then action rows from the end, then the oldest notes, then the Seen-before line; the summary line stays. |

Any of A-D creates the ledger, so they also work without `ARC3_LEDGER_ON_TRIM`; the arms run them with it, as stack-v2 does. With every `ARC3_MEM_*` flag off, the requests are byte-identical to stack-v2 (`tests/test_memory_flags_off_identity.py`) and, on the user-combo-10x stack, to main (`tests/test_memory_main_loop.py`).

**What the model sees.** Replayed from the r2 ledger run's frames (`fz-ledger-on-trim-r2`, with UP DOWN LEFT RIGHT SPACE taken as the valid actions). The first turn message of ls20's level 2 (a solved line passes the hold; L1's clearing turn only planned its last move, so the line has no `how`):

```
<memory_update>
+ solved L1: 13 acts [L3 U4 R3 U3], spent 40, 0 deaths
</memory_update>
```

A turn message on ls20's level 3, back on a board first reached 36 actions earlier:

```
Seen before (board interior matches): after action #8 and #44 of this attempt. From it you tried: RIGHT -> no board change, LEFT -> the board after action #10. Untried: UP, DOWN, SPACE.
```

The head card after a trim, at the end of ls20 (two of four solved lines, two of five action rows):

```
<memory>
  <solved>L1: 13 acts [L3 U4 R3 U3], spent 40, 0 deaths</solved>
  <solved>L2: 45 acts [U R U5 R2 D R D7 L2 R2 U D U8 L6 D L D5], spent 124, 1 death | how: The 4th DOWN takes us to (35,14); the 5th presses into the box.</solved>
  <actions level="5">
  UP: moved 5, changed 7, no change 1 | game clear 1, moved 89, changed 31, edge only 2, no change 2
  LEFT: moved 12, changed 7, edge only 1 | game clear 1, death 1 (1 board), moved 65, changed 17, edge only 6
  </actions>
</memory>
```

A click game (vc33, level 4) groups clicks by the colour clicked:

```
  <actions level="4">
  MOUSE@b 6 targets: death 1, moved 7, changed 68, edge only 6; median 63 cells; e.g. (61,9) changed, (61,39) changed
  MOUSE@w 1 target: edge only 3; e.g. (48,13) edge only
  colours on the board not clicked this level: W c B
  </actions>
  <hazard>L4: attempt 1 ended in GAME OVER after 50 actions; last action M(61,15) from the board first reached at action 48 of the attempt</hazard>
```

Harness text states counts and boards, never verdicts: none of "inert", "refuted", "impossible", "useless" or "cannot" (tested). A key that never changed the interior reads `SPACE: no interior change on 4 presses across 2 boards`.

**Where it goes, and why it is cache-safe.**
- The recorders (`memory_facts.FactRecorder`) run on every new turn message, right after the death ledger's recorder, whether or not anything is shown.
- The per-turn block (`Seen before`, then `<memory_update>`) goes into the new turn message only, after the valid-actions line (where the facts-stack `Salient:`, `Far change:` and `Step bar` lines go). It is composed once and never edited, so every request extends the last one until a trim (tested in all three placements).
- The card goes only through `_apply_due_ledger`, after a real trim, as the ledger does; `delta` sets no card. Nothing per-game enters the system message; its text depends only on which flags are on.
- A resumed turn (after a yield) gets no lines; they wait for the next full message. A turn that is abandoned and rolled back gives its deltas back, and the card flag too (when the trim came from the previous turn's persist, the rolled-back history is already trimmed and nothing else would set it).
- Nothing here calls `_note_history_evicted`; `tests/test_memory_eviction.py` runs all three placements with every line shown (`mem-trim`, `mem-delta`, `mem-both`).

**Cost (r2 replay, 25 games, no hold).** Recorders 0.39 ms per action (median, 0.7 ms max). `Seen before` on 13.8% of turn messages, mean 194 chars; deltas on 39%, mean 91 chars; about 900 tokens per game together. The final card: median 1.0k chars, max 1.4k. The 15k hold removes most per-turn lines below that point, where 85% of exact repeats and 75% of revisits do not happen.

**Known gaps.** A start-to-end object diff (`end`) on the solved line was tried and removed: within its 90 chars it fit 1 of 112 r2 level clears (100 of 112 level ends change 6 or more objects), at a median 114 ms per clear. `how` is present on 27 of 112 (65 before plans and score-2 sentences were skipped). Board identity crops the edge, so two boards that differ only in hidden state count as one ("board interior matches").

**Read-outs for the arms** (data-card scripts `m3_revisit`, `m4_deaths`, `m5`, against stack-v2, ledger-on-trim and control on matched seeds):
- exact (board, action) repeats: 9.0% of actions, target 6% or less;
- actions on net-zero loops (27%), no-op rate (5.9%), share of tokens in attempts that die (15%), next level cheaper than the last (34% of pairs);
- points lost to the action term, P(clear | k tokens on the level) and tokens per cleared level, levels per 100 active game-minutes.

Keep/drop rule, fixed before the runs: keep an arm if the mean of two runs is at least stack-v2 + 3 and exact repeats are 6% or less; drop it if it is at most stack-v2 - 1, or if repeats do not fall and its gain over control is inside control noise (44.1-46.4).

### level-path: the shortest route on record (`ARC3_MEM_PATH`)

The model re-walks boards it has already reached: 27% of r2's actions sit on net-zero loops and 9.0% repeat an exact (board, action) pair, and after a death it walks back from the start by hand. The harness already sees every board. With this flag it keeps, per level, the links between boards that the observed actions made, and after every action shows the fewest-link route from the level's start board to the current board, and how that changed.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEM_PATH` | Off | The route after every `action(...)` call (echo suffix), a `Shortest route on record` line in each turn message after actions on the level, `level_path` in python, the solved line's route, and its system-prompt line. Needs no ledger: it works alone or on top of memory-v2. |
| `ARC3_MEM_PATH_AFTER_TOKENS` | `0` | The turn line waits until the current level has used this many generated tokens (the system prompt says so when it is nonzero). `0` shows it from the level's first turn. The echo suffix and `level_path` are never held. |
| `ARC3_MEM_PATH_GAME_CHARS` | `16000` | Turn lines stop for the game after this many chars (3 of r2's 25 games would reach it); the system prompt states the cap. |

**The graph** (`memory_facts.LevelGraph`, owned by `FactRecorder`, one per level). Nodes are board hashes with the outer `ARC3_MEM_STATE_BORDER` cells cropped (4 by default, as memory-v2). Only a board-changing action adds a link `board -> board`; no-change actions, GAME OVER, clears, RESET and UNDO add none (UNDO depends on history, not on the board). A click is its exact cell, `MOUSE(r,c)`. A (board, action) pair keeps up to 4 outcomes; one seen with two outcomes, or with a move and also a death or no change, is "varied". Caps: 1024 boards, 16 actions per board, 4 outcomes; over a cap the link is refused and counted (`stats["graph_full"]`), never evicted, so a recorded link stays and, while the start board is the same, a distance never grows (a RESET that lands on another board moves the source; no recorded level had two start boards). The routes are a breadth-first search from the current attempt's start board, recomputed whenever a link or the start changes. A RESET puts the current board back on the start (route 0); a new level starts a new graph.

**What the model sees.**

System prompt, after the `Seen before` line (with memory-v2) or on its own:

```
- Route facts come from links the harness records on each level: a board-changing action links one board to the next (boards compared with the outer 4 cells cropped; UNDO, RESET, GAME OVER, clears and no-change actions add no link; a click is its exact cell). The route is the fewest links from this level's start board to the current board. Each action(...) call prints route N and what changed; a turn message after actions on the current level carries a Shortest route on record line, up to 16000 characters of these lines per game. In python, level_path is that route as a list action(...) accepts and level_path_len its length.
```

The cap is `ARC3_MEM_PATH_GAME_CHARS`; a nonzero `ARC3_MEM_PATH_AFTER_TOKENS` adds ` once the level has used N generated tokens` after `line`. Both are read when the prompt is built, so the line stays the same for the run.

The python tool's description adds `` `level_path` is the shortest route on record from this level's start board to the current board, as a list action(...) accepts, and `level_path_len` its length; `level_path_before_death` is the route on record from the start board to the last board before this level's latest GAME OVER (None before one); all refresh after every action(...) call.`` (+347 chars of tool schema), and the turn message's `Only tool:` line names `` `level_path`, `level_path_len`, `level_path_before_death` `` (+59 chars per turn message over the flag off). With `ARC3_MEM_SOLVED`, the `<solved>` sentence reads "the shortest route on record to the winning board, then the winning action (...); "(K in the winning attempt)" follows when that attempt, with its loops cut out, was longer; then the actions spent on the level and its deaths."

Each `action(...)` call's echo line (printed into the tool result being built) ends with the route after it. From the r2 replay:

```
[action] LEFT -> effect | route 1 (new board)
[action] batch of 4 actions, 4 executed: LEFT, UP, UP, UP | route 2->6 (3 4 5 6)
[action] LEFT -> effect | route 14 (shorter than 23 on record)
[action] LEFT -> GAME OVER (attempt ended immediately after this action) | route - (GAME OVER from a board at route 1)
[action] UP -> LEVEL COMPLETED | route - (level cleared; 13 on record)
[action] batch of 5 actions, 5 executed: DOWN, DOWN, DOWN, DOWN, DOWN | route 40->C (41 42 43 44 C)
```

Other single-action texts: `known board` (on record before this action, including a board whose first route this action's link made; 0 such rows in r2 or fz-experts8), `no board change`, `start board` (a RESET). In a batch, `X` is GAME OVER, `C` a clear, `-` a board no link reaches (the arrow gets spaces when a side is `-`: `route - -> 1 (0 1)`); past 12 values the first and last 5 show, and `; shorter route found` is added when one was. At most 80 chars; automatic RESETs are not shown. With `ARC3_ACTION_ECHO` off the suffix prints as its own `[route] route 1 (new board)` line.

The turn message, after the `Seen before` line and before `<memory_update>` (r2 replay):

```
Shortest route on record from this level's start board to the current board: 18 actions [U5 L2 D2 U3 D2 L3 D]; this attempt so far: 61 actions. A shorter route to a known board was found at action #56 of this attempt (25 -> 19).
Shortest route on record from this level's start board to the current board: 0 actions (this is the start board). Attempt 1 ended in GAME OVER after 7 actions; its last board before GAME OVER is 5 actions from the start on record [M(33,39) M(27,39) R3].
Shortest route on record from this level's start board to the current board: 12 actions [M(19,52) D R M(28,46) M(28,56)x2 M(28,46)x4 M(28,56)x2]; this attempt so far: 31 actions. 2 steps on the route had more than one outcome on this level.
```

and on the scripted test sequence (start, RIGHT, RIGHT, DOWN, UP, a manual RESET, DOWN to the board first reached third):

```
Shortest route on record from this level's start board to the current board: 1 action [D]; this attempt so far: 1 action. Route length after each action since the last message: 0 1. A shorter route to a known board was found at action #1 of this attempt (3 -> 1).
```

The per-action list covers this attempt's actions since the last turn message, and reads `Route length after each action of attempt N since the last message` when an earlier attempt (ended by GAME OVER or a RESET) also acted since then; `0 actions (this is the start board)` and `none on record` (no recorded links reach the board, e.g. after UNDO) are the other route texts. At most 300 chars: the list goes first, then the varied and shorter notes.

In python: `level_path` (`["RIGHT", {"action": "MOUSE", "row": 4, "col": 7}]`, `[]` on the start board, `None` when no links reach the board), `level_path_len`, `level_path_varied` (0-based indexes of varied steps), and `level_path_before_death`, the route on record from the start board to the last board before this level's latest GAME OVER, in the same form (`None` before a GAME OVER on the level or when no links reach that board; `[]` when it was the start board). They refresh at snippet start and after every `action(...)` call. Every route starts at the attempt's start board, so it replays as written only from there: right after a GAME OVER's automatic RESET, `level_path` is `[]` and `action(level_path_before_death)` walks back to the board before the death, while `action(level_path)` on the start board raises (`action([])` needs at least one action). The turn line's own copy of the pre-death route is cut at 40 chars (`...+N`); the python list is whole.

With `ARC3_MEM_SOLVED`, a solved line uses the route on record to the clearing board plus the clearing action when that is strictly shorter than the winning attempt with its loops cut out: `L1: 2 acts [D R] (4 in the winning attempt), spent 6, 0 deaths`. It also uses the route, without the `(K in the winning attempt)` note, when the loop-free walk keeps an UNDO (loop cutting keeps an UNDO that lands on a board cut from the chain; UNDO adds no link, so that walk is no route on record and does not replay from the start board; 0 of r2's 112 clears). Otherwise the line is byte-identical to memory-v2. The one case left: when no recorded links reach the clearing board at all (it was reached only through UNDO, or the level's graph was dropped after a route error), the loop-free walk is kept, UNDO included.

**Where it goes, and why it is cache-safe.**
- The system line, the tool description and the `Only tool:` line depend only on the flag and the border: the system message and the tool schema are the same object on every request (tested).
- The echo suffix goes only into the tool result being built: `_handle_action` builds the sandbox state first (which feeds the recorder the call's actions), then appends the suffix to that call's echo.
- The turn line goes only into the new turn message, through `_mem_turn_lines`. A resumed turn gets none; a rolled-back turn gives its rows back (`_route_cursor` and `_route_chars_used` are in the rollback snapshot; tested).
- A route error inside the recorder is counted (`stats["route_errors"]`) and drops that level's graph until the next level (no echo suffix or turn line, `level_path` is `None`); the memory-v2 facts, solved line included, come out as without routes (tested).
- Feeding the recorder mid-snippet only moves the hashing earlier. The events it produces wait in `_mem_pending_events` and are applied by the next turn message ahead of its own, so memory-v2's card and deltas are unchanged (tested).
- Nothing here calls `_note_history_evicted` or edits a sent message. `tests/test_memory_eviction.py` runs `mem-path` (memory-v2 both + path) and `path-only`; both pass all four #58 checks.
- With the path flags off, a memory-v2 arm sends exactly what `arm/memory-v2` sent: system prompt, every request and the tool schema (`tests/fixtures/memory_v2_on_golden.json`, made on d252fa3). Every new string passes the facts-not-advice check (`tests/test_memory_wording.py`).
- Each executed action also writes one line to `*_level_path.jsonl` (the `requests.jsonl` naming): `{act, level, attempt, i, sig, board, d, k, change, varied, path}`.

**Cost (r2 replay, 25 games, 5630 actions, 1100 turn messages).** Echo suffix on 4534 action calls, mean 24 chars (max 56). Turn line on 89% of turn messages, mean 219 chars (p90 259, max 299), about 60 tokens; median 6.7k chars per game. Recorder CPU 1.63 s to 1.73 s for the whole run with the graph; the breadth-first searches take 32 ms in all. On fz-experts8 (10 games): 91% of turn messages, mean 211 chars.

**Offline evidence (r2 replay).** The route on record beats the winning attempt's loop-free walk at 2 of 112 clears (g50t L1, 16 vs 17; sc25 L3, 19 vs 25). After 2.2% of actions (120 of 5495) the current board's route is shorter than the attempt's own loop-free walk. No cap was reached and no level had two start boards. So the line rarely finds a shortcut the model missed; what it adds is the count and route back after deaths and loops, in the turn where the model plans.

**Read-outs** (fixed before the runs): actions on net-zero loops (27%), exact repeats (9.0%), median `k - d` at the end of each attempt, level spend minus the route (median 4), excess actions on re-walks after deaths, share of executing snippets that reference `level_path`, tokens per executing turn (budget +60).

Keep/drop rule, fixed before the runs: keep `level-path` if the mean of two runs is at least the paired `memory-v2-both` control + 2 and the loop share falls by at least 3 points; drop it if it is at most control - 1, or if loops and repeats do not change and `level_path` is in under 1% of snippets.

**Known gaps.** Routes use only links the model's own actions made, so "on record" is a lower bound on what the level allows. Board identity crops the edge, so boards that differ only in hidden state merge; a merged board shows up as a varied pair. Boards reached only through UNDO have no route until a move links them. The turn line repeats the route every turn, even when nothing changed.

### memory-main: the memory arms on the user-combo-10x stack

`arm/memory-main` brings memory-v2 and level-path, written on stack-v2, onto main. Seven arms in `kaggle/arms.json` add memory flags to `user-combo-10x`'s env and serve settings (14 streams, ledger on trim, animation layers, temperature 0.65, `EXPOSE_RESET=on`, `ARC3_OPENER_DEDUP=1`, 10 experts), for full 121-minute runs on all games:

| Arm | Memory flags on top of `user-combo-10x` |
|---|---|
| `mem-solved-10x` | `ARC3_MEM_SOLVED=1`, `ARC3_MEMORY_PLACEMENT=both`, `ARC3_MEM_REPLY_WINS=1` (the `solved-levels` component: solved lines on the plain ledger, no card rules) |
| `mem-v2-10x` | `ARC3_MEM_SOLVED`, `ARC3_MEM_CARD`, `ARC3_MEM_EFFECTS`, `ARC3_MEM_SEEN` (all `1`), `ARC3_MEMORY_PLACEMENT=both`, `ARC3_MEM_REPLY_WINS=1` (`memory-v2-both`) |
| `level-path-10x` | `mem-v2-10x`'s flags plus `ARC3_MEM_PATH=1` |
| `mem-v2-fixed-10x` | `mem-v2-10x`'s flags with `ARC3_MEMORY_PLACEMENT=trim`, plus `ARC3_MEM_DEBRIEF=1` and `ARC3_MEM_KEEP_EDGE=1`: the audit's corrected memory (`research/archive/reports/memory-main.md`, Audit) |
| `mem-layout-10x` | `mem-v2-fixed-10x`'s flags plus `ARC3_MEM_EVERY_CALL=1`, `ARC3_MEM_LEGIBLE=1` and `ARC3_LEVEL_COMPACT=1`: the user's layout (`research/archive/reports/memory-layout.md`) |
| `mem-recap-10x` | `mem-layout-10x` plus `ARC3_MEM_RECAP_ASK=1` and `ARC3_LEVEL_COMPACT_NEEDS_RECAP=1`: the memory holds the model's recap of each cleared level, and a level's turns are dropped only once it has one |
| `mem-recap-keep-10x` | `mem-recap-10x` without `ARC3_LEVEL_COMPACT` and `ARC3_LEVEL_COMPACT_NEEDS_RECAP`: the recap is saved and a cleared level's turns are kept (trimmed only by the window) |

`ARC3_MEM_CODE_TAGS` is on in all of them (any memory flag turns it on). On this stack the per-turn memory lines (Seen before, `<memory_update>`, the route line) sit before the opener's static tail, so `ARC3_OPENER_DEDUP=static` still finds an identical tail to leave out. A 25-minute panel cannot judge these arms: the first trim, and with it the first card, comes around request 65 of about 203 per game. Read-outs and the keep rule are in `research/archive/reports/memory-main.md`.

### memory v2 on pf-keep-10x (`arm/mem-v2`)

Built on `pf-keep-10x` (run B, 55.79). Every flag is off by default; with all of them off (or set to `0`), every request of the game-loop sim under pf-keep-10x's env is byte-identical to `arm/pf-mem` @ 1882bcb3 (`tests/fixtures/pf_keep_golden.json`, `tests/test_memory_v2_loop.py`). They need `ARC3_MEM_EVERY_CALL` (the memory layout) and do nothing without it. Two rules shape them. On this hybrid-Mamba server any edit before the end of the previous request re-prefills the whole context (a median of 58.6k tokens in B), so model-visible text is either appended to the newest message or changed on a request that already trims, and the system prompt stays the same for the whole run. Text states facts, not advice; Franzen's bullets are not reordered; the `Level N recap:` label is kept exactly; every `(old, new)` row goes through `prompts.apply_rows`, which skips (and logs) a row whose old text does not occur exactly once, and each row has a `count(old) == 1` test on the real rendered text.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEM_APPEND_COMMIT` | Off | A settled level clear no longer redraws `<previous_levels>` at message 1 (87 such pure commits in B re-prefilled 4.99M tokens, 31% of its uncached prefill). The same frozen entry is appended once to the newest turn message as a `<cleared_level>` block, after the opener dedup and before `<this_level>`, and folded into `<previous_levels>` at the next trim, a request that re-prefills anyway; the trim strips every `<cleared_level>` block. A turn that is rolled back sends the block again; a level with no later trim keeps its block in history. A clear staged on a trim request folds at once, as before. The staging keeps the commit's slot handover, so scheduling matches B. The memory section's "It changes only when a level is cleared." becomes "A newly cleared level's entry is shown once, in a `<cleared_level>` block near the end of a turn message, until older turns are next dropped from this conversation; then it joins `<previous_levels>`. The python `history` and `transitions` lists are never trimmed: they hold every action of this game." (static per run; the last sentence answers the "python history is trimmed" reading of the ledger wording in runs A and PF). Inert with `ARC3_LEVEL_COMPACT`. Logs `cleared_level staged` and a `MEMORY FOLD AT TRIM` transcript section. |
| `ARC3_MEM_APPEND_NO_HANDOVER` | Off | With `ARC3_MEM_APPEND_COMMIT`: a staging no longer prices the game like a trim or hands its slot over. A scheduling factor (B's commits admitted the waiting games 5-9 min earlier), kept apart from the memory change. |
| `ARC3_MEM_TRIM_STRIP` | Off | On a request a trim already re-prefills, retained turn messages keep only the newest `<this_level>` block of each level (B: a median of 7 copies per request, 1.73% of prompt tokens). The loop's messages are what persist, so the next request extends this one. Changes the context after a trim (and so, slightly, the trim cadence). |
| `ARC3_MEM_RECAP_CHARS` | `0` (420 captured, 300 shown) | With `ARC3_MEM_RECAP_ASK`: one cap, e.g. `800`, for the recap's capture (`extract_recap`) and its `<previous_levels>` line. The capture drops sentences that carry no rule ("Hmm.", "Level 3 completed!", questions, harness talk), never splitting inside brackets; a correction ("Wait - actually ...") is kept together with the sentence it corrects; a cut ends in ` ...` and, when it would lose the only goal sentence, keeps it after ` ... `. On B's recaps at 800: 5 of 105 still cut (82 before), no goal lost (20 of 100 before), about +44 tokens per entry. |
| `ARC3_MEM_COMPRESS_ROUTES_FIRST` | Off | Over `ARC3_MEM_HEAD_CHARS`, `<previous_levels>` is compressed breadth-first: every older entry loses its route (the head line keeps its length) before any recap text is cut, then recaps go to 300 and 160 chars, then the head line alone; the newest entry last. Never triggered on B; the insurance longer recaps need. |
| `ARC3_MEM_ROUTE_FORMAT` | unset | `readable` (needs `ARC3_MEM_LEGIBLE`): routes keep their order and are written to be read: keys run-length (`UP x3`), clicks in the `(row,col)` form the model writes and named by the colour of the clicked cell before the click (`click (48,57) red x4`), up to `ARC3_MEM_ROUTE_CHARS`. A longer route keeps its first runs and its last two, cut only between runs: `UP x8, RIGHT, ... 17 more actions ..., RIGHT, DOWN` (the model quoted route ends in B). On B's 117 clears: 9,450 route tokens against 8,678, 27 routes cut against 28. |
| `ARC3_MEM_ROUTE_CHARS` | `320` | The readable route's cap; read only with `ARC3_MEM_ROUTE_FORMAT=readable` (B's coordinate routes stop at 240). |
| `ARC3_MEM_HEAD_WORDS` | Off | (Needs `ARC3_MEM_LEGIBLE`.) Head lines say what each count is: `Level 2: cleared. 20 actions spent in 2 attempts, no deaths. Winning route without loops (12 actions): ...` instead of `cleared in 12 actions (20 spent, no deaths). Route: ...` (read as spend summed over attempts in B). The clear's attempt number reaches `add_solved` only under this flag, so the solved record's data is pf-keep-10x's with it off (tested). |
| `ARC3_MEM_RECAP_ASK_V2` | Off | (Needs `ARC3_MEM_RECAP_ASK`.) The level-start ask, by an `(old, new)` row on `level_recap_ask`, asks for `Level N recap: Goal: ... Rules: ...` (goal first) and an optional last `Corrects level K: ...` when the level showed an earlier recap's rule was wrong. A recap written that way is shown as the entry's `Goal:` and `Rules:` lines; entry K gets `Corrected by level N: <claim>` at the next redraw. A recap that ignores the form is shown as `Your recap:`. |
| `ARC3_MEM_ATTEMPT_COUNTS` | Off | From a level's second attempt, `<this_level>`'s summary reads `Level 3 so far: 61 actions over 2 attempts, 22 of them in this attempt (attempt 2), no deaths.` (B read `61 actions, attempt 2` as this attempt's spend in about 20 reasonings; wa30 L3 reset a live attempt that had taken 22). |
| `ARC3_MEM_SEEN_GAME_ACTION` | Off | Seen-before and GAME OVER lines give the game's action number first: `after action 144 (action 5 of this attempt)`, `ended in GAME OVER at action 33 (action 4 of that attempt)`. The recorder keeps each visit's game action number either way; only this flag shows it. |
| `ARC3_MEM_EFFECT_WORDS` | Off | `<this_level>`'s effect words no longer imply the classes exclude each other (they are first-match: 52% of B's "changed cell colours" actions moved an object): `moved cells (no colour count changed)`, `changed colour counts (cells may also have moved)`, `animated, then left the board inside the 4-cell border as it was`, `changed only cells in the 4-cell border` (the band is `ARC3_MEM_STATE_BORDER`), with the count first (`4 times animated, ...`). |
| `ARC3_MEM_RECAP_FROZEN_FACT` | Off | (Needs `ARC3_MEM_DEBRIEF`.) The memory section's `<previous_levels>` bullet adds "Each recap is copied right after its level is cleared and is not updated afterwards." Static per run; stacks with `ARC3_MEM_APPEND_COMMIT`'s row. In no arm. |
| `ARC3_MEM_STUCK_FACT` | Off | The user's conditional `.replace()` prompt, in-level. Once per level, on the first turn message (fresh or a yield resumption's) where the level's generated tokens reach the threshold, an `(old, new)` row on that turn's text puts a count line after `Valid actions right now: ...`: `Harness count: you have written 77.4k tokens of reasoning and code on level 5 without clearing it; levels 1-4 took 14.6k, 49.5k, 27.0k and 49.6k tokens (median 38.2k).` Newest message only, before the opener's static tail. The level is marked done only when the turn is kept, so a rolled-back turn shows it again. Never on `run_complete`. Count line only: earlier recaps are not re-shown. Logs a `MEMORY STUCK FACT` transcript section. |
| `ARC3_MEM_STUCK_MIN_TOKENS` | `60000` | The threshold's floor. |
| `ARC3_MEM_STUCK_MEDIAN_MULT` | `2` | The threshold is max(floor, this x the median of this game's cleared levels' generated tokens). |
| `ARC3_MEM_STUCK_MEDIAN_FROM` | `2` | The median is used once this many levels are cleared; before that, the floor alone (a one-level "median" is that level). |

| Arm | Flags on top of `pf-keep-10x` |
|---|---|
| `pf-keep-append-10x` | `ARC3_MEM_APPEND_COMMIT=1` (cache-safe memory) |
| `pf-keep-recap-10x` | `ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_COMPRESS_ROUTES_FIRST=1` (whole recaps) |
| `pf-keep-memv2-10x` | those two plus `ARC3_MEM_ROUTE_FORMAT=readable`, `ARC3_MEM_HEAD_WORDS=1`, `ARC3_MEM_RECAP_ASK_V2=1`, `ARC3_MEM_ATTEMPT_COUNTS=1`, `ARC3_MEM_SEEN_GAME_ACTION=1`, `ARC3_MEM_EFFECT_WORDS=1` (memory readability) |
| `pf-keep-stuck-10x` | `ARC3_MEM_STUCK_FACT=1` (the stuck count line) |
| `pf-keep-v2-all-10x` | all of the above together, plus `ARC3_LEGIBILITY=1` (the combined run below) |

### Memory v2 and legibility together (`pf-keep-v2-all-10x`, `arm/pf-keep-v2`)

`arm/pf-keep-v2` merges `arm/mem-v2` (#169) and `arm/legibility` (#168) onto `arm/pf-mem`. The user chose to test the factors in one combined run, `pf-keep-v2-all-10x`: `pf-keep-10x` plus `ARC3_MEM_APPEND_COMMIT=1`, `pf-keep-memv2-10x`'s eight flags (`ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_COMPRESS_ROUTES_FIRST=1`, `ARC3_MEM_ROUTE_FORMAT=readable`, `ARC3_MEM_HEAD_WORDS=1`, `ARC3_MEM_RECAP_ASK_V2=1`, `ARC3_MEM_ATTEMPT_COUNTS=1`, `ARC3_MEM_SEEN_GAME_ACTION=1`, `ARC3_MEM_EFFECT_WORDS=1`), `ARC3_MEM_STUCK_FACT=1` and `ARC3_LEGIBILITY=1` (the six-fix bundle). `ARC3_MEM_TRIM_STRIP`, `ARC3_MEM_APPEND_NO_HANDOVER`, `ARC3_MEM_RECAP_FROZEN_FACT` and the six flag-only legibility fixes stay off. No setting is new: every factor keeps its own default-off flag, so a result can be bisected with the single-factor arms (`pf-keep-append-10x`, `pf-keep-recap-10x`, `pf-keep-memv2-10x`, `pf-keep-stuck-10x`, `pf-keep-legible-10x`). With every flag off, every request is still byte-identical to `arm/pf-mem` @ 1882bcb3 (both branches' golden fixtures pass on the merged tree).

How the two branches' text composes:

- System prompt, static for the run. `_build_system_prompt` builds the memory section with memory v2's rows (`ARC3_MEM_RECAP_FROZEN_FACT`'s, then `ARC3_MEM_APPEND_COMMIT`'s, applied to the `<previous_levels>` bullet on its own), moves it after the Python guidance (`ARC3_PROMPT_HYGIENE`), then applies level transfer, hygiene and budget truth, and the legibility rows last, so `ARC3_TURN_BASED_FACT` extends budget truth's step-bar bullet. The two branches' rows touch disjoint text: the combined prompt is `pf-keep-legible-10x`'s with the append-commit row applied, +271 tokens over `pf-keep-10x` (+58 for the append-commit sentences, +213 for the bundle; raw text, Flash-Next tokenizer).
- Turn messages. The stuck count line's row anchors on the `Valid actions right now:` line, which no legibility fix touches (`ARC3_ACTION_INDEX_FACT`'s line, flag-only, comes earlier, after `Executed actions:`); the v2 recap ask's row is applied to `level_recap_ask`'s text alone; the legibility turn rows (`ARC3_RESET_BAR_FACT`, `ARC3_TURN_BASED_FACT`) touch the game-over opener, the game-over stop detail, the edge-only opener line and the solver's no-op stop detail, none of which memory v2 changes. `<this_level>` and `<previous_levels>` words are memory v2's only (`ARC3_EDGE_NOOP_FACT` leaves them).
- Cache. Nothing edits text already sent except at a trim: the `<cleared_level>` block, the count line and the legibility sentences go into the newest message or tool output, and `<previous_levels>` is redrawn only on requests that trim.

Tests: `tests/test_pf_keep_v2.py` builds the system prompt and four turn messages with every flag on (the arm's env, and the arm plus every other flag of both branches), logs each `(old, new)` row as it applies, and asserts each old text occurs exactly once at that point, each stage applies all its rows, `pf-keep-10x`'s own rows find what they found without the new flags, and each new text appears once; `tests/test_pf_keep_v2_loop.py` plays the arm through the real game loop in 30k, 45k and 98k windows next to `pf-keep-10x`; `../kaggle/tests/test_pf_keep_v2_arm.py` checks the arm is `pf-keep-10x` plus exactly those flags.

### One memory arm (`mem-unified-10x`, `arm/mem-unified`)

The user's ask (2026-10-13): one memory arm combining the good parts of `pf-solved-10x` (A) and `pf-keep-10x` (B). It is B's memory (the every-call layout with recaps, which the model drew on in 109 of 109 first reasonings on a new level, against 7 of 113 in A), with `pf-keep-v2`'s cache-safe commits and whole recaps, plus three facts the state lab (round 2, our own W4A16 server, N=20 per cell) measured on frozen real states. Each flag is off by default; with all three off (unset or `0`) every request of the game-loop sim is byte-identical to `arm/pf-keep-v2` @ d624b0f9 under `pf-keep-10x`, `pf-keep-v2-all-10x` and the arms' base env (`tests/fixtures/pf_keep_v2_golden.json`). No system-prompt text is added: the new lines sit in `<previous_levels>`, in the `<cleared_level>` block and in the newest turn message, so every request still extends the previous one between trims.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEM_RULES` | Off | (Needs `ARC3_MEM_EVERY_CALL`.) A ledger of the rules the model confirmed, quoted verbatim from each reply's reasoning and content (`inference/agent/confirmed_rules.py`). A sentence is kept when it carries a confirmation mark (a check mark, "confirmed", "verified"), or is a "So ..." conclusion right after a marked sentence that was not kept; states a rule (a generic word list: moves, only, must, always, never, kills, opens, teleports, absorbs, matches, same, goal, win, ...); holds at most one coordinate pair, 8 digits and one `=`; is 40-260 chars; and is no question, plan ("Let me", "Then", an imperative), hedge ("might", "should", "presumably", ...) or python. A leading "Hmm," is cut, not the sentence. Per level the best three by a rule score (distinct rule words, less digits, coordinates, symbols and words about the model's own code; ties to the newest), shown in the order written; a sentence overlapping a kept one by >= 0.6 (word Jaccard) replaces it. The reasonings still read for the cleared level's recap are skipped (the recap carries them). Shown in each cleared level's `<previous_levels>` entry as `  Rules you confirmed: "..." \| "..."` after the recap (frozen with the entry: with `ARC3_MEM_APPEND_COMMIT` it rides in the `<cleared_level>` block; dropped at compression stage 3), and in `<this_level>` as `Rules you confirmed on this level: "..."`, after the notes and outside `ARC3_MEM_LEVEL_CHARS`. Lab: m0r0 L6 (A call 166) 3/20 -> 9/20 with the model's own L3/L4 rules quoted. |
| `ARC3_MEM_MOVED` | Off | (Needs `ARC3_MEM_EVERY_CALL`.) `<this_level>` lists the small marker objects whose position differs from the level start (`inference/agent/board_facts.py`): segmentation nodes of at most 4 cells, and corner frames (four single same-colour pixels at the corners of a rectangle, each pixel in exactly one such rectangle). Each line: `Moved since this level started, not by an arrow key: <the marker then> is now at <where>. It moved at action N, <display>, the click at which <the other objects that moved at that action, as "the 16-cell b/S block moved from ... to ...">.` A marker counts only when, at its last move, a marker like it left another place (a recolour or a newly drawn marker is no move); moves at UP/DOWN/LEFT/RIGHT and markers holding the clicked cell are left out. When a moved marker is a corner frame, a last line lists the boxes four single corner pixels of its colour mark, with their colour counts (`Boxes marked by four single yellow corner pixels: rows 33-36 cols 29-32 (4x4: 12 g cells and the 4 Y corners; the 6x6 frame surrounds it); ...`). At most 3 lines. Computed from the history's own frames at each new turn message. Lab: sc25 L6 (A call 101) 0/20 -> 16/20. |
| `ARC3_TOUCHING_SPLIT` | Off | A turn-message line, after the per-turn memory lines and before the opener's static tail, when one segmentation node of colour X is k >= 2 same-colour objects of the level start (its box splits along one axis into k equal boxes whose cells have those objects' shapes, each at least 4 cells) and X's pixel total is unchanged: `Same-colour objects that touch, listed separately: sky blue S x2 — a 5x5 at rows 9-13, cols 29-33 and a 5x5 at rows 9-13, cols 34-38, touching (segmentation shows them as one 10x5 node). Sky blue pixel total: 50, the same as at the level start (two 5x5 squares).` Not on a yield resumption. Independent of the memory flags. Lab: m0r0 L6 (uc1 call 117) 5/20 -> 10/20. |

| Arm | Env |
|---|---|
| `mem-unified-10x` | `pf-keep-10x` + `ARC3_MEM_APPEND_COMMIT=1`, `ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_RULES=1`, `ARC3_MEM_MOVED=1`, `ARC3_TOUCHING_SPLIT=1` (no legibility bundle, stuck fact or other v2 readability flags) |
| `mem-unified-core-10x` | `pf-keep-10x` + `ARC3_MEM_APPEND_COMMIT=1`, `ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_RULES=1` (bisects the two board facts out) |

Offline checks on runs A and B (all 25 games each, the logged replies and boards replayed): rules on 83% of 272 cleared or played levels (mean 2.2 lines per level), +188 prompt tokens per request on average (median 147, p90 419; chars / 3.6), and no `<previous_levels>` of B pushed past `ARC3_MEM_HEAD_CHARS` (6000) by them; moved lines on 11% of turn messages (median 0 per turn message, 1.5-1.7 lines when firing), none in 14 of A's and 13 of B's games; touching lines on 0.5% of turn messages, none in 21-22 games. Tests: `tests/test_mem_unified.py` (the lab's sc25 and m0r0 lines reproduced from the logged boards, the quoted rules captured from the logged replies), `tests/test_mem_unified_loop.py` (flags off byte-identical in three envs and three windows; with the flags on, a static system prompt and every request extending the previous one between trims).

### Stall fix: click-run outcome fact (`ARC3_CLICK_RUNS`, `arm/stall-fix-clicks`)

The stall read of run U's sc25 L5 (`mem-unified-10x`): with the full solution in hand, 13 actions from the clear, the model typed the 4-click plus pattern with 3 of 4 cells wrong (actions 127-130). Only the four keypad cells lit up and nothing animated; it concluded "plus does not grow the cap" and held that for ~71k tokens and ~60 minutes, although its own plus at actions 90-93 had shrunk the cap. `<this_level>` counted the dead clicks with the working ones ("changed cell colours 20 times"). This fact puts the dead run next to the earlier runs that worked. Off by default; with it off (unset or `0`) no request changes.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_CLICK_RUNS` | Off | A turn-message line after the executed-actions and animation lines (`inference/agent/click_runs.py`). A click run is a stretch of consecutive MOUSE clicks inside one executed sequence; it ends at any other action, a RESET, a level change, a game over and at the end of the sequence. Its clicked cells are each clicked cell plus the segmentation node holding it before and after its click, when that node has at most 64 cells (a 3x3 keypad button lights up as one object). A run is effective when, at any of its clicks, a cell more than 4 cells from the edge changed outside the clicked cells, or a click played a multi-frame animation (the history's `animation` summary), or the run ended the level or the attempt. When the last run of the sequence just executed has 2+ clicks and is not effective, and the game (any level) already had an effective run of the same length, the line names the dead run and what its clicked cells did, then up to three earlier effective runs of 2+ clicks (one per click sequence; same length first, then fewest differing cells, then newest), each with an effect phrase from the board before its first click and after its last (moved blocks, corner frames and objects gone, shrunk, grown or new, colours no longer on the board, whether the clicked cells are back as they were), then how the most similar one compares: `Click run at actions 127-130, (50,25) (50,35) (55,30) (60,30): only the clicked cells changed, from g to N, and they stay N; nothing else inside the board area changed and no animation played. Earlier click runs that changed more than the clicked cells or animated: (50,30) (55,25) (55,35) (60,30) at actions 90-93 (the 16-cell b/S object at rows 15-18 cols 35-38 is now 4 cells at rows 15-16 cols 35-36; the yellow 6x6 corner frame (...) is gone; the clicked cells are as before the run); ... The run at 90-93 has the same length and differs from this one in 3 of its 4 cells.` Not on a yield resumption, after a game over or on a level change. Independent of the memory flags. |

Replay over all 25 games of seven runs (boards from the event logs; animations rebuilt from `intermediate_states.pkl` with the harness's own `collapse_frames`/`summarize_animation`, or for the two older xz runs from each turn message's animation line): U fires twice, at requests 65 and 66 of sc25 (after actions 130 and 133, the moments the stall read names), Ucore twice (ft09, tn36), A once, B four times, `pf-keep-v2-all-10x` once (sc25), uc1 twice, `mem-solved-10x` seven times: 1-7 of ~1,100-1,200 turn messages per run, ~1,050 chars when it fires. Requiring an earlier effective run of any length instead fired 16 times on U and Ucore, 10 of them in ft09, where every click only recolours the clicked cell and the effective runs are level clears. Tests: `tests/test_click_runs.py` (the two sc25 lines reproduced from U's logged boards, `tests/fixtures/click_runs_states.json.gz`; ft09's paint runs silent; run splitting, the edge band, animations; through `_build_user_prompt` on a scripted keypad game: the line after `Executed actions:`, none on a resumption, none with the flag off).

### Death verdict and undone actions (`arm/stall-fix-death`)

Two facts for the stall-fix arm (2026-10-14), each off by default, computed from the solver's history frames (`inference/agent/death_facts.py`). U tu93 L4 read an empty-bar budget death as a ranged kill and spent ~72k tokens on it (it judged the bar by eye); Ucore r11l L2 died at a hidden fifth undone move of the attempt and, told "this action itself was fatal", hunted a contradiction for ~85k tokens. With both flags off (unset or `0`) every request is unchanged; with them on and nothing to say (the game-loop sim: no bar, clicks that blink only the clicked cell), every request is byte-identical to the flags-off run (`tests/test_stall_fix_death_loop.py`).

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_DEATH_VERDICT` | Off | On the GAME OVER turn message, lines about the attempt that just ended, after `The game over occurred immediately after action X.` A budget bar is one colour in one edge band (within `ARC3_NOOP_GUARD_BORDER` (4) cells of a grid edge), identified over the game's history from the bands without their corner squares: it falls at least 3 times, on at least 15% of the actions of the attempts that start with it, rises within an attempt at most a quarter as often as it falls (ls20 refills) and is restored at a RESET or level start at least once; the key with the most falls wins. Read over the whole edge strip: `Budget bar (bottom edge): 64 cells at the start of this attempt, 0 in the death frame, about 3.2 fewer per action: the bar is empty in the death frame, so this game over is the action budget running out on action 20 of this attempt.` or `Budget bar (bottom edge): 54 of 64 cells left in the death frame, about 3.3 per action: not a budget death.` No bar line when no bar is found, when the bar emptied more than 3 actions before the game over, or when the fatal action refilled it (su15's death frames). Then the attempt's undone actions (as `ARC3_UNDONE_FACT` defines them): `This attempt had 4 actions that the game undid (they animated, then the board returned to how it was): actions 8, 10, 12 and 14; the game over came on action 16.` When the bar is empty or there is an undone action, the sentence's BAR RULE clause (`Apply the BAR RULE above to decide: ... this action itself was fatal ...`) is dropped; otherwise it stays and the bar line follows it. The rest of the GAME OVER text is unchanged. |
| `ARC3_UNDONE_FACT` | Off | (Needs `ARC3_ANIMATION`.) In an ordinary turn message, after the animation line and digest, one line per action of the just-executed sequence that the game undid in the attempt still running, the latest two: `Action 8 MOUSE(row=30, col=30) was undone by the game: it animated, then the board returned to how it was (1st undone action in this attempt; cells that changed during the animation: rows 4-32, cols 10-47).` Undone: the board outside the border band is as before the action (`gameplay_changed` false), the animation's changed-back cells (`transient_bbox`) reach inside it, and the action is not a click whose changed-back cells are centred (within 1.5 cells) on the clicked cell, which is the game marking the click (su15's ripple, lf52's box). The range is the animation's changed-back bounding box. |

Offline checks (the frames in `intermediate_states.pkl`, animation summaries recomputed as the solver does). Verdict, all 720 game overs of 52 runs (U, Ucore, pf-keep-v2-all and 49 older runs): one bar key per game edge (sp80's top bar on L1, bottom after), 265 budget deaths (attempt lengths match each level's budget: tu93 L4 20 actions x21, tn36 61, vc33 50, wa30 70/100/125), 369 not, 86 without a bar line (bp35's 67: its level-start frames hold no bar; su15's 8 death frames that refill the bar; sk48 and sb26, whose only falling keys empty early and stay empty). U and Ucore: 18 game overs, tu93 U action 96 budget, every other tu93 death "not a budget death", Ucore r11l action 16 not budget with undone actions 8, 10, 12 and 14. Undone fact, all 25 games of U and Ucore: 73 of 1,189 and 47 of 1,132 turn messages (84 and 49 lines), silent in 14 and 15 games; most in tn36 (21 of 40 and 21 of 31: its program runs revert), dc22 (15 of 86) and s5i5 (11 of 31). The click filter removes 13 and 11 firings (su15, lf52); counting only from the 2nd undone action per attempt (45 and 9) would have dropped r11l's first undone click, so it is not used. Tests: `tests/test_stall_fix_death.py` (the U tu93 and Ucore r11l texts from the logged boards, synthetic edges, the turn message with the flags off and on, both facts through the real sandbox), `tests/test_stall_fix_death_loop.py`.

### Stall fixes: the action() contract and the colours row (`arm/stall-fix-tool`)

Three default-off flags from the stall reads of runs U (`mem-unified-10x`) and Ucore (`mem-unified-core-10x`). Each is off by default; with all three off every request is byte-identical to `arm/mem-unified` @ addf9879 (the golden tests pass). None adds system-prompt text.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_ACTION_ERROR_RAISE` | Off | A python `action(...)` call that the host refuses as invalid input, before anything runs, raises instead of returning silently. The refusals: the request does not parse (`Unknown action at index 1: 'L'`, `MOUSE action at index N requires integer row and col arguments.`, an empty list, a non-object item) and a first action that is not available now (`SPACE is not valid right now.`). The solver tags them `stop_reason: "invalid_input"`; the sandbox prints `[action] NOT EXECUTED: Unknown action at index 1: 'L'` into the snippet's stdout and raises `InvalidActionError: Unknown action at index 1: 'L'. Nothing in this call was executed. Valid action names: RESET, UP, DOWN, LEFT, RIGHT, SPACE, MOUSE.` (the payload's own valid list). `InvalidActionError` derives from `BaseException`, like the guard refusals, so `except Exception:` cannot swallow it, and is a runtime global only with the flag on. Unchanged: guard refusals (`known_noop`, `known_death`, `repeated_action_in_state`, `stale_state`, the RESET refusals), terminal payloads, a later action of a batch that is unavailable (that batch stops with `invalid_action` after what ran), and an engine error while stepping (the step may have happened). When a snippet executed nothing because of such a refusal, the next turn message says `Your last python snippet executed NO actions: its action() call was refused as invalid input (Unknown action at index 1: 'L'), so nothing in that call ran.` instead of "(it only computed or inspected state - that is fine)"; the executed-actions lines are unchanged. Evidence: U m0r0 L3, where an executor's `action(['L'])` was refused silently three times and the model built two false rules on the phantom no-ops (~73 min, ~89k tokens). |
| `ARC3_ACTION_ALIASES` | Off | `U`, `D`, `L`, `R` (any case, stripped) are accepted as `UP`, `DOWN`, `LEFT`, `RIGHT` by the solver (`action_names.to_engine_action`); the action then runs and is displayed under its full name. Nothing else is aliased (not `S`: U used it for "select squares"); valid-action lists, prompts and displays do not change. |
| `ARC3_MEM_COLOURS_FIRST` | Off | (Needs `ARC3_MEM_EVERY_CALL` and `ARC3_MEM_LEGIBLE`.) In `<this_level>`, the `Colours on the board not clicked on this level:` row comes first among the action rows and outside their 700-char budget (it rendered last and the budget loop's `break` dropped it), a row that does not fit is skipped instead of ending the rows, and colours absent from the level's first board come first, each with where it is now: `Colours on the board not clicked on this level: orange (new on this level; rows 43-54, cols 12-14), white.` The place is the bounding box of the colour's cells inside the edge crop; when they form several 4-connected groups, `40 cells in 3 groups, the largest at rows 43-54, cols 12-14`. Which colours are listed is unchanged (on the board inside the crop, not the background, never clicked on this level). The block's own `ARC3_MEM_LEVEL_CHARS` cap still applies (it cuts rows from the end, so this row goes last). |

Offline checks on runs U and Ucore (all 25 games each), and for the first two flags also on `pf-keep-v2-all-10x` (comb) and `pf-keep-10x` (B):

- `ARC3_ACTION_ERROR_RAISE`, `ARC3_ACTION_ALIASES`: every logged python snippet (2,679 in U, 2,704 in Ucore, 2,620 in comb, 2,552 in B) re-run in the real sandbox with the run's retained functions, its `action()` calls validated as the solver does and served the logged boards (97-99% of U's and Ucore's snippets replay exactly the actions they ran in the run). Snippets with a call refused as invalid input: U 4 (m0r0 L3 three times, re86 L6; all silent, all a single arrow letter), Ucore 3 (lf52 L4 `UNDO` x4, su15 L4 `UNDO`, tu93 L1 `D`), comb 4 (m0r0 L3 twice with 19 letter calls, tu93 `L`, wa30 `MD`), B 4 (`L`, `D`, `UNDO` twice). All are unknown names; no MOUSE without row/col and no unavailable first action occurred. With the flag each would have raised at its first refused call; with aliases 10 of the 15 snippets would have run as intended (the letter ones), not the `UNDO`/`MD` ones.
- `ARC3_MEM_COLOURS_FIRST`: the logged boards replayed through `FactRecorder` (the flag-off colours row equals the logged one in 2,361 of 2,369 turn messages). The row was due in 1,729 turn messages and dropped by the 700-char budget in 370 (21%; dc22 91 of 139, lf52 94 of 147, sc25 36 of 123, sb26 31 of 61, su15 30 of 154, m0r0 26 of 121, cd82 21 of 88, vc33 7 of 102). A `new on this level` colour is listed in 315 (18%), mostly in games that add colours as they are played (cn04, su15, ar25, lf52, sb26; m0r0 L3's yellow selected dot and grey squares; vc33's orange gates); 108 of the 370 dropped rows held one. The row is 100 chars at the median (p90 153, max 250), +26 chars per turn message on average; the 1,100-char block cap then drops a row that used to show in 13 of 2,369 turn messages.

Tests: `tests/test_action_error_raise.py` (the solver's tags, the sandbox's echo and exception, the guard and terminal refusals untouched, the m0r0 shape end to end and the next turn message), `tests/test_action_aliases.py`, `tests/test_mem_colours_first.py` (Ucore vc33 level 4 from the logged boards, `tests/fixtures/vc33_ucore_boards.json.gz`: the dropped row at requests 51, 60 and 62 and the flag-on row), `tests/test_stall_fix_tool_loop.py` (the game loop with the three flags on: the same game, only the colours row moves).

### Stall-fix arm (`arm/stall-fix`)

The three subsections above were built in parallel off `arm/mem-unified` and merged as `arm/stall-fix`. Why: seven stall reads of mem-unified-10x and mem-unified-core-10x (`research/archive/reports/stalls-2026-10-14.md`) found 6 of 7 single-level stalls were false negatives from invalid or misread tests, with the harness hiding or blurring the evidence. Each flag stays separately switchable for bisection.

| Arm | Env |
|---|---|
| `stall-fix-10x` | `mem-unified-10x` + `ARC3_ACTION_ERROR_RAISE=1`, `ARC3_ACTION_ALIASES=1`, `ARC3_MEM_COLOURS_FIRST=1`, `ARC3_DEATH_VERDICT=1`, `ARC3_UNDONE_FACT=1`, `ARC3_CLICK_RUNS=1` |

### Mem-repair arm: confirmed rules and retained functions (`arm/mem-repair-rules`)

The run-wide memory audit of `stall-fix-10x` (SF) found the quoted rules line often wrong, stale or noisy: of SF's 512 distinct quoted rules, hand-labelled, 36% were real mechanics, 24% board trivia (states, coordinates, arithmetic), 17% plans, 15% hypotheses stated as fact (30% of the rule impressions on stalled levels), 4% notes on the model's own code and 3% echoes of the memory or prompt; 23 were later contradicted, 7 shown next to their contradiction. Refuted rules stayed on screen while the model was stuck (re86 L7 "Level 6: the dots were absorbed when covered…" for 11 turns, dc22 L3 "So the purple region can NEVER be entered.", wa30 L3 "the 3 left boxes can never be delivered"). The retained-functions list was 34% of a turn message's text. Each flag is off by default; with all three off every request is byte-identical to `arm/stall-fix` (the golden tests pass).

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_RULES_FILTER_V2` | Off | (Needs `ARC3_MEM_RULES`.) The confirmed-rules ledger (`confirmed_rules._rule_ok_v2`) also drops a sentence that (a) holds a coordinate pair, a pair of ranges or a row/col range (`(12, 40)`, `(43-46, 3-6)`, `rows 42-53`); (b) is a fragment: ends in `:` or `...`, opens with `)`, `,`, `;`, `→` or a dash, or has unbalanced brackets or double quotes; (c) is an echo: quoted text together with a level name, the words note, recap, harness, memory, says or said, or an opening `Level 6:` or quote; (d) is a plan or about the model's code: let me, let's, I'll, I will, I need, we need, first,, actually, next, my/the code, solver, search, BFS, helper, function, simulator, model; (e) is hedged: may, might, maybe, perhaps, probably, likely, seems, appears, could, possibly, guess, assume(d), a `?`, or an opening If, Unless, Assuming or Suppose; (f) is an unmarked "So ..." conclusion (one that inherits the previous sentence's mark) stating never, cannot, can't, impossible, only, must or always. Same extraction for the `<this_level>` line and the rules frozen into `<previous_levels>`. Ranking, the three per level and the overlap replacement are unchanged. |
| `ARC3_RULES_ACTION_TAG` | Off | (Needs `ARC3_MEM_RULES`.) Each quoted rule, in `<this_level>` and `<previous_levels>`, ends with the game action it was written after (the turn's executed-action count; a turn ends at its first executed action, so every reply of a turn shares it), oldest first: `Rules you confirmed on this level: "Either way, hitting any dark red destroys all dark red ✓." (action 210) \| "So the teleport chose the left-area marker (not the nearest)." (action 233) \| "In this level, the beam hit the dark red SEGMENT (plain cells) and destroyed nothing ✓ confirmed." (action 248)` (SF sc25 request 97). The labels are unchanged. |
| `ARC3_RETAINED_DELTA` | Off | (Needs `ARC3_PERSISTENT_FUNCTIONS`.) The turn message's `Your retained functions from earlier Python calls ...: ...` list is sent as `Your retained functions are unchanged (N functions).` when the newest retained-functions statement in the messages the request keeps (decided after the opener-time trim, like `ARC3_OPENER_DEDUP`) is that same full list and no `were cleared` / `is no longer retained` notice follows it. A trim that drops the last full list makes the next turn message carry it again; nothing already sent is edited. In the list, a default value longer than 30 chars is shown as its type: `blkplan(L0, R0, walls, FREE=<tuple, 74 items>)`, `P=<str, 99 chars>`. |

Offline checks (all 25 games; the replies and turn messages of SF's raw logs replayed through the harness's own ledger, level changes and recap skips; flags off the replay reproduces every logged `<this_level>` rules line, 1,237 of 1,237, and every frozen entry, 85 of 85, and likewise on `mem-unified-10x` (U)):

- `ARC3_RULES_FILTER_V2`, on the labelled set: mechanics 36% -> 50% of kept entries, 159 of 186 mechanics kept (85.5%), hypotheses 78 -> 34, contradicted 23 -> 12; mechanics' share of this-level impressions 36% -> 52%, hypotheses' 22% -> 15%. Tuned from the spec's list, which kept only 53% of the mechanics: any quoted text (dropped 16 mechanics), any level name (30: "In level 2, clicking a piece moved the active dot to it ✓."), a lowercase opening (14) and an and/but/or opening (9) are not rejected; quoted text with a level name, the echo words and an opening `Level N:` are; an opening If and pairs of ranges were added. Replayed: a rules line on 763 of 1,237 SF turn messages (832 before), 2.4 rules a line; the rules that move up into the top three in place of dropped ones are about 55% mechanics (hand count of the 123 new ones). At the stalls: re86 L7 no longer shows the "Level 6: the dots were absorbed" echo (requests 94-109), dc22 L3 and wa30 L3 lose "can NEVER be entered" and "can never be delivered" (wa30 shows "So the \"push into socket\" theory is refuted." instead), m0r0 L3 loses "dots are impassable walls" for "CONFIRMED: The yellow dot MOVES with the arrows!", m0r0 L5 at request 149 shows "So the S block CAN move onto a colored block's cell — it ABSORBS it" instead of three plan fragments, tr87 L2 loses "The order list: I'll maintain as a list…". sc25's "So the outer frame may just be a \"big marker\"…" and su15's "Actually careful: …" are gone.
- `ARC3_RULES_ACTION_TAG`: +30 chars per rules line on average (252 -> 282).
- `ARC3_RETAINED_DELTA` (tokens with the Qwen3.8 tokenizer): of SF's 1,702 turn messages with a list, 1,147 (67%) send the unchanged line; list tokens 408,378 -> 132,828 (102,163 of the saving from the default summaries alone; m0r0 127,153 -> 12,001); per turn message median 117 -> 10 tokens, p90 464 -> 252. U: 268,787 -> 120,258. Of 156 SF context trims (150 in U), 30 (40) dropped the last full list and the next turn message carried it again in every case; one SF request (wa30, a trim inside a turn) held the unchanged line without a full list until the next turn message.

Tests: `tests/test_mem_repair_rules.py` (each v2 rejection on SF's own sentences, the lab-quoted rules kept, the tags and their order, frozen entries and copies; SF's sc25 L6 and re86 L7 replies replayed from `tests/fixtures/sf_rules_replies.json.gz`, flags off equal to the logged lines), `tests/test_retained_delta.py` (summaries, the newest-statement check, clear notices), `tests/test_mem_repair_rules_loop.py` (the game loop with a retained helper, flags off and on in the 98k and 30k windows: the same game, only the tags and the retained line change, every request extends the previous between trims, the list re-sent after the trim that dropped it).

### Mem-repair arm: board facts (`arm/mem-repair-facts`)

Eight reads of `stall-fix-10x`'s stalled levels and a run-wide memory audit (2026-10-14) found that the harness-written memory often carries wrong, stale or noisy content, or drops a correct fact exactly when it matters. These flags repair existing memory lines rather than add advice. Each is off by default; with them off (unset or `0`) every request is byte-identical to `arm/stall-fix` (the golden tests pass).

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MOVED_PERSIST` | Off | (Needs `ARC3_MEM_MOVED`.) The `<this_level>` moved lines keep reporting a marker frame after it is hidden, and report one that appears (`BoardFacts.moved_lines(persist=True)`, `inference/agent/board_facts.py`). A marker frame is a corner frame that is square, of side 4-8, with at least half of its outline between the corners in the board's most common colour (four marks around a box, not a sprite's cut corners). For the current attempt (since the level start or the latest RESET on it), each frame whose latest change hid or showed it, at an action other than an arrow key, gives one line per action, newest first, at most 2: `Hidden since action 229, MOUSE(row=60, col=30) (the click at which the 16-cell b/S object at rows 37-40 cols 45-48 became 4 cells at rows 37-38 cols 45-46): the yellow 6x6 corner frame (last around rows 33-36 cols 29-32). Appeared at that action: a purple 6x6 corner frame around rows 40-43 cols 12-15; the 4-cell b/S block moved inside it at action 233, MOUSE(row=55, col=30), from rows 37-38 cols 45-46.` The clause in brackets is what else changed inside the board area (4 cells from the edge) at that action: the largest moved block, else an object that changed size, else the largest other change. Left out: a frame gone at an action at which one of its colour and size appears (a move, which the moved line reports), a shown frame that was on the attempt's first board, and a hidden frame under a block that moved onto it. The boxes line (`Boxes marked by four single yellow corner pixels: ...`) is shown whenever a box of a marker-frame colour that moved, hid or showed on this level is on the board, not only beside a moved line, and names a surrounding frame of another colour (`the purple 6x6 frame surrounds it`). The `ARC3_MEM_MOVED` lines themselves are unchanged and come first. |
| `ARC3_CLEARED_BOARDS` | Off | The solver keeps, at each level clear, the board the clearing action produced: the last distinct state it passed through before the next level's board (None when it returned the next board alone), and answers a no-action `step_env` query for them. The python sandbox gets `cleared_boards[N]` (a frame view like `current_frame`, `step` = the clearing action's number). The turn message after a clear (after the level-start text, before the recap ask) and, with `ARC3_MEM_EVERY_CALL`, that level's `<previous_levels>` entry (after its head line, dropped at compression stage 3) say which frame is which: `Level 3 was cleared by action 50, click (56, 46). history's last level-3 frame (step 49) is the board BEFORE that action; the board it produced is cleared_boards[3].` or `... The board it produced was never shown; history's last level-3 frame (step 49) is one action short of the clear.` (`inference/agent/cleared_boards.py`). The tool description is unchanged; the line names the variable. |
| `ARC3_THIS_LEVEL_AFTER_TRIM` | Off | (Needs `ARC3_MEM_EVERY_CALL`.) A turn message that carries no `<this_level>` block of its own (a yield resumption's, "executed NO actions", "produced no tool call") gets the current block when the messages kept for its request hold none for the current level, as after a trim that dropped the turn's opener. Added after the trim, the opener dedup and the `<cleared_level>` blocks, to the newest message only, so no sent prefix changes. A trim in the middle of a turn (the newest message a tool result) is restored at the next turn message. |

Offline checks (2026-10-14) on `stall-fix-10x` (SF) and `mem-unified-10x` (U), all 25 games, boards from the event logs, clearing frames from `intermediate_states.pkl`:

- `ARC3_MOVED_PERSIST`: replayed at every request; every flag-off line is kept, in order, at every request of both runs. New lines on 17 of SF's 1,237 turn messages (sc25 14, r11l 2, re86 1) and 41 of U's 1,212 (sc25 30, r11l 5, re86 6), 421-648 chars added at the median when it fires (max 709); silent in 22 games. sc25 L6 (SF): the moved and boxes lines vanished from request 92 (action 229 hid the frame); with the flag the hidden/appeared line and the boxes line are there at requests 92-96 and again in the second attempt (98-107, the frame hidden at action 254). Tuning (lines over all 2,864 SF requests): every corner frame fired in 10 games (640 lines; dc22 174, sk48 127, lf52 90: pattern pixels, sprites' cut corners, a cursor's highlight); square side 4-8 brought it to 6 games, then dropping arrow-key actions and requiring the outline in the board's colour to 3.
- `ARC3_CLEARED_BOARDS`: every level clear had a produced board, 107 of 107 in SF and 114 of 114 in U; in every one the next level's board arrives as the action's single last frame, so the produced board is the state before it.
- `ARC3_THIS_LEVEL_AFTER_TRIM`: in SF 8 trims in 6 games left no `<this_level>` of the current level, for 61 requests (vc33 L4 requests 58-71, tr87 32-51); 6 of them came at a resumption turn message. Replayed on the logs, 3 requests stay without (ls20 2, bp35 1: trims at a tool result, restored at the next turn message). U: 53 requests in 3 games, 0 with the flag.

Tests: `tests/test_mem_repair_facts.py` (sc25 level 6 from SF's event log, `tests/fixtures/mem_repair_states.json.gz`: the lines at requests 92, 93, 95 and 99, none without the flag, the moved line unchanged where one exists; sprite corners and arrow-key hides left out; the solver keeping the produced board and answering the query; `cleared_boards` in the real sandbox; the clear lines in the turn message and the entry), `tests/test_mem_repair_facts_loop.py` (stall-fix-10x's env in a 26k window, flags off and on: the same game, the request after the trim at request 5 carrying level 1's `<this_level>`, the clear line after the level-start text and in the entry, every request extending the previous one between trims, no other difference).

### Mem-repair arm: object facts (`arm/mem-repair-objects`)

Reads of `stall-fix-10x`'s stalled levels found the memory's object facts click-framed and colour-level: m0r0 L5 never entered the purple pressure plate for ~40 requests while the colours row led the model to click gates; tn36 L2's row dropped "yellow" after one yellow block was clicked, although the yellow in the two legend boxes never was; su15 L4's novel pink invader was never tagged new while the model's own merge products were; cn04 L2 doubted for ~11 requests that a click into the hole of a U-shaped piece did nothing; the 700-char budget hid a clicked colour in 156 of SF's 1,237 `<this_level>` blocks; 46% of arrow presses were counted as "changed cell colours", many of them moves. Four facts, each off by default (`inference/agent/object_facts.py`); with all four off (unset or `0`) every request is byte-identical to `arm/stall-fix` (the golden tests pass). None adds system-prompt text.

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MEM_CONTACT_ROW` | Off | (Needs `ARC3_MEM_EVERY_CALL` and `ARC3_MEM_LEGIBLE`.) Replaces the `Colours on the board not clicked on this level` row with an object-level row, first among `<this_level>`'s action rows and outside their 700-char budget (a row that does not fit is skipped, as with `ARC3_MEM_COLOURS_FIRST`): `Not touched yet on this level (never clicked, entered or bumped). New in this game: light green 4x4 at rows 26-29, cols 34-37; light green 12x4 at rows 22-25, cols 14-25. Others: orange 4x4 at rows 26-29, cols 58-61; 2 purple 4x4 blocks at rows 6-9, cols 14-17 and rows 50-53, cols 14-17; purple 12x4 at rows 22-25, cols 42-53.` (SF m0r0, request 108). Objects: 8-connected same-colour components of 4-256 cells, not the background (the crop's most common colour), not reaching the grid's outermost row or column (walls, panels), not wholly in the 4-cell edge band (HUD, budget bars); identical ones grouped (`2 ... blocks at A and B`, `12 ... blocks within <box>`). Touched on this level: a click inside its bounding box (not a click on the background that changed nothing), holding the clicked cell after a click, moved by the model (any move at a key press; at a click, a move holding the clicked cell), entered by such a moved object, merged by a key press, or lying where a blocked avatar (an object that moved at a key press, at its learned step) would have gone. `New in this game`: its colour was on no board of an earlier level and no action recoloured its cells (they held another non-background colour one frame earlier), never on level 1. Order: new first, then smaller, then fewer copies; at most 6 groups and 360 chars, then `; N more not listed`. No row when every object was touched. Works in keyboard games too. |
| `ARC3_CLICK_MISS` | Off | A turn-message line after the executed-actions, animation and click-run lines, for each click of the sequence just executed (the latest two) that changed nothing inside the board area and hit the background inside, or within 3 cells of, an object's bounding box (objects as above; the smallest box holding the click, else the nearest): `Action 19 MOUSE(55,54) hit background (orange) inside the outline of the blue object (rows 53-58, cols 50-58); it did not touch it. Nearest blue cells: (56,54), (55,52).` (SF cn04, request 20), or `... 2 cells outside the bounding box of the red object (rows 14-17, cols 5-7); ...`. Nearest cells by Manhattan distance, then reading order. Not on a yield resumption, a game over or a level change. Independent of the memory flags. |
| `ARC3_MEM_CLICKED_ALSO` | Off | (Needs `ARC3_MEM_EVERY_CALL` and `ARC3_MEM_LEGIBLE`.) When `<this_level>`'s 700-char budget or the 6-row cap leaves click rows out, a row after the shown click rows names their colours with this level's click counts: `Also clicked on this level: blue x2, black x2, dark red x1, charcoal x1.` (SF dc22, request 7), outside the budget. |
| `ARC3_EFFECT_MOVED_FIX` | Off | The memory recorder counts a key press (not a click) as `moved cells` when objects of 2+ cells translated and every interior cell that changed is explained by one of their offsets d: a non-background colour moved in (`post[p] == pre[p-d]`) or out (`post[p+d] == pre[p]`), with up to 2 cells (and 5%) unexplained (lf52's drop shadow). Before, any colour-count change made it `changed cell colours`: a mover crossing the edge band (m0r0's squares) or covering a floor of another colour (lf52's box with its black outline). A gate that opened at the same press is not explained, so that press stays `changed cell colours`. Clicks are left alone (s5i5's growing arms pass the translation test). |

Offline checks: SF's (`stall-fix-10x`) 25 games, every turn message that carried `<this_level>`, with the trackers fed the logged boards up to that request (`lab2/objects/replay.py`; U's `mem-unified-10x` for comparison):

- Contact row: in 1,101 of 1,237 turn messages (U 1,098 of 1,212), where the colours row was in 923; median 265 chars (p90 346, max 360) against the colours row's 97, +146 chars per turn message on average (~40 tokens). With `New in this game` in 287 (U 211). The tag first used the spec's colour-and-shape rule: 779 tagged objects, 75% of them new shapes of known colours (r11l's stretching arms, tr87's glyphs, sc25's grey tracks), so it was narrowed to new colours. The spec's "larger first" order cut m0r0 L5's purple plates behind the bars; smaller first lists them. The row at the spec's moments: m0r0 requests 100 and 108 (above; the plates stay listed until request 141, after the model first stepped on one), tn36 request 26 (`yellow 3x2 at rows 55-56, cols 20-22; yellow 2x3 at rows 57-59, cols 8-9; ...`: the two legend boxes' yellow, after the main yellow block was clicked), su15 request 41 (`New in this game: pink 8-cell shape at rows 19-22, cols 52-56`), cn04 (4 rows with a new tag, all level 2's first unclicked pieces; no selection recolouring), sc25 request 93 (`... light green 2x2 at rows 33-34, cols 17-18; ...`). The 1,100-char block cap then drops a row at the block's end in 58 of 1,237 turn messages (0 before).
- Click miss: 16 lines in 8 SF games (U: 8 in 7), 169 chars at the median.
- Clicked-also: 156 of 1,237 SF turn messages (lf52 69, m0r0 25, dc22 24, sc25 13), 59 chars at the median; U 129.
- Moved fix: arrow presses counted `changed cell colours` 1,622 -> 864 of 3,523; lf52 L4 `RIGHT (24 presses on this level): moved cells once, changed cell colours 19 times, changed only the edge 4 times.` -> `moved cells 14 times, changed cell colours 6 times, ...`; m0r0 L5 `LEFT (23 presses ...): moved cells 17 times, changed cell colours 5 times, ...` -> `moved cells 21 times, changed cell colours once, ...`.

Tests: `tests/test_mem_repair_objects.py` (the rows, the miss line and the effect counts above from SF's logged boards, `tests/fixtures/mem_repair_objects_states.json.gz`; touches, bumps, new colours, the HUD and walls left out, the translation test, `effect_rows`' lead and clicked-also rows; through `_build_user_prompt`), `tests/test_mem_repair_objects_loop.py` (the game loop with the four flags on next to `stall-fix-10x`: the same game and system prompt, only the object rows and miss lines change, no new prefix breaks).

### Mem-repair arm (`arm/mem-repair`)

The three subsections above were built in parallel off `arm/stall-fix` and merged as `arm/mem-repair`. Why: eight reads of `stall-fix-10x`'s stalled levels and a run-wide memory audit (`research/archive/reports/sf-memory-reads-2026-10-14.md`) found that the harness-written memory was often wrong, stale or noisy, or dropped a correct fact when it mattered.

| Arm | Env |
|---|---|
| `mem-repair-10x` | `stall-fix-10x` + `ARC3_RULES_FILTER_V2=1`, `ARC3_RULES_ACTION_TAG=1`, `ARC3_RETAINED_DELTA=1`, `ARC3_MOVED_PERSIST=1`, `ARC3_CLEARED_BOARDS=1`, `ARC3_THIS_LEVEL_AFTER_TRIM=1`, `ARC3_MEM_CONTACT_ROW=1`, `ARC3_CLICK_MISS=1`, `ARC3_MEM_CLICKED_ALSO=1`, `ARC3_EFFECT_MOVED_FIX=1` |

## 11. Priority scheduling and compute allocation

### Admission control and cache-aware scheduling

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_MAX_ACTIVE_STREAMS` | `0`: disabled | Limits concurrently admitted game streams through a shared priority gate. |
| `ARC3_PRIORITY_REFRESH_QUEUE` | Off | Recomputes waiting priorities against a common current clock before admission. Pace or tail fade also activates refreshed ranking. |
| `ARC3_DIAG_CONCURRENCY` | Off | Prints periodic concurrency/gate diagnostics. |

**Oversubscribing the gate by one (issue #7).** A game holds its slot through python, the engine step and prompt build between requests. With exactly as many gate slots as the server's `--max-running-requests`, every such gap is a hole in the decode batch. The `fz` control averaged 8.9 of 10 running requests, falling to about 8.2 late in the run. Those are means over SGLang's decode log lines, one per 40 decode iterations; weighting each line by the wall time it covers (`kaggle/serve_report.py`, issue #121) moves such figures by about 0.1 to 0.2. The `v4-control` run at `519c4f5` averaged 8.93 time-weighted (8.77 per line), and `tput-safe` with gate11 9.72 (9.66). Setting `ARC3_MAX_ACTIVE_STREAMS` one above the server limit (11 over 10) keeps a request queued to backfill those holes.

- Context is unchanged; the earlier 12/14-stream arms also cut context, which confounded them.
- Costs: weaker harness-side priority ordering, slightly more KV pressure, and queueing: the extra request waits at the server, which adds latency to every game.
- Watch KV usage (peaks at 0.99), retractions and prefix-cache hits.

**Measured, and it lost.** `tput-safe` v1 (gate11 with the other throughput levers and the ledger) scored 42.09 against `v4-control` v1's 50.62 in one batch at `519c4f5`. The batch was fuller (full for 81% of decode time against 41%), but requests sat queued for 60% of decode time against 4%, 66.6 min of queue wait against 7.1, and harness throughput did not rise (591 against 602 tok/s). One pair is a screen, not a verdict, but the lesson in `research/archive/notes/arms.md` is to keep `ARC3_MAX_ACTIVE_STREAMS` equal to the server's max running requests. The recommended `user-combo-10x` does: 14 streams with `MAXREQ` 14.

The gate can admit a new game when another finishes, rather than leaving backend capacity idle. It preserves an admitted stream across tool calls and resumptions and can hand over when context trimming makes a new prefill necessary. This requires the runner to have enough game sessions available; it does not itself replace every outer worker/concurrency limit.

Priority bands protect prefix reuse. Already warm continuation requests receive a 1,000,000 band above requests needing prefill, and the untrimmed continuation path uses its higher 2,000,000-based ordering. Score-based priority operates within that cache-aware policy. Request priority is also sent to compatible backends; backend support and the local admission gate are distinct mechanisms.

A game hands its slot over only after `_note_history_evicted`, which must mean that already-sent messages were really dropped: by the trimmer, the force-reduce fallback, or the turn-end persist. Memory features must never edit a message that was already sent. If the persist compare sees such an edit (a stripped delta, a refreshed or re-inserted pin), the game gives its slot away every turn, as Fable's journal did. If the compare does not see it, the prefix breaks with no handover. `tests/test_memory_eviction.py` (issue #58) checks both directions. It plays a scripted game under the Kaggle knobs, including multi-request turns, yields, a context-length retry and a rolled-back turn. Run it with `nice -n 19 python -m pytest -q tests/test_memory_eviction.py`; it takes a few seconds.

### Base scoring model

The normal score term estimates the value of finishing the current level plus future progress, discounted by effort already spent:

```
efficiency = (h / (h + actions_spent_on_level))²
A = current_level × efficiency
B = min(tail_cap, tail_base + current_level − 1)
C = w × 0.5^((actions/action_scale)²)
  + (1 − w) × 0.5^((tokens/token_scale)²)
priority_score = (A + B) × C
```

C is a heuristic based on measured completion-hazard decay, not an exact conditional probability. The implementation scales the score for integer priorities and combines it with the continuation bands.

| Setting | Default | Role |
|---|---|---|
| `ARC3_PRIORITY_HUMAN_ACTIONS` | `25` | h in the action-efficiency estimate. |
| `ARC3_PRIORITY_ACTION_SCALE` | `115` | Action falloff scale. |
| `ARC3_PRIORITY_TOKEN_SCALE` | `62000` | Generated-token falloff scale. |
| `ARC3_PRIORITY_ACTION_WEIGHT` | `0.25` | Action contribution to C; tokens contribute the remaining 0.75. |
| `ARC3_PRIORITY_TAIL_BASE` | **`2`** | Starting future-level tail. |
| `ARC3_PRIORITY_TAIL_CAP` | **`4`** | Maximum ordinary tail. |

The effective harness defaults are 2/4 even though the standalone helper’s function signature has 6/6 defaults. The wrapper supplies the environment-derived values. The successful 5/5 competition configuration is an explicit override.

**Counter correction is unconditional.** Completed-action accounting now takes the real engine action number into the call summary’s `end_action_num`; priority reads that corrected summary. There is no remaining `ARC3_PRIORITY_FIX_ACTIONS` switch.

### Pace adjustment

| Setting | Default | Role |
|---|---|---|
| `ARC3_PRIORITY_PACE` | Off | Adjusts expected cost using the speed of earlier completed levels in this game. |
| `ARC3_PRIORITY_PACE_REFERENCE` | Bundled reference JSON | Optional replacement reference data. |

The implementation smooths relative completion costs, shrinks estimates toward neutral when little evidence is available, and clips the multiplier to 0.5–2. It adjusts the token falloff scale and discounts value by expected cost. It does not require knowledge of the game’s final number of levels.

### Tail fading and endgame

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PRIORITY_TAIL_FADE` | Off | Linearly reduces B near the deadline while retaining C. |
| `ARC3_PRIORITY_TAIL_FADE_FRACTION` | Unset | A value in (0, 1] selects the final fraction of total runtime for fading; 0.2 means the final 20%. |
| `ARC3_PRIORITY_TAIL_FADE_MINUTES` | `90` | Fallback fade duration when no valid fraction is configured. |
| `ARC3_ENDGAME_START_MINUTES` | `0`: disabled | Enables endgame after this many elapsed minutes. Ignored when tail fade is enabled. |

The solver configures a shared clock from session start and the applicable finite per-game/soft deadline. Fade needs a usable finite deadline. A log line announces fade start and its remaining duration.

**Tail fade takes precedence over endgame.** When `ARC3_PRIORITY_TAIL_FADE` is enabled, the endgame switch is disabled even if `ARC3_ENDGAME_START_MINUTES` is set. Scheduling uses `(A + faded_B) × C` throughout the run. With tail fade disabled, enabling endgame switches to `level × (0.25 + efficiency) / pace`, dropping both B and C. Without queue refreshing, endgame requests also receive a 500,000 phase offset. Refreshed ranking evaluates candidates consistently instead of mixing priorities computed in different phases.

### Optional remaining-level lookup and score normalization

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PRIORITY_TAIL_LOOKUP` | Off | `remaining` selects the heuristic 8/7/5/0 tail by remaining levels; `1` selects the empirical 80k-token continuation table. Off keeps the constant/capped tail. |
| `ARC3_PRIORITY_TAIL_EFFICIENCY` | `0.8` | Multiplies empirical lookup-table future reward by assumed future action efficiency. Has no effect with `remaining` or with lookup disabled. |
| `ARC3_PRIORITY_SCORE_NORMALIZATION` | Off | Normalizes A only across games with different total level counts. B and the large cache-priority bands are unchanged. Independent of lookup. |

The total level count comes from harness metadata when available, is clipped to 6–10, and falls back to 10 when unavailable/invalid.

`ARC3_PRIORITY_TAIL_LOOKUP=remaining` chooses B according to the number of levels left **after completing the current level**:

| Remaining levels | B |
|---|---:|
| 3 or more | 8 |
| 2 | 7 |
| 1 | 5 |
| 0 | 0 |

These are final heuristic tail values, without an efficiency multiplier. This is the lookup mode used in the open-sourced submission, together with A-only score normalization.

`ARC3_PRIORITY_TAIL_LOOKUP=1` instead selects the empirical table estimating future weighted progress within an 80,000-token continuation budget. It does not bake in the separate efficiency factor. It was constructed from empirical level-completion data, with pooling/extrapolation where late-level data were sparse. Ordinary tail fade applies to either lookup result.

Normalization multiplies A by `55 / (N(N+1)/2)`, using a ten-level game as the reference. B is not normalized. The lookup and normalization switches remain off by default; the notebook enables the submission settings explicitly.

### Victory lap
Ported from PR #159 (tag `archive/arm/probe-lap`, design `research/reports/probe-lap-design.md` there) without the stuck probe. It never executed a lap on Kaggle: in `v4-probe-lap-10x` every won game already scored 100, so all 8 laps were skipped.


| Setting | Default | Behavior |
|---|---|---|
| `ARC3_VICTORY_LAP` | Off | After a game is won, play it again without the model: send RESET, which in the WIN state starts a new play with fresh counters, then replay each level's winning attempt (its actions after its last RESET, clicks included) with no failed attempts. The official scorer keeps the best play of each game (arc_agi `scorecard.py`), so a shorter second play raises the score and an unfinished one is ignored. Details below. |

- **When it runs.** The game first gives back its priority-gate slot, since the lap needs no model. The lap is skipped when no level gets shorter, when the visible baselines say the shorter routes score no higher, when the replay would start from a state the first play did not start from, or when less than 60 s of the run's wall clock is left.
- **Checks.** Every action of the first play is recorded as (state hash before, action, state hash after, levels, engine state) while the flag is on; a state hash is an md5 of the whole final frame. The lap checks each step against the recorded hash, level count and state, and stops at the first difference, a GAME_OVER, an engine error, a stop request or the deadline. It never improvises a move. On 77 saved wins from 11 runs every lap reproduced every frame.
- **What the run reports.** The `GameRun` keeps the first play's `history`, `actions_per_level` and `final_score`, so `benchmark.json`, the score timeline and `make eval` still show the first play, comparable with other runs. The lap appends ` victory_lap=<won|aborted|skipped> play2=<score> official=<score>` to the game's `solver_note` in `benchmark.json`; `official` is max(play 1, play 2) by the official formula, or `?` when baselines are hidden. Each game also writes `<game>_p<pass>_victory_lap.jsonl` (route lengths per level against the first play's actions, both plays' scores, status and reason, and in an offline run the per-play scores read back from arc_agi's own scorecard), and the log gets a `[VICTORY_LAP]` line with a running total.
- **Offline run caveat.** The notebook sets `ONLY_RESET_LEVELS` before the Arcade makes each game, so the make-time RESET is a level reset and the first play never reaches the local scorecard; that scorecard then lists the lap only. The harness's `official` value does not rely on it.
- **Kernel-kill guard (competition rerun only).** The per-game deadline (532 min after the benchmark starts) lands after Kaggle's 9 h kernel kill, so in a rerun `_lap_time_left` also counts down to `ARC3_VICTORY_LAP_KILL_LIMIT_S` (default 32400) minus `ARC3_VICTORY_LAP_KILL_MARGIN_S` (default 300) from the notebook process's start (`/proc/self/stat`). A win inside that margin starts no lap, so the kill cannot cut a lap part-way. Off a rerun nothing changes unless `ARC3_VICTORY_LAP_KILL_LIMIT_S` is set.
- **Tools.** `analysis/victory_lap_replay.py` replays saved wins through the same code offline (`analysis/README.md`).

### Final-level hold and park cap (win-reliability levers 3 and 4)

Both act on the refreshed queue (`ARC3_PRIORITY_REFRESH_QUEUE`, on in the Kaggle notebook) and the gate (`ARC3_MAX_ACTIVE_STREAMS`). Both are off by default, and off they leave every price and admission exactly as before. Results: `research/reports/final-level-hold.md` on tag `archive/arm/final-level-hold` (CPU checks, 2026-10-10). Arm: `final-hold-10x` (mem-repair-10x + `ARC3_SCHED_FINAL_HOLD=1`).

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_SCHED_FINAL_HOLD` | Off | Protects a game on its last levels. Two parts: the B floor below, and the hold at a trim. |
| `ARC3_SCHED_FINAL_HOLD_FLOOR` | `3` | B on the game's last level becomes `max(B, floor)`. Both tail tables give 0 there, so a long last level otherwise prices at A × C alone. The floor is applied after the tail fade. |
| `ARC3_SCHED_FINAL_HOLD_RATIO` | `0.75` | At a trim on one of the last `ARC3_SCHED_FINAL_HOLD_LEVELS` levels, the game keeps its slot instead of handing it over when its price is at least this × the best waiter's price (repriced now). It never holds while a never-started game waits. With nobody waiting it holds, which is what a handover would do anyway. |
| `ARC3_SCHED_FINAL_HOLD_LEVELS` | `2` | How many final levels the hold covers (2 = the last and the second-to-last). |
| `ARC3_SCHED_PARK_CAP_MIN` | `0` (off) | **Rejected in simulation: don't use** (−2.3 to −2.5 mean at the hidden ratio, no wins gained offline). A waiting game that has cleared at least one level and has waited this many minutes is ranked above every priced waiter, but still below never-started games. Among capped waiters, the higher price goes first. |

The level count comes from the same metadata as the lookup (clipped to 6–10). A game with more than 10 levels is treated as being on its last level at level 10.

Log lines: `final-level hold: level L of N, price P vs best waiter W; keeping the slot` (only when someone was waiting), and `park cap: re-admitting a level-L game after M minutes waiting`.

## 12. Yielding, reasoning, and provider compatibility

### Resumption behavior

| Setting | Default | Behavior |
|---|---|---|
| `LOCAL_ANALYZER_YIELD_TOKENS` | `0`: disabled | Adds a generated-token threshold for yielding an analyzer turn. Checked between completions; it is not a hard mid-stream token cutoff. |
| `ARC3_YIELD_RESUME_PROMPT` | `full` | Selects the full opener, a shorter continuation prompt, or `state_only`. |
| `ARC3_YIELD_RESUME_TONE` | `commit` | Selects commit-oriented, `neutral`, or `minimal` resumption wording. |
| `ARC3_YIELD_ON_TIMEOUT` | On | Allows a read timeout to yield while preserving useful history rather than losing the turn’s work. |
| `ARC3_OPENER_DEDUP` | Unset: off | Stops re-sending opener content the retained context already holds word for word. `static`: the opener's fixed tail (from "Only tool:" to the end of the prompt text, about 360 tokens of tool and frame guidance the system prompt also carries) is sent only when no retained message carries the identical block, so one copy stays in the window each invocation starts from. `resume_image`: a yield resumption leaves out its current-grid image when the most recent retained image is that same board, with a one-line caption in its place. `1`/`all`: both; parts combine with commas. Decided after the opener-time trim and applied only to the new opener: nothing already sent changes, and the system prompt is untouched. |

These extend the original time-based yielding/tool-step controls. They do not mean that every individual generation is interrupted exactly at the yield threshold. Image behavior on resumed turns is controlled separately, as listed above.

`LOCAL_ANALYZER_YIELD_SECONDS` counts from the start of the invocation, so time spent waiting at the priority gate after a trim counts too. A yield does not release the gate slot; its only effect is a new invocation that re-sends an opener. `ARC3_OPENER_DEDUP` is the code half of the prefill-v2 arm; its sizing is in `research/archive/reports/prefill-v2.md`.

### Reasoning and tool-call compatibility

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_REASONING_HISTORY_KEY` | `reasoning` | Configures the reasoning field(s) retained in request history, including `reasoning_content` for compatible servers/models. |
| `ARC3_PRESERVE_THINKING` | `1` | Sends `preserve_thinking` in chat-template kwargs. False disables it; an empty value omits it. Preserves other template options such as enable-thinking. |
| `ARC3_VERBATIM_HISTORY` | Off | Stores each reply's reasoning and content in history as the server returned them, with only the outer whitespace trim the chat template applies itself, so the next request can reuse the cached reply (issue #3). Off keeps the normalised text described below. |
| `ARC3_REASONING_EFFORT_LADDER` | Empty: disabled | After output-length truncation, moves through configured reasoning-effort values, usually toward less reasoning. Resets after an action executes. |
| `ARC3_TOOLCALL_TEXT_FALLBACK` | On | Recovers recognizable tool calls emitted as text/markup when the backend did not produce structured tool-call objects. |
| `LOCAL_ANALYZER_TOOL_CHOICE` | `auto` | Overrides tool choice; empty, `omit`, or `none` omits the field while still sending tools. Other values pass through. |

The last point is deliberately different from sending the API value `tool_choice="none"`: this harness uses that environment value to omit the field for backend compatibility.

#### What `ARC3_VERBATIM_HISTORY` changes for the model

By default history stores reasoning and content through the transcript normaliser. It removes every line that is empty or whitespace-only, removes literal `<think>`/`</think>` tags, and rejoins lines split on any Unicode line boundary with `\n`. The next request then renders a reply that differs from the tokens the model generated. The hybrid model keeps recurrent-state checkpoints only at the end of prefill and the end of decode, so the server falls back to the previous prompt's end and prefills the whole reply again. Issue #3 measured that fallback on about half of all turns. Checkpoint-slot pressure is the other candidate cause, and `ARC3_PREFIX_REUSE_DIAG` below separates the two.

With the flag on, only the assistant messages in history change, in these ways:

- **Reasoning and content keep their blank lines.** Paragraph breaks and whitespace-only lines stay, as do literal think tags and line separators other than `\n`. This covers the retained reasoning (`ARC3_REASONING_HISTORY_KEY`), the assistant content (before a tool call and in replies without one), and the rolling-summary reply when summaries are on. A truncated summary keeps its appended note. Leading and trailing whitespace is still stripped, which the template's `|trim` does anyway, so that part renders identically.
- **Tool-call markup found in text** is still removed from the stored reasoning and content, but blank lines around it are kept.
- **Unchanged:** tool-call arguments, which were already stored exactly as the server returned them (the python tool rstrips the code it runs from a parsed copy). Also unchanged: degenerate-reply and dead-reasoning stubs, the transcript, prompt logs, the reasoning returned to the solver, knowledge and world-model extraction, and every decision that reads the reply.
- **Context estimate:** retained text is slightly longer (each dropped blank line comes back, at least one character), so the context estimate grows by that amount and trims may come marginally earlier.

### OpenRouter support

Added streaming response assembly, including incremental text, reasoning, tool-call fragments, usage, and provider metadata, plus provider-selection controls and an API-key fallback.

| Setting | Default | Behavior |
|---|---|---|
| `OPENROUTER_STREAM` | Streaming enabled for OpenRouter | Can disable automatic streaming for this provider. |
| `OPENROUTER_PROVIDER_ORDER` | Empty | Optional ordered provider preferences. |
| `OPENROUTER_ALLOW_FALLBACKS` | Off when an order is supplied | Controls fallback outside the requested provider order. |
| `ARC3_OPENROUTER_PROVIDER` | Empty | Request-level provider pinning; takes precedence over general provider-order preferences. |
| `OPENROUTER_API_KEY` | Empty | Additional credential source for OpenRouter. |

These are separate from the existing local/OpenAI-compatible base URL, provider, model, temperature, top-p, top-k, and enable-thinking settings. Changing the served model is a deployment choice, not part of the cumulative harness source patch.

## 13. Request recovery, runtime budgets, and compatibility fixes

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_HTTP_RETRIES` | `3` | In-request retry count for transient HTTP/connection failures. Zero disables retries; negative allows unlimited retries. |
| `ARC3_HTTP_RETRY_BASE_SECONDS` | `5` | Base retry delay. |
| `ARC3_HTTP_RETRY_MAX_SECONDS` | `5` | Backoff ceiling, with jitter and separate handling of server Retry-After. |
| `ARC3_HTTP_RETRY_INITIAL_SECONDS` | `0` | Optional startup grace period for the first request while the model server is becoming available. |
| `ARC3_ANALYZER_RETRY_BACKOFF_SECONDS` | `10` | Delay between outer analyzer retries. |
| `ARC3_MAX_ANALYZER_FAILURES` | `10` | Limits consecutive outer analyzer failures; negative means unlimited, zero stops on the first failure. |
| `ARC3_WARMUP_ACTION_GAMES` | `0` | Lets the first N game sessions execute a counted automatic RESET before waiting for analyzer readiness. |

Additional recovery work:

- Recognizes SGLang’s context-overflow error wording as well as existing compatible-server forms. An overlong ordinary request now triggers history reduction and retry instead of endlessly resending the same oversized prompt.
- Preserves usable analysis/tool history across retries and timeout yields, and avoids treating a new unrelated prompt as the continuation of a failed call.
- Separates transient HTTP retries from context-overflow recovery and read-timeout handling.
- Passes the available timeout into summary generation and provides a finite 8192-token fallback where an unlimited output setting would otherwise make summary generation unbounded.
- Adds a per-game generated-token budget through `--max-generated-tokens-per-game`, the solver’s `max_generated_tokens_per_game`, and Makefile/config plumbing via `MAX_GENERATED_TOKENS_PER_GAME` / `environment.max_generated_tokens_per_game`.
- Uses compatibility fallbacks for newly optional solver state, allowing older pickled solver objects to run without assuming newly added attributes already exist.
- Releases admission resources on completion/failure and supplies the scheduler’s shared deadline information from the solver.

These changes do not increase the server’s true context limit. Correctly matching the harness context setting to the serving configuration is still necessary for efficient operation.

## 14. Diagnostics, transcript viewer, and build plumbing

### Viewer and request accounting — unconditional

- Associates analysis steps with exact recorded request snapshots, including multimodal-aware matching, rather than relying only on reconstructed text.
- Separates the full model context from the current turn’s local transcript in both normal and lazily loaded views.
- Aggregates request count, prompt tokens, and generated tokens across the requests/attempts belonging to an analysis step.
- Makes token badges available in the lightweight initial listing without expensive transcript/segmentation hydration.
- Displays pending transcript state explicitly while a step is still running.
- Combines retry attempts so earlier attempts and executed tool calls are not hidden by the last attempt.
- Prevents the latest-board-state card from duplicating the prior turn’s transcript or token totals.
- Shows world-model updates as separate information and improves their extraction/display.
- Updates runtime-global wording to match `last_action_call_result` and the current tool interface.

### Viewer over saved runs — unconditional, viewer only

No new option; `inference-view` keeps `--host`, `--port`, `--runs-dir`, `--run-dir` (and `make view` its `VIEW_HOST`, `VIEW_PORT`, `VIEW_RUNS_DIR`, `VIEW_RUN_DIR`).

- `--runs-dir` is searched three levels deep, so `../runs/kaggle` lists every `<kernel>/v<N>/` run, newest first by start time, with its mean score. Run ids in URLs are paths relative to `--runs-dir`; paths outside it are refused.
- Every run file may be stored as `<name>.xz` (`*_viewer_data.json`, `*_events.jsonl`, `*_requests.jsonl`, `benchmark.json`, `transcripts/*.txt`); it is decompressed in memory and parsed once per game.
- Games without viewer data are built from `benchmark.json` (actions, levels) and `transcripts/` (reasoning).
- New read-only endpoints `/api/runs`, `/api/frames`, `/api/timeline`, `/api/file`; `lite=1` on `/api/game` and `/api/game-step` drops per-turn boards and the full model window for small phone responses.
- The page is an action-by-action player with a wall-clock score/levels/tokens chart and a phone layout.

### Makefile — unconditional

Configuration-derived variables now use explicit undefined checks and immediate assignment, reducing repeated/recursive shell configuration evaluation while respecting external overrides. This spans game selection, budgets, concurrency, model/server settings, deployment settings, and related paths. Generated-token budget arguments are passed through to the runner.

These are build/run reliability changes, not new gameplay policies.

### Prefix-reuse diagnostic

| Setting | Default | Behavior |
|---|---|---|
| `ARC3_PREFIX_REUSE_DIAG` | Off | Logs one `PREFIXDIAG` line per analyzer request comparing the previous reply with the history message that re-renders it, next to the cache hit the server reported. Every 100 requests a `PREFIXDIAG totals` line sums them. Logging only. |

Each line reads `PREFIXDIAG <game> action=… request=… history=… prefix=… cache=… cached=… prompt=… prev_prompt=… prev_output=… gap_s=… lost=…`:

- **`history`** is `match` when the stored message renders, through a Python mirror of the served Qwen3.x template, to the same text as the reply. `mismatch` adds `diverge_at=<offset>/<length>`, both hashes and the text on each side. `replaced` means a stub stands in for the reply, `absent` that it is not in the request (rolled back or trimmed), and `none` that there was no previous reply on the stream. A request retried after a failure (a context overflow, a timeout) is checked against the same previous reply.
- **`prefix`** is `same` when every message of the previous request is still in place, so only the reply can explain a hit that stops short.
- **`cache`** is where `usage.prompt_tokens_details.cached_tokens` ended, in 64-token pages: `reply_end` (reused, at or past `floor64(prev_prompt + prev_output)`), `prompt_end` (exactly `floor64(prev_prompt)`, so the whole reply was prefilled again), `partial`, `below_prompt`, or `first`.
- **`lost`** is previous-reply tokens prefilled again, counted only with `prefix=same`. The totals line splits it as `lost_tokens_match` / `lost_tokens_mismatch`. Tokens lost with a matching history point at the server (Mamba checkpoint slots), lost with a mismatch at the harness.

The comparison is on text. A reply whose text matches can still re-tokenize differently from the generated ids, which the harness cannot see.

### Settings outside the harness (Kaggle notebook and SGLang)

These are read by the notebook or the SGLang server, not by `ARC3-Inference`, and are listed so every flag of the merged arms has an entry. Details are in `serving/CUSTOM.md` and `kaggle/README.md`.

| Setting | Default | Behavior |
|---|---|---|
| `SGLANG_ARC_MOE_ROUTING` | Unset: off | SGLang process env (an arm's `serve_env`). `base_norm`, `nucleus` or `temp` re-weights the experts the router chose (`serving/runtime_patches/20-moe-routing.py`); unset, the patch changes nothing. Used with `TEXT_CONFIG_OVERRIDE` `num_experts_per_tok` by the `experts*-basenorm` and `user-combo` arms. |
| `SGLANG_ARC_MOE_BASE_K` | 10 | `base_norm`: how many top experts keep their trained weights. |
| `SGLANG_ARC_MOE_PARAM` | 0.9 | `nucleus`: the probability mass kept; `temp`: the temperature. |
| `SGLANG_ARC_MOE_MIN_K` | 1 | `nucleus`: the fewest experts kept. |
| `SGLANG_ARC_MOE_DRAFT` | Off | `1` also re-weights the MTP drafter's MoE blocks. |
| `TAAF_ARM` | Unset | Set by `kaggle/build_notebook.py` to the arm name so a run's output says which arm it was. Inert. |

## 15. Publication scope and exclusions

For the public writeup, the principal additions can be grouped as:

1. Better action access and explanations: UNDO, RESET, and action information.
2. Cross-level learning and clearer observation semantics.
3. Richer perception: structured diffs, rotation-aware shapes, and intermediate animations.
4. Protection against ineffective batches, repeated mistakes, and stale plans.
5. Reusable model-written Python functions with controlled lifetime.
6. Larger-context operation: estimation, blockwise trimming, summaries, and recovery.
7. Cache-aware admission and progress-based allocation of computation.
8. Provider compatibility, diagnostics, and operational fixes.

Describe which of these were enabled in a particular submission separately. The inventory deliberately retains optional unsuccessful or inconclusive experiments that are still in the code. It does not attribute a measured score gain to a single mechanism.

The SGLang Mamba/prefix-cache patches, end-of-prefill LRU refresh, bounded parallel prefetcher, wheelhouse builder, model resharding, quantization choice, and serving performance configuration belong in a separate **serving and deployment** section. They are not changes in this harness diff.

## Source map

All paths below are relative to `ARC3-Inference/`.

| Area | Main source files |
|---|---|
| Prompts, context, summaries, memory, request loop | `inference/agent/prompts.py`, `inference/agent/tool_agent.py` |
| Legibility flags (`ARC3_LEGIBILITY`) | `inference/agent/legibility.py` — new; rows in `inference/agent/prompts.py` |
| Verbatim history text and the prefix-reuse probe | `inference/agent/prefix_reuse.py` — new |
| Memory ledger and memory-v2 card, deltas | `inference/agent/memory_ledger.py`; harness facts in `inference/agent/memory_facts.py` — new |
| Action names and exposure | `inference/agent/action_names.py` |
| Action execution, guard integration, budgets, resets | `inference/framework/solver.py` |
| Runtime state and Python interface | `inference/agent/runtime_state.py`, `inference/agent/python_tool_sandbox.py` |
| Guard bookkeeping | `inference/agent/noop_repeat_guard.py` — new |
| Segmentation and differences | `inference/utils/segmentation.py`, `inference/utils/frame_diff.py` — latter new |
| Animations and images | `inference/utils/animation.py` — new; `inference/agent/vision_context.py` |
| Function retention | `inference/utils/retained_functions.py` — new |
| Priority gate, formulas, references | `inference/agent/priority_scheduler.py`, `inference/agent/priority_pace_reference.json` — new; integration in `tool_agent.py` and `solver.py` |
| Provider transport | `inference/utils/openai_compat.py` |
| Runner/build | `inference/framework/run.py`, `Makefile` |
| Viewer | `viewer/data.py`, `viewer/index.html`, `viewer/server.py`; `viewer/runfiles.py`, `viewer/timeline.py` — new |
