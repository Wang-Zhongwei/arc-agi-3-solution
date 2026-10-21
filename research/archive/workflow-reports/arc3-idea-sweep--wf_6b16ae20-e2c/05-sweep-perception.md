# arc3-idea-sweep: sweep:perception

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Turn on the automatic object diff (ARC3_AUTO_FRAME_DIFF) and make it group objects that move together",
   "category": "perception",
   "description": "Set ARC3_AUTO_FRAME_DIFF=1 (keep ARC3_AUTO_FRAME_DIFF_BUDGET=300). Every opener then carries an 'Auto frame diff' covering all actions since the previous analyzer turn (tool_agent.py _prepare_auto_diff / _render_auto_frame_diff). The text is appended to the new user message, so prefix caching is unaffected. Then patch inference/utils/frame_diff.py compute_frame_diff and the renderer in five ways. (a) Add delta=[dr,dc] to each 'moved' entry; today it has only from/to, so the model has to subtract. (b) Merge moved components that share the same delta and touch or are adjacent into one line, e.g. 'B 9x9 + 2x W 1x1 moved (+0,+3) -> rows 12-20 cols 24-32'. Multicolour sprites currently come out as several unrelated 4-connected moves. (c) Collapse changes that sit entirely inside the ARC3_NOOP_GUARD_BORDER band into one 'HUD: Y bar 40->39 px' line. (d) When identical copies vanish in one place and appear in another, print 'gone at ... / new at ...' instead of a fake 'moved'. (e) Skip 1x1 same-hash matches that jump more than about 6 cells. Ship it as its own arm against a same-day control with at least 2 runs each.",
   "mechanism": "The harness hands the model exact change facts, so it learns what moved even when its own probe code filtered that object out. In the user's GB10 runs on this exact checkpoint, the mirror-stripe move in ar25 level 2 reached the model only through the harness diff line. All 4 diff/perception-block runs cleared that level, against 2 of 8 controls (pooled Fisher p about 0.03, pooled post hoc). The diff also replaces inspection-only python calls, which are 58% of calls in Franzen's control run, with 100-300 appended tokens that are cheap to prefill. The model rarely pulls this information itself: frame_diff() appears in only 6% of calls.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md:76-77 (ARC3_AUTO_FRAME_DIFF exists, default Off)",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/franzen-notebook/notebook_cells.py:10223-10228 (submission perception flags: images, diff images, animation; no AUTO_FRAME_DIFF)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:291-303 (section 6 lists failed experiments; auto frame diff is not among them, so its result was never reported)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:3689-3766,3812-3819 (renderer + system-prompt text already written)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/utils/frame_diff.py:243-246 (moved entries carry only from/to; no delta, no rigid grouping)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/GB10-OVERNIGHT-RESULTS.md:59-66 ('Exact change information is the one lever that showed up across arms'; 5 of 5 change-info runs cleared ar25 L2 vs 2 of 8 controls)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/numpy-perception/notes/gb10-arms/numpy-perception.md (mechanism traced in transcripts; matcher fixes v2: fake 1x1 moves 21/31 -> 0/33, vanish/reappear wording)",
    "Franzen control transcripts /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/transcripts: 2275 python calls, 1323 without action(), frame_diff( in only 140 (mined with scratchpad/percep/mine.py, mine2.py)",
    "https://www.greaterwrong.com/posts/Rdwui3wHxCeKb7feK/getting-50-sota-on-arc-agi-with-gpt-4o (an explicit input/output diff representation was one of the text views that got GPT-4o to 50%)"
   ],
   "effort": "half-day",
   "expected_impact": "high",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Patch-aligned board images: A/B MULTIMODAL_UPSCALE=8 (and 16) against 10",
   "category": "perception",
   "description": "Qwen3.8-Flash-Next's vision tower uses 16-px patches merged 2x2, so each image token covers 32 px. At Franzen's 10x a cell is 10 px. Each ViT patch therefore covers 1.6 cells, and the patch-to-cell phase repeats only every 8 cells (lcm(10,16) = 80 px). The same sprite produces different patch contents depending on where it sits. Run three arms. 8x: 512 px, 258 tokens, each patch exactly 2x2 cells and each token 4x4 cells, 36% fewer image tokens. 16x: 1024 px, 1026 tokens, each patch exactly 1 cell. 10x is the control. Optionally add VISTA-style 1-px dark gridlines between cells at 8x; the image stays 512 px. Any rendered size must be a multiple of 32 px, or the processor's smart_resize will bicubic-resample and blur cell edges. Also update _IMAGE_TOKENS_FALLBACK. Offline 25-game noise is about \u00b11.1, so each arm needs at least 2 runs.",
   "mechanism": "Cell-aligned patches give the vision encoder translation-consistent, single-colour patch embeddings. That should improve object and colour recognition, which is where the image already pays off. Franzen's largest image gain was 4x -> 10x. At 8x the same picture costs 144 fewer tokens per image (two images per opener with diff images on), which leaves more text history per trim window.",
   "evidence": [
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/preprocessor_config.json (patch_size 16, merge_size 2, shortest_edge 65536)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:4082-4084 (32 px per vision token; 402 tokens at upscale 10)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:203 (the 4x -> 10x change was the biggest image-related gain; nothing says 8 or 16 was tested)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/reports/run-atlas/versions.md:55 (Wang harness, same Flash-Next checkpoint: upscale 8 scored 19.43, the best Wang-era mean, vs the 17.74 parent at 4x)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739801 (nobody has A/B'd upscale on Qwen3.8; Tufa picked 4x on Qwen3.6, forks use 8)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133 (Tufa: 'Qwen's encoder processes images in 16x16 pixel patches'; 4x worked best for them, which is also patch-aligned)",
    "https://vista-research.github.io/ (strongest published ARC-AGI-3 visual agent renders 8x nearest-neighbour, 512x512, with 1-px gridlines)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Deterministic 'scene card' at each level start (objects, background, HUD bar, click candidates)",
   "category": "perception",
   "description": "Add a new flag ARC3_SCENE_CARD. On turn 1 and on the first opener of every new level, insert a text block of at most 250 tokens built from segment_layer. Contents: the background colour (largest component touching at least 3 edges); HUD candidates (thin edge-flush components, marked 'likely timer/budget bar'); objects grouped by shape_hash with count, colour, h x w and bbox/centre, for up to about 12 groups ordered rarest/most salient first; enclosures (nodes with children); and, when MOUSE is valid, a 'click candidates' list of centres of non-background, non-HUD components. Reuse the ARC3_LEVEL_INVENTORY start-vs-start diff (tool_agent.py:6073-6146) to tag types that are new compared with the previous level. Never include the full board as text.",
   "mechanism": "Whether the model lands on the right mechanic hypothesis in the opening minutes of a level largely decides the outcome. A harness-computed object list removes the 1-2 exploratory inspection calls (about 1.5-2k generated tokens each) the model now spends just finding the objects. On GB10 the model copied the perception block's bboxes straight into its level-1 plan. It is capped and fires about once per level, which matters because the Wang-era run that added the full ASCII board to every turn lost badly.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md:78 (ARC3_LEVEL_INVENTORY exists, Off; it compares only types between level starts)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/numpy-perception/notes/gb10-arms/numpy-perception.md (ar25 level-1 plan '6 LEFT, 11 DOWN' computed from the block's bboxes; per-turn completion tokens 1052 vs 1472/2478 for same-batch controls)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060 (P3.1 'perception hints': deterministic connected-component click candidates kept; also the 'lottery effect' of the opening hypothesis)",
    "https://arxiv.org/abs/2305.18354 (Xu et al., TMLR 2024: an object-based representation nearly doubled GPT-4's solved ARC tasks)",
    "https://github.com/michaelhodel/arc-dsl (objects(grid, univalued, diagonal, without_bg) with mostcolor background: the standard ARC object abstraction)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/reports/run-atlas/versions.md:62-63 (caution: ASCII-only 4.63 and full ASCII + image 9.86 vs 17.74 control, so keep the card compact and infrequent)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Sandbox perception helpers: labelled crops, run-length rows, frame indexing, bbox fields and a multicolour objects() view",
   "category": "tools-repl",
   "description": "Add the following to the FrameView in python_tool_sandbox.py: .rows (a list of 64 strings), __getitem__ so frame[r][c] works, .at(r,c), and show(r0,r1,c0,c1), which prints a two-line column ruler (tens/ones) and row-number prefixes. Add runs(frame, r), which returns absolute-column spans [(c0,c1,char)], and find(frame, ch), which returns cells and bbox. In segmentation.py, add per-node bbox [r0,c0,r1,c1], center, h, w, area (an alias of pixels) and touches_edge. Add objects(frame, bg=None, diagonal=True, multicolor=True), which groups non-background cells into multicolour 8-connected objects in the style of ARC-DSL. Update STRUCTURED_RUNTIME_STATE_ADDENDUM and the line saying the raw grid is 'intentionally not exposed' (prompts.py:193) to match. Optionally expose the numeric grid as numpy when it can be imported. All of this is injected into the sandbox bootstrap, so retained functions keep working.",
   "mechanism": "The model parses .ascii by hand in 39% of calls and recomputes bboxes from boundary points in 17%. Its transcripts show repeated column-offset misreads from unlabelled crops. Flash-Next's tokenizer merges letter runs ('ww', 'WWW', about 28 tokens per 64-cell row, only 11% single-character tokens), so counting columns in printed text is unreliable. Labelled helpers cut wrong coordinates, retries and the boilerplate code the model generates over and over, which lowers tokens per cleared level.",
   "evidence": [
    "Franzen control transcripts /Users/advaith/Documents/studying-stuff/arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/transcripts: .ascii.split in 879/2275 calls, 'boundary' in 391, 247 row-print loops, 3x \"'FrameView' object is not subscriptable\"",
    "Same transcripts, thinking text: 'I misread before due to the column offset' (ka59); 'crop(17,31,14,60), so offset 0 = col 14 ... I misread earlier' (ar25); 113 'misread' mentions in total",
    "Flash-Next tokenizer (~/.cache/huggingface/hub/models--Intel--Qwen3.8-Flash-Next-W4A16-AutoRound/.../tokenizer.json) on 18 real 64-cell rows: letters 27.7 tokens/row (17-36), 11.3% single-char tokens; hex digits 86% single-char",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/python_tool_sandbox.py:195-215 (FrameView exposes only ascii/step/level/shape/segmentation)",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/siriki_src.txt:2393 (3rd-place M2 team: 'the model regularly asks for bbox, area, centroid, width' -> 198 code-error recoveries in 25 games)",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/siriki_src.txt:3304-3306 (they ported the Wang/Ludvig sandbox, whose segmentation nodes carry id/area/bbox)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/numpy-perception/notes/gb10-arms/numpy-perception.md (with .grid plus helpers the model merged probe and act; non-acting calls 11 -> 4 on ar25)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Add a per-action change summary and an animation flag to the action echo",
   "category": "perception",
   "description": "ARC3_ACTION_ECHO=1 is already on in the submission, but each action only reports 'effect' or 'NO-OP'. Extend _action_verdict/_build_action_echo in framework/solver.py to append a cheap summary computed from the previous and new grids: the interior changed-cell count, its bbox, and the top 3 colour transitions (e.g. 'RIGHT -> effect: 18 cells, rows 12-20 cols 21-32, c>b x9 b>c x9'). Add 'HUD-only' when only the border band changed. Add 'animated: N frames, T transient px' when the action's animation had transient pixels, so a click that left the final board unchanged but flashed is not read as inert. Limit it to one line per action and collapse long batches to the first and last 5.",
   "mechanism": "The model often acts in python loops of single action() calls, and the per-turn diff covers only the turn, so exact per-action change facts arrive too late for the loop to branch on. Putting them in stdout at the moment of the call lets the code check results immediately. Bare 'NO-OP' labels also mislead the model: in sc25, clicks echoed 'NO-OP' actually triggered informative red-flash animations.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/framework/solver.py:246-330 (verdict is only effect/NO-OP/LEVEL COMPLETED/GAME OVER)",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/franzen-notebook/notebook_cells.py:10213 (ARC3_ACTION_ECHO=1 in submission)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/frame-diff/notes/gb10-arms/frame-diff.md (the 'last call only' gap: the model runs python loops of single action([...]) calls; on sk48 a block-grab hidden behind a no-op UP took 3 requests (6.6k tokens, ~20 min) to recover)",
    "Franzen control transcripts sc25-635fd71a_p0.txt: 'MOUSE(row=20, col=30) -> NO-OP' while last_animation_timeline showed 'N>R': 36, 'g>R': 45 (an informative red flash); 158 of 1019 clicks were echoed NO-OP across the run"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Labelled coordinate axes on board images (no per-cell glyphs)",
   "category": "perception",
   "description": "Behind a new flag MULTIMODAL_AXES, render each board and diff image with a 32-px margin on the top and left. One merged token wide keeps the patch grid aligned; at 8x the image becomes 544x544 = 289 tokens. Put tick labels every 8 cells (0, 8, ..., 56) in the margin and draw faint major gridlines every 8 cells. Do not add per-cell letters: the user's 12x ASCII-overlay arm failed. Pair this with the image-geometry sentence from the prompt-fixes idea, and tell the model to confirm exact coordinates in Python before a MOUSE action.",
   "mechanism": "Explicit axis labels anchor image positions to cell indices. On Qwen3.5 (the closest model family studied) labelled axes raised grid-planning accuracy, while a blank margin raised \u00b11 offset errors to 17-46% (4-5% with labels). Qwen3.8 has been observed inventing its own canvas size and 0-1000 scales when reading positions off images, and Franzen's transcripts show the same thing.",
   "evidence": [
    "https://arxiv.org/abs/2608.21170 (Kaesberg et al. 2026, Appendix E.2/Table 5: grid lines alone +0.2 to +2.6 pp; labelled margin +2.4 to +7.6 pp; blank margin worse than the original; \u00b11 offset rate 4-5% with labels vs 17-46% with a blank margin)",
    "https://arxiv.org/abs/2402.12058 (SCAFFOLD, COLING 2025: a coordinate dot-matrix overlay plus textual coordinates improves LMM spatial reasoning)",
    "https://voxel51.com/blog/qwen38-max-visual-grounding-fiftyone (Qwen3.8 'invents its own canvas dimensions' and switches to a 0-1000 grid)",
    "Franzen control transcripts: 'image is 736 px for 64 cells -> ~11.5 px/cell' (the real image is 640 px) and 'I misread earlier from the image (the image is scaled)'",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/reports/run-atlas/versions.md:56 (caution: 12x + ASCII overlay + ruler scored 5.60 vs 17.74, with 9x the image tokens)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/ascii-upscale/notes/gb10-arms/ascii-upscale.md (overlay letters went unread and the ruler was misread; the system prompt never mentioned the ruler)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Image token hygiene: no duplicate boards, no HUD-only diff images, cheaper diff images, prune old images only when trimming",
   "category": "memory-context",
   "description": "Four changes. (a) In _build_user_message (tool_agent.py:6573), skip the current-grid image when the board is identical to the last image still in context. This covers no-execution openers ('Your last python snippet executed NO actions'); write 'Board unchanged since the last image' instead. (b) In _prepare_auto_diff (tool_agent.py:5788-5799), skip the diff image when only cells in the border band changed. (c) Add ARC3_DIFF_IMAGE_UPSCALE, e.g. 4x = 66 tokens: the diff image only needs to show where the change is, and full detail is in the current image. (d) At drain time only, when the prefix is being rebuilt anyway, run _strip_message_images on image parts older than the newest 4 image-bearing messages. ARC3_HISTORY_IMAGE_KEEP filters every request, which breaks prefix reuse; this does not.",
   "mechanism": "About 30% of openers re-attach an unchanged 402-token board (about 185k image tokens over 25 games), and full-size diff images cost another 402 tokens each. Removing these frees roughly 5-10% of each trim window for text history at no prefix-cache cost, and cuts vision-encoder work. In the user's runs, retained knowledge per game moved the score more than raw throughput did.",
   "evidence": [
    "Franzen control transcripts: 461 of 1560 openers are no-execution openers and every one carries an image; 2509 images in total; about 79 of 912 diff images followed a single NO-OP action (HUD-only change)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md:92-94 (ARC3_RESUME_PROMPT_IMAGES covers yields only; ARC3_HISTORY_IMAGE_KEEP filters outgoing context and 'can affect prefix reuse')",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:6573-6615 (opener always attaches current_grid_image_part)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/gb10-wt/ascii-upscale/notes/gb10-arms/ascii-upscale.md (historical images displaced 2.5-3.5k tokens of text history; identical copies re-attached on re-prompts)",
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/reports/run-atlas/findings.md:36,63-74 (ledger-on-trim +6.3 mean; more streams with less context per game was flat)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Budget-bar readout: harness-computed 'actions left before reset'",
   "category": "perception",
   "description": "Add a host-side detector to the solver/tool_agent. Within a level, find edge-band components (thin, edge-flush, one colour) whose pixel count falls monotonically as actions are taken. Estimate the drop per action and the length remaining. Report 'Budget bar (top edge, Y): ~N actions left' in the opener, and in the echo when N < 15. Reset the estimate on level change and on death. After a game over, use the death frame to state automatically whether the bar was depleted, which answers the BAR RULE the prompt currently asks the model to work out.",
   "mechanism": "The model spends reasoning tokens re-deriving the bar in about 16% of thinking blocks and still dies of budget exhaustion. A deterministic count turns 'how many moves do I have' into a fact, which prevents budget deaths (deaths waste actions and reset the level) and shortens game-over diagnosis.",
   "evidence": [
    "Franzen control transcripts: 371 of 2299 thinking blocks discuss a bar/timer/HUD; 19 game-overs; sc25: 'Confirmed: budget death ... drains ~1 row/action; ~64 actions per attempt'",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:223-229 (BAR RULE prompt: the model must judge bar depletion from history[-2].frame by itself)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:135-136 (HUD guidance is prose only)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Zoomed crops: a model-requested zoom() image, plus a zoomed change crop when the change is small",
   "category": "perception",
   "description": "Add a sandbox helper zoom(r0,r1,c0,c1) that queues a crop rendered at 16-32 px per cell to be attached to the next opener (at most 1 per turn). The crop should be at least 256 px per side and a multiple of 32, so that the processor neither upscales nor resamples it. Separately, when the interior change bbox is at most 16x16 cells, attach a zoomed crop of that region (labelled with its row/col range) instead of the full-board navy diff image. Off by default; test it as an arm.",
   "mechanism": "At 10x a vision token covers 3.2x3.2 cells, so 1-cell features (legend glyphs, small keys, pattern pieces) fall below one token. A zoomed crop at 16-32 px per cell gives one cell per patch or per token, and only where the model or the change points. This is how VISTA's inspect tool lets frontier models read fine detail. The small-change crop also costs fewer tokens than the full diff image it replaces.",
   "evidence": [
    "https://vista-research.github.io/ (an inspect tool returns enlarged views of board regions; 100% RHAE with Opus 5.0)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/vision_context.py:431-489 (diff image is always full-board)",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/preprocessor_config.json (32 px per token; shortest_edge 65536 means crops under 256x256 get rescaled)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133 (caution: Tufa saw small models struggle with extra frames)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:301 (caution: composite animation images gave no gain, so keep zoom opt-in)"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Re-show the current level's start board (and scene card) at the head of history after each trim",
   "category": "memory-context",
   "description": "When the context drain fires (ARC3_CONTEXT_DRAIN_TOKENS, 58k in the submission), insert one short user message right after the system prompt. It contains the current level's start-frame image (self._cur_level_start_frame) and, if built, the scene card. Mark it 'reference: this level's starting layout'. Insert it only at drain time, so it is prefix-neutral, and stack it with the user's ledger-on-trim arm.",
   "mechanism": "After a 58k-token drain the model loses the image of the level's initial layout. It needs that layout for goal inference, for reasoning about RESET/UNDO, and for comparing against the current state. Re-injecting one image of 258-402 tokens per trim restores it cheaply. Trim-time injection is the pattern behind the user's best Franzen-era arm.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/reports/run-atlas/findings.md:30-44 (ledger placed at the head of history after each trim: 52.56 vs 46.25 control, tokens per cleared level 38.7k vs 42.7k)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:91-97 (blockwise trim drops old history in one go)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:5760-5770 (per-level start frames are already tracked for ARC3_LEVEL_INVENTORY)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Perception prompt fixes: state the image geometry, fix the run-on bullet and broken lines",
   "category": "bugfix-quickwin",
   "description": "Fix prompts.py:132: the bullet lacks a trailing '\\n', so the live system prompt reads '...Sometime they might also be 1x1 tokens.- Some games are logic...'. Also fix 'are usually be rendered' and 'Sometime'. Fix prompts.py:136: the 'A common failure mode...' line is missing its leading '- '. In MULTIMODAL_CONTEXT_ADDENDUM (prompts.py:241-246), add: 'The board image is the 64x64 grid drawn at N px per cell (N = MULTIMODAL_UPSCALE, i.e. 640x640 px); cell (row, col) occupies pixels y in [N*row, N*row+N), x in [N*col, N*col+N); a position read on a 0-1000 scale maps to cell floor(v*64/1000). Use Python for exact coordinates.' Fill in N from the env var at prompt-build time. If the sandbox-helpers idea lands, also reword the repeated opener line 'raw numeric color IDs are not available' (tool_agent.py:7100).",
   "mechanism": "These are zero-cost fixes to text the model reads on every request. The model has been seen guessing the image size wrongly ('736 px ... ~11.5 px/cell'), which produces wrong cell estimates from the picture. The run-on bullet blurs two separate instructions together.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:132 (missing newline; visible verbatim as '1x1 tokens.- Some games' in fz-arms-v4-control transcripts)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:136 (orphan line without bullet)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:241-246 (multimodal addendum gives no scale)",
    "Franzen control transcripts: 'image is 736 px for 64 cells -> ~11.5 px/cell' and 'the image is scaled differently than I think'",
    "https://voxel51.com/blog/qwen38-max-visual-grounding-fiftyone (Qwen3.8 defaults to 0-1000 or invented canvas sizes when reading positions)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  }
 ]
}
```
