# arc3-idea-sweep: verify:5/10

Workflow run `wf_6b16ae20-e2c`, phase Verify. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "verdicts": [
  {
   "id": "animation-frames-last8",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.5,
   "confidence": "medium",
   "reason": "Gap confirmed in code. solver.py:1812-1830 keeps ONE animation_record. Every executed non-RESET action overwrites it, and a non-animating action sets it to None. tool_agent.py:~7990 `_handle_animation` serves only that record, and sandbox TransitionView (python_tool_sandbox.py:230) has no `.frames`. Per-action count/bbox is already in transitions[i].result['animation'], so only the frames part is missing. Franzen's own control transcripts (fz-arms-v4-control) show the model hitting the limit. m0r0: 'animations are only recorded for the last executed action of a call'. sp80: 'last_animation_frames now refers to test D. I can't re-read test C's frames'. su15 twice: it spends real actions on UNDO plus a re-click just to re-read an animation. sc25 also. Siriki's M111 docstring (competitors/kaggle/arc-agi-3-duck-18-1gc-submit .txt:3471-3520) claims mid-animation-only evidence in 22/25 games. That is his own engine replay on his harness, and I found no score result for the M111 arm, so the size of the gain is unproven. WRITEUP s6 negatives cover auto-inserted timelines and composite images. This idea is lazy and adds nothing to the prompt, so it is a different variant. Impact is capped: in the control run 97% of lost points are incomplete levels (51.98 of 53.75). Any gain is confined to animation-heavy games such as sp80 (4.8), su15 (13.3), sc25 (0), tn36 (0) and s5i5 (41.7). Store frames as hex-row strings: 110 live sessions x 8 x 32 frames of tuple-of-int grids is ~1 GB of host RAM. Skip the LHS tween/blink classifier.",
   "first_step": "framework/solver.py ~1812-1830: append each animated record (chain compressed to hex rows, deduped, <=32 frames, even subsample, first and last kept) to a deque(maxlen=8) keyed by action_num. Add an optional action_num to the `query=animation` branch (solver.py:1490) and pass it through tool_agent._handle_animation (~7990). In python_tool_sandbox.py give TransitionView a lazy `.frames`, fetched via _animation_fetch('frames', action_num=result['action_num']). Add one clause to the transitions line in the prompt."
  },
  {
   "id": "submission-protocol",
   "keep": true,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 0.1,
   "confidence": "medium",
   "reason": "This is selection hygiene, not a score lever, so impact is indirect. Checked: inference/tools/significance.py exists and does paired per-game tests (`make significance`). The sglang wheel really is cp312 (fz-arms.log:146 installs sglang-...-cp312-...whl; the interpreter is python3.12), so a fail-fast assert is cheap. I could NOT verify the 'default image moved to Py3.13 on Oct 1' claim (Kaggle pages are JS-only and web search found nothing). The user's Oct 3 Save&Run still ran 3.12, and Save&Run would catch an import failure anyway. 'Prioritize completion arms' holds on the user's own data: recomputing from fz-arms-v4-control/summary.txt, 51.98 of 53.75 lost points are incomplete levels and only 1.77 are efficiency below the cap. Noise evidence is real: the 'why one run lies' notebook quotes 25.77 mean / SD 3.93 for Franzen copies (author-reported, discussion 745062), and the user's own copy scored 20.87 vs 27.89 and 27.80. Two caveats. The FT09/SP80/TN36/VC33 panel was derived from Tufa's Qwen3.6 20-pass data and the notebook itself warns it may not transfer (worst fold 0.573), so build a panel from the user's 8 existing fz-* runs instead. Three matched seeds x 2 arms x ~2 h is ~12 GPU-h of the weekly quota, so reserve it for the one confirmation that matters (ledger-on-trim vs control, which the user already wants as r2).",
   "first_step": "Add `assert sys.version_info[:2]==(3,12)` and a printed CFG/env diff vs the control notebook at the top of the notebook builder (franzen/kaggle/build_notebook.py). Run `make significance` pairwise over the existing fz-* score files, and derive the reject-panel from those."
  },
  {
   "id": "sim-110-scale-testing",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.5,
   "confidence": "high",
   "reason": "Half of this is already built by the user. franzen/analysis/replay_scheduler.py docstring: 'the scored rerun puts 110 games on 10 ... replays per-level token costs at the hidden-set ratio', with `--games 110` as the default. arms.md already reports replays land within ~1.5 points of real runs, and backlog #7 is already 'test scheduling in the replay simulator first'. The ledger-fitted and stallswap arms are built (474a0ab). CompetitionArcadeServer.official_110() does exist (taaf/competition_arcade.py:155). The new part is a real 110-clone run, which is ~532 min (~9 GPU-hours, about a third of the weekly quota) per sample. It buys no variance reduction over 4 independent 25-game runs at equal compute, only fidelity of the scheduling regime. That only matters if a scheduler arm survives the sim, and the odds are poor: UCB failed, Franzen's pace and lookup variants gave no gain, and findings.md puts the cost of the never-cleared final level at about a quarter of tokens, so any win is capped at roughly +1-3. It is a testing method, not an improvement. Fold it into backlog #7 and use official_110 once, only as a final infra check of any change that adds per-game host memory.",
   "first_step": "No new work. For any scheduler arm run `python analysis/replay_scheduler.py --games 110 --reps 200` first (franzen/base 9f64abb)."
  },
  {
   "id": "grid-lines-10x",
   "keep": false,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "low",
   "reason": "vision_context.py has no grid or axis drawing, so it is absent. The cited paper is real (arXiv 2608.21170, Kaesberg et al., SPaRC puzzle boards). Appendix E.2 does report grid lines alone as +1.4/+0.2/+2.6 pp for the three strongest models. But those are small puzzle boards, and the same section shows the benefit comes from the labelled margin (+2.4 to +7.6 pp), not the lines. No evidence exists for 64x64 ARC boards at 10 px per cell. Drawing a 1-px dark separator over 19% of each cell's area can blur dark/charcoal/black cells against the line colour, so it could hurt. The effect is at best ~1 point offline, below the ~3-point noise floor, so it cannot be validated in the time available. In the control transcripts the model's coordinate problem is not cell boundaries but a wrong belief about the image size (see the labelled-axes entry), which lines do not fix.",
   "first_step": "n/a (dropped). If revisited: vision_context.py image render (~line 150), overwrite the last pixel row/col of each cell."
  },
  {
   "id": "labelled-axes",
   "keep": true,
   "status_in_franzen": "absent",
   "impact": 1,
   "effort_days": 0.1,
   "confidence": "medium",
   "reason": "Keep ONLY a reduced version: a one-sentence image-geometry line, with no image or margin changes. The axis-margin part is refuted by the user's own history. b3 (labelled grid + diff image) scored 12.34 vs 17.74. c2 (12x + ASCII overlay) scored 5.60 at 117k tokens per level. GB10 ascii-upscale: 'got 2 of 4 coordinate reads wrong'. Also arXiv 2608.21170 says a BLANK margin hurts. The geometry sentence is backed by the control transcripts. Counting explicit image-size beliefs in fz-arms-v4-control, the model says the board image is 720 px 13 times and 736 px 3 times, versus the correct 640 px only 4 times. Games affected include lf52, g50t, ls20, re86, sc25, sp80 and tu93. It then converts pixels to cells with 11.25 or 11.5 px/cell, a 12-15% position error, e.g. re86: 'image is 736 px for 64 cells'. The system prompt (prompts.py:241 MULTIMODAL_CONTEXT_ADDENDUM) never states the scale. The line goes in the static system prompt, so it costs nothing per turn and does not disturb the cache. Expected gain is small (the model usually cross-checks with segmentation), but the risk is about zero.",
   "first_step": "tool_agent.py:3811, right after `prompt += MULTIMODAL_CONTEXT_ADDENDUM`, append a line built from vision_context.current_grid_image_upscale(): board images are drawn at {s} px per cell (64x64 board = {64*s}x{64*s} px); image pixel (x, y) maps to col=x//{s}, row=y//{s}; confirm exact coordinates in Python before MOUSE. It applies to the diff images too."
  },
  {
   "id": "prediction-checked-batches",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 1,
   "confidence": "medium",
   "reason": "Weak upside and it overlaps things already tried. (1) It targets action efficiency, but 97% of lost points on the control run are incomplete levels (51.98 of 53.75; only 1.77 is efficiency below the cap). Cleared levels already sit near the cap. (2) Tier A (a `prediction` text parameter) is the user's hypothesis-loop (compliance 93%, no level gain, 'the model never reads the ledgered refutations'). Related tries: thinking-guide had 0/28 Predict lines, e1 predict+uniform scored 2.97, and Franzen's stronger step-verification hint (STEP_VERIFICATION_ADDENDUM, off in the comp config) gave no gain. Tier B would need per-action ASCII predictions, costing roughly 50-200 generated tokens per batch. Generated tokens are the binding resource (~611 tok/s, ~42k tokens per cleared level). (3) The harness already stops batches on no-op, game_over, level_completed and stale state (solver.py:1626 region). In the control transcripts those stops are rare: no_op_action 19, game_over ~9, stale_state 3. The model also already does per-step checks in its own loops. (4) The cited evidence is real but frontier-only. Schema (schema-harness.github.io) reports 42.83 -> 98.98 with Opus 4.8/Fable 5. Tycho (arXiv 2607.28287) uses Opus 5 and GPT-5.6 Sol. The NOOA abstract (2607.20709) says nothing about prediction-checked batches. The Schema number is also vs 'the final Claude Code scratch snapshot', and most of its gain is the backtested world model plus BFS, not the commit_actions stop. Different variant, but expected value is below the cost.",
   "first_step": "n/a (dropped). If revisited: add stop_reason='prediction_mismatch' next to BATCH_NOOP_BLOCK in solver.py:~1626."
  },
  {
   "id": "world-model-kit",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 2,
   "effort_days": 1,
   "confidence": "low",
   "reason": "Not refuted, but a poor use of one of 5 days. Evidence and checks: (a) The cited papers are real but frontier-only. Rodionov 2607.15439 (gpt-5.5/5.6-sol) concludes the three imposed mechanisms are 'not required' at max reasoning and textual beat executable in several conditions. Schema and Tycho are Opus/GPT-5.6 class. Nothing shows a Flash-Next-class model benefits. (b) The model already writes its own simulators and search. In the control transcripts `def bfs` appears 18 times, `def step` 16, `def sim` 8, and 364 loops over transitions/history occur in 24 games. So the backtest habit exists ad hoc. (c) Adoption risk is documented. GB10 sandbox.md: `find` used 82x, `path()` 0x, 'ignores the structural ones'. Tufa's own writeup (via the user's digest, line 54) says hand-crafted tools appeared to hurt. rellik13's writeup shows added prompt text lowered scores (6.42 -> 4.16). (d) The claim '3 of 7 control timeouts were search blowups' is wrong. I read all 7 'Tool timed out after 30s' cases. Only 2 are pure compute (cn04 DFS, re86 4-nested brute force). The other 5 (g50t, sk48, tr87 x2, sc25) are loops of 26-150 action() calls. Host time inside action_handler counts against the 30 s deadline (python_tool_sandbox.py run_sandboxed_python). Per-action latency grows from ~0.1 s to 0.4-1.3 s once history passes ~200 actions, consistent with write_runtime_state rewriting the whole history JSON per action (solver.py:801). On timeout stdout is lost and retained functions are dropped. A bounded bfs helper would fix at most 2 of 7. (e) Schema's 1.6-5x action savings are efficiency gains, the part that is not the bottleneck here. If revisited, use only a narrow check_model(parse, step) replay with mismatch crops and no forced prompt text.",
   "first_step": "Separate quick fix first: stop counting action_handler time against the sandbox deadline (python_tool_sandbox.py ~962, `deadline = time.monotonic() + timeout`) and/or make the per-action state refresh O(1). If the kit is tried, add only check_model to the bootstrap near frame_diff (python_tool_sandbox.py:~400) with no prompt text. Measure adoption on a 10-game x 25-min smoke and abort if under 5% of snippets use it."
  },
  {
   "id": "seed-helper-library",
   "keep": false,
   "status_in_franzen": "partial",
   "impact": 1,
   "effort_days": 0.75,
   "confidence": "medium",
   "reason": "The savings are too small and the same hand-crafted-tool risk applies. The def counts check out: in fz-arms-v4-control `def crop` appears 31 times, `bfs` 18, `snap` 11, about 1,127 defs in total. But the model's helpers are mostly bespoke (rotk, tipblocks, orb, runs, build). Even if every generic redefinition were removed, that is ~100 tokens each, so at most a ~1-2% cut of 4.48M generated tokens, and realistically under 1%. Game-scope function retention is already on (ARC3_PERSISTENT_FUNCTIONS_SCOPE=game, plus IMPORTS), so cross-level redefinition is already avoided. Seeding would need signatures listed in the prompt (more static text) and risks the model trusting a buggy helper. Adoption is uneven: GB10 shows helpers matching the model's habits get used (`find` 82x, delta/changes) while structural ones are ignored (`path()` 0x). Tufa reported hand-crafted tools hurt. Voyager (2305.16291) is real, but its skill library is self-generated and verified, not hand-seeded. The 'Rodionov future work' citation is not supported by the abstract of 2605.05138, which does not mention a helper library. Overlaps with world-model-kit's bfs.",
   "first_step": "n/a (dropped). If revisited: preload self._kept_functions in tool_agent at game start with crop/find/delta only."
  }
 ]
}
```
