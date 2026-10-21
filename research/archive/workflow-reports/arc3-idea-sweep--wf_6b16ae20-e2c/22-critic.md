# arc3-idea-sweep: critic

Workflow run `wf_6b16ae20-e2c`, phase Gaps. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Static reasoning_effort (medium/low) from game start via chat_template_kwargs; make the truncation ladder cache-safe",
   "category": "model-config",
   "description": "Add an env var (e.g. ARC3_REASONING_EFFORT) read in ToolAgent._harness_template_kwargs() (inference/agent/tool_agent.py ~7208) so EVERY request carries chat_template_kwargs.reasoning_effort. Today the key is only sent after an output-length truncation and only if ARC3_REASONING_EFFORT_LADDER is set, so the model runs at the template default (xhigh) all game. Run three arms on the same game panel with the token-indexed metric: xhigh (current), medium, low; optionally add temperature 1.0 (model-card value) vs the current 0.7 as a second factor. Do NOT toggle effort mid-game: the Qwen3.8 template renders the effort sentence at the very start of the system message (xhigh adds a 'think carefully, validate assumptions, consider alternatives' sentence, medium adds nothing, low adds a 'keep thinking brief' sentence), so any change rewrites token 0..N and invalidates the entire KV and Mamba prefix. The existing ladder flips the prefix twice per truncation (down on truncation, back to default after the next executed action, tool_agent.py ~9412-9423 and ~9664-9671), i.e. two full re-prefills of up to 128K tokens (~12 s each at the 10.5K tok/s effective prefill in WRITEUP). Either drop the ladder, or apply it only at a trim boundary where the prefix is being rebuilt anyway. For per-turn variation without breaking the cache, put a one-line 'keep reasoning brief' nudge in the user opener (the tail), not the system prefix. Do not rely on a server-side thinking_budget: SGLang's is reported as not enforced for Qwen3.6 (issue 25536).",
   "mechanism": "The run is token-starved (WRITEUP: scores were still rising at the end of runs; ~110 games share 10 streams), so tokens per useful turn is the main currency. xhigh is the default, so every turn pays for long deliberation, and because preserve_thinking keeps all prior reasoning in history, long traces also fill the 58K-128K window sooner, forcing more trims and re-prefills. About 4% of replies hit the 12,288-token ceiling (prompts.py comment at ~336-349), each wasting up to 12K generated tokens and yielding no tool call; if mean reply length is 2-3K tokens that alone could be a sizeable minority of all generated tokens (estimate, measure it). medium/low should give more turns per game, longer retained history, and fewer truncations. Risk: lower per-turn accuracy on hard levels, hence the A/B and the option of low only for routine follow-up turns.",
   "evidence": [
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next (reasoning_effort levels xhigh default / medium / low; thinking-mode sampling temperature=1.0, top_p=0.95, top_k=20)",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/chat_template.jinja (reasoning_effort resolved with default 'xhigh' and rendered at the start of the system message; medium emits no instruction, low emits a brevity instruction)",
    "https://unsloth.ai/docs/models/qwen3.8-next (reasoning_effort: xhigh/medium/low/none, medium 'balancing accuracy and speed')",
    "ARC3-Inference/inference/agent/tool_agent.py:1778-1800 (_reasoning_effort_ladder, empty by default), :7208-7233 (_harness_template_kwargs only adds reasoning_effort from the ladder rung), :9412-9423 and :9664-9672 (ladder moves down on truncation, resets after an action)",
    "ARC3-Inference/inference/agent/prompts.py:336-349 (comment: truncation touched about 4% of replies)",
    "WRITEUP.md section 1 settings table (temperature 0.7, 12 Ki max output) and section 2 (reasoning is deliberately retained in history)"
   ],
   "effort": "<2h",
   "expected_impact": "high",
   "status_in_franzen": "partial"
  },
  {
   "title": "Config lint and effective-config dump: fail fast on silent env fallbacks, stale bundle, and the default 30-assistant-message history cap",
   "category": "bugfix-quickwin",
   "description": "At notebook start (and again from inside the first analyzer request) print every resolved ARC3_*/LOCAL_ANALYZER_*/MULTIMODAL_*/EXPOSE_* value with its source (env vs code default), the chat_template_kwargs actually sent, the bundle/git hash from git_status.txt, and the server argv/env. Diff it against a checked-in golden JSON copied from Franzen's published notebook and abort or loudly warn on mismatch. Known silent traps to assert on: (1) EXPOSE_UNDO accepts only on/off/tufa and any other value, including 1 or true, silently maps to 'tufa', which advertises ACTION7 but rejects it (action_names.py:57-71); (2) images are attached only if MULTIMODAL_CONTEXT is exactly current_grid, else no board images at all (vision_context.py:76-83); the Kaggle setup defaults MULTIMODAL_UPSCALE to 4 and the code default is 16 (kaggle.py:199 vs vision_context.py:118); (3) ARC3_HISTORY_ASSISTANT_TURNS defaults to 30 and ARC3_HISTORY_TURN_DRAIN to 0, and the code's own docstring says the cap binds with thousands of tokens unused at a 98K window (tool_agent.py:395-411), so with the defaults the history window slides every turn and the prefix cache is defeated; set it to 0 for the 128K setup; (4) ARC3_PRIORITY_TAIL_FADE silently disables ARC3_ENDGAME_START_MINUTES (CONFIGURATION.md section 11); (5) ARC3_NOOP_GUARD_BORDER and ARC3_GUARDS_FROM_LEVEL raise at import on non-numeric values (solver.py:167, :215) and ARC3_ANIMATION_IMAGE raises on 1; (6) ARC3_TEXT_TOKEN_CHARS_MAX clamps calibration at 3.3 so a clamp that is always active over-estimates context and trims early. Also fail if the attached source dataset version is not the expected one.",
   "mechanism": "The user's best submission (22.88) trails the reference (27.89), and several features Franzen credits (UNDO, images at 10x, retained history, hysteresis) are gated behind exact-string env flags whose invalid values fall back silently rather than erroring. One mis-set flag can quietly remove a feature or defeat prefix caching for the whole 12-hour run, and Kaggle competition reruns return no logs to debug it (diagnostics are minimal in a rerun). A 30-minute lint that runs on the first request is the cheapest possible insurance and also makes every later A/B trustworthy.",
   "evidence": [
    "ARC3-Inference/inference/agent/action_names.py:57-71 (invalid EXPOSE_UNDO maps to tufa) and :107-111 (to_engine_action rejects UNDO unless mode is on)",
    "ARC3-Inference/inference/agent/vision_context.py:76-83 (images only when MULTIMODAL_CONTEXT == current_grid); ARC3-Inference/inference/framework/kaggle.py:199 (MULTIMODAL_UPSCALE default 4)",
    "ARC3-Inference/inference/agent/tool_agent.py:395-411 (ARC3_HISTORY_ASSISTANT_TURNS default 30, docstring says it binds at large windows), :361-376 (turn drain default 0)",
    "ARC3-Inference/inference/framework/solver.py:167 and :215 ('a bad value here raises at import')",
    "ARC3-Inference/CONFIGURATION.md section 11 (tail fade supersedes endgame) and section 8 (text-token calibration bounds)",
    "taaf-duck-harness-kaggle-share.ipynb cell 2 (TAAF_MINIMAL_DIAGNOSTICS=1 in a rerun, so no logs come back)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Offline replay rig from requests.jsonl: screen serving flags for tok/s, TTFT, MTP accept length and teacher-forced NLL drift in minutes instead of 12-hour runs",
   "category": "other",
   "description": "Build inference/tools/replay_requests.py. Input: the requests.jsonl produced by --save-request-logs on one local or Kaggle public-25 run (each line has the full messages incl. images, tools, chat_template_kwargs, usage, finish_reason; tool_agent.py:4808-4880 and :9346-9372). Mode A (throughput and cache): replay N=10-14 concurrent recorded sessions in recorded order against a candidate server, free-generating with the original sampling params, and log aggregate decode tok/s, TTFT versus cached-prefix length and versus number of history images, and avg_spec_accept_length from /server_info. Mode B (quality gate): for each recorded request, teacher-force the next recorded assistant message (it appears in the following request) through SGLang with logprobs and compare mean NLL and top-1 agreement against the baseline config. Use it to approve or reject every risky serving change on the survivor list (top-10 to top-8 experts, relaxed MTP acceptance, FP8 dense stack, Triton _MAX_M change, bf16 SSM state, FR-Spec map rebuild) with a numeric threshold before spending a Kaggle run, and to find the best num-streams and chunked-prefill size under real traffic. Strip or cap images only as a separate arm so image-preprocessing cost shows up in TTFT.",
   "mechanism": "Today any serving change can only be validated by a full 12-hour game run whose score noise (25 games, single pass, Duck notes a run losing 43% after an unrelated change) is far larger than the effect of a lossy kernel or routing tweak, so quality regressions are invisible and throughput gains are unproven. A deterministic replay gives a paired, near-noise-free measurement of speed and of distribution drift, turning the high-risk, high-reward serving ideas into safe screened candidates. It also exposes whether per-request multimodal preprocessing of re-sent history images (SGLANG_MM_PREPROCESS_DEVICE=cpu) is a hidden latency tax at 10+ streams.",
   "evidence": [
    "ARC3-Inference/inference/framework/run.py:1989-1994 (--save-request-logs flag)",
    "ARC3-Inference/inference/agent/tool_agent.py:4808-4880 (requests.jsonl schema) and :9346-9372 (request and response snapshots with usage and chat_template_kwargs)",
    "serving/build_bundle_pennyroyal.sh:1232-1246 (runtime env incl. SGLANG_MM_PREPROCESS_DEVICE=cpu) and :1405-1408 (lossless acceptance thresholds, hot-token map)",
    "ARC3-Inference/inference/agent/../framework/solver.py:173-181 (comment: a run lost 43% of the previous competition score after bundled prompt changes, showing confounded noisy A/Bs)",
    "https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding (server_info exposes avg_spec_accept_length)",
    "WRITEUP.md section 1 (P90 aggregate decode 836.6 tok/s, prefill 10.5K tok/s as baseline numbers to beat)"
   ],
   "effort": "1 day",
   "expected_impact": "high",
   "status_in_franzen": "absent"
  },
  {
   "title": "Halve Mamba state memory with --mamba-ssm-dtype bfloat16 and re-spend it on more slots or KV (gated by the replay rig)",
   "category": "serving",
   "description": "The runtime env already sets SGLANG_MAMBA_CONV_DTYPE=bfloat16 but nothing in the bundle or README sets the SSM state dtype, so GDN recurrent state is presumably the fp32 default. Launch a variant with --mamba-ssm-dtype bfloat16 and compare against baseline on the replay rig (NLL drift on 60K-120K-token prompts, plus a long-context recall probe on the retained function and board facts). If it passes, re-spend the saved memory: raise --max-mamba-cache-size well above 60 slots (fewer evicted checkpoints for the frozen games waiting in the priority gate, headroom for 12-14 streams) or give it to the KV pool/static-mem fraction safety margin. Check the Flash-Next per-slot size in the server startup log first, since the quoted numbers below are for a different 48-GDN-layer geometry.",
   "mechanism": "Cache pressure is the thing Franzen's whole serving work fights (60 Mamba slots, sparse checkpoints, LRU refresh). Every slot is a large fixed state; per the SGLang cookbook the 27B geometry is 153.9 MB per slot at fp32 versus 78.4 MB at bf16, so bf16 roughly halves the pool and hands the difference to KV cache. With 60 slots that is on the order of 4-5 GB on a card already run at static fraction 0.96, which can convert into more concurrent streams or fewer re-prefills. Accuracy risk (bf16 recurrent state drifting over long contexts) is exactly what the teacher-forced NLL gate is for.",
   "evidence": [
    "https://docs.sglang.ai/basic_usage/qwen3.html (Qwen3-Next usage lists --mamba-ssm-dtype bfloat16|float32, default float32, and --max-mamba-cache-size)",
    "https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B (state slot 153.9 MB at fp32 vs 78.4 MB at bf16 for 48 GDN layers x 48 heads x 128 x 128; as summarized by search, verify against the Flash-Next startup log)",
    "serving/build_bundle_pennyroyal.sh:1234 (only SGLANG_MAMBA_CONV_DTYPE=bfloat16 is set) and :1401-1408 (retained launcher choices list no SSM dtype)",
    "WRITEUP.md section 1 settings table (Mamba cache capacity 60 slots, static memory fraction 0.96)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Sweep MTP draft depth at the real batch (steps 1-4, draft tokens, topk) and try SGLang adaptive speculative steps",
   "category": "serving",
   "description": "Speculative steps are fixed at 3. On the replay rig, sweep --speculative-num-steps in {1,2,3,4} with --speculative-num-draft-tokens = steps+1 (and eagle-topk 1 versus 2 if the Pennyroyal build supports tree drafts for the GDN path) at 10, 12 and 14 streams, recording tokens/s, accept length and GPU time per verify step. Then try --speculative-adaptive (optionally with a small adaptive_spec.json such as candidate_steps [1,2,3] for batch sizes 8-31) and read avg_spec_accept_length from /server_info. Fill the CUDA-graph batch sizes that each step count needs (ties to the graph-gap idea).",
   "mechanism": "In an MoE with 512 experts and 10 routed per token, verifying more draft tokens per stream touches more distinct experts per forward, so the best draft depth at batch 10-14 is not the best at batch 1; SGLang's own defaults cap candidate steps at [1,3] for batch sizes 8-31. Because this workload is decode-throughput bound (score still improving at the end of runs), even a 5-8% tok/s change translates to more turns for every game. This is distinct from relaxed-acceptance and expert-count ideas: it changes how many tokens are verified, not how permissive verification is. Adaptive mode is documented only for EAGLE with topk 1 and does not mention hybrid/Mamba models, so treat it as experimental and keep the static sweep as the fallback.",
   "evidence": [
    "https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding (flags --speculative-adaptive / --speculative-adaptive-config; EAGLE and topk 1 only; default candidate_steps [1,3,7] for BS 1-7, [1,3] for BS 8-31, [1] for BS>=32; falls back to static if unsupported)",
    "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next (speculative flags and --speculative-adaptive listed for this model)",
    "WRITEUP.md section 1 settings table (Speculative steps 3, 10 parallel streams)",
    "serving/build_bundle_pennyroyal.sh:1401-1408 (MTP draft, hot-token map, gdn-mtp-cache-mode none)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Lower CPU priority and cap concurrency of sandbox tool subprocesses so they cannot starve SGLang's scheduler threads",
   "category": "bugfix-quickwin",
   "description": "Each python tool call spawns a fresh `python -I -S` child (python_tool_sandbox.py:946-957) with RLIMIT_CPU = timeout+1 (about 31 CPU seconds, :320-333) and no nice level, and up to 10 games run these concurrently on a Kaggle VM that also hosts SGLang's scheduler, tokenizer manager and detokenizer processes plus CPU image preprocessing (SGLANG_MM_PREPROCESS_DEVICE=cpu). Add preexec_fn=lambda: os.nice(10) (or start the children via nice/ionice), optionally os.sched_setaffinity away from the cores SGLang is pinned to, and a process-wide semaphore limiting simultaneous sandbox children to max(2, cpu_count-3). First check `nproc` in the notebook and log decode tok/s versus number of live sandbox children during a replay to confirm contention exists.",
   "mechanism": "SGLang's per-step scheduling and detokenization are latency-sensitive CPU work; a burst of model-written pure-Python BFS (the system prompt explicitly encourages BFS/beam search, prompts.py PYTHON_ADDENDUM_TAIL) from several streams at once can steal cores and add milliseconds to every decode step for all streams, costing aggregate throughput without any visible error. Niceing the children converts that into slightly slower tool calls (cheap) instead of slower decode (expensive). If the VM has many spare cores the change is harmless.",
   "evidence": [
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:946-957 (subprocess.Popen with no nice or affinity) and :320-333 (RLIMIT_CPU)",
    "ARC3-Inference/inference/agent/prompts.py:263 (prompt pushes explicit BFS, DFS, beam search)",
    "serving/build_bundle_pennyroyal.sh:1235-1246 (runtime env: CPU image preprocessing, MAX_JOBS=4 JIT builds)",
    "ARC3-Inference/inference/framework/solver.py:5 and :2146 (each game runs on its own worker thread in one process)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Fix multi-tool-call replies: an executed action followed by skipped calls rolls back the entire turn's history",
   "category": "bugfix-quickwin",
   "description": "In the tool loop (tool_agent.py ~9658-9690), if the model emits several tool calls in one assistant message and an action executes in a call that is not the last, the remaining calls are skipped and preserve_history is set to False; the finally block then restores previous_history_messages (~9776-9779), discarding the whole turn: the reasoning, earlier inspection exchanges in the same turn, and the executed action's own tool result, while the game state has already advanced. Same for a control yield mid-list. Fix: commit the history but trim assistant_message['tool_calls'] to the calls actually dispatched (so no tool_call_id is left without a result), or answer each skipped call with a short synthetic tool message ('skipped: an earlier call in this reply executed an action; re-issue if still needed'). Add counters for (a) replies with more than one tool call and (b) rollbacks caused by this path, and print them at game end.",
   "mechanism": "Throwing away a turn burns its generated tokens (often thousands of reasoning tokens) and leaves the model's next context without the knowledge that produced the action or its result, so it may re-derive it or repeat probes that cost game actions. Frequency is unknown, which is why the counters come first, but the repo author already flagged it with a Note comment and the fix is a few lines.",
   "evidence": [
    "ARC3-Inference/inference/agent/tool_agent.py:9681-9690 ('Note: skipped later calls discard this turn's history'; preserve_history = False when tool_index < len(tool_calls) - 1)",
    "ARC3-Inference/inference/agent/tool_agent.py:9774-9779 (finally: history restored to previous_history_messages when preserve_history is False)",
    "ARC3-Inference/inference/agent/prompts.py:374-376 (system prompt encourages multiple python calls per step)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Cross-game priors primer and one worked example in the cached system prompt, keyed by the game's valid-action set",
   "category": "prompting",
   "description": "The system prompt is fully static and generic (prompts.py: scene guidance, HUD-bar warning, Python rules) and the same text is shared by every game, so it is prefix-cached for free. Add a compact primer (target under 600 tokens) of recurring ARC-AGI-3 mechanics and goal types distilled from the 25 public games (use the offline environment files shipped in the competition dataset and the saved transcripts: movement-to-target with blockers, push/collect/key-door, click-to-toggle or paint, pattern matching, resource/budget bars, level-to-level rule additions), plus one short exemplar of an efficient discovery sequence (probe each available action once, diff, infer, then search). Select among 4-6 variants by valid_actions (movement-only, movement+SPACE, click-only/MOUSE, movement+MOUSE, UNDO available) which the solver already passes to the agent every turn; each variant stays shared across all games of that type so the cache still works. A/B on the fixed game panel with the token-indexed metric; keep the primer short and hedged ('common, not guaranteed') to avoid biasing the model on unfamiliar games.",
   "mechanism": "Level 1 of a game is where the model knows least and burns the most tokens and probe actions (the code comments note level 1 is where it 'probes hardest'); each wasted probe action lowers (baseline/actions)^2 for that level, and each wasted token delays other games. Genre priors cut both: fewer probe actions and a faster first hypothesis. The private games are new, but they come from the same engine and design vocabulary as the public 25, so priors transfer better than game-specific hints would. Cost is near zero at inference time because it sits in the cached prefix.",
   "evidence": [
    "ARC3-Inference/inference/agent/prompts.py:116-142 (GAME_OVERVIEW and VISUAL_GAME addenda contain no genre knowledge or exemplar) and :262-278 (generic Python guidance)",
    "ARC3-Inference/inference/agent/tool_agent.py:3792-3888 (_build_system_prompt is static per process, so adding variants keeps prefix sharing)",
    "ARC3-Inference/inference/framework/solver.py:1054-1058 and :168-176 (valid_actions passed each turn; comment that level 1 is where the model 'probes hardest')",
    "ARC3-Inference/inference/framework/kaggle.py:63-89 (the 25 public game ids available as source material)",
    "tufa-arc-agi-framework/src/taaf/game.py:381-413 (score is quadratic in actions per level)"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Failure-taxonomy miner over saved runs to rank which fixes to build first",
   "category": "other",
   "description": "Write a script on top of inference/tools/traces.py and the requests.jsonl/events.jsonl artifacts that tabulates, per game and per level over one full public-25 run with diagnostics on: tokens and wall time before the first action; actions, game_overs, RESET and UNDO counts; fraction of no-op or HUD-only actions; replies with finish_reason=length; replies with no tool call; multi-tool-call rollbacks; tool timeouts (and retained-function loss after them); python errors grouped by type (blocked imports such as numpy, KeyError on segmentation fields, NameError after function loss, recursion errors); tool outputs hitting the 3072-token cap; yield and resume counts; 'text_token_calibration ... clamped from X' occurrences; context trims per game and prefill tokens spent after trims; tokens spent after the last level completion (dead time). Emit one ranked table of 'tokens wasted by cause' and 'actions wasted by cause'. The real competition rerun returns no logs (minimal diagnostics), so this must be fed by local or Kaggle offline public-25 runs; do the first one as soon as a working configuration exists and re-run after each major change.",
   "mechanism": "The survivor list is large and based on code reading; the build order should come from measured waste. Franzen already did this by hand once (SAFE_MODULES comment: difflib imported 9 times across 32 transcripts), which is how that fix was justified; automating it turns every later run into guidance on whether numpy imports, timeouts, truncations or no-op loops are the real top sink, and gives per-game failure labels (e.g. games that die from budget bars versus games never understood) for targeted prompt fixes.",
   "evidence": [
    "ARC3-Inference/inference/tools/traces.py:1-40 (per-game chat traces with level_transitions and per-level scoring from saved artifacts)",
    "ARC3-Inference/inference/agent/python_tool_sandbox.py:76-95 (comment: difflib measured across 32 transcripts, imported 9 times, used 0)",
    "ARC3-Inference/inference/agent/tool_agent.py:8296-8303 (calibration note format with 'clamped from')",
    "taaf-duck-harness-kaggle-share.ipynb cell 2 (TAAF_MINIMAL_DIAGNOSTICS in competition reruns)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Token-indexed score curves and per-level hazard metrics for lower-variance A/B decisions",
   "category": "other",
   "description": "Extend inference/tools/eval.py and significance.py: from each game_run's history (ActionRecord.generated_tokens and level boundaries via actions_per_level) compute, for every game, the cumulative generated tokens at each level completion, then report (a) score as a function of generated tokens per game (e.g. at 50K, 100K, 150K, 250K tokens, interpolated) so arms are compared at equal token budgets and prompt/harness changes are separated from serving-speed noise, (b) per-level time-to-solve survival curves with censoring (the same construction as Franzen's C-calibration plots) and a paired bootstrap over level attempts (about 100-150 per run) instead of only 25 per-game means, and (c) action efficiency (baseline/actions) conditional on completion. One long run then yields several budget points at once. Use --max-generated-tokens-per-game as the budget cap for cheap short arms.",
   "mechanism": "significance.py pairs 25 per-game mean scores with a normal approximation, and wall-clock runtime budgets mix model quality with throughput. Equal-token comparisons remove the throughput confound for prompt/perception/reasoning-effort changes, and level-level paired statistics give several times more samples, so smaller real effects become detectable within the user's limited Kaggle GPU hours. This sharpens the existing panel and paired-seed protocol rather than replacing it.",
   "evidence": [
    "ARC3-Inference/inference/tools/significance.py:1-35 (paired per-game mean, normal approximation Phi(mean/(sd/sqrt(n))), n = games)",
    "tufa-arc-agi-framework/src/taaf/game.py:240-300 and :381-413 (ActionRecord generated_tokens, actions_per_level, per-level score formula)",
    "WRITEUP.md section 3 (725 level attempts from four runs used to calibrate the completion hazard C, so the data and method already exist)",
    "ARC3-Inference/CONFIGURATION.md section 13 (--max-generated-tokens-per-game)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Bbox-cropped, auto-zoomed diff image with a coordinate caption instead of a second full-board image",
   "category": "perception",
   "description": "ARC3_DIFF_IMAGE currently attaches a full-board diff image (changed cells in true colour, the rest dark navy) next to the full current-grid image on every post-action opener. Replace it, behind a flag, by a crop of the padded bounding box of changed cells (minimum about 12x12 cells, at 16x scale, cell-aligned to the 32 px vision patch) with a one-line caption such as 'diff crop rows r0-r1, cols c0-c1 of 64x64; N cells changed'. Fall back to the full-board diff when the change spans more than about half the board or at level start. A/B against the current diff using the token-indexed metric and the viewer.",
   "mechanism": "At 10x a 640x640 image is about 400 vision tokens (32 px patches); typical actions change a handful of cells, so a crop is roughly 60-150 tokens, saving about 250-350 tokens per turn (several percent of each turn's context, and less to re-prefill at every trim) while showing the change at higher zoom, where patch quantisation hurts small sprites most. The caption restores the absolute position that cropping removes. Smaller effect than the history-image ideas on the list, but additive with them.",
   "evidence": [
    "ARC3-Inference/inference/agent/vision_context.py:1-40 and CONFIGURATION.md section 4 (diff image: changed cells true colour, unchanged dimmed to navy, full board)",
    "ARC3-Inference/CONFIGURATION.md section 8 (image tokens estimated from 32-pixel patches plus two marker tokens)",
    "WRITEUP.md section 4 (diff images credited with a perception gain; 10x scale)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Fast-forward late-level evaluation: replay recorded solved prefixes at zero token cost, run the LLM only on later levels",
   "category": "other",
   "description": "Add an eval mode that, for each game, replays the action sequence of levels 1..k taken from a previous run's saved action history (games are deterministic: WRITEUP states replaying a fatal sequence fails identically) directly through env.step with no LLM calls, rebuilds history_entries via the solver's normal step_env path, then hands over to the agent at level k+1 with the standard level-start prompt, so the model sees the real history/transitions in its Python globals. Evaluate arms on levels above k only (use the per-level score weights, level L counts L times) on a 6-10 game panel. Optionally also preload the saved trimmed transcript from requests.jsonl as context for a higher-fidelity variant. Caveat: retained functions and the model's own reasoning from levels 1..k are absent, so use it to compare arms against each other, not to estimate absolute score.",
   "mechanism": "Score is weighted by level index (level L weighs L), so late levels dominate the final score, yet each dev run spends most of its tokens and wall time getting through cheap early levels before it reaches them, and variance there is high. Skipping the prefix concentrates compute on the levels that matter and shortens the iteration loop on prompt, memory, and scene-card ideas aimed at late-game difficulty.",
   "evidence": [
    "tufa-arc-agi-framework/src/taaf/game.py:381-413 (level weight = level_idx + 1; per-level score (baseline/actions)^2)",
    "ARC3-Inference/inference/framework/solver.py:962-985 (seed_initial_history and history_entries path used to rebuild runtime state)",
    "WRITEUP.md section 4 (deterministic games, 'replaying the sequence that just killed you will kill you again')",
    "WRITEUP.md section 6 (individual runs varied considerably; compute limited)"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  }
 ]
}
```
