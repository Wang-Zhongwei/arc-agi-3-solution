# arc3-idea-sweep: sweep:prompting

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Level-completion debrief: a harness-written summary of how the last level was won, plus an image of the winning board",
   "category": "memory-context",
   "description": "At level-up, tool_agent._build_user_prompt appends LEVEL_START_USER_PROMPT (tool_agent.py:6947-6950). Extend that opener with a block the harness writes itself, with no model call: (a) 'Level N cleared in K actions (A attempts)'; (b) a run-length-encoded list of that level's actions, taken from history between the level's start and the completing transition, e.g. 'UP x3, RIGHT x5, MOUSE(57,36) x12', capped at about 300 chars with attempts split at RESET/game over; (c) the last ~600 chars of the reasoning that came before the completing action (already captured as captured_reasoning; the model's own stated rule); (d) one-line digests of every earlier solved level, so level 1-2 knowledge survives the 58Ki drain; (e) an image of the board just before the winning action, labeled 'this configuration plus action X satisfied the goal', built through the existing _prepare_gameover_images path (tool_agent.py:6148). Optionally also set ARC3_LEVEL_INVENTORY=1 (see the separate idea). The block sits at the end of context, so it has no prefix-cache cost: about 300-700 text tokens plus 402 for the image, once per level. If ledger-on-trim is ported, also render its level notes here.",
   "mechanism": "Score comes mostly from later, heavier levels, and cross-level transfer is what makes them cheap. In the control run, later levels cost a median of 26.4k generated tokens against 15.3k for level 1. The ledger-on-trim arm, which carries cross-level notes, cut the later-level median to 18.8k while level 1 stayed flat (16.6k). Carried knowledge therefore lowers the cost of every later level. The debrief delivers objective, harness-verified facts at the exact moment they are needed, without depending on the model writing tags. The winning-board image shows what 'goal reached' looked like, and objective identification is the stated bottleneck in Tycho. Tufa's talk notes that small models struggle to connect information across distant turns, which favors explicit digests over hoping the old evidence is still in context.",
   "evidence": [
    "computed from arc-agi-3-v3/kaggle-outputs/fz-arms-v4-control/benchmark.json vs fz-ledger-on-trim/benchmark.json: median generated tokens per cleared level 1 = 15,294 vs 16,599; later levels = 26,439 vs 18,767 (single runs each)",
    "arc-agi-3-v3/reports/run-atlas/findings.md \u00a73: ledger-on-trim 52.56 vs 46.25 control, 38.7k vs 42.7k tokens per cleared level",
    "https://arxiv.org/abs/2607.28287 (Tycho): actor gets parsed 'completed-level terminal events'; 'a model may reproduce motion while misidentifying the goal'; terminal-outcome recall 0.0-77.7%",
    "WRITEUP.md \u00a74: game-over fatal-frame images helped, and the level-transfer guidance was part of 'a large improvement'",
    "arc-agi-3-v3/external/harness-feature-matrix.md C#2: Siri M85 pins solved-level rule + winning actions + reasoning",
    "arc-agi-3-v3/final-c1/talk_by_tufa_lab.md: smaller models 'struggle to connect information across distant turns'",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:45-58 (current level-start prompt carries no solved-level facts)"
   ],
   "effort": "half-day",
   "expected_impact": "high",
   "status_in_franzen": "partial"
  },
  {
   "title": "Executable goal hypothesis: a retained level_goal(frame) function the harness scores and falsifies every turn",
   "category": "planning-exploration",
   "description": "Add one rule to the tool-session addendum (prompts.py COMPACT_TOOL_SESSION_ADDENDUM) and to LEVEL_START_USER_PROMPT: 'Encode your current goal hypothesis as a retained function level_goal(frame) -> float, where 1.0 means the goal is reached. Redefine it whenever your goal hypothesis changes.' In _build_user_prompt, if 'level_goal' is in self._kept_functions, the harness runs a tiny sandbox call on previous_frame and current_frame (the same sandbox used by _run_python_tool) and adds a line such as 'Your level_goal(): 0.40 -> 0.65'. If the score reaches 1.0 or more after an action and the level did not complete, the harness adds: 'level_goal() reports the goal reached but the level did not complete. Your goal hypothesis is wrong or missing a condition.' At level-up, add 'Re-check level_goal for this level'. Errors are reported once and then ignored. This builds on retained_functions.py and _retained_function_context (tool_agent.py:6461).",
   "mechanism": "The goal hypothesis now lives only in prose inside the model's thinking. It is never tested directly, so wrong goals survive through long deliberation; in ls20, several single requests spent 15-21k characters debating what the legend glyph meant. A code-level goal gets checked automatically, a wrong goal is falsified in one turn, and search code gets a ready-made heuristic. It uses the channel this model reliably writes: only 11 of 2,299 replies contained any visible content text, while ls20 kept 20 retained helper functions. Prose-structure arms failed for that reason (thinking guide 0/28 Predict lines, scratchpad 0 tags). The prompt already asks for 'a small scorer' (prompts.py:268), but nothing persists or checks it.",
   "evidence": [
    "https://arxiv.org/abs/2607.28287 (Tycho): executable world models; 'Correct mechanics, unresolved objective' failure class; no-world-model 79.07 vs orchestrated 88.49 RHAE with Opus 4.8",
    "computed from fz-arms-v4-control transcripts: content_chars>0 in 11/2,299 replies; ls20 opener lists 20 retained functions",
    "arc-agi-v4/arms.md 'Didn't work': thinking guide 0/28 Predict lines; scratchpad zero tags (prose-format compliance fails)",
    "arms.md Inconclusive: hypothesis-loop (next-state prediction param) inconclusive at 93% compliance; this differs because it targets the goal and is checked by the harness",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:268 ('write a small scorer') ; tool_agent.py:6461-6496 (retained function notice shows signatures only)"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Free control-mapping sweep at game start: the harness presses each key once on level 1",
   "category": "planning-exploration",
   "description": "Before the first analyzer turn of a game (action_num == 0, level 1), the harness runs each available non-MOUSE action once through the solver's step_env (UP, DOWN, LEFT, RIGHT and SPACE, whichever are valid). When UNDO is valid it sends UNDO after each probe to restore the state, and it aborts the sweep on game_over or level_completed. It computes frame_diff for each probe and puts a compact table in the first opener, e.g. 'Control sweep by the harness (4 actions): RIGHT -> 5x5 O/b block moved +5 cols; UP -> moved -5 rows; SPACE -> no change inside the board area'. It skips MOUSE-only games (7 of 25 public). A cheaper, prompt-only fallback: add a ~10-line exemplar snippet to the first opener (the action_num == 0 branch in _build_user_prompt) that probes each key in one call and prints a per-action frame_diff summary.",
   "mechanism": "The first actions of a game are the most expensive in tokens. In the control run the median was 6.8k generated tokens before the 5th action and 12.0k before the 10th, and 36% of turns that executed anything executed a single action. Level 1 carries the least weight, so extra actions there are almost free: replaying the control run with +4 or +8 actions on level 1 lowers the mean by only 0.13 or 0.23 (ledger run: 0.06 or 0.15). The sweep should save several model turns per game (~5-10k tokens) and prevent control-mapping mistakes early on.",
   "evidence": [
    "computed from fz-arms-v4-control/benchmark.json: median tokens to 5th action 6,758, to 10th 12,010; executing-turn batch sizes 36% singles (mean 5.2)",
    "re-scoring with taaf/game.py:_compute_final_score: +4/+8/+15 actions on level 1 -> mean 46.12/46.02/45.88 (control 46.25); ledger 52.50/52.41/52.18 (52.56)",
    "tufa-arc-agi-framework/src/taaf/game.py:381-413 (level weight = index+1, per-level cap 115, game capped at cleared weight fraction)",
    "fz-arms-v4-control transcripts: valid actions at start show 18/25 games have directional keys, 7/25 MOUSE-only",
    "https://docs.arcprize.org/methodology (actions = environment-changing inputs; per-level (human/AI)^2, cap 1.15x)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Per-request thinking cap with a forced tool call, instead of letting replies run to 12k tokens",
   "category": "model-config",
   "description": "Use a two-pass request in ToolAgent.analyze (truncation handling at tool_agent.py:9411). Send each request with max_tokens of about 6k (new env, e.g. ARC3_THINK_CAP_TOKENS=6144). If finish_reason == 'length' and there is no tool call, immediately resend the same messages plus the partial assistant message (reasoning_content = the truncated reasoning, closed with a short 'Time to act; writing the code now.') with SGLang's continue_final_message=true, so generation resumes after </think> and emits the python call. Keep the full reasoning in history and do NOT enable ARC3_STUB_DEAD_REASONING: transcripts show the model acts on the very next request by building on that reasoning. If the Pennyroyal template will not close the block, fall back to today's behavior with ARC3_REASONING_EFFORT_LADDER=medium. Also run a cheap global reasoning_effort=medium A/B: the Flash-Next card documents xhigh/medium/low, and the user's earlier 'medium does not shorten thinking' result predates this model. Avoid fixing SGLang's Qwen3ThinkingBudgetLogitProcessor: its token IDs are wrong for Qwen3.6+, and it still failed after correction (issue #25536).",
   "mechanism": "About 90% of generated characters are thinking (11.9M reasoning chars vs ~1.4M code chars in the control run), and the distribution has a heavy tail. Requests over 21k chars (~6k tokens) are 4.0% of requests but carry 6.1% of all reasoning as excess beyond 21k; over 14k chars, it is 8.8% of requests and 14.5% of reasoning. 25 replies (1.1%) hit the 12,288-token cap with no tool call, about 307k tokens or ~6.9% of all generation, each followed by a yield and a full re-opener. Capping turns that tail into an earlier probe whose observation is likely worth more than further speculation. Risk: the model card warns that lower effort can raise failures in agentic tasks, so run this as an arm.",
   "evidence": [
    "computed from fz-arms-v4-control transcripts: 2,299 requests; finish_reason=length 25 (all tool_call_count 0, 25.8k-41.2k reasoning chars); tail-excess figures above",
    "transcript s5i5: after a length truncation the turn yields on turn_token_budget and the next request immediately executes a 12-action batch (so reasoning should be kept, not stubbed)",
    "https://github.com/sgl-project/sglang/issues/25536 (thinking_budget / Qwen3ThinkingBudgetLogitProcessor not enforced for Qwen3.6; hard-coded token IDs wrong)",
    "https://mintlify.wiki/sgl-project/sglang/api/http/chat-completions (continue_final_message: 'Continue from the last assistant message'; reasoning_effort param)",
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next (reasoning_effort xhigh default, medium, low; warning that lower effort can increase retries in multi-turn agents)",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:1210 (ARC3_STUB_DEAD_REASONING, off), 1778 (ARC3_REASONING_EFFORT_LADDER, empty by default)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Turn on the level-start object inventory (ARC3_LEVEL_INVENTORY=1): a flag-only 'what is new on this level' list",
   "category": "perception",
   "description": "Set ARC3_LEVEL_INVENTORY=1. It is not in Franzen's notebook env, and with ARC3_DIFF_IMAGE already on, no other flag is needed. On the first opener of each new level, _level_transition_diff_lines (tool_agent.py:6073) then lists a START-vs-START comparison against the previous level within about 300 tokens. 'appeared' marks object types not present before ('likely new mechanics, investigate these first'), and 'moved' marks recurring types. This gives LEVEL_START_USER_PROMPT's instruction to 'inspect the new board for unfamiliar elements' a concrete list to work from. It can run as an env-only arm or as part of the debrief idea.",
   "mechanism": "The level-transfer prompt asks the model to find new elements, but leaves it to do so by eye or with ad-hoc code. Each new level costs about 4k generated tokens (median) before its first action, ~20% of cleared-level tokens. A harness-computed list of new object types should shorten that orientation and point the first probes at the new mechanic.",
   "evidence": [
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:5737-5779 (gate: inventory only when ARC3_LEVEL_INVENTORY), 6073-6146 (inventory text)",
    "CONFIGURATION.md \u00a73: ARC3_LEVEL_INVENTORY default Off; arc-agi-3-v3/external/dfranzen/nb.txt setup_env does not set it",
    "computed from fz-arms-v4-control/benchmark.json: median 3,853 generated tokens before the first action on a cleared level; 19.2% of cleared-level tokens",
    "prompts.py:45-58 (level-start prompt asks to inspect unfamiliar elements)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Level-weighted action-economy prompt: probe cheaply on early levels, stay thrifty on late ones",
   "category": "prompting",
   "description": "Replace 'Optimize for as few in-game actions as possible while still being reliable.' (prompts.py:122) and the WORLD_MODEL_FREE line (prompts.py:313-317) with the actual scoring facts and a per-level policy: 'Each level scores (human_actions/your_actions)^2, capped at 1.15x the human reference (a typical first-time player, who also explores). An uncleared level scores 0, and later levels weigh more. Clearing the level comes first. On early levels, a 1-3 action probe that separates two hypotheses is cheaper than long deliberation; on later levels, probe new elements but plan routes before executing them.' Add to the opener's state line: 'Actions on this level so far: k (level L of N; this level is L/(N(N+1)/2) of the game score)', with N from _priority_total_levels (tool_agent.py:8773).",
   "mechanism": "Cleared levels are usually already cheaper in actions than the human reference: 84% are at or above it, with a median human/agent ratio of 1.57 (88 of 105 levels). Thinking is ~90% of generation, and 28% of tokens go to levels that are never cleared, so the 'as few actions as possible' instruction pushes compute into deliberation. Blanket action laxity is not free, though: replaying with +20% actions on every cleared level costs -1.65 mean and +50% costs -5.1, while +8 actions on level 1 costs only -0.23. The policy therefore spends extra actions where the level weight is lowest. Risk: the user's old-harness 'spend actions' arm was mixed (mean 10.92 vs 13.65, trimmed 8.43 vs 6.34) and its scoring guide was negative (11.12 vs 17.74). This is a re-test in a much stronger regime with a level-weighted design, so measure the trimmed mean and levels cleared.",
   "evidence": [
    "computed from fz-arms-v4-control/benchmark.json: 105 cleared levels, median baseline/agent 1.57, 84% >=1.0, p25 1.06 (ledger run: 112, 1.55, 82%)",
    "re-scoring via taaf/game.py:_compute_final_score: all cleared levels x1.2 actions -> 44.60 (from 46.25), x1.5 -> 41.13; +8 on level 1 -> 46.02",
    "https://docs.arcprize.org/methodology ('upper median human' baseline; (human/AI)^2; cap 1.15x; level-index weights)",
    "arc-agi-v4/arms.md Didn't work: Spend-actions prompt c5cfbb0 and Scoring guide b4 (old harness)",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:122, 313-317"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Stop telling the model to ignore the HUD: read edge bars as the action budget and side panels as goal hints",
   "category": "prompting",
   "description": "In VISUAL_GAME_ADDENDUM (prompts.py:135-136), replace 'If you identify such a bar, do not get distracted by it...' and the 'A common failure mode... DON'T DO THIS!' paragraph with: 'Edge bars usually count remaining actions or lives: read their length to know your budget, and do not click them segment by segment. Panels separated from the play area (legends, rows of colored squares, a framed glyph, counters) often show the target configuration, the required order, or a key shape. Compare them with the play area when you infer the goal.' Ship it together with the newline fix in the same block (see the bug-fix idea).",
   "mechanism": "ARC-AGI-3 games often encode the goal in a HUD panel. The current wording primes the model to dismiss HUD content, which pushes goal inference into long speculative thinking. In sk48 (0 levels cleared) the bottom panel shows the player icon, then R, N, b: the collection order. In ls20, the legend glyph was the same shape as the room glyph, and the model spent several 15-21k-character requests reasoning about it. Siri deleted both lines. Deaths are rare in Franzen's control (no game over in 18 of 25 games), so the goal-hint half is the part expected to matter.",
   "evidence": [
    "daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:135-136",
    "arc-agi-3-v3/external/harness-feature-matrix.md C#1 (Siri removed both lines; reported 46% of actions in attempts ending in game over in their setup)",
    "fz-arms-v4-control transcripts: sk48 first-turn reasoning identifies bottom HUD 'M-square, R, N, b \u2014 the required order'; ls20 longest reasoning blocks 21,572/19,940/19,147 chars on legend vs room glyph",
    "computed: GAME OVER prompts in control = 0 for 18/25 games"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Turn on the two decisiveness lines that were built but never tested: ARC3_PREFER_TOOL_CALLS and ARC3_OPENER_COMMIT_HINT",
   "category": "prompting",
   "description": "An env-only arm: set ARC3_PREFER_TOOL_CALLS=1, which inserts PREFER_TOOL_CALLS_LINE (prompts.py:336-349: 'when you are unsure, prefer another tool call over more reasoning') into the tool-session rules. Also set ARC3_OPENER_COMMIT_HINT=1, which adds 'When your evidence supports a choice, call action(actions) with the sequence it supports... otherwise batch the route you have' as the last line of every fresh opener. Neither is in Franzen's notebook env, and neither appears in WRITEUP \u00a76's list of things tried. If combined with the action-economy idea, reconcile the wording, because PREFER_TOOL_CALLS frames actions as the only costly resource.",
   "mechanism": "Franzen's own docstrings record the motivation. Sequences after a resumption batched 6.73/6.91 actions, against 5.43/3.91 after a fresh opener. One turn spent 18,012 characters reasoning and executed nothing. In the control run, 36% of executing turns ran a single action, and the top 10% of requests held 41% of all reasoning characters. Both lines target that deliberation tail at zero code cost.",
   "evidence": [
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:2515-2533 (_opener_commit_hint docstring with batch-size measurements), 2586-2602 (_prefer_tool_calls docstring)",
    "CONFIGURATION.md \u00a72 (both default Off); arc-agi-3-v3/external/dfranzen/nb.txt setup_env (neither set)",
    "computed from fz-arms-v4-control transcripts: top 10% of requests hold 41.3% of reasoning chars; benchmark.json: 36% single-action executing turns"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Strip old board images only when history is trimmed (no cache cost), so more text history survives each trim",
   "category": "memory-context",
   "description": "In _persistent_history_messages (tool_agent.py:8713), and only on calls where a token trim or drain actually fired (the prefix is already invalidated at that point), replace images in retained messages older than the last K (e.g. 4) image-bearing messages with the existing '[grid image omitted from history]' placeholder via _strip_message_images (tool_agent.py:1282). Between trims, leave history byte-stable. Unlike ARC3_HISTORY_IMAGE_KEEP (tool_agent.py:1309), which rolls every turn and so changes an earlier message, forcing re-prefill, on every request, this adds no prefill. Keep the 58Ki drain target in tokens, so the space freed holds more turns of reasoning and tool output.",
   "mechanism": "Openers carry 1.6 images on average (2,509 images over 1,560 openers), at 402 estimated tokens each for a 640x640 image. That is ~645 tokens per turn, about an eighth of a turn's context. Removing old images at trim time keeps more turns of reasoning and tool evidence in the retained 58Ki and makes trims less frequent. The model can still inspect any old frame through Python (history[i].frame). The user's findings show per-game context mattered more than raw throughput: 14 streams with less context gave no gain.",
   "evidence": [
    "computed from fz-arms-v4-control transcripts: 2,509 '[image attached]' over 1,560 USER PROMPT sections",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:4213-4243 (ceil(w/32)*ceil(h/32)+2 = 402 tokens at 10x upscale), 1309-1331 (rolling image window), 8713-8771 (persistence pipeline)",
    "arc-agi-3-v3/reports/run-atlas/findings.md \u00a74 (streams14: +15% tok/s, no score gain; context per game matters)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Fresh-eyes reset for stalled levels at the next trim boundary",
   "category": "planning-exploration",
   "description": "When the current level has used more than about 60k generated tokens without completing (near the 62,000-token scale in the priority formula's C term), the next natural trim for that game drains deeper (e.g. to ~25k instead of 58Ki). The harness then puts a written block at the head of the retained history containing: the level's facts (actions, resets, deaths); a run-length-encoded list of the action sequences tried in each attempt; the solved-level debriefs (idea 1) and the ledger, if ported; and one sentence: 'The approaches above have not cleared this level after N tokens; your goal or mechanics model is probably wrong. Re-derive it from the board.' Fire it at most once or twice per level. The cache cost is ~0, because the trim already invalidates the prefix and is the scheduler's handover point.",
   "mechanism": "28.1% of generated tokens in the control run went to levels that were never cleared, and completion odds fall as tokens on a level grow (Franzen's C calibration). A context full of failed hypotheses anchors the model. A prompt-only audit did not help (audit60k 44.10 vs 46.25), but memory injected at the trim did (ledger-on-trim +6.3). So change what is in context rather than adding another instruction. Risk: Franzen found that larger context helped on difficult levels, so gate this tightly to genuinely stalled levels.",
   "evidence": [
    "computed from fz-arms-v4-control/benchmark.json: unsolved-level share of generated tokens 0.281 (ledger run 0.286)",
    "WRITEUP.md \u00a73 (C calibrated on 725 level attempts; token scale 62,000)",
    "arc-agi-v4/arms.md: audit60k 44.10 (no gain); ledger-on-trim 52.56 vs 46.25",
    "arc-agi-3-v3/final-c1/talk_by_tufa_lab.md ('cutting context does not appear to hurt much' for smaller models)"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Fix prompt and tool-doc consistency bugs, and skip tool-call-format work",
   "category": "bugfix-quickwin",
   "description": "(a) prompts.py:132: the 'Game entities are usually be rendered ... 1x1 tokens.' bullet lacks a trailing \\n, so it runs into '- Some games are logic or layout puzzles...' in every system prompt (visible verbatim in the transcripts). Add the newline and fix the grammar. (b) tool_agent.py:9569-9624: the no-tool-call follow-up nudge always advertises seven memory sections ('World model:', ..., 'Each section you write replaces the stored one entirely'). Under ARC3_MEMORY_SECTIONS=off, _memory_section_labels() returns None (tool_agent.py:2048-2073), so _advertised_section_labels() returns all seven (1870-1890), even though nothing is parsed or shown back. Gate those clauses on _memory_sections_disabled(). (c) tool_agent.py:7200: the code parameter is described as 'The snippet is ephemeral and is not saved across tool calls', which contradicts game-scope retained functions; and _PYTHON_TOOL_DESCRIPTION (tool_agent.py:420) omits frame_diff, last_animation_frames and last_animation_timeline. Align both with the system prompt. (d) Don't spend time on XML vs JSON tool-call format: it is not a problem with this model and parser.",
   "mechanism": "These fixes remove garbled or contradictory instructions the model reads on every request, at zero risk and almost no effort. Point (d) saves engineering time for the ideas that matter.",
   "evidence": [
    "fz-arms-v4-control transcript sk48 system prompt: '...Sometime they might also be 1x1 tokens.- Some games are logic...'",
    "daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:1870-1890, 2048-2073, 9569-9624, 7200, 420-431",
    "computed from fz-arms-v4-control transcripts: tool_call_markup_in_text=yes 0, tool_calls_recovered_from_markup=yes 0, malformed 0 across 2,299 requests; 2,273 single-call replies",
    "arc-agi-v4/arms.md Didn't work: xml-format arm 'no effect'"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  }
 ]
}
```
