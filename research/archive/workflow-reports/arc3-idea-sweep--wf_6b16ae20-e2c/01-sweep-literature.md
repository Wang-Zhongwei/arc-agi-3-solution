# arc3-idea-sweep: sweep:literature

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Stagnation-triggered 'trouble' reflection prompt (Reflexion + Rodionov v1.5)",
   "category": "prompting",
   "description": "Add ARC3_TROUBLE_PROMPT_ACTIONS (default 50). Keep a per-level marker `_trouble_mark`, reset at level-up. In the game-over branch of ToolAgent._build_user_prompt (tool_agent.py ~6961), when actions_on_level - _trouble_mark >= N, append a one-off 'trouble' block and set the marker. Use Rodionov's exact prompts (sec 12.5) as the base: a level-1 variant ('this is a visual puzzle, you may be missing a simple visual cue') and a later-level variant ('what is missing from your model?'). Each asks the model to (1) analyze what it already tried, knowing every attempt restarts from the same deterministic state, (2) say what it may have missed, (3) state a new hypothesis about the goal, and (4) say what it will try next and why. Add two lines from his simplification prompts ('assume the mechanics are simpler than your current explanation; drop distinctions not forced by evidence'). Add one line asking for 2-3 competing goal hypotheses and the single cheapest action whose outcome would tell them apart. Also fire it without a death when the level has had no progress after X generated tokens (e.g. 40k) and at least 30 actions, using the counters the priority scheduler already tracks. The block goes in the opener at the end of history, so it costs no prefill. Log how often it triggers.",
   "mechanism": "Stuck levels burn actions, which are penalized quadratically, plus tokens the scheduler could give to other games. Trace studies show models lock onto a false rule and do not update on contradicting evidence. A reframe that fires only in stuck states costs a few hundred prompt tokens and leaves games that are going well untouched. Franzen's always-on prescriptive verification hint is a different trigger, and it did not help.",
   "evidence": [
    "https://arxiv.org/abs/2607.15439 \u2014 Rodionov component study, sec 8.1/12.5: the 'trouble' prompt fires on GAME_OVER after >50 new actions on a level. v1.5 went from 74.78 to 82.01 RHAE (GPT-5.5 xhigh), confounded with a faster client.",
    "https://arxiv.org/abs/2303.11366 \u2014 Reflexion: verbal self-reflection after failed trials gives +22% absolute on AlfWorld.",
    "https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis \u2014 Opus 4.7 'aggressively locks onto a false rule and never lets go'.",
    "https://arxiv.org/abs/2510.19788 \u2014 AutumnBench: reasoning models often fail to update when faced with contradictory evidence.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:303 \u2014 the always-on step-verification hint did not improve score, so this one is gated on stagnation.",
    "grep for stuck/trouble/stagnation in inference/agent/tool_agent.py, prompts.py and framework/solver.py finds no such trigger."
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Level-completion hindsight + pinned per-game level ledger that survives context drains",
   "category": "memory-context",
   "description": "(a) Prepend to LEVEL_START_USER_PROMPT (prompts.py:45) a request to write a short 'Level N notes:' block before planning: the win condition the completing action satisfied (compare `last_transition.before_frame` with the end of `last_animation_frames`), mechanics confirmed, things seen but never explained, and the game goal in one sentence. (b) Parse that block from the next assistant message with the existing tolerant `_extract_labeled_blocks`. Cap it at about 250 tokens per level and 1.5k in total. (c) Keep the ledger as one pinned user message directly after the system prompt, never in the system prompt itself, so the shared system-prompt Mamba checkpoint stays valid. Have `_drop_oldest_history_block` / `_trim_messages_for_context` (tool_agent.py:8564/8902) skip it. Insert or refresh it only at the next drain, where the prefix is rebuilt anyway, so it costs no extra prefill. Flag: ARC3_LEVEL_LEDGER.",
   "mechanism": "Level k carries weight k, so later levels dominate a game's score. The drain target of about 57k tokens discards the reasoning in which earlier levels were understood. Asking 'why did I win?' targets the documented failure mode 'solved the level, didn't learn the game'. This differs from what failed for Franzen. His structured world model was rewritten and re-inserted every turn and lost updates to parse failures. The ledger is written once per level and re-inserted only at drains.",
   "evidence": [
    "https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis \u2014 failure mode 3, 'Solved the Level, Didn't Learn the Game' (ka59, ar25): strong runs would interrogate why earlier levels succeeded.",
    "https://arxiv.org/abs/2607.28287 \u2014 Tycho: a consolidation pass writes level summaries at each level boundary, and the workspace persists across levels.",
    "https://arxiv.org/abs/2607.20709 \u2014 NOOA: world-model + memory skill 50.2% vs 38.4% for the same skill using markdown files instead of the memory subsystem (GPT-5.5). Memory includes reflection at level boundaries.",
    "https://gigazine.net/gsc_news/en/20260731-how-enabling-two-settings-tripled-arc-agi-3-benchmark/ \u2014 OpenAI: keeping reasoning plus compaction instead of rolling cuts took GPT-5.6 Sol from 13.3% to 38.3% (original post https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/).",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:295-297 \u2014 caveat: the per-turn structured WM and interval summaries showed no clear gain, which is why this design is written once per level and re-inserted only at drains.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:97 \u2014 drains trim to about 57 Ki retained context."
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Prediction-checked batches: action(..., expect=...) that stops at the first mismatch",
   "category": "tools-repl",
   "description": "Two tiers, behind flag ARC3_PREDICTION_CHECK. Tier A: `action(actions, expect='free text')` stores the prediction in last_action_call_result. The next opener shows it as 'You predicted: \u2026' directly above the observed change (diff image or auto frame-diff text), so a surprise is explicit. Tier B: each action item may carry `expect`, a predicted board given either as an ascii string with '?' for cells the model doesn't claim, or as a {'r,c': char} dict. In solver.py's batch loop, next to the BATCH_NOOP_BLOCK stop (solver.py:1626), compare only the claimed interior cells (outside ARC3_NOOP_GUARD_BORDER) against the new grid. On a mismatch, stop with stop_reason='prediction_mismatch' and return the first mismatched cells plus a small crop. Add a sandbox helper `checked(actions, predict)` that threads predicted grids through the model's retained predict(grid, action) and submits one call. That is one host round trip, whereas looping action([a]) reserializes the full history on every call. Gotcha: the helper must clear `_stale_state` when it predicted a no-op, or the stale-state guard fires.",
   "mechanism": "Every action counts toward the per-level score, which falls with the square of actions taken. Today a wrong plan runs to the end unless one of its actions is a pure no-op. Stopping at the first divergence turns a wrong long batch into one informative probe. This is the shared core of Schema's commit_actions, Rodionov's plan executor, Tycho's expected-frame hashes and NOOA's predict-in-rationale contract.",
   "evidence": [
    "https://schema-harness.github.io/ \u2014 commit_actions discards the remaining plan after any prediction error, one of 3 constraints that took Opus 4.8/Fable 5 from 42.83% (Claude Code) to 98.98% (best-of-2 fallback protocol).",
    "https://arxiv.org/abs/2605.05138 \u2014 Rodionov's plan executor compares the predicted settled frame after each step and stops immediately on divergence.",
    "https://arxiv.org/abs/2607.28287 \u2014 Tycho stores an expected-frame hash per step and surfaces the next action only while the observed frame matches.",
    "https://arxiv.org/abs/2607.20709 \u2014 NOOA: every turn ends with submit_actions(..., rationale='predict: ...'), checked on the next turn.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/framework/solver.py:1626 \u2014 the only in-batch stops today are no-op/terminal (BATCH_NOOP_BLOCK).",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/python_tool_sandbox.py:524 \u2014 action() closure and _stale_state handling to extend."
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "World-model kit in the sandbox: backtest(), bounded bfs(), partial-prediction contract",
   "category": "tools-repl",
   "description": "Add bootstrap helpers next to frame_diff (python_tool_sandbox.py:400). (1) `backtest(predict, level=None, border=4, show=3)` replays every recorded transition of the level from `transitions`, skipping automatic RESET and level changes. For each one it calls predict(before_grid, action), scores only the cells the model claims (None or '?' means abstain), and reports accuracy, coverage and the first mismatches as 5x5 crops. predict may also return (grid, 'win'|'dead'|None) so terminal predictions get checked. (2) `bfs(start, step, is_goal, actions, key=None, max_nodes=20000, time_limit=20)` with a visited set and a time cutoff, returning (path or None, stats), so it can never hit the 30 s kill. (3) `grid(frame)` returns a mutable list-of-lists copy. Describe them in about 150 tokens of system prompt as optional: 'when mechanics look deterministic, write predict() for just the parts you understand, backtest it, then bfs'. Pair with ARC3_PERSISTENT_FUNCTIONS_SCOPE=game so predict() persists across levels.",
   "mechanism": "This moves trial and error out of the scored game and into free search inside a checked model. BFS in the model yields shortest plans; Schema's M0R0 cleared a level in 42 actions where the human took 500. Partial predictions (abstaining on unknown cells) lower the bar for a small model, which can model only the avatar. Built-in bounded search saves the tokens spent re-writing BFS and avoids unbounded in-cell searches. Keep it optional: forced executable models and auto-triggered repair underperformed.",
   "evidence": [
    "https://arxiv.org/abs/2607.15439 \u2014 the replay-verification variant ranked first in all four model/effort settings (at higher cost). Textual beat the flexible-executable variant in both GPT-5.5 conditions, so do not force it.",
    "https://schema-harness.github.io/ \u2014 run_backtest checks rules against every recorded transition. In 14/25 games the exact-backtested model used 1.6\u20135.0x fewer actions than the human reference, and BFS expanded 10^3\u201310^4 modeled states.",
    "https://arxiv.org/abs/2607.28287 \u2014 Tycho's render may abstain on undetermined cells and verification reports claimed-cell accuracy plus coverage. The harness-triggered repair policy was the weakest model-using policy.",
    "https://arxiv.org/abs/2607.20709 \u2014 NOOA failure mode: two games hung in ad-hoc in-cell searches without max_depth, visited sets or node budgets.",
    "https://arxiv.org/abs/2505.10819 \u2014 PoE-World: world models built from many small programmatic rules, each encoding one causal law.",
    "https://arxiv.org/abs/2402.12275 \u2014 WorldCoder; https://arxiv.org/abs/2405.15383 \u2014 GIF-MCTS fixes code world models using trajectory tests as feedback.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:263-264 \u2014 the prompt already pushes BFS but provides no helper or verifier."
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Interaction-coverage ledger + state-graph frontier (OPINE ontology error / graph exploration)",
   "category": "planning-exploration",
   "description": "New module inference/agent/coverage.py, updated after each executed action. Hook it next to _record_guard_observations (tool_agent.py:7773), which already sees before/after grids and action signatures, and reuse _interior_state_hash (tool_agent.py:617) and segmentation. Keep two per-level tables. (1) Object type (color + shape hash) x action kind -> counts of effect classes (no change, moved, recolored, appeared, disappeared, death, level-up). For MOUSE the target is the segment containing (row, col); for keys and SPACE it is the set of object types that changed. (2) A directed graph: interior state hash -> {action: successor}. Expose read-only in the sandbox via the state payload, as death_ledger is: `coverage()` returns the compact table, `untested()` lists object types on the board never clicked or never changed and actions never tried from this state, and `path_to_untested()` gives the shortest known-action path to a state with untested actions. When the level is stalled (same trigger as the trouble prompt), the opener adds one line: 'Never-interacted object types: \u2026' (max 5).",
   "mechanism": "Catastrophic failures usually come from never testing the one interaction a level needs. A harness-kept record of what has not been tried is exact, costs no tokens to maintain, and survives context drains that wipe the model's own memory of what it tried. That lets the small model aim its few probe actions at real unknowns instead of re-testing familiar ones.",
   "evidence": [
    "https://arxiv.org/abs/2607.01531 \u2014 OPINE-World: 78.4 RHAE vs 58.2 for the ablation without its Bayesian object-action uncertainty guidance (Table 1). The largest gains came where the ablated agent left an interaction needed for progress untested (g50t, dc22, cn04, sc25).",
    "https://arxiv.org/abs/2512.24156 \u2014 graph-based exploration: tracking visited states and tested actions and heading for the nearest untested state-action pair solved 19 levels under a 4000-interaction cap vs 5 for LLM+DSL on the private games, ranking 3rd on the preview.",
    "https://schema-harness.github.io/ \u2014 DC22: Opus's BFS exhausted a model graph missing one edge, while Fable found the untested transition. LF52: a negative experiment rules out only the exact combination tested.",
    "https://arxiv.org/abs/2510.19788 \u2014 AutumnBench: models have 'a narrow view of what counts as informative actions'.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md:78,125-126 \u2014 LEVEL_INVENTORY and the no-op/death repeat guards track only new object types and refusal state-action pairs."
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Ranked click-target helper with HUD masking (graph-explorer frame processor)",
   "category": "perception",
   "description": "Sandbox function `click_targets(frame=None, k=20)` built on segmentation. It returns one representative interior cell per distinct object (the cell of the component nearest its centroid), grouped by shape hash and color, and sorts them into salience tiers in the style of Rudakov et al. Small-to-medium compact objects and enclosed children come first. Large background regions and thin strips touching the border (probable HUD or timer) come last. If the coverage ledger exists, each entry is annotated with 'already clicked: effect'. Add one prompt line advertising it for MOUSE games.",
   "mechanism": "Click games offer about 4096 actions per state. Models waste clicks on HUD bars, a failure mode Franzen's prompt has to shout about, and on background or pixels just off an object's edge. Ranked, deduplicated, guaranteed-on-object targets make each exploratory click more informative, which means fewer wasted actions on a quadratically penalized score.",
   "evidence": [
    "https://arxiv.org/abs/2512.24156 \u2014 frame processor: segmentation, status-bar masking and five priority tiers by size, morphology and color salience. Adding segmentation to random exploration alone added levels (5 on vc33, 2 on ft09).",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/prompts.py:136 \u2014 the prompt warns that mistaking a segmented edge bar for clickable pieces is a common failure ('DON'T DO THIS!').",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/python_tool_sandbox.py:400 \u2014 frame_diff and segmentation are the only perception built-ins today."
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Seed a curated helper library at game start (Voyager-style, mined from your own transcripts)",
   "category": "tools-repl",
   "description": "Offline: scan saved run transcripts and request logs from earlier Kaggle and local runs for functions the model kept retaining or re-defining, such as find player/cursor, crop region, track object across frames, path to target, and enumerate clickable objects. Curate 5\u201310 generic ones with no game-specific logic and unit-test them on frames from the 25 public games. Online: register them as pre-retained functions at game start through the retained_functions machinery, so they are restored into every snippet. Add one prompt line listing signatures and one-line docs, labeled 'library functions' to keep them distinct from 'your retained definitions'.",
   "mechanism": "The budget is roughly 609 tok/s x 540 min / 110 games, or about 180k generated tokens per game. Re-deriving the same helpers in every game is pure overhead and a source of bugs. Tokens saved feed the priority scheduler's redistribution, and tested helpers misread the board less often.",
   "evidence": [
    "https://arxiv.org/abs/2305.16291 \u2014 Voyager: a library of executable, reusable skills boosts performance and transfers to other agents.",
    "https://arxiv.org/abs/2605.05138 \u2014 Rodionov's future work calls for reusable routines (BFS, A*, constraint solving, subgoal decomposition) instead of reinventing them each run.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:20 \u2014 retaining model-written functions reduced unnecessary token generation.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md:110 + ARC3-Inference/Makefile:552 \u2014 609 tok/s harness-side generation and a 540-minute Kaggle experiment runtime, the basis of the per-game token estimate."
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "A/B the existing model RESET exposure, plus a guard so local runs match competition semantics",
   "category": "bugfix-quickwin",
   "description": "Run EXPOSE_RESET=on with ARC3_ACTION_INFO=1, which adds RESET_INFO_ADDENDUM, against your current config. Frame RESET as the way out of an unrecoverable position instead of burning the rest of the bar. Before trusting local A/Bs, add a guard in the RESET-first check (tool_agent.py:8071-8095) that refuses a model RESET when no action has been taken on the current level. Per the official API docs, such a RESET restarts the whole game; competition mode allows only level resets (ONLY_RESET_LEVELS), so offline runs and the real scorer would otherwise behave differently.",
   "mechanism": "A dead-end state otherwise costs every remaining bar action before the auto-reset, each of them counted against the level. One RESET action recovers that. Humans use resets far more than models, partly as deliberate experiments.",
   "evidence": [
    "https://arxiv.org/abs/2510.19788 \u2014 AutumnBench: humans use resets in 12.5% of actions, reasoning models in fewer than 7% (Claude 2.1%), and humans use them 'as experimental tools'.",
    "https://docs.arcprize.org/api-reference/commands/start-or-reset-game-instance \u2014 a RESET with no ACTION since the last level transition resets the entire game.",
    "https://stridenote.net/arcbasegame-explained/ \u2014 ARCBaseGame: with ONLY_RESET_LEVELS=true every reset is a level reset, the behavior competition mode requires.",
    "https://arxiv.org/abs/2605.05138 \u2014 sec 4: an agent-sent RESET before any action on a level triggered a full-game restart in local mode (s5i5, GPT-5.4).",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md:27 \u2014 EXPOSE_RESET defaults to off. The WRITEUP credits UNDO, not RESET."
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Programmatic memory of evicted context + per-action intent notes (PRO-LONG)",
   "category": "memory-context",
   "description": "When _drop_oldest_history_block / _trim_messages_for_context (tool_agent.py:8564/8902) drop messages, append their assistant text (reasoning, content, tool code) and tool outputs, tagged with level and step, to a per-game in-memory log. Accept `action(actions, note='why')` and store the note in each transition's result, so `transitions[i].result['note']` pairs intent with outcome. Expose `search_log(regex, k=5, width=300)` in the sandbox through a new host query message, using the same request/reply pattern as the animation fetch. Add one prompt line: 'reasoning evicted from context is searchable with search_log'.",
   "mechanism": "Each drain throws away about 70k tokens of reasoning. The frames survive in `history`, but the conclusions drawn from them and the reasons for past moves do not. A lossless, searchable archive gives back precise recall of earlier hypotheses and tests without growing the context or touching the prefix cache.",
   "evidence": [
    "https://arxiv.org/abs/2607.20064 \u2014 PRO-LONG: an append-all log (action, level, attempt, plan, result) searched with code. Gains are attributed to full-log access, and log parsing takes 20.3% of tool calls (0% for the no-log baseline).",
    "https://gigazine.net/gsc_news/en/20260731-how-enabling-two-settings-tripled-arc-agi-3-benchmark/ \u2014 OpenAI found that discarding reasoning and earlier actions explained much of GPT-5.6 Sol's low ARC-AGI-3 score (13.3% to 38.3% once fixed).",
    "https://stridenote.net/arcbasegame-explained/ \u2014 ActionInput has a 'reasoning' field echoed back with the frame, showing the engine expects intent to travel with actions.",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:8564 \u2014 the drain drops history blocks outright, with no archive."
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Actor-requested modeling subcall (Tycho 'orchestrator' policy)",
   "category": "planning-exploration",
   "description": "Add a sandbox function `ask_modeler(task)`. It triggers a host-side side request: a separate chat completion that is never appended to the main history. The context is a short modeling system prompt, the current level's transitions rendered as compact frame_diff outputs, a crop of the current board, and the current predict() source. The reply is Python for predict()/is_goal(), which the main agent can check with backtest() (kit idea) and then retain. Cap it at max_tokens of about 6k and at most 2\u20133 calls per level. Account for its tokens in the priority scheduler's C term.",
   "mechanism": "The model writes simulator code in a short, clean context instead of at the tail of a 100k-token rolling history, which suffers context rot and keeps debugging noise around. The main agent decides when, matching the best-performing allocation policy in a matched study. It costs one fresh prefill of about 15\u201320k tokens per call.",
   "evidence": [
    "https://arxiv.org/abs/2607.28287 \u2014 Tycho's matched comparison of four policies: actor-requested delegation ('orchestrator') scored highest, its interval vs no-world-model excludes zero, and the harness-triggered builder was weakest despite more exact models.",
    "https://arxiv.org/abs/2605.09998 \u2014 Continual Harness: scaffold gains scale with model capability and fall below a capability floor on Flash-Lite. A/B before committing.",
    "https://arxiv.org/abs/2605.09650 \u2014 DreamTeam: specialized roles (modeler, prober, planner) around a shared executable world model, 36% to 38.4% with 31% fewer actions."
   ],
   "effort": "2-3 days",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  }
 ]
}
```
