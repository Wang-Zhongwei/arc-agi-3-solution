# arc3-idea-sweep: sweep:scheduling

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Recalibrate the scheduler: drop the h=25 action penalty and halve the continuation tail B",
   "category": "scheduling",
   "description": "In the notebook env, set ARC3_PRIORITY_HUMAN_ACTIONS=300 (anything from 300 to 1000 works). This makes A = level x 55/W with almost no action discount. Then shrink B by about half. Code route (best): add an ARC3_PRIORITY_TAIL_SCALE multiplier on `tail` in priority_value() (inference/agent/priority_scheduler.py, around lines 240-252) and keep TAIL_LOOKUP=remaining with scale 0.5, which gives 4/3.5/2.5/0. Env-only route: unset ARC3_PRIORITY_TAIL_LOOKUP so B = min(TAIL_CAP=4, TAIL_BASE=2 + level - 1). Keep SCORE_NORMALIZATION=1, REFRESH_QUEUE=1 and TAIL_FADE=0.2, and leave TOKEN_SCALE at 62k (the sim gets worse at 40k). While editing, also fix parse_total_levels clipping N to [6,10] (line 84): for N<6 the last level still gets B=5, and for N>10 the tail drops to 0 too early. Validate with the replay simulator, not 25-game demo runs (see the 110-clone idea).",
   "mechanism": "The real per-level score is min(1.15,(b/a)^2) x level/W. The scheduler's A uses (25/(25+a))^2, which says 0.15 at 40 actions and 0.09 at 60. On 838 completed levels, measured efficiency was 1.05-1.10 in the 10-60 action range and 0.95 at 60-100. So A undervalues finishing deep, high-weight levels by 3-10x, and the flat B of 8 (with no token cost attached to reaching future levels) dominates the ranking. The scheduler effectively prefers starting level 1 of a fresh game (worth 1/W) over finishing level 5-7 of a deep game (worth 5-7/W) at about the same per-token completion rate. Measured completion hazard is about 0.38 per 20k tokens at every level index, with median 17-30k tokens per level.",
   "evidence": [
    "tufa-arc-agi-framework/src/taaf/game.py:381-413 - score = sum(level x min(115,(b/a)^2 x 100)) / sum(level), capped at the completed-weight fraction; matches https://docs.arcprize.org/methodology (per level (human/ai)^2 capped at 1.15, weights = 1-indexed level, median-human baseline)",
    "ARC3-Inference/inference/agent/priority_scheduler.py:224 efficiency=(h/(h+a))^2; tool_agent.py:3156 default h=25; notebook template.ipynb cell 4 sets TAIL_LOOKUP=remaining (B=8/7/5/0) and SCORE_NORMALIZATION=1",
    "Measured in the user's runs (~/arc-agi-3-v3/kaggle-outputs/fz-*/benchmark.json, 200 game-runs): 82% of 838 completed levels used <= baseline actions (median a/b=0.67). Efficiency costs only 45.7 vs 47.5 points. Mean efficiency by actions at clear: 1.10 (10-20), 1.05 (20-40), 1.07 (40-60), 0.95 (60-100), 0.76 (100-200), against the h=25 formula's 0.51/0.31/0.15/0.09/0.04",
    "User's replay simulator (~/arc-agi-3-v3/franzen/analysis/replay_scheduler.py + levels.csv), 110 games / 10 slots / 532 min, paired reps (scratch: scratchpad/sim/mysim4.py, mysim5.py). Franzen h25+remaining = 45.0. h1000 + remaining x0.5 = +5.3 (40/40 reps better). Env-only h1000 + base2/cap4 = +5.6 (40/40). h300 + base2/cap4 = +5.4. With 40% of stuck levels assumed unclearable: +2.1 to +2.9 (35-40/40). The user's 'fitted' index scored -0.1/-1.2 and stall-swap -0.2 to -1.6, so this is not just 'any change helps'",
    "Simulator caveats: level costs are drawn from public-game runs (local ~46 vs private ~28), and levels are assumed independent of scheduling. Expect roughly 60% of the sim gain on the private scale (about +1.5 to +3)"
   ],
   "effort": "<2h",
   "expected_impact": "high",
   "status_in_franzen": "partial"
  },
  {
   "title": "Expose RESET, guard against a free-looking zero-action RESET, and state that UNDO and RESET each cost one action",
   "category": "bugfix-quickwin",
   "description": "Set EXPOSE_RESET=on (it is off in Franzen's notebook; ACTION_INFO already adds the RESET line). Add a harness guard in solver.step_env / tool_agent around lines 8072-8085 that refuses RESET when zero actions have run since the last level start or reset. In competition mode that RESET is a server-side no-op that still counts as a scored action. Edit UNDO_INFO_ADDENDUM (prompts.py:156) to say 'UNDO counts as one action and does not refund the undone action'. Add to RESET_INFO_ADDENDUM: 'If the attempt is lost, RESET once instead of spending actions to drain the bar; to back out more than ~2 moves, RESET (1 action) plus replaying the good prefix is usually cheaper than chained UNDOs.' Locally, game_api sets ONLY_RESET_LEVELS=true, so test the guard by checking it fires; don't rely on the score.",
   "mechanism": "Every RESET (inc_reset_count) and every UNDO (ACTION7 in ids 1-7) increments the level's action count `a`, which enters (b/a)^2. Without RESET the model kills hopeless attempts by spending the bar: 10-30 counted actions plus several turns of tokens, and the batch no-op/stale guards make no-op 'burn' moves cost one turn each. Now a single action. The guard prevents a counted no-op. Full-reset replay is impossible in competition (one scorecard per game, ONLY_RESET_LEVELS, zero-count RESET swallowed), so no full-reset trick to recover from mistakes is available.",
   "evidence": [
    "Transcripts: 6-11 of 25 games per run contain explicit 'burn the bar / force a reset / deliberately die' plans (fz-arms-v4-control 11, fz-streams14 9, fz-arms-v5-temp06 7, fz-ledger-on-trim 6, fz-ucb 6). Example: ~/arc-agi-3-v3/kaggle-outputs/fz-arms-v5-temp06/transcripts/su15-1944f8ab_p0.txt:11801-11812 'burn the bar ... before burning ~14 actions'",
    "arc_agi 0.9.9 scorecard.py:701-703 inc_reset_count also increments actions, and update_scorecard (834+) counts ids 1-7 (UNDO included). arc_agi/api.py:325-334: in competition mode a RESET with _action_count==0 returns observation_space and re-runs update_scorecard, so it is counted with no effect. arcengine base_game.py:305-316 handle_reset. Version caveat: these are from the user's local venv; the gateway may differ",
    "CONFIGURATION.md section 1: EXPOSE_RESET default off, 'RESET counts as an action', first-action-of-snippet restriction already implemented. Notebook template cell 4 has no EXPOSE_RESET. WRITEUP section 6 does not list RESET as a failed experiment",
    "external/harness-feature-matrix.md: Siri allowed RESET and found 46% of actions went into attempts that ended in game over"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Two-stage thinking budget: early-stop runaway reasoning into an action instead of a 12k-token truncation",
   "category": "model-config",
   "description": "Today a reply that hits LOCAL_ANALYZER_MAX_OUTPUT=12k finishes with finish_reason=length, executes nothing, and triggers a 'you spent the turn reasoning' re-prompt. Implement Qwen's documented two-stage budget in tool_agent's request loop (near line 9411). Cap stage 1 at about 6-8k tokens. If the reply is still inside reasoning, send a continuation of the same assistant turn (raw /generate with the rendered prefix, or continue_final_message if Pennyroyal supports it) that appends Qwen's early-stop text and '</think>', so the model must emit the tool call. The prefix is cached, so stage 2 costs only the short decode. ARC3_REASONING_EFFORT_LADDER exists but is off, and the user's GB10 notes found that 'medium' effort does not shorten this model's thinking.",
   "mechanism": "Generated tokens are the binding resource: the user's simulator moves about +2 points per +5% tokens, and scores were still rising at the deadline. Truncated replies are about 1% of replies but 5-7% of all reasoning. Each costs about 10-12k tokens with no action, plus a follow-up re-think. Ending them at about 7k with a forced action reclaims about 3-5% of generation and turns dead turns into actions.",
   "evidence": [
    "fz-ledger-on-trim: 24 of 2414 replies finish_reason=length, 7.2% of reasoning chars, mean 34.5k chars each. fz-arms-v4-control: 25/2299, 7.3%. fz-streams14: 20/2735, 5.1%",
    "Successful replies: median reasoning 2.65k chars, p99 27k chars, so a 7-8k-token cap clips about 1% of good replies",
    "Qwen thinking-budget recipe (stage 2 inserts 'Considering the limited time by the user, I have to give the solution based on the thinking directly now.\\n</think>.\\n\\n'): https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html",
    "SGLang's server-side thinking_budget was reported as not enforced for Qwen3.6, which argues for doing this in the harness: https://github.com/sgl-project/sglang/issues/25536",
    "tool_agent.py:1778 _reasoning_effort_ladder (off by default); tool_agent.py:9411 truncation handling"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "KV-budget-aware admission: let 12-14 short-context games run, falling back to 10 as contexts grow",
   "category": "scheduling",
   "description": "Replace the fixed ARC3_MAX_ACTIVE_STREAMS=10 with a gate (tool_agent.py _PriorityGate, around line 2761) that admits another game while the sum of the active games' estimated contexts plus a reply reserve stays under the KV pool (~0.95M of the ~1.01M FP8 tokens). Cap it at a new ARC3_MAX_ACTIVE_STREAMS_HARD=14, and launch SGLang with MAXREQ=14, CUDAGRAPH_MAXBS=14 and MAMBA_CACHE=84. The harness already has a calibrated per-game token estimate (_estimate_request_input_tokens), so the gate can read it. Keep the 128k harness window for every game, unlike the streams14 arm, which cut it to 96k for all games.",
   "mechanism": "Decode throughput rises with batch size (611 tok/s at 10 streams, 644 at 12, 703 at 14 in the user's runs). Context size only binds when most active games are near 116k. With 110 games, the first-pass waves (each game runs about 15 min before its first trim; 11 waves) cover roughly the first 30% of the run, and in them average context is about half the window. More streams fit there with no evictions, giving about +4-5% total tokens with no loss of history depth.",
   "evidence": [
    "User runs: fz-arms-v4-control 611 tok/s (10 streams), fz-ledger-streams12 644, fz-streams14 703. streams14 also cut context to 96k/drain 42k for everyone and scored 46.38 vs 46.25, a null result that mixes the throughput and context effects",
    "~/arc-agi-3-v3/franzen/kaggle/ARMS.md: FP8 KV pool about 1.01M tokens; 10 streams x mean ~87k = 0.87M",
    "First action of second-wave games came at about 890-1060 s in 121-min runs (transcripts' run_elapsed_seconds), so first-pass slices are about 15 min",
    "Simulator: +5% tokens gave +1.97 points (44.9 to 46.8), an optimistic upper bound",
    "SGLang HiCache (host KV tier) for hybrid Mamba models was listed as future work as of Dec 2025, so don't rely on hicache for this: https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/"
   ],
   "effort": "1 day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Endgame 'bank it' mode: in the final ~10-15% of runtime, favor fast action-heavy attempts over long deliberation",
   "category": "scheduling",
   "description": "When time_remaining_seconds (already in every step payload) drops below about 0.12 x the run budget, add a short line to the next user prompt (and resume prompts). Suggested text: 'Under N minutes remain. An unfinished level scores 0. A finished one scores even if inefficient, and deaths only cost actions. Commit to your best hypothesis, try the top candidate solutions in batches, and keep reasoning short.' At the same time, switch off the batch no-op and stale-state guards (BATCH_NOOP_BLOCK/STALE_STATE_BLOCK) and lower LOCAL_ANALYZER_YIELD_TOKENS so turns end in actions before the hard cutoff. Tail fade (B fading over the last 20%) already changes which game gets a slot, but nothing tells the agent the deadline is near.",
   "mechanism": "Near the deadline the score from the current level is 0 unless it completes. A completion at 3x baseline actions still banks 0.11 x level/W x 100, and on deep levels that is large. Actions take milliseconds while deliberation costs about 850 tokens per action, so at the end it is worth trading efficiency for completion probability. The last ~10% of runtime is about 10% of all compute, which is otherwise mostly spent deliberating on levels that won't finish.",
   "evidence": [
    "Scoring: per-level min(1.15,(b/a)^2) and 0 if unfinished (game.py:381-413; https://docs.arcprize.org/methodology)",
    "Transcripts show 'time_remaining_seconds' in each action result (e.g. fz-ledger-on-trim/transcripts/ar25-0c556536_p0.txt:267), but no prompt uses it",
    "About 30% of all generated tokens (9.7M of 32.1M across 7 runs) went to the level still unsolved at run end",
    "Tail fade only changes scheduling: CONFIGURATION.md section 11, tool_agent.py _snapshot_priority"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Action-for-token trade: tell the model actions are cheap relative to long deliberation, especially on level 1",
   "category": "prompting",
   "description": "Turn on ARC3_PREFER_TOOL_CALLS=1 (prompts.py:336, currently off) and add one system line. Suggested text: 'Your score per level depends on (human_actions/your_actions)^2, but levels you never finish score 0, later levels are worth more (weight = level number), and thinking time is your scarcest resource. When two hypotheses compete, run the 1-3 action experiment that separates them rather than reasoning further. On level 1, the least valuable level, test each control and object freely.' Optionally also enable ARC3_OPENER_COMMIT_HINT. Measure tokens per completed level and levels completed against the same-day control.",
   "mechanism": "Efficiency has a lot of slack: completed levels typically finish under baseline, so extra exploratory actions on most levels cost nothing until a/b > 1. Meanwhile tokens are binding: 30% of tokens are spent on final levels that never finish, and completion per token decays. Moving effort from speculative reasoning to cheap experiments raises level completions per token, and level-1 actions carry only 1/W of a game's weight.",
   "evidence": [
    "82% of 838 completed levels finished with a <= baseline (median a/b 0.67). Efficiency costs only about 1.8 of 47.5 points (fz-* runs, 200 game-runs)",
    "Level weights: level 1 is 1/21 to 1/55 of a game (game.py:397 weight=level_idx+1)",
    "Repo prompts never mention scoring or efficiency (grep of prompts.py/tool_agent.py). PREFER_TOOL_CALLS and OPENER_COMMIT_HINT are present but off (CONFIGURATION.md section 2), and WRITEUP section 6 does not report them as tested",
    "external/harness-feature-matrix.md: Hansolo ships a scoring explanation plus 'full certainty is not required'; Wang's spend-actions arm had its batch's best trimmed mean. The user's GB10 scoring-guide arm was null on a different harness/model, so the effect is uncertain"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Harness auto-probe at game start: run each simple action once on level 1 and hand the model a diff table",
   "category": "tools-repl",
   "description": "In solver._play_inner, before the first analyzer turn (after any warmup), execute each available simple action once (UP/DOWN/LEFT/RIGHT/SPACE; skip MOUSE and UNDO; at most 5-6 actions). Record frame_diff for each and put a compact table in the first user prompt: 'Probe results: UP -> blue 2x2 moved (-1,0); SPACE -> no gameplay change; ...'. Stop early on game_over or level_completed. Gate it behind a new ARC3_AUTO_PROBE flag and apply it to level 1 only.",
   "mechanism": "These are level-1 actions: weight 1/W, and usually under baseline anyway, so the score cost is about 0. They replace the model's first few probe turns, which cost thousands of tokens each (level 1 median is about 17k tokens to clear). The saved tokens go to later, higher-weight levels.",
   "evidence": [
    "Median tokens to complete level 1 = 16.7k (mean 24.8k) across 165 completions in 7 fz runs",
    "frame_diff helper already exists (inference/utils/frame_diff.py), and step_env batches actions with guards (solver.py ~1480-1640)",
    "Risk: irreversible state changes on level 1; the history keeps every frame, and level 1 is the cheapest level to get wrong"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Trim-synchronized image stripping: drop old images only when a trim already invalidates the prefix",
   "category": "memory-context",
   "description": "Hook into the drain/trim path (tool_agent _drop_oldest_history_block / hysteresis drain, where _note_history_evicted fires). When a trim happens, strip images from every retained message except the newest K (about 4-6) image-bearing messages, and leave those images untouched until the next trim. Don't use the existing ARC3_HISTORY_IMAGE_KEEP: it filters every request, so each new image moves the cut-off point mid-prefix and forces a large re-prefill.",
   "mechanism": "Images (402 tokens each at 10x upscale; board plus diff per turn) make up 10-20% of prompt tokens. Removing old ones at trim time lets about 15% more text history fit in the retained 58k block and spaces trims further apart, which means fewer re-prefills and fewer forced handovers. It costs no cache misses because the prefix is already broken at a trim.",
   "evidence": [
    "Transcript calibration lines: 'prompt_tokens 60329, images 6030' and 'prompt_tokens 56011, images 11256' (fz-ledger-on-trim/transcripts/ar25-0c556536_p0.txt:13000, 21934)",
    "CONFIGURATION.md section 4: 'Filtering historical images changes request prefixes and can affect prefix reuse'; tool_agent.py:1227 _history_image_keep",
    "WRITEUP section 2: prefill was 15.25% of wall time; trims are the main source of re-prefill"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Anchor the per-game deadline to notebook start and shave server startup",
   "category": "bugfix-quickwin",
   "description": "In the submission branch (template cell 16), replace the fixed max_runtime_s_per_game = 532*60 with max(0, NOTEBOOK_START_TIME + LIMIT - TEARDOWN_MARGIN - now). Set LIMIT from the competition's actual runtime limit, and TEARDOWN_MARGIN to about 5 min. Then cut the 8-14 minutes spent before gameplay starts: start model-shard precaching in parallel with the wheel install (cell 6 currently delays it 60 s and uses 3 threads), drop '--warmups structured_output', and reuse a cached flashinfer autotune (or set AUTOTUNE=False if the cache is missing).",
   "mechanism": "Every minute of gameplay adds compute. In the user's simulator, +2% minutes gave +0.8 points (an upper bound). The game clock starts only after the server is healthy, so a slow load either makes the notebook overrun the limit or wastes slack when the load is fast. Anchoring the deadline makes the run safe and uses every available minute.",
   "evidence": [
    "User runs: 'Starting benchmark' at 528-702 s after notebook start, server 'READY after' 490-820 s (fz-*/fz-*.log)",
    "Template cell 16: TRUE_SUBMISSION sets max_runtime_s_per_game=532*60 after a server wait of up to 12 min (cell 12, SERVER_STARTUP_TIMEOUT=12*60). Interactive runs assert a 32,400 s notebook budget",
    "Confirm the actual rerun time limit on the competition page before changing LIMIT; a timeout scores 0"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Test scheduling and allocation changes at 110-game scale (replay sim, then one 110-clone competition-mode run), not on 25-game demos",
   "category": "other",
   "description": "Before spending Kaggle GPU, sweep every ARC3_PRIORITY_* knob in the user's replay simulator (the scratch copy runs about 40 paired reps in under a minute). Then confirm the winner once with TAAF's CompetitionArcadeServer.official_110(). It exposes 110 cloned public games in competition mode with hidden baselines, so the gate has about 100 waiting candidates, as in the real rerun. For a cheaper check, clone 50 games for about 242 min to keep compute per game equal. Score with `make significance` using paired per-game tests.",
   "mechanism": "In a 25-game demo only about 15 games wait at each handover, against about 100 in the real run. Priority differences that matter at 110 games are therefore mostly invisible in demos, and per-game swings of 50 points are normal. Franzen's 'elaborate priority rules didn't clearly help' and the user's ucb/fitted arms were judged in this setting. Testing at the right scale avoids false negatives on the highest-leverage, lowest-effort knobs.",
   "evidence": [
    "tufa-arc-agi-framework/src/taaf/competition_arcade.py: official_110() and clone_game_ids() (110 cloned IDs, competition_mode=True)",
    "~/arc-agi-3-v3/franzen/analysis/replay_scheduler.py docstring: 'scheduling matters ~4x more there than our offline scores can show'",
    "WRITEUP section 6 ('More elaborate priority rules ... did not establish a clear advantage') was based on demo runs. ARMS.md: 'Per-game swings of 50 points between identical runs are normal'",
    "Scratch sim scripts: scratchpad/sim/mysim4.py, mysim5.py (patched copies; repo untouched)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  }
 ]
}
```
