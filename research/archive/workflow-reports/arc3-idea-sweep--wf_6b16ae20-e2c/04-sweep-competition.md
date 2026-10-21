# arc3-idea-sweep: sweep:competition

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "ideas": [
  {
   "title": "Solved-level memory: pin each cleared level's rule + winning action tail (port of Siriki M85)",
   "category": "memory-context",
   "description": "When the harness sees summary['level_transition'] (tool_agent.py ~5603-5640), record: the level number, the number of actions spent on it, its last <=30 executed actions (from history_entries), and the last ~900 chars of the model's reasoning from the turn that cleared it. Append one sentence to LEVEL_START_USER_PROMPT (prompts.py:45-58) asking the model to write a paragraph starting exactly with 'Rule for level N:' (goal, what each control did, the plan that won, at most 5 lines). Capture that paragraph from content OR reasoning_content, because most Qwen tool-call replies have no visible text. Ask on at most 2 turns, then mark it 'not stated'. Keep one 'SOLVED LEVELS' block: the newest 3 levels get rule + action tail + reasoning, older levels get the rule only, capped at about 1.5k tokens. Placement: Siriki appends it to the end of the system prompt. That breaks the prefix cache once per level-up, roughly one ~116k re-prefill (~10 s at Franzen's measured ~10.5k tok/s). A cheaper option is to put the block in the level-up opener and re-insert it as the first retained message whenever _trim_messages_for_context (tool_agent.py:8902) drains, since the prefix is invalidated at that point anyway.",
   "mechanism": "With the structured world model off (Franzen WRITEUP \u00a76) and drains down to ~57k tokens, the turns that established level N-1's mechanics get evicted during long levels. The model then re-derives them, spending actions and tokens. In Siriki's transcripts, cd82 L2 took 122 actions vs 8 and sb26 L4 took 189 without clearing vs 37. The score gap is mostly incomplete levels, and later levels carry more weight, so carrying a verified rule forward speeds up exactly the levels that matter most.",
   "evidence": [
    "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/main/WRITEUP.md - 'Memory of solved levels' moved LB 7.14 -> 9.96 (+2.82), one of the largest single gains in the 3rd-place M2 solution",
    "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/af105202327dcf35bd670a50b2558adab12e8dd3/cells/42_patch_cell_v16_8_m85_solved_level_memory.py - full M85 implementation (30-action tail, 900-char reasoning, 'Rule for level N:' capture from text or reasoning, 3 full levels)",
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - M85 docstring: the 'Cross-level notes' field was filled in 0 of 989 level>=2 prompts; lost levels re-derived the previous rule (cd82 L2 122 vs 8 actions)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734843 - 66.8% of Qwen tool-call replies carry reasoning with zero visible content, so capture must read reasoning_content too",
    "https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-100-trajectory-audit - on Franzen's own 100 trajectories, 50.88 points of the gap to 100 come from incomplete levels vs 2.63 from action inefficiency",
    "daniel-franzen-sol/WRITEUP.md:295 (world model disabled); ARC3-Inference/inference/agent/prompts.py:45-58 (level-up opener, no carried rule); tool_agent.py:8902 (trim)"
   ],
   "effort": "half-day",
   "expected_impact": "high",
   "status_in_franzen": "absent"
  },
  {
   "title": "Stall-triggered strategy/assumption audit (Lord Han Solo's STRATEGY_AUDIT_PROMPT)",
   "category": "prompting",
   "description": "Copy STRATEGY_AUDIT_PROMPT from Lord Han Solo's prompts.py. It tells the model to check whether recent observations still match its predictions. If they do, it states the evidence and continues. If not, it separates observed facts from assumptions about the goal and controls, picks one uncertain assumption, and designs one short discriminating experiment with predicted outcomes. Append it to the next opener only when a level stalls, at most once per interval, and reset at level-up. LHS fires it every 25% of the per-game runtime spent on one level (their solver.py:239-249). In Franzen, reuse the per-level counters the priority scheduler already keeps: fire when generated tokens on the level cross a multiple of ARC3_PRIORITY_TOKEN_SCALE (62k), or when actions on the level cross ~115. Gate it behind a new flag, e.g. ARC3_STRATEGY_AUDIT.",
   "mechanism": "Controlled experiments found that stuck runs usually fail because of a wrong goal hypothesis formed early and then defended. In 83% of the forks, the stuck run had already tried the decisive action. An always-on hint did not help Franzen (STEP_VERIFICATION_HINT), and extra text costs context. A one-shot audit triggered only on stall spends ~600 tokens exactly where games are burning budget. It also complements the scheduler, which only deprioritizes stalled games and never redirects them.",
   "evidence": [
    "https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py - STRATEGY_AUDIT_PROMPT text (2nd place M2, 23.84 LB)",
    "https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/framework/solver.py - lines 239-249/437/476: audit interval = 0.25 x runtime limit per level",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792 - comparison table: 'strategy-audit prompt added to the turn once a level has used 25% of the game's time'",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723 - 83% of stuck-vs-solved forks: the stuck run had already tried the decisive action type; failure is early goal judgement, defended",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060 - 'When several distinct hypotheses have failed... change the hypothesis class' line was part of the +28% hidden-set prompt patch",
    "daniel-franzen-sol/WRITEUP.md:303 (always-on stronger verification hint did not help); ARC3-Inference/CONFIGURATION.md \u00a711 (per-level action/token counters already tracked)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Add bbox/area/mask to segmentation nodes and replace bare KeyErrors with field hints",
   "category": "bugfix-quickwin",
   "description": "Franzen's segmentation nodes (utils/segmentation.py:327-338) expose id, color, hash, rotation, rotational_symmetry, pose_hash, shape_hash, pixels, boundary and children. There is no bbox, area, centroid, width or mask. Add `bbox` [r0,c0,r1,c1], `area` (alias of pixels), `height`, `width`, and `mask` (row strings of '#'/'.' when bbox <= ~256 cells, as in LHS). Return nodes as a dict subclass whose __missing__ names the right field, e.g. 'centroid -> use bbox midpoint' or 'size -> area', copying _NODE_FIELD_HINTS from the Wang/Ludvig segmentation in Siriki's M90 cell. The module source is spliced into the sandbox via inspect.getsource (python_tool_sandbox.py:774), so the class must live in segmentation.py. Check that frame_diff and the JSON paths still accept the subclass, and update the node description in STRUCTURED_RUNTIME_STATE_ADDENDUM (prompts.py ~187).",
   "mechanism": "The model routinely asks for node['bbox'] or node['area'], gets a bare KeyError, and loses a whole model call (thousands of reasoning tokens) to the retry. Tokens and wall-clock are the binding constraint, so removing these failures converts wasted calls into productive ones. Both other M2 winners expose bbox on nodes.",
   "evidence": [
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - M75 markdown: model regularly asks for bbox/area/centroid/width; '198 code-error recoveries in 25 games'",
    "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/af105202327dcf35bd670a50b2558adab12e8dd3/cells/40_patch_cell_v18_0_m90_wang_ludvig_harness_ported_on.py - Wang/Ludvig Node(dict) with __missing__ + NODE_FIELD_HINTS; port gave +3.44 LB in Siriki's writeup",
    "https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py - LHS nodes carry bbox, pixel_count, mask",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369 - 'Tokens are the real currency, not actions': every run is wall-clock bound",
    "ARC3-Inference/inference/utils/segmentation.py:327-338 (no bbox/area/mask); agent/python_tool_sandbox.py:774 (source splice)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Make the prompt truthful: the edge bar is the action budget; state the real scoring rule",
   "category": "prompting",
   "description": "Three text-only changes with net prompt length about unchanged. (1) Replace prompts.py:135 ('...do not get distracted by it or treat it as core gameplay state unless there is concrete evidence...') with LHS's wording: the edge bar is a per-level action budget; every executed action, including dead clicks and no-ops, draws it down; exhausting it ends the level in a game over. Today this line contradicts the flag-semantics line in PYTHON_ADDENDUM_TAIL ('count remaining budget into your plans'). (2) Replace prompts.py:122 ('Optimize for as few in-game actions as possible while still being reliable') with the actual rule: an uncleared level scores 0; later levels weigh more; the game score cannot exceed the weighted share of levels cleared; per-level score ~ (baseline/actions)^2, so 2x the actions gives 1/4 credit; every game is solvable. (3) Fix the tool schema text at tool_agent.py:7200 ('The snippet is ephemeral and is not saved across tool calls'), which contradicts the retained-functions instructions when ARC3_PERSISTENT_FUNCTIONS=1.",
   "mechanism": "The model is currently told both to ignore and to budget for the same bar, and it never learns that depth beats polish. Competitors found that removing misleading lines helps cheaply. Stating the true objective pushes the model to secure level completion before saving actions, and makes it treat probe actions as costly because dead clicks count.",
   "evidence": [
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792 - rellik13 (Siriki): 'Some harness changes that helped were simply to not mislead the agent... UNDO instead of ACTION7, level restarted instead of game over, removing lines that tell the agent to ignore the budget bar'",
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - v18.1gs changelog: 'two phrases deleted from the prompt (the edge bar is the action budget; do not tell the model to ignore it)'",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060 - adding the true scoring formula + pacing line: hidden 1.12 -> 1.43 (+28%, together with click hints)",
    "https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py - LHS GAME_OVERVIEW_ADDENDUM: 'Every game is solvable', level weighting, budget-bar semantics",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/718638 - host-confirmed: a frame-unchanged ACTION6 counts toward both RHAE actions and the level budget",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/728299 and https://docs.arcprize.org/methodology - exact scorer: min(115,100*(h/a)^2), level-index weights, cap = completed weight share",
    "ARC3-Inference/inference/agent/prompts.py:122,135; agent/tool_agent.py:7200"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Lean per-turn opener: move ~500 tokens of static reminders into the system prompt",
   "category": "memory-context",
   "description": "Every opener built by _build_user_prompt repeats ~10 static lines plus TOOL_CALL_FORMAT_GUIDANCE (tool_agent.py:7096-7103 and 7128-7156): 'Only tool: python. It receives...', 'Only letter-coded board views...', 'Keep tool output compact...', 'For the most recent change...', 'When ready, call action(...)...', the MOUSE line, and so on. That is ~1.7k chars, about 450-550 tokens per turn. With ARC3_YIELD_RESUME_PROMPT unset (= 'full') and yields every 2,048 generated tokens, the block is sent again on every resume, along with a re-attached ~400-token image (ARC3_RESUME_PROMPT_IMAGES defaults on). Move the static lines once into the system prompt. Reduce the per-turn text to the Wang/Ludvig 5-line message (executed count, executed actions, level status, state line, valid actions) plus the existing event-specific blocks (game over, guard stops, level-up). As a separate arm, also try ARC3_YIELD_RESUME_PROMPT=short with ARC3_RESUME_PROMPT_IMAGES=0. First measure the share of retained context taken by user-message boilerplate in a saved run.",
   "mechanism": "The retained window is the model's only memory, and repeated boilerplate crowds out its own recent reasoning and tool history. It also causes more frequent drains, and each drain forces a full re-prefill. Siriki found that adding instruction or summary blocks cost 2.8-4.3 LB points for exactly this reason. The Wang/Ludvig harness with the 5-line turn message was a +3.44 step for them.",
   "evidence": [
    "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/main/WRITEUP.md - 'Instruction bloat: Adding advice or summary blocks reduced scores by 2.8-4.26 points because old turns were trimmed to fit'; 'the agent's own recent turns, its only evidence' mattered most",
    "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/af105202327dcf35bd670a50b2558adab12e8dd3/cells/40_patch_cell_v18_0_m90_wang_ludvig_harness_ported_on.py - Wang/Ludvig _build_user_prompt = 5 dynamic lines; port +3.44 (9.96 -> 13.40)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133 - Tufa: models 'dump full boards into the context that dilutes their attention'; per-turn repetition was an upstream design choice",
    "ARC3-Inference/inference/agent/tool_agent.py:7096-7103, 7128-7156 (static lines per opener); daniel-franzen-sol/WRITEUP.md:299 (Franzen tested shorter resume prompts only: inconclusive)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "SGLang hierarchical cache on host RAM freed by an FP8 PLE table, then more streams or more retained history",
   "category": "serving",
   "description": "Franzen keeps the BF16 PLE table in host RAM (95.37 GB). Swap it for the FP8 PLE version (albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE, or RadixArk's FP8 PLE as Siriki used), which takes 47.68 GB and frees ~48 GB. Spend the freed RAM on Pennyroyal's host KV+Mamba tier with Siriki's flags: `--enable-hierarchical-cache --hicache-size 48 --hicache-write-policy write_through --hicache-io-backend kernel`. Then run two arms: (a) raise MAXREQ / ARC3_MAX_ACTIVE_STREAMS / CUDAGRAPH_MAXBS from 10 to 12-14, since evicted prefixes can be reloaded from host instead of re-prefilled; (b) keep 10 streams but drain less (ARC3_CONTEXT_DRAIN_TOKENS 58Ki -> ~40Ki, so ~76k retained after a trim, closer to LHS's 81.5k target). Also check whether Pennyroyal on SM120 supports an FP4 KV dtype, since one top competitor reports negligible NLL penalty and it would roughly double the 1.01M-token pool. Validate with the existing --enable-cache-report and decode tok/s logs. Avoid expert pruning as a memory source: Siriki reports it freed memory but degraded play.",
   "mechanism": "Franzen says scores were still climbing at the end of runs, so throughput and retained context are the binding constraints. He capped admission at 10 streams only to keep prefixes warm. LHS served 14 streams from a 1.417M-token KV pool with 1,135 peak decode tok/s, vs Franzen's 1.011M pool and 946 tok/s. Siriki's largest single jump (14.49 -> 22.53) came from freeing KV memory and spending it on longer history. Another competitor reports concurrency saturating around 16.",
   "evidence": [
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792 - table: Franzen KV pool 1,011,264 tok / 10 streams / 946 tok/s; LHS 1,417,100 / 14 / 1,135; Siriki FP8 PLE 47.68 GB + 48 GB hierarchical KV+Mamba cache, 16 streams",
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - exact Pennyroyal flags: --enable-hierarchical-cache --hicache-size 48 --hicache-write-policy write_through --hicache-io-backend kernel, --max-running-requests 16, --chunked-prefill-size 4096",
    "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/main/WRITEUP.md - FP8 KV + longer history: 14.49 -> 22.53; 'cache remained 91-97% full'; 'Model expert pruning (freed memory but degraded play)'",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792 - xz: 'very little NLL penalty to use nvfp4 kv'",
    "https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE - FP8-PLE variant Franzen names as the low-RAM alternative",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739938 - 'going beyond 16 concurrent requests hits a saturation point'",
    "daniel-franzen-sol/WRITEUP.md:21,31-60,79-81 (throughput still improving score; 10-stream admission to protect cache)"
   ],
   "effort": "1 day",
   "expected_impact": "high",
   "status_in_franzen": "absent"
  },
  {
   "title": "Stop rewriting and re-parsing the whole game history on every action and python call (Siriki M86)",
   "category": "bugfix-quickwin",
   "description": "write_runtime_state (solver.py:801-810) runs after every action (solver.py:1876). It serializes every past frame as nested int lists with json.dumps(indent=2) (runtime_state.py:281). Every python tool call then parses that file twice: once at tool_agent.py:7936, a load the code itself notes is unused, and again at 7957. It also rebuilds ascii payloads for all frames (_ascii_history_view_payload, tool_agent.py:4060) and re-ships them to the sandbox after every in-snippet action. Host time inside action handlers counts against the 30 s tool timeout (python_tool_sandbox.py:906). Fix: keep history in memory on the session; cache each entry's payload once (append-only); drop indent=2 or store rows as hex strings; delete the unused load; on an in-snippet refresh send only the new entries. Benchmarked in this session on a 1,000-entry history: 65 MB file, 0.14 s dump + 0.22 s load on a fast desktop CPU. Expect several times worse on Kaggle's CPU with GIL contention across ~110 game threads (ThreadPoolExecutor, solver.py:2144). Log time spent in load/write per call before and after.",
   "mechanism": "The cost grows with history length, so it peaks in long multi-level games, which is where the high-weight levels are. Snippets that loop over many single actions (BFS execution) can be killed by the 30 s timeout mid-plan, wasting the turn and leaving state confusing. Siriki saw tool-timeout kills appear as histories grew and fixed this pattern in M86.",
   "evidence": [
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - M86: 'GAME STATE WITHOUT AN O(HISTORY) REWRITE AND RE-PARSE ON EVERY ACTION'; ascii+grid payloads for every past frame cost 0.61 s; action() calls killed by the 30 s tool timeout rose from 0",
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - M111 stores frames compactly as hex-digit rows joined by '/'",
    "ARC3-Inference/inference/framework/solver.py:801-810,1876; agent/runtime_state.py:281 (indent=2 full rewrite); agent/tool_agent.py:7936 (unused load), 7957, 4060; agent/python_tool_sandbox.py:906 (host time counts against timeout)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Server watchdog + games pause instead of quitting on analyzer failures",
   "category": "bugfix-quickwin",
   "description": "Franzen's notebook checks the SGLang process only during startup (proc.poll in the boot loop). If the server dies mid-run (GPU peak is already 94.5 GB of ~96 GB), each game fails ARC3_MAX_ANALYZER_FAILURES=10 consecutive times (3 HTTP retries at 5 s, then 10 s backoff) and stops. The run then ends hours early with a partial score. Add a background watchdog: probe /v1/models; after 4 consecutive failures, kill the process, confirm the port is free, and relaunch with the same args; allow at most 2 restarts (Scott Le Grand's design). Make sure the restart path uses identical chunked-prefill and context settings; Siriki's restart failed when those differed. Set ARC3_MAX_ANALYZER_FAILURES=-1 and a long backoff so games wait out the ~9-minute reboot. The scheduler and tail fade will redistribute the remaining time.",
   "mechanism": "This is insurance against catastrophic draws. Several teams reported identical notebooks finishing in 2-4 h with depressed scores around Oct 1-2, and the private score is computed in that same run, so it is lost too. Restarting keeps 5+ hours of play alive in the worst case.",
   "evidence": [
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744995 - runs 'completed in 4 hours' and '2 hours' instead of ~9; cause unresolved (outage vs crash)",
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - v14.64 'server-alive watchdog': 'v1 lost the server mid-run and the harness spun'; restart path mis-set chunked prefill so 'the run spent 30 of its 45 minutes with no model server'",
    "https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922 - owned-server watchdog: checks /v1/models, restarts after four consecutive failures, at most two restarts",
    "https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-100-trajectory-audit - Franzen v3 run: observed GPU/RAM peaks 94.5/125.5 GB",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985 - private scores are computed at the original run; not rerun",
    "ARC3-Inference/CONFIGURATION.md \u00a713 (ARC3_MAX_ANALYZER_FAILURES default 10, negative = unlimited)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Click-target helper with dead-signature memory for MOUSE games",
   "category": "perception",
   "description": "When MOUSE is in valid_actions, add a sandbox helper `click_candidates(frame=None)`. It returns segmentation components excluding the 4-cell HUD border (ARC3_NOOP_GUARD_BORDER) and the largest background component, each with center (row, col), color, area and shape_hash. Rank them like the Reki heuristic: small, rare-colored, button-like shapes first. Tag each candidate tried_no_change=True if an earlier MOUSE on a component with the same (shape_hash, color) produced gameplay_changed=False on this level. That is a 'dead signature', which generalizes beyond Franzen's optional exact-state no-op guard. Optionally add one compact opener line listing the top ~8 untried candidates, as gedouluhui did. Document it in the system prompt as optional.",
   "mechanism": "In click games the model has to map an image to exact coordinates through text reasoning. It often clicks dead cells or HUD strips, and every dead click costs score and level budget. Pre-computed targets plus a record of object types that did nothing cut wasted actions and the tokens spent recomputing coordinates. Predicting which clicks change the frame was the core advantage of the 2025 preview winner.",
   "evidence": [
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743060 - deterministic connected-component click candidates kept in the submission that went 1.12 -> 1.43 hidden",
    "https://arcprize.org/blog/arc-prize-2026-milestone-1 - Milestone-1 2nd place (Reki): numpy click heuristic preferring small, rare-colored, button-like shapes + 'dead-signature' that stops clicking object types that never change anything",
    "https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings - StochasticGoose (1st, 2025 preview) predicts which actions change the frame; Blind Squirrel prunes actions that don't change state",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/718638 - dead clicks are counted as actions on the scored server",
    "ARC3-Inference/CONFIGURATION.md \u00a75 (ARC3_NOOP_REPEAT_GUARD is exact state/action only and off in the submission)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Durable action annotations: per-call `plan` stored in history + a `game_overs` ledger (LHS)",
   "category": "tools-repl",
   "description": "(1) Add an optional `plan` string parameter to the python tool schema (tool_agent.py:7194-7203). Store it on every HistoryEntry that call executes (runtime_state.HistoryEntry and its payload) and expose it as history[i].plan and transitions[i].plan. Document it with LHS's line: 'Once a turn leaves your context this is the only record of what you were testing then.' (2) Add `game_overs` for the current level: one object per death with .actions (the run since level start or the previous death, excluding the harness RESET), .action_count, and .frame for the newest. Clear it on level-up. Copy the wording from LHS's STRUCTURED_RUNTIME_STATE_ADDENDUM, including: 'A level on a fixed action budget ends every run at the same .action_count'.",
   "mechanism": "Franzen's blockwise drain discards about half the context at once, and the hypothesis behind each action is lost with it. A one-line plan per call survives inside Python state at almost no token cost and supports predict-then-check behavior. Equal action_count across deaths diagnoses budget deaths from data instead of from reading the bar in images, which is what Franzen's long BAR RULE game-over prompt is trying to achieve.",
   "evidence": [
    "https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py - history[i].plan and game_overs definitions (2nd place M2, 23.84 LB)",
    "https://vista-research.github.io/ - VISTA: 'Before executing each action, briefly state what you expect to see. Afterward, briefly state all visible changes' (100% public win rate with frontier models)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792 - LHS history compaction keeps plans/modules across trims",
    "ARC3-Inference/inference/agent/tool_agent.py:6961-6990 (game-over prompt relies on image/bar reading), 7194-7203 (schema has only `code`)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Keep in-between animation frames for the last ~8 animated actions, not just the last one",
   "category": "perception",
   "description": "Franzen exposes `last_animation_frames` and `last_animation_timeline` only for the most recent executed action that animated. Inside a batch, or after several actions, earlier animations are gone. Port Siriki's M111: on the host, keep the in-between frames from GameState.all_frames for the last 8 animated actions, at most 32 each, with consecutive duplicates removed and rows stored as hex strings. Expose them as `transitions[i].frames`, read-only views with the same API as current_frame, and add an `animation_frames` count per executed action to the action result. LHS also adds a cheap classifier (tween/path/blink with object hash/positions) that turns frames into one-line facts. Keep it lazy and out of the prompt: composite images and auto-inserted timelines did not help Franzen.",
   "mechanism": "Franzen found animation access to be the strongest perception gain, but batch execution throws most of it away. Siriki's engine replay found mid-animation-only evidence in 22 of 25 public games, and 4 of their 6 biggest time sinks were games whose decisive feedback is mid-animation.",
   "evidence": [
    "https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit - M111 docstring: 22 of 25 public games have actions whose in-between frames differ from both before and after; 'evidence destroyed by the final frame = 18.8% of available score'; sp80/tn36/s5i5/cd82 stalls",
    "https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py - per-history-entry .animation_frames and animation['type'] tween/path/blink",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369 - Taaf Anim: 13/25 public games return multi-frame responses; model rarely inspects the right animation unprompted",
    "daniel-franzen-sol/WRITEUP.md:181-185,301; ARC3-Inference/CONFIGURATION.md \u00a74 (only the last animated action is kept)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Submission hygiene: pin the Docker image, sanity-check each scored run, and screen changes on a validated 4-game panel",
   "category": "other",
   "description": "(1) Pin each submitted notebook version to its original environment. Kaggle's default image moved to Ubuntu 24.04 / Python 3.13 on Oct 1, and Franzen's offline wheelhouse plus the competition wheels are cp312. Add an assert on sys.version_info[:2]==(3,12) before installing. (2) Calibrate your fork: copies of Franzen's notebook scored mean 25.77, SD 3.93, and the same immutable version scored 27.80 and then 21.01. Your two Franzen-based scores (20.87, 22.88) sit about 0.7-1.3 SD below that mean. Diff your env/CFG block against the original before attributing anything to new features, and check each run's wall time and logs for early termination. (3) Screen changes offline with the 4-game core panel FT09/SP80/TN36/VC33 (cross-validated median held-out correlation 0.953, worst fold 0.573) over >=3 seeds. Only reject on large regressions, and use the bundled tools/significance.py paired comparison. (4) For the two final picks, choose complementary configs rather than near-duplicates.",
   "mechanism": "With an SD of about 4 points per run, a +3 change is invisible on one LB draw. Wasting a daily slot on a broken environment, or a short run, costs both the public and the private score of that run. This idea does not raise the mean; it protects the expected final score and keeps you from discarding real improvements as noise.",
   "evidence": [
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745062 - Franzen-copy LB distribution: mean 25.77, SD 3.93, median 26.19",
    "https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-100-trajectory-audit - same immutable Franzen version: 27.80 then 21.01",
    "https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-why-one-run-lies - FT09/SP80/TN36/VC33 panel, 0.953 median held-out correlation, worst 0.573; experiment ladder",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745654 - Oct 1 default image switched to Python 3.13; cp312 wheels; pin original environment",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985 - one run plays both public and private games; private scores are not rerun",
    "Your own submissions list (kaggle competitions submissions): 22.88 (Oct 4), 20.87 (Oct 3); ARC3-Inference/inference/tools/significance.py"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "partial"
  },
  {
   "title": "Draw 1-px cell grid lines on the 10x board image",
   "category": "perception",
   "description": "In vision_context.py, after nearest-neighbour upscaling by MULTIMODAL_UPSCALE=10, overwrite the last pixel row and column of each cell with a thin dark separator, optionally a slightly brighter line every 8 cells as a coordinate ruler. The image size stays the same, so the token cost stays the same (~400 tokens at 640x640). Leave the diff image and death images unchanged, or apply the same treatment consistently. A/B it on click-heavy games.",
   "mechanism": "Cell boundaries make it easier to count cells and to localize MOUSE coordinates from the image. Qwen is reported to have the right visual intuition but to be weak at inferring coordinates. Franzen's single largest image gain came from changing the rendering scale (4x -> 10x), so rendering details matter for this model.",
   "evidence": [
    "https://vista-research.github.io/ - VISTA renders 512x512 PNGs: 8x nearest-neighbour upscale 'with one-pixel grid lines between cells'",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/734369 - comment: Qwen gets the initial intuition from images but 'cannot accurately infer coordinates based on image'",
    "daniel-franzen-sol/WRITEUP.md:203 (4x -> 10x was the biggest image gain); ARC3-Inference/inference/agent/vision_context.py (no grid lines)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  }
 ]
}
```
