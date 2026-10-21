# m1-vs-m2-trajectory: research:ms1-duck

Workflow run `wf_903996f8-246`, phase Research. Research what changed from ARC-AGI-3 Milestone 1 (Tufa Duck) to Milestone 2 solutions, then verify the claims

---

## MS1 Duck harness, as shipped (Tufa Labs)

**Source keys.**
- **[WU]** is Tufa's Kaggle write-up, discussion/717133. I read it in Chrome on 2026-10-04. It has no line numbers.
- **[BLOG]** is https://tufalabs.ai/research/duck-harness/.
- **[APB]** is https://arcprize.org/blog/arc-prize-2026-milestone-1.
- **inf/** is `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/duck-harness/ARC3-Inference/inference/`.
- **nb** is `.../competitors/duck-harness/taaf-duck-harness-kaggle-share.ipynb`.
- The duck-harness repo HEAD is 7652836 and is clean (`git log` / `git status`). The commit is dated 2026-07-01 16:59 +0200, by M. Tesnar. It only edits the README, so it is the repo snapshot, not a code change.
- The shipped notebook attaches `jeroencottaar/taaf-kaggle-source-share` for the solver code (nb cell 6). I could not inspect that dataset. I did not diff the snapshot against it, so "identical to the 1.21 submission" is unverified.

### Score, placing, date
- **Placing:** 1st place in ARC Prize 2026 Milestone Prize #1, from the $37.5K milestone pool [APB]. The per-place split is unknown from these sources. The prize "ran through June 30th" [APB]. The other two winners (Reki, forge) were "pick a JSON action" agents on Gemma-4-31B [APB].
- **Kaggle LB:** best submission 1.21%. It first showed 1.30%, which Kaggle retracted. Scores for the same submission ranged as low as 0.77% [WU].
- **Public 25 games, 20 tries each:** mean 1.6002 +/- 0.4475, with a standard deviation of up to 0.4 across runs [WU][BLOG].
  - `example-run/summary.txt` (mean 1.60, median 0.07, 500 runs, 68,682 actions, 2026-06-02 11:23 to 17:33, 6h10m) matches.
  - Best game is ft09 at 10.28. tr87 scores 0.00 (same file).
- **Dates:** [BLOG] is dated 2026-07-01 (page meta). The [APB] publication date is unknown. The write-up page says "posted 3 months ago" relative to 2026-10-04.
- **Team (alphabetical, per [WU]):** Bessis, Cottaar, Pressman, Smit, Tesnar, Viel. The Duck is the successor to Dries Smit's Stochastic Goose [WU]. [WU] says it takes "heavy inspiration" from RGB Agent (OpenCode) and Symbolica ARCgentica.

### Architecture
- **Model and serving:**
  - Model is `vrfai/Qwen3.6-27B-FP8` (`inf/framework/kaggle.py:10-14`; `inf/framework/kaggle.py:12` for the model name).
  - One vLLM server (vllm 0.19.0, torch 2.10.0) with tensor-parallel size 1 on a single RTX Pro 6000. Flags: `--max-model-len 65536`, prefix caching on, `qwen3_coder` tool-call parser, `qwen3` reasoning parser, `preserve_thinking: true` (`inf/framework/kaggle.py:15,168-198`).
  - Sampling is thinking on, temperature 0.6, top_p 0.95, top_k 20 (`configs/inference.json:analyzer`).
  - Tufa itself says prefix caching is "not optimally" used [WU].
- **Agent loop:** `ToolAgent.analyze` (`inf/agent/tool_agent.py:1706+`) runs inside `HarnessSolver.play` (`inf/framework/solver.py:263-338`).
  - Each turn builds a prompt, calls the model, runs its single `python` tool call, and the turn ends once an `action(...)` has executed.
  - The solver then calls `analyze` again until the game ends, the time cap hits, or an action cap hits.
  - A response with no tool call gets a "You have not acted yet. Investigate first." nudge (`tool_agent.py:1907`).
- **Python REPL tool:**
  - The tool is a fresh subprocess every call (`README.md:87`; `inf/agent/python_tool_sandbox.py`).
  - Limits: 30 s timeout, output capped at 1024 tokens (about 4096 chars; `tool_agent.py:939`), CPU/file/fd rlimits, and an import allowlist of 14 stdlib modules (`prompts.py:81`, `python_tool_sandbox.py:245-264`).
  - Tufa warns the sandbox is "not fully isolated" [WU].
  - `action([...])` can batch actions or sit in a loop, and each call refreshes `current_frame`, `history`, `transitions`, `valid_actions` and `last_action_result` (`prompts.py:37-67`).
  - The raw numeric grid is hidden on purpose (`prompts.py:49`).
- **Actions:** UP, DOWN, LEFT, RIGHT, SPACE, MOUSE(row, col), RESET (`inf/agent/action_names.py:8-15`). UNDO is withheld because the model misuses it [WU]. A game-over triggers an automatic RESET (`solver.py:279-282`). The notebook sets `ONLY_RESET_LEVELS=true` (nb cell 2) and `bm.n_passes = 1` (nb cell 14).
- **World-model text sections:**
  - The model is told to write a revised world model before acting (`tool_agent.py:1229`).
  - Seven labeled blocks are parsed from assistant text: World model, Goal model, Action model, Recent findings, Open questions, Plan, Cross-level notes (`tool_agent.py:263-296`). They are re-injected into the next user message (`tool_agent.py:1128-1145`).
  - On a level transition, run complete or game over, all of them are wiped except Cross-level notes (`tool_agent.py:1113-1126`). [WU] describes only a single "World model:" note, so the code is more structured than the write-up.
- **Images and perception:**
  - Every turn's user message carries a 4x nearest-neighbour upscaled PNG of the current frame (`inf/agent/vision_context.py:52-82`, `configs/inference.json:79-82`). [WU] says 16-px Qwen patches make 4x the best choice.
  - Text views are a letter-coded `.ascii` grid and `.segmentation`: 4-connected components with a translation-invariant hash, boundary corners, children (containment) and an adjacency list (`prompts.py:41-44`, `inf/utils/segmentation.py`).
  - There is no frame stack or video, so the model never sees animation. [WU] names sb26 and tn36 as games where that hurts.
  - Prompt guidance covers HUD/timer bars (`prompts.py:28-29,95`), BFS/search for navigation (`prompts.py:86`), and "never print full boards" (`prompts.py:90`).
- **Memory and history:**
  - The system prompt is always kept. When the estimate exceeds the budget, the oldest messages are dropped (`tool_agent.py:1672-1690`).
  - The budget is `context_window` 32768 minus 512 reply reserve minus 512 safety margin, about 31.7k tokens (`tool_agent.py:944-948`; `configs/inference.json:5`). [WU] gives the same figures: maintain 32k, cap 64k.
  - The token estimate is a rough len(json)/3 (`tool_agent.py:462-467`).
  - History carried across `analyze()` calls is capped at the last 30 assistant turns (`tool_agent.py:151,1653-1670`).
  - Caveat: the example run's `git_info.txt` shows dev branch `user/periodic-chat-clearing` with the message "Make history turn cap configurable, default to unlimited". The run is labelled "0-history-turns", and I could not determine what that means. The history setting used for the 1.21 submission is unknown.
- **Guards:**
  - Tool-call markup found in text is recovered into a real call (`tool_agent.py:91-125`).
  - Syntax is checked before the code runs, and a server "context too long" error triggers a trim and retry (`tool_agent.py:1837`).
  - Batched `action()` stops early on level_completed, game_over or an invalid action (`solver.py:611,637`), and further actions after a terminal result are refused (`tool_agent.py:1518`).
  - `board_changed` and a "verify it wasn't HUD-only" warning are fed back each turn (`tool_agent.py:1070-1103`).
  - On Kaggle, `LOCAL_ANALYZER_YIELD_SECONDS` defaults to 60 and tool steps are unlimited (`kaggle.py:111,114`). A turn that runs past 60 s yields to the solver and the same analysis step is retried (`solver.py:312-322`).
- **Scheduling across games:** There is no priority scheduler.
  - Games run in list order through an asyncio semaphore (`solver.py:896-926`).
  - Kaggle settings are `make kaggle-duck`: 132 min per game, 540 min whole run, 1 pass, 28 concurrent games, 900 s analyzer timeout (`Makefile:400-405`).
  - All games share the one vLLM server. By my arithmetic (not stated in the source), 110 games / 28 is about 4 waves, so 4 x 132 min = 528 min, which fits the 540 min budget.
  - The Kaggle environment has 9 h of runtime and 110 games "in any order" [WU].

### Tufa's own account of why it worked [WU][APB]
1. Design principles are an open model that fits 96 GB, generic play with no game-specific encoding, and a lightweight harness that keeps "the model in the driver seat."
2. Code as the interface lets the model use its post-trained coding and reasoning skills. [BLOG] compares against Executable World Models: similar games solved, with the Duck "an order of magnitude cheaper" per game on GPT-5.4. That comparison is Tufa's own.
3. Main drivers were better base models and adding multimodality. Hand-crafted tools "did not help" and hindered creativity [WU]. [APB] repeats this.
4. Heavy prompt tuning was needed, and Tufa says so openly:
   - Don't treat the energy bar as the goal.
   - Don't hallucinate sprites or Atari-style games.
   - Don't dump whole boards.
5. Tufa's own limitations:
   - It has not been ablated over many design decisions.
   - Context compaction and curated memory are missing.
   - Perception is weak, since coding models don't reason over ASCII crops.
   - Variance is large: 0.77% to 1.21% on Kaggle for the same submission.

### Gaps for the MS1-to-MS2 comparison
- **Not found in MS1:** no priority scheduling, no speculative decoding or custom serving stack, no animation frames, no compaction or summarisation memory beyond the world-model note. I only checked these in the 7652836 snapshot, not in MS2 code.
- **Unknown:** the MS1 `taaf-kaggle-source-share` dataset contents versus this snapshot, and the exact history cap used in the 1.21 run.
