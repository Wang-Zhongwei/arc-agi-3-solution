# arc3-idea-sweep: sweep:competitors

Workflow run `wf_6b16ae20-e2c`, phase Sweep. Research sweep for ARC-AGI-3 harness improvement ideas: multi-angle search, dedup, adversarial feasibility check, ranked synthesis

---

```json
{
 "repos": [
  {
   "name": "Tufalabs/duck-harness",
   "url": "https://github.com/Tufalabs/duck-harness",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/duck-harness",
   "summary": "Base Duck harness (Milestone 1, single July 1 snapshot, 7652836). Tufa Labs now leads the Kaggle LB at 55.89 (Yi-Chia Chen 48.59, third 37.54) and has said it will not open-source before the competition ends; no top-10 team has a public ARC-AGI-3 notebook (checked)."
  },
  {
   "name": "jpezzulli/sglang-rtxpro6000 (Pennyroyal)",
   "url": "https://github.com/jpezzulli/sglang-rtxpro6000",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-rtxpro6000",
   "summary": "SGLang fork Franzen builds on. Latest release is v2.5.3 (Sep 28), the version he uses; later commits (to Oct 3) are docs and reporting only. No newer serving code to pull."
  },
  {
   "name": "mratsim/sglang-qwen38fn-sm120-turbo",
   "url": "https://github.com/mratsim/sglang-qwen38fn-sm120-turbo",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-qwen38fn-sm120-turbo",
   "summary": "Docker/SGLang stack (r24). Reports 800 tok/s at 4 streams and 1170 tok/s at 8 streams on RadixArk NVFP4 plus online MXFP8 (the MXFP8 path the user already measured as slower). Franzen already uses its recover-ssm budget idea; the abort-ghost and single-GPU token-sync patches are minor. The QAD W4A16_NVFP4 checkpoint needs TP=2."
  },
  {
   "name": "gabrielolympie/sglang-flashnext-sm120",
   "url": "https://github.com/gabrielolympie/sglang-flashnext-sm120",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120",
   "summary": "Source of Franzen's low-M GEMM and FR-Spec map. Reports 231 tok/s single stream (jpezzulli 171). Three further levers Franzen does not use: relaxed MTP acceptance 0.3 (179 to 231 tok/s), W8A16 fp8 weight-only dense stack, and fp8 lm_head/HyperConnection."
  },
  {
   "name": "LohitSiriki/arc-agi-3-milestone2-solution (rellik13) + Kaggle sirikilohit/arc-agi-3-duck-18-1gc-submit",
   "url": "https://github.com/LohitSiriki/arc-agi-3-milestone2-solution",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/rellik13-milestone2",
   "summary": "Milestone 2 #3, LB 22.53 (progression 0.99 to 22.53 in documented steps). Duck plus Wang sandbox on SGLang. Biggest jump (14.49 to 22.53) came from FP8 KV and a longer retained history with no prompt changes. Also: solved-level memory pinned in the system prompt (+2.8), an O(history) state-I/O fix (cell 38), UCB scheduler, exact /v1/tokenize guard. Includes analysis/sawtooth.py. The Kaggle notebook copy is in competitors/kaggle/arc-agi-3-duck-18-1gc-submit."
  },
  {
   "name": "lordhansolo Milestone 2 (Kaggle notebook + public source dataset taaf-kaggle-source)",
   "url": "https://www.kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2",
   "kind": "kaggle-notebook",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source",
   "summary": "Milestone 2 #2, LB 23.84, vLLM NVFP4. 1,745-line tool_agent rewrite with in-memory runtime state, saved modules with regression tests, a python `plan` parameter, game_overs object, changed-region action results, animation readings (tween/path/blink), strategy audit. Notebook is in competitors/kaggle/built-on-tufa-labs-duck-harness-milestone-2. Note: the 2.3 MB source dataset was fetched with `kaggle datasets download` (notebook pulls only fetch the launcher)."
  },
  {
   "name": "tonghuikang/daniel-franzen-arc-agi-3 (Milestone 2 comparison)",
   "url": "https://github.com/tonghuikang/daniel-franzen-arc-agi-3",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/tong-m2-comparison",
   "summary": "Line-linked side-by-side of Franzen, lordhansolo and rellik13 (serving, scheduling, context, prompts); kaggle/SUMMARY.md is the fastest way to see where each differs from Franzen. Peak decode: Franzen 946, lordhansolo 1135, rellik13 1159 tok/s."
  },
  {
   "name": "scottlegrand/taaf-flashnext-sheetu12b-0922 (agentfix)",
   "url": "https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922",
   "kind": "kaggle-notebook",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/taaf-flashnext-sheetu12b-0922",
   "summary": "Duck plus six runtime 'agentfix' patches (HUD-band detection, per-action effect ledger with UNTESTED actions, loop guards, image keep-last-2, stall triage). Author reports only about 5 on the LB from harness fixes alone; several fixes ship default-off after his own regression hunt. Idea source, not a score source."
  },
  {
   "name": "desihnurseha/kinetic-learning-by-playing-v8",
   "url": "https://www.kaggle.com/code/desihnurseha/kinetic-learning-by-playing-v8",
   "kind": "kaggle-notebook",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/kinetic-learning-by-playing-v8",
   "summary": "Franzen fork plus a patch with add-ons ported from other M2 solutions: `memo` dict persisted across python calls, a knowledge pack and mechanics catalog, game-over memory, stuck-turn audit, solved-level action tails. Public LB 27.28, which is inside Franzen-copy noise (mean 25.77, SD 3.93)."
  },
  {
   "name": "skarin 27.80 LB trajectory audit + 'Why One Run Lies' (Kaggle)",
   "url": "https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-100-trajectory-audit",
   "kind": "kaggle-notebook",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/arc-agi-3-27-80-lb-100-trajectory-audit",
   "summary": "Unchanged Franzen reproduction (27.80, then 21.01 on rerun) plus an audit of his 100 public-game trajectories: 50.88 points lost to incomplete levels versus 2.63 to action inefficiency; six weak games use about 18% of tokens for 14% of levels with 0 wins. Companion notebook (why-one-run-lies) gives a validation protocol and a 4-game rejection panel."
  },
  {
   "name": "shiiin9 duck-plus-ours / affectify-arc-dfranzen-form-d1 (Osaka Metropolitan Univ.)",
   "url": "https://www.kaggle.com/code/shiiin9/affectify-arc-dfranzen-form-d1",
   "kind": "kaggle-notebook",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/affectify-arc-dfranzen-form-d1",
   "summary": "Franzen with only the slot-priority formula replaced (pace term, annealed exponent): one LB submission at 31.54 versus 27.62-31.47 for unchanged Franzen, so within noise. Consistent with Franzen's own finding that pace tweaks give no clear gain. Lottery scheduler variant in competitors/kaggle/duck-plus-ours (Duck-era, ~5.7)."
  },
  {
   "name": "Franzen near-copies: bang1850 sovereign (best 29.08), matterhorn3838 ak47, amatlas dfranzen-fork, keithtyser/wuliao0/chiakazirim vLLM Duck, cihanatak REAP-448, richardcsaky (20.00)",
   "url": "https://www.kaggle.com/code/bang1850/arc-prize-2026-sovereign-agent",
   "kind": "kaggle-notebook",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle",
   "summary": "Skimmed only. Sovereign turns on every optional Franzen flag (best 29.08, not better than the plain submission). keithtyser's vLLM Duck (no prefix caching, 32K ctx) carries a health-check/restart watchdog. REAP-448 is a 12.5% expert-pruned checkpoint (rellik13 found 25% pruning plays worse). richardcsaky's 258 MB runtime bundle was NOT downloaded."
  },
  {
   "name": "leejianrong/solve-arc-agi-3",
   "url": "https://github.com/leejianrong/solve-arc-agi-3",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/solve-arc-agi-3",
   "summary": "Planning repo for a ~27-30B Duck-control reimplementation (Qwen3.6, vLLM) with a hypothesis ledger and replay-verified executable world model. Mostly strategy docs; no scored results or new mechanism beyond what is already in the user's backlog."
  },
  {
   "name": "Mininglamp-AI/polyphony-arc-3",
   "url": "https://github.com/Mininglamp-AI/polyphony-arc-3",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/polyphony-arc-3",
   "summary": "File-based 'Heuristic System' agent on open Qwen models (8-GPU config): the agent maintains a Python state/engine/planner that must reproduce observed transitions before planning. Compaction notes confirm per-turn trimming hurt and breaks prefix cache. Too heavy for one GPU, but supports the regression-tested-module idea."
  },
  {
   "name": "Frontier/preview references: PrimeIntellect-ai/arc-agi-3-prime-agent, NVIDIA/dream-team, DriesSmit/ARC3-solution (StochasticGoose), arcprize/ARC-AGI-3-Agents",
   "url": "https://github.com/PrimeIntellect-ai/arc-agi-3-prime-agent",
   "kind": "github",
   "local_path": "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors",
   "summary": "Prime Agent plus Claude Opus 5 reaches ~95% on the 25 public games (AGENTS.md: plan before every call, batches of 1-2 for new hypotheses, at most 20 once supported). DreamTeam is a six-agent shared executable world model. StochasticGoose is a CNN that learns which actions change the frame. The official repo is templates only. Only the Prime AGENTS.md batching rule is cheaply borrowable."
  }
 ],
 "ideas": [
  {
   "title": "Stop the O(history) state rewrite/re-parse on every action; keep runtime state in memory and honor minimal_diagnostics",
   "category": "bugfix-quickwin",
   "description": "In Franzen's harness every executed action calls solver.write_runtime_state (solver.py:1876), which dumps the whole game history to tool_runtime_state.json with json.dumps(indent=2) (runtime_state.py:255-290). Every python tool call then parses that file (tool_agent.py:7936; the comment there says the result is unused), and every action() inside the snippet calls _serialized_runtime_state (tool_agent.py:8067/8095/8151/8164), which re-parses the file and rebuilds ascii+grid payloads for every past frame (_ascii_history_view_payload, tool_agent.py:4060) and pipes them to the sandbox. Fix, following rellik13's cell 38 and lordhansolo's design: (a) serialize each HistoryEntry once and cache it; (b) pass an in-memory RuntimeState via a callable (lordhansolo's `runtime_state_source`, tool_agent.py:936/1412, solver.runtime_state()) and delete the unused first load; (c) defer per-action writes inside a batch to one write at batch end (wrap step_env); (d) honor solver.minimal_diagnostics so transcript/viewer/analysis writes are skipped in submission mode (Franzen's ARC3-Inference never reads it, although TAAF's solver.py:45 says solvers should); (e) optionally send ascii only, not ascii+grid. Add per-call timers (rellik13's m86_summary counters) and log write/parse seconds and tool-call latency versus history length on a local run first, to confirm the gain before and after.",
   "mechanism": "Per-turn overhead that grows linearly with game length is removed. rellik13 measured +4.5 s per 100 actions of history, and at 800 actions a 0.8 s write (50 MB), 0.36 s parse and 0.61 s payload rebuild per call. 36 of 150 action() calls hit the 30 s tool timeout in the last 12 minutes, which can leave a batch half executed. Franzen's priority scheduler concentrates compute on the best games, which have the longest histories, so the slowdown lands on the games that earn score. His own numbers show harness-side generation of 609 tok/s against 837 tok/s aggregate decode (WRITEUP section 2), so about a quarter of engine time is idle or prefill. All ten streams share one Python process, so the JSON work also contends for the GIL. Score follows turns per game (lordhansolo writeup), so recovered time converts to score.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/rellik13-milestone2/cells/38_patch_cell_v16_9_m86_game_state_without_the_per_ac.py:1-24 (WHY block: 0.8 s write, 0.36 s parse, 0.61 s payload at 800 actions; +4.5 s/turn per 100 actions; 36/150 killed action calls)",
    "https://www.kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2 ('Keeping game state in memory instead of on disk removed most of the tool latency and kept the engine fed')",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/framework/solver.py:253-256,551,644 (diagnostics gated on minimal_diagnostics)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:7935-7936,7957,8067,8151,4060 and inference/framework/solver.py:1876"
   ],
   "effort": "half-day",
   "expected_impact": "high",
   "status_in_franzen": "absent"
  },
  {
   "title": "Relaxed MTP acceptance (SPEC_ACCEPT_SINGLE/ACC = 0.3) instead of lossless 1.0",
   "category": "serving",
   "description": "Franzen's notebook CFG sets SPEC_ACCEPT_SINGLE=1.0 and SPEC_ACCEPT_ACC=1.0 and passes them as --speculative-accept-threshold-single/--speculative-accept-threshold-acc (his notebook cell 12, lines 10438-10439 and 10715-10716 of the pulled copy). Change both to 0.3, with 0.5 as a milder arm. Gabriel Olympie's serve.sh force-accepts a draft token whenever the target gives it at least that probability. Measure decode tok/s and accept length from the SGLang log, then run a token-matched level-completion comparison (at least 3 matched seeds) against 1.0. The flag already exists in the Pennyroyal build Franzen uses, so no patching is needed.",
   "mechanism": "Raises tokens accepted per speculative step, so more tokens per second per stream and per GPU. Gabriel's ladder at temperature 0.6, single stream: 1.0 gives 179 tok/s, 0.5 gives 203, 0.3 gives 231 (accept length 2.1-2.2 greedy vs 2.5-3.0 relaxed). It is lossy at temperature above 0 (it sharpens sampling toward high-probability tokens), but the user's temp 0.6 vs 0.7 arm showed no score effect, so a small sharpening should be tolerable. Gain at 10 streams will be smaller than the single-stream 29% because batching already amortizes weight reads.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120/scripts/serve.sh:73-85",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120/docs/STATUS.md:50-52 (ladder 1.0=179, 0.5=203, 0.3=231; all quality gates pass at 0.3)",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/franzen/franzen.txt:10438-10439,10715-10716 (Franzen's published notebook: thresholds 1.0)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md (temp06: 45.92 vs 46.25 control; greedy draft accept 2.67 to 2.81)"
   ],
   "effort": "<2h",
   "expected_impact": "medium",
   "status_in_franzen": "present-but-off"
  },
  {
   "title": "Pin a solved-level memory block (rule + last ~30 winning actions + closing reasoning), re-shown after trims",
   "category": "memory-context",
   "description": "Port rellik13's M85. On a level transition, record the cleared level's number, action count, last 30 actions (the winning tail), and the tail (about 900 chars) of the model's reasoning from the clearing turn. On the next prompt ask for one 'Rule for level N:' paragraph (goal, what each control did, winning plan, at most 5 lines), captured from the reply or its reasoning. Keep the newest three levels in full and older ones as rules only. Franzen's harness has level-transfer guidance and game-scope retained functions but nothing that preserves how earlier levels were solved once they are trimmed. For prefix-cache economy, inject the block as the level-up user message and again right after each history trim (the user's ledger-on-trim hook), instead of editing the system prompt. rellik13 edited the system prompt, which re-prefills the whole prompt once per solved level; that cost is acceptable but not necessary. Fold in the ledger-on-trim port from arms.md rather than building two memories.",
   "mechanism": "Returns evidence the agent would otherwise lose (rellik13: in all four lost levels the agent re-derived the previous level's rule for 60-100+ minutes; cd82 L2 took 122 actions vs 8 with the rule kept). It was the only text addition that helped (+2.8 on a 7.14 base, a single run). The user's ledger-on-trim measured +6.3 offline (52.56 vs 46.25, one run). Both are the same class: re-inject facts after trimming without adding standing instructions. Franzen's world-model memory was switched off precisely because model-written notes were parsed unreliably; this version is harness-captured (winning actions are facts) so it avoids that failure.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/rellik13-milestone2/cells/42_patch_cell_v16_8_m85_solved_level_memory.py:1-30",
    "https://www.kaggle.com/writeups/sirikilohit/arc-prize-2026-arc-agi-3-3-milestone-2 (solved-level memory 7.14 to 9.96; 'More text made the agent worse')",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/kinetic-learning-by-playing-v8/kinetic-learning-by-playing-v8.txt:10325-10335 (ARC3_X_SOLVED_MEMORY, an action-tail-only variant by another author)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md (ledger-on-trim: 52.56 vs 46.25)"
   ],
   "effort": "1 day",
   "expected_impact": "high",
   "status_in_franzen": "partial"
  },
  {
   "title": "Level-aware history trimming, gated on a sawtooth natural-experiment check of the user's own runs",
   "category": "memory-context",
   "description": "Step 1 (<2h): run rellik13's analysis/sawtooth.py on the user's Franzen-style run transcripts. For each turn it measures how much of the current level's own history is still visible and whether the level is cleared within the next W turns, comparing visible-more vs visible-less at the same point of the same level (stratified by turns into the level and turn size). Step 2: if more visible in-level history helps, change _drop_oldest_history_block (tool_agent.py:8564-8617, a plain pop(0) with no level notion) so a trim removes whole completed-level blocks first (leaving the solved-level block from the previous idea) and preserves the first K turns of the current level (the initial probes) plus the most recent ones, instead of cutting the current level's earliest turns once it outgrows the window.",
   "mechanism": "rellik13's stuck-level transcripts showed the right goal but no record of what had been tried, because trims left a handful of turns on levels that take dozens of turns; fixing retention alone (FP8 KV + 57k/45k watermarks) took 14.49 to 22.53 on their setup. Franzen's 118k/59k window is already much longer, but his trim is blind to levels, so a long hard level still loses its early evidence first. The diagnostic keeps this from becoming another unvalidated memory change.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/rellik13-milestone2/analysis/sawtooth.py:1-12",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/rellik13-milestone2/WRITEUP.md (FP8 KV and longer history section)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:8564-8617"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Adopt a decision protocol before spending the remaining submissions: paired seeds, rejection panel, token-matched comparison",
   "category": "other",
   "description": "Use the evidence others published on noise. (a) Unchanged Franzen copies scored mean 25.77, SD 3.93 on the LB, so a change needs about +8 to be 90% sure; the user's 22.88 is inside that spread. (b) The audit of Franzen's 100 public trajectories shows 50.88 of the 53.5 lost points come from incomplete levels and only 2.63 from action inefficiency, so prioritize completion arms over efficiency arms. (c) Six weak games (sk48, bp35, lf52 and others) complete 27/196 levels with 0 wins and take about 3.12M of 17.5M tokens (18%). (d) The 'Why One Run Lies' notebook finds ft09, sp80, tn36 and vc33 recur in cross-validated 5-game panels (0.953 median held-out correlation, worst fold 0.573): use them only to reject regressions, then promote on three matched full-25 seeds with paired per-game deltas and compare at equal generated tokens, not equal wall time (rellik13). Reuse their embedded 20-pass data and RHAE scorer, and the user's existing make significance.",
   "mechanism": "With one submission per day and SD of about 4, most apparent gains in the next 5 days will be noise; a cheap rejection panel plus paired seeds avoids both promoting lottery wins and discarding real +2-3 point improvements. The completion-versus-efficiency accounting shows where the 35-point headroom is.",
   "evidence": [
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745062 (copies: mean 25.77, SD 3.93, median 26.19)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/arc-agi-3-27-80-lb-100-trajectory-audit/arc-agi-3-27-80-lb-100-trajectory-audit.ipynb (markdown cells 6, 9, 11)",
    "https://www.kaggle.com/code/skarin/arc-agi-3-27-80-lb-why-one-run-lies (panel and experiment ladder)",
    "https://www.kaggle.com/writeups/sirikilohit/arc-prize-2026-arc-agi-3-3-milestone-2 (equal-token comparison)"
   ],
   "effort": "<2h",
   "expected_impact": "high",
   "status_in_franzen": "absent"
  },
  {
   "title": "Add an optional `plan` field to the python tool, store it per action, and render a plan trail after trims",
   "category": "tools-repl",
   "description": "Franzen's python tool takes only `code` (tool_agent.py:7188-7200). Add an optional one-line `plan` string, normalized like lordhansolo's `_normalized_plan` (tool_agent.py:1070) and stored on every action executed by that call (`history[i].plan`, `step_env` payload at tool_agent.py:1042). Expose `history[i].plan` in the sandbox state. Then let the harness, with no extra model call, print a compact trail of the last N (action index, plan, result flags) for the span a trim just removed. This is a harness-built variant of the user's ledger-on-trim that does not rely on parsing model-written notes. Keep the 'plan' text out of the cached prefix except at the trim event.",
   "mechanism": "After a trim the REPL history survives but the reasoning behind each action does not; the plan field keeps the model's own hypothesis for each action queryable (and renderable) at about 20 tokens per call. It converts the unreliable free-text world model (Franzen turned it off) into structured, harness-captured memory.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/agent/tool_agent.py:734,1042,1070",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py (`history[i].plan` text: 'Once a turn leaves your context this is the only record of what you were testing')",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py:7188-7200",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md (hypothesis-loop 93% compliance, inconclusive)"
   ],
   "effort": "half-day",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Add a `memo` dict that persists JSON-serialisable variables across python calls (functions are already retained)",
   "category": "tools-repl",
   "description": "Port the ARC3_X_MEMO patch from the Kinetic v8 notebook: the sandbox exposes a `memo` dict restored at the start of each call and collected at the end (JSON-serialisable values only, capped at 20,000 chars, offenders dropped with a note), passed through run_sandboxed_python as `memo=`. Add one system-prompt line explaining it. Franzen retains only function source; ordinary variables vanish after each call, so a value computed in call 1 (object list, level map, measured step size) is recomputed or raises NameError in call 2.",
   "mechanism": "Cuts repeated generation of recomputation code and NameError retries, which cost tokens and turns. The Kinetic author reports NameErrors on values from earlier calls were the most common error in their logs (author claim). The patch is self-contained and low risk, but the notebook that carries it scored 27.28, i.e. no measurable gain over plain Franzen, so expect a small effect.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/kinetic-learning-by-playing-v8/kinetic-learning-by-playing-v8.txt:10205-10236,10292-10297 (memo patch)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md (Retaining Python functions: 'Ordinary variables are discarded')"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "Harness-measured action effects: changed_pixels / largest_changed_regions / per_step_changed_pixels in every result, plus a per-level 'UNTESTED actions' line",
   "category": "perception",
   "description": "Franzen's action result carries only booleans (board_changed, gameplay_changed, no_op, animation). Add lordhansolo's magnitude fields: changed_pixels, total_pixels, largest_changed_regions (bbox + count, top 4), changed_region_count and per_step_changed_pixels for batches (his runtime_state.py changed_regions/changed_pixel_count), plus the one-line reading guide (0 = identical, a few cells = timer tick, a moderate count = one object moved, a huge count = reset or new level). Optionally add Scott Le Grand's ledger: per level and per action name, tried count and measured outcome (no change, border-only, shifted colour X up, changed N cells; click near/away), and a deterministic line 'UNTESTED on this level: ACTION5, MOUSE' listing valid actions not yet tried (_ledger_add/_ledger_lines).",
   "mechanism": "Gives the model the cheapest discriminating signal (magnitude and location of change) without it having to call frame_diff and print boards, and nudges it to probe every control once before writing one off. Costs about 30-80 tokens per turn. Evidence is mixed: Scott shipped the ledger off by default after a regression hunt and rellik13 found a model ledger went unused and 'what changed' notes did not help at a 37k cap; Franzen has 128k, so the token cost is smaller.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/agent/prompts.py (`Result keys in last_action_result`)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/agent/runtime_state.py:154-200",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/taaf-flashnext-sheetu12b-0922/taaf-flashnext-sheetu12b-0922.txt:817-900",
    "https://www.kaggle.com/writeups/sirikilohit/arc-prize-2026-arc-agi-3-3-milestone-2 (ledger and 'what changed' notes: no gain)"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Replace the fixed 4-cell no-op border with a learned HUD-band mask",
   "category": "perception",
   "description": "Franzen's gameplay_changed / batch no-op / stale-state guards ignore a fixed 4-cell border (ARC3_NOOP_GUARD_BORDER=4); his writeup concedes 'genuine game elements can exist near the border'. Scott Le Grand's F5 learns the budget bar instead: a band cell counts as HUD if it flips on at least 90% of single actions since first seen with one fixed colour transition (HUD_RATE=0.9, HUD_CELL_MIN=5, at most 3 cells per action, HUD_BORDER=2) and reports board_changed_ex_hud, changed_bbox and distinct frames. His survey found a 1-2 cells/action budget bar in 25/25 public games. Before building, measure on the user's transcripts how often a guard fired where the changed cells were within 4 of the edge but not part of a ticking bar; build only if that is non-trivial.",
   "mechanism": "Avoids false 'no gameplay change' verdicts that truncate valid batches or block follow-up actions when a real mover or counter sits near the edge, and avoids false progress from HUD flicker outside the 4-cell strip. Wasted or wrongly blocked actions cost level efficiency and turns.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/taaf-flashnext-sheetu12b-0922/taaf-flashnext-sheetu12b-0922.txt:289-312,380-420",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md (ARC3_NOOP_GUARD_BORDER default 4)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/WRITEUP.md (section 5)"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Add lordhansolo-style saved modules with frozen-observation regression tests alongside retained functions",
   "category": "tools-repl",
   "description": "Franzen restores model-defined functions by source; a later rewrite can silently break behavior that earlier levels validated. Port lordhansolo's python_workspace.py (301 lines) and the prompt text: `save_module(name, source, tests, replace_tests)`, `load_module`, `read_module`, `saved_modules`. Each test string runs `check(actual, expected, label)` against a frozen snapshot of the observations captured when the test was added, and a save that breaks any earlier suite is rejected unless replace_tests=True. Show a catalog of module signatures in the turn prompt. Keep Franzen's function retention as the default and treat modules as an opt-in arm.",
   "mechanism": "Makes learned simulators/planners cumulative across levels and across history trims without trusting the model's memory of the rules. The Rodionov papers find executable, verified world models help most at lower reasoning effort and are 'not universally beneficial' (textual variants matched them with 41% fewer actions at max effort), and lordhansolo says he cannot confirm any gain, so expect a small and uncertain effect for a large port.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/agent/python_workspace.py",
    "https://www.kaggle.com/writeups/lordhansolo/arc-agi-3-milestone-2 ('I cannot confirm that any of these changes improved the score')",
    "https://arxiv.org/abs/2607.15439 (executable world models / verification ablation)"
   ],
   "effort": "2-3 days",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Rebuild the FR-Spec hot-token map from ARC transcripts (smaller draft vocab)",
   "category": "serving",
   "description": "Franzen's draft head scores the generic pinned 64k map (hot_tokens_64k.pt, sha-asserted in build_bundle_pennyroyal.sh:1408,1576-1579 and the notebook's TOKEN_MAP_SHA). lordhansolo instead ships configs/draft_vocab_32k.json built from 23,853 token ids actually generated in 125 Flash-Next ARC transcripts, padded to 32,000. Generate a map from the user's own Franzen-run transcripts (reasoning plus python) with Gabriel's scripts/make_hot_tokens.py (e.g. size 32768, base-ids 8192, corpus = the saved transcripts), update the sha assert, and A/B accept length and decode tok/s.",
   "mechanism": "The draft lm_head GEMV runs three times per step; a map matched to the real output distribution can keep accept length while halving draft logits cost, or raise accept length at fixed size (Gabriel measured +9% decode from the 64k map vs full vocab). Expected gain is small (a few percent) and acceptance could drop if the map is too narrow.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/configs/draft_vocab_32k.json (description field)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120/scripts/make_hot_tokens.py:1-20",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120/scripts/serve.sh:78-84"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "partial"
  },
  {
   "title": "Retune the history drain (118k to about 80k rather than to 59k), after the sawtooth check",
   "category": "model-config",
   "description": "Franzen trims at 118k and drains to about 59k in one go (drain ~50%). lordhansolo trims at ~126k and drains to 81.5k; rellik13 drains 57k to 45k (21%). Sweep ARC3_CONTEXT_DRAIN_TOKENS so the post-trim window is about 80k (the Mamba/prefix-cache patch already handles the re-prefill). Compare in-window history, trim frequency, prefill share (Franzen: 15% of wall time) and level completions at equal generated tokens. Only run it if the sawtooth analysis shows more visible in-level history raises clear rates.",
   "mechanism": "A shallower drain keeps about 18% more history in the window on average, at the cost of trimming roughly 25% more often (each costing a long prefill). Whether evidence beats prefill cost is empirical. Franzen's earlier 52k drain and the user's streams14 arm (96k/42k) showed no clear difference, so expect a small effect and treat it as a cheap config arm, not a bet.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/tong-m2-comparison/kaggle/SUMMARY.md (Context management row)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/agent/tool_agent.py:458,1291",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md (streams14: 46.38 vs 46.25)"
   ],
   "effort": "<2h",
   "expected_impact": "low",
   "status_in_franzen": "tried-no-gain"
  },
  {
   "title": "FP8 weight-only dense stack and FP8 lm_head/HyperConnection (Gabriel Olympie patches 0005/0006) on the Pennyroyal base",
   "category": "serving",
   "description": "Intel's W4A16 AutoRound keeps self_attn, linear_attn, shared_expert, hyper_connection, in_proj_a/b and lm_head in BF16 (HF model card), so Franzen's decode still reads that BF16 dense stack every step; Gabriel measured it at 7.2 of 10.14 GB per step on the NVFP4 checkpoint and halves it with per-row-scaled fp8e4m3 copies served by his Triton low-M kernel (SGLANG_SM120_LOWM_FP8_WEIGHT=1, SGLANG_SM120_LM_HEAD_FP8=1). Franzen ported only patch 0004. Port 0005 and 0006 onto Pennyroyal v2.5.3 (they build on 0004), cost about 3.6 GB VRAM (Franzen's notebook shows 4.25 GB free), validate with greedy/needle/cached-prefix gates and a decode benchmark at 10 streams on an RTX PRO 6000 before any Kaggle run.",
   "mechanism": "Cuts dense per-step bytes by about 40% and moves those GEMMs off Ampere-class WMMA kernels; Gabriel lists it among the largest wins after the GEMM kernel (kernel half 1.3-1.5x). At batch 10 the dense stack is a smaller share of step time than at batch 1, so expect a modest aggregate gain. Risks: porting effort, quality drift on attention weights, and the 3.6 GB of KV pool it displaces. The user's MXFP8 trial lost 27% decode, but that was activation-quantized MXFP8; this is weight-only.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120/docs/STATUS.md:44-49",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/sglang-flashnext-sm120/docs/PERF_CEILING.md:68,88,123-124",
    "https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (layers kept in BF16)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md (MXFP8 online activations: decode -27%)"
   ],
   "effort": "2-3 days",
   "expected_impact": "medium",
   "status_in_franzen": "absent"
  },
  {
   "title": "Guard the 9-hour run: SGLang health watchdog with bounded restarts, and an exact server-argument check",
   "category": "bugfix-quickwin",
   "description": "keithtyser's vLLM Duck bundle runs a background watchdog that polls /v1/models every 15 s, restarts the server only after 4 consecutive failures, confirms the old process and port are gone, and allows at most 2 restarts (notebook cell 0 and cell 15). Franzen launches SGLang once with --watchdog-timeout 1800 and no supervisor, so a server death leaves the remaining games with no model (the user's arms log already has an SGLang-death stop, which ends the run rather than recovering it). Add a supervisor that relaunches the same argv (about 530 s startup, all prefix caches lost) when the health check fails, with the existing stop as the final fallback. Also add rellik13's check that the final server argv equals an intended list: his FP8 build silently lost two flags to commented-out lines and was nearly shipped.",
   "mechanism": "Protects the remaining budget against a single crash or Kaggle-side outage. Son Pham and rellik13 report competition reruns finishing in 2-4 hours instead of 9 around a platform outage; a restart would recover part of such a run, and a flag audit prevents a silent regression. Probability of a crash is unknown, so this is insurance, not an expected-score lever.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/duck-qwen3-8-anim-base/duck-qwen3-8-anim-base.txt (cell 0 description, cell 15 watchdog start)",
    "https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744995 (runs completing in 2-4 hours)",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/rellik13-milestone2/WRITEUP.md (server-argument check)",
    "/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/franzen/franzen.txt:10701 (--watchdog-timeout 1800, single launch)"
   ],
   "effort": "half-day",
   "expected_impact": "low",
   "status_in_franzen": "absent"
  },
  {
   "title": "A/B a short, cached 'mechanics catalog' and a no-action circuit breaker (tried by others with no measured gain)",
   "category": "prompting",
   "description": "Kinetic v8 adds (a) a 19-line knowledge pack in the cached system prompt and (b) a mechanics catalog sent once per game and once per new level, written from the public games: movement by fixed step, push/carry, select-then-act, linked or mirrored objects, cycle controls, remote buttons, goal types (cover/reach/reproduce/submit), budgets and hazards. It also includes rules such as 'if three calls in a row sent no action, your next call must act' and one measured stat (levels never cleared spent 76% of calls with no action, cleared levels 55%). The private set comes from the same authors, so the families may transfer. Test it as one flag in the system prompt (cache-friendly at 128k context), plus a harness-side nudge after K consecutive no-action tool calls (Scott Le Grand's F4 used K=6). Note the competing evidence: rellik13's score dropped from 6.42 to 4.16 and 13.40 to 9.98 when text was added at a 37k cap, and Kinetic v8's own LB result is 27.28, equal to plain Franzen.",
   "mechanism": "A prior over mechanic families could shorten hypothesis search on new games, and a circuit breaker cuts long no-action deliberation (tokens per level). But added standing text competes with the model's own history for context, the user's own prompt arms (scoring guide, thinking guide, audit60k) found nothing, and the only measured comparison here is flat. Treat as low priority; run only once the higher-confidence arms above are done.",
   "evidence": [
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/competitors/kaggle/kinetic-learning-by-playing-v8/kinetic-learning-by-playing-v8.txt:10114-10175 (knowledge_pack.txt, mechanics_catalog.txt), 10600 (ARC3_X_CATALOG enabled)",
    "https://www.kaggle.com/code/desihnurseha/kinetic-learning-by-playing-v8 (public score 27.28)",
    "https://www.kaggle.com/writeups/sirikilohit/arc-prize-2026-arc-agi-3-3-milestone-2 ('More text made the agent worse')",
    "/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/arms.md (audit60k 44.10 vs 46.25; scoring guide, thinking guide)"
   ],
   "effort": "1 day",
   "expected_impact": "low",
   "status_in_franzen": "tried-no-gain"
  }
 ]
}
```
