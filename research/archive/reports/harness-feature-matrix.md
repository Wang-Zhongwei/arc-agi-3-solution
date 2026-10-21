# ARC-AGI-3 Duck harness variants: feature matrix (2026-10-03)

Legend: ✓ on in the scored/final run · ○ implemented but off · ✗ absent.

Column sources:
- **Tufa:** `tufalabs-duck-harness/ARC3-Inference`, public commit 7652836.
- **Wang:** `final-c1/ARC3-Inference`. The env values come from the rendered RUN_CONFIG in `kaggle-outputs/final-c1-ledgerfix/taaf-final-c1-ledgerfix.log`.
- **Franzen:** `franzen/ARC3-Inference`, with the env from `external/dfranzen/nb.txt` lines 10160-10250 and serving CFG at 10402-10425.
- **Hansolo (lordhansolo):** `external/hansolo/latest/src`, branch `solution-improve-qwen38-flash-next`. This is newer than his scored notebook v293/bundle v297. `S:` means `setup_script.sh`.
- **Siri (sirikilohit / rellik13):** `external/sirikilohit/notebook/nb.txt`. Its bundle is byte-identical to Franzen's `bundle-base`, and every change is a notebook monkey-patch (M##).

Path notes:
- File paths are relative to `ARC3-Inference/inference/` unless stated otherwise.
- "FZ env" means a variable set in Franzen's notebook.
- "fz-ours" marks our additions on `franzen/base`; none of them were in Franzen's LB run.

## A. Matrix

### Serving / throughput
| Feature | Tufa | Wang | Franzen (as run) | Hansolo | Siri |
|---|---|---|---|---|---|
| Engine | vLLM (`framework/kaggle.py:311-336`) | vLLM, 2-node TP on Slurm; Kaggle bundle | SGLang Pennyroyal v2.5.3 + 5 patches (`serving/README.md`) | vLLM nightly e975732 + 6-file Mamba-cache overlay (S:24) | SGLang Pennyroyal v2.5.0 (nb 645-665) |
| Model / quant | Qwen3.6-27B-FP8 (`configs/inference.json:3`) | Flash-Next W4A16 AutoRound | Flash-Next W4A16, BF16 PLE offloaded (nb 10669-10680, `--ple-offload-embedding`) | Flash-Next mixed NVFP4/FP8 (`kaggle.py:20-23`) | W4A16 with FP8 PLE spliced in (M82) |
| FP8 KV cache | ✗ | ✗ (`KAGGLE_KV_CACHE_DTYPE`="", bf16) | ✓ `fp8_e4m3` (CFG KVDTYPE) | ✓ fp8_e4m3 | ✓ |
| MTP / spec decode | ✗ | ✓ mtp 3 (bundle profile, run log) | ✓ NEXTN 3 + FR-Spec hot-vocab map (nb 10694-10706) | ✓ MTP 3 + 32k draft vocab | ✓ MTP 3, NVFP4 draft (M108) |
| Mamba prefix-cache patch | ✗ | ✗ | ✓ sparse prefill checkpoints + LRU refresh | ✓ finer Mamba prefix cache | Pennyroyal stock |
| Hierarchical host KV cache | ✗ | ✗ | ✗ | ✗ (embeddings offloaded only) | ✓ 48 GB hicache (nb 661) |
| Concurrent streams | 32 jobs | 8 games, max-num-seqs 12 | 10 (MAXREQ, `ARC3_MAX_ACTIVE_STREAMS=10`) | 14 | 16 requests, 28 games live (nb 991) |
| Temperature | 0.6 | 1.0 | 0.7, top_p 0.95, top_k 20 | 0.6 (code default 1.0) | 0.6 |
| Measured decode | n/a | ~297 tok/s | ~610 tok/s (our fz-arms v4) | n/a | n/a |

### Context & cache
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Context window | 65,536 server (`kaggle.py:14`) | 49,152 | 131,072 harness / 139,264 server | 127,488 harness / 147,072 server (S:31-32) | 69,632 |
| Trim with hysteresis | ✗ (trim-to-fit + 30 assistant turns, `tool_agent.py:151`) | ✓ 36,864→26,624 | ✓ `ARC3_CONTEXT_DRAIN_TOKENS`=58Ki, `HISTORY_ASSISTANT_TURNS=150`, `TURN_DRAIN=30`, coalesce | ✓ ~126.5k→81.5k (`tool_agent.py:1281-1350`) | ✓ 57,344→45,056 (M84, 3115) |
| Calibrated token / image estimate | ✗ | ✗ (double-counts reasoning) | ✓ default `ARC3_CALIBRATE_TEXT_TOKENS`, image-dimension estimate | partial (server overflow → drop block, 1524-1537) | ✓ exact server counts, images = 66 tok (M74, M79) |
| Context-overflow recovery | ✗ | ✗ | ✓ SGLang wording recognised | ✓ | ✓ + emergency repair (3175) |
| Cache-aware admission (slot held across tool calls) | ✗ | ✗ | ✓ priority bands (`tool_agent.py` ~2000-2100) | ✗ | ✗ |
| Rolling summaries | ✗ | ✗ | ○ `ARC3_SUMMARY_INTERVAL_TOKENS=0` | ✗ | ✗ |
| History image pruning | ✗ | ✗ | ○ `ARC3_HISTORY_IMAGE_KEEP` unset | ✗ | ✗ |

### Scheduling / compute allocation
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Fixed per-game budget, FIFO concurrency | ✓ | — | — | ✓ 3,918 s, 14 at a time (`solver.py:326-340,1348`) | initial slice |
| UCB time allocation | ✗ | ✓ ucb_c 2e-4, 300 s quanta (`framework/allocation.py`) | ✗ | ✗ | ✓ 2,400 s slice, then UCB (M72, 4154) |
| Exploration floor | ✗ | ✓ 0.25 (`allocation.py:204-257`) | ✗ (B-tail plays this role) | ✗ | ✗ |
| Priority gate (A+B)·C, all games started | ✗ | ✗ | ✓ `ARC3_MAX_ACTIVE_STREAMS=10`, TAIL_LOOKUP=remaining, SCORE_NORMALIZATION, REFRESH_QUEUE | ✗ | ✗ |
| Tail fade near deadline | ✗ | ✗ | ✓ last 20% | ✗ | ✗ |
| Pace adjustment | ✗ | ✗ | ○ `ARC3_PRIORITY_PACE=0` | ✗ | ✗ |
| Yield trigger | 60 s | 60 s | 2,048 generated tokens (`LOCAL_ANALYZER_YIELD_TOKENS`) | 150 s, growing ×3 (`tool_agent.py:381-395`) | none, unlimited calls per turn (2416) |

### Memory / world model
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Tufa structured world-model sections | ✓ (`tool_agent.py:270-283`) | ✓ (kept) | ○ `ARC3_MEMORY_SECTIONS=off` | ✗ (prompts rewritten) | ✗ (replaced by Wang turn msg, 3448) |
| Tagged memory ledger (world_model/hypothesis/level_note) | ✗ | ✓ `LOCAL_ANALYZER_MEMORY_LEDGER` default on (`tool_agent.py:323`, `memory_ledger.py`) | ○ fz-ours `ARC3_LEDGER_ON_TRIM` (`tool_agent.py:137`) | ✗ | ○ M24 ledger disabled (3457) |
| Solved-level memory pinned in system prompt | ✗ | partial (`<level_note>` tag) | ✗ | ✗ | ✓ rule + last 30 actions + 900 chars of reasoning for the 3 newest levels (M85, 3902-4103) |
| Per-action plan note that survives trimming | ✗ | ✗ | ✗ | ✓ `plan` ≤500 chars (`prompts.py:73-75`) | ✗ |
| Game-over log for the level | ✗ | ✗ | ○ `ARC3_DEATH_LEDGER` | ✓ `game_overs` list (`tool_agent.py:151-165`) | ✗ |
| Logprob uncertainty feedback ("least certain last turn") | ✗ | ✓ TOP_LOGPROBS default 5 (`tool_agent.py:339,2393`) | ✗ | ✗ | ✗ |
| Scratchpad | ✗ | ○ | ✗ | ✗ | ✗ |

### Perception
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Current-grid image, upscale | ✓ ×4 | ✓ ×8 | ✓ ×10 | ✓ ×4 | ✓ ×4 |
| Diff image | ✗ | ○ `MULTIMODAL_DIFF_IMAGE` | ✓ `ARC3_DIFF_IMAGE` (`tool_agent.py:351`) | ✗ (text change stats, 660-680) | ✗ |
| Game-over fatal frame / diff image | ✗ | ✗ | ✓ `ARC3_GAMEOVER_DIFF_IMAGE` (4406) | ✗ | ✗ |
| Animation frames | ✗ | ✓ images, ≤8 frames (`MULTIMODAL_ANIMATION=true`) | ✓ `ARC3_ANIMATION`: REPL vars + transient hint, no images (`ANIMATION_IMAGE=off`) | ✓ ≤16 frames + tween/path/blink classifier (`agent/animation.py:37`) | ✓ ≤4 mid-animation images (M76) + sandbox frames (M111) |
| `frame_diff` helper | ✗ | ✓ colour-transition diff (`python_tool_sandbox.py:313`) | ✓ object-level (`utils/frame_diff.py`) + `ARC3_FRAME_DIFF_HINT` | ✗ | ✓ (Wang port) |
| Auto text frame diff in prompt | ✗ | ✗ | ○ `ARC3_AUTO_FRAME_DIFF` | ✓ change stats | ✗ |
| Segmentation | ✓ | ✓ | ✓ + rotation-aware hashes | ✓ + `.crop()`, adjacency | ✓ |
| Numpy grid | ✗ | ○ `PERCEPTION_NUMPY` | ✗ | ✗ | ✗ |
| Label grid / ASCII overlay | ✗ | ○ | ✗ | ✗ | ✗ |
| Level inventory diff | ✗ | ✗ | ○ `ARC3_LEVEL_INVENTORY` | ✗ | ✗ |

### Action API & guards
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| UNDO (ACTION7) usable | ✗ (listed, not executable) | ✗ (hidden, `action_names.py:35`) | ✓ `EXPOSE_UNDO=on` | ✓ (`agent/action.py:20`) | ✓ (M99, 3709) |
| Model-initiated RESET | legacy | legacy | ○ `EXPOSE_RESET=off` | ✗ removed (`solver.py:177`) | ✓ allowed, not advertised |
| Terminal latch (stop the batch on level/game over) | prompt only | prompt only | ✓ unconditional | ✓ | ✓ (Tufa) |
| Batch no-op / stale-state guards | ✗ | ✗ | ✓ `ARC3_BATCH_NOOP_BLOCK`, `ARC3_STALE_STATE_BLOCK` from level 2 | ✗ (prompt line, 167) | ✗ |
| Repeat no-op / death / repeat-state guards | ✗ | ✗ | ○ (`tool_agent.py:3617-3630`, 2396) | ✗ | ✗ |
| `action(until=)` guarded chunks | ✗ | ○ `LOCAL_ANALYZER_ACTION_CHUNKING` (not in the final build) | ✗ | ✗ | ✗ |
| Action-info system prompt | ✗ | ✗ | ✓ `ARC3_ACTION_INFO` (2747) | ✓ state line (697-708) | ✗ |

### Prompts
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Game-over wording | "The game is over." (`tool_agent.py:1210`) | same (2367) | ✓ long diagnosis + BAR RULE + "do not replay" (`tool_agent.py:5039-5090`) | "game over… RESET counted as one action" (684-688) | ✓ "Level restarted." (M102, 3795) |
| Level-transfer guidance | ✗ | ✗ | ✓ `ARC3_LEVEL_TRANSFER_GUIDANCE` | ✗ (one line, 170) | via solved-level memory |
| Budget-bar "ignore it" lines removed | ✗ | ✗ | ✗ (kept at `prompts.py:139,378`; BAR RULE only after death) | ✓ (rewritten as an action budget, 58,147) | ✓ 2 phrases deleted (3882) |
| Scoring explained | ✗ | ○ `LOCAL_ANALYZER_SCORING_GUIDE` (`prompts.py:157`) | ✗ | ✓ (`prompts.py:56-57`) | ✗ |
| Act without certainty | ✗ | ○ (inside the scoring guide) | ✗ | ✓ "Full certainty is not required" (156) | partial |
| Strategy audit when stuck | ✗ | ✗ | ○ fz-ours `ARC3_STRATEGY_AUDIT_TOKENS` (port of Hansolo) | ✓ every 25% of the budget (`solver.py:239-250`) | ✗ |
| Thinking guide | ✗ | ○ | ✗ | ✗ | ✗ |
| Game priors / turn protocol | ✗ | ✓ (`prompts.py` GAME_PRIORS, TURN_PROTOCOL) | ✗ | ✗ | ✓ Wang/Ludvig port (M90) |
| Clarified repeated opener prompts | ✗ | ✗ | ✓ `ARC3_NEW_CHANGED_PROMPTS`, `EXPLAIN_GAMEPLAY_CHANGED` | n/a | ✗ |
| Commit / prefer-tool / step-verify hints | ✗ | ✗ | ○ | ✗ | ✗ |

### Python tool
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Model | fresh subprocess per call | ✓ persistent interpreter + `store` / `register_predictor` / `mechanics()` (`prompts.py:109-112`) | ✓ retained functions, game scope + imports (`ARC3_PERSISTENT_FUNCTIONS*`) | ✓ tested saved modules (`python_workspace.py`) | fresh per call (M44b sticky vars overridden) |
| Tool output cap | 1,024 | 1,024 | 3,072 + middle truncation + action echo | 1,024 | 1,024 |
| Ruff format / lint feedback | ✗ | ✓ `SANDBOX_RUFF_FORMAT=1` | ✗ | ✗ | ✗ |

### Reliability
| Feature | Tufa | Wang | Franzen | Hansolo | Siri |
|---|---|---|---|---|---|
| Model-request HTTP retries | ✗ | ✗ | ✓ `ARC3_HTTP_RETRIES=3` (5367), startup grace 900 s | ✓ 1 s retry (`solver.py:462`) | ✓ |
| Unsent game-API request retry | ✗ | ✗ | ✓ fz-ours (`taaf/game_api.py:58-77,240`); absent from his run | ✓ (`game_api.py:63-75`) | ✗ |
| Auto-RESET retry | ✗ | ✗ | partial | ✓ (885-920) | ✓ 4× backoff (M102) |
| Warm-up action while the server boots | ✗ | ✗ | ✓ `ARC3_WARMUP_ACTION_GAMES=1` | ✗ | ✗ |
| Server watchdog / restart | ✗ | ✗ | ✗ | ✓ restart ≤5 | ✓ abort after 3 failed polls |
| Text tool-call fallback | ✗ | ✗ | ✓ (511) | ✓ (1545) | ✗ |

## B. Franzen breakdown

**Implemented and ON in his run:**
- Serving and context:
  - SGLang Pennyroyal + FP8 KV + NEXTN-3 MTP + FR-Spec, with the Mamba prefix-cache patch.
  - 128 Ki context with 58 Ki drain hysteresis, calibrated token estimates and overflow recovery.
- Scheduling:
  - Priority gate over all games started at once, admitting 10 streams.
  - (A+B)·C with the remaining-level tail lookup and A normalisation; tail fade over the last 20%.
- Perception and actions:
  - Upscale ×10, diff image, game-over diff image, animation REPL variables with a transient hint.
  - `frame_diff` helper with its hint.
  - UNDO, action info.
- Prompts:
  - Level-transfer guidance, detailed game-over prompt, new "changed" prompts.
- Guards and Python tool:
  - Batch no-op and stale-state guards from level 2, terminal latch.
  - Game-scoped retained functions with imports.
  - Tool output 3,072 with middle truncation and action echo.
- Reliability and operations:
  - Yield after 2,048 tokens, HTTP retries, warm-up action, text tool-call fallback.

**Implemented but OFF:**
- Memory: structured memory sections, rolling summaries, death ledger.
- Actions and guards: EXPOSE_RESET; noop-repeat, death-repeat and repeat-state guards; board-dump guard.
- Perception: auto text frame diff, level inventory, gameover text diff, animation images, history image keep.
- Prompts and scheduling: commit, prefer-tool and step-verification hints; pace adjustment; reasoning-effort ladder; control-message pruning.
- Ours, not in his run: `ARC3_STRATEGY_AUDIT_TOKENS`, `ARC3_LEDGER_ON_TRIM`.

**Not implemented (but present elsewhere):**
- Serving and context: 48 GB hierarchical host KV cache, more streams (14 or 16 vs 10), FP8 PLE (Siri).
- Memory:
  - Solved-level memory pinning (Siri).
  - Per-action plan note surviving trims, saved modules with regression tests (Hansolo).
  - Logprob uncertainty feedback (Wang).
- Python tool: persistent interpreter with `store` / `register_predictor` / `mechanics` (Wang).
- Prompts:
  - Removal of the "ignore the budget bar" lines and "level restarted" wording (Siri, Hansolo).
  - Scoring explanation and "act without certainty" (Hansolo; Wang off).
  - Game priors and turn protocol (Wang, Siri).
- Perception and actions:
  - Animation classifier with up to 16 frames (Hansolo).
  - `until=` chunks (Wang, off).
- Scheduling: UCB and exploration floor (Wang, Siri).
- Reliability: server watchdog/restart (Hansolo, Siri).

## C. Top 8 missing from Franzen's run, ranked by likely score impact

1. **Removing the budget-bar "ignore it" lines (`prompts.py:139,378`).** Siri deleted both after finding that 46% of actions went into attempts that ended in game over (M102 notes). Franzen's own BAR RULE contradicts these lines only after a death. Cheap, a prompt-only change.
2. **Solved-level memory pinned in the system prompt.** Siri's M85 pins the rule, the last 30 winning actions and the reasoning for the 3 newest levels. This survives the 131k→58k trim, which currently drops the evidence for earlier levels; level transfer is where depth-weighted score comes from.
3. **Strategy audit when stuck.** Hansolo's is time-based at 25% of the budget. Our token-based port scored 44.10 vs a 46.25 control at 60k (single run, within noise). Try it at a different threshold or time-based before dismissing it.
4. **More concurrent streams or a host KV tier.** Siri runs 16 streams with a 48 GB hicache; Hansolo runs 14. The writeups say throughput is still the binding constraint, since scores are still rising at the deadline. Our streams14 arm is queued.
5. **Scoring explanation plus "full certainty is not required to act" (Hansolo `prompts.py:56-57,156`).** Wang's spend-actions arm had the best trimmed mean in its batch (8.43 vs ~6.2).
6. **Per-action plan notes or a ledger that survives trimming.** Hansolo keeps a ≤500-char `plan` per action; Wang has the tag ledger. Franzen loses everything older than the drain point. Our `ARC3_LEDGER_ON_TRIM` arm targets this.
7. **Saved, tested Python modules (Hansolo)** instead of untested retained functions. The tests guard against reusing a wrong planner after a level change. Medium effort, unproven.
8. **Server watchdog/restart (Hansolo, Siri).** No score gain, but it protects a rerun from a 0. Our fz-arms v1/v2 hung at startup.

Lower priority:
- **Temperature 0.6** (Hansolo, Siri) is already tested: 45.92 vs 46.25, a null result.
- **UCB** is replaced by Franzen's gate.
- **The logprob uncertainty trace and the persistent interpreter** were judged low value: there is a measured 37 ms spawn cost, and they duplicate retained functions.

## Other public notebooks
- No other public Duck derivative is both distinct and above 20.
- Most are Franzen copies:
  - `amatlas/dfranzen-fork` and the ctx-port/pf-unit forks.
  - `obstacledeveloper/...duck-sota`.
  - `skarin/...27-80-lb`, which reproduces 27.80 and gives a 20-pass SD of 0.459.
- `yukailiu/...champion-solution` hard-codes plans for the public games.
- The leaderboard now has teams at 33-52 (Tufa 52.51), but their code is not public.
