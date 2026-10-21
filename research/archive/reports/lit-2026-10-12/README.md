# Literature sweep, 2026-10-12

The user asked for "more papers related to AI like this, whether its new configs/things we can change to the model, new training techniques, etc." after Memento 3 (arXiv 2610.11794) and DreamTeam / Workspace Optimization (arXiv 2605.09650). Four researcher agents covered four angles; each verified every arXiv id against arxiv.org and judged transfer to our stack: Qwen3.8-Flash-Next W4A16, ~6B active, 121 min for 25 games, ~185k tokens per game, offline score ~53.

| File | Angle | Papers |
|---|---|---|
| `training-rl.md` | training through games: self-play, multi-turn agent RL, GRPO fixes for MoE, shorter reasoning, rejection sampling, world-model objectives, LoRA caveats | ~45 |
| `world-models.md` | agents that learn an unknown environment: ARC-AGI-3 papers, code world models, exploration, memory, small-model evidence | ~30 |
| `inference-configs.md` | model and serving levers: drafters, thinking control, MoE tricks, quantization, sampling, steering | ~35 |
| `game-evals.md` | game-reasoning benchmarks and eval statistics, for the `game-reasoning/` framework (`arm/game-reasoning`) | ~25 |

## What the four sweeps agree on

- **No small-model evidence for world models on ARC-AGI-3.** Every executable or written world-model result is frontier-only (Memento 3, Tycho, OPINE-World, Twin, DreamTeam). The one open-model data point, Qwen3.8-27B in a plain coding agent (2610.11450), scores 6.8–8.4, and its failures match ours: notes never revised, no script reuse across levels, a hard-coded constant carried into a level where it no longer holds. Agents ignore condensed experience and use raw experience (2601.22436), which fits our ~1% note rate versus the harness-written facts that help.
- **No paper trains goal inference in unknown-rule games and shows transfer.** Self-play and game RL (SPIRAL, ViGaL, MARSHAL) raise math scores, not play in new games, so they rank last for us.
- **Thinking length is the lever that fits both prongs.** Inference: margin or confidence stops (ThinkBrake, DEER) and reflection-token suppression (NoWait). Training: budget-aware selection (ThinkPrune, DLER), and a per-step effort router (Ares, up to 52.7% fewer reasoning tokens on agent tasks). This is the user's "more actions per unit of thinking", without lowering the thinking level (which lost).
- **Speed alone is not the limit.** Drafter and tree work transfers worst: no EAGLE-3, DSpark or DFlash2 drafter exists for Flash-Next; Pennyroyal's GDN verify supports only a linear chain; our accept length 2.67 is at the cookbook's ~2.5 for thinking output.

## Already tested by us (don't re-run as new)

- Relaxed acceptance at 0.5 (`accept05`): 46.77, lower (inference-configs item 6).
- Temperature 0.6 (`temp06`): 45.92, no effect. Higher temperatures (0.85, 1.0; Qwen recommends 1.0 for thinking, 2512.12895 reports looping at low temperature) are untested; `dyn-temp` is built (PR #154).
- Duplicate-experiment / dead-signature guard (world-models item 3): its gate is "build only if ≥10% of effort goes to repeated (state, action) pairs". On 16,331 recorded transitions from three runs, repeated (board, action) pairs are 517, 3.2% (`arms.md`, Memento 3 entry). Below the gate.

## Combined ranking

Harness / inference (each is one new factor on the leaders, per the no-repeat rule):
1. **NoWait reflection-token suppression** (2506.08343). Flag-only: `--enable-custom-logit-processor`, about −5 bias on "Wait/Hmm/Alternatively". First run also answers whether logit processors work under our MTP build (SGLang #25536: thinking budget bypassed under spec decoding).
2. **ThinkBrake / DEER early stop** (2510.00546, 2504.15895). Offline first: on ~30 saved pre-action thinking segments, the share of tokens after the point where the action was already fixed. Build only if >15%.
3. **Temperature 0.85 or 1.0**, one env change.
4. **Level-indexed raw memory with contradiction pruning** (world-models item 1). Offline first: how many carried facts are contradicted within 10 actions of a level change.
5. **No-LLM state-graph explorer** (2512.24156) run offline on never-cleared levels under a 4,000-action cap.

Training (model prong in `arms.md`; deadline 2026-11-02):
1. **M2 expert iteration**, with three selection rules: drop all-equal groups (2504.11343: RAFT matches GRPO); select by levels cleared, then fewest tokens; count only levels cleared inside a token budget (ThinkPrune / DLER).
2. **Per-step effort router** (Ares 2603.07915). Offline: replay stored steps at low effort; proceed only if ≥40% pass.
3. **Measure MoE routing mismatch** between ms-swift and SGLang on 50 rollouts before any RL (2512.01374; GSPO 2507.18071 if updates go off-policy).
4. **World-model auxiliary examples** in the SFT mix (Early Experience 2510.08558, ALFWorld +5.5; RLVR-World 2505.13934). A/B two LoRAs that differ only by it.
5. **Exploration-coverage reward** (Look Before You Leap 2605.16143: +2–4 on 4–8B models). Check first whether coverage on logged runs predicts clearing a level.
6. LoRA settings: attention-only LoRA underperforms ("LoRA Without Regret"); with the routed experts frozen int4, include the shared expert, rank ≥64, LR ~10× full fine-tune.

## Corrections made in review

- `game-evals.md` said DreamTeam ships no generated games. The paper doesn't describe them, but the NVIDIA/dream-team repo does (`environment_files_generated/`, 25 games, and `arc_agi_3/game_creator`). Corrected in place.
- `training-rl.md` notes that web-fetch summaries of PDFs were unreliable (one gave the wrong model and method for SPIRAL); its numbers come from abstracts or PDF text, with the rest marked UNVERIFIED.
