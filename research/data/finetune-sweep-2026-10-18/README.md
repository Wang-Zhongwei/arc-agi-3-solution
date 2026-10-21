# Fine-tuning research sweep, 2026-10-18 (agent notes)

Inputs to [`../../reports/finetune-sweep-2026-10-18.md`](../../reports/finetune-sweep-2026-10-18.md). Each research note
ends with an independent fact-check; where it corrects the body, the correction wins. Start with `_evidence-base.md` and
`_plan-final.md`.

**Consolidation and plans**
- `_evidence-base.md`: all 23 research notes merged, with fact-check corrections applied; settled contradictions and open questions.
- `_plan-final.md`: the final H1/H2 plan (decisions, Day-0/Day-1 checklist, gate ladder, calendar, costs, pick rule), revised after three red-team passes.
- `_critic.md`: completeness critic of round 1: gaps, contradictions, strongest signals, and the questions the G-notes answer.
- `plan-efficiency-first.md`: plan draft through the efficiency lens (levers ranked by LB per GPU-hour).
- `plan-eval-centric.md`: plan draft through the measurement lens (build the evaluation machine first).
- `plan-risk-first.md`: plan draft through the deadline-realist lens; the judges' top pick and the final plan's base.
- `plan-tufa-imitation.md`: plan draft through the max-upside lens (game factory, rollout fleet, RL burst).

**Round 1: local audits**
- `L1-data-inventory.md`: what our 85 logged runs contain, what can be rebuilt exactly, context lengths, decision counts, tooling and throughput.
- `L2-competitors.md`: competitor code, public notebooks and model artifacts; who trains; what Tufa and Chen have said.
- `L3-headroom.md`: exact scoring, what tokens/action and extra levels are worth offline and on the LB, and the noise model.

**Round 1: literature and web**
- `W1-sft-pedagogy.md`: SFT of an RL-tuned reasoner: on-policy data, forgetting, LoRA targets and LR, agent SFT mechanics.
- `W2-rl-fundamentals.md`: RL algorithms (GRPO, GSPO, CISPO), MoE routing replay, train–inference mismatch, rewards, compute.
- `W3-agentic-game-rl.md`: training LLM agents on games, credit assignment through trims, environment diversity, ARC-AGI-3-specific work.
- `W4-lab-pipelines.md`: post-training pipelines at Qwen, DeepSeek, Kimi, GLM, MiniMax, NVIDIA and others, plus small-team case studies.
- `W5-arc-lineage.md`: ARC-AGI-1/2 prize methods, ARC-AGI-3 format and scoring, the milestones, and evidence on fine-tuning for ARC-AGI-3.
- `W5b-kaggle-intel.md`: Kaggle discussion threads on fine-tuning, the leaderboard, public Flash-Next models, and the rules.
- `W6-toolchain-infra.md`: trainers that support Flash-Next, kernels, merging without re-quant, serving, Azure SKUs and prices.
- `W7-efficient-reasoning.md`: training for shorter thinking (SFT and RL methods, agent specifics) and a local state-lab measurement.
- `W8-exploration-worldmodel.md`: training exploration, world-model learning, goal inference and novelty bonuses.
- `W9-env-generation.md`: synthetic game generation, how many environments transfer needs, curricula, ARC-likeness checks, cost.
- `W10-distillation-context.md`: distillation routes: teachers, on-policy and self-distillation, frontier APIs, context distillation, TTT.

**Round 2: gap notes**
- `G-A-token-tail-stalls.md`: where the long-turn token tail sits, what R3 can reach, a 30-stall audit, and sandbox-error costs.
- `G-B-eval-design.md`: evaluation protocol: subset noise, the H10/T15 split, external validation games, LB draw allocation, canary.
- `G-C-render-logprobs.md`: training-render parity with the server, history normalisation, the verbatim-history arms, and logprobs on our SGLang build.
- `G-D-variants.md`: existing post-trained Flash-Next variants, which tensors they changed, and how to overlay them on our W4A16.
- `G-E-mtp-head.md`: R1 (retraining the MTP draft head): published evidence, our accept→throughput slope, recipe, and the post-LoRA MTP check.
- `G-F-intel-rules.md`: hidden-set composition, account and team rules (exact quotes), and top-team intel.
- `G-G-deep-reads.md`: full-text reads of the key cautions (Arora, OPSD, AWM, SDPO, ...) and verdicts on five round-1 contradictions.
- `G-H-elasticity.md`: how score responds to effective compute, sim2's errors, what a token cut is worth, and the cheapest upward test.
