# Training techniques for LLM reasoning / agentic play via games: literature check (2026-10-12)

Method: every arXiv id below was resolved through the arXiv API (export.arxiv.org) for title, date and authors, and numbers were read from the abstract or the downloaded PDF text. Where a number came only from a summarizing fetch tool, or could not be read, it is marked UNVERIFIED. Note: the WebFetch summarizer hallucinated on PDFs (it said SPIRAL used Qwen3-32B and LoRA; the PDF says Qwen3-4B-Base, full-parameter, 8xH100). Treat only the entries below as checked.

Dead ends already in arms.md that I did not re-propose: prompt-side thinking guides, scratchpads, expert-count reduction, DFlash2, model swaps. Not proposed before 11-02: online multi-turn GRPO, full-parameter training, test-time training.

Cross-cutting facts for our setup (Qwen3.8-Flash-Next, LoRA on non-expert BF16 modules only, routed experts frozen int4):
- The measured loss is goal inference plus wall-clock. Almost none of the game-RL papers below measure either; they measure transfer to math or to in-distribution game win-rate. Treat "games make models reason better" as weak evidence for us.
- The two things with direct evidence for agent tuning on interactive tasks are rejection-sampling fine-tuning (RFT/expert iteration) and length-controlled training.

---

## A. Self-play / game RL that transfers to general reasoning

1. **SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn RL** - arXiv 2506.24119, 2025-06-30. Bo Liu, Leon Guertler, Simon Yu, Zichen Liu et al. (NUS/Sea AI/ TextArena group; 12 authors). VERIFIED (API + PDF).
   - Models: Qwen3-4B-Base mainly (also Llama/DeepSeek-distill variants, up to 4 models). Full-parameter, online, 400 steps x 128 samples, 8xH100, about 25 h per run, lr 1e-6, temp 1.0.
   - Result: multi-game (TicTacToe, Kuhn Poker, Simple Negotiation) training lifts 8-benchmark average 34.0 -> 44.5 (+10.5); Kuhn-only +8.7ish on math; beats SFT on 25k expert trajectories. Without role-conditioned advantage estimation (RAE) the model abandons reasoning after about 200 steps (empty thinking traces).
   - Transfer to us: gains are on math/general QA, not on playing novel games. Needs online two-player rollouts, which we cannot run on this MoE in 3 weeks. Useful only as a warning: multi-turn RL collapses thinking unless advantage normalization is per-role/per-turn. Risk high, payoff for ARC unproven. Do not pursue.

2. **Play to Generalize: Learning to Reason Through Game Play (ViGaL)** - arXiv 2506.08011, 2025-06-09. Yunfei Xie, Yinsong Ma, Shiyi Lan, Alan Yuille et al. (6 authors). VERIFIED.
   - Qwen2.5-VL-7B, RL on Snake and Rotation, 6xA100-80G, lr 1e-6, batch 128. Average accuracy gain on math benchmarks +2.9% (vs +2.4% for RL on 12K math problems); also beats specialists on MMMU series by 5.4% vs R1-OneVision-7B.
   - Transfer: small gains (+2 to 3 points) on static benchmarks; no evidence for goal inference in new games. Skip.

3. **Game-RL / Code2Logic (GameQA)** - arXiv 2505.13886, 2025-05-20. Jingqi Tong et al. (24 authors). VERIFIED (abstract). RL only on GameQA (30 games, 158 tasks) improves VLMs on 7 vision-language benchmarks. Transfer: confirms procedural, code-verifiable game data is a cheap RL data source (same idea as our generated games); numbers not read. Low priority.

4. **MARSHAL: Multi-Agent Reasoning via Self-Play with Strategic LLMs** - arXiv 2510.15414, 2025-10-17. Huining Yuan, Zelai Xu et al. (13 authors). VERIFIED (abstract). Qwen3-4B; turn-level advantage estimator plus agent-specific advantage normalization; up to +28.7% on held-out games and up to +10.0% AIME, +7.6% GPQA-Diamond when used inside multi-agent systems. Transfer: the turn-level advantage idea is what we would need if we ever do multi-turn RL; not feasible by 11-02.

5. **Reasoning Gym** (2505.24760, 2025-05-30, Stojanovski et al., VERIFIED abstract), **Enigmata** (2505.19914, Jiangjie Chen et al., VERIFIED abstract), **SynLogic** (2505.19641, Junteng Liu et al., 35 tasks, 7B and 32B models; beats DeepSeek-R1-Distill-Qwen-32B by 6 points on BBEH, VERIFIED abstract), **Logic-RL** (2502.14768, 7B, 5K logic problems generalize to AIME/AMC, VERIFIED abstract), **TextArena** (2504.11442, 57+ text games, VERIFIED abstract). Procedural verifiable puzzle suites. Transfer: confirm the "generated environments + verifiable reward" recipe; none targets unknown-goal discovery. Only relevant as environment/tooling source.

6. **Thinking by Doing: WMAct (world-model reasoning via multi-turn interaction)** - arXiv 2511.23476, 2025-11-28. Bao Shu, Yan Cai, Jianjian Sun, Chunrui Han et al. (14 authors). VERIFIED (abstract); numbers from html fetch (moderately reliable).
   - Qwen3-8B (own variant). Reward rescaling by action efficacy plus interaction-frequency annealing (shrinks allowed turns). Sokoban standard 64.2 (PPO-interactive) -> 73.7 (+rescaling) -> 78.6 (+annealing); Hard-2 0.35 -> 49.9 vs a plan-in-one-shot baseline. Transfer to HMMT25 +5.05, GPQA-D +2.24, AIME24 +4.97 (UNVERIFIED beyond the fetch).
   - Transfer to us: the efficiency-scaled reward matches our RHAE-style scoring (levels weighted by actions). We can use it as a rollout-selection rule in expert iteration (prefer fewer actions for the same levels) without RL. Risk: they train online PPO.

7. **Absolute Zero** (2505.03335, Andrew Zhao et al., VERIFIED abstract): proposer/solver self-play with a code executor, no external data, SOTA among zero-setting models on code and math. Transfer: the proposer-solver idea is analogous to our game generator; not directly trainable in 3 weeks.

---

## B. Multi-turn agent RL and credit assignment

8. **RAGEN / StarPO** - arXiv 2504.20073, 2025-04-24. Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang et al. (18 authors). VERIFIED (abstract + PDF).
   - Qwen2.5-Instruct 0.5B (Bandit, Sokoban, FrozenLake) and 3B (WebShop), H100s, 100-200 rollout-update steps. "Echo Trap": reward variance collapse, entropy drop, gradient spikes; PPO collapses later than GRPO. StarPO-S: filter trajectories by reward variance, critic baseline, decoupled clipping. Reasoning does not emerge in multi-turn RL without reasoning-aware reward.
   - Transfer: variance-based prompt/trajectory filtering is the same as "drop zero-variance games" in our M2 plan, and has independent evidence. Small models only; no MoE.

9. **GiGPO: Group-in-Group Policy Optimization** - arXiv 2505.10978, 2025-05-16. Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An. VERIFIED (abstract + PDF).
   - Qwen2.5-1.5B/3B/7B-Instruct. Anchor-state grouping gives step-level advantages with no extra rollouts or critic. >12% over GRPO on ALFWorld, >9% on WebShop, same GPU memory.
   - Transfer: requires repeated identical states across trajectories. ARC games reset to identical initial boards and have many re-visited states, so anchor grouping is plausible, but it is multi-turn online RL. Defer past 11-02.

10. **ArCHer: hierarchical multi-turn RL** - arXiv 2402.19446, 2024-02-29 (older than the preferred window; kept as the origin of turn-level value + token-level policy). Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine et al. VERIFIED (abstract). Off-policy utterance-level critic plus token-level actor. Transfer: needs a critic; not for 3 weeks.

11. **Turn-level reward / credit assignment for multi-turn agents** - arXiv 2505.11821, 2025-05-17. Quan Wei, Siliang Zeng et al. (12 authors). VERIFIED (abstract). Dense per-turn rewards beat terminal or delayed rewards for both GRPO and PPO on search and game agent tasks.
    **Caveat paper: Multi-turn RL for tool-calling agents with Iterative Reward Calibration** - arXiv 2604.02869, 2026-04-03, Modecrua et al. VERIFIED (abstract): naive dense per-turn rewards DEGRADE performance by up to 14 points; calibrated rewards plus GTPO give Qwen3-30B-A3B 58.0 -> 69.5 on Tau-Bench airline (+11.5), Qwen3.5-4B +2.9. Transfer: if we add shaping (e.g., "board changed"), it must be validated against outcome; the Stanford CS224R project (not arXiv, found via search) used exactly such a dense reward.

12. **SimpleTIR** - arXiv 2509.02479, 2025-09-02. Zhenghai Xue, Longtao Zheng, Qian Liu et al. VERIFIED (abstract). Multi-turn instability comes from tool feedback shifting the token distribution; filter trajectories containing "void turns" (no code block / no answer). Qwen2.5-7B AIME24 22.1 -> 50.5. Transfer: our harness has tool-call failures (unknown tool calls silently dropped); filtering malformed-turn trajectories from the SFT set is a free analogue.

13. **AgentGym-RL / ScalingInter-RL** - arXiv 2509.08755, 2025-09-10. Zhiheng Xi, Jixuan Huang et al. (23 authors). VERIFIED (abstract). Curriculum on horizon length (few interactions first, then more) stabilizes long-horizon RL; agents match or beat commercial models on 27 tasks (number from abstract truncated; UNVERIFIED). Transfer: horizon curriculum maps to our "short budget first" curriculum on generated games.

14. **LOOP: RL for Long-Horizon Interactive LLM Agents** - arXiv 2502.01600, 2025-02-03. Kevin Chen, Marco Cusumano-Towner et al. (7 authors; Apple). VERIFIED (abstract). Critic-free PPO variant, one copy of the LLM in memory; 32B agent beats o1 by 9 points (15% relative) on AppWorld. Transfer: shows memory-efficient on-policy agent RL is possible at 32B; the agent learns to consult docs and avoid unwarranted assumptions (relevant to goal inference), but it ran on dense clusters.

15. **EPO, T^2PO, SGE (exploration fixes for multi-turn RL)**
    - EPO: arXiv 2509.22576, 2025-09-26, Wujiang Xu et al. VERIFIED (abstract). Entropy regularization with smoothing and phase weighting; up to +152% ScienceWorld, +19.8% ALFWorld.
    - T^2PO: arXiv 2605.02178, 2026-05-04, Haixin Wang et al. VERIFIED (abstract). Token-level "thinking intervention" when uncertainty stops changing; turn-level resampling of no-progress turns; improves WebShop/ALFWorld/SearchQA stability.
    - SGE (Strategy-Guided Exploration): arXiv 2603.02045, 2026-03-02, Andrew Szot, Michael Kirchhof, Omar Attia, Alexander Toshev (Apple). VERIFIED (abstract). Explore in space of language strategies (strategy first, then actions); solves tasks too hard for the base model.
    - Transfer: SGE is the closest match to our failure ("96% of stuck decisions had an untried action"): force a hypothesis-then-act structure. But arms.md notes Flash-Next ignores optional structure; a required strategy field trained in via SFT might stick where prompting did not. Medium.

16. **On Information Self-Locking in RL for Active Reasoning (AREW)** - arXiv 2603.12109, 2026-03-12. Deyu Zou, Yongqiang Chen, Fan Feng, Mufei Li et al. (7 authors). VERIFIED (abstract + PDF).
    - Outcome-only RL on active-reasoning tasks makes agents stop eliciting informative observations and stop updating beliefs (action selection and belief tracking lock each other). Advantage reweighting with cheap directional critiques gives up to +60 points over 9 tasks.
    - Transfer: this is the closest published description of our bottleneck (not trying new actions, not using evidence). It argues that outcome-only reward (levels cleared) alone can reinforce conservative behavior, so RFT data should be audited for action diversity and the selection rule should reward information-gaining actions. Cheap to analyze offline.

---

## C. GRPO-family fixes and MoE stability

17. **DAPO** - arXiv 2503.14476, 2025-03-18. Qiying Yu, Zheng Zhang et al. (35 authors; ByteDance Seed/Tsinghua). VERIFIED (abstract + PDF). Qwen2.5-32B base, 50 AIME24 vs 30 for naive GRPO and 47 for DeepSeek R1-Zero. Four tricks: Clip-Higher, dynamic sampling (drop all-correct / all-wrong groups), token-level loss, overlong reward shaping. Transfer: all four are config flags in verl/ms-swift-type trainers and are what we want for single-turn world-model GRPO at small batch. Low cost, solid evidence (though on a dense 32B).

18. **Understanding R1-Zero-Like Training (Dr. GRPO)** - arXiv 2503.20783, 2025-03-26. Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi et al. (8 authors; Sea AI Lab). VERIFIED. GRPO's per-sequence length normalization and std normalization inflate length, especially of wrong answers; Dr. GRPO removes them. 7B base, 43.3% AIME24. Transfer: directly relevant since our binding constraint is length; use Dr. GRPO normalization from the start.

19. **GSPO: Group Sequence Policy Optimization** - arXiv 2507.18071, 2025-07-24. Chujie Zheng, Shixuan Liu et al. (12 authors; Qwen team). VERIFIED (abstract + PDF).
    - Sequence-level importance ratio and clipping. On Qwen3-30B-A3B-Base cold-start (48 layers), about 10% of activated experts for the same response change after each gradient update, making token-level ratios noisy; GRPO needs Routing Replay to converge on MoE, GSPO does not. Used in Qwen3 models.
    - Transfer: strongest MoE argument. With LoRA on attention/GDN only, routing still shifts via hidden states. If we run GRPO use GSPO or on-policy-only updates.

20. **Rollout Routing Replay (R3)** - arXiv 2510.11370, 2025-10-13. Wenhan Ma, Hailin Zhang, Liang Zhao, Yifan Song et al. (7 authors). VERIFIED. Records the inference engine's routing and replays it in training; cuts training-inference KL; beats GSPO and TIS on Qwen3-30B-A3B. Transfer: needs SGLang to export routing ids and the trainer to accept them; Flash-Next with 512 experts and a custom engine makes this uncertain (UNVERIFIED that SGLang/Pennyroyal exports routing). Prefer measuring KL first.

21. **Stabilizing RL with LLMs: Formulation and Practices** - arXiv 2512.01374, 2025-12-01. Chujie Zheng, Kai Dang, Bowen Yu, Mingze Li et al. (13 authors; Qwen). VERIFIED (abstract + PDF).
    - 30B MoE (Qwen3-30B-A3B), hundreds of thousands of GPU hours. For strictly on-policy training, basic policy gradient with importance-sampling correction is most stable. Clipping plus Routing Replay becomes essential only once off-policy updates are used. After stabilization, final performance is independent of cold-start init.
    - Transfer: gives a decision rule for our M3: run one gradient step per fresh batch (fully on-policy), then neither clipping tricks nor routing replay is required, which suits a tiny budget. Best single MoE-RL reference.

22. **Towards Stable and Effective RL for MoE (RSPO)** - arXiv 2510.23027, 2025-10-27. Di Zhang, Xun Wu, Shaohan Huang, Furu Wei et al. (9 authors; MSRA). VERIFIED (abstract + PDF). Router-shift-ratio reweighting; tested on Qwen2.5-MoE (Countdown) and MATH/code; better and more stable than GSPO/GMPO. Transfer: small-scale evidence only; alternative to R3 when routing cannot be replayed.

23. **A Minimalist Approach: From Rejection Sampling to Reinforce (RAFT++ / Reinforce-Rej)** - arXiv 2504.11343, 2025-04-15. Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang et al. (11 authors). VERIFIED (abstract). RAFT (train only on positive samples) is competitive with GRPO/PPO; GRPO's advantage comes mostly from discarding all-wrong prompts, not reward normalization. Reinforce-Rej also filters all-correct prompts. Transfer: strongest argument that our M2 (expert iteration with group filtering) captures most of what GRPO gives. Raise M2's priority, and treat M3 as optional.

---

## D. Shorter / budget-aware reasoning (our binding constraint)

24. **ThinkPrune** - arXiv 2504.01296, 2025-04-02. Bairu Hou, Yang Zhang et al. (7 authors; UCSB). VERIFIED. RL with a hard token limit (over-limit = zero reward), iterative tightening; R1-Distill-Qwen-1.5B reasoning length halved with 2% AIME24 drop. Transfer: truncation reward = simplest scheme; for us it maps to the 150k-token rollout budget in M2 (reward is only counted within budget).

25. **DLER: Doing Length pEnalty Right** - arXiv 2510.15110, 2025-10-16. Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao et al. (12 authors; NVIDIA). VERIFIED (abstract). Truncation penalty works if RL is done right: batch-wise reward normalization, higher clip, dynamic sampling. Cuts output length >70% on R1-7B with accuracy at or above baselines; also an update-selective merging method that keeps accuracy while retaining brevity. Transfer: the recipe overlaps DAPO; the merging trick is relevant because we merge LoRA anyway.

26. **Training Language Models to Reason Efficiently** - arXiv 2502.04463, 2025-02-06. Daman Arora, Andrea Zanette. VERIFIED. R1-Distill 1.5B/7B; a single hyperparameter alpha trades accuracy for length; for the 7B, 50% fewer tokens with <5% accuracy loss. Transfer: simple objective, adaptable as "levels per token" selection.

27. **L1 / LCPO** - arXiv 2503.04697, 2025-03-06. Pranjal Aggarwal, Sean Welleck (CMU). VERIFIED (abstract). Length-controlled RL: the prompt states a token budget and the model obeys; 1.5B L1 beats GPT-4o at equal length. Transfer: prompts already carry a time budget in our harness; budget-conditioned behavior could be taught but needs RL.

28. **ShorterBetter** - arXiv 2504.21370, 2025-04-30. Jingyang Yi, Jiazheng Wang, Sida Li. VERIFIED (abstract). Reward the length of the shortest correct sample in the group ("Sample Optimal Length"); 50-80% shorter outputs on R1-Distill 1.5B/7B at similar accuracy. Transfer: this is directly implementable as a rollout-selection rule in expert iteration (pick the shortest successful rollout per game), no RL needed.

29. **O1-Pruner** (2501.12570, Jan 2025, VERIFIED abstract), **Kimi k1.5 long2short** (2501.12599, VERIFIED abstract; long2short methods reach 60.8 AIME, 94.6 MATH500 with short CoT), **Draft-Thinking** (2603.00578, 2026-02-28, Jie Cao et al.; 82.6% budget reduction at 2.6% MATH500 drop, VERIFIED abstract). All math-only evidence.

30. **Ares: Adaptive Reasoning Effort Selection for LLM Agents** - arXiv 2603.07915, 2026-03-09. Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao et al. VERIFIED (abstract). A small router trained to pick the minimum reasoning level per agent step; up to 52.7% fewer reasoning tokens with minimal success loss on Tau-Bench, BrowseComp-Plus, WebArena. Notes static low effort at every step degrades performance (consistent with our arms.md static-effort test). Transfer: closest agent-specific result to our wall-clock problem; a LoRA/classifier router that sets `reasoning_effort` per step is trainable in days from our own transcripts (label = cheapest effort that still passes). Needs per-step labels, which means re-running steps at several efforts.

---

## E. Rejection sampling / expert iteration / distillation on interactive tasks

31. **Exploring Expert Failures Improves LLM Agent Tuning (EEF)** - arXiv 2504.13145, 2025-04-17. Li-Cheng Lan, Andrew Bai, Minhao Cheng, Cho-Jui Hsieh. VERIFIED (abstract). RFT favors easy subtasks and stalls; mining beneficial actions from failed expert trajectories solves previously unsolvable subtasks. WebShop 62% vs RFT 53.6% vs GPT-4 35.6%. Transfer: our levels-never-cleared are the hard tail RFT will never see; add a "partial-progress" segment from failed rollouts (the prefix up to the last level-up, as M2 already does) and, if available, hints from a stronger teacher run.

32. **ReST-EM (Beyond Human Data)** - arXiv 2312.06585, 2023-12-11 (older; foundational). Avi Singh, John Co-Reyes et al. (DeepMind; PaLM-2). VERIFIED (abstract). Generate, filter by binary reward, fine-tune, repeat; scales with model size and beats human-data SFT on MATH/APPS. Transfer: justifies the 2-3 round plan; note the original refits from the base model each round to limit drift.

33. **SWE-smith** - arXiv 2504.21798, 2025-04-30. John Yang, Kilian Lieret et al. VERIFIED (abstract). 50k synthetic task instances from 128 repos; SWE-agent-LM-32B 40.2% pass@1 on SWE-bench Verified via trajectory SFT (rejection sampling with Claude 3.7). Transfer: synthetic environments at scale plus teacher trajectories are the scalable route; GAMES version = generated games + best-available teacher traces.

34. **Self-Challenging Language Model Agents** - arXiv 2506.01716, 2025-06-02. Yifei Zhou, Sergey Levine, Jason Weston, Xian Li. VERIFIED (abstract). Agent generates tasks with a verifier (Code-as-Task) then trains with RL on its own tasks; >2x improvement for Llama-3.1-8B on M3ToolEval and Tau-Bench with only self-generated data. Transfer: supports using model-generated games, but our generator quality (DreamTeam games) is the risk.

35. **Agent Distillation** - arXiv 2505.17612, 2025-05-23. Minki Kang, Jongwon Jeong et al. VERIFIED (abstract). Distilling whole agent trajectories (with first-thought prefix) from a teacher into 0.5B-3B models; matches next-tier larger CoT-distilled models. Transfer: teacher traces from frontier models cost API spend (OpenRouter credit was $0); a 125B/6B student does not need it unless teacher traces beat self-generated ones.

36. **TRACE: Training Reasoning Agents for Causal Exploration with Synthesized Rewards** - arXiv 2609.10315, 2026-09-09. Rui Sun, Zhan Shi, Bing He. VERIFIED (abstract). Qwen3.5-35B-A3B (a MoE): SFT 0.159 -> 0.637, then RL with simulator-derived rewards -> 0.757, above prompted Claude Opus 5 (0.686) and prompted Qwen3.5-122B-A10B; fewer tool calls. Transfer: the nearest recent example of the exact pipeline we plan (SFT on traces, then RL, on a small-active MoE, with a simulator reward that is the oracle). Domain is ad diagnostics, not games; the SFT stage gave most of the gain. Compute not read (UNVERIFIED).

---

## F. World-model / next-state prediction as objective

37. **Agent Learning via Early Experience** - arXiv 2510.08558, 2025-10-09. Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue et al. (30 authors; Meta/OSU). VERIFIED (abstract + PDF table).
    - Llama-3.2-3B, Qwen-2.5-7B, Llama-3.1-8B on 8 environments. Implicit world modeling (predict the next state for the agent's own non-expert actions, as auxiliary SFT) and self-reflection beat imitation. Example ALFWorld (3B): IL 78.1 -> implicit world modeling 83.6 (+5.5), self-reflection 85.9 (+7.8); TravelPlanner gains +9 to +15; also a better starting point for later RL.
    - Transfer: cheapest world-model use for us, as an auxiliary in the SFT mix (examples: [board, action] -> next board/diff text) built from our own rollouts for free. Gains are 3-8 points on short-horizon text tasks; our rendering of 64x64 grids is far longer, so token cost per example is high.

38. **RLVR-World** - arXiv 2505.13934, 2025-05-20. Jialong Wu, Shaofeng Yin, Ningya Feng, Mingsheng Long (Tsinghua). VERIFIED (abstract); numbers via html fetch (UNVERIFIED in the PDF, which was image-heavy).
    - R1-Distill-Qwen 1.5B/7B as text-game world models: SFT 32.87% -> RLVR 57.0-63.2% (1.5B), 65.5% (7B); web page state F1 49.9 -> 65.1; hundreds of RLVR steps. Downstream: web-agent success 12.06 -> 14.29 (+18% relative) when the better world model is used for planning.
    - Transfer: strongest evidence that cell-exact verifiable rewards on transitions work in a few hundred steps, which supports M3's reward. But the downstream acting gain is small (+2.2 points absolute), matching the arms.md caution that better replay need not give better decisions.

39. **Dyna-Think** - arXiv 2506.00320, 2025-05-31. Xiao Yu, Baolin Peng, Ruize Xu, Michel Galley et al. (8 authors; Microsoft/Columbia). VERIFIED (abstract). Imitation of R1 reconstructed to include world-model simulation, then training of state prediction/critique generation then policy. On OSWorld/WindowsAgentArena matches R1 best-of-n with 2x fewer tokens; critique-generation training helps policy more than state prediction; agent success correlates with world-model quality. Transfer: gives a second reason to train short, simulate-then-act thinking (token savings), and a note that critique ("why was this action wrong") objectives beat raw state prediction.

40. **Code World Models for General Game Playing** - arXiv 2510.04542, 2025-10-06. Wolfgang Lehrach, Daniel Hennes, Miguel Lazaro-Gredilla, Xinghua Lou et al. (16 authors; Google DeepMind). VERIFIED (abstract). LLM turns rules + trajectories into executable Python world models (transition, legal moves, termination) used by MCTS; outperforms or matches Gemini 2.5 Pro in 9 of 10 games (4 novel; 5 imperfect-information). No weight training. Transfer: harness-side (the `step()` that M3 would train). It supports a `check_model()` tool plus search; not a training result.

41. **Benchmarking World-Model Learning with Environment-Level Queries (WorldTest / AutumnBench)** - arXiv 2510.19788, 2025-10-22. Archana Warrier, Dat Nguyen et al. (11 authors). VERIFIED (abstract). 43 interactive grid-worlds, 129 tasks, 517 humans vs five frontier models; humans win, gap attributed to exploration and belief updating. Transfer: evaluation only, but a held-out grid-world set for checking whether training transfers beyond our generated games.

---

## G. LoRA vs full fine-tuning for RL; MoE caveats

42. **LoRA Without Regret** (Thinking Machines blog, not arXiv) - https://thinkingmachines.ai/blog/lora/, 2025-09-29, John Schulman et al. VERIFIED (page fetched).
    - LoRA matches full FT on SFT when the dataset fits in capacity (about 1 bit/token), and matches it in policy-gradient RL even at rank 1 (about 1 bit per episode). Optimal LoRA lr about 10x full-FT lr (about 15x for short runs ~100 steps). Attention-only LoRA underperforms: apply to all matrices, especially MLP and MoE layers (rank-128 MLP-only beat rank-256 attention-only). LoRA tolerates large batch sizes less than full FT. Models: Llama 3 and Qwen3 including the 30B MoE; MATH/GSM8K/DeepMath-103K RL; Tulu3/OpenThoughts3 SFT. About 2/3 the FLOPs of full FT.
    - Transfer: encouraging for M3 (RL at low rank is fine) but a warning for M2: our LoRA can only cover attention, GDN, shared expert and (if we choose) the router; the routed expert MLPs (where the blog says capacity matters) are frozen int4. Expect weaker SFT fits than the blog; use rank >= 64 and the shared experts, LR 10-15x, small batch.

43. **LoRA Learns Less and Forgets Less** - arXiv 2405.09673, 2024-05-15 (older; kept for the SFT caveat). Dan Biderman, Jacob Portes et al. (12 authors). VERIFIED. In standard low-rank settings LoRA underperforms full FT on code and math SFT but forgets less; full FT learns perturbations with rank 10-100x higher than typical LoRA. Transfer: for knowledge-heavy SFT LoRA lags; for behavior shaping (our case) the "forgets less" side is a plus because we must keep the harness tool-call format and MTP acceptance.

44. **Tina: Tiny Reasoning Models via LoRA** - arXiv 2504.15777, 2025-04-22. Shangshang Wang, Julian Asilis et al. VERIFIED (abstract). LoRA during RL on a 1.5B model: 43.33% Pass@1 AIME24 for about $9; hypothesis that LoRA learns the format of rewarded reasoning while preserving knowledge. Transfer: consistent with the LoRA-RL claim; scale is tiny.

---

## H. ARC-AGI-3-specific and related items found (context)

- **ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence** - arXiv 2603.24621, 2026-03-24, ARC Prize Foundation. VERIFIED (abstract). Humans 100%, frontier AI below 1% as of March 2026. Describes efficiency-based scoring against human action baselines. No training results.
- **Sensi** - arXiv 2603.17683, 2026-03-18, Mohsen Arjmandi. VERIFIED (abstract). Test-time curriculum agent for ARC-AGI-3; 2 levels solved (v1), 0 (v2); failure is perception hallucination cascade. Harness-side; not training.
- Naestro-AGI3-27B (QLoRA on procedurally generated ARC-like games): non-arXiv model card; already in arms.md; no numbers published.
- Memento 3 (frozen LLM + rulebook + compiled world model, claims RHAE 100 on 25 public games): a news item from the search results, NOT verified, not a training paper.

## Items I did not verify or could not read
- Compute for TRACE, RLVR-World, ViGaL step counts, Dyna-Think, WMAct (GPUs): not read.
- Whether SGLang/Pennyroyal can export per-token routing ids for R3-style replay on Flash-Next: not checked.
- Whether ms-swift's GRPO supports GSPO / Dr. GRPO / dynamic sampling for this model: not checked; confirm in the ms-swift docs before M3.
- Any paper that trains an LLM specifically to infer goals in unknown-rule interactive grid games and reports a transfer number: I found none. This remains an open gap.

---

## Ranked top-8 "what we should try" for our setup (deadline 2026-11-02)

1. **Expert iteration with group filtering plus shortest-success selection (M2, strengthened).** Evidence: RAFT matches GRPO and the gain comes from discarding all-wrong prompts (2504.11343); ReST-EM (2312.06585); ShorterBetter's shortest-correct-sample rule (2504.21370); WMAct efficiency reward (2511.23476). Selection rule: highest levels cleared, then fewest tokens, then fewest actions; drop zero-variance games; truncate at last level-up. LoRA: all BF16 non-expert linears including shared experts, rank 64, lr about 10-15x the full-FT value, small batch (LoRA Without Regret). Cheapest first test: take existing logged rollouts, apply the selection rule offline, train one LoRA epoch, and run the 20-min panel; also report mean tokens per level.

2. **Length-aware selection as the efficiency lever (budget-in-training).** ThinkPrune/DLER truncation reward (2504.01296, 2510.15110) implemented as "only count levels within 150k tokens" at data-selection time, plus DLER-style update-selective merging of the LoRA delta. Cheapest test: compare mean tokens per request and score on the panel for M2 data chosen by (a) levels only vs (b) levels per token; the pass condition is at least 25% fewer tokens with no score drop.

3. **Per-step reasoning-effort router (Ares-style, 2603.07915).** Train a small classifier on our own transcripts to set `reasoning_effort` per step. Cheapest test: replay a stored run's steps at low/high effort on a subset (the harness has `ARC3_REASONING_EFFORT` hooks per research-sweep) to see how often low suffices; only train if at least 40% of steps pass at low.

4. **Audit RFT data for information self-locking (2603.12109) and add a strategy-first field (SGE, 2603.02045).** Our stuck decisions skip untried actions, exactly the SeL pattern. Cheapest test: offline, count untried-action rate in the rollouts selected for SFT vs in discarded ones; if the selected set is just as conservative, reweight toward rollouts that tried new actions before the level-up, and add a short required "hypothesis" line to the SFT targets (flash-next ignores optional structure, so make it required).

5. **Early-experience auxiliary: implicit world-model examples in the SFT mix (2510.08558).** [board, action] -> compact diff text from our own rollouts, mixed at about 10-20% with the M2 data. Cheapest test: A/B two LoRAs differing only by this auxiliary; compare held-out transition accuracy and the 20-min panel. Risk: longer sequences and the arms.md note that replay accuracy did not translate into decisions.

6. **MoE-safe single-turn GRPO for M3: strictly on-policy, Dr. GRPO normalization, DAPO dynamic sampling, GSPO if off-policy (2512.01374, 2503.20783, 2503.14476, 2507.18071).** Reward from RLVR-World-style cell-exact transitions (2505.13934). Cheapest first test (before any RL): measure training-vs-SGLang logprob KL and routing mismatch on 50 stored rollouts with the LoRA loaded; if KL is high, apply R3/RSPO or stay on-policy-with-one-step.

7. **Failure-mining in the hard tail (EEF-style, 2504.13145).** Include partial-progress segments from failed rollouts and, for games the model never clears, a teacher or longer-budget rollout. Cheapest test: count games with zero successes at 8 rollouts; run the longer-budget or teacher rollout on those only and see whether any level-ups appear that the base never produced.

8. **Dense-reward guard before any online RL (2604.02869, 2505.11821, 2504.20073).** If we ever add per-turn shaping (e.g. "board changed"), validate that shaping correlates with levels cleared; naive dense rewards cost up to 14 points. Cheapest test: offline correlation of the candidate shaping signal with eventual level-ups on logged rollouts, plus StarPO-S-style variance filtering as an Echo-Trap monitor (reward-variance and entropy curves) in any RL run.

Not recommended for this window: SPIRAL/MARSHAL-style self-play, ViGaL/Game-RL transfer from games to math, GiGPO/ArCHer multi-turn RL (no evidence they improve goal inference in novel games; need online multi-turn infrastructure on a 125B MoE).
