# Fine-tuning Flash-Next for ARC-AGI-3: research sweep and plan (2026-10-18)

**Request (the user, 2026-10-18):**
> do a complete research sweep on what we could do in regards to fine tuning and training qwen 3.8 flash next. basically,
> first go through just general training advice (like pedagogy on this, and what specifically to train on and whatever,
> you have a lot of freedom), then slowly hone in on what labs are doing, and then create an optimized plan for this
> benchmark

The report follows that order:
- Part 1: how fine-tuning works and what to train on;
- Part 2: what labs do;
- Part 3: what has been tried on agents, games, ARC and this competition;
- Part 4: our own numbers;
- Part 5: the plan;
- Part 6: what this changes in the 2026-10-17 plan ([Prior]);
- Part 7: risks and open questions.

**Method.**
- **Round 1: 14 notes.**
  - Three local audits: our data ([L1]), competitors ([L2]) and score headroom ([L3]).
  - Eleven literature and web notes: [W1]–[W10] and the Kaggle intel note [W5b].
- **Fact-checks.**
  - Each of the 22 research notes ends with an independent verifier's check against the primary source (arXiv full text,
    HF configs and safetensors headers, Kaggle pages, repo lines).
  - Where a check corrects the body, this report uses the correction. The corrections that change a decision are in
    Appendix A.
- **Critic and gap round.** A completeness critic ([Critic]) listed gaps and contradictions. Eight gap notes
  ([G-A]–[G-H]) answered them.
- **Consolidation and planning.**
  - One agent merged the 23 notes into an evidence base ([EB]).
  - Four planners wrote plans through different lenses: efficiency-first ([P-eff]), evaluation-centric ([P-eval]),
    risk-first ([P-risk]) and Tufa imitation ([P-tufa]).
  - Three judges ranked them, and two of the three put risk-first first.
  - The final plan ([Plan]) starts from risk-first, takes grafts from the other three, and was revised after three
    red-team passes.
- **Measured locally from existing logs (no new GPU runs):**
  - the token tail and a 30-stall audit over 85 runs ([G-A]);
  - 10-game subset noise and screen back-tests ([G-B]);
  - render parity, 200 of 200 requests, and the ms-swift training render, 25 of 25 windows ([G-C]);
  - safetensors diffs of the public Flash-Next variants, and a third party's Kaggle A/B logs ([G-D]);
  - the MTP accept-length → throughput slope from serve logs ([G-E]);
  - compute elasticity over 23 reference runs ([G-H]);
  - the corpus inventory ([L1]) and the headroom model on 26 runs ([L3]);
  - a per-level clear-rate count ([W3]) and a re-analysis of 700 state-lab samples ([W7]).
- **Not measured, so they are the plan's first gates:**
  - more compute than 121 minutes;
  - training at the deployed shape;
  - logprobs on our server;
  - Flash-Next on external games.
- **Conventions.** I did not re-run the fact-checks. UNVERIFIED marks a claim no primary source confirmed. [I] marks an
  inference or arithmetic, not a measurement. Note files are cited by short name; the links are at the end.

## Answer

1. **A consistent LB 60 is out of reach by 11-02.**
   - "Consistent 60" means a draw mean of about 64.5, since one draw has sd 3.50. We are at about 31 ([L3] FC 7).
   - Every lever we can ship by 11-02 is *projected* at +0.05 to +1.5 LB, from a compute curve that has been checked
     only downward ([Plan] §3.1; [G-H] FC 4–5).
   - Efficiency alone would need 2.7–6× effective compute ([G-H] Q4).
   - So 60 needs new capability: clearing levels with mechanics no current run solves. One extra level per hidden game
     is worth about +13–15 LB ([L3] Q2).
2. **Training pedagogy (Part 1).** Train on the model's **own** engine-verified outputs, **regenerated every round**;
   teacher text transfers badly to an RL-tuned model. Use LoRA on the BF16 non-expert modules at LR 1e-5 to 3e-5, on the
   **exact served context**. Pick checkpoints by λ-merges judged on held-out play, not loss. Generic replay protects
   only partly. RL on this MoE needs routing replay and 10⁴–10⁵ GPU-h at published scales, so not in H1.
3. **What labs do (Part 2).** Everything on-policy (own rejection samples, on-policy distillation, RL), training against
   the low-bit weights they deploy, with evaluation as the bottleneck. Qwen says that of Flash-Next itself.
4. **ARC and this competition (Part 3).** No public result shows a fine-tune helping on ARC-AGI-3, and none on
   Flash-Next. Every public notebook scoring ≥20 runs stock Flash-Next. Franzen won Milestone 2 untrained. Tufa (68.91)
   has described RL on generated games as a research pipeline, without saying its submission used it, and releases
   after the deadline.
5. **Our main training lever is smaller and riskier than the 10-17 plan assumed.**
   - R3, "same decision in fewer tokens", can reach only about 30% of leader-run tokens. Under its own safety rules it
     projects to about −4% of total tokens (at most −6.3%), worth about +0.3 to +1.0 LB ([Plan] §3.2).
   - Its closest published analogue collapsed: AIME 0.563 → 0.183 ([G-G] FC 1).
   - The retrained MTP drafter (R1) is the safer trained artifact: about +0.5–1.5 LB for under 15 GPU-h ([G-E] Q4).
6. **Deployment is cheap, which is new.** Intel's W4A16 checkpoint quantizes only the routed experts. A LoRA on the
   other modules merges into BF16 tensors with **no re-quantization** ([W6] FC 1; [G-D] §3).
7. **The plan.**
   - **D0 first:** write to Kaggle support about the second account (item 8).
   - **Today:** the elasticity test E1 (the leader at 161 minutes) and the throttle draw T½.
   - **Ship the near-lossless line:** harness hygiene fixes, the victory lap, any serving lever that benches ≥1%, and R1.
   - **The default rents no 8×H200 node before 11-02 (Branch B).** The one R3 LoRA round becomes the first step after
     11-02, unless every gate passes on 10-23 and you opt in.
   - **Cost:** Branch B $70–170 (cap $300) plus about 31 Kaggle GPU-h. Branch A $1.7–2.5k (cap $3k) ([Plan] §10).
   - **Odds at 11-02:** pick-1 LB mean low 28 / mid 30 / high 32.5. P(a fine-tuned trunk is truly better and is picked)
     ≈ 0.04. P(mean ≥ 60) < 1% ([Plan] §3).
8. **The largest single risk is the account rule.**
   - Rule 3.5.a: "You will be disqualified if you make Submissions through more than one Kaggle account."
   - snowiers has 8 competition submissions, the latest on 10-17 ([Plan] Answer 5; [G-F] G17).
   - Stopping now doesn't undo them, so write to Kaggle support today, before the 10-26 entry deadline.
9. **After 11-02 (H2)** the path to 60 runs through capability: an LB-proxy suite of ≥55 unseen games, iterated
   on-policy rejection-sampling rounds, breakthroughs on external games, then frozen-state RL with routing replay. About
   $15–45k over 3 months, P(consistent 60) ≈ 0.1, mostly conditional on a reproducible top-team release ([Plan] §12).

---

## Part 1. Training pedagogy: how fine-tuning works, what to train on, and why

Benchmarks named in this part (AIME, GPQA, IF-eval/IFEval, MMLU, MT-Bench, SWE-bench, Terminal-Bench, τ², BFCL, MCP,
WebShop, ALFWorld, Procgen) are public test suites for maths, science, instruction following, coding, tool use and agent
tasks. Higher is better.

### 1.1 Where we start: continued post-training of an RL-tuned MoE with no base checkpoint

- **Pre-training** teaches a model next-token prediction on raw text. **Post-training** (supervised fine-tuning or SFT,
  preference tuning, and reinforcement learning or RL; §1.2, §1.8, §1.9) shapes it into an assistant or agent.
  Flash-Next has had both. Our job would be a third stage on top.
- **Flash-Next's own post-training is unpublished.**
  - Its tech report covers architecture and pre-training only. It evaluates a "Qwen3.8-Flash-Next-Base" checkpoint (a
    saved set of weights) that was never released ([W4] FC 1; [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)).
  - Hugging Face lists only the post-trained model and its FP8 copy (weights stored as 8-bit floats) ([W2] §5).
  - So nothing can be redone "from base". Every change starts from an RL-tuned reasoning model, one that writes a
    *thinking* trace before each reply.
- **Architecture** ([W1] Q3 from the [HF config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json);
  [W6] §3, FC):
  - 48 layers, hidden size 2,560 (the width of the vector each token carries between layers);
  - 512 routed experts plus 1 shared expert per layer: a **mixture of experts (MoE)**. Each expert is a feed-forward
    block (an MLP, also called an FFN). A small **router** picks 10 of the 512 routed experts for each token, and the
    shared expert processes every token;
  - attention is a hybrid of Gated DeltaNet (GDN), a linear attention that keeps a fixed-size running state, and QSA, a
    sparse attention whose small **indexer** picks which earlier tokens each token attends to;
  - 180.0B parameters in BF16 (a 16-bit float format, 2 bytes per parameter: 360 GB), including a 51.2B PLE embedding
    table (a large hashed n-gram lookup) that we keep in host RAM. Routed experts are 120.8B, GDN 2.09B, QSA 0.60B,
    shared experts 0.24B.
- **What we serve** ([W6] FC 1; [Intel config](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/main/config.json)):
  - **Quantization** stores weights at lower precision to save memory. Intel's AutoRound W4A16 checkpoint stores 4-bit
    integer (int4) weights, with one scale per group of 128, and computes with 16-bit activations. It does this for the
    **routed experts of the 48 main (*trunk*) layers only**.
  - Attention, GDN, the shared expert, the router, the hyper-connections (learned mixing between 4 parallel copies of
    each token's 2,560-wide vector), the QSA indexer, the MTP layer (multi-token prediction: an extra layer, with its
    own 512 experts, that drafts upcoming tokens; §1.11), PLE, embeddings and vision all stay BF16.
  - This one fact sets the cheapest deploy path (§1.5).

### 1.2 SFT, and why on-policy data matters most

- **Supervised fine-tuning (SFT)** trains the model to raise the probability of chosen target replies, token by token
  (cross-entropy loss). **Loss masking** applies that loss only to chosen tokens, usually the assistant's reply, never
  the prompt or tool outputs.
- **SFT teaches whatever text it is given, including its style.**
  - **On-policy data** is text the current model itself produced.
  - **Off-policy data** is text from anyone else: a teacher model (a stronger model whose outputs are copied; §1.10), a
    human, a script, or an older version of the model.
  - Off-policy text pulls the model toward a different distribution, and that pull is what causes forgetting (losing
    skills that weren't in the training data; §1.6).
- **Rejection sampling (RS), also called rejection-sampling fine-tuning (RFT), STaR, ReST-EM or expert iteration:**
  sample k replies from the model, keep those a **verifier** (an automatic correctness check; for us, the game engine)
  accepts, SFT on them, and repeat with the new model.

**What the evidence says, ranked from safest to riskiest data** ([W1] Q1):

| Data | Evidence | Strength |
|---|---|---|
| Own samples, filtered, **regenerated each round** | Iterative-SFT shows "mild to no forgetting" ([Retaining by Doing §4.2](https://arxiv.org/abs/2510.18874)) | Strong; 1.5B and 7B only |
| Own samples, filtered **once** | Still forgets. Qwen-2.5-1.5B non-target drop: Self-SFT 3.0 / 14.0 / 19.5 vs GRPO (an RL method, §1.9) 0.6 / 0.5 / 0.9 (same paper, Table 2). Thinking Machines: SFT on Qwen3-32B's own unfiltered samples degraded IF-eval at any practical learning rate (LR) ([blog](https://thinkingmachines.ai/blog/on-policy-distillation/); [W1] FC 3) | Strong |
| Lab precedent | Qwen3 Stage 3 ran continual SFT on its own rejection samples "to ensure that the performance of the Stage 2 model is not compromised", and **still** lost AIME'24 83.8 → 81.9 ([2505.09388 §4.3](https://arxiv.org/html/2505.09388); [W1] FC 1) | Strong |
| Teacher or expert text | Orak SFT on GPT-4o trajectories (recorded sequences of play) barely moved held-out games (games kept out of training). SPIRAL's SFT stayed flat when data doubled. ViGaL Snake SFT went 49.1 → 47.2 while RL reached 62.3. Chu et al.: out-of-distribution (OOD: on tasks unlike the training data) accuracy 80.8 → 1.3 under SFT vs 91.8 under RL ([W3] §1, §5) | Strong pattern |

- **Why on-policy forgets less** [I]: relative to the model, a filtered own sample carries mainly the *selection*, about
  log₂ k bits per kept sample, not new text ([W1] Q1). That is also why its gains per round are modest.
- **The ablations credit on-policyness, not the KL penalty** (a loss term that keeps outputs close to the starting
  model's; §1.6) ([Retaining by Doing §4.1](https://arxiv.org/abs/2510.18874)).
- **A caveat for us.** Chu's "SFT first, then RL" holds only for *base* models that can't follow instructions
  ([W3] FC 2). Flash-Next needs no format warm start.

### 1.3 What to train on: curation, diversity and the right signal

- **Curation beats volume.**
  - LIMO: 800 curated samples reached a benchmark average of 78.1. An uncurated 100k set dropped it from 49.9 to 32.3
    ([2502.03387](https://arxiv.org/html/2502.03387); [W1] FC).
  - s1: random, longest-trace or diversity-only selection of 1,000 samples were "significantly worse" than joint
    selection ([2501.19393](https://arxiv.org/html/2501.19393)).
  - In this competition, a STaR LoRA author found "more trajectories at the same recipe made it worse"
    ([739047](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739047); [W5b] §1).
- **Diversity across environments matters more than count, and small pools overfit.**
  - Agent World Model (Qwen3-4B only): 10 training environments *degraded* held-out scores; 100 gave about +2–3 BFCL and
    0 on MCP ([2602.10090](https://arxiv.org/abs/2602.10090); [G-G] FC 6; [W9] FC 1).
  - At fixed compute, 30 coverage-selected environments gave +40.3% out-of-distribution vs +7.5% for all 170
    ([2608.03571](https://arxiv.org/html/2608.03571v1) Table 3; [W3] FC 6). The absolute numbers are small.
  - Procgen needed up to 10,000 levels to close the generalization gap for classic RL
    ([1912.01588](https://ar5iv.labs.arxiv.org/html/1912.01588)).
- **The right signal.**
  - *Early-experience* auxiliaries help: predicting the next state of the agent's own alternative actions, or explaining
    why the chosen action beat them. They gave +2.3 to +18.4 over imitation
    ([2510.08558](https://arxiv.org/html/2510.08558); [W1] Q6).
  - A *standalone* world-model objective (training the model to predict what the environment does next) does not
    improve decisions. In Meta's CWM ablation, execution traces left agentic SWE-bench at 18.6 → 18.4, while agentic
    trajectories raised it to 22.1 ([2510.02387](https://arxiv.org/html/2510.02387v1)). World-model SFT alone collapsed ALFWorld to 2.8
    ([2602.05842](https://arxiv.org/html/2602.05842v2); [W8] FC 2–3).
  - **Hindsight relabelling** (relabel a failed trajectory with the goal it did reach) beat success-only SFT by
    +7.6–11.4% ([2603.21357](https://arxiv.org/html/2603.21357v4)). Every method needs an explicit goal, and ARC-AGI-3
    goals are hidden ([W8] Q4).
- **For us** ([EB] §2.1): train on the model's own engine-verified decisions at frozen states (saved game positions
  where the model can be sampled again), and on mechanic-discovery breakthroughs from external games. Not on teacher
  text, scripted-solver rationales, world-model SFT or novelty rewards (bonuses for reaching unseen states).

### 1.4 Mechanics of agent SFT: render exactly what was served

- **Context and turns.** A multi-turn agent call is one long prompt: the system prompt, history, tool results and images,
  then one assistant reply. Flash-Next keeps the thinking of past turns in history by default ("especially beneficial
  for agent scenarios", [HF README](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/README.md); [W1] Q4).
  Franzen found that dropping it was "a large regression" ([W5] Q3).
- **Train on the exact token sequence the server saw.**
  - Our renders (the token sequences we rebuild from the logs) reproduce the server's token counts on 200 of 200
    logged requests ([G-C] Q1).
  - In ms-swift 4.6.0 (ModelScope's training framework, our planned trainer), the `qwen3_8` chat template (the rules
    that turn messages into tokens) matched 25 of 25 full logged windows token for token, after four exporter
    transforms, i.e. changes our log exporter makes ([G-C] Q1; [EB] §4.2).
- **Train on the served context length, not short windows.**
  - Agent World Model's history-limited model fell on τ² from 22.57 to 9.35 when evaluated with full history
    ([G-G] FC 6).
  - GLM-4.5 found that shorter multi-stage RL made the model "unlearn" long context "irreversibly"
    ([2508.06471 §3](https://arxiv.org/html/2508.06471)).
  - Our prompts have a median of 57,313 tokens; only 2.2% are ≤8k and 6.8% ≤16k ([L1] Q3, FC 3). The published recipe
    is measured only to 8k ([W6] FC 2).
- **Per-call examples vs packing.**
  - *Packing* puts several turns of one conversation into one training sequence, with loss on each.
  - Labs that keep thinking in history train each call on exactly the context it was served ([G-G] Q1).
  - Our logs store a *normalized* history (past thinking reformatted rather than kept verbatim) and no raw replies,
    token ids or logprobs (the log-probability the model gave each token) ([L1] Q2, FC 2). So behaviour cloning (SFT on
    logged replies) copies the policy, the model's current behaviour, rather than improving it, and new data should
    log raw replies ([G-C] Q2–Q3).
- **Mask what shouldn't be learned.** DeepSWE masked trajectories cut by the context limit or a timeout ("compact
  filtering", [DeepSWE card](https://huggingface.co/agentica-org/DeepSWE-Preview)), so the model isn't trained to run out
  of budget ([W4] §6).
- **Don't judge by training loss.** On-policy data starts at low loss, so loss barely moves while behaviour changes.
  Short runs can also underestimate generalization ([W1] Q5; [2604.06628](https://arxiv.org/html/2604.06628)).

### 1.5 LoRA: what it is, where to put it, and what LR

- **Full fine-tuning** updates every weight. **LoRA** freezes the weight W and learns a low-rank update B·A of rank r,
  scaled by α/r. The update can then be **merged** into W, so serving costs nothing extra.
- **LoRA learns less and forgets less.** It underperforms full fine-tuning on code and maths, and it preserves
  out-of-domain ability and output diversity better ([2405.09673](https://arxiv.org/abs/2405.09673)). Poor α and LR
  choices can still make it forget more ([2410.21228](https://arxiv.org/html/2410.21228)).
- **Where it goes matters** ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/); [W1] FC 4):
  - Attention-only LoRA underperforms MLP LoRA. The parameter-matched comparison is on Llama-3.1-8B; the same direction
    holds on Qwen3-30B-A3B-Base and in maths RL.
  - The best LoRA LR is about 10× the full-FT LR (fit 9.8), nearly independent of rank.
  - LoRA is less tolerant of large batches (many examples per update step).
  - Rank 1 matches full fine-tuning in RL.
- **Our constraint.**
  - A LoRA merges into our served checkpoint without re-quantization only if it touches BF16 modules: GDN, QSA, the
    shared expert and hyper-connections.
  - The trunk's only BF16 MLP is the shared expert: 1 of 11 active FFNs per token, about 4.9M parameters per layer.
    Shared-expert-only LoRA is untested anywhere ([W1] Q3; [W2] FC 7).
  - The router and the QSA indexer stay frozen (untrained); the indexer has no gradient (training signal) in any
    trainer ([W6] §2).
  - Routed-expert LoRA would force an int4 re-quantization ([W6] FC 1).
- **Learning rate** ([G-G] (d), FC 9):
  - Full-FT anchors are 5e-6 (Arora, 7B), 2e-6 (OPSD, 8B), 1e-6 (PAPRIKA SFT) and 7e-7 (Agent World Model).
  - ×10 gives a **LoRA SFT band of 1e-5 to 3e-5**. A DPO-style stage (§1.8) would sit near 2e-6. Global batch
    (examples per optimizer step) ≤32.
  - The ms-swift recipe's 1e-4 is a sweep ceiling, not a default.
- **Train against the served weights.** Build the trainer's base from Intel's **dequantized** int4 experts (the int4
  values converted back to BF16), not Qwen's BF16 experts. The LoRA then learns the deployed function, and the merge is
  exact [I]. Labs do the same with quantization-aware training (QAT), which trains against the low-bit weights
  (§2.1; [W4] §3).

### 1.6 Forgetting, and how to control it

- **Catastrophic forgetting** is losing skills that weren't in the training data. **KL divergence** measures how far
  the new model's output distribution has moved from the old one. A KL penalty is the classic guard.
- **The KL penalty is not what protects.**
  - GRPO at KL weight 0 forgot about as little as at 0.05; the on-policy data did the work
    ([Retaining by Doing](https://arxiv.org/abs/2510.18874)).
  - RL's Razor finds that forgetting tracks the KL on the new task ([2509.04259](https://arxiv.org/abs/2509.04259)).
    Retaining by Doing finds that correlation only moderate (0.52).
- **Replay** (mixing general data into training) **anchors only partly**
  ([TM on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/); [G-G] FC 10):
  - mid-training (continued training) on documents cut IF-eval from 85 to 45;
  - a 70/30 mix with the model's own chat outputs kept 79;
  - no mixing ratio kept the full 85, not even 100% chat;
  - on-policy distillation (§1.10) from the original model then lifted the 70/30 model to 83.
- **Weight interpolation is free with LoRA.**
  - Interpolating the fine-tuned weights back toward the start (WiSE-FT) "nearly fully" restores the pass@k (the
    chance that at least one of k samples is right) lost to reasoning SFT ([2504.10478](https://arxiv.org/abs/2504.10478)).
  - The AIMO-2 winner's α = 0.7 merge scored 69.1 vs 66.8, and was shorter than the fine-tuned end
    ([2504.16891](https://arxiv.org/html/2504.16891) Table 10; [W4] FC 6).
  - With LoRA, a λ-scaled merge W + λ·ΔW costs only evaluation.
- **Game training costs general skill.**
  - SIMA 2 (Gemini Flash-Lite, 10 environments, pre-training data mixed in) lost AIME −15 to −25% and GPQA −16 to −20%
    relative ([2512.04797](https://arxiv.org/html/2512.04797v1); [W3] FC 4).
  - PAPRIKA, with self-generated data at LR 1e-6, showed no regression (MT-Bench 7.88 → 8.14; GPQA 33.5 → 32.8)
    ([2502.17543](https://arxiv.org/html/2502.17543); [W3] FC 5).
- **What we must not forget is not MMLU** ([W1] Q2):
  - valid tool calls over long contexts;
  - diverse proposals on stuck levels;
  - near-optimal play on levels we already clear.

  Gate on game metrics and tool-call validity, with a small GPQA/IFEval canary (an early-warning test set) as a
  back-off trigger ([G-B] Q6).

### 1.7 Shorter thinking (R3): the lever, and how it fails

**R3** is "same decision in fewer tokens". Sample k replies at a logged decision state, keep the shortest one the
engine verifies, and SFT on it.

- **Self-selection cuts length modestly** ([Munkhbat et al.](https://arxiv.org/html/2502.20122) Table 2; [W7] FC 1):
  - shortest-of-16 own samples: −12.8% / −10.1% length at about equal accuracy;
  - conditioning the sampler with concise few-shot exemplars (worked examples placed in the prompt), then training
    without them: −32.9% / −19.6% at a matched sampling budget;
  - all on 1.5–7B models with short chains of thought (short-CoT).
- **The closest analogue to us collapsed** ([Arora & Zanette](https://arxiv.org/abs/2502.04463) Tables 8–13;
  [G-G] FC 1):
  - recipe: shortest correct of 8 own samples, R1-Distill-7B, full fine-tuning at 5e-6;
  - AIME24 fell from 0.563 to 0.183 at 5.8k tokens, worse than simply truncating the base model at 8k (0.390);
  - normalized over benchmarks, it kept 0.822 of accuracy at 0.597 of the tokens. DPO (§1.8) kept 0.960 at 0.737, and
    RL 0.959 at 0.516.
- **Our own hard states don't reward shortness** (700 samples from state-lab, our tool that rebuilds a logged game
  state and samples the model there again; 30 state-and-prompt cells on 14 states; [W7] §5, FC 7):
  - right replies are not shorter than wrong ones: P(success is shorter) = 0.445;
  - success by length tercile is 10.4 / 10.0 / 13.2%;
  - the shortest verified reply is 0.68× the median verified one;
  - success is 11.1%, so about 20 samples per state are needed. Easy states are unmeasured.
- **RL length methods** cut 40–80% on verbose 1.5–7B distilled models (§1.10) ([W7] §2).
  - On a saturated RL-tuned model they cost more: ThinkPrune took QwQ-32B −49% tokens at −3.2 average and −6.3 AIME
    ([2504.01296](https://arxiv.org/html/2504.01296); [W7] FC 4).
  - Targeting the shortest reply *regardless of correctness* collapses ([ShorterBetter](https://arxiv.org/html/2504.21370);
    [W7] FC 3).
  - Negative advantages (penalties on below-average samples; §1.9) push replies longer
    ([Fatemi](https://arxiv.org/abs/2504.05185)).
- **In agents, cutting thinking by fiat hurts; selective cuts work.**
  - o1 at low effort scored 21.0% vs 29.1% at high on SWE-bench ([Cuadron](https://arxiv.org/html/2502.08235); [W7] FC 6).
  - RACE (Qwen3.5-9B) skipped reasoning on turns earlier reasoning had covered: −32 to −80% reasoning tokens within 1.4
    points ([2610.12061](https://arxiv.org/html/2610.12061); [W7] FC 5).
  - A general "think less" fine-tune (Swift 1.5) *raised* mean output 11.9% on Terminal-Bench, though its median fell
    17.9% ([L2] FC 5).
- **A failure to watch.** In one Kaggle fine-tune, sentence-level down-weighting of reasoning gave 1,474 of 4,767 turns
  with no tool call, against 132 of 3,668. It is a single run with the base model unnamed
  ([743319](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743319); [W5b] FC 3). Weak, but
  cheap to monitor.
- **The recipe that survives the evidence** ([G-G] (a); [Plan] §7.6):
  - whole-turn targets (every call of a turn, up to the one that acts), scored on total turn tokens and calls;
  - per state, keep the shortest verified sibling (another sample from the same state) at **≥0.7× that state's median
    verified length**;
  - add 30% median-length verified siblings as a length anchor;
  - no compression of a level's first turn or of the turn after a death;
  - one epoch (one pass over the data), then regenerate; LoRA LR about 2e-5; a λ-sweep;
  - gates on right-move rate, verification markers (phrases where the model checks its work), action diversity and
    no-tool-call rate.

  Rated **medium risk** ([EB] §5.1 item 14).

### 1.8 Preference optimization (DPO)

- **DPO** trains on pairs of replies to the same prompt, one *chosen* and one *rejected*. It raises the chosen reply's
  probability relative to the rejected one, with no reward model (a learned scorer of replies). β sets how far it may
  move from the start.
- **It works on agents only with an SFT anchor and the right pairs**
  ([ETO](https://arxiv.org/html/2403.02502) Tables 3–5; [W1] FC 8):
  - trajectory-level success/failure pairs: WebShop 63.1 → 67.4 (Llama-2-7B);
  - step-level pairs collapsed to 8.3;
  - without behaviour cloning first, ETO scored 12.5, below the untuned model's 17.9.
- **Online or semi-online DPO** (pairs re-sampled from the current model) matched GRPO; purely offline methods fell short
  ([2506.21495](https://arxiv.org/abs/2506.21495)).
- **PAPRIKA used one pair per task.** Several pairs per task caused "unintentional unalignment" ([W8] FC 5).
- **Kimi's length pairs:** chosen = shortest correct; rejected = wrong longer replies and correct replies 1.5× longer
  ([2501.12599](https://arxiv.org/html/2501.12599) §2.4).
- **For us:** same-state sibling pairs only, after SFT, test-first ([G-G] (e)).

### 1.9 Reinforcement learning, and why it is hard on this MoE

- **Definitions.**
  - In **RL** the model (the *policy*) acts, gets a **reward**, and is updated by a *policy gradient* to make
    higher-reward replies likelier.
  - The **advantage** is how much better a reply did than expected.
  - **PPO** estimates "expected" with a learned **critic** as large as the policy. That is impossible on one node (one
    8-GPU server) for a 125B model ([W2] §1).
  - **GRPO** drops the critic: it samples a group of replies per prompt and uses each reply's reward relative to the
    group mean.
  - An **importance ratio** corrects for data sampled by a slightly different policy.
- **GRPO learns only where the group disagrees (0 < pass rate < 1).**
  - Labs drop all-pass and all-fail prompts: ScaleRL drops prompts at ≥0.9 pass rate
    ([2510.13786](https://arxiv.org/html/2510.13786)); DeepSeek-V3.2 kept environments with pass@100 > 0
    ([2512.02556](https://arxiv.org/html/2512.02556)).
  - Whole games from reset are mostly always-won or never-won. So the usable unit for us is a **frozen decision state**
    with 0 < p < 1 ([W2] §5).
  - Selecting states by reward variance matters: in RAGEN-2 the highest-variance quartile gave 21.1% vs 11.0% for the
    lowest ([G-G] FC 5).
- **MoE routing breaks naive RL** ([R3 paper](https://arxiv.org/html/2510.11370v2); [W2] FC 1–2):
  - About 10% of routers pick different experts in training than at inference.
  - On Qwen3-30B-A3B, GRPO crashed in 3 of 3 settings. All 6 runs with **rollout routing replay** stayed stable;
    routing replay means forcing training to use the experts the rollout (the inference pass that generated the
    sample) used.
  - GSPO (a sequence-level ratio, used for Qwen3-Next) also collapsed without replay in three sources (R3 paper App. B,
    [SAPO](https://arxiv.org/abs/2511.20347) §5.1, ScaleRL App. A.17).
  - **Routing replay is required whichever loss is used.**
  - The load-balancing auxiliary loss (an extra loss that pushes the router to spread tokens evenly across experts)
    should be 0 during RL. The ms-swift Flash-Next recipe still sets 1e-3 ([W2] FC 4).
- **Train–inference mismatch.** The trainer must compute the same probabilities the server sampled from ([W2] §3, FC 5):
  - INT4 post-training-quantized rollouts with a BF16 trainer drift as training goes on;
  - NVFP4 (NVIDIA's 4-bit float format) rollouts collapse after about 150 steps ([QUADS](https://arxiv.org/abs/2607.15810));
  - INT4 QAT "almost overlaps" BF16 ([LMSYS](https://www.lmsys.org/blog/2026-01-26-int4-qat));
  - an FP32 (32-bit) LM head, the final layer that turns hidden states into token probabilities, raised MiniMax's
    train/inference correlation from about 0.9 to 0.99 ([2506.13585](https://arxiv.org/html/2506.13585)).
  - Our variant, LoRA on BF16 modules over frozen dequantized experts, is untested anywhere. The FP8 KV cache (the
    stored attention keys and values of past tokens) and kernel (GPU routine) differences remain.
- **Rewards.** "The frame changed" is true for 94.5% of actions, so it is useless. Use level clears, verified routes or
  progress over earlier attempts, with length terms on correct samples only ([W2] §6).
- **Scale.**
  - MiniMax-M1's RL, with its CISPO loss (clipped importance ratios that keep every token's gradient), cost $534,700
    (512 H800 for 3 weeks).
  - ScaleRL's full 8B run used 100k GPU-h (GPU-hours) ([W2] FC).
  - Our 450–950 GPU-h would buy a few hundred steps at most.
  - The cheaper alternatives (RAFT, i.e. online rejection-sampling fine-tuning; semi-online DPO; on-policy distillation,
    §1.10) recover much of RL's gain early. Vanilla RAFT still trailed GRPO later, 52.3 vs 56.3
    ([2504.11343](https://arxiv.org/abs/2504.11343); [W2] FC 8).
- **Collapse monitors.** Reward-sd collapse precedes reward collapse. "Template collapse" keeps entropy (the spread of
  the model's token choices) stable while reasoning stops depending on the input
  ([RAGEN](https://arxiv.org/html/2504.20073); [RAGEN-2](https://arxiv.org/abs/2604.06268); [W3] §1).

### 1.10 Distillation and self-distillation

- **Distillation** trains a *student* toward a *teacher*'s outputs.
  - Off-policy: on the teacher's text.
  - **On-policy distillation (OPD):** the teacher scores the student's own samples token by token (reverse KL), so the
    student stays on its own distribution.
- **OPD is about 10× cheaper than RL.** Qwen3-8B reached 74.4 AIME in 1,800 GPU-h via OPD, against 67.6 in 17,920 via
  RL ([2505.09388](https://arxiv.org/html/2505.09388) Table 21; [W4] FC 2).
- **We have no usable teacher.**
  - Anthropic (Commercial Terms D.4), Gemini and OpenAI terms bar training competing models on outputs.
  - Their reasoning is hidden, and their tokenizers (text-to-token vocabularies) differ.
  - Qwen3.8-2.4T-A95B shares our tokenizer under an MIT-style licence, but it is **text-only** and can't read our
    image contexts ([W10] FC).
- **Self-distillation with privileged context hurts thinking models.**
  - Here the teacher is the same model, shown extra information such as the right answer.
  - OPSD lowered all 5 thinking models tested (Qwen3-8B 0.635 → 0.578). With thinking off it held
    ([2607.05184](https://arxiv.org/abs/2607.05184); [G-G] FC 2).
  - SDPO's wins were all with thinking off, at ≤8B ([2601.20802](https://arxiv.org/abs/2601.20802); [G-G] FC 7).
- **Still useful.** OPD with the *original* model as teacher is the documented repair for forgetting (§1.6).
  ms-swift 4.4.3 can use the base through `disable_adapter()` as that teacher, with no second model load ([W4] FC 8).
- **Context distillation** (baking the system prompt into the weights) targets the wrong tokens. The system prompt is
  4,134 tokens and prefix-cached, i.e. reused from earlier requests rather than recomputed ([W10] FC 7).

### 1.11 Test-time training and speculative-decoding heads

- **Test-time training (TTT)** updates weights on each test task during inference.
  - It drove ARC-AGI-1/2: the ARChitects went 18.3 → 44.5 on TTT alone
    ([2505.07859](https://arxiv.org/html/2505.07859v2) Table 2).
  - It is infeasible on our card: the weights take 69.85 of 96 GB while serving 10–14 streams (games played in
    parallel) ([W5] Q5, FC 7).
  - aTTT, a TTT method for agents, ran the updates behind its "up to +5" on dedicated training GPUs
    ([2607.03441](https://arxiv.org/abs/2607.03441)).
- **Speculative decoding.** A small **draft head** (our MTP layer) proposes several tokens; the main model checks them
  in one pass. *Accept length* is how many are kept per step. Ours is 2.68 ([G-E] Q3).
  - Clean decode speed (output tokens per second, free of prompt-processing stalls) is proportional to accept length.
  - Harness throughput (tokens per second over a full game run) rises about 2.25% per +0.1 accept ([G-E] Q2).
  - Retraining a head on the target's (main model's) own outputs helps most when the head started weak. Qwen3-Next's
    native MTP gained +0.21 in-domain
    ([Red Hat](https://developers.redhat.com/articles/2026/09/08/optimize-vllm-speculative-decoding-fastmtp-heads)),
    and AngelSpec's production MoE head +0.36–0.41 ([2607.25852](https://arxiv.org/abs/2607.25852); [G-E] FC 8).
  - Botched retrains regress: hard labels (training on the top token only, not the full distribution) at LR 2e-4 cut
    accept 2.756 → 2.493 ([Holodeck](https://huggingface.co/Wiself/Holodeck-Lounge-Regressed-MTP)).
  - Any trunk change moves the features the head reads, so the head must be re-checked after every trunk LoRA
    ([G-E] Q5).

### 1.12 Summary: methods against our constraints

In the last column, H1 is the run-up to the 11-02 deadline and H2 the work after it (§5.10).

| Method | Buys | Main risk | Fits us before 11-02? |
|---|---|---|---|
| R3: own shortest verified reply, SFT, regenerate | Fewer tokens per decision | Reasoning collapse (Arora); small reachable share (§4.3) | Only one round, opt-in; otherwise right after 11-02 |
| R4: own breakthroughs on external games | Mechanic discovery | No external-game data yet; 25 families is the overfitting regime | H2 |
| Same-state DPO after SFT | Extra length or quality contrast | Step-level collapse; unalignment from many pairs | H2, test-first |
| GRPO / GSPO / CISPO on frozen states | The largest gains in the literature | Needs routing replay, aux loss 0, matched weights; 10⁴+ GPU-h at published scale | H2-3 |
| On-policy distillation from the base | Forgetting repair | Needs support for the model in vLLM, the inference server ms-swift samples with | H2 |
| Privileged self-distillation (OPSD / SDPO) | Credit assignment (knowing which tokens earned the outcome) | Hurts thinking models | No |
| Teacher or frontier-API distillation | — | Terms of service; text-only teacher | No |
| World-model SFT, novelty rewards | — | No decision gain; our model over-covers (§4.4) | No |
| R1: retrained MTP drafter head | +3–6% throughput, lossless | Capture-hook plumbing (recording the server's hidden states); botched-retrain regression | **Yes** |
| TTT on the card | — | No memory | No |

---

## Part 2. What labs do

### 2.1 Pipelines, lab by lab

- **Qwen3** ([2505.09388](https://arxiv.org/html/2505.09388); [W4] §1, FC):
  - **Pipeline:** long-CoT cold start, then reasoning RL, then "thinking-mode fusion" SFT on its own rejection samples,
    then general RL over 20+ tasks.
  - **Reasoning RL:** 3,995 query-verifier pairs took Qwen3-235B's AIME'24 from 70.1 to 85.1 in 170 steps.
  - **Cost of generality:** the general stages cost Qwen3-32B AIME'24 83.8 → 81.4 and LiveCodeBench 68.4 → 65.7.
    Qwen accepted this "to enhance the model's overall versatility".
- **Qwen3-Next** used GSPO for RL stability with hybrid attention and sparse MoE
  ([model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)). **Flash-Next's** report has no post-training
  section. It says changes invisible in pre-training "degraded clearly after post-training". It names evaluation
  throughput as "the tightest bottleneck" ([tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) §2.2, §5).
- **DeepSeek** ([V3.2](https://arxiv.org/html/2512.02556v1); [V4](https://arxiv.org/html/2606.19348); [W4] §2):
  - **V3.2:** one mixed RL stage, to avoid multi-stage forgetting. It used "Keep Routing" (routing replay, "crucial" for
    MoE), "Keep Sampling Mask" and off-policy sequence masking, on 1,827 synthetic agent environments kept only if
    pass@100 > 0. Post-training cost more than 10% of pre-training.
  - **V4:** "the mixed RL stage was entirely replaced by On-Policy Distillation" from more than ten teachers, with FP4
    QAT on the MoE experts.
- **Moonshot (Kimi)** ([k1.5](https://arxiv.org/html/2501.12599v1); [K2](https://arxiv.org/html/2507.20534);
  [K2-Thinking card](https://huggingface.co/moonshotai/Kimi-K2-Thinking); [W4] §3, FC 7):
  - **k1.5 "long2short":** shortest-of-8 rejection SFT, length-contrast DPO, a long/short weight merge and length-penalty
    RL. This is the closest lab precedent for R3.
  - **K2:** per-task token budgets, and a PTX loss on curated data against forgetting.
  - **K2-Thinking:** INT4 weight-only QAT on the MoE components, with every benchmark reported at INT4.
- **Zhipu** ([GLM-4.5](https://arxiv.org/html/2508.06471); [GLM-5](https://arxiv.org/html/2602.15763); [W4] §4):
  - **GLM-4.5:** expert models, then a unified SFT at 128K, then general RL. Reasoning RL ran as a single 64K stage,
    because shorter multi-stage RL made the model "unlearn" long context.
  - **GLM-5:** ends with on-policy cross-stage distillation, and uses a token-in/token-out gateway so training sees the
    exact token ids the server produced.
- **MiniMax:**
  - **M1:** CISPO RL for $534,700 ([2506.13585](https://arxiv.org/abs/2506.13585)).
  - **M2:** its RL trains on (state, action) pairs as served and retains thinking (τ² 87 retained vs 64 discarded). Its MTP
    heads are co-trained with top-K KL ([2605.26494](https://arxiv.org/abs/2605.26494); [G-G] FC 8).
- **Prime Intellect, NVIDIA, Mistral** ([W4] §4, FC):
  - **INTELLECT-3:** about two months on 512 H200, from GLM-4.5-Air-Base. GSPO showed reward collapse in its ablations;
    it used masked token-level importance sampling ([2512.16144](https://arxiv.org/html/2512.16144v1)).
  - **Nemotron 3 Nano:** RLVR over 9 environments at once, because single-environment training "often results in
    un-recoverable degradation" ([2512.20848](https://arxiv.org/html/2512.20848)).
  - **Magistral Medium:** RL alone on top of a post-trained model kept most of its capabilities
    ([2506.10910](https://arxiv.org/abs/2506.10910)).
- **ByteDance Seed, UI-TARS-2** (browser games; [2509.02544](https://arxiv.org/html/2509.02544)):
  - A data flywheel: validated trajectories go to SFT, failures to continued pre-training.
  - It went SFT 44.27 → RL 59.77 on its 15-game suite.
  - It scored Sokoban 0.3 on the out-of-domain LMGame-Bench ([W4] FC).

### 2.2 Small teams that adapted a model to one benchmark

- **DeepSWE:** RL only, from Qwen3-32B on 4.5k problems for 200 steps, 23% → 42%
  ([card](https://huggingface.co/agentica-org/DeepSWE-Preview)).
- **SWE-Gym:** 491 trajectories gave Qwen2.5-Coder-32B +12.3 (Lite) and +13.6 (Verified)
  ([2412.21139](https://arxiv.org/abs/2412.21139); [W4] FC).
- **AIMO-2 winner:** heavy SFT of a 14B base, a 0.3/0.7 weight merge, FP8 serving with a self-trained draft head. W4A16
  PTQ cost 10 points (AIME24 72.7 vs 82.7) where FP8 cost nothing ([2504.16891](https://arxiv.org/html/2504.16891)
  §6, Table 9).
- **The common thread:** these gains came where the base was weak on the task format. Flash-Next is already a strong
  agent in our harness, so expect smaller gains ([W4] §6).

### 2.3 Cross-cutting patterns, and what transfers to us

| Pattern | Who | Transfers? |
|---|---|---|
| Everything on-policy: own rejection samples, OPD, RL | Qwen3, DeepSeek V4, GLM-5, Kimi | **Yes.** It is the core of R3 |
| Train against the deployed low-bit weights (QAT) | Kimi K2-Thinking, DeepSeek V4, LMSYS | **Yes, for free.** Dequantized experts as the frozen base (§1.5) |
| Routing replay for MoE RL | DeepSeek V3.2, R3 paper | Yes, when RL happens (H2) |
| Consolidate with one mixed stage or OPD, not sequential SFT stages | DeepSeek, Nemotron, GLM-5 | Partly: one stage; OPD when vLLM support exists |
| Interpolate or merge checkpoints | AIMO-2, Kimi, UI-TARS-2 | **Yes.** λ-merges cost only evaluation |
| Long-context training at deployment length | GLM-4.5, MiniMax M2 | Yes: train on the served context |
| RL at 10²–10³ GPUs for weeks | All frontier labs | **No.** 20–100× our budget |
| Evaluation is the bottleneck | Qwen (own words), INTELLECT-3 (eval every 15 steps) | **Yes, and worse for us:** our noise is larger (§4.6) |

---

## Part 3. Agents, games, ARC and this competition

### 3.1 Training LLM agents on games: what transferred

| Work | Model, scale | Result | Lesson |
|---|---|---|---|
| VHD-Play, hidden-dynamics tools ([2609.27321](https://arxiv.org/html/2609.27321v1)) | Qwen3.6-35B-A3B, 34 GRPO steps, 34,816 rollouts on 2,176 environments | Agentic 0.204 → 0.815; held-out +0.56, near-OOD +0.63, far-OOD +0.16 to +0.24 | The closest analogue. Near transfer is large, far transfer small |
| SPIRAL self-play ([2506.24119](https://arxiv.org/html/2506.24119)) | Qwen3-4B/8B-Base, 400 steps | +10.5 on base; +1.4 on R1-Distill | Gains shrink on models that already reason |
| PAPRIKA ([2502.17543](https://arxiv.org/html/2502.17543)) | Llama-3.1-8B-Instruct, 17k own trajectories + 5.3k pairs | 9 of 10 held-out groups improved; no general regression | Self-generated multi-task SFT transfers |
| ORBIT, cross-episode RL ([2602.04089](https://arxiv.org/html/2602.04089v1)) | Qwen3-8B, 100 steps | Mastermind 0.27 → 0.59, where single-episode RL fell to 0.21 | Rewarding learning *across* attempts transfers (one run) |
| LaMer, meta-RL ([2512.16848](https://arxiv.org/html/2512.16848v2)) | Qwen3-4B | pass@3 up, pass@1 down; reflection memory beat raw trajectories (56.4 vs 34.8) | Train the recap, not the transcript |
| ReSum-GRPO ([2509.13313](https://arxiv.org/html/2509.13313v3)) | WebSailor-30B-A3B | Training added 29.3 → 33.3 on top of a training-free summarizer | Trims map to segments with a broadcast advantage |
| MEM1 ([2506.15841](https://arxiv.org/html/2506.15841)) | Qwen2.5-7B | SFT on the compressed format collapsed beyond about 6 objectives; RL held | Context management must be inside training |
| TTI ([2506.07976](https://arxiv.org/html/2506.07976v2)) | Gemma-3-12B, online filtered BC | "Acting more" beat "thinking more" at a fixed budget | Outcome training shrinks per-step thinking |

Sources: [W3] §1–§5, FC 1–5. Every positive result is on short contexts (2–32k) and models ≤35B total. None is at our
57k context on a 125B hybrid MoE ([EB] §2).

**Exploration and goals.**
- LLM explorers fail by greediness on bandits, and a +1 untried-action bonus helps there ([W8] Q2).
- Our model over-covers rather than under-covers (§4.4), so novelty bonuses are a weak lever for us ([W8] Q5).
- The stuck failure is judgement. In 83% of one forum author's forks, the stuck run had already tried the decisive
  action type and "drew the wrong conclusion about the goal"
  ([743723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723); [G-F] FC 6).

### 3.2 The ARC lineage

**ARC-AGI-1 and 2 (2024–2025)** ([W5] Q1, FC):
- **TTT was the largest single gain** (2.4× to 6×). The ARChitects' ladder on ARC-AGI-1: 18.3 baseline → 44.5 with TTT
  → 62.5 with 16× inference-time augmentation → 71.6 with product-of-experts scoring and DFS
  ([2505.07859](https://arxiv.org/html/2505.07859v2)).
- **Large verified synthetic data came next.** NVARC (2025 winner, 24.03%) built 103,253 puzzles and 3.2M samples, and
  fine-tuned Qwen3-4B on 4 × 8 H100 for 27 h. It found that "many ARC-AGI approaches quickly overfit to the provided
  1000 training puzzles" ([NVARC](https://github.com/1ytic/NVARC)).
- **Static inference with a pretrained model capped near 11%** ([2412.04604](https://arxiv.org/html/2412.04604)).

**ARC-AGI-3** ([technical report](https://arxiv.org/html/2603.24621v1); [W5] Q2, FC 4):
- **Format:** turn-based games with levels, 64×64 grids, 16 colours, 5 actions plus undo and click.
- **Scoring:** per level, the squared ratio of human to AI actions; levels weighted by index; a cutoff at 5× the human
  action count.
- **Easier public set:** the public set is "intentionally easier", and the private set has "limited overlap with the
  mechanics".
- **ARC's overfitting taxonomy:** "domain-specific overfitting" covers training on lookalike games *or* "using a harness
  that contains ARC-AGI-3 specific strategies". It is defined for ARC's own leaderboard, not Kaggle.
- **Frontier scores** on the semi-private set: GPT-6 Astra high 99.95% through a provider adapter, and GPT-6 Astra max
  62.7% on the standard harness. No Qwen model has a row
  ([evaluations.json](https://arcprize.org/media/data/evaluations.json)).

**The 2025 preview and the 2026 milestones** ([W5] Q3, FC):
- **Preview:** won by StochasticGoose (Tufa, 12.58%), a CNN trained *during play* and reset each level
  ([preview blog](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)).
- **Milestone 1:** Tufa's Duck on stock Qwen3.6-27B FP8. "Hand-crafted tools actually hurt"
  ([M1 post](https://arcprize.org/blog/arc-prize-2026-milestone-1)).
- **Milestone 2:** Franzen, the ARChitects co-author, at 27.89, on stock Flash-Next. In his words: "I did not train or
  fine-tune the models" ([W5] FC 3).
- **arXiv:** of 27 ARC-AGI-3 papers, only one trains weights, and it reports no ARC-AGI-3 fine-tune result. Every
  near-100 public-set result is a training-free harness on a frontier model ([W5] FC 5).

### 3.3 Competitor and Kaggle intel

**Leaderboard, 2026-10-18** (about 50% of the hidden games; [W5b] §2, FC 4):

| # | Team | Score | Entries | What is known |
|---|---|---|---|---|
| 1 | Tufa Labs | 68.91 | 166 | Jumped 56.52 → 68.91 on 10-13. Model unknown |
| 2 | Yi-Chia Chen | 67.49 | 34 | Solo. Steady climbs 48.59 → 67.49 between 10-08 and 10-12 |
| 3 | mtg | 50.37 | 72 | Nothing public |
| 9 | NVARC3 | 43.94 | 41 | ARC-AGI-2 winners plus the DreamTeam authors, per a competitor-run board scrape ([L2] §2) |
| 16 | Daniel Franzen | 38.85 | 98 | Our base stack |
| 18 | us (mellamoa) | 38.22 | 33 | A lucky draw; mean about 31 |

- **Tufa's own words** (MLST, 2026-07-01, [transcript](https://app.rescript.info/api/public/sessions/23f8cebdc5ce8f5f/pdf);
  [L2] FC 1, 7):
  - Per-level ARC-score rewards are "the most naive implementation".
  - Their rewards included level rewards, ARC-AGI scores, "whether the code gets executed or not" and "length of the
    reasoning steps".
  - On self-generated games: "we can for example train on them".
  - Because training at 100–200k tokens is "extremely difficult", they train on shorter sequences and generalize to
    longer ones.
  - This describes a research pipeline. It does not say a submitted model was RL-trained.
- **Tufa's compute and release.** 48 B200 now, with 144 B300 "expected" ([tufalabs.ai](https://tufalabs.ai/)). They
  will release "when the competition ends", worrying about "the best of 4000 copies"
  ([742801](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742801); [G-F] FC 4).
- **Tufa's model.** It ran stock Qwen3.6-27B FP8 at M1. A Qwen3.8-27B-FP8 snapshot was uploaded by a Tufa member on
  08-18. A dead vLLM LoRA guard sits in their M1 code ([L2] §1, FC 2). That is all that's known.
- **Yi-Chia Chen** has GRPO and on-policy-distillation history. Her OPD-32B and DFlash models are AIMO artifacts, not
  ARC ones ([L2] FC 4).
- **Forum reports on fine-tuning Flash-Next** ([742835](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/742835),
  [748593](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/748593); [W5b] FC 1):
  - two H200 users (SFT; an RL pipeline on miles) and two others report no gain;
  - "MoE structure makes training unstable";
  - on 27B bases, a STaR LoRA moved LB 1.25 → 1.94;
  - behaviour cloning's "two-point jump on the public games … didn't translate" to the hidden set.
- **Harness saturation.** One author found nearly every harness layer at or below baseline. He concluded "the remaining
  lever appears to be the model itself, or training" ([743723](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/743723)).
  None of these people has a positive Flash-Next training result.
- **Best-of-N inflation.** The public LB shows each team's best draw. At our draw sd of 3.5, the best of 166 draws
  averages about mean + 9.4. So Tufa's typical draw could be about 59.5. That is an upper bound on the inflation, since
  the submissions are different systems ([EB] §1.7; [Critic] C10).
- **Does fine-tuning separate 60+ from 35–50?** No direct evidence either way ([L2] §6).
  - For: Tufa's July statement, their compute, Chen's track record, and a 17-point gap above #3 that exceeds best-of-N
    alone.
  - Against: Franzen's untrained win, no public fine-tune shown to beat the base ([L2] FC 6), and NVARC3's strong
    synthetic-data pedigree at 43.94.

### 3.4 Existing post-trained Flash-Next variants

Of 547 HF repos derived from Flash-Next, about 8 are real post-trains ([G-D] §1, FC):

| Variant | What changed | Overlays on our W4A16 with no re-quant? | Evidence in an ARC harness |
|---|---|---|---|
| Swift 1.5 (UkisAI), general efficient-thinking tune ([card](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next)) | 300 BF16 tensors, 5.82 GB: GDN, QSA, shared expert | **Yes** | One third-party 10-game, 8–11-minute screen: decode 4–8% *faster*, tokens per action −4%, 7 vs 13 levels ([G-D] FC 7). Teams that used it at some point: LB 26–35.6, not attributable |
| Tinfield-1 | 518 tensors, 6.64 GB, tiny deltas (0.03–0.12% RMS) | **Yes** | NVFP4 probe +5.6 pp, NO_GO, p ≈ 0.54 ([G-D] FC 8) |
| AZ-Focus, Agnes-3.0, CYBER-FROST, abliterations | Routed experts | No: needs an int4 re-quant | — |
| Tong Hui Kang's LoRA | Merged into NVFP4, scope unpublished | Unknown | Team best 32.48; whether it used the LoRA is unknown ([L2] FC) |

[I] Swift and Tinfield prove that the deployable shape exists: a non-expert post-train overlaid on Intel's experts.
Swift's overlay is the cheapest in-harness test of whether *any* post-training of Flash-Next helps us.

### 3.5 Rules and licences

- **Submissions:** 1 per day and 2 final selections. Entry deadline 2026-10-26; final deadline 2026-11-02 23:59 UTC.
  The scoring box is an RTX 6000 (g4-standard-48) with ≤9 h and no internet ([W5b] FC 5).
- **The hidden set is 110 games**, half on each LB ([Kaggle data page](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data);
  [G-F] FC 1).
- **One account per person.** Rule 3.5.a quoted in the Answer. Team members must be separate individuals (3.5.b)
  ([rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules); [G-F] FC 2). snowiers shows as a
  separate LB entry ([G-B] FC 11).
- **Fine-tuning on rented compute is not barred.**
  - External data and models must be "reasonably accessible to all" at "minimal cost" (2.6).
  - Winners must open-source code, model and weights under CC-BY 4.0, with a carve-out for incompatible pretrained
    licences (2.5, 2.8).
  - Staff won't say whether the Qwen licence is prize-eligible ([W5b] FC 5).
- **Licences of the data and models we might use** ([EB] §4.5):
  - DreamTeam games: Apache-2.0. arc-interactive: MIT.
  - sonpham and ARC-AGI-3-Training-Data: no licence, so validation only.
  - Swift adds a US$1M revenue threshold.
  - The GPQA terms forbid revealing items online.

---

## Part 4. Our situation in numbers

### 4.1 The data we have

| Item | Value | Source |
|---|---|---|
| Usable runs | 85 Franzen + Flash-Next runs with request logs; 1,642 game-runs; **all on the 25 public games** | [L1] Q1 |
| Calls | 141,483 responses; 1.13% ended without a tool call | [L1] Q1; [G-B] Q6 |
| Tokens | 269.6M generated (87% reasoning); 8.04B prompt (90.5% cached) | [L1] Q1 |
| Prompt length | Median 57,313, p90 84,800, max 120,756; ≤8k 2.2%, ≤16k 6.8%, ≤32k 16.5% | [L1] Q3, FC 3 |
| Outcomes | 289,520 actions; 6,034 level clears; 354 wins from 14 games | [L1] Q1, FC 5 |
| Diversity | **151 distinct cleared levels**, about 40 clears each | [L1] Q1 |
| Frontier cells (clear rate 0.2–0.8) | 38 across 18 games; about 23 after a 10-game holdout | [W3] FC 8 |
| What is missing | No token ids, logprobs or raw replies; history is normalized | [L1] Q2, FC 2 |
| R3-eligible states on the deployed stack | 2,857 in the corpus, but only 64 / 43 / 51 on the three runs of today's stack | [Plan] §7.6 |

**External games** ([EB] §1.1):
- **DreamTeam:** 25 games × 8 levels, Apache-2.0. Every game embeds a solver. They have never been run in our harness.
- **arc-interactive:** 249 games in 150 families, MIT. Most are small.
- **sonpham and Training-Data:** unlicensed, so validation only.
- **The DreamTeam generator** has only ever been run on frontier APIs, and its yield is unknown.

**The binding constraint is diversity, not volume.** 151 distinct levels from 25 games is NVARC's "1000 training
puzzles" regime ([W5] Q4).

### 4.2 Throughput and budget

- **A full run:** 25 games in 121 min, 5.03–5.26M generated tokens, about 690–720 tok/s, about 2.3 Kaggle GPU-h
  ([L1] Q6; [G-H] FC 11).
- **Serving:** MTP accept 2.684. Prefill stalls take about 20.6% of wall ([G-E] FC 4; [W10] FC 7).
- **Kaggle rates:** 2.5–2.9M tokens per GPU-h, and 730–960 frozen-state samples per GPU-h all-in ([L1] FC 6). For long
  (≥5k-token) turns, that is only about 200–240 samples per GPU-h [I] ([Plan] §7.6).
- **mellamoa quota:** 40.9 GPU-h left to 10-24, then 45 a week, under a 6 h daily cap ([P-eval] §0).

### 4.3 Where the tokens go

- **Actions and turns** ([L3] Q2; [G-A] Q1):
  - 80% of actions carry 0 tokens (batched actions).
  - A deciding turn averages 4,411 tokens (median 2,234).
  - Turns of ≥5k tokens hold 69.3% of all tokens.
- **Long turns are call loops, not long thoughts.** A 12k+ turn averages 7.7 calls, and 85.5% of its tokens are in calls
  that did not act ([G-A] Q1, FC 3).

| Level class (all 85 runs) | Token share |
|---|---|
| Cleared, never stalled, on the hindsight route | 39.4% |
| Cleared, never stalled, off-route | 21.8% |
| Stalled, then cleared | 4.4% |
| Not cleared here, cleared in another run | 32.4% |
| Never cleared in any run | 1.9% |

Source: [G-A] Q1.

- **What R3 can reach.**
  - Verifiable on-route states hold about 41% of tokens.
  - The recipe's stratum, cleared on-route turns of ≥5k tokens, is about 30% of leader tokens.
  - With the 0.7 floor and a 30% anchor, the projected total cut is at most 6.3%, about 4% at a realistic ratio
    ([Plan] §3.2; [G-A] FC 1–2).
  - **Correction:** the 10-17 plan's "~50–60% of tokens on stuck levels" counted all stalled levels. Never-cleared levels
    take 22–28% in leader runs ([L1] correction 1; [G-A] FC 9).
- **Sandbox errors** are 8.9% of calls and 9.2% of tokens.
  - About 25–45% of the errors are made by the harness (stale-state guard, blocked imports).
  - Fixing them saves the *repair call*, about 0.3% of tokens directly, up to about 1.7% at issue #18's upper bound
    ([G-A] FC 7–8; [Plan] §7.1).
  - `finish=length` calls are 6.8% of tokens ([G-A] FC 3).

### 4.4 What stalls are made of

- **Novelty stalls are rare.** Stretches of ≥15 actions with no new board occur in 302 of 7,204 level attempts and cost
  0.6% of tokens ([G-A] Q2, FC 4).
- **Exploration is not the deficit.**
  - New (state, action) pairs are 96.0% of actions on cleared levels against 89.1% on failed ones ([G-A] Q2).
  - On stuck levels, 96% of decisions had an untried action that reached a new board ([W8] FC 1).
- **A 30-stall audit** (one coder, labels not reproducible; [G-A] Q3, FC 6):
  - 8 were planned revisits;
  - of the 22 real stalls, 12 were **mechanic errors**, 5 goal inference, 4 execution and 1 perception.
- **So the target is judgement about mechanics and goals**, which agrees with the forum's 83% (§3.1). Novelty bonuses
  are dropped.

### 4.5 Score mechanics and headroom

- **Scoring** ([L3] Q1, FC 1; [docs](https://docs.arcprize.org/methodology)):
  - Level score = min(115, 100·(human/AI actions)²).
  - Game score = the level-index-weighted mean, capped at the weighted share cleared.
  - RESET counts as an action.
- **Depth dominates.** Clearing k of n levels at human efficiency scores 100·k(k+1)/(n(n+1)) ([L3] Q1, FC 2).
  - Our cleared levels already take 0.64× human actions (median).
  - Human efficiency everywhere would lift a run only from 51.17 to 53.06.
- **Stuck levels are mostly soft** ([L3] Q2, FC):
  - Of 451 final stuck levels, only 45 were never cleared in any run.
  - At the deadline, 15.6 of 17.3 live games sit on a level another run cleared ([G-H] FC 10).
- **Counterfactuals** (26 runs, mean 51.17; [L3] Q2):
  - +1 level per non-won game gives 61.6 offline.
  - Every game at its best-ever depth gives 75.4.
  - +1 level per hidden game is worth about +13–15 LB.

### 4.6 Compute elasticity: what a token saving is worth

- **The reference curve** (23 runs): S(60 / 80 / 100 / 121 min) = 30.5 / 39.7 / 45.7 / 51.6 ([G-H] Q2, FC 1–2).
- **Elasticity: +2.7 offline per +10% effective compute** (95% CI 1.9–3.5), about 28 points per unit of ln compute
  ([G-H] Q4, FC 8).
- **Validated only downward.**
  - 8 uniform slowdowns land on the curve (residual −0.01 ± 1.68).
  - No run has exceeded 121 min.
  - All 4 runs that bought tokens by changing streams or sampling fell below the curve, by −4.3 to −23.2
    ([G-H] Q1, FC 4–5).
- **Value of a tokens-per-action cut at equal decisions** ([G-H] Q4, FC 9):

  | Cut | Offline [range] | LB, P1 | LB, P3 |
  |---|---|---|---|
  | −15% | +3.6 [1.9–4.7] | +2.0 | +3.7 |
  | −25% | +6.3 [3.3–7.6] | +3.5 | +6.5 |
  | −30% | +7.8 [4.0–8.8] | +4.3 | +8.0 |

- **P1 and P3** bracket how offline gains reach the hidden games ([L3] Q2):
  - **P1 (floor):** LB gain = 0.548 × offline gain.
  - **P3 (ceiling):** the hidden games behave like the public ones at about 0.6× compute.
  - One LB draw at a halved private deadline separates them by only 1.6–1.9 draw-sd ([G-H] FC 12).
- **The internal simulator (sim2) runs 15–28% too steep.** It over-predicted every upward lever and the 27B (24.3
  predicted vs 14.51 real) ([G-H] FC 6).

### 4.7 Noise: what can be detected

- **Offline:** a full run has sd 3.15–3.28. A 10-game subset has sd 5.8, so a paired difference has sd 8.2
  ([G-B] Q1, FC 1).
- **LB:** per-draw sd 3.50. The 10x family's 19 draws average 29.12 ([L3] FC 7).
  - **mem-repair-10x drew 38.22 / 28.73 / 29.54 / 26.90.** Two of those (28.73, 29.54) were on snowiers ([G-B] FC 11).
- **Draws needed:** about 6 per arm to detect +5 LB, about 1.5–2.4 for +8 ([G-B] FC 4).
- **LB/offline = 0.548** (sd 0.051). Within the 10x family, r = 0.44 ([L3] FC 3).
- **Screens can't rank arms.** A single run's reliability within the family is 0.08. A minute-25 panel correlates 0.21
  with the final rank. Public-score split-half correlation is −0.45 to −0.48, so finals must never be picked by public
  score ([G-B] Q4–Q5, FC 8–9).

### 4.8 What LB 60 requires

- Efficiency alone needs about 2.7× effective compute under P3 and about 6× under P1 ([G-H] Q4; [L3] Q3).
- The realistic 15-day levers, R3 and R1, give single digits at best, and much less at R3's real reach (§4.3).
- **So LB 60 needs new capability:** clearing levels no run clears today, on unseen mechanics. The "~63 ceiling" belongs
  to our agent's efficiency levers, not to the benchmark. Tufa and Chen already exceed it ([L3] Q3; [Critic] C19).

---

## Part 5. The optimized plan

Source: [Plan], which starts from the risk-first draft. Dates and times are UTC; Kaggle GPU-h are counted per Pacific
day. Tags: **[$]** paid compute, **[K]** Kaggle push, **[LB]** submission, **[U]** your decision.

### 5.1 How the four lenses compared, and why risk-first won

| Lens | Bottom line | Mid LB at 11-02 | What the final plan took from it |
|---|---|---|---|
| Efficiency-first ([P-eff]) | Harness token fixes + R1; one gated LoRA round as a pipeline proof | 32 | Track ordering by LB per GPU-h; RTX spot boxes instead of snowiers |
| Evaluation-centric ([P-eval]) | Build the evaluation machine first; train only what can be judged | 32 | Adoption by mechanism; matched comparators; LB as non-inferiority check |
| Risk-first ([P-risk]) | Cheapest falsifier first, kill fast, protect the fallback | 31.5 | The skeleton: gate ladder, branches, pick rule |
| Tufa imitation ([P-tufa]) | Game factory + rollout fleet + RL burst; cap about $19k | 31.5 (high 40) | H2's shape; "train on the weights we serve" |

The final plan's mid case is lower (30). It shrinks F's 30.85 toward the 10x family's 29.12, and counts harness fixes at
their repair-call value (+0.1, not +0.5) ([Plan] §3.1, Review log).

### 5.2 Decisions for you

| # | Decision | Recommendation | When |
|---|---|---|---|
| **D0 [U]** | **The two-account exposure** | Write to Kaggle support now. Say mellamoa and snowiers are both yours, and ask how to fix it: remove snowiers' entry or its 8 submissions, and confirm mellamoa's standing. Not choosing a snowiers final doesn't fix it, because Kaggle normally auto-selects an entry's best submissions. A team merger can't fix it (3.5.b). No [$] item starts until Kaggle answers or you accept the risk in writing | **10-18, first** |
| D1 [U] | Accounts from now on | No more snowiers submissions and no snowiers compute | 10-18 |
| D2 [U] | The no-repeat rule | (a) Allow LB redraws of an unchanged notebook: they use no GPU quota, and each final needs ≥3 draws. If declined, §9.7 of [Plan] stacks a new config per slot. (b) Allow offline replicates on rented boxes (n ≤ 4 per arm) for non-inferiority tests. If declined, U25 is a catastrophe gate only | 10-18 |
| D3 [$] | RTX PRO 6000 boxes | Azure NC144lds, one at a time in Branch B (up to 3 in Branch A): spot about $1.02/h for restartable work, on-demand $5.50/h for scored runs ([W6] FC 7). Also open a no-quota provider and request GCP g4-standard-48 quota (prices UNVERIFIED). Cap $300 in Branch B; $500 of RTX time in Branch A | Request today; use after D0 |
| D4 [$][U] | Branch A in H1: the 8×H200 node for one R3 round | **Default no.** Run it after 11-02 as H2-1's first step. Request the quota today anyway; approval time is unpublished, so P(allocatable by 10-24) ≈ 0.2–0.3. Opt in only if late submissions turn out to be closed and every GO condition passes on 10-23. Node $84.80/node-h; caps $1k through G5, $2.5k in H1 | Request 10-18; decide 10-23 12:00 |
| D5–D7 [K][LB] | Kaggle pushes, LB slots, private model uploads | Per the ledger (≤6 GPU-h per Pacific day, 6 GPU-h weekly reserve) and the 16-slot calendar | Daily (uploads 10-20, 10-24, 10-25) |
| D8 [U] | Licences | Swift's (US$1M revenue threshold), the Qwen licence against CC-BY, and the R1 head's lineage. This affects prizes, not score | 10-22 |
| D9 [U] | GPQA canary | Keep GPQA items out of `runs/` on GitHub | 10-20 |
| D10 [U] | Final picks | Apply the pre-registered rule (§5.7); lock on 11-01 | 11-01 |
| D11 [$][U] | H2 | Whether to continue after 11-02, and its budget. Gated on Kaggle's late-submission answer and on reading the top-team releases | 11-03 |

### 5.3 Day 0 and Day 1 (cheapest gates first)

**Day 0, Sunday 10-18.**
1. **[U] D0, then D1 and D2.** Free.
2. **G0, rules admin.** A Sonnet browser agent reads the Kaggle pages and reports:
   - whether the private score is the best of the two picks;
   - whether late submissions are scored;
   - whether a still-running submission can be selected;
   - whether Kaggle auto-selects when none is chosen;
   - what happens to snowiers' entry;
   - the exact deadline.
3. **Quota requests (free):** Azure H200 node; NC144lds on-demand and spot; GCP g4-standard-48; Lambda; one no-quota RTX
   PRO 6000 provider.
4. **[K] E1, the elasticity test:** the leader (mem-repair-10x) at `demo_minutes` 161, otherwise unchanged (3.0 GPU-h).
   This is the first upward compute test ever run (§4.6).
5. **[K] Logging smoke** (0.3 GPU-h): raw replies, output token ids and logprobs on our SGLang build, with disk cost.
6. **Laptop work** (`nice -n 15`, ≤3 heavy jobs):
   - `arm/harness-fixes`: the #176 import shims, #18's narrow fix, the #27 deadline flag and a repair-call counter;
   - a `private_minutes` builder option, unit-tested. The template hard-codes `532*60` in the submission branch;
   - the `arms.json` entries `elastic-161`, `throttle-half` and `fallback-plus`;
   - the FR-Spec hot-map coverage count;
   - archive final-hold-lap-10x into `runs/` and `arms.md`.
7. **[K][LB] T½, the throttle draw:** F with the private deadline halved to 266 of 532 min. Submit by about 16:00 so the
   score is back before the 10-19 slot.

**Day 1, Monday 10-19.**

8. **Read E1 by its pre-registered rule (G2)** and log it in `arms.md`, wins and time-to-win first.
9. **Read T½ as one input only.** ≥17 leans P1; ≤15 leans P3. It never decides a go/no-go alone; one draw misreads
   10–20% of the time.
10. **[K] fallback+ on the public 25,** with raw logging on as a data run (2.3 GPU-h). Also the serving bench for #10
    (0.7).
11. **[$] Bring up the RTX box,** if D3 is approved and D0 resolved. First prove the wheelhouse runs outside Kaggle's
    image, then check serving parity: decode within 5% of 852.7 tok/s at batch 10 ([G-E] FC 4), greedy match and accept
    within ±0.03.
12. **Laptop:**
    - port state-lab from `archive/tooling/state-lab`, and validate it (G-SL);
    - build the overlay tool with positive-control tests;
    - DreamTeam plumbing;
    - start the whole-turn sampler (due 10-21) and the R1 capture hook (due 10-22).

### 5.4 The tracks, in order of expected LB per engineering day

| Track | What | Expected effect on pick 1 | Basis |
|---|---|---|---|
| **H**, harness hygiene | #176 inert import shims (a `time` shim with a step counter, no wall clock); #18 narrow fix (clear the stale flag on raise); #27 deadline from notebook start | ≈ +0.1 (+0.05–0.3 if it ships) | Repair calls ≈0.3% of tokens, up to 1.7% ([G-A] FC 7–8) |
| **V**, victory lap | `ARC3_VICTORY_LAP` (on `main`): replays a winning attempt after a win; can't lower a score | ≈ +0.15 | +0.34 offline per run, 11-run mean (`arms.md:568`) |
| **SB**, serving levers | #10 prefill flags, #12 low-M GEMM entry, a KV-pool audit; adopt at ≥1% clean tok/s | ≈ +0.05 | +1% tok/s ≈ +0.15 LB [I] |
| **R1**, MTP head | Retrain albucino's ≈71M dense tensors with the mixer frozen and int4 experts byte-identical; CE + top-K KL; LR 1e-5–5e-5 | ≈ +0.3 (+0.5–0.8 P1 / +0.9–1.5 P3 if it ships, P ≈ 0.3) | [G-E] Q2–Q4, FC 8 |
| **S**, Swift overlay | Swift's 5.82 GB BF16 tensors on our experts; pick 2 only | ≈ +0.01 to the best-of-two | [G-D] §3–4 |
| **A**, one R3 round (opt-in) | §5.6 | ≈ 0 to pick 1 in H1 | [Plan] §3.2 |

- **The near-lossless line for pick 1:** F (mem-repair-10x v1) → **fallback+** (F + H + V) → **fallback++** (+ R1 + adopted
  SB levers).
- **Adoption** needs the mechanism to show, plus ≥3 error-free draws with mean ≥ the previous config − 2. That is a
  non-inferiority check, not an alpha ([Plan] §9).
- **A slip kills only the track that slipped.**

### 5.5 The gate ladder, with kill points

| Gate | Cost | Pass (pre-registered) | A fail kills | By |
|---|---|---|---|---|
| G0 rules | Free | §5.3 item 2 answered; quota confirmed or refused | Nothing; sets D0, the branch and the pick rule | 10-19 |
| G1 overlay boot + Swift panel | 1.6 GPU-h | **Zero-delta overlay:** decode ±2%, accept ±0.03, greedy match ≥ a base-vs-base repeat, tensor hashes match, and a deliberately perturbed overlay changes output. Startup ≤ base + 2 min; first action ≤ 12 min. **Swift:** kill if decode < 0.95×, accept < 0.9× or parse errors > +2 pp; promote to a full run if tokens/action ≤ 0.85× and levels/action ≥ 0.95× | Every trunk-weight path in H1 / the Swift arm | 10-20 |
| G2 elasticity (E1) | 3.0 GPU-h | Valid if S(97) is within 44.7 ± 6.5. **Falsified** if S(161) − 51.57 < +3.5 (false-falsification ≈13–15%); **supported** if ≥ +6 | No node and no R3 in H1; R1 and SB continue only under $50 | 10-19 |
| G-SL state-lab fidelity | Laptop | ≥95% of rebuilt boards, actions and tool results identical on ≥100 states per config; every mismatch classed benign or not, and any non-benign one fails | R3 data from that config; every frozen-state screen if it fails on the mem-repair stack | 10-20 / 10-22 |
| G3 R3 yield | 3.3–4.0 GPU-h | On 100 states × 8 whole-turn samples, all three of: (a) ≥50% of states have ≥2 verified siblings; (b) within each state, the shortest tercile's verified rate ≥0.8× the longest's (the only local datum, 0.79 at hard states, already fails); (c) projected total cut ≥4% | Branch A in H1 | **10-22**; no result → Branch B |
| G-SV screen validation | ≈$8 spot | Base, a base repeat and accept05 on 450 held-out states; the screen must flag accept05 as worse | Branch A in H1 | 10-22 |
| GO (opt-in) | — | D0 resolved, your opt-in, G1/G2/G-SL/G3/G-SV pass, RTX parity, node allocatable by 10-24 08:00 | Branch A → H2 | **10-23 12:00** |
| G4 R1 capture | 0.5 GPU-h | Offline E[accept] within ±0.05 of 2.68 | R1 | 10-22 |
| G5 node smoke (A) | ≤10 node-h, cap $1k | ≥32k multimodal LoRA step at CP2/CP4; measured throughput; packed/unpacked GDN loss parity; zero-LoRA merge byte-identical | The trunk prong in H1 | 10-24 22:00 |
| G6–G8 (A) | 5–6.5 + 2.3–4.6 GPU-h + 3–4 slots | Integrity, MTP check, canary, frozen-state screen; U25 catastrophe/non-inferiority; LB confirmation | That checkpoint | 10-26 / 10-27 / 11-01 |

**Missed dates.** A missed date counts as a fail for that track only. No [$] item bills before D0 resolves and its gate
passes ([Plan] §4).

**Evaluation sets** ([G-B] Q1–Q3):
- **H10, held out:** lp85, tr87, cn04, sc25, vc33, dc22, ka59, lf52, re86, wa30.
- **T15:** the other 15 public games, the only source of training states.
- **U25:** H10 plus 15 DreamTeam games, with 10 more DreamTeam games kept as a single-use final panel.
- **Never score H10 inside a run that contains trained games.** Their freed compute flows to the other games, worth up
  to +3–8 ([G-B] FC 5).

### 5.6 Branch A, if you opt in: one R3 round (FT1)

- **States:** about 400 T15 on-route states on cleared levels, with logged turns ≥5k tokens and no level-first or
  post-death turns. They come only from the deployed stack: three existing runs, E1, the fallback+ data run and two
  raw-logged RTX runs. k = 8 whole-turn samples at served settings (T 0.65, top-p 0.95, top-k 20, xhigh) give about
  3,200 samples, 13–16 GPU-h.
- **Verify:** the next board hash matches the logged step, or the step lands on the hindsight route; no death.
- **Select:** the shortest verified sibling ≥0.7× the state's median verified length, plus 30% median-length anchors.
  No generic replay. Dataset gate: ≥200 selected turns by 10-24 12:00.
- **Trainer base:** BF16 with the routed experts set to dequant(Intel int4), checked against the served Marlin output.
  If that fails, Branch A stops.
- **Recipe:**
  - ms-swift Megatron pinned to ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1 and transformers 5.19.0
    ([ms-swift #10376](https://github.com/modelscope/ms-swift/issues/10376));
  - LoRA r16 on GDN in/out, QSA qkv/o and the shared expert, never experts, router or indexer;
  - LR 2e-5, global batch ≤32, one epoch, aux loss 0;
  - one example per call on the served post-trim context, with the four exporter transforms.
- **Node window:** 10-24 08:00 to 10-25 08:00 at most, ≤24 node-h ≈ $2.0k. Stage the container and checkpoint on
  persistent storage first. Allocate once and deallocate when done: an idle node costs $2,035 a day, and the NVMe is
  wiped on deallocation.
- **After training:**
  - a movement check on held-out sibling pairs; about 30–60 optimizer steps may barely move the policy [I];
  - at most 2 λ merges;
  - the MTP check: an accept drop >0.05 kills FT1 in H1;
  - the canary;
  - a frozen-state screen on H10 states, with thresholds set from G-SV's base-vs-base spread;
  - U25 runs;
  - a 9-h, 110-game dress rehearsal under scoring-box limits.
- **Thresholds calibrated on public games**, to be named in the PR: the 5k stratum, S = 0.30, the 0.7 floor, the anchor
  share and every gate delta.

### 5.7 Final-pick rule (pre-registered)

1. **Pick 1:** the furthest-adopted config in the near-lossless line (F → fallback+ → fallback++).
2. **Pick 2,** a hedge with a different policy:
   - Compute Δ* = s × (candidate draw mean − comparator draw mean). The comparator is loader-flagged base draws only,
     and s ≈ 0.52–0.56 at 3–4 vs 3 draws under a skeptical prior.
   - Pick 2 is the trunk candidate (FT1 or Swift) if Δ* > −0.33, it has ≥3 error-free draws, the dress rehearsal passed
     and D8 is cleared.
   - Otherwise pick 2 is F, or mem-unified-10x v1 if F is pick 1.
3. **A trunk change never takes both picks in H1.** At 3–4 draws, Δ* = +1 is inside noise ([G-B] FC 10).
4. **Choose by config, never by a draw's public score.** Select actively, in case Kaggle auto-selects.

### 5.8 Calendar to 11-02 (Branch B default; "A:" marks opt-in changes)

| Date | Kaggle (GPU-h) | Off-Kaggle | LB slot | Decision |
|---|---|---|---|---|
| Sun 10-18 | E1 161 min, logging smoke, T½ commit (3.55) | Day-0 laptop list | **T½** by ≈16:00 | **D0**, D1, D2 |
| Mon 10-19 | fallback+ public-25 data run, bench #10 (3.25) | RTX bring-up; state-lab port; overlay tool | T½ if missed, else final-hold-lap-10x copied to mellamoa (D2a) | Read G2, T½ |
| Tue 10-20 | G1 pair + perturbed boot, DT25 base (4.15) | Two raw-logged RTX runs; canary build | fallback+ #1 | G1, G-SL, D9 |
| Wed 10-21 | U25 base with loader flag; Swift full run if promoted (4.85) | R1 capture | Z1 (fallback+ via the overlay loader) | — |
| Thu 10-22 | G3; bench #12 (≤4.95) | G4, R1 training, G-SV | Z2 | G3, G-SV, G4, D8 |
| Fri 10-23 | State-lab sampling: R3 data or the H2 bank (5.0) | R1 selection; A: node pre-staging | Swift #1 (A: Z3) | **GO 12:00** (A) |
| Sat 10-24 | R1 MTP panel; fallback++ public-25 and U25 (5.55) | A: node 08:00 → G5 → training | Swift #2 (A: Swift #1) | R1 gates; A: G5 |
| Sun 10-25 | A: FT1 MTP check + canary; B: H2 sampling (≤4) or reserve | A: merges, export, deallocate by 08:00; dress rehearsal (≈$52) | C1 #1 (fallback++ or fallback+) | — |
| Mon 10-26 | A: λ screens | — | C1 #2 | Entry deadline; G6 (A) |
| Tue 10-27 | A: FT1 U25 | A: replicates | C1 #3 | G7 (A) |
| 10-28 → 10-30 | H2 data runs ≤4/day if reserve allows | A: FT1 rehearsal | B: Swift #3, Z3 or C1 #4, a stacked variant. A: FT1 #1–#3 | Δ* on 10-30 |
| 10-31 → 11-01 | Hold a 6 GPU-h reserve | — | Top-ups (A: Swift #2, #3) | **Picks locked 11-01** once that draw completes |
| Mon 11-02 | — | — | Information-only draw, never a final | Confirm by 20:00 |

Totals: Kaggle about 31 GPU-h in Branch B (≤45 with H2 sampling) and 42–50 in Branch A ([Plan] §6).

### 5.9 Cost

| Branch | Kaggle GPU-h | Rented | $ |
|---|---|---|---|
| **B (default):** no node | ≈31 (≤45) | RTX ≈27–33 h | **$70–170** (cap $300) |
| **A (opt-in):** node + one R3 round | ≈42–50 | RTX ≈75–90 h; node ≤24 node-h (≤192 GPU-h) | **$1.7–2.5k** (cap $3k) |
| **C:** no paid compute, or D0 unresolved | ≈35 | — | $0 |

Prices: ND96isr H200 v5 is $84.80 per node-hour ($10.60 per GPU-hour), with spot priced the same
([Azure Retail Prices API](https://prices.azure.com/api/retail/prices?$filter=armSkuName%20eq%20%27Standard_ND96isr_H200_v5%27);
[W6] FC 6). Lambda 8×B200 is $53.52 per node-hour ([pricing](https://lambda.ai/pricing)).

### 5.10 H2: after 11-02, toward a consistent 60

| Phase | Work | Gate / kill | Cost |
|---|---|---|---|
| H2-0 (11-03 → 11-10) | Harvest every mellamoa submission's private score (extra draws on the fixed private set). Read and reproduce the Tufa, NVARC3 and Lord Han Solo releases: their base model, and whether they fine-tuned. Run the 242-min elasticity ceiling on the RTX box | Reproduction within ±3 of the release's public-25 claim | ≈$20–110 |
| H2-1 (weeks 1–3) | Branch A's R3 round if not done. **An LB-proxy suite** of ≥55 unseen games whose base score sits near the LB (≈31). An environment pool of ≥100 mechanic families: a 20-seed open-model `game_creator` pilot, a CPU acceptance script, family holdouts. An exact-stack RTX spot fleet. Log raw replies, token ids, logprobs and routing ids | The proxy ranks known generations correctly; ≥150 frontier cells. Kill: generation yield <25–30% → coverage-select existing pools | ≈$0.5–1.5k + node ≈$1.1–2.0k |
| H2-2 (weeks 3–7) | **Iterated on-policy RFT**, regenerated each round: R3 rounds 2–3; R4 breakthroughs on external games aimed at mechanic errors; cross-attempt pairs; λ-merges; R1 per trunk | Each round: proxy +3 over ≥4 paired runs. Kill: two rounds without gain | $1.6–4.2k per round |
| H2-3 (weeks 6–12) | **Frozen-state RL:** states with 0 < p < 1 chosen by reward variance; clears or verified progress as reward; routing replay, aux loss 0, FP32 head; auto-stop monitors | Trainer↔server KL ≤0.004; no collapse over 150 steps; proxy +5. Kill: collapse in 3 seeds, or no gain by step 300 | $10–30k |
| H2-4 | Multi-turn game RL (Tufa's route) only if H2-3 gains. Re-check any new base that shows ≥30 LB | — | Open |

- **H2 odds** [I] ([Plan] §12):
  - our own pipeline alone: P(mean ≥ 64.5 within 3 months) ≈ 0.05, because the best transfer evidence is small
    (far-OOD +0.16 to +0.24, §3.1);
  - conditional on a reproducible top-team release near 60: ≈ 0.3;
  - blended: ≈ 0.1.
- **Stop rule:** if H2-2 doesn't move the proxy by ≥ +6 within about $10k, stop training and revisit the base model.

### 5.11 Deliberately not doing (and why)

| Not doing | Why |
|---|---|
| More snowiers submissions or quota | Rule 3.5.a |
| Renting the node in H1 by default | Worth ≤ +0.8 / +1.5 LB even if it works, at P ≈ 0.01 in H1 ([Plan] §3.2) |
| Online RL / GRPO / GSPO before 11-02 | Routing replay required; the one-node recipe tops out at 1k + 8k tokens; budget buys few steps ([W2]; [W6]) |
| Routed-expert LoRA or any AutoRound re-quant | Forces int4 re-quant; non-expert LoRA needs none ([W6] FC 1) |
| R4 or external-game SFT in H1 | No external-game data yet; 25 families is the regime that overfits ([G-G] FC 6) |
| Behaviour cloning of logged replies | Normalized history; copies the policy rather than shortening it ([L1] FC 2) |
| OPSD / SDPO; frontier or 2.4T distillation | Hurts thinking models; terms; text-only teacher ([G-G] FC 2; [W10]) |
| Novelty rewards, world-model SFT | Novelty stalls are 0.6% of tokens; no decision gain ([G-A] Q2; [W8] FC 2–3) |
| A low-effort teacher; relaxed acceptance | Stay on-policy at deployed effort; accept05 lost 6–10 points ([G-E]) |
| A sandbox import whitelist | Would expose `os` and guard state through `builtins`, `sys`, `inspect`, `pickle`; inert shims instead ([Plan] §7.1) |
| Test-time training; a 27B swap | No memory on the card; the 27B scored 14.51 offline vs 54.69 ([L3] FC 6) |

---

## Part 6. What changed versus the 2026-10-17 plan

| 2026-10-17 plan ([Prior]) | Now | Why |
|---|---|---|
| Keep Flash-Next | **Unchanged** | Still the only base with ≥30 LB evidence; the 27B scored 14.51 offline ([L2] §3; [L3] FC 6) |
| After training, re-quantize with AutoRound and redo the MTP | **No re-quant:** a non-expert LoRA is a file overlay. R1, the MTP retrain, is a lever in its own right | Intel quantized only the routed experts ([W6] FC 1; [G-D]); [G-E] |
| 8×H200 node as the minimum, 450–950 GPU-h, cost unknown | **Node is opt-in and by default an H2 investment.** ≤24 node-h if run in H1, at $84.80/node-h; RTX PRO 6000 boxes ($1.02–5.50/h) for rollouts and evaluation | [Plan] §3.2; [W6] FC 6–7 |
| Train on trimmed 8–16k windows | **The served post-trim context** (31–85k), gated at 57k; short windows only as a fallback | AWM: 22.57 → 9.35 from context mismatch ([G-G] FC 6); Tufa trains shorter on purpose, but generalizes up ([L2] FC 7) |
| MiniCPM-style WSD with 30–50% self-replay; a small-sibling wind tunnel | **0% generic replay;** within-state length anchors and λ-merges; no wind tunnel | Replay anchors only partly ([G-G] FC 10); a dense sibling can't rank MoE choices ([Critic] C12) |
| LoRA LR around the recipe's 1e-4 | **1e-5 to 3e-5** (full-FT anchors × 10) | [G-G] (d), FC 9 |
| "~50–60% of tokens on stuck levels"; a novel-board bonus | **Never-cleared levels 22–28%** in leader runs; novelty rewards dropped | [L1] correction 1; [G-A] Q2 |
| R3 = a big lever (−25% tokens on most decisions) | **≈ −4% of total tokens** (≤ −6.3%) at the recipe's reach; **medium risk** | [G-A] FC 1–2; [G-G] FC 1 |
| Data = logs + DreamTeam/arc-interactive trajectories in H1 | **Own whole-turn verified siblings** at ≈400 states from the deployed stack; external games for evaluation or H2 | [L1] FC 2; [G-G] FC 6; [Plan] §7.6 |
| Test = 10 held-out public games + 2-h runs | **U25 = H10 + 15 DreamTeam games**, a frozen-state screen validated against accept05, and never H10 inside a run with trained games | [G-B] FC 5 |
| Kaggle mellamoa + snowiers for collection | **mellamoa only;** the existing snowiers exposure goes to Kaggle support | Rule 3.5.a ([G-F] G17) |
| LB sd ≈5, ~16 draws to detect +5 | **sd 3.50; about 6 draws per arm for +5;** the LB is a crash and non-inferiority check | [L3] FC 7; [G-B] FC 4 |
| Finals = the two best mean configs | **A pre-registered rule;** a trunk fine-tune takes at most pick 2 | [G-B] FC 10 |
| One SFT round then a ReST-EM round 2 in H1; no upward-compute test | **At most one round, by default after 11-02.** E1 (161 min) and T½ first | [G-H] Q1, Q5 |
| Hidden set size not settled | **110 games, 55 per LB** | [G-F] FC 1 |
| H100 ≈ $12/GPU-h (third-party) | H200 $10.60/GPU-h (Azure price API); spot has no discount | [W6] FC 6 |
| "Nobody has shown fine-tuning works here" | **Unchanged, and stronger:** no positive Flash-Next result in any forum, paper or notebook. Franzen won M2 untrained. Two existing post-trains have no ARC gain | [W5] Q4; [W5b] FC 1; [G-D] |

Still valid from 10-17:
- the RSI toy result (score doubled through shorter answers);
- unlicensed data stays out of released weights;
- "60 is reachable, but we don't know the route".

---

## Part 7. Risks and open questions

### 7.1 Risks that can silently cost days

| Risk | Mitigation |
|---|---|
| Disqualification over the existing snowiers entry | D0 today; no [$] until resolved; mellamoa only |
| Our SGLang build doesn't run outside Kaggle's image | Wheelhouse test on a non-Kaggle box before anything depends on it; Branch C otherwise |
| No box or node quota | Requests filed today; Branch B needs no node |
| An overlay is silently ignored by the loader | G1's tensor-hash check and the perturbed-overlay positive control |
| The overlay or slower startup breaks the 9-h private rerun | Startup ≤ base + 2 min, first action ≤ 12 min; the #27 flag; Z draws; a dress rehearsal under scoring-box limits |
| R3 collapses reasoning (the Arora pattern) | The 0.7 floor, anchors, low LR, λ-merges, the movement check, a screen validated against accept05 |
| R1's capture hook overruns its 2–4 days | G4 on 10-22; fallback+ ships without R1 |
| The whole-turn sampler overruns | No G3 result on 10-22 → Branch B automatically |
| A final is still running at the deadline | Picks lock on 11-01 from completed draws |
| Overfitting to public games | Never score H10 next to trained games; U25 and the LB are the only valid evaluations for trained checkpoints |

### 7.2 Open questions, with the cheapest test

1. **Does more compute convert to score** (R3's premise), and what is the curve's shape above 121 min? Test: E1 today;
   the 242-min run in H2-0 ([G-H] Q5, FC 3).
2. **P1 vs P3 on the hidden set.** T½ is one input; the H2-0 private scores add more ([G-H] Q5).
3. **R3 yield at easy states, and where samples should come from.**
   - Some notes recommend sampling at low effort and training under xhigh; another says never use a low-effort teacher.
     The evidence for the second is a distillation result, not a selection one ([Critic] G4).
   - Test: G3 (about 3–4 GPU-h).
4. **Spill-over.** Does R3 on cleared states shorten turns on stalled levels? Does compressed thinking cost the working
   memory that later turns read? Unmeasured ([G-A] Implication 1; [Critic] G14).
5. **Training at the deployed shape:** a 57k multimodal step with context parallelism, GDN packing parity, real tokens
   per second. Only a node smoke can answer it ([Critic] G5; [W6] §3).
6. **How Flash-Next does on external games**, and whether those games track the LB (~31) or the public set (~51). One
   panel is about 2.3 GPU-h ([G-B] Q3).
7. **`game_creator` yield with an open model.** A 20-seed pilot ([W9]; [L2] §2).
8. **Whether frozen-state screens predict full-run score.** Existing data can't back-test it ([G-B] Q5).
9. **LoRA scope:** shared-expert vs attention-only on a partial-module MoE LoRA ([W1]; [G-G]).
10. **Whether verbatim history is score-neutral.** Its only run was confounded ([G-C] Q3).
11. **Tufa's and Chen's methods.** Releases come after 11-02 ([G-F] G9).
12. **Rules:** compute-only use of a second account, Qwen-licence prize eligibility, and whether the best of the two
    picks counts (G0).

---

## Appendix A. Fact-check corrections that matter

These come from the notes' fact-check sections. Each one changed a number or a decision.

| Claim in a note body | Verdict after checking | Source |
|---|---|---|
| A merged LoRA needs an AutoRound re-quant | Not for a non-expert LoRA: Intel keeps everything but trunk routed experts in BF16 | [W6] FC 1; [W4] FC 5 |
| The overlay is about 5.9 GB | About 72 GB if whole Intel shards are copied; small only with new shards keyed by tensor name | [W6] FC |
| Only transformers ≥5.18 renames `full_attention` | Every `qwen4_exp` release rewrites it; keep the original `config.json` | [W6] FC 8 |
| GSPO removes the need for routing replay | No: GSPO without replay collapsed in three sources | [W2] FC 2 |
| R3 paper: 4 stable replay runs | 6 stable runs; the 3 GRPO crashes are 3 different settings | [W2] FC 1 |
| Shortest-correct self-SFT "keeps accuracy in every paper" | Arora's RS+SFT on a long-thinking model collapsed AIME 0.563 → 0.183 | [W7] other errors; [G-G] FC 1 |
| GPT-4o concise SFT gives a "severe drop" | 97.65% (GSM8K) / 90.52% (MATH) relative; the severe drop is other external data | [W1] FC 7; [W7] FC 2 |
| Qwen3 Stage 3 self-data prevents regression | It limits regression; AIME'24 still fell 83.8 → 81.9 | [W1] FC 1 |
| Chu et al.: SFT is needed before RL | Only for base models that can't follow instructions | [W3] FC 2 |
| AWM 10/100/526 environment curve, 4B–14B | Qwen3-4B only; 100 environments gave about +2–3 BFCL and 0 MCP | [W9] FC 1; [G-G] FC 6 |
| "All 200 environments" vs 30 selected | All 170 non-held-out environments | [W3] FC 6 |
| Self-replay does not anchor | Partly: IF-eval 79 with a 70/30 mix vs 45 without, out of 85 | [G-G] FC 10 |
| Replay of 30–50% as the forgetting guard | 0% generic replay; within-state anchors + λ-merges | [EB] §5.1 item 15 |
| Swift decoded ~16% slower | 4–8% *faster* per the serve log; the A/B was a 10-game screen | [G-D] FC 7 |
| "Four independent H200 users" got nothing from Flash-Next | Two stated H200; one used 2×96 GB; one unstated | [W5b] FC 1 |
| The no-tool-call collapse was on Flash-Next | The base is unnamed; at the 1-h budget the arm was ahead | [W5b] FC 3 |
| Tufa trains at 100–200k tokens | Tufa trains on shorter sequences to avoid that | [L2] FC 7 |
| Tufa's RL pipeline produced its submission | The transcript describes a research pipeline only | [L2] FC 1 |
| Chen's OPD-32B is an ARC model | It was built for AIMO proof-pilot | [L2] FC 4 |
| The 2.4T can teach our model | It is text-only | [W10] FC |
| SDPO outputs are 3× shorter | Relative to GRPO, in a thinking-off setting | [W10] FC 3; [G-G] FC 7 |
| INTELLECT-3 kept CISPO | It used masked token-level importance sampling | [W4] other errors |
| UI-TARS-2 scored Sokoban 0.3 in-suite | That is the out-of-domain LMGame-Bench | [W4] other errors |
| ARChitects +18 from data augmentation | Inference-time 16× augmentation, not training data | [W5] other errors |
| AngelSpec 52.8 → 66.4% unverified | Confirmed in the full text; it is the closest MTP analogue (+0.36–0.41) | [G-E] FC 8 |
| R1 reaches 3.0–3.2 accept | That is the upside; the central case is +0.15 to +0.25 | [G-E] Q1, FC 8 |
| LB sd 5 or 3.04 | 3.50; a consistent 60 ≈ a mean of 64.5 | [L3] FC 7 |
| The hidden set is 55 games | 110, about 55 per LB | [G-F] FC 1 |
| State-lab does 1.2–1.6k samples per GPU-h | 730–960 all-in; ≈200–240 for long turns | [L1] FC 6; [Plan] §7.6 |
| DreamTeam ships no solutions | All 25 embed `solve_level`; 20 of 25 match their baselines | [W9] FC 6 |
| H10 gains of +5–8 from freed compute | An upper bound; about +3–8 | [G-B] FC 5 |
| Le Grand's harness "at 43" | 43 is his LB rank, not a score | [G-F] FC 7 |
| Harness fixes worth ≈ +0.5 LB | ≈ +0.1: count the repair calls, not the whole error calls | [Plan] Review log |

## Appendix B. Notes index

All notes are in [`../data/finetune-sweep-2026-10-18/`](../data/finetune-sweep-2026-10-18/), which has its own README.
Each note ends with a fact-check whose corrections override its body.

- **Consolidation and plans:**
  - [EB]: the evidence base;
  - [Plan]: the final plan;
  - [Critic]: the completeness check;
  - [P-eff], [P-eval], [P-risk], [P-tufa]: the four lens drafts.
- **Local audits:**
  - [L1]: our data and tooling;
  - [L2]: competitors and public artifacts;
  - [L3]: the headroom model.
- **Literature and web:**
  - [W1]: SFT pedagogy;
  - [W2]: RL fundamentals and MoE issues;
  - [W3]: agentic and game RL;
  - [W4]: lab pipelines;
  - [W5]: the ARC lineage;
  - [W5b]: Kaggle intel;
  - [W6]: toolchain and compute;
  - [W7]: efficient reasoning;
  - [W8]: exploration and world models;
  - [W9]: environment generation;
  - [W10]: distillation.
- **Gap notes:**
  - [G-A]: the token tail and stalls;
  - [G-B]: evaluation design;
  - [G-C]: render parity and logprobs;
  - [G-D]: existing Flash-Next variants;
  - [G-E]: the MTP head (R1);
  - [G-F]: top-team intel and rules;
  - [G-G]: deep reads of the key cautions;
  - [G-H]: compute elasticity.
- **Earlier work:** [Prior] and its notes in
  [`../data/finetune-research-2026-10-17/`](../data/finetune-research-2026-10-17/).

[EB]: ../data/finetune-sweep-2026-10-18/_evidence-base.md
[Plan]: ../data/finetune-sweep-2026-10-18/_plan-final.md
[Critic]: ../data/finetune-sweep-2026-10-18/_critic.md
[P-eff]: ../data/finetune-sweep-2026-10-18/plan-efficiency-first.md
[P-eval]: ../data/finetune-sweep-2026-10-18/plan-eval-centric.md
[P-risk]: ../data/finetune-sweep-2026-10-18/plan-risk-first.md
[P-tufa]: ../data/finetune-sweep-2026-10-18/plan-tufa-imitation.md
[L1]: ../data/finetune-sweep-2026-10-18/L1-data-inventory.md
[L2]: ../data/finetune-sweep-2026-10-18/L2-competitors.md
[L3]: ../data/finetune-sweep-2026-10-18/L3-headroom.md
[W1]: ../data/finetune-sweep-2026-10-18/W1-sft-pedagogy.md
[W2]: ../data/finetune-sweep-2026-10-18/W2-rl-fundamentals.md
[W3]: ../data/finetune-sweep-2026-10-18/W3-agentic-game-rl.md
[W4]: ../data/finetune-sweep-2026-10-18/W4-lab-pipelines.md
[W5]: ../data/finetune-sweep-2026-10-18/W5-arc-lineage.md
[W5b]: ../data/finetune-sweep-2026-10-18/W5b-kaggle-intel.md
[W6]: ../data/finetune-sweep-2026-10-18/W6-toolchain-infra.md
[W7]: ../data/finetune-sweep-2026-10-18/W7-efficient-reasoning.md
[W8]: ../data/finetune-sweep-2026-10-18/W8-exploration-worldmodel.md
[W9]: ../data/finetune-sweep-2026-10-18/W9-env-generation.md
[W10]: ../data/finetune-sweep-2026-10-18/W10-distillation-context.md
[G-A]: ../data/finetune-sweep-2026-10-18/G-A-token-tail-stalls.md
[G-B]: ../data/finetune-sweep-2026-10-18/G-B-eval-design.md
[G-C]: ../data/finetune-sweep-2026-10-18/G-C-render-logprobs.md
[G-D]: ../data/finetune-sweep-2026-10-18/G-D-variants.md
[G-E]: ../data/finetune-sweep-2026-10-18/G-E-mtp-head.md
[G-F]: ../data/finetune-sweep-2026-10-18/G-F-intel-rules.md
[G-G]: ../data/finetune-sweep-2026-10-18/G-G-deep-reads.md
[G-H]: ../data/finetune-sweep-2026-10-18/G-H-elasticity.md
[Prior]: finetune-plan-2026-10-17.md
