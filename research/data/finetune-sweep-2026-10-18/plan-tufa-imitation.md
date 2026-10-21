# Plan: Tufa imitation, the max-upside lens (2026-10-18)

**Lens.** Tufa Labs is the only first-hand source for a 60+ method. In July they described end-to-end RL on games they generated themselves (L2 §5, FC 1). This plan builds the closest copy we can afford. It states what the copy costs and how likely it is to work, and each stage is built to leave something useful behind if the next one fails.

**Sources.** Numbers come from the 2026-10-18 sweep, with each note's fact-check applied over its body (`_evidence-base.md` indexes them). Note names are file prefixes (W2 = `W2-rl-fundamentals.md`); "FC n" is that note's fact-check item. A † marks a threshold calibrated on public games; any PR that uses one must name it (CLAUDE.md).

## 0. Answer

- **What we copy from Tufa.** Self-generated games. Per-level ARC-score rewards, shaped by "whether the code gets executed" and "length of the reasoning steps". Training on shorter sequences and generalising to longer ones (L2 FC 1, 7).
- **What we can't copy.** Their compute: 48 B200 now, 144 B300 "expected" (L2 FC 2). Their 166 submissions. Their months of iteration.
- **Trick 1: rollouts are cheap.** Azure rents the exact Kaggle GPU. One RTX PRO 6000 spot box costs about $1.02/h and generates about 2.5M tokens/h on our exact serving stack, about $0.41 per million tokens (W6 §7–8, FC 7).
  - A fleet of these boxes does all game generation, difficulty screening, rollouts and replicate evaluation.
  - The $84.80/h 8-GPU node does only gradient steps.
- **Trick 2: train on the weights we serve.**
  - Dequantise Intel's int4 routed experts to BF16 inside the trainer.
  - Put LoRA only on BF16 modules (GDN, QSA, shared expert).
  - The merged adapter then needs no re-quantisation, and fleet rollouts are on-policy up to kernel noise and the FP8 KV cache (W2 §3; W6 FC 1; G-D).
- **What H1 (to 11-02) can deliver at best.** One RL-shaped checkpoint from 2–3 semi-online rejection-sampling rounds. If the plumbing passes by 10-27, a ~30-step online GSPO/CISPO burst with routing replay is added on top. That matches VHD-Play's 34 steps, but with about a quarter of its 34,816 rollouts (W3 FC 1).
- **Odds** (my estimates; §1 gives the reasoning):
  - LB mean at 11-02: **low 29.5, mid 31.5, high 40**.
  - P(a fine-tuned model beats the harness fallback by 11-02) ≈ **0.12**.
  - P(LB mean ≥ 60 by 11-02) < 1%.
- **H1 cost.**
  - Cap ≈ **$19k** with Azure nodes, or ≈ $13k with Lambda B200 nodes.
  - Expected spend ≈ $8–10k, because money is released phase by phase and each kill gate stops it.
- **H2: the real attempt at a consistent 60.**
  - A 2–3-month program: ≥300 generated mechanic families, 4 training nodes, a 32–64-box fleet, online RL with a horizon curriculum.
  - Cost ≈ $50k (minimum version) to ≈ $250k+ (Tufa-scale).
  - Even at Tufa scale, P(mean ≥ 64.5) is about 0.10.

## 1. Odds, with reasoning

**What the target means.**
- At a per-draw sd of 3.50, a consistent 60 is a draw mean of about 64.5. We sit at about 31: mem-repair-10x averages 30.85 and mem-unified-10x 30.91 (L3 FC 7; `runs/submissions.csv`).
- One extra level per hidden game is worth about 13–15 LB (L3 Q2), so 60 needs about +2.4 levels per hidden game.
- Efficiency alone would need 2.7× (P3) to 6× (P1) effective compute (G-H Q4). The gap has to come from new capability.
- Tufa may not average 60 either. At sd 3.5, the best of 166 draws sits about 9.4 above the mean, so their typical draw could be about 59.5 (evidence base §1.7; `_critic.md` C10).

**P(a fine-tuned model beats the fallback by 11-02), as a chain:**

| Step | P | Basis |
|---|---|---|
| A training node runs by 10-21 (Azure or Lambda) | 0.70 | Azure quota timeline unknown; a vendor guesses 1–4 weeks (W6 §7). Lambda needs only a card |
| Bring-up passes by 10-23: zero-delta overlay plus a ≥32k multimodal LoRA step | 0.60 | The recipe is measured only to 8k, text-only (W6 FC 2). Training against frozen dequantised experts is untested anywhere (W2 FC 5) |
| A checkpoint clears the offline gates at ≥ +3 offline | 0.35 | No public Flash-Next fine-tune has gained (L2 FC 6). H200 users got nothing (W5b FC 1). The closest shortest-correct analogue collapsed AIME 0.563 → 0.183 (G-G FC 1). On our side: on-policy data, served weights, exact render parity (G-C Q1) |
| Three LB draws confirm Δ* ≥ +0.5 | 0.60 | Within the 10x family, offline and LB correlate at only r = 0.44 (L3 FC 3) |
| **Product** | ≈ 0.09 | Swift/Tinfield overlays add about 0.03, for a total ≈ **0.12** |

**LB mean at 11-02** (the config we would actually pick):
- **Low, 29.5.** Every training stage dies and both picks are the fallback. Its true mean is about 31, and its four mem-repair draws ranged 26.9–38.2.
- **Mid, 31.5.**
  - The trunk fine-tune is not adopted (p ≈ 0.88).
  - The degrade track's MTP-head retrain (R1) ships with p ≈ 0.35, worth +1.1 to +1.9 LB (G-E Q4).
- **High, 40.** An RFT/RL checkpoint combines three gains, for 31 + ≈ 9:
  - it cuts tokens per action by 25–30% at equal decisions, worth +3.5 to +8.0 LB (G-H Q4, FC 9);
  - it removes part of the 9.2% of tokens spent on sandbox errors (G-A Q4);
  - R1 adds about +1.5.

## 2. What we build

### 2.1 Game factory: Flash-Next writes its own games

**Why it matters.** Diversity binds, not volume:
- our logs cover 151 distinct cleared levels, all on public games (L1);
- about 23 frontier cells remain after the 10-game holdout (W3 FC 8);
- 10 training environments degrade held-out performance and 100 help (AWM, measured on a 4B model only; W9 FC 1);
- 30 coverage-selected environments beat all 170 at fixed compute (W3 FC 6).

**Generator.** DreamTeam `game_creator` (Apache-2.0).
- It writes 8 levels per game, must replay a winning solver on every level, and has a 30-minute, 75-round budget (L2 FC 7).
- We drive it with our W4A16 Flash-Next on fleet boxes, through an OpenAI-compatible profile. Flash-Next both authors and plays the games, the closest reading of Tufa's "games that we've generated ourselves".
- **Licence.** The Qwen licence allows derivatives. Anthropic's and OpenAI's terms bar using outputs to train models without authorisation (W9 Q5, FC 8). So Claude writes infrastructure only (acceptance script, fleet, trainer glue), never game content. **User decision** if they want otherwise.

**Pools.**
- **arc-interactive** (MIT, 249 games, about 25–55 usable multi-level ones). Exclude the ft09, vc33 and ls20 copies by id prefix, since their SHA-256s differ from ours (W9 FC 7).
- **Generated games.**
- **ACCEL-style level mutations** of accepted games, re-verified by solver or small-grid BFS (W9 Q3).
- **Never in training:** the DreamTeam 25 (reserved for evaluation), H10, and the unlicensed sonpham and Training-Data sets (G-B Q3; W9).

**Acceptance (W9 Q4; a CPU script).**
- Checks, in order:
  - six or more levels;
  - at least two mechanics introduced;
  - no glyph or digit sprites;
  - solver replay reaches WIN on every level;
  - 50k random steps win no non-tutorial level;
  - novel against accepted games and against the public set.
- All generated code runs sandboxed.
- Difficulty: two base attempts per game. Keep games with at least one level at 0 < p < 1.

**Targets.** At least 100 training families in H1; at least 300 in H2.
- **Pilot gate:** 20 seeds must yield ≥25% and cost ≤1 GPU-h per kept game, or generation stops (W9 implication 1).
- **Expected yield:** 2–7 kept games per GPU-h (W9 Q5, UNVERIFIED).

### 2.2 The rollout fleet

**SKU.** `Standard_NC144lds_xl_RTXPRO6000BSE_v6`: 1 GPU, 264 GiB RAM, $5.50/h on-demand, $1.02/h spot (W6 FC 7). It runs the Kaggle stack offline: Pennyroyal, Intel W4A16, the albucino MTP drafter and our harness.

**Calibration first.**
- Decode must be within 5% of Kaggle's 895–909 tok/s at full batch.
- Two reference public-25 runs must land within 51.17 ± 3.28.
- Every comparison carries a platform term. Fleet runs are never pooled with Kaggle runs (G-B Q7).

**Logging on every fleet run** (flags, default off): `ARC3_VERBATIM_HISTORY=1`, raw replies, prompt and output token ids, behaviour logprobs, and routed-expert ids for RL (G-C Q3–4; W2 §3).

**Runtime patches** (test first, in `serving/runtime_patches/`):
- (a) An in-place LoRA-delta update: W += (α/r)·BA on about 300 BF16 tensors.
- (b) Routed-expert capture for rollout routing replay, ported from slime/miles `--use-rollout-routing-replay` (W2 FC).
- (c) Output logprobs under MTP speculation, checked against the `/generate` prompt-logprob path (G-C Q4).

**Unit costs:**

| Unit | Kaggle | Fleet (spot) | Source |
|---|---|---|---|
| Full 25-game, 121-min run | 2.3 GPU-h of quota | ≈ $2.5 | G-B Q3 |
| 1M generated tokens | 0.35–0.4 GPU-h | ≈ $0.41 | L1 Q6; W6 §8 |
| Frozen-state sample | 730–960 per GPU-h | ≈ $0.0012 | L1 FC 6 |
| Level-segment rollout (~40k tokens) | — | ≈ $0.016 | 43k tokens per cleared level (W3 implication 6) |
| Kept generated game | — | $0.15–0.50 (UNVERIFIED) | W9 Q5 |

**Quota.** 16–24 boxes need 2,300–3,500 vCPUs of spot quota in this family. **Approval needed**, plus an Azure quota request.

### 2.3 The trainer: one 8-GPU node

**Hardware.**
- **Node A:** ND96isr H200 v5 at $84.80/node-h, with 1,850 GiB RAM and 28 TiB NVMe (W6 §7, FC).
- **Fallback:** Lambda 8×B200 at $53.52/node-h, which carries a risk of sm100 kernel bugs in fla (W6 §8).

**Stack.** ms-swift Megatron, pinned to ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1 and transformers 5.19.0 (W6 §1). Use padding_free, PLE offload and cross-entropy fusion, plus context parallelism above 16k (W6 §3).

**Served-weights checkpoint.**
- Dequantise Intel's int4 g128 sym experts to BF16 offline. Assemble them with Intel's BF16 tensors, which are byte-identical to Qwen's (G-D FC 6).
- LoRA targets: GDN in/out, QSA qkv/o and the shared expert.
- Never the routed experts, the router or the QSA indexer (the indexer has no gradient; W6 §2).
- Keep the original `config.json` (W6 FC 8).

**Data.**
- **Rendering:** the closed four-transform `qwen3_8` path, id-exact on 25 of 25 windows (G-C Q1).
- **Context:** post-trim windows (31–85k) at CP4 if the day-1 smoke fits. Otherwise use Tufa-style short windows (the last 32k) and gate at real ~57k contexts. AWM lost 59% relative when trained short and tested long (G-G Q1).
- **Packing:** with verbatim-history fleet data, one window supervises all its turns, using about 12.6× fewer tokens (G-C Q2).
- **Hyperparameters:** LoRA SFT LR 1e-5 to 3e-5, RPO about 2e-6, global batch ≤32. One epoch per round, then a λ-merge sweep (G-G (d); W4 FC 6).

### 2.4 RL formulation

**Unit: a level segment.**
- Start from an exact state-lab rebuild of the harness context at a level's first turn, taken from base or previous-round runs, where the clear rate p satisfies 0 < p < 1. Full games have near-zero group variance (W2 §5).
- Run k = 8 rollouts per start through the real harness on fleet boxes.
- Weight starts by p(1−p) and keep the top reward-variance quartile. In RAGEN-2 that quartile reached 21.1% against 11.0% for the lowest (W2 §5; G-G FC 5).

**Reward (Tufa's list, L2 FC 1).**
- A clear scores min(1.15, (baseline/actions)²).
- Each sandbox error costs −0.02.
- On clears only, a length term of ±0.1 relative to the group's median clear tokens (W2 §6).
- Budget per segment: 60k generated tokens or 5× baseline actions, whichever comes first.
- No novelty reward: novelty stalls hold only 0.6% of tokens (G-A Q2).

**Credit assignment.** Every call is a training sample that carries the segment's advantage, broadcast ReSum-style (W3 §2). Add GiGPO step-level advantages at identical frames.

**Loss and stabilisers (W2 §1–3).**
- GSPO-token or CISPO, chosen by a 5-step smoke.
- **Rollout routing replay is required.** Plain GRPO crashed in all 3 settings without it, and GSPO without it collapsed in three sources (W2 FC 1–2).
- TIS truncation, plus a mask on negative-advantage, off-policy sequences.
- Aux loss 0, FP32 LM head, no KL term.
- Rollouts at T = 0.65 with top-p 1 and top-k −1. Staleness at most 4 steps.

**Auto-stop monitors:**
- reward std and entropy;
- distinct actions across different boards (a proxy for RAGEN-2's template collapse);
- no-tool-call rate (base 1.13%; G-B FC 12);
- train-vs-rollout k3 KL ≤ 0.004 (W2 FC);
- mean reply length.

**Cost per step.**
- 32 groups × 8 = 256 rollouts ≈ 10M rollout tokens ≈ 4 fleet box-hours ≈ $5.
- About 27M packed training tokens ≈ 0.5–1.9 node-h, at an unverified MFU (W6 §3).
- So the burst is trainer-bound: 30 steps ≈ 15–57 node-h.

**Semi-online fallback** (used whenever online plumbing isn't ready).
- Each round re-collects on-policy data with the current checkpoint, then runs RFT plus one DPO round on same-state pairs.
- Online and semi-online DPO roughly match GRPO (W2 FC 8).
- It needs neither importance sampling nor routing replay.

### 2.5 Deployment and evaluation

**Deploy** (G-D §3; W6 FC).
- Write new shard files holding only the changed tensors (5.8–6.7 GB) plus an index. Symlink the rest and leave `config.json` untouched.
- **G-MTP gate** (G-E Q5): greedy output identical with speculation on and off; accept ≥ base − 0.05 (2.63 in the full run); decode ≥ 0.98× base; step time within ±1 ms.

**Splits** (G-B Q1–2):
- **Train:** the T15 public games, arc-interactive and generated games.
- **Held out:** H10.
- **U25:** H10 plus 15 DreamTeam games. The other 10 DreamTeam games are a single-use final panel.
- **GH:** generated families held out by family, not by instance.
- Never score H10 in a run that also contains trained games (G-B FC 5).

**Gates before any LB draw** (G-B):
- **G0:** serving parity.
- **G1:** a canary of tool-call validity plus GPQA-100 and IFEval-150, backing off past 5–6 points.
- **G2:** a frozen-state screen on 420–470 states.
- **G3:** U25 ≥ base − 2.

**Promotion to LB draws** requires all three:
- a pooled paired gain of at least +3 offline† across at least two different panels (`make significance`);
- tokens per action ≤ 0.9×†;
- a curve residual of at least −3† (G-H implication 3).

Offline reports lead with wins and time-to-win.

**No exact repeats.** Fleet replicates always use different panels (H10, DT15, GH-A, GH-B) or different λ-merges. **Question for the user:** does the no-repeat rule also apply off Kaggle, where repeats cost about $2.5?

## 3. H1 plan, 2026-10-18 to 11-02 (UTC)

### 3.1 Phases

**P0. Decide and prep (Sun 10-18 → Mon 10-19)**
- **Work.**
  - Get approvals.
  - Open branches.
  - Build:
    - the `environments_dir` override;
    - the logging arm;
    - the acceptance script;
    - the dequantiser;
    - the name-keyed overlay tool (G-D);
    - the Swift overlay;
    - the `game_creator` endpoint profile;
    - the fleet image script.
  - Draft the designs for the weight-sync, routing-capture and logprob patches.
- **Kaggle (mellamoa).**
  - A DreamTeam-25 base panel (2.3 GPU-h). It closes G7 and defines U25.
  - Swift-overlay 20-minute paired panels (about 1.5 GPU-h) with G-D's gates: kill below 0.95× decode; promote at ≤0.85× tokens per action with ≥0.95× levels per action.
- **Gate (10-19 23:59).**
  - The DreamTeam panel has 0 harness errors and at least 5 games with a level at a 0.2–0.8 clear rate†.
  - A logging smoke returns token ids and logprobs.
- **Kill.** No training node booked by 10-20 18:00 → the no-node branch (§4).
- **Cost.** About 4 Kaggle GPU-h; $0.

**P1. Bring-up (Tue 10-20 → Fri 10-23)**
- **Node A.**
  - Install the pins and build the served-BF16 checkpoint.
  - Run LoRA forward/backward at 16k, 32k and 64k, multimodal, under CP, plus a 100-step throughput run.
  - Measure trainer-vs-SGLang KL on 200 logged decisions.
  - Export a zero-delta merge.
- **Fleet (1 → 4 boxes).**
  - Pennyroyal running outside Kaggle; calibration; 2 reference runs.
  - The 20-seed `game_creator` pilot.
  - Difficulty screens on arc-interactive.
- **Kaggle.** Push the zero-delta overlay notebook on 10-20, then draw it on the LB on 10-21. This tests the 9-hour symlink-overlay path.
- **Gate (10-23 12:00).**
  - The zero-delta overlay passes G-MTP.
  - The LoRA step runs at ≥32k, multimodal, with measured tokens/s.
  - The fleet calibrates within 5%.
  - The pilot's yield is reported.
- **Kill (10-23 23:59).** If the overlay or the LoRA step fails, trunk training ends for H1. Release Node A and switch to the fleet-only branch.
- **Cost.** Node A ≤ 40 node-h ($3.4k Azure / $2.1k Lambda); fleet ≤ 80 h (about $150, including on-demand boxes); Kaggle 2.3 GPU-h.

**P2. Corpus and RFT-1 (Thu 10-22 → Mon 10-26)**
- **Fleet (8–16 spot boxes).**
  - Generate games at scale, if the pilot passed.
  - Run the base model on the train pool with full logging, 4 attempts per game, to find PLR cells.
  - Resample frozen states at 0 < p < 1 with state-lab, k = 8–16.
- **Node A: RFT-1 on 4–8k whole-turn targets.**
  - Targets are engine-verified and at most 0.7× the median verified length†.
  - 30–50% of targets are median-length anchors.
  - Include stall→clear breakthroughs, verified by replay.
  - Prefer turns with error-free code.
  - Never compress a level's first turn or the turn after a death.
  - At most 2% of states per game.
  - Export λ-merges at 0.5, 0.75 and 1.0.
  - Test the RL loss and patches in Node A's idle hours.
- **Gate (10-26 18:00).** The best λ passes G0–G3 and the promotion threshold.
- **Kill.**
  - On a fail, one repair by 10-28 18:00: λ 0.5, LR halved, more anchors.
  - A second fail ends trunk training, and the finals go to the fallback.
- **Cost.** Node A ≤ 60 node-h ($5.1k / $3.2k); fleet ≤ 500 h (about $650). The Kaggle FT1 push on 10-26 (2.3 GPU-h) happens only if the gate passes.

**P3. RL rounds (Mon 10-26 → Fri 10-30)**
- **Round 2** (always, if P2 passed). Serve RFT-1 on the fleet, re-collect, then train RFT-2 plus one DPO round on same-state pairs.
- **Online burst.**
  - Runs only if, by 10-27 18:00, routing capture, weight sync and logprob parity all pass on the fleet with KL ≤ 0.004.
  - 30 steps from RFT-1 on starts from generated and arc-interactive games.
  - Optional Node B (≤48 node-h) runs the other loss (GSPO-token vs CISPO) in parallel.
- **LB.** FT1 draws on 10-27, 10-28 and 10-29.
- **Gate (10-30 12:00).** FT2 beats FT1, paired, on at least two panels and passes G0–G3 → FT2 gets the draws.
- **Kill.**
  - If an auto-stop monitor fires, keep the last good checkpoint.
  - If the burst hasn't started by 10-28, skip it.
- **Cost.** Node A ≤ 40 node-h; Node B ≤ 48; fleet ≤ 700 h (about $900); one Kaggle FT2 push.

**P4. Select and ship (Fri 10-30 → Mon 11-02)**
- **Work.**
  - A 9-hour, 110-game dress rehearsal of the FT finalist on an on-demand box (about $50; G-B Q7). It must show no OOM, no server stall and the deadline margin.
  - Top-up draws.
  - Finals under the Δ* rule (§3.3).
- **Release.** Node A is released by 10-31 after its last exports (≤10 node-h).

### 3.2 Calendar and the 16 LB slots (mellamoa only)

| Date | Build / train | Fleet | LB draw |
|---|---|---|---|
| Sun 10-18 | P0 tooling; approvals | — | Fallback anchor (frozen mem-repair-10x v1) |
| Mon 10-19 | Swift overlay; endpoint profile | Box 1 calibration | Fallback anchor |
| Tue 10-20 | Node A boot; served-BF16 checkpoint | Reference runs; pilot starts | Best harness-only candidate, or Swift if its panel promoted |
| Wed 10-21 | LoRA steps at 16–64k, CP | Pilot; arc-interactive screens | Zero-delta FT path |
| Thu 10-22 | Zero-delta merge check | Spot fleet to 8–16; collection | Base model with FT harness settings (verbatim + logging); the FT's true comparator |
| Fri 10-23 | **Gate P1 12:00; kill 23:59** | Collection; frozen resamples | Fallback anchor (mem-unified) |
| Sat 10-24 | RFT-1 training | Collection; patch tests | Second draw of the 10-22 config |
| Sun 10-25 | λ-merges; MTP check | λ evaluations on U25/GH | R1 head if ready, else harness-best #2 |
| Mon 10-26 | **Gate P2 18:00**; FT1 push | On-policy collection with RFT-1 | Reserve |
| Tue 10-27 | RFT-2 / **RL plumbing gate 18:00** | RL rollouts | FT1 #1 |
| Wed 10-28 | RL burst (repair deadline 18:00) | Rollouts | FT1 #2 |
| Thu 10-29 | Burst ends; FT2 export | FT2 vs FT1 | FT1 #3 |
| Fri 10-30 | **Gate P3 12:00**; FT2 push | 9-h rehearsal | FT2 #1, or FT1 #4 |
| Sat 10-31 | Release Node A | Final evaluations | FT2 #2 |
| Sun 11-01 | — | — | Top-up for the config nearest a threshold |
| Mon 11-02 | — | — | Last copy of the leading config by 12:00; finals by 23:59 |

LB re-draws of a frozen notebook are the repeats the user's alpha rule asks for, and scoring reruns use no GPU quota (CLAUDE.md). No offline notebook is ever run twice.

### 3.3 Final picks
- Use G-B's rule: Δ* = 0.6 × (FT draw mean − fallback draw mean).
  - Both picks FT if Δ* ≥ +0.5.
  - One of each if Δ* lies within ±0.5.
  - Both fallback if Δ* ≤ −0.5, or if any FT draw errored (G-B Q4).
- My stricter addition: keep one harness-only pick unless the FT passes the user's alpha rule. That means every FT draw above the fallback mean and Δ ≥ 2·SE.
- **Fallback:** mem-repair-10x v1 / mem-unified-10x (means 30.85 / 30.91), or the harness prong's leader frozen by 10-26. If R1 passes G-MTP and shows at least +0.1 accept on held-out games, it may ride on one fallback pick.

## 4. Branches when compute doesn't arrive

| Situation (decide by) | What runs | What H1 can still deliver |
|---|---|---|
| No Azure ND node (10-20 18:00) | Lambda 8×B200, after approval, with an fla sm100 smoke on day 1; the plan is otherwise unchanged | The same plan, 1 day late |
| No training node at all (10-23) | **Fleet only.** Game factory, U25/GH evaluation, and R1, the MTP head on one box: <40 GB with features cached, so it fits (G-E Q3). Replicate evaluation for the harness prong lets it detect +5 offline (G-B Q7) | +1 to +2 LB if R1 ships; a corpus and validation set ready for H2 |
| No Azure at all | **Kaggle only, mellamoa at 6 GPU-h/day.** DreamTeam panel; Swift/Tinfield overlays; the upward elasticity test (`demo_minutes` 242, about 4.3 GPU-h; G-H Q5); the harness prong | Evidence on whether post-training helps in our harness; no trunk training |

**What each failure still leaves behind:**

| Fails at | What we keep |
|---|---|
| Compute | Overlay tooling, logging arm, acceptance pipeline, DreamTeam U25 |
| Bring-up | All of the above, plus a calibrated fleet, a game corpus and evidence on whether the Swift/Tinfield post-trains help in our harness |
| RFT-1 | Verified decision datasets with token ids and logprobs, a measured negative result, and R1 |
| RL burst | FT1 if it passed, and a debugged RL loop (routing replay, weight sync) for H2 |

## 5. H1 cost summary

| Item | Cap | Expected | Where | Approval |
|---|---|---|---|---|
| Node A | 150 node-h (1,200 GPU-h): $12.7k Azure / $8.0k Lambda | 70–110 node-h | Azure ND H200 v5 or Lambda B200 | Yes (paid) |
| Node B (burst only) | 48 node-h (384 GPU-h): $4.1k / $2.6k | 0–15 node-h | Same | Yes (paid) |
| Fleet | 1,400 box-h: ≈ $1.9k spot, worst case $7.7k on-demand | 600–1,000 box-h | Azure NC RTX PRO 6000 | Yes (paid, quota) |
| Storage and egress | ≈ $0.5k | ≈ $0.3k | Azure P40, blob | Yes |
| Kaggle | ≈ 15 GPU-h (mellamoa) | ≈ 12 | Kaggle | Each push |
| LB draws | 16 slots | 16 | Kaggle | Each submission |
| **Total** | **≈ $19k (Azure) / ≈ $13k (Lambda nodes)** | **≈ $8–10k** | | |

## 6. Rules, accounts and approvals

**Accounts.**
- The rules say "you cannot enter or submit from multiple accounts", and rule 3.5.a disqualifies anyone who submits from more than one (G-F G17).
- snowiers already holds its own LB entry: draws of 28.73, 29.54, 34.04 and 28.06 that are absent from mellamoa's list (G-B FC 11). That is a live disqualification risk.
- **Recommendation:**
  - Stop all snowiers submissions now.
  - The user decides whether to raise the existing entry with Kaggle.
  - Budget no snowiers quota. Whether compute-only use is allowed isn't addressed by the rules (G-F, UNVERIFIED), and the fleet replaces those 5 h/day at about $2.5 per full run.

**Prize eligibility.**
- Winners must open-source the model and the training code (rules 2.5 and 2.8).
- Staff won't interpret the Qwen licence (W5b FC 5).
- A private Kaggle model is enough for scoring.

**Needs the user's approval:**
- Azure ND and NC spot quota requests, and a spend cap per phase.
- A Lambda account as the fallback.
- Every Kaggle push and every LB submission.
- Uploading private Kaggle models.
- Any non-Flash-Next game authorship.
- Exact replicates off Kaggle (§2.5).

## 7. H2: after 11-02, toward a consistent 60

**Precondition.** Find out whether the LB keeps scoring after the deadline. Late submissions are unverified here. Without them, the arbiter becomes held-out generated families plus DreamTeam plus H10, which within our family track the LB at only r = 0.44 (L3 FC 3).

| Stage | Dates | Work | Gate | Kill | Cost |
|---|---|---|---|---|---|
| H2-A Intel | 11-03 → 11-16 | Read Tufa's promised release (G-F G9), NVARC3's if they reach gold, and any from Chen. Diff against our pipeline; settle base model, rewards and context length | A memo naming the 3 largest method differences | No measurable target (no LB and no validated proxy) → run only H2-min | ≈ $0 |
| H2-B Game factory v2 | 11-10 → 12-07 | ≥1,000 accepted games in ≥300 families; generation at the policy's frontier (ACCEL/SPADE-style regret selection; W9 Q3); the public action mix (13 mixed / 7 click / 4 keyboard) | A 50-family GH set whose base clear rates resemble LB difficulty (about 0.55× public-like) | Yield below 15%, or more than 50% of games single-mechanic | ≈ 600–2,000 box-h, ≈ $1–3k |
| H2-C Online RL at scale | 11-17 → 01-31 | 4 nodes (32 GPUs; miles-style SGLang routing replay or ms-swift with an external fleet) and 32–64 fleet boxes. Horizon curriculum from 1 level to the full game (TTI; W3 §2); train short to long, 16k → 32k → 64k (Tufa); cross-level rewards (ORBIT/LaMer, W8 FC 4; W3 §1); MTP co-trained each promotion (MiniMax top-K KL; G-G FC 8) | Every 50 steps: GH +1 level per game; each milestone ≥ +5 LB over 3 draws, if the LB exists | Two milestones in a row with no GH gain → stop and revisit the base model with Tufa's facts in hand | H2-min: 1 node + 16 boxes, 8 weeks ≈ 6.5k GPU-h ≈ $45–70k. H2-Tufa: 4 nodes + 32–64 boxes, 10 weeks ≈ 27k GPU-h ≈ $180–285k plus about $27k fleet |
| H2-D Expert training | After a LoRA plateau | Full-parameter training including routed experts, with INT4 QAT (LMSYS/slime fake-quant; W2 §3) | Trainer↔INT4 logprob error flat over 200 steps | QAT drift grows | ≥8 nodes; priced only if H2-C works |

**H2 odds** (my judgement):
- H2-min: P(mean ≥ 64.5) ≈ 0.03; P(≥ 45) ≈ 0.2.
- H2-Tufa: P(mean ≥ 64.5) ≈ 0.10; P(≥ 50) ≈ 0.3.
- Tufa reached about 69 at the best of 166 draws with more than 20× our compute, and its typical draw may be near 59.5 (§1).

## 8. What this plan deliberately does not do

- **Train routed experts or the full model in H1.** That forces an int4 re-quantisation and breaks the overlay deployment (G-D implication 3). Reserved for H2-D.
- **Plain GRPO, or GSPO without routing replay.** 3 of 3 GRPO settings crashed, and GSPO collapsed in three sources (W2 FC 1–2).
- **INT4 PTQ rollouts with a BF16 trainer.** Their mismatch grows during training (W2 §3). We train against the dequantised experts we serve.
- **Teacher or frontier text, OPSD or SDPO.** Their terms bar training on outputs; the 2.4T teacher is text-only; privileged self-distillation hurt every thinking model tested (W10 FC; G-G FC 2).
- **World-model SFT and novelty rewards.** Neither has shown decision gains, and novelty stalls are 0.6% of tokens (W8 FC 2–3; G-A Q2).
- **Selective down-weighting of reasoning tokens.** It caused a no-tool-call collapse, 1,474 of 4,767 turns (W5b FC 3).
- **Switching the base model.** The 27B scored 14.51 offline, and the user's rule requires a base with LB ≥ 30 (finetune plan 2026-10-17).
- **Test-time training on Kaggle.** The served weights already take 69.85 GB of the card's 96 GB (W5 FC 7).
- **One training job spanning nodes in H1.** Our documented LoRA path is one node (ms-swift). The multi-node recipes (NeMo RL, miles) run at 4k and don't map the MTP (W6 FC 4). Nodes A and B run independent jobs.
- **Adversarial co-evolving generators before 11-02.** Each designer step costs a generation, a verification and rollouts (W9 Q3).
- **Training on DreamTeam, H10 or unlicensed games.** They are reserved for evaluation, or barred by their missing licence.
- **Efficiency as the main bet.** It needs 2.7–6× effective compute to reach 60 (G-H Q4).
- **SGLang LoRA serving.** It is undocumented for MoE, quantised bases and speculation, so we merge instead (W6 §4).
- **Exact-repeat offline runs, and snowiers submissions.**

## 9. Biggest risks

1. **Compute arrives late or never.** Azure quota has no published timeline, and the plan's upside is entirely downstream of it (W6 §7).
2. **The trainer at the deployed shape.** A 32–85k multimodal step under CP, mcore-bridge on dequantised experts, and real tokens/s are all untested (`_critic.md` G5).
3. **The RL-tuned model degrades.** Shortest-correct SFT collapsed in the closest analogue; a no-tool-call collapse has been seen on Kaggle; there is no base checkpoint to fall back on (G-G FC 1; W5b FC 3).
4. **Generated games are too easy or too far off-distribution.** Generators simplify their own tasks, DreamTeam's games are about 4× shorter than the public ones, and the hidden set is out of distribution by design. Expect meta-skill transfer only: far-OOD +0.16 to +0.24 in VHD-Play (W9 Q1–2; W3 FC 1).
5. **Evaluation can't resolve the gains.** Sixteen draws at sd 3.5 detect only about +5 LB or more, and offline↔LB r = 0.44 (G-B Q4).
6. **Novel fleet patches fail silently.** Routing capture, weight sync and logprobs under speculation are new code; a silent bug makes "on-policy" RL off-policy.
7. **MTP drift.** Each −10% of effective throughput costs about −2.7 offline (G-H Q4).
8. **Rules and money.** The snowiers entry, Qwen-licence eligibility and an uncapped spend.
9. **Bandwidth.** One user with agents across four parallel tracks.
10. **The fleet platform.** Spot evictions, and the BSE card versus Kaggle's card (G-B Q7).

## 10. Project-rule compliance

**Arms**, each on its own branch with default-off flags, run on the leader and stacked, never repeated:
- `arm/ext-games`: `environments_dir`;
- `arm/rl-logging`: verbatim history, raw replies, token ids, logprobs, routing ids;
- `arm/overlay-loader`: Kaggle mount and symlink view;
- `arm/ft-rft1`, `arm/ft-rft2`, `arm/ft-rl1`: the FT models' paths behind a model flag;
- `arm/mtp-r1`.

**Tooling branches:** `tooling/gamegen`, `tooling/fleet`, `tooling/trainer`.

Each arm gets a PR with hypothesis, changes, tests, results and compute cost. Research and run records go straight to main. CONFIGURATION.md and the area READMEs are updated in the same PR.

**Agnostic code.**
- Rewards and data never branch on a game's id, grid, palette or level count.
- The model never sees minutes or game counts.
- † thresholds are named in each PR.

**Claims.**
- Every improvement claim needs `make significance` on the score files.
- An alpha needs repeated LB draws.
- Offline reports lead with wins and time-to-win.
