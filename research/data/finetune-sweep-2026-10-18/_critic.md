# Critic: completeness check of the 2026-10-18 fine-tuning sweep

Completeness critic, 2026-10-18. I read all 14 notes files in this folder, including every fact-check section. I also read the prior plan (`research/reports/finetune-plan-2026-10-17.md`), the `rsi-proposals-2026-10-15.md` ground rules and the Model prong / FT-1 entries in `arc-agi-v4/arms.md`. I ran three spot checks:

1. **Vision path in the trainer.** mcore-bridge registers a multimodal Flash-Next loader: `src/mcore_bridge/model/mm_gpts/qwen4_exp.py` (30 lines, `Qwen4ExpMMLoader` with `visual_cls=Qwen3_5Vit`, present since #174 on 2026-08-26). So image training exists in code. No ms-swift doc or example trains images, or uses context parallelism, for this model.
2. **Longer wall-clock budgets.** No arm in `kaggle/arms.json` has ever run longer than the standard 121-minute budget. The only explicit `demo_minutes` values are 20 and 5. Real upward compute elasticity has never been measured.
3. **Tinfield-1.** `badtheorylabs/Tinfield-1` is a full-weight BF16 post-train of Flash-Next (HF, 2026-09-21, 131 shards, Qwen community licence). Its card self-reports Terminal-Bench 4.0 at 33.0 vs 29.0 for the base and DeepSWE v1.1 at 62 vs 58.7. The method is not disclosed, and nobody in the sweep evaluated it.

## Verdict

The sweep covers the user's three asks well:
- **General pedagogy:** W1, W2, W7, W8, W10.
- **What labs do:** W4, W3, W5.
- **Our facts:** L1, L2, L3, W5b, W6, W9.

The fact-checks caught most of the errors. Four kinds of hole remain:
- **The plan's central bet is untested.** That bet is "same decision, fewer tokens → more score". Its conversion rate has never been measured on our own runs, and a cheap test exists.
- **Topics conflict on where to cut tokens.** The long-token tail and the "safe to compress" states may be different states.
- **Day-1 feasibility rests on items nobody can check without the node.** These are the node itself, long multimodal sequences, exact rendering and logprobs. Some of them can be checked today on the laptop.
- **Free evidence was skipped.** Two existing post-trained Flash-Next models are untested in our harness.

Taken together, the notes say LB 60 "consistently" (mean ≈ 64.5) is out of reach by 11-02 for every measured lever. The plan should say so and optimise LB mean per GPU-hour.

## 1. Gaps

### Critical

**G1. Training compute may not exist in time, and there is no fallback branch.**
- Why it matters:
  - Every training step needs the Azure node.
  - W6 found no official quota timeline. A vendor estimate is 1–4 weeks, the H200 spot meter equals on-demand, and new subscriptions may be refused.
  - There are 15 days left. No topic defines a no-node branch or go/no-go dates.
- Research:
  - Have the user confirm today the ND H200 v5 quota and the start date. Pre-arrange Lambda 8×B200 ($53.52/h) as the fallback.
  - Write dated kill points. Example: no zero-delta merged-LoRA boot reproducing today's tok/s on Kaggle by about 10-23 → stop the training prong, and make the final picks from harness arms.
  - Check whether R1 (MTP-head self-distillation) can train on one 96 GB GPU. Its trunk would be frozen, with hidden states dumped by prefill. If it can, there is a Kaggle-only weights lever.

**G2. Real upward compute elasticity is unmeasured, and the R3 payoff rests on it.**
- Why it matters:
  - L3's "−25% tokens per action → +7.6 offline" comes from sim2.
  - sim2 has no real-data check above s = 1.15 and over-predicted the 27B by about 10 points.
  - No run has ever had more than 121 minutes (spot check 2). The only conversion evidence is cross-model (the 27B).
- Research:
  - Run the leader offline with a scaled time budget (`demo_minutes` 161 and/or 242; about 3–4.5 Kaggle GPU-h each). Compare levels and score with sim2's 58.5 (s = 1.33) and 64.4 (s = 2), and read `score_timeline`.
  - This is a conservative proxy for fewer tokens per action, because shorter thinking would also slow context growth.
  - LB side, optional: one or two throttled submissions at 0.5× compute tell P1 from P3. My sim2 extrapolation gives about −12 LB under P1 and about −18 under P3.

**G3. Nobody measured where the long-turn token tail sits.**
- Why it matters:
  - L3 says the cut must come from turns ≥ 5k tokens, which hold about 70% of tokens.
  - W7 says to compress only cleared-level, easy states and leave hard or stuck states alone.
  - If the tail sits mostly on stalled or never-cleared levels, the "safe" R3 touches well under half the tokens. Its effective compute gain is then much smaller than the −25% headline.
- Research: one laptop query over L1's `decisions2_joined.json` and the `benchmark.json` files.
  - Tabulate tokens in turns ≥ 5k by level class: cleared on-route, cleared off-route, stalled (≥ 15 actions without a new board) and never-cleared.
  - Compute the share of total tokens that W7's recipe would act on.

**G4. The R3 yield and achievable cut at easy states are unmeasured.**
- Why it matters:
  - The literature gives −10 to −13% for naive shortest-of-16.
  - The only local measurement (W7, 700 samples) is at hard states: the shortest success is 0.68× the median success, and right replies are not shorter than wrong ones.
  - Whether −25% is reachable at all depends on easy-state numbers nobody has.
- Research: restore state-lab from `archive/tooling/state-lab`. On about 200 on-route cleared-level states, at k = 8, measure (about 1–2 Kaggle GPU-h):
  - the share of resamples that reproduce the logged action or board;
  - the shortest-to-median verified length ratio;
  - the same two numbers under `reasoning_effort=low`;
  - the yield of the prefill-only truncation probe (40/60/80%).

**G5. Training at the deployed shape has not been tested.**
- The shape is a 32–98k-token, image-bearing, multi-turn sample with LoRA on GDN/QSA/shared expert.
- Why it matters:
  - 93% of calls have prompts over 16k (L1), and the recipe is measured only up to 8k, text-only.
  - Images are 12% of prompt tokens (about 9 per request).
  - A vision path exists in mcore-bridge code (spot check 1), but CP with `all_gather`, `padding_free` with GDN state resets, and image tokens under CP have no doc or test.
  - Training tokens/s is unpublished at any length (W6 estimate: 2–7 node-hours per 100M tokens, at an unverified MFU).
- Research, a day-1 node smoke:
  - one 64k multimodal LoRA step at CP2 and CP4 (peak memory);
  - a 100-step throughput run;
  - packed-vs-unpacked loss parity on GDN.

  The result picks between whole-window, segment and short-context data. Decide it before building datasets.

**G6. Exact token parity between training renders and served prompts is unverified.**
- Why it matters:
  - Every example depends on it. ms-swift documents `preserve_thinking` only for inference, and training-time rendering of historical thinking is undocumented (W1 fact-check).
  - The leader normalises history: no blank lines or think tags (L1).
  - Yield-continued replies and `_arc3_control` keys also need handling.
- Research, closable on the laptop today with no GPU:
  - Pull `chat_template.jinja` from the Kaggle or HF model.
  - Render about 100 logged requests through it plus the Qwen processor, and through ms-swift's `qwen3_8` + `qwen3_coder` encode.
  - Diff against the logged `prompt_tokens` and `image_tokens`.
  - Document how a 2,048-token yield continuation appears in history.

**G7. Flash-Next's pass rate on external games (M0(c)) is unmeasured, and so is whether they work as a proxy.**
- Why it matters:
  - The data plan depends on it: W9's PLR-lite, W3's frontier cells and R4 at scale.
  - So does the screening plan: L3's external-game proxy, needed once public games are trained on.
  - The DreamTeam games have never run in our harness.
- Research:
  - Add the `environments_dir` override.
  - Run one 25-game DreamTeam panel plus 20–25 arc-interactive games with 6 or more levels on Kaggle (about 3–4 GPU-h).
  - Report per-level clear rates and the count of 0.2–0.8 cells, and whether scores sit near LB (~31) or public (~51) difficulty.
  - DreamTeam baselines are solver lengths, so report levels, not RHAE.

### Important

**G8. Evaluation design breaks once public games are in training.**
- Why it matters:
  - The project's offline score (25-game run, 10-win target) becomes contaminated.
  - A 10-game held-out run has a paired sd of about 7–9. That is my arithmetic from L3's per-game sd of 27.2, deflated by its own 4.5/5.4 shared-compute factor. So +5 needs about 13 paired runs.
  - About 16 LB submissions remain (deadline 11-02, 23:59 UTC), at a per-draw sd of about 3.5.
  - No topic has shown that the proposed frozen-state screens (right-move rate, tokens per decision) predict full-run score.
- Research:
  - Compute the 10-game subset noise from the 26 runs.
  - Decide explicitly between rsi-proposals' "never train on public games" and the prior plan's 15/10 split.
  - Draft a dated LB calendar, for example at least 3 draws per finalist plus the fallback.
  - Back-test the screen: does the state-lab right-move rate rank past harness arms the way their full-run scores do?

**G9. The top-team methods are still unknown, and they are the only evidence that 60+ exists.**
- Why it matters: the user's premise is that only fine-tuning reaches 60. Nothing in the sweep establishes that.
- Unread sources:
  - Tufa's 10-09 X post, seen only via a competitor's site.
  - Any Tufa or Chen posts or podcasts from October.
  - Kaggle forum searches for LoRA, GRPO, distill, Tufa and "60". W5b did not run them.
  - The meaning of `driessmit1/qwen3-8-27b-fp8-hf` (uploaded 08-18).
- Research:
  - A Sonnet browser agent on X, LinkedIn, the Kaggle forum and the ARC Daily Digest.
  - Line up the leaderboard event history (Tufa 56.5 → 68.9 on 10-13; Chen's +4 steps) against model and tool releases.

**G10. Existing post-trained Flash-Next models are untested in our harness.**
- The candidates:
  - **Tinfield-1:** full weights; self-reported +4 on Terminal-Bench and +3.3 on DeepSWE.
  - **Swift-1.5 W4A16:** its 20-minute smoke has been pending since `arms.md:674`.
  - **huikang's merged LoRA.**
- Why it matters: they are free evidence on whether a post-trained Flash-Next turns into more levels per wall-clock in our harness. Tinfield is also an existence proof that a small lab can post-train this MoE for agentic gains.
- Research:
  - Diff one Tinfield routed-expert tensor against the base.
    - If only non-expert tensors changed, overlay its BF16 trunk onto the Intel W4A16 shards (no requant) and run a panel.
    - Otherwise use the NVFP4 probe build, or skip it.
  - Run the Swift-1.5 smoke and one full run.

**G11. R1, the MTP draft head, is under-researched.**
- Why it matters:
  - rsi-proposals ranked it the safest weights change (accept length 2.68 → 3.0–3.2).
  - It is required after any trunk change.
  - L3 values +10–15% throughput at +2 to +4 LB.
- What's missing:
  - Recipes and expected domain gains for self-distilling MTP/EAGLE heads on the model's own reasoning traces.
  - Whether it trains on one GPU.
  - Measured tok/s per +0.1 accept length at 12–14 streams. Arms notes say 14 streams is near compute-bound.
- Research:
  - A literature pass (EAGLE-3/HASS-style head training, ms-swift `--mtp_num_layers`).
  - Derive the accept-length → tok/s slope from existing config-sweep runs with relaxed acceptance.

**G12. Logprob support on the Pennyroyal SGLang build is unknown.**
- Why it matters: W10's prefill-only SDPO, W2's trainer↔server KL gate (≤0.003–0.004) and any later IS-corrected RL all need top-k input-token logprobs. The model runs MTP speculative decoding and has a hyper-connection forward.
- Research:
  - Read the build's `return_logprob` / `logprob_start_len` / `top_logprobs_num` path for `qwen4_exp`, then do one Kaggle smoke.
  - Meanwhile, turn on `ARC3_VERBATIM_HISTORY=1`, raw-reply logging and logprob logging in every data-collection run now (L1).

**G13. There is no stall taxonomy or novelty-by-outcome measurement.**
- Why it matters: it decides three things:
  - R4's target (win-condition vs rule rationales);
  - the novelty rewards (W2, prior plan stage 3);
  - the side model (W5, StochasticGoose-style).
- Research:
  - A log query over the 85 runs: new-pair and new-board rate on stalled vs cleared levels.
  - Audit about 30 stalls against their later clears (goal-inference error vs mechanic error).

**G14. Nobody has tested whether compressed thinking costs working memory.**
- Why it matters: thinking stays in history (WRITEUP.md:69). R3 shortens what later turns read, which could raise repeated probes or drop facts.
- Research:
  - In state-lab, replace history thinking with truncated or shortest-sibling versions at frozen states, then compare right-move rate and distinct actions.
  - After training, track the repeated-probe rate on cleared levels.

**G15. Rollout throughput on the node is unmeasured (L1, W2, W3).**
- The configurations in question are FP8 TP4/EP4 ×2, or W4A16 on sm90, with our request shapes.
- Why it matters: it sizes any ReST round on external games and the game_creator pilot.
- Research: run `kaggle/bench_corpus.json.gz` on day 1. Optionally test the Azure RTX PRO 6000 BSE spot box with the Pennyroyal wheelhouse as an exact-stack rollout machine (W6).

**G16. game_creator yield with an open model is unmeasured.**
- Why it matters:
  - W9's "≥100 mechanic families" needs 50–150 new verified games.
  - Yield brackets are weak: ScriptDoctor isn't one-shot, and AWM-Qwen3.5 is about 59% end to end.
  - The generator has only ever run on GPT-5.5, Gemini or Opus.
- Research: W9's 20-seed pilot with Flash-Next on the node (yield, tokens per attempt, glyph/number violations). Accept that the route may not fit before 11-02.

**G17. Second-account compute was not checked against the rules.**
- Why it matters:
  - Data budgets assume 11 GPU-h a day across mellamoa and snowiers.
  - No topic read the rules' account clauses, or checked whether snowiers is a team member.
  - If that quota can't be used, the data budget shrinks by about 45%.
- Research: have a browser agent read the eligibility and accounts section of the rules, and confirm before budgeting snowiers' quota.

### Minor

- **G18. Hidden-set composition.** The technical report assigns the fully private 55 to the competition (W5 fact-check). Kaggle says the public LB uses about 50% of the test data. What "110" means, and the size of the final private half, set the per-draw variance and the final-pick logic. Research: read the Kaggle data and overview pages.
- **G19. Lab pipelines not covered.** MiniMax-M2's interleaved-thinking training is directly relevant to training with retained thinking across tool calls. Also missing: Kimi K2.5/K3, Seed/VAPO, OLMo 3 delta learning and DeepSeek-V4 OPD compute. Research: read the MiniMax-M2 report first.
- **G20. A key caution was read only as an abstract.** "Rethinking OPSD for Thinking Models" (2607.05184) is the main warning against W10's top recommendation. SMRC-SD, π-Distill, RAGEN-2 tables, AgentGym-RL and AWM Fig. 4 points are also abstract-only. Research: read 2607.05184 in full before committing to SDPO.
- **G21. Perception and tool-error targets were not examined.**
  - No topic measured how many stalls come from misreading frames, or the token share of failed sandbox code.
  - Both are cheap, verifiable targets: auto-generated frame → object-list pairs, and RFT that keeps error-free code.
  - Research: a log query on sandbox-error turns; read Kaggle thread 743060.
- **G22. No general-ability canary was chosen.** One should run on our stack within quota, for example tool-call validity on logged states plus a small GPQA or LiveCodeBench slice through the same server.
- **G23. No pedagogy primer exists.** The user asked for a progression from general advice to labs to the plan. The material is spread across W1, W2, W3, W7, W8 and W10. The final report must order it; no new research is needed.

## 2. Contradictions

| # | Between | Issue | Resolution hint |
|---|---|---|---|
| C1 | W8 vs L3 lever table, W2 §6, W5 Q5, prior plan stage 3 | W8 says the model already explores (91% new pairs, 87% new boards; "shortfall is volume, not choice", arms.md:183-185). L3 reads the 96% statistic as "explores too little"; W2 rewards "board not seen before"; W5's side model "targets the 96%". | Read the 96% as "an untried new-board move existed", which is almost guaranteed at 91% new pairs. Run G13. If stalled and cleared levels have equal novelty rates, drop novelty rewards and the side-model rationale, and aim R4 at hypothesis and goal quality and at probes per minute. |
| C2 | W1 ("R3 low-risk"), W2, W4 vs W7 fact-check, W7 §5, RSI toy | Others call shortest-correct self-SFT safe. But on R1-Distill-7B, Arora's rejection-sampling SFT (shortest of 8 own samples) took AIME24 from 56.3 to 18.3. At our hard states right replies aren't shorter (P = 0.445). In the RSI toy, Connect-4 accuracy when answered fell from 0.43 to 0.25. | Treat R3 as medium risk. Select relative to each state, mix in median-length correct replies, keep the LR conservative, and save several checkpoints with a λ-merge sweep. Gate on the frozen-state right-move rate and the no-tool-call rate, not only on tokens. |
| C3 | L3 vs W7 | L3: the cut must come from the long-turn tail (70% of tokens in turns ≥ 5k). W7: compress only cleared, easy states; leave hard states alone. | Measure G3 first. W7's own data shows length barely predicts success at hard states (terciles 10.4 / 10.0 / 13.2%). So relative shortest selection at hard states with k ≈ 20 may be acceptable, given the G14 check. |
| C4 | Prior plan, W3, W4 (30–50% self-replay) vs W1, W2 | One side treats replay of the model's own outputs as the forgetting guard. W1 and W2 cite Thinking Machines (self-sample SFT drifts) and Retaining by Doing (one-shot Self-SFT forgets), and prefer data regenerated each round, LoRA, λ-merges or an OPD anchor. | The failures were unfiltered T = 1 chat samples or full-FT runs. PAPRIKA's no-regression used a tiny LR. Run a 0% vs 20% replay pair at a low LR, judged by tool-call validity plus the canary (G22). Use OPD only if vLLM boots by about day 5. |
| C5 | W3 vs W1 | W3 recommends an LR of 1e-6 to 2e-6 (full-FT numbers from PAPRIKA, ORBIT and VHD). W1 recommends 3e-5 to 2e-4 for LoRA (the recipe uses 1e-4; LoRA ≈ 10× full FT). | State the LR as a full-FT equivalent and multiply by about 10 for LoRA (Thinking Machines). Applying W3's numbers to LoRA would undertrain it 10–100×. |
| C6 | W1 ("never trajectory-level DPO over whole games") vs W1's own ETO table, W8 (PAPRIKA RPO) | W1 cites trajectory-level ETO adding 4–11 points while step-level DPO collapsed. PAPRIKA's per-task trajectory pairs beat SFT-only on 6 of 6 groups. | Whole-game pairs are impractical at 55k contexts anyway. As a test-first step after SFT, use per-level segment pairs: one pair per level, with an NLL term, low β and LR ≈ 2e-7 full-FT equivalent. |
| C7 | W4 vs L2 (Tufa), W1 (ProLong), W3 (ReSum) | W4: don't assume short-window training is harmless (GLM-4.5 "unlearned" long context). Tufa trains on shorter sequences on purpose; ProLong says short SFT keeps long-context ability; ReSum trains through segments. | GLM-4.5's finding was for multi-stage RL. Prefer real post-trim windows (31–85k), which are contexts deployment actually shows, if G5 allows. Otherwise train short and gate at 57k with state-lab frozen states. |
| C8 | L1 vs W1, W6 | W1 and W6 want segment packing with loss on every turn in a trim window, saving up to about 98% of compute. L1 shows the leader normalises history, so a turn as context (normalised) differs from the turn as target (generated). | Either deploy with `ARC3_VERBATIM_HISTORY=1` after a full run shows no score loss (six arms already set it), or train only the final turn of each window at much higher compute. Decide this with G5 and G6. |
| C9 | rsi-proposals-2026-10-15 vs prior plan, L1, W3 | rsi-proposals: "Never train on the 25 public games. They are the eval set." The prior plan and L1 use a 15-train / 10-test split as the minimum. ARC calls this task-specific overfitting, and the private set is out of distribution. | Pick one explicitly (G8). If public logs are used, every offline comparison must be on the 10 held-out games plus external games, and the 25-game score and the 10-win target stop being valid for trained models. |
| C10 | L2 (sd 5.0, best-of-N), prior plan (sd ~5) vs L3 (3.0–3.3) and its fact-check (3.50) | Draw noise sets both the sample sizes and the top-team "typical draw" inference. | Use the 10x-family pooled sd of 3.50 (df 7). Then a consistent 60 needs a mean of about 64.5. Best-of-166 inflation is about 9.4 points, which puts Tufa's typical draw near 59.5 if its noise is like ours. |
| C11 | W10 vs W4 fact-check | W10 says it is unverified whether ms-swift accepts a different teacher prompt, so a custom KD loss may be needed. W4's fact-check found ms-swift v4.4.3 supports OPSD (`teacher_prompt`, GRPO path, needs vLLM) and a LoRA `disable_adapter()` teacher with no second load. | Online OPSD is in ms-swift but needs vLLM for Flash-Next. W10's offline variant (teacher logits from SGLang prefill) still needs a custom loss plus G12. Choose based on whether vLLM boots. |
| C12 | Prior plan (Qwen3-4B or 27B dense), W1 (MoE sibling), W6 (Axolotl QLoRA on the real model, 8 single-GPU jobs) | Three different wind tunnels. Axolotl's ~120 GiB figure was measured on a 288 GB B300; it uses NF4 experts plus routed-expert LoRA, unlike our deployment. | If Axolotl fits on an H200 with expert LoRA off, use the real model. Otherwise skip the wind tunnel and rank with 2–3 short ms-swift runs. A dense sibling can't rank MoE target choices. |
| C13 | W3 vs L1, W8 | W3: "1–2k level attempts a day on Kaggle is feasible." L1 measures about 120 game-runs a day across all quota. W8: about 1.9k attempts before 11-02, with nothing left for evaluation. | Use L1's measured 12 game-runs per GPU-hour. Use state-lab's all-in 730–960 samples per GPU-hour (not 1.2–1.6k). Name the unit each time (game-run, level attempt or frozen state). |
| C14 | L3 vs W5 fact-check | L3 treats the hidden set as 110 games, with about 55 on the public LB. W5's fact-check: the technical report reserves the fully private 55 for the competition and the semi-private 55 for API testing. | Close G18. If the final ranking rests on about 55 games, final-pick logic should favour lower-variance configurations. |
| C15 | Prior plan, W5b implication 5 vs W4, W6 | Re-quantize after merging (50–100 GPU-h, "AutoRound base not trainable directly") vs no re-quant for non-expert LoRA. W6's own 9-hour AutoRound figure was not an experts-only run. | Settled: non-expert LoRA needs no requant. Only an MTP check and a zero-delta boot. Strike the stale items. |
| C16 | W4 (general benchmarks as canary only) vs W3 (regression gate, SIMA 2) vs prior plan (< 2-point gate) | Different forgetting gates. | Gate on game metrics and tool-call validity. Treat the general canary as a back-off trigger, for example > 5 points → a λ-merge toward base. |
| C17 | L1 vs L1 fact-check, W9 | "No DreamTeam solutions; rollouts only" vs `solve_level` embedded in 25 of 25 games, with 20 of 25 matching their baselines. | Use the solvers as verifiers and hindsight-route sources (R4) and for curriculum difficulty, not as SFT text: it is off-policy (W1 type 4). |
| C18 | W5b (H200 users: no gain; "MoE makes training unstable") vs Tinfield-1 card (my check) | One full-weight post-train of Flash-Next self-reports agentic gains. Forum failures may reflect recipes rather than the architecture. | Not decisive (self-reported, method unknown), but it argues for G10. Weight the forum reports by their method: off-policy teacher data, down-weighted reasoning, and RL without routing replay. |
| C19 | L3 (P3 ceiling ~63 as "this setup's physics") vs L2/W5b (Tufa 68.91, Chen 67.49) | Two teams already exceed the modelled ceiling on these hidden games and this hardware. | The ceiling is sim2's set of levels our agent never cleared, a property of our agent, not of the games. Use it as the limit of efficiency-only levers, not of the benchmark. |

## 3. Strongest cross-topic signals

1. **LB 60 "consistently" is beyond every measured lever by 11-02.**
   - At sd 3.5 it needs a mean of about 64.5.
   - Even the optimistic hidden-set model saturates near 63 on efficiency alone.
   - L3 puts realistic 15-day outcomes at LB 33–45. Applying W7's realistic −15 to −30% tokens per action to L3's lever table gives about +3 to +10 LB.
   - The plan should say this plainly, target the LB mean, and rank levers by LB per GPU-hour (L3, L2, W7, W5b).
2. **No public evidence that fine-tuning Flash-Next on ARC helps.**
   - Every notebook scoring ≥ 20 runs stock weights.
   - Franzen won M2 without training.
   - At least two H200 teams got nothing.
   - The only first-hand top-team signal is Tufa's RL on self-generated games, with rewards for per-level score, code execution and reasoning length, at more than 20× our compute.
   - So gate the training prong on cheap, falsifiable steps with dated kill points, and keep a harness fallback for both final picks (L2, W5, W5b, W3).
3. **The deployable path is narrow and cheap.**
   - LoRA only on BF16 non-expert modules (GDN, QSA, optionally the shared expert), trained against the dequantised int4 experts.
   - Merge into the Intel shards and keep the original config.json.
   - No AutoRound re-run; MTP acceptance and greedy spec-on/off identity are the remaining checks.
   - The first gate is a zero-delta merge that reproduces today's offline score and tok/s (W1, W2, W4, W6).
4. **The best-supported training signal is the model's own engine-verified samples at frozen decision states, regenerated each round.**
   - That means RFT/ReST/Iterative-SFT and Qwen3's Stage 3 precedent.
   - Not teacher or frontier text (licence-barred, and the 2.4T is text-only), not scripted-solver rationales, not world-model SFT, not novelty bonuses.
   - State-lab (exact rebuilds, Kaggle backend, ~730–960 samples per GPU-hour all-in) is the ready pipeline (W1, W2, W3, W7, W8, W10, L1).
5. **R3 is the primary lever but is not low-risk, and its payoff is unvalidated.**
   - Realistic cut: −15 to −30%.
   - The closest literature analogue collapsed accuracy (Arora RS+SFT).
   - At our hard states length doesn't predict correctness.
   - Where the token tail sits (G3) and real elasticity (G2) are unmeasured.
   - Measure G2, G3 and G4 before building the training set (L3, W7, W1).
6. **Context length and rendering are the main data-format decision.**
   - 93% of calls exceed 16k; the recipe is tested to 8k.
   - The leader's normalised history blocks exact segment packing.
   - Tufa also trains short and generalises long.
   - Day 1: run the 64k multimodal CP step (G5) and the laptop parity check (G6).
   - From today: turn on verbatim history and raw-reply/logprob logging in every collection run (L1, W1, W4, W6, L2).
7. **Diversity, not volume, limits transfer, and our pool is tiny.**
   - Only 151 distinct cleared levels from 25 public games, and about 23 frontier cells after the holdout.
   - The private set is out of distribution by design.
   - Published transfer needs about 100 coverage-selected environments.
   - External games are mandatory, and their base pass rate (G7) is the first measurement to make (L1, W3, W5, W9, L3).
8. **The model doesn't lack novelty; it lacks probes per unit of compute and correct hypotheses.**
   - Drop novelty rewards.
   - Make R4 hindsight-filtered by engine replay toward an actual clear.
   - Use cross-attempt filters (the ORBIT/PAPRIKA signal offline) (W8, W2 fact-check, W10).
9. **RL is out of scope before 11-02, but its prerequisites should be logged now.**
   - MoE RL needs routing replay; GSPO alone collapsed in three independent sources.
   - INT4 PTQ rollouts drift.
   - The only single-node recipe is ms-swift GRPO-LoRA with vLLM colocated at 1k + 8k tokens.
   - Start logging behaviour logprobs now so a later stage is possible (W2, W4, W6).
10. **Evaluation is the bottleneck.**
    - About 16 LB draws remain at sd 3.5.
    - Offline correlates with LB at only r = 0.44 within the family.
    - 20-minute panels don't predict score, and a held-out 10-game set can only see effects of about +10.
    - Screen on mechanism (tokens per decision at a matched action, frozen-state right-move rate, no-tool-call rate, stuck-state action diversity), then full runs, then pre-planned LB draws (L3, W4, W5, W5b).

## 4. Already settled by the fact-checks (carry into the plan; don't reopen)

- **DreamTeam solvers:** they ship as `solve_level` in all 25 games; 20 of 25 match their baselines.
- **Re-quantization:** none needed for non-expert LoRA.
- **Overlay size:** about 72 GB unless the changed tensors are resharded into new files.
- **Config:** keep the original `config.json` on every transformers version.
- **Single-node RL:** ms-swift GRPO-LoRA on one 8-GPU node exists (vLLM colocated, smoke-test sized).
- **Routing replay:** GSPO is not a substitute for R3 routing replay on MoE.
- **Aux loss:** NeMo's "aux loss severely degrades" note applies to its AutoModel backend; setting it to 0 is still sensible.
- **Qwen3.8-2.4T:** text-only, so it is not a teacher for our multimodal contexts.
- **Noise:** LB sd is 3.50 for the 10x family; offline run sd is about 3.15–3.2.
- **Token shares:** never-cleared levels take 22–28% of tokens in leader runs (32% overall). "50–60%" refers to stalled levels.
- **State-lab throughput:** 730–960 samples per GPU-hour all-in, not 1.2–1.6k.
- **Deadline:** 2026-11-02, 23:59 UTC; 1 submission a day; 2 final picks.
- **Swift-1.5:** a general efficient-thinking fine-tune, not an ARC one. On Terminal-Bench its median length fell while its mean rose.
- **Chen's OPD-32B and DFlash models:** AIMO artifacts, not ARC ones.
- **Arora RS+SFT:** its collapse is the closest analogue to R3 and belongs in R3's risk estimate.
