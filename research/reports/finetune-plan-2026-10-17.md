# Fine-tuning toward a consistent 60: model, training scheme and data (2026-10-17)

**Request (the user, 2026-10-17):**
> i think we have to switch tactics now. i am trying to get azure compute for fine tuning/training of these models. the new
> goal is very high, at 60 now. even while having good rng, we got a max of 38.2. now we need to consistently score high now,
> which we have seen can be only done by fine tuning.

The user asked for three things:
- other model candidates that could replace Qwen3.8-Flash-Next ("better, more recent, or have a easier to train
  architecture"), which "HAS to have a base score of atleast 30";
- the training disciplines to use, and the training and testing data we'd need;
- a combined training scheme like MiniCPM's.

**Method.** Four research agents covered models, the leaderboard landscape, training methods and data. Their raw reports are
in [`../data/finetune-research-2026-10-17/`](../data/finetune-research-2026-10-17/). I then checked the claims this plan rests
on most:
- the ms-swift training recipe;
- the Flash-Next licence;
- the competition rules;
- our own measurements.

Unverified items are marked as such.

## Answer

1. **Keep Qwen3.8-Flash-Next. It is the only candidate that meets the ≥30 rule.**
   - Every public notebook we found scoring 20 or more on the leaderboard uses it, and our six draws average about 31.
   - Nothing released up to October 2026 fits on the 96 GB card and is clearly better.
   - The models that are easier to train (Qwen3.8-27B, Qwen3.6-27B, Gemma 4 31B) have no evidence of reaching 30.
   - Our own Qwen3.8-27B run scored 14.51 offline against 54.69 (2026-10-17).
2. **Training Flash-Next is possible, but it needs a big node.**
   - ms-swift publishes the only recipe: Megatron LoRA SFT and GRPO on 8 GPUs.
   - Measured LoRA SFT peaks: 77.8 GiB per GPU at 4k tokens and 93.3 GiB at 8k. GRPO needs about 135 GB per GPU.
   - Our agent prompts are about 55k tokens (median). H100 80 GB is therefore too small, and 8× H200 (141 GB) is the
     minimum useful Azure node.
   - Even on H200 we must either train on shortened decision windows or add context parallelism, which the recipe doesn't
     cover.
   - Flash-Next has no base checkpoint. After training, the W4A16 quantization and the MTP draft head must both be redone.
3. **What training should target, from our own measurements:**
   - **Tokens per action.** The 27B run needed the same actions per level as Flash-Next (49.6 vs 49.8) but twice the tokens
     per action (1,793 vs 872). It played half as far and scored 14.5 vs 54.7. Thinking length is a huge lever.
   - **Exploration on stuck levels.** About half of all tokens go to stuck levels, and in 96% of stuck-level decisions an
     untried action would have reached a new board.
   - Moves on levels that do clear are already near-optimal.
4. **Combined scheme, adapted from MiniCPM.**
   - MiniCPM's gain came from what it mixed into the decay phase: high-quality and SFT data mixed with general data. The
     shape of the WSD learning-rate curve didn't matter.
   - For us: one mixed SFT stage on a short re-warmup, plateau and decay schedule. Mix verified game decisions with 30–50%
     replay of the model's own general reasoning outputs. The best data (shortest correct decisions) goes into the decay
     phase.
   - Hyperparameters are ranked on a small sibling model first, MiniCPM's "wind tunnel".
   - Then one rejection-sampling round with the new model (ReST-EM, retraining from the base each time). RL comes only if
     time allows.
5. **Time is the binding constraint.**
   - The competition ends around 2026-11-02, 16 days from now, with one leaderboard submission a day.
   - One SFT round, re-quantization and 3–5 leaderboard draws is the realistic scope. RL isn't.
   - Keep mem-unified as the fallback final pick.
6. **Nobody has shown that fine-tuning works in this competition.**
   - The top two, Tufa Labs (68.91) and Yi-Chia Chen (67.49), are 18+ points above #3 (50.37). Neither has said what model
     they use or whether they fine-tune.
   - One rank-38 competitor guessed that "part of that is some fine tuning". That is speculation.
   - A score of 60 is clearly reachable, but we don't know the route. Fine-tuning is our best bet, not a proven one.

## 1. Model candidates

| Rank | Model | Fits 96 GB card | Evidence for base LB ≥30 | Trainability | Notes |
|---|---|---|---|---|---|
| 1 | **Qwen3.8-Flash-Next** (2026-08-26) | Yes (W4A16 + PLE in host RAM) | **Yes**: ours 26.9–38.2 (mean ~31); all ≥20 public notebooks | Heavy: ms-swift Megatron LoRA / GRPO, 8 GPUs; no base checkpoint | 125B language + 51B PLE table + 4B MTP head, 6B active; 512 experts (10 + 1 shared); GDN + sparse attention hybrid; hyper-connections; licence qwen-community-1.0 |
| 2 | Qwen3.8-27B (dense, Apache-2.0) | Yes | **No**: 14.51 offline (ours) | Easy (well-trodden Qwen3.5/3.6 paths; unverified for 3.8) | Same tokenizer and template as Flash-Next; loses on tokens/action, not on decisions. Good "wind tunnel" |
| 3 | Qwen3.6-27B (Apache-2.0) | Yes | Unverified. One secondary blog claims Tufa runs it now; Tufa's Milestone-1 entry on it scored 1.21 | Easy; a community ARC-AGI-3 QLoRA exists (no scores) | Only worth a look if Tufa's open-sourced solution (after Nov 2) confirms it |
| 4 | Mistral Small 4 119B-A6B (Apache-2.0) | NVFP4 only | Unverified, likely no (GPQA 71 vs Flash-Next 92) | Axolotl; official EAGLE head | — |
| 5 | Gemma 4 31B (Apache-2.0) | Yes | Unverified, likely no (Milestone-1 entries on it scored ~1) | Easy; base checkpoint exists | — |

**Too big for one 96 GB card:** GLM-5.3-Flash (320B), DeepSeek V4.1 Flash (550B+), Kimi K3, MiniMax M3, Nemotron 3 Ultra.
Step 3.7 Flash (198B) is marginal even at 4-bit.

**Weaker or unproven:** Nemotron 3.5 Super VL/Bolt (released 2026-10-15), gpt-oss-120b, Kolibri-1.

**Qwen 4 27B** was announced on 2026-09-22 with no weights yet, so it is too late for 11-02.

**Verified details:**
- **ms-swift recipe** ([best-practice page](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)):
  - LoRA SFT on 8 GPUs with TP2/EP4/PP2; peak per GPU 75.6 / 77.8 / 93.3 GiB at 2k / 4k / 8k tokens.
  - Offloading the PLE table saves about 16% at 4k. The table is then frozen, which doesn't matter for LoRA.
  - GRPO LoRA uses about "8*135G" with a vLLM built from source.
  - Context parallelism isn't mentioned.
  - Requires transformers ≥5.16, mcore-bridge, Transformer Engine and flash-linear-attention ≥0.5.2.
- **Licence** ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)):
  - Fine-tuning, derivative works, distribution and sublicensing are allowed if the notice is kept.
  - A separate licence is needed only for model-as-a-service or AI-work-assistant businesses.
  - It does not name CC-BY 4.0 or OSI licences.

## 2. What fine-tuning has to change (our evidence, not the literature)

| Lever | Measurement | What training can do |
|---|---|---|
| Tokens per action | Flash-Next 872 tokens/action. Thinking is 87–92% of every reply. The 27B at 2× tokens/action played half as far (52 vs 116 levels) | Train on the **shortest correct** reply for the same decision (R3). Same decisions at fewer tokens means more actions in the 9 h |
| Stuck-level exploration | ~50–60% of tokens go to stuck levels. In 96% of stuck-level decisions an untried action reached a new board. A forced restart doesn't beat continuing | Train on decisions that **reached a new board or cleared a level** from stuck states (hindsight relabelling, R4). Later, RL with a novel-board bonus |
| Cleared-level play | Moves are already near-optimal (81% on the hindsight-shortest route) | Nothing to gain. Don't degrade it: keep replay data |
| Harness facts | The model uses computed facts; it ignores advice and optional structure | Keep the harness. Train with the same prompts the notebook serves (mem-unified) so the weights learn to use those facts |

## 3. Training scheme (combined, MiniCPM-style, adapted to continued training)

**What MiniCPM did** ([arXiv 2404.06395](https://arxiv.org/abs/2404.06395)):
- **Schedule.** WSD = warmup, a long constant-LR stable phase, then a short exponential decay (about 10% of steps beat 2.5%).
- **Decay-phase mix.** At the start of decay they mixed high-quality and SFT-style data into the pretraining data, then
  ran a short SFT.
- **Result.** At 1.2B, "decay-mix + 6B SFT tokens" beat "plain decay + 12B SFT tokens": MMLU 49.6 vs 47.9, MATH 10.5 vs
  7.3. GSM8K was the exception (31.8 vs 34.4).
- **Curve vs mix.** The curve itself isn't the source of the gain: cosine reaches lower loss at a fixed length. WSD's real
  benefit is that you can anneal from any stable-phase checkpoint.
- **Wind tunnel.** Hyperparameters were tuned on a ~9M model and transferred with muP.
- **OLMo 2** does the same thing as "Dolmino" mid-training, spending the last 5–10% of compute on high-quality data.

**What transfers to us:**
- **Mixed data, not pure SFT.** Train on game decisions and replay data together in one stage. This is also the guard
  against forgetting. Replay should be the model's **own** outputs on general reasoning and tool-use prompts, so its
  thinking style stays as it is.
- **The schedule.** A short re-warmup to a low LR, a plateau, then about 10% exponential decay. Save a checkpoint before
  the decay, so one run gives several models to evaluate. The highest-quality slice goes into the decay phase: the
  shortest correct decisions and verified stuck-level breakthroughs.
- **The wind tunnel.** Rank LR, LoRA rank, replay share and epochs on a small sibling with the same data, and transfer the
  rankings, not the values. Candidates: Qwen3-4B (the RSI notebook's model) or Qwen3.8-27B (same template).
- **What doesn't transfer:** the muP values and the 20B-token scale. Our data is about 20–100M tokens.

**Stages:**

| Stage | What | Data | Compute (estimate) | Gate |
|---|---|---|---|---|
| 0 | Eval set and wind tunnel | Validation and test splits (below); small-model sweeps | <50 GPU-h | Pipeline reproduces the base model's offline score through the trained-model path (merge → W4A16 → MTP) with no change |
| 1 | **Mixed SFT / expert iteration** on Flash-Next (LoRA, Megatron) | Verified decisions from our logs and external games; shortest-correct resamples (R3); stuck-level breakthroughs (R4); 30–50% self-generated replay | ~150–400 GPU-h on 8× H200 | Held-out games: more levels at matched wall-clock and fewer tokens/action. General reasoning drop under ~2 points |
| 2 | One rejection-sampling round with the stage-1 model (ReST-EM: retrain from the base) | Re-collected trajectories on train games, including external ones | ~200–400 GPU-h (collection can use Kaggle quota) | Same as stage 1, against stage 1 |
| 3 | RL: GRPO/GiGPO with per-level reward, novel-board bonus and per-action length penalty; Router Replay for MoE stability | Train games | ~400–800 GPU-h | **Not before 11-02.** No public RL precedent on a Qwen3-Next-class model or on ARC-AGI-3 |
| 4 | Deployment | Re-run AutoRound W4A16 with calibration on agent transcripts; retrain or verify the MTP head on rollouts of the fine-tuned quantized model | ~50–100 GPU-h | MTP acceptance and tok/s within ~5% of today (≈690 tok/s real); offline score at least equal on the held-out games |

**Training on long contexts.** Our real requests are ~55k tokens median and 78k at p90. The recipe was measured up to 8k.
Two ways to handle this:
- **(a)** Train each decision with the context the harness shows after a trim: system prompt, memory card, recent turns
  and the current board, capped at ~8–16k. Loss falls only on the assistant turn.
- **(b)** Add Megatron context parallelism on more GPUs.

Recommendation: (a) first, because the harness already writes compact memory. Check on held-out games that the model still
behaves at 55k.

## 4. Data

**Our logs** (inventory script in the data folder):
- 150 Kaggle kernel versions, 110 of them full 25-game runs.
- About 3,100 game-runs, 7,785 level clears, 369 games won, 378M completion tokens.
- 89 runs kept full request logs (about 180k model calls, 6,083 level clears).
- Almost all are Flash-Next W4A16 on Franzen-based harnesses.
- **All are on the 25 public games.**

**External games in the same engine** (so our harness and scoring run unchanged):

| Source | Games | Licence | Solver | Use |
|---|---|---|---|---|
| DreamTeam generated games (`competitors/dream-team`) | 25 × 8 levels (+ generator) | Apache-2.0 | `solution.json` replays to a win | Train / validation |
| arc-interactive (`competitors/arc-interactive`) | ~250 (many tiny) | MIT | none | Train / validation, after filtering by difficulty |
| sonpham-org/arc-agi-3 (`competitors/arc-agi-3`) | ~60 custom (10 levels each) | **none** | scripted wins in changelog | Validation only (we can't redistribute it) |
| ARC-AGI-3-Training-Data (`competitors/ARC-AGI-3-Training-Data`) | 434 (PuzzleScript, MiniGrid, homemade) with 390 solvers; **includes augmented copies of all 25 public games** | **none** | yes | Ideas and solvers only (no licence); drop the public copies |
| PuzzleScript corpus ([script-doctor](https://github.com/smearle/script-doctor)) | ~950 scraped, ~500 validated | MIT tool, per-author games | tree search | Rule-inference warm-up, needs an adapter |
| Boxoban, MiniGrid/BabyAI, our `game-reasoning` games | unlimited | Apache / MIT / ours | BFS / bot | Warm-up only |

**Ways to generate trajectories** (cost estimates from the data report):
- **Our own model on external games, with rejection sampling.** Cost ≈ 80 GPU-h per 1k full-game attempts on the
  RTX PRO 6000. This can run on the **free Kaggle quota** (mellamoa 6 h + snowiers 5 h a day, roughly 4–5 full runs a day).
- **Scripted solvers plus a hindsight-written rationale.** Nearly free, but the text isn't the model's own style, so use it
  sparingly.
- **Shortest-correct resamples of logged decisions (R3).** The state-lab tooling already resamples a frozen decision k
  times on our Kaggle server. Keep the shortest reply that takes the logged successful action, which the game engine can
  verify by replay.

**Splits.** Given the deadline, this departs from the data report's "external only":

| Split | Content |
|---|---|
| Test (never trained on) | **10 of the 25 public games**, chosen across difficulty: 2 always won, 4 sometimes won, 4 never won. We already have ~15 baseline runs on them for paired tests |
| Train | The other 15 public games' logged decisions (about 3.6k level clears) + external-game trajectories + replay |
| Validation | 15–20 external games held out by mechanic family (about 8 DreamTeam, 6 sonpham, 6 arc-interactive) |

Contamination guards:
- Drop the Training-Data and sonpham copies of public games.
- Exclude by id and by mechanic family.
- Check engine SHA-256s against the public manifest.

The leaderboard (the private 110) stays the final judge.

**How much evaluation is needed.**
- One full offline run has an sd of about 2.7 on the mean, so detecting +5 needs ~5 runs per arm. Leaderboard draws (sd ~5)
  need ~16.
- So the screen is a paired 20-minute panel on validation games with 3 seeds. Then 2–3 full runs for the best checkpoint,
  then leaderboard draws.
- Report wins and time-to-win first.

## 5. Rules and licences

From the Kaggle rules page, read by the landscape agent on 2026-10-17:
- External data and pretrained models are allowed if public and reasonably accessible.
- Prize winners must open-source code, model and weights (CC-BY 4.0, per the OSI Open Source AI definition), and deliver
  training and inference code. A pretrained model with an incompatible licence is exempt from the licence grant.
- A fine-tuned Flash-Next is distributable under its licence (notice kept).
- **Training data without a licence (sonpham, Training-Data) shouldn't go into released weights.**
- The rules don't forbid training on public games, but ARC calls it overfitting. Here it can only help on the private set if
  it teaches general behaviour.

## 6. Compute ask for Azure

- **Node:** 1× **8× H200 141 GB** node (ND H200 v5 class) is the minimum for Flash-Next LoRA at useful sequence lengths.
  8× H100 80 GB fits only ≤4k-token samples. GB200/B200 would do too.
- **Hours, order of magnitude** (agent estimates; no source gives these for this model):

  | Step | GPU-h |
  |---|---|
  | Stage 0 | <50 |
  | Stage 1 | 150–400 |
  | Re-quantization and MTP | 50–100 |
  | Optional stage 2 | 200–400 |
  | **Total** | **~450–950** |

- **Cost:** H100 on-demand is about $12 per GPU-hour, per a third-party blog. H200 pricing wasn't found, so check Azure's
  price sheet.
- **Kaggle quota** covers data collection and all offline evaluation.
- **Software:** ms-swift + mcore-bridge + Transformer Engine + flash-linear-attention, transformers ≥5.16. AutoRound for
  the re-quantization. Our SGLang Pennyroyal build for serving.

## 7. Schedule to 11-02 (16 days)

| Days | Work |
|---|---|
| 10-17 → 10-20 | Data collection: our harness on DreamTeam and arc-interactive games (Kaggle quota); R3 resamples of logged decisions; the split and contamination guards. In parallel, set up ms-swift on the Azure node and reproduce the base model's offline score through merge → AutoRound → MTP |
| 10-20 → 10-24 | Wind-tunnel sweep on the small sibling; stage-1 SFT on Flash-Next; merge, re-quantize, check the MTP head; upload as a private Kaggle model |
| 10-24 → 10-28 | Offline 2-h runs: held-out public games + validation games, against mem-unified at matched wall-clock (2–3 runs). If it wins, take leaderboard draws (1/day) |
| 10-28 → 11-02 | Optional stage 2 if stage 1 worked; final picks = the two best **mean** configurations, with mem-unified as the fallback |

## 8. Risks

1. **The model was already RL-tuned and has no base checkpoint.** SFT can degrade its thinking. Self-generated data and replay
   are the guards.
2. **Context-length mismatch.** It trains at ≤16k tokens but plays at ~55k.
3. **Re-quantization and MTP drift** can erase gains or cut throughput. Self-distilled MTP retraining has regressed
   acceptance in one community experiment.
4. **Overfitting to public games.** A higher offline score may not carry to the private 110.
5. **Noise.** Leaderboard draws vary by ±5, so one good draw proves nothing. That was the user's own lesson with the 38.22.
6. **Toolchain risk.** Only one recipe exists (ms-swift), with unpinned dependencies and a source-built vLLM for GRPO. A
   failed setup costs days we don't have.

## 9. RSI test (the user, 2026-10-17: "also do a test of the RSI notebook, just to see")

`mellamoa/v4-rsi-test` runs the `arm/rsi` loop (PR #177) on a Kaggle RTX PRO 6000:
- Qwen3-4B, thinking off, on two verifiable toy tasks (tic-tac-toe best move, Connect Four tactic);
- baseline, then up to 4 rounds of: sample 8 answers per task, keep the verified ones, train a fresh LoRA, run two
  evaluations on a pinned 200-task suite, and promote only on a paired, repeated gain.

### 9a. Result

It stopped after round 2 at its time budget. It used 118 GPU-minutes. Logs are in `runs/kaggle/v4-rsi-test/v2`.

| Version | Status | Mean score | Answered within 256 tokens (c4 / ttt) | Correct when answered (c4 / ttt) |
|---|---|---|---|---|
| base | baseline | 0.100 | 0.24 / 0.57 | 0.43 / 0.17 |
| round 1 | **promoted** | **0.200** | 0.73 / 0.75 | 0.29 / 0.25 |
| round 2 | **rejected** (CI not above 0) | 0.232 | 0.84 / 0.88 | 0.25 / 0.28 |

- **The loop works end to end on the competition GPU.** It ran collection, LoRA training, paired evaluation, the gate
  (promote and reject) and lineage.
- **The score doubled because the model learned to answer in fewer tokens.** Training on its own shortest verified answers
  cut how often it ran past the budget. Accuracy when it did answer barely moved.
- **This is the "same decision in fewer tokens" lever from §2 at toy scale.** It says nothing yet about Flash-Next on
  ARC-AGI-3.
