# How to fine-tune with no resources (2026-10-11)

Paths are relative to `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4`. "scratch" means `(session scratch, not kept)/`. Anything marked (est.) is an estimate; the rest is sourced.

## Verdict

You can't do a normal fine-tune of Qwen3.8-Flash-Next for free.
- The documented LoRA recipe needs 8 GPUs at 70–75 GiB each ([ms-swift](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html)).
- Kaggle's TPU v5e-8 has 128 GB, which can't hold even the 181 GB int4 checkpoint.

The only free route is a small LoRA on the model's BF16 parts, trained on Kaggle's 96 GB RTX PRO 6000.
- It needs about a week of custom engineering.
- It may break the rule on how RTX machines can be used.
- It should add 0 to +3 leaderboard points (est.), which we can't confirm before Nov 2.

The cheaper model-side test is to try a fine-tune someone has already published.

**Free GPU budget:** about 217 GPU-h until Nov 2, of which about 140 come before an Oct 27 freeze. That pool is shared with every harness arm, and a full run costs 2.3 GPU-h.

## Options, ranked

### 1. Swap in a published Flash-Next fine-tune (no training)
- **What:** use `phuongncn/arc3-qwen38-swift-w4a16-autoround` in place of the base model and keep our MTP drafter. This is Swift-1.5, trained with RL to think less, in the same W4A16 format. Also list the files of `huikang/qwen-38-flash-next-finetune` (public, 15 versions, no description) on CPU.
- **Compute:** a 25-min panel (0.5 GPU-h), then 2 full runs (4.6 GPU-h).
- **Steps:**
  1. Read the "Swift Open License". It has a revenue cap, and the winner rules require an open-source model (`scratch/rules.txt`).
  2. If the licence passes, run the panel, then the 2 full runs.
- **Expected gain:** +0.5 to +1 on average (est.).
  - For: on Aider it used a median of 6,991 thinking tokens against 17,646, at a similar pass rate ([prismix](https://prismix.dev/news/e4614f039e53)).
  - Against: rellik13 found it "no better per token, ~35% slower" (`arms.md`).
  - Against: speed-ups haven't turned into score for us. streams14 decoded 15% faster and scored 46.38 against 46.25 (`arms.md:197,605`).
- **Risks:** the licence makes it ineligible; our drafter accepts fewer of its tokens.
- **Kill test:**
  - Check the licence first (0 GPU).
  - Kill on the panel if decode speed is below 0.9× the base or accept length is below 2.4.
  - Confirm only on score at matched wall-clock against control (`make significance`), not on token counts.

### 2. LoRA spike on Flash-Next, trained on generated games (arms.md M2)
- **What:**
  - LoRA r16 on the BF16 modules only: the GDN projections (`in_proj_qkv`, `in_proj_z`, `out_proj`), attention `q/k/v/o`, and `shared_expert`. The regex must not match the int4 experts' `gate_proj`.
  - Router, int4 experts, MTP, the PLE table and vision stay frozen.
  - SGLang's `qwen4_exp` has no runtime LoRA support, so a patch in `serving/runtime_patches/` merges W + (α/r)BA into the BF16 tensors at load time. No re-quantization is needed.
  - The adapter is about 61 MB and ships as a public dataset.
- **Data:**
  - Collect trajectories on DreamTeam's 25 generated games (Apache-2.0, `competitors/dream-team/environment_files_generated/`). That keeps all 25 public games as a clean eval, which is M2's own rule (`arms.md:565-570`).
  - Drop levels cleared in 80% or more of runs, weight the rest by inverse clear rate, and prefer segments where the model corrected a wrong idea.
  - Train whole level segments as consecutive 8k windows, carrying the GDN state and attention cache forward (detached).
  - Put loss on assistant tokens only.
- **Compute:**
  - 3–4 collection runs: 7–9 GPU-h.
  - Gate session: 1.5 GPU-h.
  - Training: about 1,100 windows, 6–14 GPU-h at 20–45 s per window (est.).
  - Evaluation: 3 runs, about 7 GPU-h.
  - Total: about 22–32 GPU-h for one round. A second round doesn't fit.
  - Engineering: 4–7 days (est.).
- **Blockers I checked in the source code:**
  - The auto-round quantizer marks the model as not trainable, so we need our own training loop.
  - The Marlin int4 kernel has no backward pass, so the Triton one must be forced (`auto_round_extension/triton/triton_utils_zp/dequant.py:170-190`).
  - The attention layers pick tokens in a Python loop, one query token at a time, and each pass waits on the GPU (`scratch/dl/modeling_qwen4_exp.py:722-758`). At that speed one example takes 2–12 min (est.), so the loop must be vectorized first.
  - The GDN kernels load through a hub wrapper with a fallback. With internet off they may silently fall back to slow torch code. Their backward kernels are tuned for Hopper and may not compile on sm120.
  - Memory: frozen weights take about 67.7 GiB. Peak is about 82–85 GiB with an 8k gradient window; 16k runs out of memory (est., `scratch/hdr_sum.py`).
- **Expected gain:** 0 to +3 on the leaderboard (est.), with a real chance of a loss.
  - Self-training gains of +5 to +19 points came from weak models on in-distribution tasks, and mostly in round 1 ([ReST-EM](https://arxiv.org/html/2312.06585), [RFT](https://arxiv.org/abs/2308.01825), [SOAR](https://arxiv.org/html/2507.14172)).
  - The hidden games are "intentionally out-of-distribution" ([paper](https://arxiv.org/pdf/2603.24621)).
  - LoRA mostly on attention underperforms ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)), and `shared_expert` is the only dense MLP we can train.
- **It can't be confirmed in time:**
  - 3 runs against about 30 baseline runs at sd about 4 gives a standard error of about 2.4 (est.).
  - Under your repeated-runs rule, the best we can show by Nov 2 is "not worse".
- **Risks:**
  - Account moderation: "use of RTX for any other activity could result in moderation action" (`scratch/rules.txt`). The collection runs face the same question.
  - The training stack never runs.
- **Kill test:** run it only after a Kaggle host confirms this use is allowed. Internet off, wheels attached as a dataset, 1.5 GPU-h or less. Kill if any of these happen:
  - a 58k-token forward pass takes over 60 s;
  - it runs out of memory at 8k;
  - the fallback GDN code runs instead of the fla kernels, or their backward fails;
  - a random nonzero adapter merged in SGLang does not match transformers+PEFT (at least 99% top-1 agreement). A zero adapter proves nothing.
  - the measured step time implies more than 14 GPU-h of training.

  After training, kill if accept length falls by more than 0.2, or if the public-game score at matched wall-clock is more than 1 point below control.

### 3. Action-effect CNN on the laptop (free, low ceiling)
- **What:** a small 4-layer CNN that predicts which keys and clicks change the frame. Labels come from the local engines plus the DreamTeam games, generated on CPU. The harness feeds its output to the model as computed facts.
- **Compute:** 0 GPU until it passes the offline gate, then about 6 GPU-h. Plus 2–4 days of development.
- **Expected gain:** 0 to +1 (est.).
  - Removing every no-op click is worth only +0.4 to +0.8.
  - StochasticGoose scored 12.58% in the preview and 0.25% on the official benchmark ([datacamp](https://www.datacamp.com/zh/blog/arc-agi-3)).
  - Our segmenter lost 3.0 F1 on held-out games (`arms.md`).
- **Kill test:** hold out 5 games. It must reach AUC of at least 0.85 and flag at least half of the no-op clicks, while wrongly flagging at most 5% of useful ones.

### Dropped
- **LoRA on public-game trajectories:** it contaminates the eval.
- **TPU small model or helper LLM:** 4 GB or less of spare VRAM, and rellik13's TPU LoRA "scored badly".
- **Retraining the MTP drafter:** a ±15% speed change is worth about 0 points.
- **Distilling frontier-model traces:** they cover the same 11 public games, come from other harnesses, and the providers' terms may forbid it.
- **Test-time weight updates during the 9 h run.**

## Recommended next 3 actions
1. **Today, 0 GPU:**
   - Ask on the competition forum whether offline training in an RTX notebook is allowed.
   - Check the Swift licence against the open-source rule.
   - List the files in huikang's model with `kaggle models`.
2. **If Swift's licence passes:** run its panel and 2 full runs (about 5 GPU-h), judged on score. Meanwhile, on the laptop under `nice`:
   - build the data-selection script;
   - write the vectorized token selector and test it on CPU against the loop;
   - write the merge patch and test it on a tiny fake model.
3. **Spend the rest of this week's GPU on repeat runs of harness arms that are already built:** traj-arms #145, sched #144, sandbox-robust #76 and prefill-v2 #129. user-combo-10x already shows +2.3 over control on 3 runs. Start option 2's gate only after a host says yes. Target dates: gate Oct 16–18, training Oct 18–21, evaluation Oct 21–25, leaderboard submission by Oct 26.

## Open questions only you can answer
- **Money:** will you spend any? Nothing has been spent; prices only:
  - Renting an RTX PRO 6000: $0.80–2.09/h. This avoids the RTX rule question and costs about $20–60 per round (est.).
  - A standard BF16 LoRA on 8×H100: $14–21/h for 3–5 h, about $50–110, plus downloading the 360 GB base model.
  - Free but unproven: Modal's $30 starter credit (that it needs no card is unverified), and AMD Developer Cloud's 25 h on an MI300X (needs an application; the int4 backward is untested on ROCm).
- **GPU split:** will you spend about 30 of the roughly 140 GPU-h before the freeze on option 2, which means about 13 fewer harness repeat runs?
- **Evidence bar:** for this fine-tune, will you accept "not worse" plus the prior, instead of your rule that repeated runs must score consistently higher?
- **Deadline for the rules answer:** if no host answers by about Oct 16, do we drop option 2 or pay to rent?
- **Publishing:** are you OK making the adapter and training code public? The prize requires it.