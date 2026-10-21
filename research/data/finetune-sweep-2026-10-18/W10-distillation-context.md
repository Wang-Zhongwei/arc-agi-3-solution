# W10: Distillation routes for Qwen3.8-Flash-Next (teachers, on-policy distillation, context distillation, search distillation, test-time training)

Researcher W10, 2026-10-18. Builds on the 10-17 plan, its raw notes, `rsi-proposals-2026-10-15.md` (R1–R6) and siblings W1–W5b. On-policy distillation basics (Thinking Machines, Qwen3 Table 21, SDFT, ms-swift GKD) are in W4 §5 and W2 §4, and ARC-AGI-1/2 TTT is in W5 Q5; not repeated here. New here: usable teachers (tokenizer hashes checked), 2026 privileged/hindsight self-distillation (including negative results for thinking models), multi-turn OPD, frontier-API terms and cost, context distillation against our token budget, search distillation, and test-time adaptation on the Kaggle card.

Tags: [E] = evidence I read in a primary source; [S] = abstract or search snippet only; [I] = my inference; UNVERIFIED = I couldn't check it.

## Q1. On-policy distillation: which teacher, and self-distillation with privileged or hindsight context

### Takeaway
Two same-tokenizer teachers exist: **Flash-Next itself with privileged context** (free, runs on our Kaggle GPU) and **Qwen3.8-2.4T-A95B** (open-weight, byte-identical tokenizer, too big for one 8×H200 node, no ARC-AGI-3 evidence). The best 2026 fit is SDPO-style self-distillation: the same model, shown feedback (an action's outcome or a successful sibling sample), re-scores the student's own reply. For thinking models the evidence is mixed: privileged teachers can suppress the reconsideration tokens stuck-level exploration needs, so test first on frozen states.

### Cited Findings
**Teachers with the same tokenizer (checked 2026-10-18 through the HF API, `?blobs=true`):**
- `tokenizer.json` has SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` in **Qwen3.8-Flash-Next, Qwen3.8-27B and Qwen3.8-2.4T-A95B**. `vocab.json` and `merges.txt` blob ids also match, and `vocab_size` is 248,320 in all three configs [E] (`https://huggingface.co/api/models/Qwen/<repo>?blobs=true` for Qwen3.8-Flash-Next, Qwen3.8-2.4T-A95B, Qwen3.8-27B).
- `chat_template.jinja` is **identical** for Flash-Next and the 27B (blob `c0c686f9…`) but **different** for the 2.4T (blob `7015404d…`) [E] (same sources).
- Qwen3.8-2.4T-A95B config: `Qwen3_5MoeForCausalLM`, 92 layers, hidden 8192, 512 experts top-10. Created on HF 2026-08-08. Licence tag `qwen3.8-max` [E] ([config.json](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/config.json)).
  - LICENSE is MIT-style ("use, copy, modify, … fine-tune, and create derivative works"), plus a naming clause above 100M MAU / $20M monthly revenue and a separate licence for MaaS/AI-Work-Assistant businesses above $50M. No clause restricts training on outputs [E] ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/LICENSE), lines 5–11).
- **OpenRouter serves `qwen/qwen3.8-2.4t-a95b`** at $2/M input, $6/M output and $0.25/M cache read, with a 1,048,576 context.
  - `logprobs` and `top_logprobs` are supported on the Novita, Alibaba and Together endpoints, but not on SiliconFlow (fp8) or Venice [E] ([api/v1/models/qwen/qwen3.8-2.4t-a95b/endpoints](https://openrouter.ai/api/v1/models/qwen/qwen3.8-2.4t-a95b/endpoints), queried 2026-10-18).
  - These are logprobs of **generated** tokens. Whether any provider returns prompt-token (echo) logprobs, which on-policy distillation needs to score the student's tokens, is UNVERIFIED.
- No Qwen model has an ARC-AGI-3 row on ARC's verified leaderboard data. The only Qwen3.8 entry is the 27B on ARC-AGI-2 (semi-private 42.4% at xhigh) [E] ([evaluations.json](https://arcprize.org/media/data/evaluations.json), fetched 2026-10-18). Our own 27B run scored 14.51 offline against 54.69 for Flash-Next (`finetune-plan-2026-10-17.md` §1). So the 27B is not a teacher.
- Cross-tokenizer on-policy distillation exists (TRL's experimental GOLD; [arXiv 2606.09456](https://arxiv.org/pdf/2606.09456)) [S]. We don't need it.

**Self-distillation with privileged information (2026):**
- **OPSD, "Self-Distilled Reasoner"** (Zhao et al., UCLA/Meta, [arXiv 2601.18734v2](https://arxiv.org/html/2601.18734v2), 2026-03-05) [E]:
  - Setup: the teacher is the same model shown the reference solution; the student sees only the problem; the student samples, and the loss is full-vocabulary JSD (β=0.5).
  - Qwen3 1.7B/4B/8B, average@16 over AIME24/25, HMMT25 and AMO-Bench:
    - 8B: base 50.0, SFT 50.0, GRPO 51.3, OPSD 52.2;
    - 4B: 48.3 / 49.6 / 49.6 / 50.4;
    - 1.7B: 28.8 / 28.0 / 30.5 / 30.0.
  - The authors say OPSD "can match GRPO performance but not significantly outperform it".
  - "8–12× token efficiency" compares a 1,024-token cap, 1 rollout, against GRPO at 16,384 tokens, 8 rollouts; no GPU-hours reported. Full-vocab JSD beat sampled-token (Qwen3-4B AIME25 pass@8 84.1 vs 82.1). LoRA r64 on 8×A100/H100.
- **SDPO, "Reinforcement Learning via Self-Distillation"** (Hübotter et al., ETH/MIT/Stanford, [arXiv 2601.20802](https://arxiv.org/html/2601.20802v1), ICML 2026) [E]:
  - The teacher is the **current policy conditioned on feedback**: environment output such as runtime errors, and/or **a successful sibling rollout from the same group**. It re-scores the student's original attempt, so there is "no sampling overhead".
  - Loss: JSD with top-K = 100 logits plus a tail term. The teacher is regularised by an EMA, or by interpolation with the initial teacher.
  - Results: science/tool-use aggregate 68.8% vs GRPO 64.1%; LiveCodeBench v6 48.8% vs 41.2% (Qwen3-8B), GRPO's final accuracy with ~4× fewer generations; Chemistry: GRPO's 5-h accuracy in ~30 min.
  - **Outputs are more than 3× shorter than GRPO's** (about 7× on Chemistry).
  - Putting the student's own attempt in the teacher prompt "biases the teacher toward the student and reduces exploration".
  - The gain grows with model size; SDPO underperforms GRPO on Qwen2.5-1.5B.
- **"Rethinking On-Policy Self-Distillation for Thinking Models"** (Kaur, …, Arora, [arXiv 2607.05184](https://arxiv.org/abs/2607.05184), 2026-07-06) [S, abstract]:
  - On five Qwen3/OLMo thinking models, privileged self-distillation causes **up to a 17% relative drop** in avg@16 on AIME/HMMT, growing with withheld context and rollout budget; "privileged OPD reverses" vanilla OPD's gains.
  - Mechanism: privileged context lowers forking at high-entropy positions and "penalizes sampled reconsideration tokens"; trained models produce fewer verification and backtracking markers.
  - RLSD ([2604.03128](https://arxiv.org/abs/2604.03128)) and DemoPSD ([2607.02502](https://arxiv.org/abs/2607.02502)) target the same leakage [S].
- **π-Distill** (Penaloza et al., ServiceNow/Mila, [arXiv 2602.04942](https://arxiv.org/abs/2602.04942), v3 2026-02-16) [S, abstract]:
  - Targets multi-turn agents where "closed-source systems typically hide their internal reasoning and expose only action trajectories".
  - The teacher is conditioned on action-only privileged information; π-Distill and an OPSD variant (an RL reverse-KL penalty to the PI-conditioned teacher) "outperform … SFT followed by RL". No numbers in the abstract.
- **Multi-turn on-policy distillation** [S, abstracts and snippets]:
  - TCOD ([2604.24005](https://arxiv.org/pdf/2604.24005)): vanilla OPD is unstable from "inter-turn error compounding" (the student leaves the teacher's support); a short-to-long curriculum adds up to 18 points on ALFWorld/WebShop/ScienceWorld.
  - Guided-OPD ([2606.15912](https://papers.cool/arxiv/2606.15912)): decaying teacher-turn mixing, +21.1% score over vanilla OPD. ATOD ([2606.27814](https://arxiv.org/pdf/2606.27814)): OPD gains "saturate once the student approaches the teacher".
  - SMRC-SD ([2608.05219](https://arxiv.org/abs/2608.05219), v2 2026-10-14): distil only on turns where the student's state matches the privileged reference, because an off-reference teacher "can penalize valid actions". Snippet: Qwen3-1.7B ALFWorld 0.746 → 0.865, WebShop 0.574 → 0.693 (not on the abstract page).
- **Cost against RL** (from W4): Qwen3-8B OPD 74.4 AIME'24 in 1,800 GPU-h vs RL 67.6 in 17,920 GPU-h ([arXiv 2505.09388 Table 21](https://arxiv.org/html/2505.09388), `W4-lab-pipelines.md:14`); Thinking Machines 9–30× less compute than SFT (`W4-lab-pipelines.md:138`). GKD ([2306.13649](https://arxiv.org/abs/2306.13649), ICLR 2024) originated student-sampled distillation [E, abstract].

**Our stack:**
- No harness code requests logprobs, so the ~180k logged calls have no behaviour logprobs (`W2-rl-fundamentals.md`, "Our stack" bullets).
- The server is Intel AutoRound W4A16 with an fp8 KV cache (same source).
- The state lab already resamples frozen decisions on the Kaggle server: 400 samples in 14.9 min and 638k generated tokens, 27 GPU-min end to end (`arc-agi-v4/arms.md:313`).

### Inferences
- [I] **The teacher signal can be computed on the Kaggle GPU as prefill only.** The 180k logged replies were sampled by the W4A16 model we serve (on Franzen-based harness arms), so for round 0 they are close to on-policy. Re-score each reply with SGLang input-token top-k logprobs under a teacher prompt: the original prompt plus a short privileged block before the reply. The prompt prefix stays in the radix cache, so the cost is about the hint plus reply tokens. The Azure trainer minimises per-token reverse KL/JSD from the student to these teacher targets: SDPO without online rollouts. Whether Pennyroyal SGLang returns top-k prompt logprobs for this model is UNVERIFIED; test it first.
- [I] **Privileged blocks we can produce, game-agnostically, from data we already have:**
  - (a) **outcome feedback** ("this action produced no change / a new board / a level-up"), in every logged call; SDPO's environment-feedback teacher;
  - (b) **a sibling success** at the same frozen state (state-lab k-sampling); SDPO's "successful rollout as implicit feedback";
  - (c) **hindsight rule or route** from the model's own later recap when the level was cleared later or in another run; OPSD/π-Distill-style.
  - (a) and (b) leak less than (c): they don't reveal the solution.
- [I] **The thinking-model warning targets exactly the behaviour we need.** Stuck-level failure is holding a wrong hypothesis while an untried action would reach a new board (96% of stuck decisions). If a privileged teacher suppresses "wait, maybe…" forks (2607.05184), it could make stalls worse.
  - Gate on frozen stuck states with distinct actions across k samples, reconsideration-marker rate and right-move rate, not only tokens per action.
  - Per-decision distillation on frozen states satisfies SMRC-SD's state-match rule and avoids TCOD's inter-turn compounding.
- [I] SDPO's >3× shorter outputs bears on our biggest lever, tokens per action (872 vs 1,793 decided the 27B gap), the same direction as R3 and the RSI toy test.
- [I] **Qwen3.8-2.4T as a teacher is a test-first long shot.**
  - 2.4T parameters is about 2.4 TB even at FP8, against 8 × 141 GB = 1.13 TB of HBM, so it can't run on our node; over the API it is an off-policy text teacher unless a provider returns prompt logprobs; its ARC-AGI-3 skill is unmeasured.

### Gaps
- No OPD, OPSD or SDPO result on a model of our class (125B MoE, hybrid GDN/QSA) or on ARC-AGI-3.
- Whether ms-swift Megatron GKD accepts a teacher prompt different from the student's, or precomputed teacher logits, is UNVERIFIED; a custom KD loss may be needed.
- The "Rethinking OPSD" numbers per model and its remedies come from the abstract only; the full text wasn't read.
- The SMRC-SD, TCOD, Guided-OPD and ATOD numbers come from search snippets.

## Q2. Distillation from frontier APIs (Claude, GPT, Gemini): legality, cost, and whether they play ARC-AGI-3 well

### Takeaway
2026 frontier models do play ARC-AGI-3 far better than we do. GPT-6 Astra scores 99.9% and Gemini 4 Argon 97.5% on the semi-private set with provider adapters.

All three providers' terms forbid using outputs to build competing models. Their reasoning is hidden or summarised, their tokenizers differ (so there are no usable logits), and the prize rules require releasing the trained weights. **Don't distil from closed APIs.** The legal teacher-text option is the open-weight Qwen3.8-2.4T-A95B, and even that is off-policy text, which the sibling notes show transfers poorly to a strong RL-tuned model.

### Cited Findings
- **Anthropic** Commercial Terms D.4: customers may not "access the Services to build a competing product or service, including to train competing AI models" (effective 2025-06-17) [E] ([anthropic.com/legal/commercial-terms](https://www.anthropic.com/legal/commercial-terms)).
- **Google** Gemini API terms, Use Restrictions: "You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)"; last modified 2026-04-28 [E] ([ai.google.dev/gemini-api/terms](https://ai.google.dev/gemini-api/terms)).
- **OpenAI**'s terms list "use Output to develop models that compete with OpenAI" as prohibited [S]. The page returned HTTP 403 to my fetch; the wording is from search snippets of [openai.com/policies/terms-of-use](https://www.openai.com/policies/terms-of-use/).
  - A Kaggle participant raised the same conflict (`W5b-kaggle-intel.md:21`).
- **Kaggle rules** (from W5b's read of the rules page):
  - external data and models are allowed if "reasonably accessible to all" and of "minimal cost" (2.6.b) (`W5b-kaggle-intel.md:117`);
  - winners must open-source under CC-BY 4.0, with an exemption for pretrained inputs whose licences are incompatible (`finetune-plan-2026-10-17.md` §5).
- **Hidden reasoning:** Claude 4-generation models return a *summary* of their thinking, and full thinking needs a sales contact [S] ([Anthropic extended-thinking docs](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)). π-Distill exists because closed agents "expose only action trajectories" ([arXiv 2602.04942](https://arxiv.org/abs/2602.04942)) [S].
- **Frontier ARC-AGI-3 scores** (`v3_Semi_Private` rows in [evaluations.json](https://arcprize.org/media/data/evaluations.json), fetched 2026-10-18) [E]:
  - with provider adapters: GPT-6 Astra high 99.9%, Gemini 4 Argon low 97.5%, GPT-6.1 Sol xhigh 96.4%, Gemini 3.8 Flash high 35.0%;
  - standard harness: GPT-6 Astra max 62.7%;
  - Claude Opus 5 high 30.2%;
  - the file has no cost per task for v3 rows.
- **Cost of a frontier harness:** DreamTeam's workspace-optimisation agent (Opus 4.6 + GPT-5.5, six roles) used about 693k input and 26k output tokens, and about 205 s, per environment action. It ran under a $2,000-per-game cap for **38.4 RHAE on the 25 public games** (mean of 2 runs) (`arc-agi-v4/arms.md:657`; [arXiv 2605.09650](https://arxiv.org/abs/2605.09650)).
- **Free frontier traces already exist:** AgentNativeResearchLab's `arc-agi3-*` datasets (Claude, Kimi k2.7, GPT-5.6, Gemini 3.1, GLM 5.2) are CC BY 4.0, but they cover **public games only** (`finetune-research-2026-10-17/data.md:50`).
- **Off-policy teacher text transfers poorly:** Retaining by Doing (`W1-sft-pedagogy.md:21-27`); RAGEN/SPIRAL/ViGaL/Orak (`W3-agentic-game-rl.md:134`).

### Inferences
- [I] **Cost in our harness for the open-weight 2.4T teacher, if it spent Flash-Next's tokens:**
  - one 2-h, 25-game run is about 2.6k calls and 5.25M completion tokens (`data.md:66`), with prompts around 55k (median);
  - that is about 143M prompt tokens at 93.6% cache hit (`research/archive/workflow-reports/throughput-900--wf_2e6382c4-06f/01-research-logs.md:54`);
  - at OpenRouter's prices: about $33 cached + $18 uncached + $32 output ≈ **$80 per run-equivalent**. A teacher that thinks 2× longer costs roughly $115–170.
  - Rough: provider caching may differ, and OpenRouter credit was $0 at last check (auto-memory).
- [I] **Frontier APIs at DreamTeam's token rate:** about 50 actions per level × 693k input means about 35M input tokens per level. Thousands of trajectories would cost tens of thousands of dollars or more, before the legal problem.
- [I] **Even legal frontier traces would teach the wrong thing:** hidden reasoning and huge per-action budgets, while our lever is fewer tokens per action.
- [I] If a test shows the 2.4T beats Flash-Next in our harness, its best use is as a **verifier or labeller**: it picks which of Flash-Next's k samples to keep (search distillation, Q4), so the training text stays Flash-Next's own. Do not SFT on its text.

### Gaps
- OpenAI's clause text is from a snippet, not a fetched page.
- Whether the frontier-trace datasets' CC BY licence frees downstream users from the generators' provider terms is a legal question I can't answer.
- No ARC-AGI-3 score for Qwen3.8-2.4T-A95B in any harness.

## Q3. Context distillation and prompt baking: can it shrink the ~55k context or the opener cost?

### Takeaway
The method works (Snell 2022; Prompt Baking 2024) but targets the wrong tokens for us: the system prompt is only about 4–5.5k tokens and prefix-cached, prefill is about 10% of wall, and decode is the bottleneck. The 55k is mostly retained conversation, including past thinking. Long-to-short context distillation (full-history teacher, trimmed-history student) is the version that could matter; it is unproven for agents, so after 11-02. R5 (harness facts) is better done as on-policy training *with* the facts in the prompt.

### Cited Findings
- **Snell, Klein, Zhong, "Learning by Distilling Context"** ([arXiv 2209.15189](https://arxiv.org/abs/2209.15189), 2022-09-30) [E, abstract]:
  - The model with instructions plus a scratchpad is the teacher; the same model fine-tuned on the answer alone is the student.
  - It internalises instructions, step-by-step reasoning (8-digit addition) and examples. On SPIDER it "outperforms directly learning with gradient descent by 9%".
- **Prompt Baking** (Bhargava et al., [arXiv 2409.13697](https://arxiv.org/abs/2409.13697), 2024-09-04) [E, abstract]:
  - Minimises KL(P_θ(·|u) ‖ P_θu(·)), so the weights behave as if the prompt u were present.
  - Baked chain-of-thought prompts improve zero-shot GSM8K, ASDiv, MBPP, ARC-Easy/Challenge and CSQA. The abstract gives no numbers.
- OPSDL ([alphaxiv 2604.17535](https://www.alphaxiv.org/abs/2604.17535), 2026-04) self-distils long-context behaviour from short-context behaviour [S].
- **Our token budget:**
  - The system prompt is about 18.6k characters, about 5.5k tokens (`research/archive/workflow-reports/fable-layer-vs-ours--wf_c4d3dccd-2d7/01-read-handbook.md:28`); another read puts it at about 4.1k tokens, "mostly Franzen advice" (`…/03-read-ours+rellik.md:41`).
  - Prefix-cache hit is 93.6% overall. Prefill token cost is about 730 s, 9.7% of wall; cache-miss prefill is about 4.5% of wall (`…/throughput-900--wf_2e6382c4-06f/01-research-logs.md:51-54`).
  - Openers carry 1.6 images on average at about 402 tokens each (`…/arc3-idea-sweep--wf_6b16ae20-e2c/02-sweep-prompting.md:141`).
- The Flash-Next chat template keeps thinking blocks from all historical messages (`W1-sft-pedagogy.md:144`). So history, not the system prompt, fills the 55k.
- In the state lab, facts in the prompt moved decisions (moved objects 0 → 16 of 20; own rules 3 → 9) (`rsi-proposals-2026-10-15.md`, R5).

### Inferences
- [I] **Baking the system prompt saves at most about 5k of about 55k tokens**, which are mostly cached anyway. That is under 1–2% of wall time. It also changes behaviour we can't easily audit, since the prompt strings are behaviour per CLAUDE.md. **Don't** do it.
- [I] **Harness facts (R5):** keep the facts in the prompt. Training the student to compute them itself would cost thinking tokens, the very thing we want to cut. The useful part is making the model *use* the facts. That happens naturally if every training sample is rendered with the served mem-unified prompt, as the prior plan says.
- [I] **Long-to-short context distillation** (teacher = full 55k context, student = trimmed context) could let the trim fire earlier and pack more streams per KV pool. But decode already saturates at 12–13 running streams (`rsi-proposals-2026-10-15.md:28`), so the throughput gain is uncertain. Not before 11-02.

### Gaps
- No paper found on context distillation for multi-turn tool agents with retained thinking.
- I didn't measure what share of the 55k is retained thinking vs tool results vs images.

## Q4. Search distillation: best-of-N / tree search outcomes into one policy (expert iteration)

### Takeaway
The most direct fit to our data and tooling: **branch at frozen decision states** (state lab), **score siblings by an exactly checkable outcome** (new board, level progress, hindsight-shortest route), and **distil the winner** by SFT (R3/R4) or SDPO-style re-scoring (Q1). Agentic evidence (Agent Q, Tree-GRPO, BranPO) supports step-level branching.

### Cited Findings
- **Expert iteration** (Anthony, Tian, Barber 2017, [arXiv 1705.08439](https://arxiv.org/abs/1705.08439)): tree search is the "expert", the network is the "apprentice" that imitates it, and the improved apprentice strengthens the next search. Not re-read for this note; cited for the definition only.
- **Agent Q** (Putta et al., [arXiv 2408.07199](https://arxiv.org/abs/2408.07199), 2024-08-13) [E, abstract]:
  - MCTS-guided search plus off-policy DPO on the agent's own interactions.
  - Llama-3-70B on real booking tasks: 18.6% zero-shot → 81.7% "after a single day of data collection" → 95.4% with online search.
  - "Consistently outperforms behavior cloning and reinforced fine-tuning".
- **Tree-GRPO** (ICLR 2026, [arXiv 2509.21240](https://arxiv.org/pdf/2509.21240)) [S]: tree nodes are agent steps, and sharing prefixes buys more rollouts per token budget. Intra-tree GRPO is shown "equivalent" to step-level DPO.
- **BranPO** ([arXiv 2602.03719](https://arxiv.org/pdf/2602.03719)) [S]: truncates near the tail and resamples alternatives, because outcomes diverge mainly at tail decisions.
- **SDPO's** "successful rollouts as implicit feedback for failed attempts" is the self-distillation form of best-of-N ([arXiv 2601.20802](https://arxiv.org/html/2601.20802v1)) [E].
- **Our measurements:**
  - in 96% of stuck-level decisions an untried action would have reached a new board;
  - on cleared levels 81% of moves are on the hindsight-shortest route (task brief; `finetune-plan-2026-10-17.md` §2);
  - state-lab throughput is 400 samples in 14.9 min (`arms.md:313`).
- Toy runs: SFT on 24 shortest-correct examples +0.20 (p 0.0008) vs GRPO +0.07 n.s. (`arms.md:323`); the Kaggle RSI test doubled Qwen3-4B's score mostly by answering within budget (`finetune-plan-2026-10-17.md` §9a).

### Inferences
- [I] **Cost on Kaggle quota:** about 27 samples per minute, so k = 8 at 5k frozen states (40k samples) is about 25 GPU-h, about 2.3 days of the 11 GPU-h per day across both accounts. Restrict it to stuck-level and pre-clear states.
- [I] **One-step outcomes are exact and cheap** (engine replay). Multi-step value needs the model to keep playing from the branch, which costs roughly a 20-min panel per branch set. Use one-step novelty and progress as the selector on stuck states, and the hindsight route on cleared states.
- [I] **Distil the selection, not the search procedure.** Keep the shortest successful sibling (R3), or use the successful sibling as the SDPO teacher context (Q1(b)). The information carried is about log2 k bits per kept sample (`W1-sft-pedagogy.md:50`), which is why on-policy selection forgets little.
- [I] **DPO pairs** (success vs failure at the same state) are a second step, after plain SFT has been measured (W4 item 2).

### Gaps
- No agentic search-distillation result on a strong RL-tuned thinking MoE.
- The BOND-style best-of-N distribution matching ([arXiv 2407.14622](https://arxiv.org/abs/2407.14622)) and AlphaZero-like LLM search (TS-LLM, [arXiv 2309.17179](https://arxiv.org/abs/2309.17179)) were not re-read this session.

## Q5. Test-time training or adaptation during the Kaggle run

### Takeaway
Trunk TTT is infeasible on the card (W5 Q5 agrees): weights take about 70 of 96 GB and decode is the bottleneck all run. SGLang hot-loads LoRA adapters, but its docs say nothing about MoE experts, quantized bases or linear-attention hybrids. The only plausible test-time adaptation is lossless drafter adaptation (Online Speculative Decoding), and offline R1 (MTP head distilled on ARC transcripts) captures most of it since our domain is constant across games. ARC-AGI-1/2 TTT needed per-task demo pairs, which ARC-AGI-3 lacks.

### Cited Findings
- **SGLang LoRA:** dynamic `/load_lora_adapter` and `/unload_lora_adapter` endpoints exist. Adapters must sit in pinned CPU memory for async copies. Overlap loading is off by default because it hurts batching. The docs mention no MoE expert modules, no quantized bases and no linear-attention hybrids [E] ([docs.sglang.io LoRA](https://docs.sglang.io/advanced_features/lora.html)).
  - TensorRT-LLM: quantized bases are "not yet supported for MoE LoRA" [S] ([docs](https://nvidia.github.io/TensorRT-LLM/latest/features/lora.html)).
  - Our serving folder has no LoRA code (grep over `daniel-franzen-sol/serving`: no hits).
- **Card budget:** the W4A16 target weights take 69.85 GB (`W5-arc-lineage.md`, Q5, citing `arms.md:665`). The KV pool is about 1.01M tokens and peaks at 0.99 (`throughput-900…/01-research-logs.md:56`).
- **Online Speculative Decoding** (Liu et al., [arXiv 2310.07177](https://arxiv.org/abs/2310.07177), v4 2024-06): continuously distils the draft model on the live query distribution. Token acceptance rises by 0.1–0.65 and latency falls 1.42–2.17× [E, abstract].
- Offline drafter retraining: AngelSpec acceptance 52.8% → 66.4%; one community self-distilled retrain regressed (`finetune-research-2026-10-17/training.md:55`).
- **aTTT**, in-episode LoRA: +5.0 ALFWorld and +4.9 SWE-bench Lite at about 1.9× cost. It "can amplify drift when the agent gets stuck" (`W5-arc-lineage.md`, Q5, [arXiv 2607.03441](https://arxiv.org/abs/2607.03441)).

### Inferences
- [I] **No trunk TTT before 11-02:** no free memory for optimizer state, CPU backward through a 125B MoE at 55k context is far too slow, and the only stuck-level signal is failure, where aTTT drifts.
- [I] **Drafter TTT is lossless, so it is safe to try, but it is low value.** A per-game drift in drafter acceptance is unmeasured. R1 offline (accept length 2.68 → 3.0–3.2 projected in `rsi-proposals-2026-10-15.md`) should come first, and it must be redone after any trunk fine-tune anyway.
- [I] Test-time learning stays in context (memory card, confirmed rules); Q1/Q4 should make that loop cheaper per action.

### Gaps
- Whether Pennyroyal SGLang can serve an unmerged LoRA on Flash-Next's BF16 modules (attention, GDN, shared expert) is UNVERIFIED. It matters for skipping re-quantization.
- No measurement of MTP acceptance drift across games within one run.

## Implications for our plan

1. **DO (cheapest new route; test-first on days 1–4): SDPO-style hindsight self-distillation at frozen decisions.**
   - Teacher = Flash-Next itself, with a privileged block added after the prompt: (a) the outcome of the action, or (b) a successful sibling sample. Student = the same prompt without the block. Use the student's own logged or state-lab replies.
   - Teacher and student top-k logprobs come from SGLang prefill on the Kaggle GPU (free quota). Train on Azure as KD on own samples.
   - First test: does Pennyroyal return top-k input-token logprobs, and does trainer↔SGLang KL stay ≤0.003 (W2)?
   - Gate on frozen stuck states: right-move rate, tokens per decision, **distinct actions across k samples, and reconsideration-marker rate**. The last two are the guard against 2607.05184's thinking-model degradation.
2. **DO: search distillation as the selector for R3/R4.**
   - k = 8 at stuck and pre-clear states on Kaggle quota (about 25 GPU-h per 5k states); keep the shortest sibling whose action is verified good (new board, progress, hindsight route). Same pipeline as item 1.
3. **DON'T: distil from Claude, GPT or Gemini.** Their terms forbid training competing models (Anthropic D.4, the Gemini API terms, OpenAI's terms), reasoning is hidden, and logits are unusable. It also endangers prize eligibility.
4. **TEST-FIRST, low priority: Qwen3.8-2.4T-A95B as a selector or labeller, not a text teacher.**
   - Same tokenizer, permissive licence, $2/$6 per M on OpenRouter. One paid 25-game panel (about $80–170, rough) only if the credit is funded; use it only if it clearly beats Flash-Next.
5. **DON'T: bake the system prompt.** It saves under 2% of wall time. Keep the harness facts in the prompt and train with the served prompt.
6. **DON'T: trunk test-time training on Kaggle.** Do R1, the offline MTP-head distillation on ARC transcripts, after every trunk change. Treat online drafter adaptation as an after-11-02 idea.
7. **Calibrate expectations.**
   - OPSD only *matched* GRPO (+2.1 to +2.2 points over base at 4–8B); SDPO's wins are on code, science and tool tasks; nothing is shown on a 125B thinking MoE or ARC-AGI-3.
   - The realistic payoff is fewer tokens per action at the same decision quality (our measured lever), not a jump to 60 on its own.

## Fact-check (independent verifier, 2026-10-18)

I checked each claim against its primary source: the HF and OpenRouter APIs, the arXiv HTML and abstract pages, the live terms pages, and the local files at the cited lines. I did not use a browser.

| # | Claim | Verdict | Correction / note |
|---|---|---|---|
| 1 | Flash-Next, 27B and 2.4T have a byte-identical `tokenizer.json` (SHA-256 `0997f410…b9f3`) | **Confirmed** | All three report the same LFS sha256, blob `9328ce9c…` and 12,809,320 bytes. `vocab.json` and `merges.txt` blobs match, and `vocab_size` is 248,320 everywhere. `tokenizer_config.json` differs for the 2.4T (blob `c43fe0e…` vs `5de744b…`), but only in its embedded chat template. `added_tokens_decoder` is identical. |
| 2 | 2.4T is open-weight under an MIT-style licence with no anti-distillation clause; OpenRouter charges $2 in / $6 out / $0.25 cache-read per M; logprobs on Novita, Alibaba and Together | **Confirmed** | The LICENSE ("Qwen3.8-Max License") lines 5–11 match the summary. No clause covers training on outputs. The HF tag is `license:other` (`license_name: qwen3.8-max`). The endpoints API shows the stated prices on all 5 providers, and `logprobs`/`top_logprobs` only on Novita, Alibaba and Together. The 1,048,576 context is the model-level figure. The logprob-capable endpoints offer 1,000,000 (Novita, Alibaba) and 1,010,000 (Together). See the text-only note below. |
| 3 | SDPO 48.8% vs GRPO 41.2% on LCBv6 (Qwen3-8B), with outputs more than 3× shorter than GRPO's | **Partially confirmed** | 48.8 vs 41.2 on LCBv6 with Qwen3-8B is correct (Fig. 1, §4), and so is "4× fewer generations". The ">3× shorter" result is from **§3, the scalar-reward science Q&A and tool-use setting**, not LCBv6. Table 8 gives Qwen3-8B 820.8 → 255.8 tokens and Olmo3-7B-Instruct 1,095.4 → 343.9 tokens, both 3.2×. The 7× figure is Chemistry with **Olmo3-7B-Instruct**. The comparison is **against GRPO**, which grows verbose ("Hmm", "Wait", loops). It is not against the initial model, so it does not directly show that SDPO shortens the starting policy's replies. Max response was 8,192 tokens. The ICML 2026 venue checks out (icml.cc/virtual/2026 poster 64121; PMLR v306). |
| 4 | Privileged on-policy self-distillation degrades thinking models by up to 17% relative avg@16 on AIME/HMMT and penalizes reconsideration tokens | **Confirmed** | Abstract: five Qwen3/OLMo thinking models on AIME24, AIME25 and HMMT25. The drop is "most pronounced at long rollout budgets". "Privileged OPD penalizes sampled reconsideration tokens that vanilla OPD supports." Submitted 2026-07-06. |
| 5 | OPSD Qwen3-8B 52.2 vs GRPO 51.3 vs base 50.0; "can match GRPO performance but not significantly outperform it" | **Confirmed** | Table 2 of v2: Base (Instruct) 50.0, SFT 50.0, GRPO 51.3, OPSD 52.2. The quote is in the Table 2 caption. The 4B and 1.7B rows, the 84.1 vs 82.1 ablation (Qwen3-4B, pass@8, 2,048-token generations), LoRA r64 and 8×A100/H100 also match. The v2 date of 2026-03-05 is right. The current version is v3 (2026-03-20). |
| 6 | Anthropic D.4 and the Gemini API "compete" clauses | **Confirmed** | D.4 wording is exact. The page metadata gives `effectiveDate` 2025-06-17, with a formatting update on 2026-03-11. The Gemini Use Restrictions wording is exact. The date is the page footer "Last updated 2026-04-28 UTC", not a "last modified" field. |
| 7 | System prompt about 5.5k tokens (18,636 chars); prefix-cache hit 93.6%; prefill token cost about 9.7% of wall | **Partially confirmed** | The lines exist (`01-read-handbook.md:28`; `01-research-logs.md:51,54`). However, 18,636 chars / ~5.5k tokens is a **third-party, unverified [T] figure** from leoprovorov's Fable handbook, for their B003 unchanged-Franzen run. **Our own rendered measurement is 4,134 tokens** for control (4,361–4,644 with stack-v2/attention), in `03-read-ours+rellik.md:41,46-49`. Line 51 also gives a **fixed per-batch prefill cost of 820 s (10.9% of wall)** on top of the 730 s (9.7%) token cost. Total prefill stall is about **20.6% of wall**, and that file's header cites issue #10 as "prefill ~15% of wall". The run is `fz-arms-v4-control`. The conclusion still holds: the system prompt is cached, and the fixed per-batch cost doesn't depend on prompt length. But "prefill is about 10% of wall" (Q3 takeaway) understates prefill. |
| 8 | State lab: 400 replies in 14.9 min on Kaggle (27 GPU-min end to end, 638k tokens) | **Confirmed** | `arms.md:313` matches exactly (`mellamoa/v4-lab-round2` v1, 22:11–22:38 UTC). The derived "~25 GPU-h per 40k samples" uses the sampling-only rate (26.8/min). At the end-to-end rate (400 per 27 GPU-min) it is about **45 GPU-h**. Round 3 (`arms.md:320`, 300 samples in 15.5 min, about 19/min) gives about **34 GPU-h**. A fair range is 25–45 GPU-h, plus server start-up for each session. |

**Other errors and omissions in the notes:**
- **Qwen3.8-2.4T-A95B is text-only.** Its config has no `vision_config`, the HF `pipeline_tag` is `text-generation`, OpenRouter lists `input_modalities: ["text"]`, and its chat template has no image handling. Flash-Next is `image-text-to-text`, and our openers carry about 1.6 images (Q3). The 2.4T therefore can't play our harness as is, and it can't re-score Flash-Next's multimodal contexts. This affects the "$80 per run-equivalent" estimate, the "one paid 25-game panel" in Implication 4, and the "verifier or labeller" idea. Frames would have to be rendered as text first.
- Line 21: "The only Qwen3.8 entry is the 27B on ARC-AGI-2" is imprecise. `evaluations.json` also has 27B rows on ARC-AGI-1 (v1 semi-private 87.5% at xhigh). It is right that no Qwen model has a v3 (ARC-AGI-3) row.
- Line 46: the π-Distill abstract says π-Distill "and, **in some cases**, OPSD" outperform SFT followed by RL. The notes drop that qualifier.
- Line 157: the "consistently outperforms behavior cloning and reinforced fine-tuning" quote is from Agent Q's **WebShop** experiments, not the real-booking result next to it.
- Line 36: "Chemistry: GRPO's 5-h accuracy in ~30 min" is with Olmo3-7B-Instruct (a 10× speedup), not Qwen3-8B.
- Verified with no change needed: the frontier v3 scores (GPT-6 Astra high provider-adapter 99.9%, Gemini 4 Argon low 97.5%, GPT-6.1 Sol xhigh 96.4%, Gemini 3.8 Flash high 35.0%, GPT-6 Astra max standard 62.7%, Claude Opus 5 high 30.2%, null `costPerTask`); Online Speculative Decoding (0.1–0.65, 1.42–2.17×); aTTT (+5.0 / +4.9, 1.9×); Guided-OPD (21.1% relative); TCOD (up to 18 points); Agent Q (18.6 → 81.7 → 95.4); the SGLang LoRA docs (no MoE, quantized-base or hybrid mentions); no LoRA code in `serving/`; no `logprob` in the harness `inference/`; 27B 14.51 vs 54.69; `arms.md:323` toy result; `arms.md:657` DreamTeam figures. All cited arXiv ids exist with the stated titles.
