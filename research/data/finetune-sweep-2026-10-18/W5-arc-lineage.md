# W5: ARC training lineage (ARC-AGI-1/2 prizes to ARC-AGI-3) and the 2026 competition state of the art

Researcher W5, 2026-10-18. Method: primary papers (arXiv abstract and HTML pages, the NVARC PDF from its repo), arcprize.org blog posts and the leaderboard's own data file (`arcprize.org/media/data/evaluations.json` + `models.json`, fetched 2026-10-18), Hugging Face API, GitHub, and local repo files. Kaggle discussion threads are covered in depth by sibling notes `W5b-kaggle-intel.md` (browser agent); I cite them where needed and don't repeat them. Scratch copies are in the session scratchpad `w5/`: the NVARC paper text, the leaderboard JSON and the Cole & Osman HTML.

## Q1. ARC-AGI-1/2 prizes (2024, 2025): what was trained, on what data, and how big were the gains?

### Takeaway
Three levers drove ARC-AGI-1/2 progress, each with a measured effect.
- **Test-time training (TTT) on each task's own demos** was the biggest single gain, about 2.4× to 6×.
- **Large verified synthetic data** came next. NVARC used 103k puzzles and 3.2M samples; the ARChitects approach "scales with data size".
- **Augmentation-consistent sampling and scoring** added +5 to +18 points.

Static inference with a pretrained model capped out at about 11%. The 2025 theme was the "refinement loop". In the 2025 small-model work, refinement steps beat architecture.

### Cited Findings
**ARC Prize 2024 (ARC-AGI-1).**
- **Private results:** the ARChitects 53.5% (1st). MindsAI reached 55.5% but was prize-ineligible because it didn't open-source ([ARC Prize 2024 Technical Report, arXiv 2412.04604, Sec. 2.1, Table 1](https://arxiv.org/html/2412.04604)).
- **The ARChitects' recipe:** NeMo-Minitron-8B with TTT, "extensive data augmentation", and selection by how stable a solution stays under augmentation ([2412.04604 Sec. 3](https://arxiv.org/html/2412.04604)).
- **Static inference ceiling:** "there does not exist any static inference-style transduction solution that scores above 11%" ([2412.04604 Sec. 3.2](https://arxiv.org/html/2412.04604)).
- **Transduction vs induction:** the two solve "significantly distinct sets of tasks". The best single-paradigm submissions were each about 40%, and only ensembles of both reached the top ([2412.04604 Sec. 3.3](https://arxiv.org/html/2412.04604)).
- **The ARChitects' ICML 2025 paper ("Product of Experts…", Franzen, Disselhoff, Hartmann).** Table 2 gives the gain ladder on ARC-AGI-1 public eval (two-guess):

  | Model | Baseline | + TTT | + 16× augmentation | + Product-of-Experts scoring | + DFS |
  |---|---|---|---|---|---|
  | NeMo-Minitron-8B | 18.3 | 44.5 | 62.5 | 67.6 | 71.6 |
  | Llama-3.2-3B | 14.9 | 40.9 | 52.9 | 59.5 | 61.4 |

  ([arXiv 2505.07859v2 Table 2](https://arxiv.org/html/2505.07859v2))
  - Product-of-Experts scoring added about 5 points over averaging (66.6 → 71.6) (Table 3).
  - The initial fine-tune used RE-ARC only. ConceptARC and ARC-Heavy were excluded "to limit conceptual leakage". It took about 98 H100-hours for the 8B model.
  - TTT was 64 steps per task (LoRA r32), 51 s per task on an RTX 4090. The whole pipeline cost about $0.02 per task (Sec. 5, Table 4) (same source).
- **Akyürek et al. (MIT), "The Surprising Effectiveness of Test-Time Training…":**
  - TTT gives "up to 6x improvement in accuracy compared to base fine-tuned models": 53.0% with an 8B model, and 61.9% when ensembled with program synthesis.
  - The paper names three crucial components: "(1) initial finetuning on similar tasks (2) auxiliary task format and augmentations (3) per-instance training" ([arXiv 2411.07279v1 abstract](https://arxiv.org/abs/2411.07279v1)).
  - In the 2024 report the public/semi-private numbers differ: 62.8% vs 47.5% (Table 2), against 53% in the text ([2412.04604](https://arxiv.org/html/2412.04604)).
- **MindsAI (Jack Cole, Mohamed Osman):**
  - Gains: AIRV (augment, infer, reverse-augment, vote) up to +260%, and TTFT a further +300%. They claim 58% on the private set for 2024 ([arXiv 2506.14276v2 abstract + Sec. 4](https://arxiv.org/html/2506.14276v2)). ARC Prize's own figure is 55.5%, so treat 58% as the authors' number.
  - Pretraining: "substantial pre-training on ARC riddles remains indispensable for achieving state-of-the-art performance with TTFT".
  - Model size: "Benefits of increased pre-training on the ARC dataset did not beat the benefits of simply increasing the model size" for zero-shot and AIRV-only ([2506.14276v2 Sec. 4.2](https://arxiv.org/html/2506.14276v2)).
  - Synthetic data was DSL-sampled riddles, with the model also trained to predict the DSL function names (same source).
  - Affiliation header: Cole at Mindware/MindsAI; Osman at mohamed@tufalabs.ai, listed under "MindsAI, Tufa Labs" (same source).

**ARC Prize 2025 (ARC-AGI-2).**
- **Kaggle private results:** NVARC 24.03%, the ARChitects 16.53%, MindsAI 12.64%, Lonnie 6.67%, G. Barbadillo 6.53%. The top score cost about $0.20 per task, across 1,455 teams ([ARC Prize 2025 results, 2025-12-05](https://arcprize.org/blog/arc-prize-2025-results-analysis); [arXiv 2601.10904](https://arxiv.org/abs/2601.10904)). The interview section of the blog gives MindsAI 15.42%, which conflicts with the table.
- **NVARC paper (Sorokin and Puget, NVIDIA, Nov 2025)** ([nvarc_2025.pdf in github.com/1ytic/NVARC](https://github.com/1ytic/NVARC); README links the 103k and 3.2M datasets on Kaggle):
  - **Why they trained at test time:** "using models pretrained on public tasks isn't effective. To perform well one has to use some form of training on the hidden test set (test-time fine-tuning or TTFT)" (Sec. 1).
  - **Why synthetic data:** "many ARC-AGI approaches quickly overfit to the provided 1000 training puzzles", so they put "full focus on creating good synthetic puzzles" (Sec. 2).
  - **The synthetic-data pipeline, in four stages:**
    1. Human descriptions from H-ARC and 160 from BARC gave 3,268 puzzle summaries, written with claude-opus-4 and gpt-oss-120b, plus 29 hand-labelled ARC-AGI-2 evaluation puzzles.
    2. Pairs of summaries were mixed (Instruct-SkillMix style) into 266,593 new summaries.
    3. 126,901 input-grid programs passed generated unit tests and produced at least 30 unique inputs.
    4. Output programs were kept only if several independently sampled programs agreed on all 30 inputs. That left 103,253 puzzles of about 30 pairs each.

    Most of the generation used gpt-oss-120b at 15k tokens/s on one 8×H100 node (Sec. 2.1–2.4).
  - **Training mix:** 3.2M augmented samples. NVARC's own puzzles were 89.7% of them; RE-ARC, ConceptARC, MINI-ARC and ARC-AGI-2 make up the rest (Table 1).
  - **Training:** full fine-tune of Qwen3-4B on 4 nodes × 8×H100 for 27 h, with NeMo RL's SFT (Megatron backend), tensor and context parallelism, and 256k-token packing (Sec. 3.1). At test time, a LoRA (r = 256) was fine-tuned on each puzzle (Sec. 3.2). Kaggle allowed 12 h on 4×L4, which is why they stayed at 4B or smaller.
  - **"The ARChitects solution … perfectly scales with data size … the more data we add - the better model we have"** (Sec. 3). Figure 1 shows validation loss falling as the synthetic share grows; no score-per-data table is given. BARC data was dropped as not useful (Sec. 3.5).
  - **Results (Table 2):** Qwen3-4B-Thinking 29.72 on the public LB; Qwen3-VL-2B 22.22.
  - **Noise:** "1-2 points variability in the scores across different re-runs of the same notebook" (Sec. 3.5).
  - **TRM component:** 2.08% when pretrained on real data alone. 7.5% with a 3k synthetic subset plus test-time fine-tuning. 10.0% after the deadline. In the ensemble it added about 0–1 point (Sec. 4).
- **The 2025 papers:**
  - **TRM (7M params):** 45% on ARC-AGI-1, 8% on ARC-AGI-2 ([arXiv 2510.04871](https://arxiv.org/abs/2510.04871)). Its data step builds 1,000 augmentations per puzzle and trains on the evaluation puzzles' demo pairs. Run by ARC Prize on 4 nodes × 8×H100, it scored 6.9% ([NVARC paper Sec. 4](https://github.com/1ytic/NVARC)).
  - **ARC Prize's HRM analysis (2025-08-15):**
    - The hierarchy had minimal impact.
    - Outer-loop refinement was the main driver: 1 → 2 loops gave +13 pp.
    - About 300 augmentations got near the maximum.
    - Training only on the 400 evaluation tasks gave 31% vs 41%: "most performance comes from fitting the evaluation tasks themselves" ([arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)).
  - **SOAR:** an LLM is fine-tuned on its own evolutionary-search traces with hindsight relabelling, reaching 52% on ARC-AGI-1 public test ([arXiv 2507.14172](https://arxiv.org/abs/2507.14172)).
  - **Poetiq:** a refinement harness took Gemini 3 Pro from 31% to 54% on ARC-AGI-2, at about $31 per task ([results blog](https://arcprize.org/blog/arc-prize-2025-results-analysis)).
  - **The ARChitects 2025:** a 2D-aware masked-diffusion LM with recursive self-refinement ([2601.10904](https://arxiv.org/abs/2601.10904)). The details "LLaDA-8B, 102 refinement steps" come only from a secondary survey ([arXiv 2603.13372](https://arxiv.org/pdf/2603.13372)). UNVERIFIED.
- **"Knowledge-dependent overfitting":** ARC says frontier ARC scores are "fundamentally constrained to knowledge coverage". A Gemini 3 run used the correct ARC colour mapping without being told about ARC ([2601.10904](https://arxiv.org/abs/2601.10904); [results blog](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

### Inferences
- Every large ARC-AGI-1/2 gain came from adapting to the test instance (TTT, refinement loops) or from very large verified synthetic data for a small model trained from near scratch. Neither maps directly onto continued SFT of an already RL-tuned 125B MoE with no base checkpoint.
- The closest analog to our plan is SOAR or STaR: train on your own verified search traces. That is published only for single-shot program synthesis, with an exact verifier.
- NVARC's pipeline shape transfers as a recipe for generating ARC-AGI-3-like games: describe the mechanic, mix two descriptions, generate code with self-tests, and keep only what independent samples agree on. NVARC's scale (one 8×H100 node, months) does not transfer to 15 days.
- Model size beat ARC pretraining for zero-shot ability (Cole & Osman). Franzen's model switches agree (Q3), and so does our 27B result (14.5 vs 54.7 offline).

### Gaps
- No primary ARChitects 2025 write-up was found. The MindsAI 2025 write-up was not read.
- The NVARC paper gives no score-versus-data-volume table, only a loss curve.

## Q2. ARC-AGI-3: format, scoring, the Foundation's view of fine-tuning and overfitting, and frontier scores

### Takeaway
ARC-AGI-3 scores action efficiency on held-out games.
- **Format:** 64×64 grids with 16 colours; 5 actions plus undo plus click.
- **Scoring:** squared human/AI action ratio per level, levels weighted by index, cut off at 5× the human action count.
- **Training stance:** ARC explicitly calls training on synthetic ARC-AGI-3 lookalikes "domain-specific overfitting". It discounts that on its own leaderboard, but the Kaggle rules don't forbid it.
- **Harness vs model:** at the frontier the harness is everything (Opus 4.6: 0% → 97.1% on one game). No Qwen3.8 model appears on ARC's ARC-AGI-3 leaderboard.

### Cited Findings
- **Format and sets** ([ARC-AGI-3 technical report, arXiv 2603.24621, 2026-03-24, Sec. 2.3, Table 1](https://arxiv.org/html/2603.24621v1)):
  - Turn-based environments with levels; 64×64 grid, 16 colours; five key actions plus Undo plus a coordinate click.
  - Sets: Public Demo 25, Semi-Private 55, Fully Private 55.
  - The public set is easier, and public scores never appear on the official leaderboard.
- **RHAE scoring** (Sec. 4.1–4.3, same source):
  - Level score S = min(1, h/a)², where h is the second-best first-run human action count.
  - Environment score is Σ(l·S_l) / [n(n+1)/2].
  - Cutoff at 5× the human baseline per level.
- **Overfitting taxonomy** (Sec. 4.3.1, same source):
  - *Task-specific:* agents trained on, or configured with knowledge of, the public environments.
  - *Domain-specific:* agents trained on "synthetically generated ARC-AGI-3 lookalike environments", or harnesses with ARC-AGI-3-specific strategies.
  - The official leaderboard "will not use a harness" and aims to "discount score increases that come from direct targeting of ARC-AGI-3".
- **Harness effect:** on a TR87 variant, Opus 4.6 scored 0.0% without a harness and 97.1% with the "Duke" harness; on BP35 it scored 0% both ways (Sec. 4.3.1, same source). At launch, frontier models scored below 1% (Gemini 3.1 Pro 0.37%, GPT-5.4 High 0.26%, Opus 4.6 Max 0.25%) (Table 2).
- **ARC's verified leaderboard data, fetched 2026-10-18** ([evaluations.json](https://arcprize.org/media/data/evaluations.json), dataset `v3_Semi_Private`, 102 rows):

  | Model | Harness | Score |
  |---|---|---|
  | GPT-6 Astra high | provider adapter | 99.95% |
  | Gemini 4 Argon low | provider adapter | 97.5% |
  | GPT-6 Astra max | standard harness | 62.7% |
  | Claude Opus 5 high | — | 30.2% |
  | DeepSeek V4.1 Flash | provider adapter, best | 3.3% |
  | Claude Haiku 5.5 max | standard | 0.45% |
  | Claude Haiku 5.5 max | provider adapter | 2.8% |

  - No Qwen model has a v3 row. Qwen3.8-27B xhigh has ARC-AGI-2 semi-private 42.4% (same file).
  - The leaderboard page describes two ARC-AGI-3 harnesses: "Standard" (carries model-selected notes forward) and "Provider Adapter" (keeps reasoning state, compacts long conversations) ([arcprize.org/leaderboard](https://arcprize.org/leaderboard)).
- **The Kaggle competition:** milestones 2026-06-30 and 2026-09-30 at $25k / $10k / $2.5k each. "All code and methods must be open sourced to be eligible for prizes"; no internet during evaluation ([arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3)). The rules page doesn't address fine-tuning (same source).
- **Do public games reappear privately?** The Naestro model card claims "the public demo games can reappear in the private test set under different names" ([HF star-ga/naestro-agi3-27b](https://huggingface.co/star-ga/naestro-agi3-27b)). That conflicts with the technical report's separate sets ([2603.24621 Table 1](https://arxiv.org/html/2603.24621v1)). UNVERIFIED claim; treat it as false unless ARC says otherwise.

### Inferences
- The Kaggle rules allow training on lookalike games, and the private games come from the same designers and engine. "Domain-specific overfitting" may well help on the Kaggle LB, even though ARC won't count it as AGI progress.
- RHAE's squared ratio makes action count matter. But our cleared levels are already near-optimal (81% of moves on the hindsight-shortest route). The 5× cutoff and the uncleared levels are where the points go.
- Small open models with the official harness score about 0–3% on ARC-AGI-3 against our ~31 LB with the Duck lineage. At our model scale the harness is a far bigger lever than model identity, and the training target should be "use the harness better and faster", not "know ARC".

### Gaps
- ARC Prize states no explicit position on fine-tuning for the Kaggle track. I found no official statement on whether the Kaggle private 110 equals the semi-private 55 plus the fully private 55. Kaggle shows about 50% of test data on the public LB, consistent with 55 + 55 (see W5b).

## Q3. ARC-AGI-3 preview (2025) and the 2026 Kaggle competition: winners, methods, lineage

### Takeaway
**Preview (2025).** It was won by non-LLM online learners: StochasticGoose (Tufa Labs) was a CNN trained during play. That is the only test-time-training precedent on ARC-AGI-3 itself.

**2026 milestones.** Both went to training-free LLM tool agents on local open models.
- M1: Tufa's Duck on Qwen3.6-27B.
- M2: Franzen's Duck fork on Flash-Next. Franzen is the ARChitects co-author who invented the 2024 TTT/DFS/PoE stack, and he explicitly did not fine-tune.

**Lineage of the top teams.**
- Tufa Labs (68.91 now) holds the TTT lineage through MindsAI and StochasticGoose; its current method is undisclosed.
- NVIDIA's NVARC co-author Puget (CPMP) is on the LB. Team NVARC3 is at 43.94; that his team is NVARC3 is UNVERIFIED.

### Cited Findings
- **Preview (2025-07-18 to 08-19):** 3 public and 3 private games.
  - StochasticGoose (Tufa Labs, Dries Smit) 12.58%, 18 levels.
  - Blind Squirrel 6.71%: a state graph plus a ResNet18 value model, "retrained a small value model during play whenever its score improved".
  - LLM entries were weak: Fluxonian 8.04% with 0 games won ([arcprize.org/blog/arc-agi-3-preview-30-day-learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings); [2603.24621 Sec. 6](https://arxiv.org/html/2603.24621v1)).
  - StochasticGoose "first spent about 350 moves clicking with no result, then learned which actions were clickable" ([preview blog](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)).
- **How StochasticGoose works:**
  - A 4-layer CNN (32→64→128→256) on 16-channel one-hot 64×64 frames, trained online with binary cross-entropy on whether (state, action) changed the frame.
  - A 200k-entry deduplicated experience buffer; the model and buffer reset on each new level; clicks are sampled through a convolutional coordinate head.
  - Lead Dries Smit, adviser/reviewer Jack Cole (MindsAI) (`competitors/ARC3-solution/README.md`, lines 1–80; repo github.com/DriesSmit/ARC3-solution, HEAD a6e77bbf, 2026-01-27).
- **On the full benchmark,** StochasticGoose scored 0.25% ([DataCamp, secondary](https://www.datacamp.com/zh/blog/arc-agi-3)). UNVERIFIED: no primary source found.
- **Graph-Based Exploration** ([arXiv 2512.24156](https://arxiv.org/abs/2512.24156)) is training-free: median 30 of 52 levels on six preview games, 3rd on the private preview leaderboard (abstract, via arXiv API).
- **Milestone 1** ([arcprize.org/blog/arc-prize-2026-milestone-1, 2026-07-06](https://arcprize.org/blog/arc-prize-2026-milestone-1)):
  - 1st, Tufa Labs "The Duck": Qwen3.6-27B FP8, an agent writing Python in a live REPL. "Hand-crafted tools actually hurt"; gains credited to "multimodality and better base models".
  - 2nd, Reki: Gemma-4-31B, JSON actions.
  - 3rd, Md Boktiar Mahbub Murad, "forge": Gemma-4-31B. "local public-game checks were not a reliable proxy for the leaderboard".
  - The post doesn't say whether any entry was fine-tuned.
- **Milestone 2:**
  - Results: Franzen 27.89 ([`competitors/tong-m2-comparison/WRITEUP.md:5`](../../../../competitors/tong-m2-comparison/WRITEUP.md)), Lord Han Solo 23.8, Lohit Siriki 22.5 ([TechTimes, secondary](https://www.techtimes.com/articles/328542/20261006/arc-agi-3-scaffolding-beats-model-upgrades-same-ai-two-settings-49x-score-gain.htm)). No arcprize.org M2 post exists yet: `/blog/arc-prize-2026-milestone-2` returns 404 on 2026-10-18.
  - Franzen, quoted from WRITEUP.md:
    - "I did not train or fine-tune the models" (line 318).
    - "Better models help a lot … Each switch brought a substantial improvement" (line 15).
    - Keeping the model's thinking in history is deliberate; "removing it leads to a large regression in score" (line 69).
- **Franzen is the ARChitects co-author:**
  - His ARC-AGI-3 solution is github.com/da-fr/arc-agi-3-solution (`competitors/tong-m2-comparison/README.md:3`).
  - ARC Prize's model registry lists the ARChitects 2024 paper at github.com/da-fr/arc-prize-2024 ([models.json](https://arcprize.org/media/data/models.json), id "ARChitects").
  - The PoE paper's first author is Daniel Franzen ([2505.07859](https://arxiv.org/abs/2505.07859)).
- **The current LB** (2026-10-18; W5b §2): Tufa Labs 68.91, Yi-Chia Chen 67.49, mtg 50.37 … NVARC3 43.94 (9th). Methods for ranks 1–3 are undisclosed (W5b §1; `finetune-research-2026-10-17/landscape.md`).
- **Tufa's lineage, in one team:** StochasticGoose (preview winner), the Duck (M1 winner), and MindsAI's Osman at tufalabs.ai ([2506.14276v2 header](https://arxiv.org/html/2506.14276v2)).

### Inferences
- The person with the deepest ARC fine-tuning expertise in this competition chose, at M2, to put all effort into harness and throughput. No public LB evidence shows that fine-tuning Flash-Next helps. Our prior for "fine-tuning gives +29 LB" should be low.
- The two 67–69 teams are 17+ points above everyone else. If either trains, Tufa is the likeliest: MindsAI's TTT lineage plus RL infrastructure, 166 entries. That is speculation until they release after 11-02.
- StochasticGoose and Blind Squirrel show that online learning during play works on ARC-AGI-3 when done by a tiny side model, not the LLM.

### Gaps
- Tufa Labs' and Yi-Chia Chen's models and training are not public. No ARC write-up for Milestone 2 was found. NVARC3's membership is unconfirmed.

## Q4. Public evidence that fine-tuning improved an ARC-AGI-3 score

### Takeaway
There is no public evidence of a meaningful ARC-AGI-3 gain from fine-tuning, and none at all for Flash-Next. The only positive reports are small, on weak 27B bases: +0.7 LB, and +2 public points that didn't transfer to the hidden set. Every near-100 public-set paper is a training-free harness on a frontier model.

### Cited Findings
- **arXiv:** the API phrase search `all:"ARC-AGI-3"` (27 hits, newest 2026-10-15; [query URL](https://export.arxiv.org/api/query?search_query=all:%22ARC-AGI-3%22&max_results=60&sortBy=submittedDate&sortOrder=descending)) contains only one paper that trains weights.
  - That paper is MAP ([2605.13037](https://arxiv.org/abs/2605.13037)), which trains on a MAP-2K trajectory dataset. Its abstract reports no ARC-AGI-3 fine-tune vs base numbers; ARC-AGI-3 is used for frontier-model prompting (22 of 25 games above a near-zero baseline).
  - **Training-free harnesses on frontier models:**
    - Tycho ([2607.28287](https://arxiv.org/abs/2607.28287)): 88.5–100 RHAE.
    - Kepler ([2610.00834](https://arxiv.org/abs/2610.00834)): 100 RHAE.
    - Memento 3 ([2610.11794](https://arxiv.org/abs/2610.11794)): 100 RHAE.
    - Twin ([2608.14490](https://arxiv.org/abs/2608.14490)): 97.8% of levels vs 7.8% for the base model alone.
    - Executable world models ([2605.05138](https://arxiv.org/abs/2605.05138)): GPT-5.5 58.1%.
  - **The only small-open-model paper:** AERA on Qwen2.5-0.5B ([2605.25931](https://arxiv.org/abs/2605.25931)) is training-free: RHAE 0.2116 on public and "0.30" on the private 55. The units are ambiguous and the score is likely near zero. The authors claim all 25 public games are solvable by trivial heuristics.

  These summaries come from abstracts via the arXiv API; full texts not read.
- **Hugging Face:**
  - Naestro-AGI3-27B is a QLoRA r16 on Qwen3.6-27B over procedural ARC-AGI-3-style sims (frame → goal + plan). It reports no scores and says first-exposure generalisation "is the unsolved core of ARC-AGI-3" ([model card](https://huggingface.co/star-ga/naestro-agi3-27b)).
  - `ericmao/arc-agi3-qwen36-{sft,sft-v2,sft-v3,grpo,human100,spatial-continuation-100,raw-control}` are LoRA adapters on Qwen3.6-27B (TRL; created 2026-07-21) with empty cards and no scores ([HF API search "arc-agi-3"](https://huggingface.co/api/models?search=arc-agi-3)).
  - Every other "arc" model is ARC-AGI-1/2 (Trelis Qwen3-4B series).
- **Kaggle** (sibling W5b §1, Kaggle UI):
  - **739047:** STaR LoRA on Qwen3.6-27B Duck trajectories, LB 1.25 → 1.94. "Naively scaling the data hurt … Curation > quantity"; "High variance"; "Base-model-specific".
  - **Scott Le Grand:** behaviour cloning on 27B gave "a two-point jump on the public games but it didn't translate performance on the hidden data". His advice: "behaviorally clone from the same model".
  - **748593 and 742835:** several H200 users (Slow∞Loop at rank 11, AAAAAtjc, blakewest) report no gain from SFT or RL on Flash-Next ("MoE structure makes training unstable").
  - **743319:** selectively down-weighting reasoning tokens caused a "no tool call" collapse (avg 25.07 → 4.99).

### Inferences
- Treat any fine-tune as a test-first bet with a low prior. The only levers with in-house evidence are tokens/action (the RSI toy: a 2× gain mostly from shorter answers) and stuck-level breakthroughs.
- The 739047 caveat that "more data hurt" parallels NVARC's "approaches quickly overfit to the provided 1000 training puzzles". With about 6k level clears from 25 public games, our own logs are a small, correlated dataset, in NVARC's terms a "1000 training puzzles" regime. Data diversity (other games) matters more than volume from the same 25 games.

### Gaps
- No primary source reports fine-tuned vs base on the ARC-AGI-3 private set.

## Q5. Test-time training for interactive settings: is it feasible during the Kaggle run, and how would ARC-AGI-1/2 TTT translate?

### Takeaway
LLM-weight TTT on Flash-Next during the Kaggle run is infeasible. The W4A16 weights alone take 69.85 GB of the 96 GB card while the GPU serves 10–14 streams, and no ~100B training fits on that card.

Two lighter forms exist:
- **In-episode LoRA TTT (aTTT)** gives about +5 points on ALFWorld and SWE-bench Lite at 1.9× cost, on models that fit. Its own failure mode is drift amplification when the agent is stuck.
- **A tiny model trained online beside the LLM** (StochasticGoose, Blind Squirrel) is cheap and has ARC-AGI-3 precedent.

The ARC-AGI-1/2 TTT gain (2.4×–6×) came from per-task demo pairs used as a supervised signal. In ARC-AGI-3 the analog signal is the game's own transitions: frame changes and level-ups.

### Cited Findings
- **Our serving footprint:** the Intel W4A16 target weights take 69.85 GB of GPU, with the PLE table offloaded (`arc-agi-v4/arms.md:665`). The ms-swift LoRA recipe needs about 78–93 GiB per GPU across 8 GPUs at 4–8k tokens (`finetune-plan-2026-10-17.md` §1).
- **On the RTX PRO 6000:** "rtx 6000 pro can not be used for training of any ~100B model even with aggressive cpu offload and quantization" (AAAAAtjc, Kaggle 742835, via W5b §1).
- **ARC-AGI-1 TTT cost:** 64 LoRA steps, 51 s per task on a 4090 for an 8B model ([2505.07859v2 Table 4](https://arxiv.org/html/2505.07859v2)). NVARC used LoRA r256 per puzzle on 4×L4 within 12 h for 240 tasks ([NVARC Sec. 3.2](https://github.com/1ytic/NVARC)).
- **aTTT** ([arXiv 2607.03441, 2026-07-03](https://arxiv.org/abs/2607.03441)):
  - Method: continuous LoRA updates during multi-turn episodes, served through vLLM's runtime LoRA API. Tokens in n-grams repeated from prior updates are down-weighted.
  - Results: +5.0 on ALFWorld, +4.9 on SWE-bench Lite, at about 1.9× the no-TTT cost.
  - Caveats: gains appear where models "already have task competence but drift", so the method "mainly preserves existing competence rather than teaching new abilities". Plain TTT "can amplify drift when the agent gets stuck".
- **Online side models on ARC-AGI-3:** StochasticGoose (online CNN, reset per level; `competitors/ARC3-solution/README.md`) and Blind Squirrel (value model retrained during play; [preview blog](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)).
- **Small models in the refinement loop:** HRM and TRM gains came mostly from refinement loops and fitting the evaluation tasks ([HRM analysis](https://arcprize.org/blog/hrm-analysis)). CompressARC (76k params, no pretraining) reached about 20% on ARC-AGI-1 at about 20 min per puzzle ([results blog](https://arcprize.org/blog/arc-prize-2025-results-analysis)).

### Inferences
- The ARC-AGI-3 analog of TTT for us is in-context learning inside the harness: memory, retained functions, the 118k context. Franzen's stack already maximises it, and in his tests turning off the structured world model plus long context worked best (WRITEUP.md:295).
- A tiny online action-effect model is feasible: CNN-sized, on CPU or about 1 GB of GPU, trained on the run's own transitions. It is game- and time-agnostic. It targets our measured failure directly: in 96% of stuck-level decisions an untried action reached a new board. This is harness work, not fine-tuning, and it is the only TTT form with ARC-AGI-3 evidence.
- aTTT-style LoRA updates would need SGLang runtime LoRA on Flash-Next, plus a trainer co-resident on the same GPU, so not before 11-02. Its drift warning applies offline too: self-generated stuck-level text must be filtered to breakthroughs only (R4), never raw stuck traces.

### Gaps
- Not checked: whether Pennyroyal SGLang can serve an unmerged LoRA on Flash-Next's BF16 modules. That matters for serving any adapter without requantisation, and for any aTTT variant.
- No published result on TTT of an LLM's weights on ARC-AGI-3.

## Implications for our plan (15 days, one 8×H200 node, about 11 Kaggle GPU-h/day)

1. **Keep Flash-Next; don't switch to a smaller, easier-to-train model.** Model size beat ARC pretraining (Cole & Osman Sec. 4.2). Every one of Franzen's switches to a larger model gained, and our 27B scored 14.5 vs 54.7.
2. **Lower the prior on "fine-tuning reaches 60".**
   - The ARChitects co-author won M2 without fine-tuning.
   - No public Flash-Next fine-tune gain exists (several H200 teams tried).
   - Every ARC-AGI-3 paper with a high score is training-free.

   Gate the training prong on cheap, falsifiable steps, and keep the harness prong alive as the final-pick fallback.
3. **What to train for: efficiency, not knowledge.** ARC-AGI-1/2 gains came from instance adaptation; ARC-AGI-3's analog is in-context. Training should make the in-context loop cheaper (R3: shortest correct reply for the same decision, i.e. more iterations per 9 h) and better at breaking stalls (R4). It should not "teach ARC rules". The HRM finding that refinement steps beat architecture supports "more loop iterations per token".
4. **Self-improvement precedent supports ReST-EM / STaR on our own verified traces.** SOAR's hindsight relabelling reached 52%; the 739047 STaR LoRA was positive but small. Do this, with these guards:
   - curation over volume (739047);
   - train on the same model's outputs only (Le Grand);
   - no selective down-weighting of reasoning without monitoring the no-action rate (743319);
   - exclude raw stuck segments (aTTT drift).
5. **Synthetic games: borrow NVARC's pipeline shape at small scale.** Steps: describe the mechanic, mix two mechanics, generate the game with self-tests, keep it only if a solver or replay verifies it. Seed from DreamTeam's `game_creator` and arc-interactive. Target hundreds of games, not 100k.
   - NVARC's evidence: data diversity is what scaled, and training on the 1,000 given puzzles overfit. So diversity of mechanics should come before more trajectories from the 25 public games.
   - Test first: show a held-out-family gain on validation games before spending node time.
6. **Don't attempt LLM test-time training on Kaggle before 11-02** (no memory and no trainer on the card).
   - Test first: a StochasticGoose-style online action-effect predictor as a flagged harness feature (CPU or a small GPU slice, reset per level) that ranks untried actions. It is the only TTT with ARC-AGI-3 evidence, and it targets the 96% untried-action finding.
7. **Don't use ARC-style D8 and colour augmentation of trajectories now.** It was worth +18 points in ARC-AGI-1 (44.5 → 62.5), but agent traces reference directions, coordinates and colours in free-text thinking. Remapping actions and rewriting the text is error-prone. Consider it only for perception-only auxiliary data.
8. **Evaluation discipline from the lineage:**
   - NVARC saw ±1–2 points across reruns even on ARC-AGI-2, and ours is ±5 on the LB.
   - forge and 739047 both found offline public checks unreliable.
   - The public set is easier (ARC Sec. 3.6), and the AERA paper says public games fall to trivial heuristics.

   So: hold out games by mechanic family, report wins and time-to-win first, and take several LB draws before calling a gain.
9. **Rules:** training on lookalike games is "domain-specific overfitting" in ARC's taxonomy but isn't forbidden on Kaggle. Prize eligibility needs open code and weights. Keep unlicensed data (sonpham, ARC-AGI-3-Training-Data) out of released weights, as the prior plan says.

## Fact-check (independent verifier, 2026-10-18)

Method: each source re-fetched independently (NVARC README and `nvarc_2025.pdf` from `raw.githubusercontent.com/1ytic/NVARC/main`; arXiv HTML and API for 2505.07859v2, 2603.24621v1, 2605.13037v1, 2605.25931v1, 2607.03441v1 and the `all:"ARC-AGI-3"` query; `arcprize.org/media/data/evaluations.json` and `models.json`; the arcprize.org preview, M1, HRM, 2025-results, leaderboard and competition pages; GitHub API; HF API), plus the local files at the cited lines. No browser.

### Verdicts on the claims

1. **NVARC (103,253 puzzles; 3.2M samples; full FT of Qwen3-4B on 4 nodes x 8xH100 for 27 h; per-puzzle LoRA r=256 TTT; "isn't effective").** CONFIRMED. Sec. 1: "using models pretrained on public tasks isn't effective. To perform well one has to use some form of training on the hidden test set (test-time fine-tuning or TTFT)". Sec. 2.4: "103253 puzzles". Table 1 totals 3,255,481 samples ("3.2M"). Sec. 3.1: "full fine-tuning of the 4B model we used 4 nodes of 8xH100 for 27 hours". Sec. 3.2: LoRA r = 256, alpha = 32. The README links `sorokin/nvarc-synthetic-puzzles` (103k) and `sorokin/nvarc-augmented-puzzles` (3.2M).
2. **ARChitects Table 2 (NeMo-Minitron-8B 18.3 / 44.5 / 62.5 / 67.6 / 71.6).** CONFIRMED exactly (2505.07859v2 Table 2, two-guess accuracy, ARC-AGI-1 public eval). Llama-3.2-3B 14.9 / 40.9 / 52.9 / 59.5 / 61.4 is also correct.
3. **Franzen won M2 (27.89), "I did not train or fine-tune the models", and da-fr hosts both repos.** CONFIRMED. `competitors/tong-m2-comparison/WRITEUP.md` has 27.89 at line 5 and the quote at line 318 (in the compute section: "This was inference and testing compute; I did not train or fine-tune the models."). README.md line 3 links github.com/da-fr/arc-agi-3-solution, which exists per the GitHub API (owner da-fr, name "Daniel Franzen", created 2026-09-30). In `models.json`, id "ARChitects" has paperUrl github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf. The WRITEUP itself doesn't say "winner". The M2 win (1st, 27.9%, $25K) comes from ARC Prize's X announcement, seen only as search-result snippets.
4. **ARC-AGI-3 report: domain-specific overfitting, no-harness official LB, public set easier.** CONFIRMED. One nuance matters for us. Sec. 4.3.1 defines domain-specific overfitting as training on "many synthetically generated ARC-AGI-3 lookalike environments, *or using a harness that contains ARC-AGI-3 specific strategies*". Our harness work therefore falls under the same heading. The section also says "the official leaderboard will not use a harness" and "will seek to discount score increases that come from direct targeting of ARC-AGI-3". Sec. 3.6 calls the public set "intentionally easier" and the private set "significantly more difficult", with "limited overlap with the mechanics" of the public set. Table 1 gives 25 / 55 / 55.
5. **arXiv: 27 hits; only MAP trains weights; no ARC-AGI-3 fine-tuned-vs-base numbers; top public-set results are training-free.** CONFIRMED. The query returns totalResults 27, the newest 2610.22289 (2026-10-15). In the abstracts, only MAP describes training weights. The MAP full text makes the point stronger: MAP-4B (Qwen3-4B-Thinking, ms-swift, 8xH800) was evaluated only on ALFWorld, TextCraft and ScienceWorld. Its ARC-AGI-3 results (Table 2, App. C.2) are Claude 4.6 Opus prompting, so no ARC-AGI-3 fine-tune result exists anywhere in the paper. Tycho, Kepler ("one frozen Claude Opus 5 configuration"), Memento 3 ("frozen LLM agents") and Twin are all harnesses. Other near-100 public-set results not listed are also training-free harnesses: VISTA 2610.02200 (100.00), Schema 2609.39140 (99.2), PRO-LONG 2607.20064 (97.4 best@2) and Prime Agent 2608.23552 (95.5).
6. **StochasticGoose: 4-layer CNN trained online on frame-change; 200k dedup buffer reset per level; 12.58%, 18 levels; LLM entries lower.** CONFIRMED. `competitors/ARC3-solution/README.md` lines 67-79 give the 4-layer CNN (32->64->128->256), BCE on (state, action) -> frame_changed, a 200K hash-deduplicated buffer, and buffer plus model reset on a new level. `custom_agents/action.py` has `deque(maxlen=200000)`, and the model and optimizer are re-created on a level-up. Upstream HEAD is a6e77bbf (2026-01-27), matching the local copy. The preview blog gives 12.58% and 18 levels. The LLM entries were Fluxonian (DSL + LLM) 8.04%, Play Zero Agent 4.37% and Tomas Engine 3.70%, all below 12.58%. Nuance: Fluxonian's 8.04% beat the 2nd-place Blind Squirrel's 6.71%, so "LLM entries were weak" (line 148) holds only against the winner.
7. **aTTT: +5.0 ALFWorld, +4.9 SWE-bench Lite at about 1.9x cost; plain TTT amplifies drift when stuck; gains mainly preserve competence.** PARTIALLY CONFIRMED. The abstract says "improves success by *up to* 5.0 points on ALFWorld and 4.9 points on SWE-bench Lite". These are best cases, not typical gains:
   - **SWE-bench Lite:** the +4.9 is Qwen3.5-27B only; Qwen3.5-9B gained +2.9 (Table 2).
   - **ALFWorld:** the +5.0 is the best model and setting. Qwen3.5-27B "gains little" on ALFWorld.
   - **Models tested:** dense Qwen3.5-4B, 9B and 27B, and Gemma-3-12B.
   - **The 1.9x figure:** this is wall-clock for 140 ALFWorld episodes on Qwen3.5-9B with 16 concurrent episodes. *Dedicated training GPUs* compute the updates and hot-swap them into the vLLM server (App. A), so it is not a single-GPU co-resident cost.
   - **Drift and competence:** the drift-amplification and "mainly preserves existing competence rather than teaching new abilities" statements are verbatim from the abstract.
8. **evaluations.json v3_Semi_Private.** CONFIRMED. There are 102 rows:
   - **GPT-6 Astra high (provider adapter):** 0.99946.
   - **GPT-6 Astra max:** 0.6271. It is the top non-adapter row. Its models.json note says only "'max' reasoning effort", but the leaderboard page says ARC-AGI-3 results "use either the Standard harness ... or the Provider Adapter harness", so it is a standard-harness row. The best row explicitly labelled "standard harness" is GPT-6.1 Sol max at 0.527.
   - **DeepSeek V4.1 Flash:** at most 0.0326 (low effort, provider adapter).
   - **Qwen:** no v3 rows. Qwen3.8-27B appears only in v1/v2; xhigh scores 0.4236 on v2_Semi_Private, matching the 42.4% at line 118.

### Other errors found in the notes
- **Implication 7 (line 267):** the "+18 points (44.5 -> 62.5)" is the *inference-time* "16xAug" step: one candidate sampled under each of 16 random augmentations, keeping the two most probable (Table 2 caption). It is not training-data D8/colour augmentation. Table 2 has no training-augmentation ablation, so it gives no number for augmenting trajectories.
- **Line 11 ("+5 to +18"):** Table 2 increments are 16xAug +18.0 / +12.0, PoE +5.1 / +6.6 and DFS +4.0 / +1.9 (8B / 3B), so the range is about +2 to +18.
- **Line 49 (NVARC stage 1):**
  - The H-ARC and BARC descriptions covered 716 ARC-AGI-2 training puzzles. They gave 716 x 3 summaries: one from claude-opus-4 and two from gpt-oss-120b.
  - The 3,268 total = 716 x 3 + 29 hand-labelled + 91 evaluation + 1,000 training summaries. The last two groups were written with claude-sonnet-4-5, which the notes omit.
  - So the 29 are part of the 3,268, not added to it.
- **Line 60 (TRM in the ensemble):** the measured effect ranged from -1.25 to +0.97 points, not 0 to 1:
  - in-competition: 20.28 with TRM vs 21.53 without;
  - late, with Qwen3 2B: 21.53 -> 22.50;
  - late, with Qwen3 4B: 27.22 unchanged.
- **Line 194 (Twin):** the 97.8% is levels cleared (179 of 183), while the base model's 7.8% is a score. Like for like, Twin scores 93.3, against 61.1 for an off-the-shelf harness and 7.8 for the base model.
- **Line 196 (AERA):**
  - The "0.30" on the 55-game private set belongs to a different Kaggle entry, a BFS solver with an offline pre-solve cache. The paper reports it as that entry's public-LB score; it is not AERA on Qwen2.5-0.5B.
  - The abstract says every public game is "reachable through non-intelligent strategies", which is weaker than "solvable".
- **Line 129 (gap):** the report does make an official statement. Table 1 and Sec. 3.6 assign the semi-private 55 to testing models "behind an external API" and the fully private 55 to "the competition" (as of 2026-03). That bears directly on whether the Kaggle 110 equals 55 + 55.
- **Line 148:** Sec. 6 of the report, cited for the preview results, doesn't mention Fluxonian; only the blog does.
- **Line 249 (implication 1):** Cole & Osman's "model size beat pre-training" holds only for zero-shot and AIRV-only. The same Sec. 4.2 says more ARC pre-training "does significantly improve score in the TTFT + AIRV setting". Line 39 has the qualifier; implication 1 drops it.
- **Line 274:** the competition page requires "All code and methods must be open sourced". It says nothing about weights, so "open code and weights" needs another source, such as the Kaggle rules, which were not checked.
- **Line 120:** the page lists M2 prizes of $25K / $10K / $2.5K. ARC Prize's M2 announcement, seen only via search snippets of the X post, lists $25K / $7.5K / $5K. This is unverified and doesn't affect the plan.
- **Paraphrases set in quotation marks:**
  - Line 67 (HRM): the blog says "the large majority of the performance is driven by training on the tasks seen at evaluation time".
  - Line 147 (Blind Squirrel): "Whenever the score improves, it back-labels that level with distances and retrains a small ResNet18-based value model".
  - Line 159 (forge): "local public-game checks weren't a reliable leaderboard proxy".
  - Line 30: "to minimize 'conceptual leakage'".
- **Checked and correct:**
  - Q1: lines 17, 19, 20, 29-31, 33-35, 37-41, 44 and 62.
  - Q2: lines 98-105, 118 and 121.
  - Q3: lines 145-146, 149 and 155-157.
  - Q4 and Q5: lines 189, 195, 200-201, 228, 230, 236 and 239.
  - Local: `arms.md:101` (14.51 vs 54.69).
