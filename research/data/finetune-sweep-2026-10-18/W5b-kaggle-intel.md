# Kaggle competition intel: fine-tuning discussions, leaderboard, public Flash-Next model variants, rules (collected 2026-10-18)

Method: Chrome browser on the Kaggle web UI (worked, no CLI fallback needed for pages); the model listing used `kaggle models list/get` (CLI). Page header on 2026-10-18 reads "Featured Code Competition · 16 days to go". Base URL B = https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3

## 1. Discussion threads on fine-tuning / training

### Takeaway
Public evidence that fine-tuning helps is thin and mostly negative. Several people report an SFT/BC gain on Qwen 27B-class models (about +2 points on public games, small on LB), and several (H200 users) report no gain at all on Flash-Next. No one claims a Flash-Next fine-tune with a measured leaderboard gain. Nobody top-ranked discloses a method. The best-documented caution is a loss-masking failure mode (no-action collapse).

### Cited Findings
- Thread 739047 "Open-sourcing winning-trajectory datasets + a LoRA for the duck harness (STaR fine-tuning)", manas joshi (255th), posted ~2 months ago, 8 votes, 4 comments. [B/discussion/739047]
  - Method: ran the open duck (TAAF) harness with Qwen3.6-27B, kept only games where the agent reached a level, LoRA-SFT on the winning trajectories (STaR / rejection sampling). Released three CC0 datasets: arc3-sft-trajectories, arc3-duck-lora-sft (Qwen3.6 adapter), duck-eval-results.
  - Gotchas quoted: "Train assistant-tokens-only; the tool-calls are OpenAI-format, flatten before templating." Multimodal arch: serve as Qwen3_5ForConditionalGeneration, copy the processor configs into the merged checkpoint. "Offline levels don't predict the leaderboard - only submit to measure."
  - Claimed gain in a reply: "it was a big jump went from 1.25 to 1.94 fine tuning on qwen" (LB score, rank ~300 to ~133 at that time; the LB top was ~7 then). That is +0.7 absolute on a ~1-2 score scale.
  - Author's caveats: "High variance - same recipe, different runs, scores swung a lot"; "Naively scaling the data hurt - more trajectories at the same recipe made it worse. Curation > quantity"; "Base-model-specific" (an adapter for one base won't help a stronger one).
  - Ya Xu (624th): LoRA "only backfired on me"; cites Chollet saying the 25 local games "[are] not meant to be used as training data" (second-hand, UNVERIFIED).
- Thread 742835 "Finetuning a model (SFT or RL) on the RTX 6000 Pro?", Nick Pellegrin (1227th), 25 days ago, 5 comments. [B/discussion/742835]
  - blakewest (4424th), 18 days ago: SFT on H200s, "not RTX 6000 Pro"; tried Qwen Max and "Astra" as teachers, synthetic variants of LS20 and CD82, training on only "stall" points (loops, lives lost) and on "periodic" trajectories (every 10th move, teacher asked to move from the Duck game state), different learning rates, Flash-Next and 27B. "No permutation of these has been able to make meaningful improvements. I'm pretty convinced that finetuning with Qwen Flash Next is very difficult because it's an MoE model."
  - AAAAAtjc (2107th), 24 days ago: "I successfully built and validated rl training pipeline of Qwen 3.8 Next Flash based miles, but I did not gain any performance in arc-agic-3 yet"; used "a full H200 node. rtx 6000 pro can not be used for training of any ~100B model even with aggressive cpu offload and quantization".
  - Scott Le Grand (522nd): "what others have written is that you have to behaviorally clone from the same model or there's some sort of impedance mismatch on the tokens"; BC on Qwen 3.6-27B gave "a two-point jump on the public games but it didn't translate performance on the hidden data."
  - Nikolai Nedovodin: warns that distilling from OpenAI-model outputs conflicts with OpenAI terms (prohibits using output to develop competing models) and complicates the open-source obligation for winners.
- Thread 748593 "Which is a better choice for training, Qwen3.8-27B or Qwen3.8-Flash-Next?", Slow∞Loop (11th in competition), 6 days ago, 6 votes. [B/discussion/748593]
  - Author: SFT and RL on Flash-Next trouble for a week; "Its MoE structure makes training unstable"; training 27B "works well although scores grows slowly".
  - Scott Le Grand: "2 RHAE points on the public data out of 3.8-27B with SFT/behavioral cloning but I so far haven't been able to get anything out of flash-next."
  - Slow∞Loop: "I haven't been able to get anything out of flash-next too! I focus on 3.8-27B." Recipe quoted: "QLoRA 4-bit + FSDP2 + PLE CPU Offload. It can help you finetune Qwen3.8-Flash-Next in 2*96G GPU."
  - AAAAAtjc (2107th): "I can [fine-tune], but I did not gain any performance". Son Pham (5th): "You are able to fine tune at all? That is truly amazing!"
- Thread 743319 "Partial-reasoning supervision made our agent stop acting: 'no tool call' collapse in a small fine-tune", wkdrbwnd1 (977th), 23 days ago, 0 comments. [B/discussion/743319]
  - Data: 31 agent trajectories from 20 public games; eval on 5 held-out public games (lp85, r11l, sc25, sp80, su15), 1 run per config, 7-h budget per game.
  - "weighted" (tool-call tokens 1.0, all reasoning tokens 0.3) avg 25.07 vs "labeled" (sentence-level weights 1.0/0.3/0.1 by topic; ~92% of reasoning tokens ended at 0.1) avg 4.99.
  - Turns with no tool call: weighted 132/3,668, labeled 1,474/4,767; labeled lp85 stuck 1,231 turns in a row after one cut-off turn.
  - Takeaway quoted: "If you mask or down-weight reasoning selectively, check the no-action rate, not just the score." Also: harness should tell the model its previous turn produced no action. Caveat: "offline ranking has not reliably predicted the leaderboard".
- Thread 732854 "What are your agents scoring on the 25 public games?", Reki, 2 months ago, 39 listed comments (72 per list view). [B/discussion/732854]
  - Offline-to-LB relations reported: Fususu 22.26 local -> 5.37 LB; 17.34 -> 6.91 (25 runs, 2 won, 2h18m); Nick Pellegrin: 7.6% public -> 3.35% LB, "doing better on the public set ... correlates to weaker private scores", and "you need to submit at least like 5 times to get a good feel for how a notebook version scores".
  - Recent (14 days ago): Nick Pellegrin ~105 levels on public (~40% score) -> ~23% on private (one submission); Nick2187 best 56.76 offline, 124 levels, 11 games won (no LB score yet); Scott Le Grand 53.74 "and I haven't been able to repeat it"; Scott: "variance on submissions is insane".
  - CPMP (9th): "any number not from the competition leaderboard is irrelevant."
- Thread 745079 "Rules clarification: is Qwen3.8-Flash-Next (Qwen Community License 1.0) allowed in a prize-eligible solution?", Vitalii Berezniak, 16 days ago, 9 comments. [B/discussion/745079]
  - Staff (Maria Cruz, 2 days ago): "we cannot offer any guidance or provide interpretation on model licenses under the light of the competition rules." Anh-Vu Mai-Nguyen (48th) notes "All three Milestone-2 top notebooks serve Qwen3.8-Flash-Next". Unresolved.
- Thread 748591 "ARC Daily Digest" (Son Pham, 5th; site https://arc.markbarney.net/news): fan newsletter; says "Yi-Chia Chen overtakes Tufa Labs in the race". Not a method source. [B/discussion/748591]
- Other thread titles seen, not read (potentially relevant): 688114 "Tackling ARC-AGI 3: Leveraging Puzzle Games to Address Data Scarcity"; 743060 "P3.1: Perception-Augmented Duck - What Prompt Engineering Can and Cannot Buy on ARC-AGI-3"; 735662 "qwen3.8-27B model vs dataset"; 734843 "Potential persistent memory issue with the Tufa Duck harness". All via search at B/discussion?search=fine-tune and ?search=Qwen3.8-27B.
- I found no thread by Tufa Labs or Yi-Chia Chen describing a method or training in the discussion list searched (search terms tried: fine-tune, Qwen3.8-27B). Searches for LoRA/GRPO/distill/"60" were NOT run separately (UNVERIFIED completeness).

### Inferences
- Consensus among people with H200 access (blakewest, AAAAAtjc, Slow∞Loop, Scott Le Grand) is that Flash-Next fine-tuning is hard to make beneficial; no public positive result on Flash-Next. Our plan is therefore high-risk relative to the user's target; prior probability that a first LoRA beats base should be set low.
- The only reported positive effects (+0.7 LB points on a ~1-2 base; +2 public points) are on weak 27B/3.6 bases and tiny in absolute terms; they do not show that fine-tuning can move a 31-LB model to 60.
- The loss-masking failure shows a concrete risk for our plan: selective token weighting (e.g. R3 shortest-reply targets) can cause no-action loops. We must monitor no-tool-call rate and degenerate-reasoning rate, not just score.

### Gaps
- No thread explicitly links 60+ scores to training. Tufa Labs / Yi-Chia Chen / #3 methods are undisclosed on the Kaggle UI.
- Comment counts in the 732854 view are partial ("N more replies" collapsed).

## 2. Leaderboard (public, ~50% of test data), 2026-10-18
Source: B/leaderboard. Banner: "This leaderboard is calculated with approximately 50% of the test data. The final results will be based on the other 50%". Top 4,454 entries.

| # | Team | Score | Entries | Last |
|---|---|---|---|---|
| 1 | Tufa Labs (+2) | 68.91 | 166 | 1d |
| 2 | Yi-Chia Chen | 67.49 | 34 | 15h |
| 3 | mtg | 50.37 | 72 | 11h |
| 4 | the last dance | 49.85 | 85 | 10h |
| 5 | Son & Mark & Ronen | 48.92 | 81 | 1d |
| 6 | (Japanese team name, +4) | 45.11 | 88 | 15h |
| 7 | Grigoriy Zatravkin | 45.02 | 70 | 10h |
| 8 | Majkel1337 | 44.84 | 21 | 13h |
| 9 | NVARC3 (+2) | 43.94 | 41 | 13h |
| 10 | artificialagencylab.com | 41.74 | 56 | 13h |
| 11 | ARC Slayers | 41.19 | 121 | 19h |
| 12 | Vineeth Sai Varikuntla | 40.09 | 66 | 3h |
| 13 | AquaUrsa Research | 39.69 | 11 | 9h |
| 14 | dreach.ai | 39.11 | 40 | 11h |
| 15 | (Osaka Metropolitan Univ. student team) | 38.98 | 47 | 21h |
| 16 | Daniel Franzen | 38.85 | 98 | 10h |
| 18 | "speed, my mom is kinda homless" (our account, best 38.22; latest 29.59) | 38.22 | 33 | 1d |
| 24 | Lord Han Solo | 37.07 | 96 | 14h |

- Entries counts: Tufa Labs 166 entries is large relative to 1/day (a team of 3, plus the 1/day cap since ~March 25 gives ~207 max days), Yi-Chia Chen 34 entries.
- A gap of ~18 points separates #2 (67.49) and #3 (50.37); the field below is dense 36-50.
- No write-ups or notebooks are linked from the leaderboard page. The user's own page note: "Your latest submission scored 29.59. Your best score remains 38.22."

### Inferences
- A score of 60 would sit between #2 and #3: only two teams are above it. Tufa Labs and Yi-Chia Chen are the only teams who have broken out of the pack; whatever they do (UNVERIFIED: model, training) is not public.
- Daniel Franzen (author of our base stack) sits at 38.85, roughly our best draw (38.22): the Franzen-style stack plateaus near 38.

### Gaps
- Methods for ranks 1-3 are not available on Kaggle. Comparing "Tufa Labs" fine-tuning is impossible from Kaggle UI.

## 3. Public Kaggle Models derived from Qwen3.8-Flash-Next
Source: `kaggle models list --search "qwen3.8 flash next"` and `kaggle models get <ref>` run 2026-10-18 (CLI metadata; URLs https://www.kaggle.com/models/<ref>).

Fine-tuned / modified (not just quantized):
- huikang/qwen-38-flash-next-finetune (Tong Hui Kang), updated 2026-09-30, version 15. "LoRA checkpoint 09-29-21-15, merged into its pinned NVFP4 base"; base HF RadixArk/Qwen3.8-Flash-Next-NVFP4 rev 7b719225...; 83,987,919,497 bytes; PLE not included (assemble with `keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1`); instance license listed Apache 2.0 but base license file included; "See training_run.json for the actual training scope". No score claimed. Evidence that a LoRA-merged NVFP4 checkpoint can be served on Kaggle via a pre-merged upload plus an `assemble.py` that borrows PLE from the base.
- boristown/qwen3-8-flash-next-nvfp4-reap-448e (暗黑AGI): REAP-pruned 512 -> 448 routed experts "For pushing Kaggle serving context to 128K by freeing weight VRAM."
- natnitarach/tinfield-1-nvfp4-probe: conversion of badtheorylabs/Tinfield-1, "derived from Qwen/Qwen3.8-Flash-Next", NVFP4 core (84.06 GB) + separate INT8 PLE, needs "dedicated Tinfield vLLM 0.30 runtime"; "The archived B200 first-crack +10pp gate was a historical NO-GO result for the earlier export." (Tinfield-1 is an upstream derivative of Flash-Next, a third-party fine-tune/variant; its training is not described here: UNVERIFIED).
- phuongncn/arc3-qwen38-swift-w4a16-autoround and lordhansolo/swift-1-5-qwen3-8-flash-next-nvfp4: "Swift-1.5 Qwen3.8-Flash-Next" from HF ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound; sizes 124.5 GB + 52.8 GB split (part a/b). A named derivative "Swift-1.5" exists (nature of the tuning UNVERIFIED).
- davidaylward/whittle-next-27b-a3b: 27B-total/3B-active qwen4_exp MoE with hyper-connections and 2B-row n-gram memory, distilled from Qwen3.8-27B; Q4_K_M 16.8 GB; "research preview", not a finished model; GGUF for llama.cpp.
- manarmilad/qwen3.8-flash-next-uncensored; joesammy/orcarouter-...-uncensored-gguf (abliteration-type, irrelevant).

Quantizations / mirrors:
- dfranzen/intel-qwen3.8-flash-next-w4a16-autoround (our serving base), dfranzen/albucino-qwen3-8-flash-next-drafter (MTP drafter), dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple (W4A16 with FP8 PLE; id 772782, newest).
- nvidia/qwen3-8-flash-next-nvfp4 (official NVIDIA, NVIDIA Open Model License per thread 745079), keithtyser/... (RadixArk ModelOpt NVFP4), lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8 (mirror of HF primitive-ai/..., 180 GB), buianhtruc/strata-...-gsq-rco-iq3-s (GGUF IQ3_S, 83.6 GB, ISTA-DASLab), dineshkumar0705 FP8, ram2121 GPTQ 4bit, woochangsim and jamalsaeedi/xiaoz259 W4A16 AutoRound copies, superstringdev NVFP4 main/ple split.
- Only other fine-tuning-related ARC models found: mikedan7/qwen2-vl-2b-arc3-schema-qlora-bf16 (Qwen2-VL-2B QLoRA on "public schema-trace"; irrelevant size).

### Inferences
- Only one public Flash-Next LoRA-merged model (huikang, v15 means 15 iterations) is on Kaggle, with no claimed score. Tufa/Yi-Chia likely keep private datasets (UNVERIFIED, not on any public listing I could search).
- Re-uploading a fine-tuned model is straightforward: a merged-LoRA NVFP4 or W4A16 checkpoint plus PLE assembled in-notebook has precedent (huikang's `assemble.py` recipe). Kaggle model upload limit (50-file split) seen in phuongncn's two-part upload.
- REAP expert pruning (448 of 512) exists as a context-extension trick, relevant to our 55-78k prompts, but is a quality risk.

### Gaps
- Did not open each model page for download counts or comments; no claimed LB scores found.
- Did not check lordhansolo/richardcsaky/albucino pages beyond the CLI listing; richardcsaky models did not appear in the searches.

## 4. Rules (https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules, fetched 2026-10-18)
### Cited Findings
- Timeline (B/overview): start 2026-03-25; entry deadline 2026-10-26; final submission deadline 2026-11-02 (time of day not read; check overview).
- Team limit: "The maximum Team size is eight (8)." Submissions: "a maximum of 1 (1) Submissions per day"; "You may select up to two (2) Final Submissions for judging."
- Prizes: $850,000 total; final leaderboard $75,000 (1st $40k, 2nd $15k, 3rd $10k, 4th/5th $5k); milestone prizes $75,000 (M1 2026-06-30, M2 2026-09-30); Bonus $700,000 only if a team reaches 100%.
- Winner license: CC-BY 4.0; "Submissions are required to have open source system, open source model, and open source weights/parameters, as defined in the checklist from the Open Source AI definition by the Open Source Initiative." and "In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s)." Winners must deliver "training code, inference code, and a description of the required computational environment" (2.8).
- External data: "The use of external data and models is acceptable unless specifically prohibited by the Host" but must be "reasonably accessible to all" and of "minimal cost" (2.6.b); Reasonableness Standard applied by Host. So a fine-tune on rented Azure H200 compute is not prohibited per the text; a thread (742835) raised the fairness question without an official answer.
- Public sharing: private code sharing outside teams prohibited; code shared publicly deemed OSI-licensed (3.6).
- Data: Apache 2.0 license for competition data.
- Code requirements/hardware: the rules page has no hardware text; hardware/limits are in the pinned thread "Update on Code Requirements" (María Cruz, 1 month ago, 7 comments) and "Update on accelerators", which I did not open (UNVERIFIED here; the project brief states a single RTX PRO 6000 Blackwell, offline, ~9 h).
- Private leaderboard determines winners: "The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard" (3.7). The public LB uses ~50% of the test data (leaderboard banner).

### Inferences
- Fine-tuned weights must be uploaded as a Kaggle model/dataset and (for prize eligibility) be open-licensed; Qwen Community License eligibility is unresolved by the host. Not a blocker for score-only goals.
- Only two final selections out of ~1/day submissions and a ~50% public vs 50% private split: with same-notebook variance 27-38 we should pick the final two by repeated measurement, not best draw.

### Gaps
- Exact time of day of the deadline; pinned hardware thread content.

## Implications for our plan
1. Treat Flash-Next fine-tuning as unproven: four independent H200 users (threads 742835, 748593) report no gain. Plan a cheap first screen (LoRA SFT on our own log-derived data at tiny scale, 1 LB-style 25-game run) with an explicit stop rule, and keep an unmodified-notebook fallback for the final 2 selections.
2. Do guard against the documented failure modes: no-tool-call/degenerate-reasoning collapse under selective token weighting (743319): use uniform or near-uniform weights on reasoning plus full weight on stop/act boundary; report no-action rate per run; make sure the harness re-prompts with a distinct message after a no-action turn.
3. "Behavior-clone from the same model" (Scott Le Grand) agrees with our self-generated replay plan; the 739047 caveats (curation beats quantity; high variance; base-specific) support the R3/R4 selective-data approach and argue against bulk data scaling.
4. Variance: a +0.7 or +2 point gain is invisible in noise of 27-38 for the same notebook; decide with offline paired tests (make significance) and 3+ submissions, not one.
5. Serving path exists: merge LoRA into the NVFP4/W4A16 base and upload (huikang precedent, 84 GB, PLE assembled at run time); check our W4A16 re-quantization of a merged model is needed since the AutoRound base is not trainable directly. 2x96 GB QLoRA+FSDP2+PLE-CPU-offload is claimed feasible (748593, unverified).
6. Rules: fine-tuning with external compute is allowed by text; license eligibility for Qwen weights is unanswered by staff; there is no rule against it for score purposes. Keep our own code CC-BY compatible and avoid teacher outputs from OpenAI-type APIs (742835 warning).
7. Intel gap: ranks 1-2 (68.91, 67.49) are ~18 above the pack; their method is not on Kaggle. 60 is only exceeded by two teams, so the target is aggressive; the 38.85 of Franzen is near our ceiling.

## Fact-check (independent verifier, 2026-10-18)

Method: re-opened each Kaggle page in Chrome (discussions 742835, 748593, 743319, 739047, 745079; leaderboard; rules; overview) and ran `kaggle models get huikang/qwen-38-flash-next-finetune`.

1. H200 users report no gain from SFT/RL on Flash-Next, blame MoE: CONFIRMED (minor corrections). blakewest (rank shown 4426th, not 4424th) and Slow∞Loop ("MoE structure makes training unstable") blame MoE. AAAAAtjc said "I did not gain any performance" (RL pipeline, "yet"). Scott Le Grand said "haven't been able to get anything out of flash-next" but never said he used an H200. The "four independent H200 users" in Implications is overstated: only blakewest and AAAAAtjc state H200; Slow∞Loop used 2x96GB; Le Grand's hardware is unstated. Also, Son Pham's reply ("leads to some improvement at all") is loose praise, not a reported gain.
2. 2x96GB QLoRA 4-bit + FSDP2 + PLE CPU offload; full H200 node for RL: PARTIALLY CONFIRMED. Slow∞Loop wrote "I used QLoRA 4-bit + FSDP2 + PLE CPU Offload. It can help you finetune Qwen3.8-Flash-Next in 2*96G GPU." The thread does not say that run produced "no gain" explicitly for that recipe; he says only that he has "not been able to get anything out of flash-next." AAAAAtjc: "No, I used a full H200 node" and the RL pipeline was "based miles" (as written). Both quotes are real.
3. Down-weighting reasoning caused no-tool-call collapse (4.99 vs 25.07, 5 held-out games, 1 run/config): PARTIALLY CONFIRMED. Numbers, games and counts match (weighted 3,668 turns / 132 no-call; labeled 4,767 / 1,474). Corrections: (a) the post says "small adapter" and never names the base model, so it is not shown to be Flash-Next; (b) at the 1-hour budget "labeled" was slightly AHEAD (10.83 vs 8.88), so the 4.99 vs 25.07 gap is the 7-h result only, and the author says single-run noise dominates; (c) the "weighted" arm also down-weights all reasoning to 0.3, so the failure is attributed to sentence-level selective weights (92% of reasoning at 0.1), which is the author's unverified hypothesis. Score units are per-game scores, not LB.
4. Leaderboard 2026-10-18: CONFIRMED. Tufa Labs 68.91 (166 entries), Yi-Chia Chen 67.49, #3 mtg 50.37, Daniel Franzen 38.85 (#16), user at #18 with 38.22, banner "approximately 50% of the test data". Minor: the list ends "28 - 4456" so the notes' "Top 4,454 entries" is off by about 2.
5. Rules/timeline/license: CONFIRMED. 1 submission/day (2.2.a), 2 final selections (2.2.b), entry deadline Oct 26 and final submission Nov 2, 2026 (all deadlines 11:59 PM UTC; winners announced Dec 4, 2026), CC-BY 4.0, open-source system/model/weights requirement (2.5.a.a), carve-out for incompatible-license pretrained models (2.5, third paragraph; the notes' "2.5.a" is fine). María Cruz (Kaggle Staff, 2 days ago): "we cannot offer any guidance or provide interpretation on model licenses". The notes' "Gaps: time of day not read" can be filled: 11:59 PM UTC. Hardware/limits per overview: GPU notebook <= 9 hours, internet off, RTX 6000 (g4-standard-48).
6. Only quantified positive gain is LB 1.25 -> 1.94 (STaR LoRA, Qwen3.6-27B): PARTIALLY CONFIRMED. The reply is from manas joshi to Nick Pellegrin: "went from 1.25 to 1.94 fine tuning on qwen" (rank ~300 -> ~133). Caveats (high variance, scaling data hurt, base-specific) are in the reply to Ya Xu. But "only quantified positive" is too strong as a general statement: Scott Le Grand reports "a two-point jump on the public games" (Qwen3.6-27B, thread 742835) and "2 RHAE points on the public data" for 3.8-27B (thread 748593); these are public-game, not LB, and the notes themselves list them. The base is "qwen" in the quote; "Qwen3.6-27B" comes from the main post. Also note the thread says Ya Xu's LoRA "backfired".
7. huikang/qwen-38-flash-next-finetune: CONFIRMED. CLI shows version 15, totalUncompressedBytes 83987919497, updateTime 2026-09-30T06:27:45Z, overview "LoRA checkpoint 09-29-21-15, merged into its pinned NVFP4 base", voteCount 0, no score claimed; instance license field Apache 2.0 with base license file included.

Other errors noticed: (i) "Implications" item 1 says "four independent H200 users" (see claim 1). (ii) Thread 739047 date "~2 months ago" and replies "a month ago" are consistent with the page. (iii) Leaderboard row count "4,454" vs page "4456". (iv) Section 1 says Ya Xu's Chollet quote is second-hand; the page text agrees ("I also read later"). No other clear factual errors found.
