# Top-team intel, account rules, hidden-set composition (collected 2026-10-18)

Method: Chrome (new tab, closed at end) on Kaggle pages; WebSearch/WebFetch for external sources. X/Twitter is behind a logged-out wall (profile showed only one retweet). No Tufa/Chen LinkedIn/podcast/GitHub checks beyond what L2 already holds. B = https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3

## G18: Hidden-set composition (resolves the 110-vs-55 question)

### Takeaway
Kaggle's Data page says the evaluation set is 110 private games, half for the public LB and half for the private LB. So the hidden set is 110 games; each LB sees about 55. The ARC technical report's "55 fully private" is consistent with this (the private LB has 55).

### Cited Findings
- "Note: Competition evaluation uses a separate, private set of 110 games that your agent has never seen. Half of these are used for the Public Leaderboard score, and the other half for the Private Leaderboard score." [B/data]
- Scoring: "Per-level score = min(human_actions / agent_actions, 1.0), then squared"; "Per-game score = Weighted average of level scores (weighted by level index, 1-indexed)"; "Total score = Average of all individual game scores." [B/data]
- 25 public games in environment_files; up to 7 actions; grid max 64x64, values 0-15. [B/data]
- Tufa M1 write-up: the Kaggle setup gives "a RTX Pro 6000 GPU with 96 GB of VRAM and 9 hours of runtime to play 110 games in any order." [B/discussion/717133]
- Rules 3.18(f): Public LB is "a representative sample of the test data"; 3.7: winners "determined solely by the leaderboard ranking on the Private Leaderboard". [B/rules]

### Inferences
- With ~55 games per LB, per-draw sd ~3.5 is plausible, and public/private are independent halves: a public 60 mean implies ~60 expected private but with the same noise. Final-two selection should rely on repeat means.

## G17: Account / team / sharing rules (exact quotes, B/rules)

### Cited Findings
- Header: "You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts."
- 3.5.a: "You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy."
- 3.5.b: "you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account."
- 2.1: "The maximum Team size is eight (8)." Team merger: combined submission count must be <= (submissions per day x days run). 2.2: "a maximum of 1 (1) Submissions per day"; "up to two (2) Final Submissions".
- 3.5.d: "No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums." 3.6.a repeats it and extends to "sharing Competition Code between separate Teams, unless a Team merger occurs." 3.6.b: public sharing deemed licensed under an OSI-approved license.
- 2.6.a/b External data and models: allowed if "publicly available and equally accessible ... at no cost" or satisfying the Reasonableness criteria; "reasonably accessible to all" and of "minimal cost"; "a small subscription charge ... acceptable"; "Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable."
- 2.5: winners must have "open source system, open source model, and open source weights/parameters" (OSI checklist), CC-BY 4.0; "In the event that input data or pretrained models with an incompatible license are used ... you do not need to grant an open source license ... for that data and/or model(s)."
- 2.8: winners deliver code "should include training code, inference code, and a description of the required computational environment."
- 2.7 eligibility: Competition Entity employees may enter but are not eligible for prizes. No compute/notebook-limit text in the rules page (limits are in overview/pinned threads: 9 h, RTX PRO 6000; per M1 write-up).
- A fine-tuned Flash-Next trained on rented compute is not excluded by the text (2.6 cost-reasonableness is the only hook, a Host judgement); not answered by staff (W5b thread 745079).
- Community question (thread 742801, Tong Hui Kang, 25d ago): fine-tuned models must be published before the milestone deadline as part of the notebook's dependencies; he asked whether the dataset and fine-tuning script must be published too; Van-Phuc Huynh asked whether traces from a proprietary model (Astra) are acceptable. No staff answer visible in the 27 comments. [B/discussion/742801]

### Inferences
- Multiple accounts are a disqualification risk; our two-account setup (mellamoa, snowiers) must stay out of competition submissions. Only one account may submit unless they are merged into one team (max 8). Compute use of the second account for non-submission work is not addressed by the rules (UNVERIFIED reading); do not submit from both.
- Private sharing of weights/code with another team is prohibited, public forum sharing is fine.

## G9: Top-team intel

### Takeaway
No new primary statement that Tufa or Yi-Chia Chen fine-tunes. Tufa publicly said it will release its final solution after the competition ends, skipped the M2 prize, and a Kaggle top-9 forum member guessed (not knew) that fine-tuning is part of their edge. Compute: forum consensus is that Tufa's edge is GPUs plus talent.

### Cited Findings
- Tufa "We will not be open-sourcing for the second milestone prize", Jeroen Cottaar (rank 1), 25 days ago, 41 votes: "We, Tufa Labs, will not be sharing our solution on September 30 ... since we worry it may not be possible to beat the best of 4000 copies of that solution. We still plan to release our final solution when the competition ends." Also: "the winner simply being the luckiest ... your score would need to beat the best of 4000, and variance is high." [B/discussion/742801]
- Same thread, Scott Le Grand (523rd): "You have a commanding lead. My guess is part of that is some fine tuning, but I just got to #38 on harness fixes alone (5.19)" (speculation, 25d ago). Penguin: "the major advantage is of gpus, they have 10s of TB of gpu vram ..." (speculation). Lord Han Solo: "It looks like Tufa Labs just found a big improvement." Commenters at that time cite jumps: 27.29, then 45.33, "20+ to 35+ to 45+" (about 19 days before 2026-10-18, i.e. around Sep 29-30). [B/discussion/742801]
- CPMP (9th, NVARC3): "We will open source our code at the end of competition if we end up in gold. As usual for NVIDIA teams." [same thread]
- Leaderboard timeline from a competitor-run digest (arc.markbarney.net, headlines only, scores not given): Oct 12 evening "Chen continues to extend her lead over Tufa"; Oct 13 morning "Tufa retakes the ARC-AGI-3 lead as NVARC3 climbs to fourth"; Oct 13 evening "Tufa takes ARC-3 into the evening lead"; Oct 14 evening "Tufa holds the lead". [https://arc.markbarney.net/news/competitors/arc-3-15486995] This matches L2's board scrape (Tufa 56.52 -> 68.91 on 2026-10-13; Chen 62.96 on 10-10, 67.49 on 10-12). Chen is "her". The same page paraphrases an X post of 2026-10-09 ("temporary setback", "overwhelming resources and compute"); wording NOT verifiable (X login wall). UNVERIFIED.
- Metir blog (secondary): Tufa 55.89 and Chen 48.59 in an early-October snapshot; ARC Prize posted on X that Chen had earlier taken first at 28.34 (unverified by me). [https://www.metirai.com/blog/arc-agi-3-kaggle-scores-surge-harnesses-arc-prize-2026]
- ARC Prize X post 2026-09-21 (seen on https://x.com/tufalabs, retweet): standings for the M2 open-source prize: "1. Tufa Labs 2. Lord Han Solo 3. NVARC3". [https://x.com/arcprize]
- Tufa M1 write-up (Duck): authors Bessis, Cottaar, Pressman, Smit, Tesnar, Viel; Qwen 3.6 27B FP8; "we expect this approach to scale with better small models and additional post-training"; "hand-crafting specific tools for the model did not help". Context cap 64k, evicts to hold ~32k. [B/discussion/717133]
- Sergei Makarov (588th), "Harness or model?", 22d ago: nearly every harness layer (consensus, reasoning-discipline block, higher-res images, executable world model with search planning) scored at or below baseline; the stuck-vs-success analysis: "in 83% of the forks the stuck run had already tried the action type that later turned out to be decisive. It had the evidence; it drew the wrong conclusion about the goal". Conclusion: "the remaining lever appears to be the model itself, or training that improves this judgement without memorizing particular games." Same-build full runs vary "roughly 4.5 to 9.4 on the public games". [B/discussion/743723]
- Scott Le Grand same thread: "my path in the remaining month is finding a way to fine tune"; "My heuristic has been the best of 3 runs"; harness at 43 at the time. [B/discussion/743723]
- New thread titles from search "training" (not opened): 694549 "Generating training data ideas?" (Froggy McFrogson, 23 comments); "Retro Game Training Set"; "non-official games for training" (robenten); "Comparing the Milestone 2 winning solutions" (Tong Hui Kang, 64 votes); "Incredible notebook performance variation (Daniel Franzen's copies data analysis)" (Son Pham, 15 votes). Search for "GRPO" returned only the 6-month-old "Tackling ARC-AGI 3: Leveraging Puzzle Games to Address Data Scarcity"; "LoRA" returned only the STaR thread 739047 (already in W5b). Searches "distill", "60", "RL", "fine-tune" were NOT run separately in this pass beyond "training" and "Tufa"; Kaggle search is fuzzy.
- Web search found no October-2026 public statement by Dries Smit or Yi-Chia Chen about the model/training; the Tufa website/X pages show nothing newer than L2 holds. X is not readable logged out. LinkedIn, podcasts (beyond MLST July), and GitHub for Sept-Oct were not located.

### Inferences
- Tufa's own words (commit to releasing "when the competition ends") mean the best intel source arrives after 2026-11-02, too late for us; only external signals before then are leaderboard jumps, all consistent with either harness iteration or training.
- Two independent forum voices (Makarov, Le Grand) conclude the harness is saturated and that the early-goal-judgement error is a model issue, supporting a training prong; neither has a positive training result on Flash-Next.
- The 3.5 / 2.5 rules mean our final model must be publicly releasable only if we want prize eligibility; not needed for score.

### Gaps
- Tufa/Chen model and training: still undisclosed. X posts unreadable. Leaderboard dates of jumps exist only via L2's board scrape and the headlines above.
- Staff answers on fine-tune data/script disclosure and proprietary-teacher traces: none visible.

## Implications for our plan
1. Hidden set is 110 games, 55 per LB; plan validation around ~55-game noise (sd ~3.5), and submit repeats, not best draws.
2. Account rule: single-account submissions only (or one merged team of <=8); the second Kaggle account must not submit to the competition. Do not share weights/code privately with others.
3. Fine-tuned weights are permitted by text; only Reasonableness (cost) could be raised by the Host; avoid proprietary-API teacher traces (open question raised in 742801 and by W5b).
4. Do not expect Tufa's method before Nov 2. Makarov's finding (83% of forks had evidence but wrong goal judgement) is a concrete training target: goal-hypothesis commit/revise behaviour, supervised via paired success-vs-stuck decisions from our 1,642 game-runs.
5. Scott Le Grand is also pursuing fine-tuning; no positive result yet. Keep an unmodified-notebook fallback for the two final selections.

## Fact-check (independent verifier, 2026-10-18)

Method: Chrome, new tab (closed), read pages 742801, 743723, 717133, /data and /rules in full. Skipped: result 9 (arc.markbarney.net page; X posts behind login wall) and the X/arcprize part of result 10; those stay UNVERIFIED by me.

1. 110 private games, half public LB / half private LB: CONFIRMED. Exact quote on /data matches. Note: /rules 3.18(f) still calls the Public LB "a representative sample of the test data", which is consistent (a half). Scoring formulas and "25 public games" also match.
2. Account rules: CONFIRMED. 3.5.a quote exact; 3.5.b one team only; 2.1 max team size 8; 2.2 1 submission/day and 2 final submissions; 3.5.d and 3.6.a private sharing banned, public forum sharing allowed (3.6.b). The rules page also opens with "you cannot enter or submit from multiple accounts."
3. External data / fine-tuning: CONFIRMED. 2.6.a-b wording ("reasonably accessible to all", "minimal cost") and 2.5 incompatible-licence carve-out are as stated. Nothing in the rules text bars fine-tuning on rented compute. Caveat: 2.6.b says the Host assesses cost reasonableness case by case, and 2.8 requires training code from winners.
4. Tufa will not open-source for M2: CONFIRMED. Post by Jeroen Cottaar (1st); quote "We still plan to release our final solution when the competition ends" exact; "best of 4000 copies" exact. The page shows "16 days to go", so the end date is about 2026-11-03; "2026-11-02" is approximately right but I did not check the Timeline page.
5. Scott Le Grand "My guess is part of that is some fine tuning": CONFIRMED (523rd, speculation). GPU attribution is by Penguin (422nd), also speculation; chiefkeith (11th) says compute is only a small part.
6. Makarov (588th): CONFIRMED. "Almost every added layer scored at or below the baseline"; 83% of forks (this number is in his follow-up comment, not the main post); "remaining lever appears to be the model itself, or training"; spread "roughly 4.5 to 9.4" is also in the follow-up comment.
7. Le Grand "path in the remaining month is finding a way to fine tune" and best-of-3 heuristic: CONFIRMED. CORRECTION to the note body: "harness at 43 at the time" is his leaderboard POSITION ("current position of 43"), not a score of 43. Do not read it as a score.
8. Tufa M1 write-up (717133): CONFIRMED. "RTX Pro 6000 GPU with 96 GB of VRAM and 9 hours of runtime to play 110 games in any order"; "we expect this approach to scale with better small models and additional post-training" exact. Further detail: the M1 harness base model is Qwen 3.6 27B FP8 and the post's scores were 1.21% LB.
9. markbarney timeline: NOT CHECKED (skipped).
10. CPMP (9th): CONFIRMED ("We will open source our code at the end of competition if we end up in gold."). Lord Han Solo (24th) said he will share if he makes top 3 among willing sharers (not in the note, relevant to open-source intel). The 2026-09-21 ARC Prize X standings: NOT CHECKED.
11. No staff answer on fine-tune data/script disclosure or proprietary-teacher traces: PARTIALLY CONFIRMED. The questions are by Tong Hui Kang and Van-Phuc Huynh as stated, and no visible comment is from a host or staff. But the page collapses "8 more replies" under one thread (the xz / Lord Han Solo exchange) that I did not expand, and 3 comments are shown as deleted; so "27 visible comments, no staff reply" slightly overstates what was inspected. Treat as "no staff reply seen in the expanded view".

Other corrections: none to numbers. The only wording fix is item 7 (43 is a rank).
