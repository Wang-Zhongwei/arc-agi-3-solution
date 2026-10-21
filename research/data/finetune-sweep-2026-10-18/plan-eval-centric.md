# Plan (measurement-centric lens): judge checkpoints first, train only what can be judged in time (2026-10-18)

Planner 3 of 4. Lens: **the bottleneck is knowing whether a checkpoint is better.** The plan builds an evaluation machine first, then admits only the training work that machine can give a verdict on before 2026-11-02 23:59 UTC.

Numbers come from the sweep notes, with each note's fact-check applied. Short names: EB = `_evidence-base.md`, G-B = `G-B-eval-design.md`, and so on. New facts read today: mellamoa has **40.90 GPU-h** left until 2026-10-24 00:00 UTC (`kaggle quota`, 2026-10-18). GCP `g4-standard-48`, the scoring box's own machine type (W5b FC 5), lists at **about $4.50/h on-demand in us-central1** ([cloudprice.net](https://cloudprice.net/gcp/compute/instances/g4-standard-48)) and **about $1.63/h spot** ([computeprices.com](https://computeprices.com/providers/google/gpus/rtx-pro-6000)). Both price sources are secondary, so treat them as UNVERIFIED.

## Bottom line

1. **The leaderboard cannot confirm any realistic fine-tuning gain by 11-02.** With 16 slots and per-draw sd 3.50, confirming +5 LB takes about 6 draws per arm (G-B Q4, FC 4). Realistic gains are smaller: R1 (MTP head) +1.1 to +1.9 LB (G-E Q4), R3 (shorter thinking) +2.0 to +6.5 (G-H Q4). So decide offline, and use the LB as a non-inferiority and crash check on hidden games.
2. **Offline evaluation can be made powerful cheaply.** A rented fleet of exact-stack RTX PRO 6000 boxes costs about $10–13 per full run on-demand, $2.5–4 on spot (G-B Q7; W6 §7). At about 15 runs per arm it resolves +3 offline on unseen games; Kaggle alone gives about 2 runs a day. All of H1 on the fleet is about $0.5–0.8k, against $84.80 per H200 node-hour.
3. **Lossless changes are judgeable by 11-02; decision-changing ones mostly are not.**
   - R1 is lossless under target-only verification (G-E FC 5). Its accept length and decode speed are measured to about 1% (G-E Q5), and the measured compute curve prices them (G-H Q1).
   - R3 changes the token distribution. Its closest analogue, accept05, fell 8.6 points below the curve (G-H Q1). It needs U25 replicates and LB draws, with a hard entry deadline of 10-26.
4. **Measure two things before training anything:** upward elasticity (the leader at `demo_minutes` 242, 4.3 Kaggle GPU-h, today) and P1 vs P3 (one LB draw at a halved private deadline, tomorrow; G-H Q5). Together they set R3's value and whether the node money is worth spending before 11-02.
5. **Odds.** LB mean at 11-02: 29.5 / **32** / 38. P(any trained weights beat the harness fallback) is **about 0.30**, almost all R1; a trunk LoRA is about 0.07. A consistent 60 (mean about 64.5, EB 1.8) is under 1% by 11-02 and is an H2 goal.

## 0. Where we start

| Item | Value | Source |
|---|---|---|
| Fallback F = `mem-repair-10x` v1 | LB 38.22 / 28.73 / 29.54 / 26.90, mean 30.85; offline 54.69, 10 wins. `mem-unified-10x` mean 30.91 (2 draws) | `runs/submissions.csv`; EB 1.7 |
| 10x family | 19 draws, mean 29.12; per-draw sd 3.50 (df 7) | L3 FC 7 |
| LB / offline | 0.548 (sd 0.051); within-family r = 0.44 | L3 FC 3 |
| Offline run sd | 3.15 within replicates, 3.28 across configs; 10-game subset 5.8 | G-B Q1 |
| Kaggle quota | mellamoa only: 40.90 h to 10-24, then 45 h/week, user cap 6 h/day; a full run is about 2.3 GPU-h | `kaggle quota`; memory `kaggle-gpu-budget` |
| LB slots | 16 (10-18 to 11-02, 1/day), mellamoa only | G-B Q4 |
| Screens | single-run reliability within the family 0.08; minute-25 panel r = 0.21 with final rank | G-B Q5; L3 Q4 |

**Account rule (needs the user's decision today).** Rule 3.5.a: "You will be disqualified if you make Submissions through more than one Kaggle account" (G-F G17, FC 2). snowiers has already made 9 LB submissions (`runs/submissions.csv`) and shows as a separate entry (G-B FC 11).
- This plan makes **no snowiers submissions** and budgets **no snowiers quota**. The rules don't address compute-only use, and the site header says "you cannot sign up to Kaggle from multiple accounts" (G-F).
- The rented fleet replaces snowiers' 5 h/day. Whether to contact Kaggle about the past submissions is the user's call.

## 1. The evaluation machine

### 1.1 Instruments, from cheapest to most expensive

Each instrument has a job, a measured or to-be-measured noise level, and a place in the gate order. Screens are **regression gates, not predictors**, because no log metric has predicted score (G-B Q5).

| # | Instrument | Measures | Noise / power | Cost | Status |
|---|---|---|---|---|---|
| I0 | Integrity + serving parity | SHA of drafter files; tensor byte-identity of the overlay; T=0 spec-on vs spec-off identity; boot | exact | about 0.3 h | overlay tool to build (G-D §3) |
| I1 | G-MTP throughput panel (25 min) | time-weighted accept length, clean bs10 tok/s, step time | tok/s SE about 1%; accept day-to-day about ±0.01 (2.631 vs 2.643) | 0.7 h | design exists (G-E Q5) |
| I2 | Canary | no-tool-call rate on 300 logged states (fail at >8 of 300; FC correction), sandbox/syntax errors, GPQA-100, IFEval-150 | thresholds from a base-vs-base repeat | about 50 min | to build (G-B Q6) |
| I3 | Frozen-state screen (state-lab) | paired, about 450 states: tokens and calls per decision, right-move rate, distinct actions across k, verification markers per 1k | −25% tokens needs 420–470 samples per arm unpaired (L3 FC); paired needs fewer | about 1 GPU-h per pair | state-lab at tag `archive/tooling/state-lab`; exact rebuilds (EB 1.1) |
| I4 | Branch bench | clear rate from frozen level starts at H10 frontier levels (cn04 L2, re86 L6, sc25 L6, vc33 L7) | p 0.5 → 0.7 needs 73 branches per arm | about 5 GPU-h per arm | not built (`arms.md:575`; L3 Q4) |
| I5 | Curve residual | full-run score minus the base curve S(121·m_eff) at the run's measured tokens per action | uniform slowdowns land at −0.01 ± 1.68 | free with I6 | method in G-H Q1, Q5 |
| I6 | U25 full run, 121 min, 25 games | H10 + DT15 (all unseen): **wins and time-to-win first**, levels at 20/60/121 min, H10 RHAE | sd unknown (about 3.3–4.5 expected); measure with ≥3 base runs | 2.3 h; $10–13 on-demand | needs the `--environments-dir` builder path (exists in `run.py:1213`) |
| I7 | DT10 fresh panel | single-use final confirmation on 10 never-viewed DreamTeam games | one look only | 2.3 h per run | G-B Q3 |
| I8 | LB draw | hidden 55 games, the 9-h private path | sd 3.50; +5 needs about 6 draws | free; 1 slot | 16 slots |

Rules for every instrument:
- Each fleet batch includes a same-day base run. Never pool raw runs across platforms; fit a platform term (G-B Q7).
- Score comparisons use `make significance BASELINE_SCORE=... CANDIDATE_SCORE=...`, plus the run-level test, because the per-game test is conservative under shared compute (L3 Q4).
- Never score H10 in a run that contains trained games: freed compute adds up to +3–8 with zero transfer (G-B FC 5).

### 1.2 The exact-stack fleet

| Option | Box | Price | Fit | Source |
|---|---|---|---|---|
| **GCP `g4-standard-48` (first choice)** | the Kaggle scoring machine type: 1 RTX PRO 6000, 48 vCPU, 180 GB RAM | about $4.50/h on-demand; spot about $1.63 (secondary sources) | same type as the scoring box; RAM covers the 95.4 GiB PLE table | W5b FC 5; links above |
| Azure `NC144lds_xl_RTXPRO6000BSE_v6` (second) | 1 RTX PRO 6000 BSE, 144 vCPU, 264 GiB | $5.50 on-demand; $1.02 spot (API-confirmed) | same Azure account the user is arranging | W6 §7, FC 7 |
| Kaggle (always on) | the real stack | free, ≤6 h/day | ground truth for each final notebook | memory |

**Software.**
- Run the public Kaggle GPU Docker image with the `dfranzen/pennyroyal-v253` wheelhouse and the notebook's code headless.
- Pull the weights once per box (about 15 minutes for 119 GB, G-B Q7).
- Turn on raw-reply, token-id and logprob logging in every fleet run. These are default-off flags that change no requests (EB 4.2).

**Calibration gate, by 10-21 23:59 UTC.**
- Full-batch decode must be within 5% of Kaggle's 895–909 tok/s (G-B Q7).
- Two public-25 runs of the leader config must each score ≥45.0, with a mean ≥48.3 (REF 51.57 − 1 sd, G-H Q2), and their minute curves must overlap the REF band.
- If this fails, run the fleet only for state-lab sampling and score on Kaggle.

**Sizing.** Peak demand is about 4 boxes for one day: one U25 batch of 8 candidate runs plus 2 base runs. Request quota for 4 concurrent GPUs on GCP and Azure today, and use whichever arrives first. Use spot only for restartable work (state-lab, capture), because a pre-empted 2-h run is lost.

### 1.3 Evaluation sets (pre-registered)

- **H10 (held out):** lp85, tr87, cn04, sc25, vc33, dc22, ka59, lf52, re86, wa30 (G-B Q1).
  - Nothing trains on these. Exclude arc-interactive's vc33 copy from any training pool (G-B Q2).
- **DreamTeam split.**
  - On 10-19, one base Kaggle run on all 25 DreamTeam games (2.3 GPU-h; this closes `_critic.md` G7).
  - **DT15** = the 15 games with at least one level cleared and no win before minute 60, keeping all 3 mixed-mode games. Ties go to more partial levels.
  - **DT10** = the other 10, opened once on 10-30.
  - If fewer than 15 qualify, fill DT15 with arc-interactive games of 6+ levels and grid ≥16 (bn04, ll01, mm01, mm04, mm05, pu04, sf04; exclude wm04, which references ls20). Nothing trains on external games in H1, so all of them are eligible for evaluation.
  - DreamTeam baselines are solver lengths, so report **levels**, not RHAE, on DT games (G-B Q3).
- **U25** = H10 + DT15 in the standard 121-min layout. It starts with 3 base runs and grows by one same-day base control per batch, to about 6 by 10-25.
- **Controls for validating screens** (§1.4): the base against itself (noise floor), and the base under relaxed acceptance 0.5/0.5 (`accept05`). accept05 changes decisions and is known to cost −6 offline, and −8.6 against its own compute curve (G-H Q1, Q3).

### 1.4 Validate each screen before it gates anything

No screen has ever been back-tested (G-B Q5). Before I3 or I4 gates a trunk checkpoint, it must:
- show base vs base within its own CI, which sets every threshold;
- **flag accept05 as worse**, either right-move rate down ≥2 pp or I4 clear rate down ≥0.1.

A screen that can't see a known −6-point change isn't allowed to approve R3. Cost: about 2 GPU-h (I3) on Kaggle 10-21, plus about 10 machine-h (I4) on the fleet if the branch bench gets built.

### 1.5 What the machine can detect, and by when

Power uses one-sided α 0.05 and 80% power, with n = 12.37·σ²/δ² for two arms (G-B Q1 method). U25 σ is assumed to be 3.5 until it is measured.

| Effect to detect | Instrument | n | Machine time | Wall time on 4 boxes | Earliest verdict for a 10-24 artifact |
|---|---|---|---|---|---|
| accept +0.10 / decode +2% | I1 | 1 panel vs a same-day base | 1.4 h | 1 h | 10-24 |
| −15% tokens per decision at equal right-move rate | I3 | about 450 paired states | 1 GPU-h | 1 h | 10-24 |
| +5 offline on U25 | I6 | 6 per arm (σ 3.5); 10 at σ 4.5 | 28–46 h | 7–12 h | 10-25 |
| +3 offline on U25 | I6 | 17 per arm; or about 9 against a 6-run pool | 40–80 h | 10–20 h | 10-26 |
| +5 LB | I8 | about 6 draws vs F's 8 | 6 slots | 6 days | **not reachable for a 10-24 artifact** |
| +8 LB | I8 | 1.5–2.4 draws | 2–3 slots | 3 days | 10-27 |

**What follows.**
- By 11-02, offline evaluation can resolve effects of about +3. The LB can resolve only effects of +8 or more, which no lever we have can produce.
- The LB's job is therefore to catch breakage on hidden games, settle P1 vs P3, and give the final picks a few draws each.

### 1.6 LB calendar (mellamoa only; submit about 00:10 UTC so the ~9-h scoring finishes the same day)

| Date | Draw | Why |
|---|---|---|
| 10-18 | F (anchor #5) | comparator mean |
| 10-19 | **F-half:** F with the private deadline halved (266 of 532 min) | P1 predicts about 19.5 (−11.4); P3 predicts 12–13 (−17.6 to −18.8) (G-H Q5, FC 12). ≥17 → P1-leaning; ≤15 → P3-leaning |
| 10-20 | F (anchor #6) | |
| 10-21 | **Z:** F through the zero-delta overlay path | checks the 9-h deploy path; pooled with F only if tensors are byte-identical (I0) |
| 10-22 | flex: a second F-half draw if 10-19 landed in 15–17; otherwise F anchor #7 | |
| 10-23 | F (anchor #8) | F's SE becomes 3.5/√8 ≈ 1.24 |
| 10-24 → 10-26 | **C1** = F + R1 head, 3 draws | only after its gates (§3, P2) |
| 10-27 | reserve: rerun an errored draw, or C1 #4 | |
| 10-28 → 10-30 | **C2** = C1 + R3 LoRA, 3 draws (Branch A only); otherwise C1 or F | |
| 10-31, 11-01 | top-ups for whichever config sits near a pick threshold | |
| 11-02 | a final copy of the leading config at 00:10 UTC; finals selected from completed draws by 23:00 | |

Calendar rules:
- Never leave a slot unused (`arms.md:569`).
- Never pick finals by public score: the split-half r is −0.45 (G-B Q4).

**Pick rule.** It assumes the private score is the best of the 2 picks, which is UNVERIFIED for this competition; a Sonnet browser agent should read the evaluation page on 10-18.
- **C1** is F plus a lossless drafter. Its prior is mechanism-based: the LB gain predicted from the harness tok/s measured offline, 0.548 × 28·ln(m_eff) (P1 floor, G-H Q4), with sd 1.0.
  - Once C1 has completed ≥3 draws with no error, and its offline Kaggle run shows accept ≥2.63 + the measured gain, C1 replaces F as pick 1. With a lossless change, the draws are an error check, not the evidence.
- **C2** gets G-B's skeptical prior N(0, 3²), a shrink of about 0.6.
  - Pick 2 = C2 if its posterior Δ against pick 1 is ≥ −0.29 (the FC-corrected band, G-B FC 10) and every C2 draw completed. Otherwise pick 2 = F.
  - Pairing two different policies keeps the hedge that best-of-two scoring rewards (+2.2 at Δ = 0 vs +1.97, G-B Q4).

## 2. The judgeability ledger: what to train

| Lever | Expected effect | Can I0–I8 judge it by 11-02? | Decision |
|---|---|---|---|
| **R1: self-distilled MTP dense head** (90.6M BF16 params; trunk frozen) | accept +0.15–0.25 → +2.0 offline, +1.1–1.9 LB; upside +0.36–0.41 (AngelSpec) → up to +3.9 offline, +2.2–3.9 LB; under 15 GPU-h (G-E Q4, FC 8, 12) | **Yes.** Lossless, judged by I0/I1 to about 1%; the score value comes from the measured curve (I5) | **Do. Primary weights lever.** Trains on one RTX PRO 6000 (<40 GB, G-E Q3), so it needs no node |
| **Swift-1.5 / Tinfield BF16 trunk overlays** | unknown. Swift: decode 4–8% faster, tokens per action −4% in a 10-game screen (G-D FC 7). Tinfield: tiny deltas, NO_GO probe (G-D FC 8) | Partly: I1, I3 and 2–4 U25 runs can see ±5 | **Do as free evidence** on whether post-training moves our metrics. No final pick unless U25 ≥ +5 |
| **R3: shorter thinking at the same decision** (LoRA on GDN/QSA/shared expert; per-call, served context) | −15 / −25 / −30% tokens per action → +3.6 / +6.3 / +7.8 offline; +2.0 / +3.5 / +4.3 LB (P1); +3.7 / +6.5 / +8.0 (P3) (G-H Q4). Medium risk: the Arora collapse; right replies are not shorter at hard states (G-G (a)) | Only if the artifact exists by 10-26: I3 → I2 → I1 → 8× I6 → 3× I8 | **Do in Branch A only**, behind the elasticity and yield gates (§3, P3) |
| R4: hindsight breakthroughs on external games | the capability lever LB 60 needs (EB 1.8) | **No.** No external-game data exists before 11-02; 25 families sit in the AWM degrade regime (G-G implication 7) | H2 |
| Harness fixes (#18, #176 sandbox: 2–4% of tokens; #175 hardening) | +0.5–1 offline at most (G-A Q4) | No at score level; yes as mechanism and crash checks | Fold #175 hardening into F only if a request-replay check shows identical requests; otherwise H2 |
| RL, SDPO/OPSD, world-model SFT, novelty rewards, re-quantisation | — | No | Not doing (§7) |

## 3. H1: phases, gates and kill dates (2026-10-18 → 11-02)

### 3.1 Phases

| Phase | Dates (UTC) | Work | Gate (measurable) | Kill (dated) |
|---|---|---|---|---|
| **P0: decide, launch the cheap measurements** | 10-18 | User approvals (§9). Push arm `elastic-242` (the leader at `demo_minutes` 242). LB: F #5. File GCP, Azure RTX PRO 6000 and H200 quota requests. Laptop: builder plumbing for `--environments-dir` + game manifests | The 242-min run's S(121) lies within REF 51.57 ± 3.26 (truncation validity, G-H Q5) | — |
| **P1: build and calibrate the machine** | 10-18 → 10-21 | DT25 base run (Kaggle); fleet bring-up + calibration; U25 base ×3; canary base-vs-base; state-lab restore + logprob smoke (G-C Q4); **R3 yield test** on about 200 on-route states at k=8 (`_critic.md` G4); screen validation vs accept05; overlay tool + zero-delta boot | Fleet calibrated (§1.2); U25 base sd ≤5.5; ≥10 DT games with partial clears; zero-delta overlay byte-identical, tok/s within 2%, T=0 identity ≥ base − 5 pp; I3 flags accept05 | **10-21 23:59:** fleet not calibrated → Branch C. **10-22:** zero-delta overlay fails → no trunk weights in finals |
| **P2: lossless weights (R1) + overlay evidence** | 10-20 → 10-26 | R1 capture hook (runtime patch dumping the HC stream, or the frozen-mixer 2,560-wide state, in a 4k window, FP8; G-E Q3). Check that albucino reproduces its served accept on captured rows. Train the dense-only head on **T15 transcripts only**, so H10 and DT stay unseen for the drafter. Export the 31 tensors byte-compatibly. Swift + Tinfield overlay panels | Offline E[L] on held-out features ≥ albucino + 0.10. I1: accept ≥ base + 0.08, clean decode ≥ 1.02× base, step time ±1 ms, T=0 identity ≥ base − 5 pp. ≥2 U25 runs error-free with residual ≥ −3. Kaggle stacked run (new notebook): wins and time-to-win reported, accept ≥ 2.63 + gain. Swift promote rule: tokens per action ≤ 0.85× and levels per action ≥ 0.95× (G-D §4) | **10-23:** capture doesn't reproduce albucino's accept within ±0.03 → stop R1. **10-25:** head not ≥ +0.10 held-out → stop. **10-26:** C1 not through I1 + U25 → its slots go to F |
| **P3: R3 trunk LoRA (Branch A only)** | 10-20 → 10-30 | Day-1 node smoke (`_critic.md` G5): pinned stack (ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1, transformers 5.19.0; W6 §1); one 64k multimodal LoRA step at CP2/CP4; a 100-step throughput run; packed vs unpacked parity. Siblings for about 2,000–3,000 T15 states at k=8 (state-lab, Kaggle + fleet spot). G-G recipe: shortest verified sibling capped at ≥0.7× the state's median; 30–50% median-length anchors; 0% generic replay; LoRA LR 1e-5 to 3e-5; one epoch; checkpoints at 25/50/75/100%; λ ∈ {0.25, 0.5, 0.75, 1}; whole-turn targets; never the first turn of a level or the turn after a death | **Entry:** S(151) − S(121) ≥ +2.5 **and** S(161) − S(121) ≥ +3.5 (G-H Q5); yield test: ≥50% of on-route states have ≥2 verified siblings with shortest ≤0.8× median; node smoke passes. **I3 (paired):** tokens per decision ≤0.85× base, right-move rate ≥ base − 2 pp, distinct actions ≥0.9× base, no-tool-call ≤ base + 0.5 pp, verification markers per 1k ≥0.9× base. **I2** passes. **I1** passes (or R1 retrained on the new trunk). **U25 ×8:** mean ≥ the base pool mean (non-inferiority) and residual ≥ −3. **Alpha claim** (user's rule): every run above the base mean, plus a paired p < 0.05 | **10-20:** elasticity falsified → no R3 before 11-02; node time goes to H2 prep. **10-23 23:59:** no usable node, or smoke fails → Branch B. **10-26 23:59:** no λ candidate through I3 → stop R3. **10-28:** U25 below the base pool or residual < −5 (the accept05 signature) → C2 dropped |
| **P4: confirm and pick** | 10-24 → 11-02 | LB calendar (§1.6); DT10 fresh panel on 10-30 (finalists ×2 runs each); optional 110-game, 532-min fleet rehearsal for C2 (about $45) | Each final notebook has one error-free offline Kaggle run and ≥3 error-free LB draws; DT10 not below F by >1 sd | Any C draw errors and is not fixed and re-drawn by 11-01 → F takes that pick |

### 3.2 Day by day

Kaggle GPU-h are budgeted against mellamoa's 40.9 h to 10-24 and 45 h for 10-24 → 10-30.

| Date | Kaggle (GPU-h) | Fleet (machine-h) | Node | Build / laptop | LB |
|---|---|---|---|---|---|
| 10-18 Sun | elastic-242 (4.3) | quota requests | confirm the date | ext-games builder; fleet bootstrap; read the evaluation page | F #5 |
| 10-19 Mon | DT25 base (2.3); state-lab + logprob smoke (0.7); R3 yield test (1.5) | bring-up, decode bench (4) | — | overlay tool; canary set; F-half arm | **F-half** |
| 10-20 Tue | canary base-vs-base (1.0); zero-delta panel (0.7); siblings (3) | calibration: 2 public-25 runs (5) | smoke, if here | select DT15/DT10 (pre-registered); R1 capture hook; **R3 go/no-go** | F #6 |
| 10-21 Wed | I3 validation vs accept05 (1); siblings (4) | U25 base ×3 (7); Swift/Tinfield panels (2) | smoke (G5) | R1 exporter; albucino self-test | **Z** |
| 10-22 Thu | siblings (4) | R1 capture on 4 runs (2) + training (6); siblings on spot (6) | data upload | R3 dataset build | flex |
| 10-23 Fri | C1: I1 panel (0.7) + stacked full run (2.3) | C1 U25 ×4 + base (12) | **kill date** | C1 Kaggle model upload | F #8 |
| 10-24 Sat (quota reset) | siblings / C1 follow-up (3) | siblings on spot (6) | R3 round-1 training (3–12 node-h) | — | C1 #1 |
| 10-25 Sun | — | λ-candidate I3 screens (4) | merges/export | overlays (~6 GB each) | C1 #2 |
| 10-26 Mon | C2 canary (1) | C2: I1 + U25 ×8 + 2 base (25) | R1 on the C2 trunk if accept drops >0.05 | **R3 entry deadline** | C1 #3 |
| 10-27 Tue | C2 stacked full run (2.3) | — | release node | — | reserve |
| 10-28 Wed | — | optional 110-game rehearsal (9.5) | — | — | C2 #1 |
| 10-29 Thu | — | — | — | — | C2 #2 |
| 10-30 Fri | — | DT10: finalists ×2 (9) | — | — | C2 #3 |
| 10-31 Sat | final-notebook confirmation if anything changed (2.3) | — | — | — | top-up |
| 11-01 Sun | — | — | — | — | top-up |
| 11-02 Mon | — | — | — | select finals by 23:00 | final copy |

**Branches.**
- **A (node by 10-23):** the full table.
- **B (no node):** drop P3. C2's slots go to C1 draws, then F. Fleet spend falls to about $0.4k.
  - Expected LB mean about 31.5–32.5.
  - Branch B still produces trained weights (R1), on one rented RTX PRO 6000.
- **C (no calibrated fleet by 10-21):** evaluate on Kaggle only, at about 2 full runs per day.
  - R1 is judged by I1 plus 2 stacked Kaggle runs.
  - R3 is not judgeable in time, so don't train it before 11-02 even if the node arrives.
- **D (no fleet, no node):** R1 trains inside a Kaggle session only if disk and RAM allow (UNVERIFIED, G-E Q3 gaps). Otherwise both finals are harness-only: F, plus a second F copy or the hardening variant.

**Artifact-entry deadlines** (the latest date an artifact can enter the machine and still be judged):
- a lossless head: I1 by **10-28**;
- a decision-changing trunk checkpoint: I3 by **10-26**, so training must finish by 10-25 and the data by 10-23.

### 3.3 Project rules applied

- **Branches**, each change behind a default-off flag: `arm/ext-games`, `arm/eval-fleet`, `arm/raw-logging`, `arm/weights-overlay` (`WEIGHTS_OVERLAY_DIR`), `arm/r1-mtp-head` (capture patch, exporter, `DRAFT_MODEL_DIR`), `arm/r3-lora`. `elastic-242` and `f-half` are `arms.json` entries. Write-ups and run logs go to `main`.
- **Each PR names its public-game-calibrated thresholds:** REF 51.57, accept 2.63/2.643, 0.85× / 0.95×, and the 2-pp right-move rate.
- **No exact repeats on Kaggle.** Fleet replicates (calibration, base pool, candidate n) are measurement and need the user's exception (§9). If it is declined, every fleet run carries a new default-off flag and leaders are judged across the runs that carry them (memory `no-exact-repeat-runs`), at a cost in power.
- **Game- and time-agnostic code.** F-half and elastic-242 change only the budget. **Laptop load:** `nice -n 15`, at most 3 heavy jobs.

## 4. H1 cost

| Item | Where | GPU-h | $ | Approval |
|---|---|---|---|---|
| Elasticity, DT25, smokes, yield test, canary, siblings, I3 validation, stacked confirmation runs | Kaggle, mellamoa | about 70 (cap 6/day) | 0 | each push |
| Bring-up + calibration | fleet | 9 | 40–50 | fleet budget |
| U25 base pool (about 6) | fleet | 14 | 65–80 | " |
| R1 capture + training + C1 evaluation | fleet | 20 | 75–110 | " |
| Swift/Tinfield overlays | fleet | 5 | 25 | " |
| Siblings for R3 (spot) | fleet | 18 | 20–30 | " (Branch A) |
| C2: λ screens, I1, U25 ×8 + controls | fleet | 29 | 130–160 | " (Branch A) |
| Branch bench (I4), optional | fleet | 10 | 45–55 | " |
| DT10 + optional 110-game rehearsal | fleet | 19 | 85–105 | " |
| Storage, egress | — | — | 30–60 | " |
| **Fleet subtotal** | | **about 125–135** | **about $0.5–0.8k** (cap $1k) | |
| H200 node: setup, smoke, R3 round 1, merges | Azure 8×H200 | 20–36 node-h = 160–288 GPU-h | $1.7–3.1k at $84.80/node-h (W6 §7); deallocate when idle | node budget (cap $3k) |
| **Total** | | **about 365–490 (Branch A); about 205 (Branch B)** | **$2.2–3.9k (A); $0.5–0.8k (B)** | |

R3 training is 2,000–3,000 examples at about 57k context, about 114–171M tokens, or 2.3–12 node-h at W6's unverified 2–7 node-h per 100M tokens (G-G (c)).

## 5. Odds at 11-02

**LB mean of the selected config (per draw):**

| Case | LB mean | Reasoning |
|---|---|---|
| Low | **29.5** | No weights ship. F's true mean regresses toward the family's 29.12 (its 30.85 is a 4-draw mean with SE 1.75) |
| Mid | **32** | R1 ships at +0.15–0.25 accept: +1.1–1.9 LB (G-E Q4), on F ≈ 30.5. R3 does not ship (node late, or failed I3/U25) |
| High | **38** | R1 at the AngelSpec upside (+2.2 to +3.9 LB) **and** R3 at −25% under P3 (+6.5) with no decision damage (G-H Q4). This needs every gate to pass |

Best-of-two picks add about +2 to the expected private score on top of the per-draw mean (σ/√π, G-B FC 10).

| Probability | Value | How it is built |
|---|---|---|
| LB mean ≥ 40 | about 5% | |
| LB mean ≥ 60 | under 1% | |
| Any trained weights truly beat F and ship | **0.30** | R1: capture hook works about 0.7 × head gains ≥0.1 about 0.65 × survives full runs about 0.85 × on time about 0.8 ≈ 0.31. Trunk R3: node by 10-23 about 0.5 × decision-safe gain about 0.25 × judged in time about 0.6 ≈ 0.07 |
| We can *prove* it at the user's alpha standard on the LB by 11-02 | under 5% | §1.5 |

## 6. H2: after 11-02, toward a consistent 60

The target is a draw mean of about 64.5. Even Tufa's typical draw may be about 59.5 if its noise matches ours (best-of-166 inflation, EB 1.7). Efficiency alone needs about 2.7× effective compute under P3 (EB 1.8), so H2 has to buy new capability **and** keep measuring it honestly.

### H2-0 (11-03 → 11-10): harvest free measurements
- **Private scores for all of our roughly 60 submissions** should appear once the competition closes (standard Kaggle behaviour; confirm). That is a second independent 55-game hidden half per draw. Use it to settle:
  - P1 vs P3 (the F-half draw);
  - the public/private relation;
  - every config's hidden mean.
- **Late submissions.** Check whether they stay open (UNVERIFIED). If they do, they become a continuing hidden-set channel. If not, "LB 60" must be measured on the H2 proxy below, or at the next ARC Prize evaluation.
- **Competitor releases.** Read Tufa's release ("when the competition ends", G-F G9), NVARC3's (if gold) and Lord Han Solo's. Re-plan from what actually got them to 67–69.

### H2-1 (weeks 1–3): the LB-proxy suite, before any training rung
- **≥55 unseen games** whose base score sits near the LB (about 31), not the public set (about 51). Sources: DreamTeam 25, sonpham's ~62 (evaluation only, no licence; W9), arc-interactive's larger games, and generated games. Pilot `game_creator` on 20 seeds with Flash-Next or the text-only Qwen3.8-2.4T, which can write game code ($2 / $6 per M on OpenRouter, W10).
- **Freeze a split** into train families, proxy, and a single-use fresh set.
- **Validate the proxy.** It must rank the known generations correctly (taaf ~4, fz ~25, 10x ~30), with its per-run sd measured.
- **Target:** detect +3 with ~10 replicates per checkpoint on 4–8 fleet boxes, ~$1–2k per month.

### H2-2 (weeks 3–12): training ladder, one rung at a time

Every rung is gated on the proxy (+3, paired p < 0.05, every replicate above the base mean), I1 (−2% decode at most) and I2. Each rung has a kill.

| Rung | Content | Compute | Kill |
|---|---|---|---|
| a | R1 refresh on all data; R3 rounds 2–3 (regenerate each round, G-G (a)) | 50–150 node-h | proxy gain < +2 after round 2 → stop R3 |
| b | R4: hindsight-filtered breakthroughs (engine-verified toward a clear) on ≥100 coverage-selected environments (AWM; coverage selection, EB 2.1) | 100–300 node-h plus generation | < +3 on the proxy at 100 environments |
| c | Frozen-state RL: GRPO-LoRA with rollout routing replay, aux loss 0, states with 0 < p < 1 selected by reward variance, rewards = level clears / verified progress (EB 2.4) | 450–950 GPU-h ≈ $4.8–10.1k (W6) | KL to the served weights >0.004, or the proxy flat after 200 steps |
| d | Multi-turn game RL (Tufa's route, at >20× our compute, L2) | open | only if c gains |

**H2 odds.**
- Mean ≥45 within 3 months: about 30%.
- Consistent 60 within 3 months: about 10%.
- The evidence that fine-tuning Flash-Next helps here is still nil (EB, five facts). Several H200 teams report no gain (W5b).
- If rungs a–b (about 6 weeks, about $10k) don't move the proxy by ≥ +6, stop training. Revisit the base model once Tufa's release, or a released Qwen 4 27B, shows a better starting point.

## 7. What this plan deliberately does not do

- **Pick finals by public score** (split-half r −0.45, G-B Q4).
- **Predict score from 20-min panels or sim2** (minute-25 r 0.21, L3 Q4; sim2 is 15–28% too steep, G-H Q3).
- **Use the public-25 score or 10-win target for trained checkpoints** (compute-coupling contamination, G-B FC 5).
- **Submit from snowiers or budget its quota** (rule 3.5.a, G-F).
- **Any MoE RL before 11-02.** Published runs cost 10k–100k GPU-h; GRPO without routing replay crashed 3 of 3 times (EB 2.4).
- **Distil from frontier APIs or the 2.4T** (licences; 2.4T is text-only, W10), and **no SDPO/OPSD** (hurt all 5 thinking models, G-G Q1).
- **World-model SFT or novelty rewards.** Novelty stalls are 0.6% of tokens (G-A Q2).
- **Train routed experts or re-quantise** (W6 FC 1). **No generic self-replay, MiniCPM mix or small-sibling wind tunnel** (G-G (b); `_critic.md` C12).
- **Relaxed acceptance or new drafter families** (accept05 lost 6–10 points, G-E). **No model swap** (27B 14.51 offline).
- **Tune any threshold on DT10.** It is opened once.

## 8. Biggest risks

1. **The LB can't see realistic gains** (§1.5), so a subtle regression could ship. Mitigations: lossless changes first; a mechanism prior only for them; the risky candidate gets one pick, paired with a different policy; ≥3 error-free draws per pick.
2. **The fleet isn't exact, or its quota isn't granted** (W6 §7). The wheelhouse is untested outside Kaggle's image. Mitigations: the calibration gate, a platform term, Branch C.
3. **External games may not track the LB** (solver-length baselines; U25 sd unmeasured, G-B Q3). Fallback: H10 + LB.
4. **R1 capture is infeasible.** There is no SGLang hidden-state hook for qwen4_exp (G-E Q3, FC 10), and botched heads have regressed before (Holodeck, G-E FC 8).
5. **R3 changes decisions.** It could repeat accept05's −8.6 residual or Arora's collapse (AIME 0.563 → 0.183) (EB 2.3).
6. **Upward elasticity is weak**, or the test misfires (about 13% false-falsification rate, G-H FC 11).
7. **The 9-h deploy path fails** with new weights, and an errored final scores nothing. The 10-21 zero-delta draw and ≥3 draws per pick cover this.
8. **Rules exposure from the past snowiers submissions**, whatever this plan does.
9. **Engineering bandwidth:** six branches in 8 days. R1 and the fleet come first; R3 is the first thing cut.

## 9. Approvals needed (the user)

| # | Decision | When |
|---|---|---|
| A1 | Stop snowiers submissions. Decide on compute-only use and whether to contact Kaggle | 10-18 |
| A2 | Fleet: GCP G4 and/or Azure NC RTX PRO 6000 quota (4 GPUs), budget cap $1k for H1 | 10-18 |
| A3 | H200 node: confirm the date; budget cap $3k for R3 round 1 (Branch A) | 10-18, re-confirmed 10-20 after the elasticity and P1/P3 results |
| A4 | Exception to the no-repeat rule for fleet replicates (calibration, base pool, candidate n) | 10-18 |
| A5 | Each Kaggle push (≤6 GPU-h/day) and each LB draw in §1.6 | daily |
| A6 | Uploading R1 head / LoRA overlays as private Kaggle models | 10-23, 10-26 |
| A7 | Final picks by the §1.6 rule | 11-02 |
| A8 (H2) | Proxy-suite build, OpenRouter credit for game generation, training rungs a–c | after 11-02 |
