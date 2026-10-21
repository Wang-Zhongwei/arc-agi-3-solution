# Training plan, risk-first lens (2026-10-18)

Planner 4 of 4. Lens: **deadline realist / risk-first.** Maximise the expected LB mean at 11-02 with the fewest ways to fail. Order work so the cheapest falsifiable gate runs first, kill fast, protect the harness fallback, and allow no step that can silently cost days.

Sources: `_evidence-base.md` (EB) and the notes in this folder, with every note's fact-check (FC) applied over its body. Note files are cited by prefix (G-H = `G-H-elasticity.md`, and so on). Repo paths are relative to `daniel-franzen-sol/`. "[I]" marks my own inference.

Approval legend: **[$]** paid compute, **[K]** Kaggle notebook push, **[LB]** leaderboard submission, **[U]** a user decision that isn't about money.

## 0. Bottom line

1. **LB 60 "consistently" is out of reach by 11-02.** It needs a draw mean of about 64.5 (draw sd 3.50; L3 FC 7). We are at about 31. Every lever we have measured adds +1 to +3 LB, and stacking them all gives about +8 at best (EB §1.6, §1.8; G-A Implication 1; G-E Q4). This plan maximises the 11-02 mean and sets up H2. It does not pretend H1 reaches 60.
2. **Expected LB mean of the final pick at 11-02: low 28.5 / mid 31.5 / high 36.** The expected best-of-two private score is about 2 points above the pick's mean (G-B FC 10). The probability that any fine-tuned *trunk* beats the harness fallback by 11-02 is about **0.10**. Counting the MTP drafter retrain (R1), which leaves the trunk untouched, the chance that some trained artifact lifts the final config is about 0.3.
3. **The largest single risk is not technical: the account rule.** Rule 3.5.a: "You will be disqualified if you make Submissions through more than one Kaggle account" (G-F G17, FC 2). snowiers has made at least 9 competition submissions, the latest on 10-17 07:15 UTC (`runs/submissions.csv`), and appears as a separate leaderboard entry (G-B FC 11). This plan budgets **mellamoa only**: 6 GPU-h a day and 45 a week. It never submits from snowiers and never uses its quota unless the user decides otherwise **[U]**.
4. **Order of work:** free checks → about 10 Kaggle GPU-h of falsifiers (10-18 to 10-20) → a paid 8-GPU node **only if** they pass (go/no-go 10-21) → at most one R3 round → LB draws only for configs that passed every offline gate. Harness fixes and R1 run in parallel without the node.
5. **Both final picks default to harness configs.** A fine-tuned checkpoint takes a pick only through the pre-registered Δ* rule after 3 or more clean LB draws (§7).

## 1. Honest odds

### 1.1 Expected LB mean of the final pick at 11-02

| Case | Mean | How it comes about |
|---|---|---|
| Low | **28.5** | Nothing new ships, or what ships is neutral. The fallback's true mean sits below its observed 30.85 because of winner's curse: the 10x family's 19 draws average 29.12 (L3 FC 7; G-B Q4). |
| Mid | **31.5** | Fallback true mean about 30.0 (30.85 over 4 draws, shrunk toward 29.12), plus three expected contributions: harness fixes about +0.6 (2–4% of tokens are harness-made sandbox errors; G-A Q4, FC 7–8; worth +0.5–1.1 offline at 2.7 per 10%, G-H Q4); R1 about +0.45 (P ≈ 0.3 × central +1.1 to +1.9 LB; G-E Q4); trunk FT about +0.2 (P ≈ 0.1 × +1.3 to +2.5 LB). |
| High | **36** | R1 at its AngelSpec-analogue upside (+2.2 to +3.9 LB; G-E FC 8), harness fixes +1, an R3 checkpoint that ships at −10% total tokens under P3 (+1.3 to +2.5; EB §1.3), and a fallback true mean near 31. |

Excluded from the table: a disqualification over the account rule, which would score 0, and a final pick that never completes. §11 covers both.

### 1.2 Why a fine-tuned trunk is unlikely to beat the fallback by 11-02

The chance is a product of steps that each have to go right [I]:

| Step | P | Basis |
|---|---|---|
| An 8-GPU node (Azure H200 or Lambda B200) is running by 10-22 | 0.5 | No official quota timeline; vendor estimate 1–4 weeks, UNVERIFIED; new subscriptions may be refused (W6 §7) |
| The upward-elasticity test passes (S(161) − S(121) ≥ +3.5) | 0.7 | Log and exp fits predict +7.0 / +8.6; false-falsification about 13% (G-H FC 3, 11) |
| R3 selection yield passes at easy states | 0.65 | Only hard states have been measured: shortest verified reply 0.68× the median, P(success shorter) = 0.445 (W7 §5, FC 7) |
| The stack trains at the deployed shape by 10-23 | 0.6 | Recipe measured only to 8k, text-only; context parallelism and multimodal input undocumented (W6 FC 2, 4; `_critic.md` G5) |
| The checkpoint passes screen and deploy gates | 0.5 | Closest analogue collapsed: Arora RS+SFT, AIME 0.563 → 0.183 (G-G Q1, FC 1) |
| Its true LB mean exceeds the fallback | 0.65 | Realistic gain +0.7 to +2.5 LB (EB §1.3) |

Our own R3 comes to about 0.045. The Swift-1.5 overlay adds about 0.06 (panel promotes 0.2 × full run holds 0.6 × beats fallback 0.5). Combined, about **0.10**. Even a truly better FT is unlikely to be **detectable**: +5 LB needs about 6 draws per arm (G-B Q4, FC 4), and R3's realistic edge is +0.7 to +2.5. No public positive result raises these priors: every public notebook scoring ≥ 20 runs stock Flash-Next, two H200 users got nothing, and the Tinfield probe was NO_GO (L2 FC 6; W5b FC 1–2; G-D FC 8).

## 2. Ground rules this plan imposes

1. **Accounts [U, today].**
   - From now on, submit only from mellamoa.
   - Never select a snowiers draw as a final.
   - Don't count snowiers quota. The rules page opens with "you cannot enter or submit from multiple accounts" (G-F FC 2), so using that quota deepens the exposure.
   - Whether to contact Kaggle about the past snowiers submissions is the user's call. I can't estimate the odds of enforcement.
2. **The fallback is frozen.** `mem-repair-10x` v1 (`kaggle/notebooks/v4-mem-repair-10x-v1.ipynb`; mellamoa draws 38.22 and 26.90) and `mem-unified-10x` v1 (32.23, 29.59) are already selectable mellamoa submissions, so pick A needs no new slot. No trained artifact replaces a fallback notebook in place; each gets its own notebook and flags.
3. **No billing before the gates pass.** Request quota today for free. Start the node's meter only after the 10-21 go/no-go (§3).
4. **Every kill condition has a date (UTC) and, for paid steps, a dollar cap.** A missed date counts as a fail.
5. **Every candidate completes a clean offline full run before its first LB draw.** No slot goes to an untested notebook.
6. **Lossy or confounded behaviour stays out of final candidates.** `ARC3_VERBATIM_HISTORY=1` scored 42.09 against 50.62 in its only, confounded, run (G-C FC 10–11). It goes into data-collection runs only. Logging-only flags (raw replies, token ids) go everywhere, since they cost nothing and make the H2 data on-policy (G-C Implication 3).

## 3. The gate ladder (cheapest falsifiable first)

| # | Gate | Cost / where | Pass threshold (pre-registered) | A fail kills | Deadline |
|---|---|---|---|---|---|
| G0 | Rules and admin | Free; user plus a Sonnet browser agent | The final-pick rule (best of 2?) and late-submission availability after 11-02 read on Kaggle; account decision made; Azure H200 quota and start date confirmed or refused; Lambda account opened | Nothing; sets the branch | 10-19 |
| G1 | Zero-delta overlay boot, then a Swift-1.5 overlay 20-min panel paired with a base panel | 1.5 GPU-h, Kaggle **[K]** | Zero-delta: decode tok/s and MTP accept within ±2% and ±0.03 of base; greedy output identical; startup ≤ +10 min; a 6 GB model version uploads. Swift: kill if decode < 0.95×, accept < 0.9× or parse errors > +2 pp; promote if tokens/action ≤ 0.85× **and** levels/action ≥ 0.95× (G-D §4) | Zero-delta fail → **all trunk-weight paths for H1** (no deploy route). Swift fail → the Swift arm | 10-20 |
| G2 | Upward elasticity: the leader at `demo_minutes` 161 | 3.0 GPU-h, Kaggle **[K]** | S(161) − S(121) ≥ +3.5 against REF S(121) = 51.6 (G-H Q5, FC 11) | **R3 for H1, and the paid node.** R1 continues only if it stays under $50 | Result 10-19 |
| G3 | R3 selection yield: state-lab on about 200 easy on-route T15 states, k = 8, whole multi-call turns to the acting call | 2.0 GPU-h, Kaggle **[K]** | ≥ 50% of states have ≥ 2 verified samples, and the median ratio (shortest verified whole-turn tokens / median verified) ≤ 0.75 | R3 for H1, and the paid node | 10-20 |
| G4 | R1 capture feasibility: dump the HC stream on a short replay; albucino's head must reproduce its served acceptance on the captured rows | 0.5 GPU-h, Kaggle or Azure RTX **[K]/[$]** | Offline E[accept] within ±0.05 of the served 2.68 (G-E Q3–Q4) | R1 | 10-23 |
| G5 | Node smoke: a 64k multimodal LoRA step at CP2/CP4; 100-step throughput; packed vs unpacked GDN loss parity; zero-LoRA merge → overlay → served on Kaggle | 16–24 node-h **[$]** | A ≥ 32k multimodal step fits; a merged LoRA serves on Kaggle at ≥ 0.98× base tok/s with spec on | **Trunk prong for H1** | 10-23 23:59, or $2.5k node spend, whichever comes first |
| G6 | R3 checkpoint screens (frozen-state, canary, G-MTP) | About 2.5 GPU-h, Kaggle **[K]** | §5 P4 | That checkpoint | 10-26 12:00 |
| G7 | U25 non-inferiority | 4.6 GPU-h, Kaggle **[K]** | Both λ variants ≥ U25-base mean − 2, with no errors | FT LB draws | 10-27 23:59 |
| G8 | LB confirmation | 3–4 slots **[LB]** | Δ* rule (§7) | FT as a final pick | 11-01 |

**The 10-21 go/no-go (one line):** rent the node **[$]** only if G1 zero-delta, G2 and G3 have all passed and the node is confirmed. Otherwise take the no-node branch (§6).

## 4. H1 calendar, 10-18 → 11-02 (UTC; Kaggle = mellamoa)

| Date | Kaggle GPU (mellamoa) | Other work (laptop / Azure / node) | LB slot **[LB]** | Decision |
|---|---|---|---|---|
| **10-18 Sat** | G2: leader at `demo_minutes` 161 (3.0). U25-a: leader on U25 (2.3), if the `environments_dir` override lands today; otherwise it moves to 10-19. Total ≤ 5.3 | G0 admin. Laptop: overlay tooling (name-keyed shard patching, `get_weights_by_name` check; G-D §3); read our Kaggle model's shard index on a CPU notebook; `arm/harness-fixes` (#176 import whitelist, #18 stale-state guard, `finish=length` handling) behind flags; `arm/dreamteam-eval` override; restore state-lab from tag `archive/tooling/state-lab`; build the R3 state list from `decisions2_joined.json` (T15 only) | `mem-unified-core-10x`, copied to mellamoa. Its only draw (34.04) is on snowiers and can't be selected. This makes a second selectable fallback candidate | Account rule decided [U] |
| **10-19 Sun** | G1 overlay notebook (1.5). fallback+ = leader + harness fixes, on the public 25 (2.3). State-lab restore smoke (0.5). Total 4.3 | R1 capture hook (`serving/runtime_patches/` HC-stream dump; G-E Q3) | **Throttle draw:** leader with a halved private deadline (information only, never a final). P1 predicts −11.4, P3 −17.6 to −18.8; about 1.6–1.9 sd apart per draw (G-H FC 12) | G2 read |
| **10-20 Mon** | G3 R3 yield (2.0). U25-b: fallback+ on U25 (2.3). Logprob smoke (0.3; G-C Q4). Total 4.6 | Azure RTX PRO 6000 calibration, 2 public-25 runs (≈$5 spot / $25 on-demand) **[$]**, if approved | `mem-unified-10x` v1 anchor (n = 3) | G1, G3 read |
| **10-21 Tue** | If go: R3 sampling (4.0) + G4 (0.5). If no-go: Swift full run if promoted (2.3) | If go: start the node; G5 day 1 (environment pinned to ms_swift 4.6.0 / mcore_bridge 1.6.5 / megatron-core 0.16.1 / transformers 5.19.0; W6 §1). R1 capture on an RTX box | fallback+ #1 (its offline run passed on 10-19) | **Node go/no-go** |
| **10-22 Wed** | R3 sampling (4.0) + Swift full run if promoted (2.3). Total ≤ 6 | G5 day 2. R1 training on the RTX box (2–8 GPU-h) | fallback+ #2, through the zero-delta overlay loader. This tests the 9-h private-rerun path for any weights overlay at no extra slot cost | — |
| **10-23 Thu** | R3 sampling (3.0). R1 G-MTP panel (0.6). fallback++ = fallback+ plus the R1 head, on the public 25 (2.3). Total 5.9 | **G5 kill at 23:59.** R1 offline selection (held-out E[accept] ≥ albucino + 0.10) | Best harness config by posterior, or Swift #1 if its full run held | G4/G5 kills |
| **10-24 Fri** | U25-c: fallback++ on U25 (2.3). Week 1 total ≈ 33 of 45 | Node: build the R3 dataset; SFT round 1 (§5 P3); λ-merges 0.5 / 0.7 / 1.0 | fallback++ #1 (only if R1 passed G-MTP and its full run) | — |
| **10-25 Sat** | FT1 G-MTP + canary + frozen-state screen (2.5) | Node: export the overlays; upload the Kaggle model version (about 6 GB) **[K]**; **deallocate the node** | fallback++ #2 | — |
| **10-26 Sun** | FT1 U25, λ variant A (2.3) | — | fallback++ #3, or a reserve | **G6 kill at 12:00** |
| **10-27 Mon** | FT1 U25, λ variant B (2.3) | — | Reserve (rerun an errored draw) | **G7 kill at 23:59** |
| **10-28 Tue** | H2 data runs (external games, verbatim and logging on), ≤ 4.0 | — | FT1 #1 if G7 passed, otherwise the best harness config | — |
| **10-29 Wed** | Same | — | FT1 #2 | — |
| **10-30 Thu** | Same | — | FT1 #3 | Δ* computed |
| **10-31 Fri** | Keep a 6 GPU-h reserve | — | FT1 #4, or a top-up for the config nearest a threshold | — |
| **11-01 Sat** | — | Provisional final selection made | Top-up | Provisional picks |
| **11-02 Sun** | — | Finals locked by 20:00 | **Submit by 06:00** (the run takes about 9 h; whether a still-running submission can be selected is unknown; G-B Q4 gaps) | Final picks |

Kaggle total for H1 is about 45–55 GPU-h, which leaves at least 6 GPU-h in reserve each week. The 16 slots match G-B's count from 10-18 (G-B Q4).

## 5. Phases: work, gates, kills

### P0. Protect (10-18 to 10-19), $0

- **Work:** G0; the overlay tool (G-D §3: range-pull tensors, patch by name, sha256 manifest, `config.json` untouched; W6 FC 8); the harness-fix arm; the DreamTeam override; the state-lab restore; raw-reply and token-id logging on every run.
- **Gate:** overlay unit tests pass on the laptop (patched bytes equal the source; untouched tensors byte-identical), and our Kaggle copy's shard layout is known. **Kill:** none.

### P1. Cheap falsifiers (10-18 to 10-20), about 13 GPU-h on Kaggle, $0–25

- **Work:**
  - G1, G2 and G3.
  - U25-a/b base runs. U25 = H10 (lp85, tr87, cn04, sc25, vc33, dc22, ka59, lf52, re86, wa30) plus 15 DreamTeam games picked by a pre-registered rule; the other 10 DreamTeam games stay sealed for a single final check (G-B Q1–Q3).
  - The fallback+ public-25 run. Report wins and time-to-win first, then score against the 26-run pool (51.17 ± 3.28; G-B Q1), plus `make significance` against the leader's score file.
- **Gates:** as in §3. **fallback+ passes** if it completes without errors, its score is ≥ 48 (pool − 1 sd) and it shows no per-game significant loss.
- **Kill:** G2 or G3 fails → no node in H1. G1 zero-delta fails by 10-22 → no trunk weights in H1.

### P2. Node and data (10-21 to 10-23), only after go

- **Node work:** G5. Download the BF16 checkpoint (360 GB) onto the 28 TiB NVMe at boot (W6 FC). Optionally build a "dequantised-Intel" BF16 checkpoint, whose experts are dequant(int4), so the trainer's weights equal the served ones (EB §2.4 [I], untested; skip it if it takes more than 4 node-h).
- **R3 sampling on Kaggle:** about 11 GPU-h → about 4.4–6.6k whole-turn samples (single-call state-lab runs at 730–960 per GPU-h, L1 FC 6; I assume about 1.5× more per whole turn [I]). That is about 550–800 states × k = 8. Add Azure RTX spot boxes if approved, at about $1 per GPU-h (W6 FC 7).
- **Selection** (G-G (a); thresholds are proposals calibrated on public states, so the PR names them):
  - whole-turn targets;
  - per state, keep the verified sibling with the fewest total turn tokens, if it is ≤ 0.7× the median verified;
  - add 30–50% median-length verified siblings as a length anchor;
  - no level-first turns and no post-death turns.
- **Gate:** G5. The dataset has ≥ 500 selected examples, and its mean target length is ≤ 0.8× the logged mean at the same states.
- **Kill:** G5 at 10-23 23:59 or $2.5k. If the dataset is short of 500 examples by 10-24 12:00 → don't train.

### P3. One R3 round (10-24 to 10-25), 8–14 node-h

- **Recipe:**
  - LoRA on GDN in/out and QSA qkv/o plus the shared expert, Swift-shaped. Never routed experts, the router or the QSA indexer (G-D Implication 3; W6 Implication 1).
  - LR 1e-5 to 3e-5, global batch ≤ 32, one epoch, aux loss 0 (G-G (d); W2 FC 4).
  - Per-call examples on the served post-trim context, with loss only on the target reply (G-G Implication 4). Use the four exporter transforms and the `qwen3_5` agent template (G-C Q1).
  - If G5 showed only ≤ 16k fits: train on short windows and accept the context-mismatch risk (AWM's −59% relative; G-G Q1). The gate at real 57k contexts below must then pass by a wider margin.
- **Cost:** about 1,200 examples × 57k ≈ 68M tokens ≈ 1.4–4.8 node-h at 2–7 node-h per 100M (W6 §3), plus merges and exports.
- **Gate:** at least one λ in {0.5, 0.7, 1.0} passes the frozen-state screen in P4.
- **Kill:** none pass by 10-26 12:00 → FT is out of the LB calendar.

### P4. Deploy gates (10-25 to 10-27), about 7 GPU-h on Kaggle

- **Frozen-state screen, paired on about 450 states,** including stuck and ≥ 80k-context states (L3 FC CV note: 420–470 per arm):
  - verified right-move rate ≥ base − 2 pp (McNemar not significantly worse);
  - whole-turn tokens per decision ≤ 0.85× base;
  - calls per decision not up;
  - distinct actions across k at stuck states ≥ 0.9× base;
  - no-tool-call rate ≤ base + 0.5 pp.
- **Canary** (G-B Q6, FC):
  - no tool call in ≤ 8 of 300 states;
  - malformed calls and sandbox exceptions ≤ +2 pp;
  - GPQA-100 drop > 6 pp or IFEval-150 drop > 5 pp → back off to a smaller λ.
- **G-MTP** (G-E Q5): SHA-check the drafter files; greedy identity spec-on vs spec-off against the FT trunk; accept ≥ 2.63; clean decode ≥ 0.98×; step time within ±1 ms.
- **U25:** two runs, one per λ variant, so neither repeats the other. Each ≥ U25-base mean − 2, with zero errors, and tokens/action ≤ 0.9× base.
- **Kill:** any fail by 10-27 23:59.

### P5. LB confirmation (10-28 to 11-01)

- **Work:** FT1 draws × 3–4.
- **Comparator:** the fallback+ draws, which use the same harness. The FT notebook is fallback+ with the overlay flag on.
- **Kill:** any FT draw errors, or Δ* ≤ −0.3 → both finals are harness configs.

### Parallel track R1, the MTP head (10-19 to 10-25)

- **Recipe** (G-E Q4):
  - train only albucino's 90.6M dense tensors;
  - keep the INT4 g32 experts byte-identical;
  - 3 chained steps with shared weights;
  - CE plus top-K KL loss on the 64K FR-Spec vocab;
  - LR 1e-5 to 5e-5;
  - hold out 5 games.
- **Hosts:** an Azure RTX PRO 6000 spot box (under 40 GB needed) **[$]**, or Kaggle if disk and RAM allow (UNVERIFIED; G-E Q3 gaps).
- **Cost:** under 15 GPU-h; under $15 spot or under $80 on-demand.
- **Gates:**
  - G4 by 10-23;
  - offline held-out E[accept] ≥ albucino + 0.10;
  - Kaggle panel: accept ≥ 2.78, greedy identity, harness tok/s ≥ +2%;
  - the fallback++ full run completes with no errors and a score within the pool band.
- **Kill:** G4 fail on 10-23, or no ≥ +0.10 held-out gain by 10-24 23:59.
- **Scope:** R1 v1 pairs only with the base trunk, because a trunk change moves the features. So it improves the fallback, not FT1 (G-E Risks).

### Parallel track H, harness fixes (10-18 to 10-21)

- **Work:** #176 (import whitelist; ImportError is 8.1% of errors), #18 (stale-state; 14.8%) and `finish=length` handling (6.8% of tokens), each behind its own flag (G-A Q4, Implication 5).
- **Gate:** the fallback+ run (P1). Close #18 and #176 once the arm's PR records a clean result.

## 6. Branches

| Branch | Trigger | What runs | Final picks |
|---|---|---|---|
| **A. Full** | Node confirmed, and G1, G2, G3 pass by 10-21 | Everything above | Δ* rule over FT1 vs the best harness config |
| **B. No 8-GPU node** | Azure H200 not confirmed by 10-20, and either the user declines Lambda 8×B200 ($53.52/node-h, W6 §8) or it isn't up by 10-22 | P0, P1, track H, track R1 (RTX spot), the Swift overlay (if promoted). State-lab sampling continues as **H2 data** with raw replies and token ids | Best two harness configs (fallback, fallback+ or fallback++), or Swift via Δ* |
| **C. No paid compute at all** | User declines every [$] | Kaggle only: track H, Swift, R1 only if Kaggle-only capture and training fits (test-first, 1 GPU-h; G-E gaps) | Harness configs |
| **D. Elasticity falsified** (G2 < +3.5) | 10-19 | Track H only, plus H2 data collection. R1 only under $50. No node | Harness configs |

In every branch the slots go to the best harness configs, so no slot is idle. Each of those draws is also a selectable final.

## 7. Final-pick rule (pre-registered)

- **Comparison:** Δ* = 0.6 × (mean of the candidate's draws − mean of the comparator's draws). The 0.6 is the skeptical N(0, 3²) prior's shrinkage; the posterior sd is 1.91 (G-B Q4, FC 10).
  - Δ* ≥ +1.0, ≥ 3 clean draws and Swift/Qwen licence checks done → both picks are the candidate.
  - −0.3 < Δ* < +1.0 → **one candidate draw and one harness draw** (the hedge costs ≤ 0.25 against the optimum at Δ = ±1; G-B Q4 table, FC 10).
  - Δ* ≤ −0.3, any errored draw, or fewer than 3 draws → both picks are harness configs.
- **Which draw:** among draws of the chosen config, pick by config, never by the public score of a draw. Its split-half correlation with private is −0.45 to −0.48 (G-B Q4).
- **If G0 shows the final score is *not* the best of the two picks,** use both picks on the highest-posterior config.
- **Calling it an alpha** (the user's rule) is stricter: every candidate draw above the fallback mean, and Δ ≥ 2·SE.
- The 2-pick hedge is a small, cheap insurance: at Δ = 0 the expected best-of-two is +2.20 for one-each against +1.97 for both-fallback (G-B FC 10).

## 8. Cost

| Step | Where | GPU-h | $ (on-demand / spot) | Approval |
|---|---|---|---|---|
| G2 elasticity, 161 min | Kaggle | 3.0 | 0 | [K] |
| U25 base family (a, b, c: three different configs, so no repeat) | Kaggle | 6.9 | 0 | [K] |
| fallback+ and fallback++ public-25 runs | Kaggle | 4.6 | 0 | [K] |
| G1 overlay + Swift panel; Swift full run if promoted | Kaggle | 1.5 + 2.3 | 0 | [K] |
| G3 R3 yield; logprob smoke; state-lab smoke | Kaggle | 2.8 | 0 | [K] |
| R3 sampling | Kaggle (+ RTX spot) | 11 (+ 12) | 0 (+ $12 / $66) | [K] [$] |
| R1 capture + training | Azure RTX PRO 6000 | 10–15 | $55–85 / $10–15 | [$] |
| R1 G-MTP panel | Kaggle | 0.6 | 0 | [K] |
| RTX platform calibration (2 runs) | Azure RTX | 4.6 | $25 / $5 | [$] |
| G5 smoke + P3 training + exports | Azure 8×H200 (or Lambda 8×B200) | 32–44 node-h = 256–352 | $2.7–3.7k Azure ($84.80/h, spot no cheaper; W6 FC 6) / $1.7–2.4k Lambda | [$] |
| Persistent disk (2 TiB P40, half a month) | Azure | — | ≈ $120 | [$] |
| FT1 gates + U25 × 2 | Kaggle | 7.1 | 0 | [K] |
| 16 LB draws | Kaggle scoring | 0 (no quota) | 0 | [LB] |
| **Total, branch A** | | **≈ 320–435** (Kaggle ≈ 45–55) | **≈ $2.9–4.0k** (Azure) / $1.9–2.7k (Lambda) | |
| **Total, branch B** | | **≈ 70–100** | **≈ $40–300** | |

**Hard caps:** node spend $2.5k by 10-23 (G5) and $4k for all of H1. The node is deallocated whenever it is idle, and always from 10-25.

## 9. H2: after 11-02, toward a consistent LB 60

**Starting facts.** Efficiency alone needs about 2.7× effective compute under P3 and about 6× under P1 (G-H Q4), so 60 needs new capability on out-of-distribution mechanics (L3 Q3; `_critic.md` C19). Two teams show it exists. Tufa's only method statement is RL on self-generated games with shorter sequences, at more than 20× our compute (L2 FC 1, 7), and it will release its solution when the competition ends (G-F FC 4).

| Phase | Work | Gate | Kill | Compute and cost |
|---|---|---|---|---|
| **H2-0** (11-03 → 11-10) | Post-mortem: our private scores settle P1 vs P3 with the throttle draw; read the Tufa, NVARC3 and Chen releases if published (CPMP: "if we end up in gold"; G-F FC 10); reproduce the best released system on Azure RTX. Confirm whether late submissions are scored **[U]** | Reproduce within ±3 of the release's public-25 claim | If late submissions are closed, switch the metric to a sealed U-set proxy | ≈ 20 GPU-h RTX, about $20–110 |
| **H2-1** (weeks 1–3) | Environment pool: arc-interactive (MIT, 249 games, filtered for contamination; W9 FC 7), DreamTeam (eval only), a 20-seed `game_creator` pilot with an open model (W9; `_critic.md` G16). Exact-stack rollout fleet on RTX spot at about $0.41/M tokens (W6 §8). All runs on-policy-logged (verbatim history, raw replies, token ids, logprobs) | ≥ 100 verified mechanic families; ≥ 150 cells with clear rate in 0.2–0.8; the U-set score tracks the LB across ≥ 4 configs (r ≥ 0.6) | Yield < 30% per generation attempt → coverage-select existing pools instead (30 chosen environments beat all 170 at 4B; W3 FC 6) | ≈ 2,000 game-runs ≈ 440M tokens ≈ $180 spot + about 10 node-h ($850) |
| **H2-2** (weeks 3–7) | Iterated on-policy RFT (regenerate every round): R3 rounds 2–3, R4 hindsight-filtered breakthroughs on external games aimed at mechanic discovery (12 of 22 real stalls; G-A Q3), cross-attempt pairs (ORBIT/PAPRIKA-style). Swift-shaped LoRA, λ-merges, R1 retrain per trunk | Each round: U-set +3 offline over ≥ 4 paired runs, then LB +3 over ≥ 6 draws | Two rounds with no gain → stop the SFT path | 150–400 GPU-h per round ≈ $1.6–4.2k (W6 Implication 7) |
| **H2-3** (weeks 6–12) | RL on frozen-state units with 0 < p < 1 chosen by variance (RAGEN-2: top-variance quartile 21.1% vs 11.0%), level rewards, length terms on correct samples only; routing replay mandatory, aux loss 0, FP32 LM head (W2 FC 1–5; G-G FC 5) | Trainer↔server KL ≤ 0.004; no collapse over 150 steps; U-set +5 | Collapse in 3 seeds, or no U-set gain by step 300 | 1–3k GPU-h ≈ $10–30k; published MoE RL is 10k–100k+ (W2 §5) |
| **H2-4** (ongoing) | Model refresh: any newly released base with evidence of ≥ 30 LB (the user's rule) | ≥ 30 LB in our harness | — | One full run plus 3 draws per candidate |

**Honest H2 odds [I].** Without a top-team release that shows a path, P(mean ≥ 64.5 within 3 months) is about 0.1–0.2. The best transfer evidence is small: VHD-Play far-OOD +0.16 to +0.24 after 34 GRPO steps on 2,176 environments (W3 FC 1). If Tufa's release is reproducible on our hardware, adopting and extending it is the cheapest route by far, and training then builds on that base.

## 10. What we are deliberately NOT doing (and why)

| Not doing | Why |
|---|---|
| Submitting from snowiers, or budgeting its quota | Rule 3.5.a disqualification risk (G-F G17) |
| Online RL / GRPO before 11-02 | Routing replay is required, the one-node recipe tops out at 1k + 8k tokens, and our budget buys only a few hundred steps (W2 FC 1–2; W6 Implication 4) |
| Routed-expert LoRA or any AutoRound re-quant | Forces an int4 re-quant of `down_proj`, which adds days of format risk; the non-expert LoRA needs none (W6 FC 1; G-D) |
| R4 / content SFT on public games in H1 | 25 families sit in AWM's degrade regime (10 environments below base); external games have never run in our harness (G-G FC 6; L1) |
| SDPO / OPSD | Privileged self-distillation lowered all 5 thinking models tested (G-G FC 2) |
| Frontier-API or Qwen3.8-2.4T distillation | The terms bar training on outputs; the 2.4T is text-only (W10 FC) |
| Novelty rewards, world-model SFT | Novelty stalls hold 0.6% of tokens; world-model SFT gives no decision gain (G-A Q2; W8 FC 2–3) |
| A second R3 round, multi-epoch training, a LoRA-scope sweep or a dense-sibling wind tunnel in H1 | No time; one Swift-shaped scope plus a free λ-sweep; a dense sibling can't rank MoE choices (`_critic.md` C12) |
| The Tinfield overlay | Tiny deltas (5–10× smaller than Swift's), NO_GO probe; a null says nothing (G-D FC 5, 8) |
| `ARC3_VERBATIM_HISTORY` in final candidates | Its only run lost 8.5 points, confounded (G-C FC 10) |
| Low-effort teacher sampling for R3 | Stay on-policy at the deployed effort; low-effort sampling is an H2 variant (EB §5.2.3) |
| EAGLE-3 / DFlash drafters, relaxed acceptance | Unservable on Qwen4Exp; accept05 lost 6–10 points (G-E Implication 5) |
| Test-time training on the card | Weights take 69.85 of 96 GB (W10 Q5) |
| Model swap to a 27B | 14.51 offline; fails the ≥ 30 rule (L3 FC 6) |
| Exact offline repeats | Project rule. U25 base runs are three different stacked configs, and FT1's two U25 runs are two λ values. True replicates would need the user's explicit exception |

## 11. Biggest risks, and the ones that silently cost days

| Risk | Mitigation (date) |
|---|---|
| **Disqualification over past snowiers submissions** | Stop now; never select snowiers draws; the user decides on any disclosure ([U], 10-18) |
| Node quota late or never | Nothing is scheduled on it; branch B is a full plan; Lambda is pre-opened (10-20) |
| ms-swift fails at the deployed shape | Pinned versions; 2-day timebox; short-window fallback gated at real 57k (10-23) |
| The overlay breaks the 9-h private rerun | Measured in G1; a real LB draw goes through the loader on 10-22 |
| R3 collapses reasoning (the Arora pattern) | Anchors, low LR, λ-merges, the right-move gate (P4) |
| The weekly Kaggle cap runs out late | A ledger, with a 6 GPU-h reserve held every week |
| The last submission is still running at the deadline | Submit by 11-02 06:00; provisional finals on 11-01 |
| Prize eligibility (the Swift licence; Qwen vs CC-BY) | Licence check before any Swift pick; affects prizes, not score (G-D Implication 6; W5b FC 5) |

## 12. Project-rule compliance

- **Branches and flags.** One branch per arm (`arm/harness-fixes`, `arm/dreamteam-eval`, `arm/weight-overlay`, `arm/r1-mtp`, `arm/r3-lora`, `arm/throttle-draw`), each change behind a default-off flag, each with a PR, an `arms.md` entry and docs updated in the same PR. Research docs and run logs go to `main`.
- **Game- and time-agnostic.** No code branches on game identity. The PR names every threshold calibrated on public games (the 0.7 selection ratio, the 0.75 yield, the screen deltas).
- **Reporting.** Offline runs report wins and time-to-win first. Improvements are claimed only with `make significance`, and an alpha needs repeated runs consistently higher.
