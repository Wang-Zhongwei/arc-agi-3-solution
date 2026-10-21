# Plan: efficiency-first, data-centric training of Qwen3.8-Flash-Next (2026-10-18)

**Lens.** Build everything around the levers we have measured: tokens per decision, serving throughput (including the MTP head), and stall breakthroughs. Use only the model's own engine-verified samples at frozen states, and rank every step by expected LB points per GPU-hour and per dollar.

**Sources.** Numbers come from the sweep notes in this folder, with each note's fact-check applied over its body. Note prefixes are short names (G-H = `G-H-elasticity.md`, and so on); "EB" is `_evidence-base.md`; "FC" is a fact-check row. [I] marks my own synthesis or arithmetic.

## 0. Bottom line

1. **The best LB per GPU-hour before 11-02 is making the fallback faster without touching the trunk:**
   - Harness token fixes: StaleState/Terminal guard messages (#18) and the sandbox import whitelist (#176). They address 2–4% of tokens (EB §1.3; G-A FC 7–8).
   - R1, a self-distilled MTP drafter: +0.15 to +0.25 accept in the central case, which is about +2.0 offline and +1.1 to +1.9 LB. It costs <15 GPU-h and $0–80 (G-E Q4, FC 12).
   - Together that is about +1.5 to +3 LB expected, on 15–20 Kaggle GPU-h. It is lossless, so it stacks into the final fallback pick.
2. **Only one trunk-LoRA round fits in H1, and it is gated hard.**
   - Data: verified own samples at frozen states on the 15 training games, in two strata. Stratum E holds easy, on-route, long turns (tokens per decision). Stratum S holds stuck states on levels another run cleared (breakthroughs).
   - Its expected LB is small: about 0.25 LB for about $2k, roughly 40× less LB per GPU-hour than R1 [I].
   - It runs because H2 needs the pipeline: exporter, 57k-context training, overlay, MTP gate and screens. Treat it as a pipeline proof, not a scoring bet.
3. **Measure before spending.**
   - Run one 161-minute leader run today (3.0 GPU-h). It tests the premise that more effective compute buys score, with a falsification rule fixed in advance (G-H Q5, FC 11).
   - Restore state-lab and run one yield screen on easy states (≈3 GPU-h).
   - Paid node hours start only if both pass and a 57k multimodal training step works.
4. **Odds.**
   - LB mean at 11-02: low 29.5, mid 32, high 36.5.
   - P(a trunk-changed model, our LoRA or the Swift overlay, beats the harness fallback and is picked) ≈ 0.10.
   - P(the R1 drafter ships in a final pick) ≈ 0.5.
   - A consistent 60 (a mean of about 64.5; L3 FC 7) is not reachable by 11-02 with any measured lever (G-H Q4; EB §1.8).
5. **Rules.**
   - Budget Kaggle on **mellamoa only**: 6 GPU-h a day and 45 h a week.
   - snowiers is already a separate leaderboard entry with 8 submissions, the latest on 10-17 (28.06, `runs/submissions.csv`). Rule 3.5.a: "You will be disqualified if you make Submissions through more than one Kaggle account" (G-F G17, FC 2; G-B FC 11). Make no more snowiers submissions, and let the user decide about the existing ones.
   - Rent single Azure RTX PRO 6000 spot VMs (~$1/h) for extra sampling instead of using the second account's quota.

## 1. What the lens takes from the evidence

### 1.1 The measured levers

| Lever | Measurement | Source |
|---|---|---|
| Effective compute → score (downward) | 8 uniform slowdowns land on the reference curve: residual −0.01 ± 1.68 | G-H Q1 (FC 4: holds for uniform slowdowns only) |
| Effective compute → score (upward) | **Never cleanly observed.** All 4 token-buying levers changed streams, window or sampling and fell below the curve (−4.3 to −23.2) | G-H Q1, FC 5 |
| Elasticity | +2.7 offline per +10% effective compute (95% CI 1.9–3.5) | G-H Q4, FC 8 |
| Offline → LB | P1: ×0.548. P3: ≈×1.0 on the real curve | L3 FC 3; G-H Q4 |
| Where tokens go | Turns ≥5k tokens hold 69.3%. Cleared on-route states about 41% (leader about 45%). Cleared on-route ≥5k turns 27.2%. Nx levels (not cleared here, cleared elsewhere) 32.4% | G-A Q1, FC 1–2 |
| Long turns are call loops | 12k+ turns: 7.67 calls, 85.5% of tokens in non-acting calls | G-A FC 3 |
| R3 headroom at hard states | Success 11.1%; P(success shorter) = 0.445; shortest/median verified 0.68 | W7 §5, FC 7 |
| R3 realistic value | A −25% cut on on-route states ≈ −10% of total tokens ≈ +1.3–2.4 offline, +0.7–1.3 LB (P1) or +1.3–2.5 (P3), unless the cut spills over to stalled turns | EB §1.3 [I] |
| MTP | Accept 2.684. Harness tok/s +2.25% per +0.1 accept. Clean decode is proportional to accept | G-E Q2–3, FC 2, 4 |
| Stalls | 240 stall-then-clear breakthroughs, 68% on the hindsight route. 12 of 22 real stalls were mechanic errors (one coder) | L1 Q4; G-A FC 5–6 |
| Stuck levels are soft | 15.6 of 17.3 live games per run sit on a level some run cleared. +1 level per hidden game ≈ +13–15 LB | G-H Q2; L3 Q2 |

Three consequences shape this plan:
- **Efficiency levers are worth something only if upward elasticity holds.** That has never been measured, and the test costs 3.0 GPU-h.
- **The cheapest clean upward lever is R1.** It is a uniform decode speed-up that doesn't change what the streams see, so its full run doubles as a second upward-elasticity point [I].
- **R3 must target the long-turn tail, not just cleared states.** Selection is per whole turn (fewest total turn tokens to a verified decision; G-A Implication 2). Stratum S extends verified selection to stuck states on Nx levels, using routes from other runs as hindsight.

### 1.2 Levers ranked by expected LB per GPU-hour (H1)

The products use P1–P3 midpoints [I].

| Lever | LB if it works (P1–P3) | P(ships by 11-02) | Expected LB | Cost (GPU-h; $) | Expected LB per 10 Kaggle GPU-h |
|---|---|---|---|---|---|
| Harness token fixes (#18, #176), stacked | +0.3 to +1.1 (2–4% of tokens × 2.7 per 10%) | 0.8 | +0.55 | ≈0 extra (shares the fast-fallback run); $0 | high (≈2) |
| ARC hot map for FR-Spec (only if >1–2% of tokens fall outside the 64K map) | 0 to +0.5, UNVERIFIED | 0.3 | +0.1 | laptop; $0 | high |
| R1 MTP drafter | +1.1 to +1.9 central; +2.2 to +3.9 at the AngelSpec level | 0.5 | +0.75 | 10–15; $0–80 | ≈0.6 |
| Swift-1.5 trunk overlay (no training) | −3 to +3; Felix: −4% tokens per action, decode 4–8% faster | 0.15 | ≈+0.1 | 1.4–3.7; $0 | ≈0.4 |
| One R3/S LoRA round | +0.7 to +3.7, plus an unknown S effect | 0.12 | ≈+0.25 | 20–30 Kaggle + 15–30 node-h; $1.1–3k | ≈0.1 (≈0.0001 LB per $) |

Sources by row: G-A FC 7–8; G-E Implication 6; G-E Q4, FC 8, 12; G-D FC 7; EB §1.3, §1.6.

## 2. Odds at 11-02

**Starting point.**
- mem-repair-10x has drawn 38.22, 28.73, 29.54 and 26.90 (mean 30.85); mem-unified-10x 32.23 and 29.59. The 10x family's 19 draws average 29.12 (L3 FC 7; G-B Q4).
- Shrunk toward the family mean, the fallback's true LB mean is about 30.3 [I].

| Case | LB mean | What happens |
|---|---|---|
| Low (≈10th pct.) | **29.5** | E1 falsifies upward elasticity, R1's capture hook fails, and the LoRA never clears its gates. The fallback's true mean sits at the family level |
| Mid | **32.0** | Harness fixes ship (+0.5) and R1 ships at +0.2 accept (+1.1 to +1.9). No trunk model is picked |
| High (≈90th pct.) | **36.5** | P3 holds; R1 reaches the AngelSpec level (+0.4 accept, +2.2 to +3.9); the harness fixes give +1; the LoRA gives −15% tokens per decision at equal right-move rate (+2 to +3.7) and is picked |

The mid case is below L3's "33–45 realistic". Two corrections explain the gap: R3 reaches only on-route states (−10% of total tokens, not −25%; G-A Implication 1), and P(FT ships) is low.

**P(fine-tuned trunk beats the fallback and is picked) ≈ 0.10 [I].** It decomposes as:
- a node with a passing 57k step by 10-23: 0.5;
- data, training, merge and every gate done by 10-29: 0.5;
- the true LB mean actually higher: 0.35, given no public evidence for Flash-Next fine-tunes (L2 FC 6; W5b FC 1–6) and the Arora collapse precedent (G-G FC 1);
- detection by the Δ* rule with 3 draws: 0.6.

That product is ≈0.05. The Swift overlay adds ≈0.05 (promoted 0.15 × truly better 0.5 × detected 0.6), for a total of ≈0.10. R1 is a trained model but lossless; P(it ships in a final pick) ≈ 0.5.

## 3. Compute, accounts and approvals

| Resource | Use in this plan | Price | Source |
|---|---|---|---|
| Kaggle mellamoa | All evaluation; R1 capture and training if there is no spot VM; small screens | Free. 6 GPU-h/day; 45 h/week, resetting Sat 00:00 UTC (10-24, 10-31) | CLAUDE.md; auto-memory `kaggle-gpu-budget` |
| Kaggle snowiers | **None by default.** Never submits. Compute-only use is not addressed by the rules, but the rules header says "you cannot sign up to Kaggle from multiple accounts" | — | G-F G17, FC 2 |
| Azure NC144lds RTX PRO 6000 (spot) | State-lab sampling, R1 capture and training, DreamTeam panels. The exact serving GPU | $1.02/h spot, $5.50 on-demand. Calibrate decode against Kaggle (stop if >5% off) | W6 FC 7; G-B Q7 |
| Azure ND96isr H200 v5 | LoRA training only, plus sampling if SGLang W4A16 serves on sm90. **Deallocate between jobs**: an idle node costs $2,035/day | $84.80/node-h; spot is the same price | W6 FC 6 |
| Lambda 8×B200 | Fallback node if the Azure quota slips. fla had sm100 bugs | $53.52/node-h | W6 §8 |

Unit costs used below:
- full 121-min run: 2.3 GPU-h;
- 161-min run: 3.0 GPU-h;
- 25-min panel: ≈0.7 GPU-h;
- state-lab single-call samples: 730–960 per GPU-h all-in;
- whole-turn samples: ≈330–440 per GPU-h [I: mean deciding turn 4,411 tokens vs ≈2k per call; L3 Q2];
- LB commit run at `demo_minutes` 1: ≈0.25 GPU-h [I: boot 487–523 s; L1 FC 6].

Sources: L1 Q6; G-H FC 11; G-E Q4; L1 FC 6.

**Approvals the user must give:**
- every Kaggle push and every LB submission (dates below);
- Azure RTX PRO 6000 spot, capped at $100 for H1;
- the Azure H200 node: start date, plus a cap of $3.5k for H1 (central $2k);
- Lambda as the fallback node;
- uploading private Kaggle models (overlays and the drafter);
- any use of snowiers.

## 4. H1: 2026-10-18 → 2026-11-02 23:59 UTC

### 4.1 Phases

| Phase | Dates (UTC) | Work | Compute; $ | Gate (pre-registered) | Kill (dated) |
|---|---|---|---|---|---|
| **A. Measure** | 10-18 → 10-19 | E1: leader at `demo_minutes` 161, otherwise unchanged. Throttle LB draw. Laptop builds: harness flags, hot-map count, overlay tool, exporter, state-lab restore | Kaggle 3.25; $0 | E1 gate below | 10-19 23:59: if E1 falsifies, drop stratum E for H1. R1 continues only as a cheap lossless item |
| **B. Cheap evidence** | 10-19 → 10-21 | State-lab validation on the leader stack plus the G4 yield screen. Panel pair: zero-delta overlay (also the same-day base panel) and the Swift overlay. R1 capture and training | Kaggle ≈10, or Kaggle 5.2 + spot ≈$8; $0–8 | Gates below | State-lab identity <95% by 10-21 → no R3/S data, no-node branch. R1 not through its self-test by 10-21 12:00 → the fast-fallback run goes without it; not ready by 10-24 → R1 leaves H1 |
| **C. Fast fallback** | 10-21 → 10-25 | R1 G-MTP panel. One full run of leader + harness fixes + R1 (+ hot map). 3 LB draws | Kaggle 3.0; $0 | Wins and time-to-win reported first. Accept_tw ≥ 2.78, harness tok/s ≥ +2%, score ≥ 48.3, no new error class | Mechanism gate fails → keep mem-repair/mem-unified as the fallback |
| **D. One LoRA round** (node branch) | 10-21 GO → 10-28 | Node smoke; data generation (E+S); base U25 runs; train; λ-merge; overlay; screens; U25 run | Kaggle ≈14 (≈25 if data is generated on Kaggle); node 15–30 h; spot $20–60; ≈$1.4–3.4k | Node GO (10-21 12:00); 57k smoke; data yield; G-MTP / G1 / G2 / G3 below | No node with a passing 57k smoke by **10-23 23:59** → no-node branch. No checkpoint through G1–G3 by **10-29 12:00** → FT out of the finals |
| **E. Draws and finals** | 10-28 → 11-02 | FT1 ×3 draws if it passed, else fast-fallback and fallback draws. Top-ups. Finals by the Δ* rule | LB only; ≈0.75 GPU-h of commits | Δ* rule (§4.5) | 11-02 12:00: last draw. 23:59: finals selected |

**E1 gate** (G-H Q5, FC 11):
- The run's own S(121) must fall within 51.57 ± 3.3, the REF noise. This checks that a truncated run is equivalent to a shorter one.
- **Falsified if S(151) − S(121) < +2.5 or S(161) − S(121) < +3.5.** The false-falsification rate is about 13%.
- **Supported if S(161) − S(121) ≥ +6.**

**Phase B gates:**
- **State-lab validation:** ≥95 of 100 boards and actions identical, and the tool result byte-identical, on the leader stack. Today's figures are 100/100 on mem-v2-10x and 13/14 tool results on pf-solved (L1 FC 7).
- **G4 yield, on 100 T15 on-route states whose logged turn was ≥5k tokens, k = 8 whole-turn samples:**
  - ≥40% of states have ≥2 verified siblings;
  - the median shortest/median verified turn ratio is ≤0.80;
  - the projected cut is ≥8% of *total* tokens (stratum share × (1 − ratio)), which is ≈+2.2 offline at 2.7 per 10% [I].
- **Stratum S, on 40 Nx stuck states at k = 12:** the verified-progress rate is ≥5%, against the 11.1% base success rate at hard states (W7 FC 7).
- The node GO needs the E gate **or** the S gate.
- **Zero-delta overlay** (G-E Q5; G-D §4): accept within ±0.02 of base, clean bs10 tok/s within ±1%, and the same greedy spec-on/off identity as base.
- **Swift overlay**: promote it to a full run only if tokens per action ≤0.85×, levels per action ≥0.95×, decode ≥0.95× and parse errors ≤ +2 pp.
- **R1 offline:** expected accept E[L] on H10 transcripts must beat albucino by ≥0.1 (G-E Q4).

**Phase C gate.** The score floor is the REF mean minus one run sd (G-H Q2; G-E Q5). The adoption logic is in §4.5.

### 4.2 Day-by-day calendar

Kaggle hours are mellamoa GPU-h, counted by Pacific day. A run "on spot" moves to an RTX PRO 6000 VM if the user approves one.

| Date | Kaggle (GPU-h) | Laptop / paid | LB slot (needs approval) |
|---|---|---|---|
| 10-18 Sun | E1: leader at 161 min (3.0); commit for the throttle notebook (0.25) | Flags for #18 and #176; FR-Spec hot-map coverage over the 269.6M generated tokens; overlay tool (name-keyed shard patch, symlinks, original `config.json`; G-D §3, W6 FC 8); exporter with a per-row token-parity assert (G-C §1); state-lab restored onto a branch; ask for approvals and the node start date | **Throttle draw**: the leader with the private deadline at 266 of 532 min. P1 predicts −11.4, P3 −17.6 to −18.8 (G-H FC 12). It shifts the prior; it isn't a gate |
| 10-19 Mon | State-lab validation + G4 yield + truncation probes + logprob smoke (3.5); panel pair: zero-delta overlay and Swift (1.4); zero-delta commit (0.25) | Read E1 under its rule; write the R1 capture hook and dense-only exporter (byte-identity self-test) | Fallback anchor: mem-unified-10x v1, 3rd draw |
| 10-20 Tue | R1 capture (prefill of T15 transcripts from 4 leader-family runs, spec off) and training in one session (≈5.0), or on spot (≈$8) | R1 export | **Zero-delta overlay draw**: tests the FT load path in the 9-h run, and counts as a base-model draw |
| 10-21 Wed | R1 G-MTP panel (0.7); **fast-fallback** full run (2.3); if node GO, a DreamTeam-25 base panel (2.3, or $2.5 on spot) | **12:00 node GO/NO-GO**: E1 not falsified, G4 or S yield gate met, and node start ≤10-23. Then the node smoke (≈8 node-h, $680) | Swift draw if its panel promoted it, else mem-repair anchor (5th) |
| 10-22 Thu | Data generation, stratum E + S (5.5), unless it runs on spot or the node; Swift full run (2.3) only if promoted | Node: BF16 checkpoint with dequantised Intel experts (so the trainer's weights are the served weights; W2 §3); one 57k multimodal LoRA step at CP2/CP4; 100-step throughput; packing parity; SGLang W4A16 sm90 sampling bench | Fast-fallback #1 (if Phase C passes) |
| 10-23 Fri | U25 base #1 (2.3) + data generation (3.0) | **23:59 node kill** | Fast-fallback #2 |
| 10-24 Sat | U25 base #2 (2.3); G2 base arm on H10 states (1.2); data top-up (2.0) | Node: LoRA round 1 (4–15 node-h) | Fast-fallback #3 |
| 10-25 Sun | Reserve | Node: checkpoints at 25/50/75/100%; λ-merges; overlays; private Kaggle upload | Reserve: rerun an errored draw, else a fallback anchor |
| 10-26 Mon | FT1: mini λ-screen of 2 λ × 150 states (0.8); G2 FT arm (1.2); canary FT and base (1.0); G-MTP (0.7); commit (0.25) | If accept falls >0.05, retrain R1 on the FT1 trunk (node ≈2 h or Kaggle ≈4–5) | Fast-fallback #4, or the anchor with fewer draws |
| 10-27 Tue | FT1 U25 run (2.3) | — | Reserve |
| 10-28 Wed | Reserve | — | FT1 #1 (only if G1–G3 passed) |
| 10-29 Thu | Reserve | **12:00 FT kill** | FT1 #2 |
| 10-30 Fri | Reserve | — | FT1 #3 |
| 10-31 Sat | Reserve (new quota week) | — | Top-up for whichever config sits near the Δ* band |
| 11-01 Sun | — | — | Top-up |
| 11-02 Mon | — | Finals selected by 23:59 | Last copy of the leading config by ≈12:00 |

**Totals.**
- Kaggle: ≈30 GPU-h in the week of 10-17 and ≈13–18 in the week of 10-24, so ≈45–60 GPU-h in all with reserve, under the caps.
- No-node branch: ≈20–30 GPU-h.
- LB: 16 slots, all on mellamoa.

**A precedent to confirm.** mem-repair-10x drew 4 times after the 10-12 no-repeat rule, so LB redraws of an unchanged notebook have precedent. If the user reads "never run the same notebook twice" as covering LB submissions too, replace the anchor draws with harness candidates.

### 4.3 The dataset: one pipeline, two strata

Both strata are drawn only from **T15**, the 15 non-held-out public games. **H10** (lp85, tr87, cn04, sc25, vc33, dc22, ka59, lf52, re86, wa30) is never trained on (G-B Q1–2). Each sample is the full multi-call turn, run in the real sandbox up to the acting call, with the raw reply and token ids saved (G-G (c); L1 Implications).

| | Stratum E: same decision, fewer tokens | Stratum S: stall breakthroughs |
|---|---|---|
| States | On-route states on cleared levels whose logged turn was ≥5k tokens. Not a level's first turn, not the turn after a death (W7 §5; Agent-Omit) | Stuck states on Nx levels whose board lies on another run's clearing state graph, plus the 15 actions before each of the 240 logged breakthroughs (L1 Q4) |
| k | 8 | 16 (success ≈11% at hard states; W7 FC 7) |
| Verified if | Same next board hash as the logged step, or a step onto the BFS hindsight route; no death; every intermediate frame passes (W7 §5) | The committed action(s) reach an on-route board the stuck run had not yet reached ("progress over earlier attempts"; W2 §6). No novelty reward (G-A Implication 3) |
| Selection | The verified sibling with the fewest total turn tokens, if ≥0.7× the state's median verified length; otherwise the next shortest. Skip states with <2 verified siblings (G-G (a)) | Up to 2 verified siblings per state, shortest first, under the same 0.7× cap |
| Anchor | 30–50% median-length verified siblings at the same states (G-G (a)) | — |
| Stretch (test-first in G4) | Truncation probe: cut verified thinking at 40/60/80%, decode only the tool call, keep the earliest cut that verifies (W7 §5) | — |
| Target size | ≈1,100 turns with node or spot sampling; ≈400 on Kaggle only | ≈400 turns, or ≈150 on Kaggle only |
| Sampling cost | ≈16k whole-turn samples: ≈$40–50 on spot, ≈5 node-h, or ≈40 Kaggle GPU-h (too many; cut to ≈5k) | ≈2.4–4.8k samples: 6–12 GPU-h |

**Thresholds calibrated on public games** (the PR must name them): the ≥5k-token stratum cut, the 0.7× cap, the hindsight-route construction, and the G4 and S yield gates.

**Excluded:** behaviour cloning of logged replies, teacher text, solver rationales, generic replay, and DreamTeam games, which are evaluation only (G-G (b); EB §2.1; G-B Q3).

### 4.4 Training recipe (round 1)

- **Trainer:** ms-swift Megatron (mcore-bridge), pinned to ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1 and transformers 5.19.0 (W6 Implication 2).
- **Base weights:** BF16 with the routed experts dequantised from Intel's int4. The QSA indexer is frozen (EB §4.1).
- **LoRA targets:** GDN in_proj/out_proj, QSA linear_qkv/linear_proj and the shared expert. Never the routed experts or the router. This keeps deployment a file overlay with no re-quantisation (W6 FC 1; G-D Implication 3).
- **Rank:** r16. The 10× LoRA-LR rule is rank-independent (G-G (d)).
- **LR and schedule:** 2e-5 (band 1e-5 to 3e-5), global batch ≤32, one epoch (G-G (d)).
- **Checkpoints and merge:** save at 25/50/75/100%; λ ∈ {0.5, 1.0} goes to the mini-screen (WiSE-FT/AIMO merge; EB §2.2).
- **Shape:** one example per call on the served, normalised, post-trim context (31–85k), loss on that call's raw reply. Historical thinking stays in context, as served. Use the four exporter transforms and the `qwen3_5` agent template (G-C §1).
- **No short-context training:** AWM shows a 59% relative loss from context mismatch (G-G Q1).
- **Size:** about 1,500 turns × 2.5 calls ≈ 3.7k call examples ≈ 214M tokens ≈ 4–15 node-h, about $360–1,270 [I, scaled from G-G (c)'s 3,000 × 57k ≈ 171M ≈ 3.4–12 node-h].
- **Small node smoke:** if the 57k step fits only at CP4, or measures under about 4k training tokens/s per node (214M tokens in 15 node-h), halve the dataset rather than truncating contexts [I].
- **No:** aux load-balancing loss (set to 0, as for RL), replay, or a wind tunnel (G-G (b); `_critic.md` C12).

### 4.5 Gates for any trunk change, and for picks

- **G-MTP** (G-E Q5):
  - SHA-check the drafter;
  - greedy spec-on/off identity ≥ base − 5 pp;
  - on a panel: accept ≥ base − 0.05, clean bs10 ≥ 0.98× base, step time ±1 ms;
  - in a full run: accept ≥ 2.63.
  - If accept falls by more than 0.05, retrain R1 on the new trunk. The same pipeline is built in Phase B.
- **G1, canary** (G-B Q6, FC):
  - 300 logged states: no-tool-call >8/300 fails (P ≈ 0.008 at the 1.13% base);
  - malformed calls or sandbox errors > base + 2 pp fail;
  - GPQA-100 down >6 pp, or IFEval-150 down >5 pp → λ-merge toward base.
- **G2, frozen-state screen** on 450 H10 states (300 easy, 150 stuck), one whole-turn sample per arm, paired by state:
  - tokens per verified decision ≤0.85× (paired one-sided p < 0.05);
  - right-move rate ≥ base − 3 pp (easy) and ≥ base − 2 pp (stuck);
  - calls per turn ≤ base;
  - distinct actions across k = 4 on 50 stuck states ≥0.9× base;
  - verification/backtracking markers per 1k tokens fall ≤10% (G-G (a)6; W8 Implication 6).
- **G3, U25** (H10 + 15 DreamTeam; G-B Q1, Q3):
  - wins and time-to-win reported first;
  - score ≥ the mean of the 2 base U25 runs − 2;
  - levels per M generated tokens ≥ +10%.
  - Never score H10 in a run that contains T15 (G-B FC 5).
- **Picks.** Let Δ* = 0.6 × (FT draw mean − fallback draw mean).
  - Both FT if Δ* ≥ +0.3;
  - one FT and one fallback if |Δ*| < 0.3;
  - both fallback otherwise, or if any FT draw errored.
  - The ±0.3 band is the posterior-sd correction (G-B FC 10).
  - Never pick by public score (split-half r = −0.45; G-B Q4).
- **Adopting the fast fallback.** It changes no decisions, so it replaces mem-repair as the fallback if all of these hold:
  - its full run passes the mechanism gate (accept, tok/s, no new errors);
  - its 3 draws' mean is ≥ the fallback mean − 2;
  - `make significance` on the offline score file shows no per-game regression.

### 4.6 No-node branch

If no H200 (or approved Lambda) node passes the 57k smoke by 10-23 23:59:
- **Ship:** the fast fallback (harness fixes, R1, the hot map if it qualifies) and the Swift overlay if promoted.
- **Bank for H2, at ≈$0:**
  - every collection run logs raw replies and output token ids;
  - the state-lab whole-turn E and S banks, sampled on spot or spare Kaggle hours;
  - the DreamTeam-25 base panel (G7).
- **Finals:** fast fallback + the best other harness config (or Swift). Training waits for H2.
- **Kaggle need:** ≈20–30 GPU-h.
- **Don't** attempt trunk training on 96 GB cards. A ~100B model can't be trained on an RTX PRO 6000 (W5b §1), and the 2×96 GB QLoRA claim is unverified (W5b Implication 5).

### 4.7 Project rules, applied

| Branch | Contents (each change behind its own default-off flag) |
|---|---|
| `arm/harness-token-fixes` | #18 and #176 fixes. Close both issues when merged |
| `arm/mtp-r1` | Capture hook and dense-only exporter. A drafter-directory flag; default albucino |
| `arm/trunk-overlay` | Overlay loader. Arms `zero-delta` and `swift15` |
| `tooling/state-lab` | Restored from `archive/tooling/state-lab`. Adds whole-turn sampling, raw-reply and token-id logging, and the E/S grade rules |
| `arm/ft-r3s-1` | The FT1 checkpoint flag |
| `arm/elasticity-161`, `arm/throttle-half` | Arm entries only |

Other rules:
- Stacking: the fast-fallback run stacks the fixes, R1 and the hot map, each behind its own flag.
- E1 runs unstacked, because stacked flags would confound its S(121) validity check [I].
- No notebook is pushed twice.
- Every full run reports wins and time-to-win first.
- Score comparisons use `make significance`.
- An alpha needs the user's repeated-run rule: every carrying run above the reference.
- Docs (CONFIGURATION.md, kaggle/README.md, the top-level README) are updated in each PR.
- Research notes and run logs go straight to main.
- The code stays game- and time-agnostic: training selects states from public games, but no code branches on a game id.

## 5. H2: after 11-02, toward a consistent LB 60

**What 60 requires.** A mean of ≈64.5 (L3 FC 7). Efficiency alone needs ≈2.7× effective compute under P3, if the log curve holds that far, or ≈6× under P1 (G-H Q4). So H2 must add capability, meaning more levels cleared on new mechanics (≈+13–15 LB per +1 level per game; L3 Q2). Efficiency remains the multiplier.

| Phase | When | Work | Compute; $ | Gate | Kill |
|---|---|---|---|---|---|
| **H2-0. Re-plan on new information** | 11-03 → 11-09 | Read Tufa's release, promised "when the competition ends" (G-F G9). Calibrate the platform with 2 spot reference runs. Run the leader at 242 min (S(242)). Run VERBATIM_HISTORY in isolation (it saves ≈2M prefill tokens a run; its only run is confounded; G-C FC 10–11). Check whether late LB submissions exist (UNVERIFIED) | ≈15 spot GPU-h ≈$15 | S(242) − S(121) ≥ +14 → efficiency stays a main lever; ≤ +10 → capability first (G-H Q5) | — |
| **H2-1. Environment and data engine** | 11-09 → 11-30 | A spot fleet (4–8 VMs) on the exact stack. Base runs on DreamTeam (25) and qualifying arc-interactive games, after the contamination filter (W9 Implication 5). A 20-seed `game_creator` pilot with Flash-Next. Target ≥100 mechanic families (W9; AWM, G-G FC 6). PLR-lite cell selection, p in 0.2–0.8. Every run logs raw replies, token ids and logprobs | 300–600 spot GPU-h ≈$300–600 | ≥60 families with ≥1 frontier cell; noise on held-out families measured | game_creator yield <25% or >1 GPU-h per kept game → existing pools plus level edits only (W9 Implication 1) |
| **H2-2. Iterated on-policy RFT** | 12-01 → 12-31 | 3 rounds, each regenerated from the current checkpoint (Iterative-SFT; EB §2.1). Strata E + S on all 25 public games plus the training families. Cross-attempt filters (ORBIT/PAPRIKA; W8 Implication 2). Same-state RPO pairs after SFT at LR ≈2e-6 (G-G (e)). λ-merge. R1 retrained each round | 3 × 10–15 node-h + ≈150 spot GPU-h ≈$2.7–4k | Held-out families: levels per wall-clock ≥ +10% at matched compute, with G1 clean, and LB draws if open | 2 rounds with no held-out gain → stop training the trunk; keep R1 and the harness |
| **H2-3. Frozen-state RL** | Jan → | GRPO-LoRA (ms-swift Megatron) with routing replay, aux loss 0, an FP32 LM head and a trainer↔SGLang KL gate ≤ ≈0.004 (W2 FC 1–5). Reward: verified progress, with a length term on correct samples only. States chosen by reward variance (RAGEN-2; G-G FC 5). Rollouts on the spot fleet, IS-corrected with logged logprobs | 60–210 node-h + ≈200 spot GPU-h ≈$5–18k | LB mean ≥ +5 over the RFT model across ≥6 draws (+5 needs ≈6 draws; G-B FC 4), or the held-out-family equivalent | No-tool-call >3%, or no gain by step 150 |

**H2 totals.** ≈100–260 node-h (≈800–2,100 H200 GPU-h) plus 450–950 spot GPU-h ≈ **$9–23k**, staged so that each stage spends only after the previous one gains.

**H2 odds [I].**
- P(mean ≥45) ≈ 0.3.
- P(consistent 60, a mean ≈64.5) ≈ 0.05–0.10.
- Tufa reported end-to-end RL on self-generated games at more than 20× our compute (EB §3.2). Their release is the single biggest lever on these odds.

## 6. Deliberately not doing (and why)

| Not doing | Why |
|---|---|
| Online multi-turn or full-game RL before 11-02 | MoE RL needs routing replay; published runs take 10k–100k+ GPU-h; the only one-node recipe is 1k + 8k tokens (W2 FC 1–2, §5; W6 FC 4). Full games have near-zero group variance |
| A second LoRA round in H1 | No time to regenerate and re-gate. One round proves the pipeline |
| Routed-expert LoRA, full fine-tuning, AutoRound re-quantisation | They break the free overlay path. Non-expert LoRA needs no re-quantisation (W6 FC 1; EB §5.1 #3) |
| Distilling from Claude, GPT or Gemini; the 2.4T as a text teacher | Their terms bar it; the 2.4T is text-only (W10 FC) |
| SDPO / OPSD privileged self-distillation | It lowered all 5 thinking models tested (G-G FC 2) |
| World-model SFT or transition GRPO (M3) | It doesn't improve decisions (W8 FC 2–3) |
| Novelty or curiosity rewards | Novelty stalls are 0.6% of tokens; exploration rates are similar on failed levels (G-A FC 4–5) |
| 30–50% generic self-replay | It anchors only partly. Use median anchors and λ-merges (G-G (b), FC 10) |
| Behaviour cloning of logged replies | History is normalised, and it repeats the base policy rather than teaching fewer tokens (L1 FC 2; G-C Q2) |
| Training on short rebuilt contexts | AWM: context mismatch costs 59% relative (G-G Q1) |
| A small-sibling wind tunnel | Dense siblings can't rank MoE target choices (`_critic.md` C12) |
| External-game ReST or `game_creator` before 11-02 | Pass rates and yield are unknown; ≈85 GPU-h per 1k game-runs (G-B FC) |
| EAGLE-3 / DFlash drafters, relaxed acceptance, deeper speculation | Unservable on qwen4_exp; accept05 lost 6–10 points; QSA caps a verify block at 4 (G-E Implication 5) |
| More streams or a smaller window to buy tokens | 4 of 4 such levers fell below the curve (G-H Q1) |
| `reasoning_effort=low` at play time; a low-effort teacher | Cuadron: 29.1 → 21.0. G-G (a)3 |
| Tinfield / AZ-Focus overlays in H1 | Tinfield's deltas are tiny and its probe was NO_GO; AZ-Focus touches routed experts (G-D FC 4–8) |
| The 242-min run in H1 | 161 min decides R3's go/no-go for 1.3 GPU-h less. 242 runs in H2 on spot for ≈$5 |
| Flipping VERBATIM_HISTORY in H1 | Its only full run is confounded (−8.5); per-call R3 training doesn't need it (G-G (c)) |
| System-prompt distillation; test-time training on the card | Under 2% of wall; weights take 69.85 of 96 GB (W10 Q3, Q5) |
| Using snowiers for submissions | Rule 3.5.a disqualification risk (G-F G17) |

## 7. Biggest risks

1. **Account rules.** snowiers is a separate leaderboard entry with 8 submissions, the latest on 10-17. Rule 3.5.a can disqualify regardless of score (G-F G17; G-B FC 11). This plan adds none, but the existing exposure is the user's decision.
2. **Upward elasticity may not exist.** All 4 compute-buying levers fell below the curve, and the curve's shape above 121 min is unsettled (G-H Q1, FC 3). If E1 falsifies it, R1 and R3 are worth ≤ +1.5 LB.
3. **R3 collapse.** Arora's RS+SFT took AIME 0.563 → 0.183 (G-G FC 1). At hard states length doesn't predict correctness (P = 0.445; W7 FC 7), and reasoning down-weighting has caused no-tool-call collapse (W5b FC 3). G2 and the λ-merges are the guards.
4. **Training at 57k with images is untested.** The recipe is measured only to 8k, text-only (W6 FC 2). The node may not arrive (`_critic.md` G1).
5. **Transfer from public to hidden games.** We train on T15. Within the family, offline correlates with LB at r = 0.44, and P1 vs P3 is unresolved (L3 FC 3; G-H Q5).
6. **The R1 capture hook doesn't exist** for qwen4_exp (SGLang #38589 is open). Botched head retrains have regressed (G-E FC 8, 10).
7. **Evaluation noise.** 16 draws at sd 3.5 can confirm only ≥ +5 LB (G-B Q4). Screens are regression gates, not predictors (G-B Q5).
8. **State-lab has not been validated on the leader stack** (L1 FC 7).
9. **Node cost overrun.** An idle ND H200 node costs $2,035/day. Deallocate it after every job and keep data on a persistent disk (W6 FC 6) [I].
