# Final training plan: Qwen3.8-Flash-Next for ARC-AGI-3 (2026-10-18)

This plan starts from the risk-first draft (`plan-risk-first.md`), which two of three judges ranked first. It fixes every fatal flaw the judges named and takes their grafts from `plan-efficiency-first.md`, `plan-eval-centric.md` and `plan-tufa-imitation.md`. **Revised the same day after three red-team passes;** the Review log at the end lists the findings I did not apply as written.

**Sources.** Numbers come from `_evidence-base.md` (EB) and the sweep notes, cited by file prefix (G-H = `G-H-elasticity.md`, and so on); each note's fact-check (FC) overrides its body. Repo paths are relative to `daniel-franzen-sol/`. [I] marks my own inference or arithmetic.

**Approval tags:** **[$]** paid compute, **[K]** Kaggle push, **[LB]** leaderboard submission, **[U]** a user decision that isn't about money. Dates and times are UTC, except the Kaggle GPU column in §6, which uses Pacific days (07:00 → 07:00 UTC).

## Answer

1. **A consistent LB 60 is out of reach by 11-02.** It means a draw mean of about 64.5 (sd 3.50), and ours is about 31 (L3 FC 7). Each H1 lever is *projected* (not measured) at +0.05 to +1.5 LB, from a compute curve validated only downward (G-H FC 4–5), and efficiency alone would need 2.7–6× effective compute (G-H Q4). So H1 maximises the 11-02 mean and builds what H2 needs.
2. **The H1 plan in one line.** Settle the two-account exposure first (D0). Then run cheap falsifiers on mellamoa's free Kaggle hours (10-18 to 10-22) and ship the near-lossless line: harness hygiene fixes (#176 shims, #18's narrow fix, the #27 deadline flag), the victory lap, any serving lever that benches ≥1%, and R1, a retrained MTP drafter head. **The default calendar rents no training node (Branch B).** The node plus one R3 LoRA round (Branch A) is an H2 investment: I recommend it in H1 only if G0 shows late submissions are closed and every GO condition passes on 10-23. Otherwise it starts after 11-02. A trunk fine-tune can take only pick 2.
3. **Odds at 11-02** [I].
   - LB mean of pick 1: low 28, mid 30, high 32.5. The mid case is within noise of F.
   - P(a fine-tuned *trunk* is truly better than the fallback and is picked) ≈ 0.04.
   - P(any trained weights, R1 included, lift pick 1) ≈ 0.3.
   - P(LB mean ≥ 60) < 1%.
   - Disqualification is listed separately (§3.1). It can't be estimated from our evidence.
4. **Spend.**
   - Kaggle: about 31–45 GPU-h (Branch B) or 42–50 (Branch A), mellamoa only.
   - Branch B: $70–170 of rented RTX PRO 6000 time (cap $300).
   - Branch A: $1.7–2.5k (cap $3k).
   - No [$] item starts before D0 resolves, and the node is never billed before GO.
5. **The largest single risk is the account rule, and it has already been triggered.** Rule 3.5.a: "You will be disqualified if you make Submissions through more than one Kaggle account" (G-F G17, FC 2). snowiers, your second account, has 8 competition submissions, the latest on 10-17 07:15 (`runs/submissions.csv`, checked today).
   - Stopping now doesn't undo them.
   - Kaggle normally auto-selects an entry's best submissions when it picks none, so snowiers probably reaches the private LB anyway (G0 checks this).
   - D0 recommends writing to Kaggle support today, before the 10-26 entry deadline.
6. **Both final picks default to harness configs.**
   - Pick 1 is the furthest-adopted config in the near-lossless line: mem-repair-10x v1 (F), then fallback+ (F plus the hygiene fixes and the victory lap), then fallback++ (plus R1 and any adopted serving lever).
   - Pick 2 is a different policy, as a hedge. A trunk fine-tune can take pick 2, and only pick 2, through a pre-registered rule.
7. **H2.**
   - Harvest our private scores, which are extra draws on the fixed 55-game private set, and the top teams' releases.
   - Run the node and R3 round 1 if H1 didn't. Build a ≥55-game LB-proxy suite, then climb a gated ladder: iterated on-policy RFT, external-game breakthroughs, frozen-state RL with routing replay.
   - Each step is gated on G0's late-submission answer and on what the releases show. About $15–45k over 3 months; P(consistent 60) ≈ 0.1, mostly conditional on a reproducible top-team release.

## 1. Decisions for the user

| # | Decision | Recommendation | When |
|---|---|---|---|
| **D0 [U]** | **The existing two-account exposure** | Write to Kaggle support now. Say plainly that mellamoa and snowiers are both yours, and ask how to fix it: remove snowiers' entry or its 8 submissions, and confirm mellamoa's standing. Do it before the 10-26 entry deadline. Choosing no snowiers draw as a final is not a fix on its own: Kaggle normally auto-selects an entry's best submissions when none are chosen (G0 confirms). A team merger can't fix it either: rule 3.5.b requires each team member to be a separate individual (G-F G17). No [$] item starts until Kaggle answers or you accept the risk in writing. Until then, Branch C applies | **10-18, first** |
| D1 [U] | Accounts from now on | No more snowiers submissions and no snowiers compute. The rules header says "you cannot sign up to Kaggle from multiple accounts"; compute-only use isn't addressed (G-F G17, FC 2). This only limits new exposure; D0 handles the existing exposure | 10-18 |
| D2 [U] | The no-repeat rule | (a) Confirm that LB redraws of an unchanged notebook are allowed. They use no GPU quota, mem-repair-10x drew 4 times after the rule, and each final needs ≥3 draws. If declined, §9.7's stacked-config calendar applies. (b) Allow offline replicates on rented boxes only (n ≤ 4 per arm) for non-inferiority tests. If declined, U25 is a catastrophe gate only (§7.6) | 10-18 |
| D3 [$] | RTX PRO 6000 boxes | **Azure NC144lds:** 1 box at a time in Branch B. Spot (≈$1.02/h) for restartable work; on-demand ($5.50/h) for scored runs (W6 FC 7). Each box needs 144 vCPU of quota in a new SKU family, and spot quota is a separate request. Branch A needs up to 3 boxes (432 vCPU). Confirm first that the subscription's offer type allows N-series. **Also today:** open a provider that needs no quota approval (Lambda; RunPod or similar for RTX PRO 6000; prices UNVERIFIED), and request GCP g4-standard-48 quota. That is the scoring machine itself (≈$4.50 / $1.63, UNVERIFIED). Caps: $300 (Branch B), $500 for RTX time in Branch A | Request today; use after D0 |
| D4 [$][U] | **Branch A in H1:** the 8×H200 node (ND96isr H200 v5, $84.80/node-h; W6 FC 6) for one R3 round | **Default: no; run it after 11-02 as H2-1's first step.** By the plan's own numbers it is worth ≈0 to pick 1 (§3.2). Opt in only if G0 shows late submissions are closed and every GO condition passes on 10-23. **Quota:** request it today anyway (free). Azure has no published approval time (a vendor estimates 1–4 weeks, UNVERIFIED), and high-demand SKUs may be unavailable to new subscriptions (W6 §7). So P(allocatable by 10-24) ≈ 0.2–0.3. Lambda 8×B200 ($53.52) is a separate path that needs its own sm100 GDN/QSA parity test (G5). **If opted in:** stage the container and checkpoint on persistent storage first, allocate once (10-24 08:00) and deallocate when the exports are done. An idle node costs $2,035/day, and a deallocated ND H200 v5 may fail to reallocate and loses its NVMe. Caps: $1k through G5, $2.5k for the node in H1 | Request 10-18; decide 10-23 12:00 |
| D5 [K] | Kaggle pushes | The ledger in §6: ≤6 GPU-h per Pacific day, with a 6 GPU-h reserve each week | Daily |
| D6 [LB] | LB submissions | The 16-slot calendar in §6. Throttle and information draws are never finals | Daily |
| D7 [K] | Private Kaggle model uploads | The zero-delta overlay, Swift, the R1 head, and FT1 overlays (≈6 GB each; G-D §3) | 10-20, 10-24, 10-25 |
| D8 [U] | Licences | Swift's licence (it adds a US$1M revenue threshold) and the Qwen licence against the CC-BY prize terms, before any Swift draw. The R1 head too: it derives from albucino's drafter, which comes from the RadixArk NVFP4 MTP of Qwen weights. Check it before fallback++ becomes pick 1, along with any FT1 overlay. This affects prizes, not score (G-D Implication 6; EB §4.5) | **10-22** |
| D9 [U] | GPQA canary | Keep GPQA items and outputs out of `runs/` on GitHub; the GPQA terms forbid revealing examples online (G-B FC 12) | 10-20 |
| D10 [U] | Final picks | Apply the §9 rule. Lock the picks on 11-01 once that day's draw completes; confirm by 11-02 20:00 | 11-01 |
| D11 [$][U] | H2 | Whether to continue after 11-02, and its budget (§12). H2-1 spend has two preconditions: G0's answer on late submissions, and H2-0's reading of the Tufa, NVARC3 and Lord Han Solo releases (their base model, and whether they fine-tuned) | 11-03 |

## 2. Day-0 / Day-1 checklist (cheapest gates first)

**Day 0, Sunday 10-18** (mellamoa's Pacific day runs to 10-19 07:00 UTC).
1. **[U] D0 first, then D1 and D2.** Free.
2. **G0 admin (free).** A Sonnet browser agent reads the Kaggle pages and reports:
   - whether the private score is the best of the two picks;
   - whether late submissions are scored;
   - whether a running submission can be selected;
   - whether Kaggle auto-selects an entry's best submissions when it selects none;
   - what happens to snowiers' entry and finals, and whether an entry can be withdrawn;
   - the exact deadline.
3. **Accounts and quota (free):**
   - Azure H200 node quota;
   - NC144lds vCPU quota (on-demand and spot), after checking the offer type;
   - GCP g4-standard-48 quota;
   - Lambda, and one no-quota RTX PRO 6000 provider.
4. **[K] Push E1 first:** the leader (mem-repair-10x) at `demo_minutes` 161, otherwise unchanged (3.0 GPU-h; G-H Q5, FC 11; why 161 and not 242 is in §15).
5. **[K] Logging smoke** (0.3 GPU-h): raw replies, output token ids and logprobs on Pennyroyal (G-C Q4), with disk cost measured. Logging is for offline and data runs only (§4.5).
6. **Laptop, at `nice -n 15` with ≤3 heavy jobs.** Day 0 is cut down to what Day 1 needs:
   - `arm/harness-fixes`: the #176 shims, #18's narrow fix, the #27 deadline flag and a repair-call counter (§7.1).
   - A `private_minutes` builder option with a unit test that asserts the rendered TRUE_SUBMISSION line. This is needed because `kaggle/template.ipynb` hard-codes `max_runtime_s_per_game = 532*60` in that branch, and `build_notebook.py`'s `DEMO_MINUTES` patch changes only the interactive line.
   - The `arms.json` entries `elastic-161`, `throttle-half` and `fallback-plus` (F, the Track H flags and `ARC3_VICTORY_LAP`).
   - The FR-Spec hot-map coverage count over recoverable replies (`hot_tokens_64k.pt`; G-E Implication 6).
   - Archive final-hold-lap-10x into `runs/` and `arms.md`, where it is missing (snowiers, 10-17: offline 55.08 with 9 wins; LB 28.06).
   - Moved to 10-19 and 10-20: the overlay tool, DreamTeam plumbing, the state-lab port and the whole-turn sampler.
7. **[K][LB] The throttle draw T½:** F with the private deadline halved to 266 of 532 min through `private_minutes`.
   - Diff the built notebook against F, then commit it (0.25 GPU-h).
   - Submit by ≈16:00 UTC, so the score (≈4.5–5 h later) is back before the 10-19 slot. Otherwise T½ takes the 10-19 slot.

**Day 1, Monday 10-19.**

8. **Read E1 by its pre-registered rule (§5, G2)** and log the result in `arms.md`. Report wins and time-to-win first.
9. **Read T½ as one input.**
   - ≥17 leans P1, ≤15 leans P3, and anything between is no lean.
   - With one draw against F ≈ 29.5, a P1 truth reads ≤15 about 15–20% of the time, and a P3 truth reads ≥17 about 10% of the time. The P1/P3 gap is only ≈1.6–1.9 draw-sd (G-H FC 12) [I].
   - T½ never decides a go/no-go on its own.
10. **[K] Two runs.**
    - **fallback+ on the public 25** (2.3, raw logging on, as an offline data run). It also counts repair calls per error class, plus actions and deaths per level, for the Track H gate.
    - **Serving bench for #10** (0.7).
11. **[$] Bring up the RTX box,** if D3 is approved and D0 is resolved.
    - **Prove the wheelhouse runs outside Kaggle's image first.** The notebook image (`gcr.io/kaggle-private-byod/...`) can't be pulled elsewhere, and the wheelhouse and its JIT caches depend on the exact Python, torch and nvcc versions (CUSTOM.md §5; W6 §8).
    - **Then serving parity.** Clean decode within 5% of the leader's at the matching batch size: 852.7 tok/s at bs10, 1001.4 at bs14, from `serve.log` (G-E Q2, FC 4). Alternatively, use the same benchmark run on Kaggle that day.
    - A greedy match rate ≥ a base-vs-base repeat's.
    - Accept within ±0.03 of a matched Kaggle 25-min panel (G-B Q7).
12. **Laptop.**
    - Port state-lab from `archive/tooling/state-lab` (300 commits behind `main`) to `tooling/state-lab`.
    - Run G-SL on the mem-repair stack (CPU).
    - Build the overlay tool and its positive-control tests.
    - Add DreamTeam plumbing, with a CPU smoke. `kaggle/build_notebook.py` has no `environments_dir` or manifest support today.
    - Start the whole-turn sampler (due 10-21) and the R1 capture hook (due 10-22).

## 3. Honest odds

### 3.1 LB mean of the final pick at 11-02

**Starting point.**
- F has drawn 38.22, 28.73, 29.54 and 26.90: a mean of 30.85 with se 1.75.
- The 10x family's 19 draws average 29.12 with sd 3.32. That spread is below the within-notebook draw sd of 3.50 (df 7; L3 FC 7), so the family shows no measurable spread between configs.
- Shrinking F toward 29.12 with an assumed between-config sd τ = 1 LB gives ≈29.5 [I]. τ = 0 gives 29.1, and τ = 2.2 would be needed for 30.2.
- The offline route agrees: 54.69 × 0.548 = 29.97, and that is before shrinking a single offline run whose reliability is 0.08 (G-B Q5, FC 8).
- So I use F ≈ 29.5, with a posterior sd of about 1 LB.

**What each lever adds to pick 1** (value if it ships × P(it ships)):

| Lever | LB if it ships, P1 / P3 | P(ships) | Expected | Basis |
|---|---|---|---|---|
| Harness hygiene (#176 shims, #18 narrow fix). Direct saving ≈0.3% of tokens: the repair calls after ImportError and StaleState, which are 22.9% of 12,634 errors, × the 1.26% next-call share. Up to ≈1.7% at issue #18's upper bound | +0.05–0.3 / +0.1–0.5 | 0.8 | +0.1 | G-A FC 7–8; issue #18; 28 points per ln unit (G-H FC 8); ×0.548 under P1 (L3 FC 3). Fixing an error doesn't refund the failed call's reasoning; any gain from less confusion is unmeasured |
| Victory lap (on `main`; can't lower a score) | ≈+0.2 / ≈+0.2 [I] | 0.85 | +0.15 | +0.34 offline per run (11-run mean; `arms.md:568`) × 0.548. The gain scales with wins, not compute |
| Serving levers (#10 flags, #12 m64 GEMM, KV-pool audit) | +1% tok/s ≈ +0.15 / +0.3 [I] | ≈0.3 that any reaches ≥1% [I] | +0.05 | Priced like R1, at 0.548 × 28·ln(ratio) |
| R1: accept +0.15 to +0.25 → harness tok/s +3.4 to +5.6% (+2.25% per +0.1 accept) | +0.5–0.8 / +0.9–1.5; AngelSpec-level upside +1.2–1.4 / +2.2–2.5 | 0.3 | +0.3 | G-E Q2, Q4, FC 2, 8. Priced by mechanism, 28·ln(m_eff), not by sim2, which runs 15–28% steep (G-H FC 6) |
| Trunk fine-tune (R3 or Swift), **pick 2 only** | ≈ −4% of total tokens (≤ −6.3% at the 0.7 floor) → +0.3–0.5 / +0.5–1.0 (≤ +0.8 / +1.5) | ≈0.04 | 0 to pick 1; ≈ +0.02 to the expected best-of-two [I] | §3.2; scaled from EB §1.3's −10% figures |

| Case | Pick-1 mean | How it comes about [I] |
|---|---|---|
| Low (≈10th percentile) | **28** | F's true mean is ≈28.2, one posterior sd below 29.5, and nothing ships, or what ships is neutral |
| Mid | **30** | 29.5 + 0.1 (H) + 0.15 (V) + 0.05 (SB) + 0.3 (R1). Within noise of F |
| High (≈90th percentile) | **32.5** | F ≈ 30.8, P3 holds, R1 ships at its P3 central gain (+1.2), V +0.3, H and SB +0.3 together |

- **The trunk is not in this table.** It can only take pick 2 (§9). It changes the expected best-of-two, by about +0.02.
- **The expected best-of-two private score** is about 2 points above the picks' per-draw mean when the two are similar (σ/√π = 1.97; G-B FC 10).
- **Listed separately: disqualification (D0).** It can't be estimated from our evidence. At P(DQ) ≥ 3%, its expected cost (≥ 0.03 × 30 ≈ 0.9 LB) exceeds every lever above combined (≈ +0.6).
- **Also listed separately:** a final that never completes (§14).

### 3.2 Why a fine-tuned trunk is unlikely to beat the fallback by 11-02

**What a trunk round is worth even if it works [I].**
- The recipe trains only cleared on-route turns of ≥5k tokens. Those are 30.3–30.6% of leader tokens and 27.2% over all runs (G-A FC 1–2).
- It keeps a 0.7 floor and a 30% median-length anchor. So the projected total cut is 0.30 × 0.7 × (1 − r̃): at most 6.3% at the floor (r̃ = 0.7), and ≈4% at r̃ ≈ 0.8.
- That is ≈ +0.3–0.5 LB under P1 and +0.5–1.0 under P3.
- More would need spill-over onto untrained states, which is unmeasured (G-A Implication 1).
- EB §1.3's "−10%" assumed a −25% cut on all 41% of on-route tokens. The recipe can't reach that.

**The chain for our own R3 round in H1**, if you opt in [I]:

| Step | P |
|---|---|
| The Azure node can be allocated by 10-24. There is no published approval time (W6 §7). Lambda is a separate path with sm100 risk | 0.2–0.3 |
| E1 is valid and not falsified (≈13–15% false falsification; §5 G2) | 0.75 |
| The whole-turn sampler works by 10-21, and G3 passes all three conditions as fixed in §5. The only local datum already fails (b) at hard states | 0.2–0.3 |
| About 400 states rebuild with fidelity, giving ≥200 selected turns by 10-24 12:00 (§7.6) | 0.6 |
| The stack trains at the deployed shape (the recipe is measured only to 8k, text-only; W6 FC 2, 4) | 0.6 |
| The checkpoint moves toward its targets and passes every screen (Arora's RS+SFT collapsed AIME 0.563 → 0.183; G-G FC 1) | 0.4 |
| Its true LB mean is higher | 0.6 |

**The total.**
- The R3 product is ≈0.004, or ≈0.005–0.01 allowing for correlated steps.
- The Swift-1.5 overlay adds ≈0.03–0.04 [I]: promoted 0.15–0.2 × full run holds 0.5–0.6 × truly better ≈0.4 × three clean draws on time 0.8. I put P(truly better) below 0.5 because the only in-harness datum points the other way. Felix's 10-game screen, of unequal length, had Swift at 7 levels against 13, with decode 4–8% faster and tokens per action −4% (G-D FC 7).
- **The total for a trunk change is ≈0.04.**

**No public result raises that prior.**
- Every public notebook scoring ≥20 runs stock Flash-Next (L2 FC 6).
- Two H200 users and two others report no gain from SFT or RL on Flash-Next. These are forum posts with no scores (W5b FC 1).
- The Tinfield probe was NO_GO (G-D FC 8).
- Even a truly better fine-tune is barely detectable: +5 LB needs about 6 draws per arm (G-B FC 4).

So Branch A can't pay for itself in H1. Its value is the pipeline proof and the measured result it leaves for H2.

## 4. Ground rules

1. **mellamoa only (D0, D1).**
   - The selectable fallbacks are mem-repair-10x v1 (mellamoa draws 38.22 and 26.90) and mem-unified-10x v1 (32.23 and 29.59).
   - snowiers draws stay valid as noise data. Not picking them doesn't remove the exposure (D0).
2. **The fallback is frozen.** Each trained artifact gets its own notebook and flags.
3. **No billing before D0 resolves and the gates pass.** Every kill has a date and every paid step a dollar cap. A missed date counts as a fail for the track that missed it, and only that track.
4. **Every LB candidate completes a clean offline full run first.**
   - The Z draws (fallback+ through the overlay loader) are comparators, not finals. G1 and U25-F+0 cover them.
   - Offline runs report wins and time-to-win first, then score, then `make significance` against the comparator.
5. **Logging flags.**
   - Raw replies, token ids and logprobs go on in offline and data runs only.
   - LB notebooks and their matched comparators keep them off. A competition rerun's outputs can't be retrieved, so logging there adds disk and throughput risk for no data. And a 0.3 GPU-h smoke can't resolve a <1% cost.
   - `ARC3_VERBATIM_HISTORY` stays out of finals: its only run scored 42.09 against 50.62, confounded (G-C FC 10–11).
6. **Matched comparators.** A comparator has the same harness, logging and overlay-loader flags as its candidate.
   - FT1 and Swift are compared only with loader-flagged base draws (Z1–Z3).
   - fallback++'s U25 run carries the same loader flag as U25-F+0.
   - A mean pooled over different configs is not a comparator.
7. **Claude writes infrastructure only, never training targets or game content** (W10 FC). All H1 targets are the model's own engine-verified samples.
   - In H2, environments generated by frontier models are evaluation-only unless you clear them. DreamTeam's games, for example, were made with GPT-5.5, Gemini or Opus (L2 FC 7).
   - Each pool's authorship is recorded before training on it.
8. **Identity gates are relative.** Every "greedy identical" check is a match rate at least as high as a base-vs-base repeat's at the same batch size. Verify-versus-decode kernels and the FP8 KV cache make exact identity fragile.

## 5. The H1 gate ladder

| # | Gate | Cost, where | Pass (pre-registered) | A fail kills | By |
|---|---|---|---|---|---|
| G0 | Rules and admin | Free | The questions in §2 item 2 are answered, including snowiers' auto-selection and withdrawal; quota confirmed or refused | Nothing; it sets D0's request, the branch and the pick rule | 10-19 |
| G1 | Zero-delta overlay boot with a positive control, plus the Swift panel (a 25-min pair; the zero-delta panel doubles as the same-day base) | 1.6 GPU-h, Kaggle [K] | **Zero-delta:** decode within ±2% and accept within ±0.03 of base; greedy match rate ≥ the base-vs-base repeat's; every overlaid tensor's `get_weights_by_name` hash matches the manifest, and a deliberately perturbed overlay (one tensor scaled) changes greedy output. "Last duplicate wins" loading is unverified in Pennyroyal (G-D §3), so the loader must be shown to read the overlay. Startup ≤ matched base + 2 min, and notebook start → first environment action ≤ 12 min (Kaggle requires an action within 15 min; template note). Build the overlay from symlinks plus new shards, or after the warmup RESET, so it can't block the game cell. A 6 GB model version uploads, with its processing time recorded. **Swift:** kill if decode < 0.95×, accept < 0.9× or parse errors > +2 pp. Promote to one full run if tokens/action ≤ 0.85× and levels/action ≥ 0.95× (G-D §4). A 25-min panel correlates only 0.21 with final rank (G-B Q5), so promotion buys a full run, nothing more | Zero-delta fail → every trunk-weight path in H1 (Swift and FT1). Swift fail → the Swift arm | 10-20 |
| G2 | E1 elasticity: the leader at 161 min | 3.0 GPU-h, Kaggle [K] | The tail fade scales with run length (fraction 0.2; `tool_agent.py:3119-3133`), so E1's final includes its own fade, like REF's. **Validity:** E1's S(97) within REF's S(97) ≈ 44.7 [I, interpolated] ± 6.5 (2 sd). That point is before E1's fade, which starts at 128.8 min, and at REF's fade start. **Falsified** if S_final(161) − 51.57 < +3.5 (diff sd ≈ 3.3; ≈13–15% false falsification if the truth follows the exp path's +7.0; G-H FC 3, 11). **Supported** if ≥ +6. The 242-min ceiling moves to H2 | Invalid or falsified → no node and no R3 in H1; R1 and serving levers continue only under $50 | 10-19 |
| G-SL | State-lab fidelity, for every config that supplies states | Laptop CPU (`analysis.state_lab.validate`, nice 15), free | On ≥100 states per config: ≥95% of rebuilt boards and actions identical. ≥95% of tool results byte-identical, with every mismatch classified benign (e.g. timing text) or not; any non-benign mismatch fails. The precedent: pf-solved matched only 13 of 14 (L1 FC 7). Rebuilt follow-up requests (the 2nd and later requests of a turn, rebuilt from harness memory state under mem-repair's flags) match the logged next request in text and pixels on ≥95 of 100 | All R3 data from that config; every frozen-state screen if it fails on the mem-repair stack | 10-20 (mem-repair stack); 10-22 (other configs) |
| G3 | R3 yield: 100 T15 states from configs that passed G-SL (on-route, cleared levels, logged turn ≥5k tokens, no level-first or post-death turns), k = 8 whole-turn samples at served settings | ≈800 samples ≈ 3.3–4.0 GPU-h at ≈200–240 per GPU-h [I] [K] | **All three must hold.** (a) ≥50% of states have ≥2 verified siblings. (b) **Within each state**, rank its 8 samples by total turn tokens. Pooled over states, the verified rate in each state's shortest tercile is ≥0.8× its longest tercile's. Within-state ranking stops easy states posing as short ones. The only local datum, at hard states, is 10.4% vs 13.2% = 0.79, already below the bar (W7 FC 7); easy states are unmeasured. (c) **Projected total cut ≥4%** = S × (1 − a) × (1 − r̃), with S = 0.30 (the leader's share of ≥5k cleared on-route tokens; 0.27 over all runs; G-A FC 1–2), anchor share a = 0.3, and r̃ = median selected / median verified. This needs r̃ ≤ 0.81. The 0.7 floor caps the cut at 6.3%. **Also recorded:** tokens per sibling, calls per turn and the measured samples per GPU-h, which size R3 | Branch A in H1; R3 moves to H2 | **10-22.** No result on 10-22 → Branch B automatically |
| G-SV | Screen validation | RTX spot (≈8 box-h ≈ $8), or spare Kaggle hours | Base, a base repeat and accept05 on the 450 H10 states, plus k = 4 on 50 stuck states. The base-vs-base repeat sets the CIs and the canary thresholds (§7.6). The screen must flag accept05 as worse (−6 offline and −8.6 residual; G-H Q1) | Branch A in H1: a screen that misses accept05 can't approve FT1 | 10-22 |
| GO | Branch A go/no-go (opt-in) | — | All of: D0 resolved; you opt in [$][U]; G1 zero-delta, G2, G-SL, G3 and G-SV pass; the RTX box has passed parity; the node can be allocated by 10-24 08:00. I recommend opting in only if G0 shows late submissions are closed. T½ never decides this alone | Branch A in H1; it becomes H2-1's first step | **10-23 12:00** |
| G4 | R1 capture: albucino's head reproduces its served acceptance on captured rows | 0.5 GPU-h, RTX box (Kaggle in Branch C) | Offline E[accept] within ±0.05 of 2.68 (G-E Q3–Q4), reported separately for raw-logged and normalised rows | R1 | 10-22 |
| G5 | Node smoke on the first real training rows (Branch A) | ≤10 node-h [$], cap $1k | A ≥32k multimodal LoRA step fits at CP2/CP4. A 100-step throughput run: the 3–10% MFU behind "2–7 node-h per 100M tokens" is unverified until this step (W6 §3). Packed vs unpacked GDN loss parity. A zero-LoRA merge → overlay that is byte-identical to the base tensors (G1 already proved that path serves). On Lambda B200 only: a GDN/QSA forward/backward parity test (sm100; fla had sm100 bugs; W6 §8) | The trunk prong in H1 | **10-24 22:00, or $1k of node spend** |
| G6 | FT1 screens (§7.6) | ≈5–6.5 GPU-h over 10-25 and 10-26 [K] | Integrity, G-MTP, canary, and the frozen-state screen on H10 states, which G-SV has validated | That checkpoint | 10-26 23:59 |
| G7 | U25 catastrophe or non-inferiority test (§7.6) | 2.3–4.6 GPU-h [K], plus replicates [$] | §7.6 | FT LB draws | 10-27 23:59 |
| G8 | LB confirmation | 3–4 slots [LB] | §9 | FT as pick 2 | 11-01 |

§15 gives the reasons for E1 at 161 min and for the three-part G3.

## 6. H1 calendar, 10-18 → 11-02 (Branch B is the default; "A:" marks the opt-in changes)

**How to read the ledger.**
- The Kaggle column is mellamoa GPU-h per **Pacific day** (the row's date 07:00 UTC → next date 07:00 UTC), counted from push to completion. It includes ≈0.25 GPU-h to commit each new LB notebook (efficiency-first §3).
- **Week 1:** 40.9 GPU-h are left until 10-24 00:00 (`kaggle quota`, read 10-18 in eval-centric).
- **Week 2:** 45 GPU-h, from 10-24.
- Kaggle allows at most 2 GPU sessions at once.

| Date | Kaggle (GPU-h) | RTX box / node / laptop | LB slot [LB] (≈00:10 UTC from 10-19) | Decision |
|---|---|---|---|---|
| **Sun 10-18** | E1 at 161 min (3.0), logging smoke (0.3), T½ commit (0.25). **3.55** | Laptop: the Day-0 list (§2) | **T½ throttle draw**, by ≈16:00 | **D0**, D1, D2 |
| **Mon 10-19** | fallback+ public-25, raw logging on (2.3); serving bench #10 (0.7); fallback+ LB commit (0.25). **3.25** | [$] RTX bring-up, wheelhouse test and parity (after D3 and D0). Laptop: state-lab port; G-SL on the mem-repair stack; overlay tool; DreamTeam plumbing and CPU smoke; whole-turn sampler (to 10-21); R1 capture hook (to 10-22) | T½ if it missed 10-18; otherwise final-hold-lap-10x copied to mellamoa (D2a; it has a clean offline run); idle if D2a is declined | Read G2 and T½; Track H/V gate when fallback+ ends (≈09:30) |
| **Tue 10-20** | G1 pair plus the perturbed-overlay boot (1.6); DT25 base (2.3) if the CPU smoke passed; Z commit (0.25). **4.15** | RTX spot: two raw-logged leader-stack runs (≈$5), giving R1 rows and R3 states. Laptop: canary build (GPQA kept local, IFEval scored offline); sampler | fallback+ #1 | G1; G-SL (mem-repair); D9 |
| **Wed 10-21** | U25-F+0 = fallback+ with the zero-delta overlay flag on U25 (2.3); Swift full run if promoted (2.3); commit (0.25). **4.85** | RTX: R1 capture. A: comparator U25 replicates start (D2b, on-demand box); dequantised checkpoint on a CPU VM. Laptop: arc-interactive integration (to 10-24); sampler done | Z1 = fallback+ #2 through the overlay loader, which tests the 9-h path | — |
| **Thu 10-22** | G3 (3.3–4.0); serving bench #12 if built (0.7); commit. **≤4.95** | RTX: G4; R1 training (2–8 GPU-h); G-SV and the canary base-vs-base run on spot (≈8–10 box-h) | Z2 | G3, G-SV, G4; G-SL (other configs); D8 |
| **Fri 10-23** | State-lab sampling: R3 data, or the H2 bank in Branch B (5.0). **5.0** | R1 offline selection. A: sampling on 2 spot boxes (≈10–15 box-h); node pre-staging (no meter) | Swift #1 if its full run held. A: Z3 instead, if you opted in and G3 passed by 10-22 18:00 | **GO 12:00** (A) |
| Week 1 total | **≈25–26** of ≤36 usable (40.9 left) | | | |
| **Sat 10-24** | R1 G-MTP panel on U25 (0.7); fallback++ public-25 (2.3); fallback++ on U25, loader flag on (2.3); commit. **5.55** | A: node at 08:00 → G5 on the first rows; dataset gate 12:00; training | Swift #2 (A: Swift #1) | R1 gates; A: dataset gate 12:00, G5 22:00 |
| **Sun 10-25** | A: FT1 G-MTP (0.7), canary (1.5), commit. **2.45.** B: H2 sampling (≤4) or reserve | A: movement check, ≤2 λ merges, export, upload [K], **deallocate by 08:00**. C1 dress rehearsal on the on-demand box under scoring-box limits (≈$52) | C1 #1 = fallback++ if the R1 gates passed; otherwise fallback+ through the loader | — |
| **Mon 10-26** | A: λ screens (≤2 λ, ≈3–4) + commit. **≤4.25.** B: reserve | A: comparator replicates finished | C1 #2 | **G6 23:59** (A) |
| **Tue 10-27** | A: FT1 U25, one run per surviving λ (2.3–4.6) + commit. **≤4.85.** B: reserve | A: FT1 U25 replicates (D2b, on-demand, 9.2 h) | C1 #3 | **G7 23:59** (A) |
| **Wed 10-28** | H2 data runs on external games (≤4.0, logging on) if the reserve allows | A: FT1 dress rehearsal (on-demand, ≈$52) | B: Swift #3. A: FT1 #1 | — |
| **Thu 10-29** | Same | — | B: Z3 (Swift's third comparator draw) or C1 #4. A: FT1 #2 | — |
| **Fri 10-30** | Same | Optional DT10 single-use panel for the finalists (≈$13 per run) | B: a stacked variant with a clean offline run, or a top-up. A: FT1 #3 | Δ* computed |
| **Sat 10-31** | Hold a 6 GPU-h reserve (new quota week) | — | B: top-up. A: Swift #2 | — |
| **Sun 11-01** | — | — | Top-up for the config nearest a threshold (A: Swift #3) | **Picks locked** once this draw completes (≈09:30) |
| **Mon 11-02** | — | — | Optional information-only draw: never a final; its private score feeds H2-0 | Confirm the picks by 20:00 |

**Totals.**
- **Kaggle:** ≈31 GPU-h in Branch B (≤45 with optional H2 sampling) and ≈42–50 in Branch A, with ≥6 GPU-h of reserve each week.
- **LB:** all slots from mellamoa. A slot is idle only when D2a is declined and no new clean config is ready (10-19 at most).
- **Engineering days,** roughly in order: Track H, Track V and the builder option (Day 0, ≈1 day); the overlay tool and DreamTeam plumbing (10-19/20); the state-lab port and the whole-turn sampler (10-19 → 10-21); the R1 capture hook (2–4 days, 10-19 → 10-22; G-E Q4); the canary (10-20/21); arc-interactive (10-21 → 10-24).

## 7. The tracks

The tracks run in order of expected value per engineering day [I]:

| Track | Engineering | Expected effect | Order |
|---|---|---|---|
| H: harness hygiene (§7.1) | ≈1 day | +0.1 to pick 1 | 1 |
| V: victory lap (§7.2) | Flags on `main`; hours | +0.15 | 1 |
| SB: serving levers (§7.3) | #10 hours; #12 a kernel-table build (≈1–2 days) | +0.05 | 2 |
| R1: MTP head (§7.4) | 2–4-day hook, plus training | +0.3 | 3 |
| S: Swift overlay (§7.5) | The overlay tool (≈1 day, shared with A) | ≈+0.01 to the best-of-two | 4 |
| A: one R3 round (§7.6) | State-lab port, sampler (3–4 days), canary, screen, node operations: ≥6 days | ≈0 to pick 1; ≈+0.01 to the best-of-two in H1 | 5, opt-in |

A slip kills only the track that slipped. H and V depend on nothing else, and they are built first.

### 7.1 Track H: harness hygiene (part of fallback+), 10-18 → 10-22

**What it's worth.** This is cheap, low-risk hygiene, not a score lever: about +0.1 LB expected (§3.1).
- Fixing an import or a guard doesn't refund the failed call's reasoning. It saves the repair call.
- Track H's classes are ImportError (8.1%) and StaleStateActionError (14.8%), together 22.9% of the 12,634 errors.
- The call after any error costs 1.26% of tokens (G-A FC 7–8), so the direct saving is ≈0.3% of tokens.
- Issue #18 puts stale-state at up to ≈1.7%, an upper bound measured on an older stack.
- "Less confusion from false messages" is plausible but unmeasured.

**Work** (`arm/harness-fixes`, each change behind its own default-off flag):
- **#176: imports as inert shims, not a whitelist.**
  - Why not a whitelist: the child bootstrap already imports `os`, `sys` and `builtins` (`python_tool_sandbox.py:27-34`), and `_safe_import` checks only the root module name (`:426-430`). Whitelisting `builtins`, `sys`, `inspect` or `pickle` would let model code reach `os`, the host protocol globals and the guard state, for example through `builtins.__import__('os')`, `sys.modules` or `inspect.stack()`. That would happen during the 9-h rerun, where #175's sandbox limits are still open.
  - Start from `ARC3_SANDBOX_SHIMS` (PR #76, tag `archive/arm/sandbox-robust`), with two changes. `time` becomes a shim with no `sleep` and a step counter in place of a wall clock, because the model must never see minutes. `sys`, `inspect`, `builtins` and `pickle` import successfully but expose nothing usable.
  - Unit tests show that `sys.modules`, `builtins.__import__` and `inspect.stack()` can't reach `os`, the RPC handles or the guard state.
  - The PR and `CONFIGURATION.md` name all of this.
- **#18: the issue's own narrow fix.**
  - The stale-state guard is Franzen's intentional design (`WRITEUP.md:285`), and #18 is labelled MAYBE: "A/B on action savings before turning it off".
  - So clear `_stale_state` when StaleStateActionError is raised, so the catch-and-retry the message promises actually works. Alternatively, block only a repeat of the same action.
  - The guard stays on.
- **#27: the deadline fix,** behind its own flag.
  - Derive the private deadline from notebook start + 32,400 s − margin, read from `/proc/self/stat` as the victory lap's kill guard already does.
  - mem-repair-10x's nominal end was 32,513 s against the 32,400 s limit (#175).
  - It goes into every config that changes startup or weights. The rest of #175 stays out (§13).
- **`finish=length` handling** (6.8% of tokens), only if a design exists by 10-18 night. Otherwise it moves to H2 (G-A Q4, Implication 5).
- **A repair-call counter per error class,** so the saving is measured rather than assumed.

**Gate** (the fallback+ public-25 run, 10-19):
- wins and time-to-win reported, and no new error class;
- ImportErrors on shimmed modules fall to 0, and StaleState errors per 1k calls fall by ≥50%;
- actions per cleared level, deaths per level and RHAE on cleared levels are no worse than F's, compared paired per level against F's logged action counts. A weaker guard can spend actions;
- score ≥48 (pool 51.17 − 1 sd; G-B Q1);
- `make significance` shows no per-game loss against F.

**Adoption over F:** ≥3 error-free draws with a mean ≥ F − 2. Loader-flagged draws count (§9.1). This is a crash and non-inferiority check only.

**Kill:** any miss by 10-22 → F stays pick 1. Close #18 and #176 once the PR records a clean result.

### 7.2 Track V: the victory lap (part of fallback+), 10-18 → 10-22

- **What it does.** `ARC3_VICTORY_LAP` is on `main` (`CONFIGURATION.md:822`). After a win, a harness thread with no LLM replays each level's winning attempt, and the official scorer keeps the best play.
- **Why it's safe.** A gate requires play 2 to finish with no level longer than in play 1, or not to start. So it can't lower a score under max, mean or last-play scoring. This was CPU-checked against arc_agi 0.9.9's CompetitionArcadeServer (`arms.md:568`).
- **What it's worth.** Offline +0.34 per run (11-run mean) → ≈ +0.2 LB [I]. Its kill guard already counts down from notebook start.
- **Final hold stays out.** The lap ran once together with `ARC3_SCHED_FINAL_HOLD`, as final-hold-lap-10x: offline 55.08 with 9 wins, LB 28.06 on snowiers (not selectable). Final hold changes scheduling, so it isn't lossless and stays out of fallback+. It can fill a spare slot as a stacked variant (§9.7).
- **Gate.** In the fallback+ run, `[VICTORY_LAP]` lines appear only after wins, `official` ≥ play 1 on every game, and the kill guard cuts no lap.

### 7.3 Track SB: serving levers (into fallback++), 10-19 → 10-24

- **The levers.** All are lossless and none has been benched (`arms.md` backlog, moved from issues #10 and #12):
  - the chunked-prefill, mixed-chunk and prefill-decode-interval flags (#10);
  - a low-M GEMM entry for the M = 56 decode shape at 14 streams (#12; `_MAX_M = 32`);
  - a KV-pool audit.
- **The test.** Each lever gets a fixed-prompt A/B serve bench (≈0.7 GPU-h on Kaggle, or on the RTX box after parity). It is adopted on ≥1% clean tok/s at matched batch size, with a greedy match rate ≥ the base-vs-base repeat's. Priced like R1, +1% ≈ +0.15 LB [I].
- **Order.** #10 on 10-19 (flags only); #12 on 10-22 if the kernel-table build is done; the audit on the laptop. Levers that pass join fallback++ behind their own flags and ride its gates.

### 7.4 Track R1: the MTP drafter head (fallback++ = fallback+ plus R1 plus adopted SB levers), 10-18 → 10-27

- **Recipe** (G-E Q4):
  - Train albucino's dense tensors with the hyper-connection input mixer [320, 10240] frozen in v1. That leaves ≈70.8M of the 90.6M trainable, and the cached 2,560-wide features stay at ≈26 GB per run (G-E Q3). Training the mixer would need ≈4× the disk.
  - Keep the INT4 g32 experts byte-identical.
  - 3 chained steps with shared weights; CE plus top-K KL on the 64K FR-Spec vocab; LR 1e-5 to 5e-5.
  - Export the 31 tensors with the same names, shapes and dtype.
- **Data split.**
  - Train on **raw-logged T15 transcripts**: the fallback+ public-25 run (10-19) and two leader-stack spot runs (10-20). Normalised logs are extra data only, because the leader's history is normalised: 0 of 468 history messages keep a blank line, against 486 of 498 in a verbatim run (L1 FC 2).
  - Judge on H10 and DreamTeam, which the drafter has never seen.
  - Never gate R1 on accept or throughput from a run that contains its training games.
- **Host.** Capture and training run on one host.
  - Use the RTX box (<40 GB needed).
  - On Kaggle (Branch C only), first run a 1 GPU-h capture-and-disk test on 10-20, and write features to `/`, not `/kaggle/working`, which holds only 19.5 GB (leader run census).
  - Cost: 10–15 GPU-h; $10–15 spot or $55–85 on-demand.
- **Gates:**
  - G4, reported separately for raw-logged and normalised rows;
  - offline held-out E[L] ≥ albucino + 0.10;
  - G-MTP panel on U25 games: drafter SHA check, a greedy match rate ≥ the base-vs-base repeat's, accept ≥ comparator + 0.08, clean decode ≥ 1.02×, step time within ±1 ms (G-E Q5);
  - the fallback++ public-25 run is error-free and in the pool band. This is a crash check only, since T15 is in it;
  - the fallback++ U25 run, with the same loader flag as U25-F+0, is error-free, with harness tok/s ≥ +2% over U25-F+0.
- **Adoption** (eval-centric's mechanism rule). The prior gain is 0.548 × 28·ln(m_eff), with m_eff the U25 harness tok/s ratio. fallback++ replaces fallback+ as pick 1 after ≥3 error-free draws with a mean ≥ fallback+ − 2, a clean dress rehearsal, and D8 cleared for the head.
- **Kills:**
  - G4 fails on 10-22 → stop;
  - no +0.10 held-out gain by 10-24 12:00 → out of H1;
  - not through G-MTP by 10-27 → its slots go to other configs;
  - E1 falsified → continue only under $50.
- **FR-Spec hot map** (a free check, 10-18). If more than 1–2% of generated tokens fall outside the 64K map, an ARC-derived hot map joins fallback++ behind its own flag. Turning FR-Spec off cost −5.4% decode (G-E Implication 6).

### 7.5 Track S: the Swift-1.5 overlay, 10-20 → 11-01

- **Shape.** Swift's BF16 non-expert tensors (5.82 GB) overlaid on our Intel experts, with no requant, stacked on fallback+ (G-D §1, §3).
- **Steps.**
  1. Panel gates in G1.
  2. One full run on 10-21.
  3. If the run lands in the pool band with no per-game loss, three draws, so that §9's ≥3 can be met: 10-23, 10-24 and 10-28 in Branch B; 10-24, 10-31 and 11-01 in Branch A.
- **Licence first.** D8 is cleared before the first draw (10-22).
- **Comparator and pick.** The comparator is the loader-flagged fallback+ draws (Z1–Z3). Swift can take pick 2 by the §9 rule.
- **Tinfield is not run.** Its deltas are tiny and its probe was NO_GO (G-D FC 5, 8).

### 7.6 Branch A: one R3 round (FT1), opt-in (D4), 10-19 → 10-27

**Status.** By default this is H2-1's first step after 11-02. It runs in H1 only after GO on 10-23 12:00. The state-lab work, G3 and the sampling run in Branch B too, on free Kaggle hours, as the H2 data bank.

**Data** (10-19 → 10-24 12:00):
- **States.** T15 on-route states on cleared levels: logged turn ≥5k tokens, no level-first or post-death turns, from configs that pass G-SL. Whole-turn samples at the served settings: T 0.65, top-p 0.95, top-k 20, xhigh effort (G-E FC 5).
- **Supply** (counted today in `scratchpad/ga/ana_steps.json`, before post-death turns are removed [I]).
  - The corpus has ≈2,857 such states, 1,514 of them in the 10x family. They are spread over 30 runs and roughly 25 configs.
  - Only 64 are on mem-repair-10x v1, 43 on mem-repair2 and 51 on stall-fix.
  - Many of the other configs exist only on `archive/*` tags. Contexts from their old prompts would bring back the context mismatch the recipe avoids (G-G Q1).
  - So states come from the deployed stack only: those three runs (each one must pass G-SL), E1 (the leader at 161 min), the fallback+ public-25 run (10-19), and the two raw-logged leader-stack spot runs (10-20; ≈40–60 states each [I]).
  - **Target: ≈400 states.**
- **Volume and rate.**
  - ≈400 states × k = 8 ≈ 3,200 samples.
  - Turns ≥5k average ≈11.7k tokens: 15,969 such turns hold 69.3% of 269.6M tokens (G-A Q1). At 2.5–2.9M tokens per sampling GPU-h (L1 FC 6), that is ≈200–240 samples per GPU-h if siblings are as long as the logged turns [I]. The earlier 330–440 used the 4,411-token mean over all turns.
  - Total: ≈13–16 GPU-h. G3's 800 samples run on Kaggle on 10-22, ≈1,100 on Kaggle on 10-23, and the rest on 2 spot boxes on 10-23 (≈10–15 box-h ≈ $10–15).
  - G3 measures the real rate. If it comes in below 200, the pre-registered cut is k = 6 first, then fewer states.
- **Verification.** The next board's hash must match the logged step's, or the step must land on the BFS hindsight route. No death, and every intermediate frame passes (W7 §5).
- **Selection** (the corrected G-G (a)):
  - per state, keep the shortest verified sibling whose total turn tokens are **≥0.7× the state's median verified**; otherwise the next shortest that clears the floor;
  - skip states with <2 verified siblings;
  - add 30% median-length verified siblings as a length anchor;
  - no generic replay.
- **Projected cut.** It fills G3(c), G6 and G7.
  - P_s = (1 − a)(1 − r̃) on trained-type states, and P_t = 0.30 × P_s overall.
  - At r̃ = 0.81: P_s = 13.3% and P_t = 4.0%. At the 0.7 floor: 21% and 6.3%.
  - Both, and the success lines below, are recorded in `arms.md` before training.
- **Thresholds calibrated on public games** (named in the PR): the 5k stratum cut, S = 0.30, the 0.7 floor, the anchor share, and every G3 and screen threshold.
- **Dataset gate.** ≥200 selected turns (≈800+ call examples [I]) by 10-24 12:00; otherwise don't train.

**Trainer checkpoint** (a CPU VM, 10-21/22, before the node; ≈$20–50).
- BF16, with the routed experts set to dequant(Intel int4), so the trainer's weights equal the served ones.
- Check several dequantised experts against the served Marlin int4 output on the RTX box, using the same input and bf16 tolerance.
- If it fails, Branch A stops. Training against different weights would bring back the trainer–server mismatch.

**Node window** (10-24 08:00 → 10-25 08:00 at the latest; ≤24 node-h ≈ $2.0k).
- The container is prebuilt and the checkpoint staged on persistent storage before allocation. The 28 TiB of local NVMe is scratch and is wiped on deallocation (W6 FC).
- G5 runs on the first real rows (≤10 node-h, $1k).
- Training starts after the 12:00 dataset gate, followed by the movement check, the merges and the export. Then deallocate.

**Recipe:**
- **Trainer:** ms-swift Megatron, pinned to ms_swift 4.6.0, mcore_bridge 1.6.5, megatron-core 0.16.1 and transformers 5.19.0 (W6 §1).
- **LoRA:** r16 on GDN in/out, QSA qkv/o and the shared expert. Never the routed experts, the router or the QSA indexer (G-D Implication 3).
- **Optimisation:** LR 2e-5 (band 1e-5 to 3e-5), global batch ≤32, one epoch, aux loss 0 (G-G (d)).
- **Examples:** one per call, on the served post-trim context, with loss only on that call's reply. Use the four exporter transforms, `qwen3_5` and `enable_thinking true` (G-C Q1).
- **Size:** ≈1,000 call examples × 57k ≈ 60M tokens ≈ 1–4 node-h, at the unverified 3–10% MFU (W6 §3) until G5 measures it.
- **Short-window fallback.** If G5 shows only ≤16k fits, train on recent-context windows. The 57k frozen-state screen must then also show a right-move rate ≥ base − 1 pp. AWM's context-mismatch risk is real (G-G Q1; §15).
- **Movement check** (on the node, before export). On held-out T15 sibling pairs, generated length or NLL must move toward the short siblings. ≈30–60 optimiser steps (≈1,000 examples at a global batch of 16–32) at LoRA LR 2e-5 may barely move the policy [I]. If it barely moved, try λ up to 1.5 and one alternative LR (3e-5) within the node cap. If it still hasn't moved, skip the Kaggle screens: FT1 = no change.
- **Merges:** at most 2 λ values (default 0.75 and 1.0; 1.5 if movement was small).

**Deploy gates** (G6 by 10-26 23:59; G7 by 10-27 23:59):
- **I0 integrity.** Overlay tensors byte-match the merged checkpoint, and untouched tensors are byte-identical.
- **G-MTP** (G-E Q5): greedy spec-on/off match ≥ the base's own rate − 5 pp; accept ≥ base − 0.05; clean decode ≥ 0.98×; step time within ±1 ms. **An accept drop of more than 0.05 kills FT1 in H1.** An R1 retrain on the FT trunk needs capture, training, upload and a panel, which doesn't fit in a day, so it moves to H2.
- **Canary.** Built on 10-20/21 and scored offline on the laptop; GPQA stays off GitHub (D9).
  - The states are 300 stuck states with ≥80k context, where `finish=length` concentrates: no-tool calls are 1.1% of calls but 6.5% of tokens (G-A Q4). So the pooled 1.13% base rate understates the false-fail rate.
  - Each threshold is therefore the 97.5th percentile of G-SV's base-vs-base difference on the same states. The old values act as limits it can't loosen: >8 of 300 with no tool call, +2 pp malformed calls or sandbox errors, −6 pp GPQA-100, −5 pp IFEval-150.
  - GPQA-100 has a run-to-run sd of ≈3 pp (difference sd ≈4.2), so a flat 6 pp line would fire falsely ≈8% of the time per λ [I].
  - A GPQA or IFEval miss steps down to the smaller λ.
- **Frozen-state screen,** on 450 **H10** states (300 easy on-route, 150 stuck), paired by state. G-SV validated it before GO. **Pass:**
  - tokens per verified decision on easy states ≤ 1 − 0.5 × P_s (≈0.93× at P_s = 13.3%), paired one-sided p < 0.05;
  - right-move rate ≥ base − 3 pp (easy) and ≥ base − 2 pp (stuck);
  - calls per turn ≤ base;
  - distinct actions across k = 4 on 50 stuck states ≥ 0.9×;
  - no-tool-call rate ≤ base + 0.5 pp;
  - verification markers per 1k tokens ≥ 0.9×.
- **Cost at G3's measured rate.** G-MTP 0.7 and canary ≈1.5 on 10-25; screens ≈1.5–2.0 per λ on 10-26.
- **U25 (G7).**
  - **Metric, defined first:** H10 RHAE, plus DT15 levels cleared and levels per token. DreamTeam baselines are solver lengths (median 145 actions against 638 on the public games; L1 FC 8), so RHAE is compressed there.
  - **DT15** is chosen by level depth reached and early-level clears in the DT25 base run (10-20).
  - **On Kaggle:** one run per surviving λ against U25-F+0 (loader-flagged). Kill on any of: below the comparator − 5; any error; tokens/action ≥ 1.0× (it moved the wrong way); or a residual < −5 against U25-F+0's own within-run timeline at matched tokens.
  - **Success** is pre-registered as tokens/action ≤ 1 − 0.5 × P_t. At n = 1 this is a catastrophe gate only: U25's own sd is unknown until the replicates.
  - **If D2b allows:** 4 runs of the best λ against 4 comparator runs on the RTX on-demand box (≈$100). The comparators run 10-21 → 10-24, since they don't depend on FT1. Pass if the mean is ≥ comparator − 1 (a true −5 then passes ≈5% of the time [I]) and every residual is ≥ −3. The replicates also measure U25's sd.
- **Dress rehearsal:** a 9-h, 110-game, 532-min run: public 25 + DT 25 + 60 arc-interactive games, integrated with the DreamTeam plumbing by 10-24.
  - It runs on an on-demand box limited to match the scoring machine: `systemd-run -p MemoryMax=177G` and 48 cores (`taskset` or CPUQuota). Or it runs on GCP g4-standard-48 itself.
  - The limits are needed because Kaggle's box has 176.9 GB of RAM (106.6 GB used after boot, per the leader run's census) and g4-standard-48 has 48 vCPU, while NC144lds has 264 GiB and 144 vCPU.
  - If arc-interactive isn't integrated in time, it is a partial 50-game rehearsal and says so.
- **LB comparator:** the loader-flagged base draws (Z1–Z3) only.

## 8. Branches

| Branch | Trigger | What runs | Final picks |
|---|---|---|---|
| **B. No node (default)** | The default. Also: no opt-in, any GO condition failing, or the node not allocatable | Tracks H, V, SB, R1 and S. G3 and state-lab sampling run on spare free hours as the **H2 data bank**: E states, plus S, the stall breakthroughs on Nx levels | Pick 1 from the near-lossless line; pick 2 = Swift by §9, otherwise F or mem-unified-10x |
| **A. Full (opt-in)** | D4 opt-in and GO on 10-23 | Branch B plus §7.6 | Pick 1 from the near-lossless line; pick 2 = FT1 or Swift by §9, otherwise a harness config |
| **C. No paid compute** | Every [$] declined, or D0 unresolved | Kaggle only: Tracks H, V, SB and S. R1 only if the 1 GPU-h Kaggle capture-and-disk test passes (10-20) | Harness configs |
| **D. Elasticity falsified or invalid** | 10-19 | Tracks H and V, the Swift panel as free evidence, R1 and SB under $50, H2 data. No node | Harness configs |

## 9. Final-pick rule (pre-registered)

1. **Pick 1, the near-lossless line.** Take the furthest-adopted config: F → fallback+ (§7.1–7.2) → fallback++ (§7.3–7.4).
   - Adoption is on mechanism plus ≥3 error-free draws with a mean ≥ the previous config − 2, because 16 slots can't resolve +1 LB (eval-centric §1.5).
   - fallback+'s loader-flagged draws (Z) count toward its own adoption, since they carry everything it does.
2. **Pick 2, a hedge with a different policy.**
   - Let Δ* = s × (candidate draw mean − comparator draw mean), with the comparator being loader-flagged draws only.
   - Under the skeptical N(0, 3²) prior, 3–4 candidate draws against 3 comparator draws give a sampling SE of 2.67–2.86. That makes s ≈ 0.52–0.56, the posterior sd ≈2.0–2.1, and the band ≈ ±0.33 [I]. G-B FC 10's 0.595, 1.91 and ±0.29 assumed 6 comparator draws; the band scales with the posterior variance.
   - **Pick 2 is the trunk candidate (FT1, or Swift)** if Δ* > −0.33, it has ≥3 error-free draws, the dress rehearsal passed, and D8 is cleared.
   - **Otherwise pick 2 is F,** or mem-unified-10x v1 if F is pick 1.
3. **A decision-changing trunk never takes both picks in H1,** even at Δ* ≥ +1.0. At 3–4 draws, a Δ* of +1 is well inside noise. The hedge costs ≈0.3 of expected best-of-two at Δ = +1: one-each ≈2.65 against both-FT 2.97, at the posterior sd of ≈1.9–2.0 (G-B FC 10 closed forms) [I]. It protects against a hidden-set breakage that 3 draws can't exclude.
4. **Choose by config, never by a draw's public score** (split-half correlation −0.45 to −0.48; G-B Q4). Select actively, in case Kaggle auto-selects by public score (G0).
5. **If G0 shows the picks aren't scored best-of-two,** both go to the highest-posterior config.
6. **The alpha label** (the user's rule) needs every carrying run above the reference and a paired p < 0.05. No H1 adoption is expected to meet it; `arms.md` will say "adopted on mechanism; alpha not established".
7. **If D2a is declined.**
   - Every LB slot becomes a new stacked config built on the previous clean one, each change behind its own flag: fallback+ → + loader (Z) → + R1 → + adopted SB levers → + final hold.
   - A change is adopted when every draw that carries it is ≥ F − 2, and the pooled carrying draws feed Δ*.
   - Pick 1 is the latest clean stack.
   - FT1 and Swift each get at most 3 variant draws (for example, different λ values), compared with the loader-flagged stack draws.

## 10. Cost

| Step | Where | GPU-h | $ (spot / on-demand) | Approval |
|---|---|---|---|---|
| E1 (161 min), logging smoke, T½ and ≈15 LB commits | Kaggle | 3.3 + 3.75 | 0 | [K][LB] |
| fallback+ public-25, DT25 base, U25-F+0, serving benches | Kaggle | 6.9 + 1.4 | 0 | [K] |
| G1 pair with the perturbed boot, Swift full run, G3 | Kaggle | 1.6 + 2.3 + 3.3–4.0 | 0 | [K] |
| R1 G-MTP, fallback++ public-25 and U25 | Kaggle | 5.3 | 0 | [K] |
| State-lab sampling (R3 data, or the H2 bank) | Kaggle | 5, plus ≤4 optional per day after 10-25 | 0 | [K] |
| RTX bring-up, wheelhouse test, serving parity | RTX | 3–4 | $4 / $22 | [$] |
| Two raw-logged leader-stack runs | RTX spot | 4.6 | $5 | [$] |
| R1 capture and training | RTX (Kaggle in Branch C) | 10–15 | $10–15 / $55–85 | [$] |
| C1 dress rehearsal under scoring-box limits | RTX on-demand | 9.5 | $52 | [$] |
| *Branch A:* G-SV, canary base-vs-base, R3 sampling | 2 RTX spot boxes | ≈20–25 | $20–25 | [$] |
| *Branch A:* dequantised checkpoint build and check | CPU VM + RTX | — | $20–50 | [$] |
| *Branch A:* node: G5 on real rows, training, merges, exports | 8×H200 | 13–24 node-h (≤192 GPU-h) | $1.1–2.0k (G5 cap $1k; Lambda ≈$0.7–1.3k), plus persistent storage for the checkpoint and container (tens of dollars, UNVERIFIED) | [$][U] |
| *Branch A:* FT1 gates and U25 runs | Kaggle | 7.5–10 | 0 | [K] |
| *Branch A:* comparator and FT1 U25 replicates (if D2b allows) | RTX on-demand | 18.4 | ≈$100 | [$][U] |
| *Branch A:* FT1 dress rehearsal | RTX on-demand | 9.5 | $52 | [$] |
| 16 LB draws | Kaggle scoring | 0 (no quota) | 0 | [LB] |
| **Total, Branch B** | | Kaggle ≈31 (≤45 with optional H2 sampling); RTX ≈27–33 | **≈$70–170** (cap $300) | |
| **Total, Branch A** | | Kaggle ≈42–50; RTX ≈75–90; node ≤192 GPU-h | **≈$1.7–2.5k** (cap $3k) | |
| **Total, Branch C** | | Kaggle ≈35 | $0 | |

## 11. What each step leaves behind if the next one fails

- **D0 and G0:** a clear account position and the rules facts that set every pick.
- **E1:** a full-run gain at 1.33× compute, with its own fade. It prices every compute lever and sets H2's priority between efficiency and capability.
- **T½:** one P1/P3 input.
- **Tracks H, V and SB:** lossless gains carried into every later config, plus a measured per-class repair-call share.
- **The overlay tool, its positive control and the Z draws:** a deploy path that has been through the 9-h rerun.
- **The whole-turn state-lab, with raw logging, and G3:** H2's on-policy data engine, a measured yield and a measured samples-per-GPU-h.
- **The R1 exporter:** the G-MTP pipeline that every trunk change needs.
- **The node smoke:** measured training throughput at 32–57k.
- **The R3 round:** a measured result on our own model.
- **U25, the canary and a screen validated against accept05:** H2's evaluation machine.

## 12. H2: after 11-02, toward a consistent LB 60

**Starting facts.**
- **Efficiency alone isn't enough.** It needs 2.7–6× effective compute (G-H Q4), so 60 needs new capability: clearing levels with out-of-distribution mechanics. +1 level per hidden game is worth ≈+13–15 LB (L3 Q2).
- **Tufa's method statement, with its hedges.**
  - The per-level ARC score was "the most naive implementation".
  - The rewards included level rewards, ARC-AGI scores, "whether the code gets executed or not" and "length of the reasoning steps".
  - On self-generated games: "we can for example train on them".
  - They train on shorter sequences.
  - The transcript describes a research pipeline. It doesn't say that a submitted model was RL-trained (L2 FC 1, 7).
- **Tufa's compute and release.** 48 B200, with 144 B300 "expected" (L2 FC 2). They will release their solution after the competition ends (G-F FC 4).
- **What gates H2-1 (D11).** H2-1 spends only after G0's late-submission answer and H2-0's reading of the releases.

| Phase | Work | Gate | Kill | Compute, $ |
|---|---|---|---|---|
| **H2-0** (11-03 → 11-10) | **Private scores:** harvest every mellamoa submission's private score. These are extra draws on the fixed 55-game private set: a second sample of hidden games that improves the offline → hidden calibration. They don't settle P1 vs P3 on their own. **Releases:** read the Tufa, NVARC3 and Lord Han Solo releases (base model; whether they fine-tuned; G-F FC 4, 10) and reproduce the best on the RTX box. **Also:** check whether late submissions are scored [U], and run the 242-min elasticity ceiling on the RTX box (≈$25 on-demand) | The reproduction lands within ±3 of the release's public-25 claim | If late submissions are closed, "LB 60" is measured on the H2-1 proxy | ≈20 GPU-h RTX ≈ $20–110 |
| **H2-1** (weeks 1–3) | **Branch A first, if it didn't run in H1:** the node smoke and R3 round 1 under §7.6's gates. **LB-proxy suite:** ≥55 unseen games whose base score sits near the LB (≈31), from DreamTeam, arc-interactive (contamination-filtered; W9 FC 7) and generated games; frontier-generated games stay evaluation-only (§4.7). **Environment pool** of ≥100 mechanic families: a 20-seed `game_creator` pilot with an open model; a CPU acceptance script (≥6 levels, the solver replays to WIN, 50k random steps win nothing, novelty against the public set); holdouts by family. **An exact-stack RTX spot fleet** with a calibration gate. Log raw replies, token ids, logprobs and routing ids | The proxy ranks known generations correctly (taaf ≈4, fz ≈25, 10x ≈30), with per-run sd measured; ≥100 families; ≥150 cells with a clear rate in 0.2–0.8 | Generation yield <25–30% → coverage-select existing pools (W3 FC 6) | ≈300–600 spot GPU-h ≈ $0.5–1.5k, plus the node round ≈$1.1–2.0k |
| **H2-2** (weeks 3–7) | **Iterated on-policy RFT,** regenerated every round: R3 rounds 2–3; R4, hindsight-filtered breakthroughs on external games aimed at mechanic discovery (12 of 22 real stalls; G-A Q3); cross-attempt pairs (W8). λ-merges, and R1 retrained per trunk. Test a short-to-long curriculum | Each round: proxy +3 over ≥4 paired runs, then LB +3 over ≥6 draws if late submissions exist | Two rounds without gain → stop SFT | 150–400 GPU-h per round ≈ $1.6–4.2k (W6) |
| **H2-3** (weeks 6–12) | **Frozen-state RL.** States with 0 < p < 1, chosen by reward variance (G-G FC 5). Rewards: clears or verified progress, with length terms on correct samples only. Routing replay mandatory, aux loss 0, FP32 LM head (W2 FC 1–5). Rollouts at the served sampling settings (G-E FC 5). Auto-stop monitors on reward sd, entropy, distinct actions and no-tool-call rate | Trainer↔server KL ≤ 0.004; no collapse over 150 steps; proxy +5 | Collapse in 3 seeds, or no gain by step 300 | 1–3k GPU-h ≈ $10–30k |
| **H2-4** | Multi-turn game RL (Tufa's route), only if H2-3 gains. Re-check any new base that shows ≥30 LB | — | — | Open |

**H2 odds [I].**
- Our own pipeline alone: P(mean ≥ 64.5 within 3 months) ≈ 0.05. The best transfer evidence is small: far-OOD +0.16 to +0.24 (VHD-Play; W3 FC 1).
- Conditional on a reproducible top-team release with a true mean near 60 (Tufa's typical draw may be ≈59.5 after best-of-166 inflation; EB §1.7): ≈0.3.
- Blended: ≈0.1.
- If H2-2 doesn't move the proxy by ≥ +6 within about $10k, stop training and revisit the base model.

## 13. What we are deliberately NOT doing

| Not doing | Why |
|---|---|
| More snowiers submissions, or any snowiers quota | Rule 3.5.a and the rules header (G-F G17). We can't undo the existing 8 submissions ourselves; that is D0 |
| Renting the node in H1 by default | Even if the trunk works it is worth ≤ +0.8 / +1.5 LB, at P ≈ 0.01 in H1 (§3.2). It is opt-in only (D4) |
| Online RL / GRPO / GSPO before 11-02 | Routing replay is required; the one-node recipe tops out at 1k + 8k tokens; our budget buys only a few hundred steps (W2 FC 1–2; W6 Implication 4) |
| A Tufa-style game factory or spot fleet in H1 | 16–24 boxes of unconfirmed quota and a `game_creator` with no yield data, at 5–10× worse LB per dollar (judges' review) |
| Routed-expert LoRA or any AutoRound re-quant | It forces an int4 re-quant of `down_proj`; a non-expert LoRA needs none (W6 FC 1) |
| R4 or external-game SFT in H1; stratum S in H1 training | No external-game data exists yet; 25 families sit in AWM's degrade regime (G-G FC 6); S's hindsight verifier is new and unvalidated |
| Behaviour cloning of logged replies | History is normalised, and it copies the policy rather than shortening it (L1 FC 2) |
| SDPO / OPSD; frontier or Qwen3.8-2.4T distillation | OPSD lowered all 5 thinking models tested; API terms bar training on outputs; the 2.4T is text-only (G-G FC 2; W10 FC) |
| Novelty rewards, world-model SFT | Novelty stalls are 0.6% of tokens; world-model SFT gave no decision gain (G-A Q2; W8 FC 2–3) |
| A second R3 round, LoRA-scope sweeps, a wind tunnel, the Tinfield overlay | No time; a dense sibling can't rank MoE choices (`_critic.md` C12); Tinfield's deltas are tiny (G-D FC 8) |
| A low-effort teacher; EAGLE-3 / DFlash drafters; relaxed acceptance | Stay on-policy at the deployed effort (W7 FC 6); the drafters are unservable, and accept05 lost 6–10 points (G-E Implication 5) |
| A sandbox import whitelist | It would expose `os` and the guard state through `builtins`, `sys`, `inspect` and `pickle`; inert shims instead (§7.1) |
| Disabling the stale-state guard | It is Franzen's intentional design and its benefit is unmeasured (#18 is MAYBE); the narrow fix instead |
| #175 hardening in final candidates | Only if a request-replay check shows identical requests (eval-centric). **Exception:** #27's deadline fix goes behind its own flag into every config that changes startup or weights (§7.1) |
| Raw logging in LB notebooks | A competition rerun's outputs can't be retrieved, so it adds disk and throughput risk for no data (§4.5) |
| A load-time λ-scaled LoRA merge in H1 (W6 §4b) | It is an untested SGLang loader patch; at most 2 λ file overlays instead |
| A 242-min elasticity run in H1 | Its 121–161-min values are pre-fade and not comparable with REF's finals; the ceiling moves to H2-0 |
| More than one F anchor redraw; a final-day copy of pick 1 | Draws of one config are interchangeable, and picks are never made by public score; the picks lock on 11-01 from completed draws |
| Test-time training; a 27B swap | The weights take 69.85 of 96 GB (W10 Q5); the 27B scored 14.51 offline (L3 FC 6) |

## 14. Risks that silently cost days

| Risk | Mitigation |
|---|---|
| Disqualification over the existing snowiers entry | D0 today: Kaggle support, before the 10-26 entry deadline; no [$] until it is resolved; mellamoa only from now on |
| Pennyroyal doesn't build outside Kaggle's image | The wheelhouse test on a non-Kaggle box before anything depends on it; Branch C (Kaggle only) otherwise |
| No box or node quota, or an Azure offer that blocks N-series | Check the offer type today; a no-quota provider and GCP g4 requests filed today; Branch B needs no node |
| The whole-turn sampler overruns its 3–4 days | No G3 result on 10-22 → Branch B automatically; the near-lossless line doesn't depend on it |
| The node fails to reallocate, or loses its NVMe on deallocation | Allocate once (10-24 08:00); container and checkpoint pre-staged on persistent storage |
| ms-swift fails at the deployed shape | Pinned versions; G5 on real rows with a $1k cap, killed at 10-24 22:00 |
| The R1 capture hook overruns its 2–4 days | G4 on 10-22; fallback+ ships without R1 |
| An overlay is silently ignored | G1's hash check and perturbed-overlay positive control |
| The overlay or a slow startup breaks the 9-h private rerun | G1: startup ≤ base + 2 min and first action ≤ 12 min; the #27 flag; the Z draws; dress rehearsals under scoring-box limits |
| R3 collapses reasoning (the Arora pattern) | The 0.7 floor, anchors, low LR, λ-merges, the movement check, a screen validated against accept05 |
| A Pacific-day cap breach | The §6 ledger is kept in Pacific days, from push to completion |
| The weekly Kaggle cap runs out late | ≥6 GPU-h of reserve per week |
| D2a is declined | §9.7's stacked-config calendar |
| The last submission is still running at the deadline | Picks lock on 11-01 from completed draws; the 11-02 slot is information-only |
| Spot pre-emption kills a scored 2-h run | Spot only for restartable work |

## 15. Where the judges disagreed, and what I chose

| Question | Choice and one-line reason |
|---|---|
| Elasticity run at 161 or 242 min | **161** (changed after review). The tail fade scales with run length, so the 242-min run's S(121–161) are pre-fade values that aren't comparable with REF's finals. A 161-min run ends with its own fade, which is what a ×m compute lever delivers. It also saves 1.3 GPU-h. The ceiling moves to H2-0 |
| Harness fixes in H1 (eval-centric drops them) | **Keep, as cheap low-risk hygiene (≈ +0.1 expected).** The earlier ≈ +0.5 counted whole error calls rather than the repair calls they cause |
| Both picks on a fine-tune at Δ* ≥ +0.3 or ≥ +1.0 | **Never, in H1.** The hedge costs ≈0.3, and 3–4 draws can't exclude a hidden-set breakage |
| Node GO on the E gate *or* the S gate | **E only.** The S gate would almost always pass, so it doesn't bind; S becomes H2 data |
| R3 yield thresholds | **The three-part G3 on the stratum base.** (c) is ≥4% of total tokens on S = 0.30; the 0.7 floor caps it at 6.3%. (b) is within-state, so it can be falsified. A shortest/median ratio from k = 8 is mostly an order statistic |
| R1 value: sim2 (+1.1 to +1.9) or mechanism | **Mechanism:** +0.5–0.8 LB under P1 and +0.9–1.5 under P3. sim2 runs 15–28% steep |
| A short-window training fallback | **Allowed, with a stricter 57k gate.** Otherwise a memory limit, not evidence, kills the prong |
| Fleet size: 4 boxes or 1 | **1 at a time by default (Branch B); up to 3 in Branch A,** quota permitting |
| Truncation-validity band | **±2 sd on S(97),** a point that is pre-fade in E1 and at REF's fade start. ±1 sd would reject about 32% of valid runs |
| snowiers submission count (8 or 9) | **8,** per `runs/submissions.csv`, checked today |
| U25 power claims | **A catastrophe gate at n = 1; non-inferiority only with D2b replicates,** which also measure U25's sd |

## 16. What changed versus the 2026-10-17 plan

| 2026-10-17 plan (`research/reports/finetune-plan-2026-10-17.md`) | Now | Why |
|---|---|---|
| Re-quantise with AutoRound after the merge; redo the MTP | No re-quant: a non-expert LoRA is a file overlay. R1 is a lever in its own right | Intel quantised only the routed experts (W6 FC 1; G-D); G-E |
| Train on trimmed 8–16k windows | Served post-trim context, gated at 57k; short windows only as a fallback | AWM −59% from context mismatch (G-G Q1) |
| MiniCPM WSD with 30–50% generic self-replay; a small-sibling wind tunnel | 0% generic replay, within-state anchors and λ-merges; no wind tunnel | Replay anchors only partly (G-G (b), FC 10); a dense sibling can't rank MoE choices (`_critic.md` C12) |
| "~50–60% of tokens on stuck levels"; a novel-board bonus | 22–28% in leader runs; novelty rewards dropped | L1 correction 1; G-A Q2 |
| R3 worth a −25% cut on most tokens | ≈ −4% of total tokens (≤ −6.3%) under the recipe's own stratum, floor and anchors | G-A FC 1–2; §3.2 |
| Training data = logs + DreamTeam/arc-interactive in H1 | Own whole-turn verified siblings at ≈400 T15 states from the deployed stack; external games for evaluation, or H2 | Normalised history (L1 FC 2); AWM regime (G-G FC 6); state supply (§7.6) |
| Test = 10 held-out public games + 2-h public runs | U25 = H10 + DT15 with a defined metric; frozen-state screen on H10 states, validated against accept05; DT10 single-use | Compute coupling, +3 to +8 (G-B FC 5) |
| Kaggle mellamoa + snowiers for collection | mellamoa only; an RTX box replaces snowiers; the existing exposure is raised with Kaggle (D0) | Rule 3.5.a (G-F G17) |
| Ask for 450–950 H200 GPU-h, cost unknown | The node is an opt-in H2 investment; ≤24 node-h if run in H1; caps $1k (G5) and $2.5k | W6 FC 6; §3.2 |
| LB sd ≈5, ~16 draws needed | sd 3.50; the LB as a crash and non-inferiority check | L3 FC 7; G-B FC 4 |
| Finals = the two best mean configs | Pre-registered rule; a trunk fine-tune takes at most one pick | G-B FC 10 |
| ReST-EM round 2 in H1; no upward-compute test | One round at most, by default after 11-02; E1 at 161 min first, plus a P1/P3 throttle draw | Time; G-H Q1, Q5 |

## 17. Project-rule compliance

- **Branches and flags.**
  - Branches: `arm/harness-fixes` (including the #27 flag), `arm/dreamteam-eval` (with arc-interactive), `arm/weight-overlay`, `arm/raw-logging`, `arm/serving-levers`, `arm/r1-mtp`, `tooling/state-lab` (ported from `archive/tooling/state-lab`) and `arm/r3-lora`.
  - `elastic-161`, `throttle-half` and `fallback-plus` are `arms.json` entries. `throttle-half` uses the new, unit-tested `private_minutes` builder option.
  - Each change sits behind its own default-off flag, with its own PR and `arms.md` entry. Docs change in the same PR: `CONFIGURATION.md` for every flag, `kaggle/README.md` and the README. Research docs and run logs go to `main`, including the overdue final-hold-lap-10x archive.
- **Game- and time-agnostic.**
  - No code branches on game id, grid size, palette or level count. E1 and T½ change only the budget.
  - The `time` shim gives a step counter, never a wall clock.
  - PRs name every public-calibrated threshold: REF 51.57 and S(97), the 0.7 floor, the 5k stratum, S = 0.30, G3, the screen deltas, accept 2.63.
- **No exact repeats.** Every Kaggle run is a new config. LB redraws depend on D2a (§9.7 otherwise), and rented replicates on D2b.
- **Evidence.**
  - Wins and time-to-win come first.
  - Improvements are claimed only with `make significance BASELINE_SCORE=... CANDIDATE_SCORE=...` against the matched comparator.
  - Adoption for the finals is not an alpha.
- **Laptop load:** `nice -n 15`, ≤3 heavy jobs.

## Review log (red-team findings not applied as written)

Every critical and major finding from the three red-team passes was applied, except as noted below; most minor ones were applied too. I checked each against the notes and the repo. Confirmed today:
- 2,857 / 1,514 / 64 / 43 / 51 eligible states (`ana_steps.json`), and a mean ≥5k turn of 11.7k tokens;
- 176.9 GB of host RAM and a 19.5 GB `/kaggle/working`;
- the hard-coded TRUE_SUBMISSION `532*60`;
- tail-fade fraction 0.2;
- #18's MAYBE label;
- the bootstrap's `os`/`sys`/`builtins` imports;
- final-hold-lap-10x missing from `runs/index.csv` and `arms.md`.

The exceptions:

- **"Require G3(c) ≥10% before the node is billed."** Rejected: the plan's own 0.7 floor and ≥5k stratum cap the cut at 6.3%. Instead, Branch A became opt-in and G3(c) is ≥4%.
- **"Hold all paid compute until Kaggle answers."** Changed to "until Kaggle answers or the user accepts the risk in writing" (the finding's own alternative), because Kaggle's reply time is unknown.
- **"Harness fix ≈ +0.05–0.1 LB" (pass 1) vs "+0.05–0.3" (pass 3).** I used +0.05–0.3 (P1). The low end comes from G-A FC 7's repair-call share; the high end from issue #18's 1.7% upper bound. The central value is ≈ +0.1.
- **E1 regime fix.** I took the 161-min option rather than shifting thresholds by a regime offset measured on one run. I kept the validity band at 2 sd, since the fade offset no longer enters.
- **"G7 kill at tokens/action > 1 − 0.3 × projected total cut."** Changed: at a projected 4% cut, that line (0.988×) sits inside run-to-run noise. So G7 kills at ≥1.0× (it moved the wrong way), and success is pre-registered at ≤ 1 − 0.5 × P_t.
- **"Use the load-time λ-scaled merge of one adapter (W6 §4b)."** Not adopted for H1: it is an untested SGLang loader patch, and it would add a second deploy path in two days. Instead, at most 2 λ file overlays.
- **"Second on-demand box, or drop the replicates to n = 2."** I took the rescheduling half instead: comparator replicates run 10-21 → 10-24 and the C1 rehearsal on 10-25. That removes the overlap without more quota.
- **"Add G5's Kaggle panel (0.7 GPU-h) to the ledger."** Replaced by a byte-identity check of the zero-LoRA merge, since G1 already proves the overlay path serves.
- **"Require two throttle draws."** Took the alternative instead: T½ is one input that can't trigger a no-go alone. Slots are scarce, and Branch A is opt-in anyway.
- **"Count eligible states per rebuildable config before 10-19."** Done here for the corpus. The per-config rebuild check is folded into G-SL rather than run as a separate pre-10-19 task.
