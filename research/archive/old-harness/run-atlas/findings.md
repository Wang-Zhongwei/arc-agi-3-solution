# Findings

What the runs say, each with the evidence behind it. Numbers come from `data/` and match
`versions.md`. "Mean / trimmed" are offline scores over the 25 public games at full budget.

## 1. The Wang → Franzen jump came from tokens, not from a smarter agent

| | Wang b1 (best LB 7.07) | Franzen control |
|---|---|---|
| generated tok/s | 294 | 611 |
| tokens per game | 93k | 183k |
| levels cleared (25 games) | 56 | 105 |
| **tokens per cleared level** | **41.6k** | **42.7k** |
| mean score | 17.74 | 46.25 |

Both agents spend about the same ~42k generated tokens to clear a level (charts 6 and 8).
The weaker Wang arms spent 55k-236k per level, so a bad prompt or allocation change can
waste tokens badly. No change has beaten ~39k per level.
Franzen's serving (SGLang, FP8 KV, cache-friendly trimming) roughly doubles the tokens each game
gets, which roughly doubles the levels cleared. The score rises faster than the levels (2.6×
vs 1.9×) because later levels carry more weight.

## 2. Only clearing levels counts

RHAE caps a game's score at its weighted fraction of levels cleared. Rescoring every game in
our 7 Franzen-era runs from the per-level data reproduces the harness scores exactly *only*
with the cap; without it, scores come out 3-5 points too high. Clearing a level faster than
humans therefore earns nothing once the cap binds. Every gain has to come from clearing more
levels, and later ones are worth the most.

## 3. The memory ledger, shown only after a trim, is the one clear win so far

| run | mean | trimmed | median | won | tokens per cleared level |
|---|---|---|---|---|---|
| control | 46.25 | 41.58 | 35.7 | 7 | 42.7k |
| **ledger on trim** | **52.56** | **48.43** | **58.3** | **8** | **38.7k** |

- The three near-identical control-like runs (control, temperature 0.6, audit) span 44.1-46.3
  mean and 39.2-41.6 trimmed. The ledger sits about 6 points above that whole range.
- It also made levels cheapest of any run: 38.7k tokens per cleared level, −9% vs control
  (every other Franzen-era arm is within 41.5k-48.0k). It makes levels cheaper rather than
  buying more tokens.
- It worked as designed: the block was at the head of history in 63-78% of requests (every
  request after a game's first trim). The model wrote ~1,700 tags of each kind.
- Still one run. A repeat (or the two scheduler arms, which run on top of it) will confirm.

## 4. More streams buy throughput, not score

| run | streams | context (trim to) | harness tok/s | decode P90 | mean |
|---|---|---|---|---|---|
| control | 10 | 128k (58k) | 611 | 841 | 46.25 |
| 14 streams | 14 | 96k (42k) | 703 (+15%) | 1,005 | 46.38 |
| ledger on trim | 10 | 128k (58k) | 589 | 838 | 52.56 |
| ledger + 12 streams | 12 | 110k (49k) | 644 (+9%) | 921 | 48.08 |

The KV pool (~1.01M tokens) is shared, so more streams means less context per game. Twice now
the extra tokens were cancelled by the smaller context. Knowledge kept per game (the ledger)
moved the score; raw tok/s did not.

## 5. What did not help

| run | mean | trimmed | why it did not help |
|---|---|---|---|
| temperature 0.6 | 45.92 | 41.22 | no measurable effect |
| strategy audit (60k) | 44.10 | 39.24 | fired 30 times; no gain |
| UCB scheduler | 43.84 | 38.95 | active (tokens/game sd 91k vs 69k), but sent 61% of tokens to games that ended ≥50% complete vs 74% under Franzen's priority |

## 6. Serving experiments: most "speed" ideas are blocked or slower on this stack

| idea | result |
|---|---|
| MXFP8 online activations | loads, frees 4.6 GB (weights 69.85 → 65.28 GB), KV pool 1.01M → 1.36M tokens, but **decode −27%** at full batch (555 vs 765 tok/s) |
| FP4 (nvfp4) KV cache | allocates 2.04M tokens, but needs the triton attention backend, which has no FP4 dtype: crashes on the first batch |
| 4 MTP draft steps | impossible: Qwen QSA caps speculation at 4 draft tokens (Franzen's 3 steps is the maximum) |
| greedy decoding | draft acceptance 2.67 → 2.81 (+5%); not tested alone; repetition risk |
| more streams | the only measured win (section 4) |

## 7. Noise: what a single number can and cannot tell you

- **Per game:** the same game swings 50-80 points between near-identical runs (chart 4; e.g.
  cn04 71 → 5, sc25 0 → 81).
- **Offline run mean:** near-identical runs span about ±1.1 (44.1-46.3). Gaps under ~3 are
  not evidence.
- **Leaderboard:** Franzen-copy submissions average 25.77 with sd 3.93 (community snapshot).
  Our copy of Franzen scored 20.87, about 1.25 sd low; Franzen and skarin scored 27.89 and 27.80.
  In the Wang era, one notebook scored 7.07, 4.21 and 3.95.
- **Offline → leaderboard:** about 0.56× (46.25 offline → ~25.8 mean LB). A leaderboard 35 needs
  about 62 offline on average.

## 8. The scheduler

Franzen's priority is P = (A + B)·C. Data from 869 level attempts:

- **C (chance of clearing soon):** close to the measured curve up to ~70k tokens on the level
  (chart 7). They separate only past 80k, where the data is thin (118 attempts reach 80k, 31
  reach 120k). An earlier claim of ours that C "gives up far too fast" compared only its token
  part and overstated the gap.
- **B (future levels):** hand-set 8/7/5/0. Measured next-level clear rates are 0.73-0.86, which
  a chained continuation value can use directly. Being tested as `fz-ledger-fitted`.
- About a quarter of all tokens (24% control, 28% ledger) go into the final level of a game
  that is never cleared. Being tested as `fz-ledger-stallswap` (early handover of stuck games).
- Offline runs put 25 games on 10 slots; the scored rerun puts 110 on 10, so scheduling
  matters roughly 4× more there than offline scores show.

## 9. Wang-era lessons that still hold

- The memory ledger was the first change to win games (memory-fixed 13.65 vs 9.86 baseline);
  in the Franzen era its trim-only version is again the best arm.
- Images matter: ASCII-only scored 4.63; image upscale 8 gave the best Wang mean (19.43).
- One prompt arm (spend actions / batch when the path is known) had the best robust metrics
  of its batch and was reverted on the mean alone.

## 10. Infrastructure lessons (each cost a run)

- Two notebooks cold-starting together both hung loading the 190 GB model; stagger pushes ≥20 min.
- Kaggle mounts inputs at `/kaggle/input/datasets/<owner>/<slug>` or `/kaggle/input/<slug>`
  depending on the notebook; resolve both.
- Kernel sources must be < 1 MB; the documented patch is shipped gzip+base64.
- `kaggle kernels status/output owner/slug/N` ignores `N`; use the `version_label="vN"` API.
- A server crash during startup used to leave the games running against nothing; the notebook
  now stops at once.
