# Object-perception benchmark: can a learned vision model help? (2026-10-10)

**Short answer:** no candidate passes the gate for a Kaggle arm.
- **MobileSAM, the off-the-shelf ViT**, is the most accurate segmenter, but it is about 40-60x over the CPU budget.
- **The tiny hybrid CNN-ViT trained on engine ground truth** memorises the public sprites. On games it has never seen, it scores below the plain colour-component baseline.
- **The cheap lexicon heuristic** gains about +3 F1, far short of the +10 gate.
- **Player-role inference** is limited by the tracking rule, not by segmentation. Perfect (oracle) segments do no better than colour components.

## Data and protocol
- **Ground truth.** `vision/gt/data` holds 1,000 frames, 40 from each of the 25 public games. Every replayed step was verified pixel-exact (5,717/5,717).
- **Full-trajectory dump.** `bench/full/` holds every replayed step's frame, instance map, HUD map and object list (5,717 frames, 16 MB). It was made by `bench/gt_full.py`, which reuses `gt_dump.py`, and all frames match. It supplies the training data for candidate 4 and the action stream for candidate 5.
- **GT objects.** Visible engine sprite instances, with HUD pixels removed. The *group* level merges 4-adjacent instances of the same design (tile maps).
- **Ignored segments.** Predicted segments that are at least 50% GT background or HUD are ignored. This rule is the same for every candidate.
- **Metrics.** F1@IoU 0.5 is one-to-one, which is exact for partitions. The headline is the **macro average over games**.
  - frag: at least 2 predicted segments, each holding at least 10% of the object.
  - shatter: no single segment holds half of the object.
  - merge: the best segment also takes at least 10% of another object.
  - Hard-subset columns give recall at IoU 0.5.
- **Code.** `bench/eval.py` (shared evaluator), `cand_*.py` (candidates), `report.py` and `report_roles.py` (tables), `roles_eval.py` (candidate 5), `train_tiny.py` and `tiny_model.py` (candidate 4), and `timing.py` (latency).
- **Load.** All jobs ran niced inside the shared lock. The machine was heavily shared during the run, with load average between 7 and 45, so all ms figures are noisy upper bounds. The clean-ish numbers are in the latency table.

## 1. Segmentation, all 25 games, 1,000 frames (instance level; group level in brackets)

| candidate | F1@0.5 macro | P / R | exact | frag | shatter | merge | segs per GT obj | ARI | R: touch-diff-design | R: multicolour | R: scaled | R: tile games | names V | CPU ms/frame (median, 1 thread) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 baseline colour CCs (`segment_layer`) | **0.467** [0.494] | 0.41 / 0.68 | 0.40 | 0.24 | 0.15 | 0.20 | 3.21 | 0.66 | 0.25 | 0.68 | 0.45 | 0.56 | 0.88 | 110 (p95 540, max 2,057) |
| 2 lexicon, `cc` mode (frozen choice) | **0.502** [0.529] | 0.48 / 0.64 | 0.41 | 0.15 | 0.11 | 0.30 | 2.66 | 0.69 | 0.28 | 0.71 | 0.48 | 0.49 | 0.88 | 3.8 (p95 13) |
| 2 lexicon, `group` mode (competitor-style 4-conn multicolour blobs, then cuts) | 0.395 [0.385] | 0.47 / 0.48 | 0.28 | 0.16 | 0.11 | 0.52 | 2.27 | 0.54 | 0.17 | 0.58 | 0.45 | 0.28 | 0.85 | ~9 |

- **Dev/non-dev split for the lexicon.** On the 5 dev games (dc22, g50t, sb26, sc25, sk48) it scores 0.524 vs baseline 0.527. On the 20 non-dev games it scores 0.497 vs 0.452. The gain is not an artefact of tuning on the dev games, but it is small. Period cutting (`LEX_PERIOD=1`) made the dev results worse (0.465), so it was left off.
- **Cell size.** The lexicon recovers it correctly on 78.9% of frames (85.8% excluding bp35/lf52, whose scene is drawn at its own scale). The competitor reported 93%. Our estimator fails on ar25 (scale 3), m0r0 (scale 4-5) and tu93 (3-px maze cells at camera scale 1).

## 2. MobileSAM vs the same frames (every 4th GT frame: 250 frames, 10 per game, all 25 games)

| candidate | F1@0.5 macro | P / R | exact | frag | shatter | merge | segs per GT obj | R: touch-diff | R: multicolour | R: tile games | CPU ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 0.468 [0.496] | 0.41 / 0.67 | 0.40 | 0.23 | 0.15 | 0.20 | 3.18 | 0.25 | 0.66 | 0.55 | 110 |
| 2 lexicon (cc) | 0.494 [0.518] | 0.45 / 0.66 | 0.42 | 0.18 | 0.12 | 0.25 | 2.85 | 0.27 | 0.69 | 0.51 | 4 |
| **3 MobileSAM, zero-shot, CC-point prompts** | **0.550 [0.587]** | 0.59 / 0.60 | 0.37 | 0.19 | 0.10 | 0.33 | 1.41 | 0.26 | 0.69 | 0.34 | **3,165 at 1 thread, 1,873 at 4 threads; 5.1 GB RSS** |

- **What it does well.** MobileSAM groups textured and multi-colour sprites that colour components shatter. Gains over the baseline: m0r0 0.04 → 0.35, re86 0.09 → 0.37, bp35 0.06 → 0.48, r11l 0.21 → 0.62.
- **What it does badly.** It merges tile maps (ls20 0.18) and loses exactness.
- **Overall.** +8.2 instance F1 and +9.1 group F1 over the baseline, still under the +10 gate.
- **Timing.** The encoder alone takes about 0.55-0.65 s/frame at 1 thread. The decoder with 21-64 prompts takes another 0.7-1.6 s, even with the 256-px low-res masks.
- **Verdict: infeasible.** It is roughly 40-60x over the 50 ms budget, and its RSS is far over the "<1 GB" bar. It would need to run on the GPU, which has only about 1.2 GB free.

## 3. Tiny hybrid CNN-ViT trained on engine ground truth, leave-5-games-out (2 splits; 10 held-out games)

The model has 845k params. It takes a one-hot input plus same-colour-neighbour channels, uses convs, then 3 transformer blocks on 16x16 tokens, then a U-Net decoder. Its heads predict affinity at d = 1, 2, 4, semantic class (bg/sprite/HUD), and clickable.
- **Training.** Unique frames from the 20 training games (4,454 for F1) plus 30% synthetic frames built from training sprites. Augmentation: palette permutation and D4. 4,000 AdamW steps on MPS.
- **Splits.** F1 holds out ls20, cd82, bp35, sp80, vc33. F2 holds out ar25, tu93, lf52, cn04, r11l.

| on the 10 held-out games | F1@0.5 macro | P / R | exact | frag | shatter | merge | R: touch-diff | R: multicolour | R: tile games | CPU ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 0.395 [0.480] | 0.38 / 0.63 | 0.26 | 0.27 | 0.10 | 0.31 | 0.26 | 0.76 | 0.14 | 110 |
| 2 lexicon (cc) | **0.426 [0.504]** | 0.42 / 0.63 | 0.29 | 0.23 | 0.08 | 0.34 | 0.32 | 0.81 | 0.13 | 4 |
| **4 tiny CNN-ViT (out-of-fold)** | 0.365 [0.434] | 0.41 / 0.46 | 0.26 | 0.14 | 0.31 | 0.27 | 0.15 | 0.56 | 0.11 | 16 (torch CPU, 1 thread; 276 MB RSS) |

- **Generalisation gap.**
  - F1 model: 0.617 on its own training games (baseline 0.458 on the same games) vs 0.455 held out (baseline 0.502).
  - F2 model: 0.696 in-fold vs 0.274 held out (baseline 0.287).
  - In-fold it beats the baseline by 16-18 points. Held out it loses by 3. Worst held-out game: bp35 at 0.04.
- **Conclusion.** The net memorises the public sprite designs and does not transfer to unseen ones. This is exactly the risk for the 110 hidden games.
- **Clickable head.** Not scored separately. It was trained, but the segmentation result made it moot.
- **Skipped.** The CNN-only and 2-frame ablations were not run (time box).

## 4. Behavioural role tracker (gap-analysis Stack A) on the full recorded action stream

- **Data.** 5,717 steps. Player = the stable-named segment that moves on more than 30% of arrow presses and in the arrow's direction more than 60% of the time, top-1 by agreeing moves (at least 2), online and causal.
- **Eligible levels.** Levels where a GT-controlled sprite visibly moves under an arrow action: 58 levels in 12 games. (ar25 has no eligible level: by this filter, its GT-controlled sprites never visibly move under an arrow.)
- **HUD.** Segmentation-agnostic: a 3-px edge strip that changes on at least 30% of a level's steps.

| segments fed to the tracker | player correct at level end | player committed within 5 actions | median actions to commit | click: segment under the click = GT target (IoU ≥ 0.5) | HUD-change pixels P / R | seg ms/frame |
|---|---|---|---|---|---|---|
| colour CCs (harness) | **43/58 (74%)** | **28/58 (48%)** | 3 | 1,124/1,550 (73%) | 0.40 / 0.40 | ~1 |
| oracle GT instances | 42/58 (72%) | 27/58 (47%) | 3 | 1,550/1,550 (100%) | 0.40 / 0.40 | n/a |
| lexicon (cc) | 37/58 (64%) | 11/58 (19%) | 25 | 1,280/1,550 (83%) | n/a | ~16-60 under load |
| tiny CNN-ViT (held-out games only: ls20, sp80, tu93, cn04) | 5/20 (25%) vs CC 16/20 on the same levels | 0/20 vs CC 11/20 | 38 | 22/37 vs CC 31/37 | 0.49 / 0.27 | 17 |
| colour CCs, ratio rule (variant) | 38/58 (66%) | 23/58 (40%) | 4 | (subset) | | |

- **Better segments do not help find the player.** Oracle segments are no better than colour CCs (72% vs 74%). The failures are in the rule:
  - games with several controllable pieces (re86, ka59 commit late);
  - sc25 and m0r0, where other sprites also move with the arrows;
  - a sprite whose look changes when it is occluded, which breaks name-based tracking.
- **Gate check.** The gate needs the correct player within 5 actions in at least 80% of levels. The tracker manages 48%, so it **fails**.
- **Where segmentation does matter: click targets.** The segment under a click is the real clicked object 73% of the time with colour CCs, 83% with the lexicon and 100% with the oracle. That is the one place a better segmenter feeds a role fact directly.
- **Not evaluated.** Blocker and collectible roles, and the cross-frame identity metric (time box).

## Latency (laptop M3 Pro CPU, nice 15, 50 frames, load average 7-13)
| candidate | 1 thread median / p95 / max | 4 threads median | RSS | weights |
|---|---|---|---|---|
| baseline `segment_layer` (pure Python, includes containment and hashes) | 110 / 540 / 2,057 ms | n/a | 63 MB | none |
| lexicon (cc) | 3.8 / 12.6 / 40.5 ms | n/a | 66 MB | none |
| tiny CNN-ViT (torch) | 16.4 / 20.2 / 21.4 ms | 31 ms (contended) | 276 MB (torch) | 3.3 MB per fold |
| MobileSAM (torch, encoder + batched decoder) | 3,165 / 7,123 / 11,381 ms | 1,873 ms | 5.1 GB | 40.7 MB |

Earlier quieter measurements: baseline 32 ms median and 253 ms max (segtime.py); tiny 10-13 ms per frame inside eval.

## Decision gate
**Gate:** held-out macro F1 at least 10 points over the baseline at both the instance and group levels, AND better on the hard subset, at 50 ms median / 250 ms p95 or less, deterministic.

| candidate | Δ instance F1 | Δ group F1 | hard subset | latency | deterministic | verdict |
|---|---|---|---|---|---|---|
| 2 lexicon (cc) | +3.5 (all 25), +3.1 (held-out 10) | +3.5 / +2.4 | touch-diff +0.03, multicolour +0.03, tile −0.07 | 4 ms | yes | **fail** (too small); best deployable segmenter |
| 3 MobileSAM | +8.2 (subset) | +9.1 | mixed (tile −0.21) | 1.9-3.2 s, 5 GB | yes | **fail** (latency and RAM) |
| 4 tiny CNN-ViT | −3.0 (held-out) | −4.6 | worse | 16 ms | yes | **fail** (no generalisation to unseen sprites) |
| 5 role tracker | n/a | n/a | n/a | ~1 ms with CC | yes | **fail** on the 80% within-5-actions gate (48%); 74% correct by level end |

## Deviations from the plan
- **Lexicon.** Reimplemented from the discussion's description. The competitor's code is not in `competitors/`, and no browser subagent was available to fetch it. Of its two modes, `group` (multicolour blobs, then library and period cuts) was clearly worse, so `cc` (colour CCs, then merging library designs seen isolated earlier in the game) was frozen.
- **MobileSAM.** Ran in PyTorch on CPU, not ONNX. It used the 256-px low-res decoder masks, and was scored on every 4th frame because of its cost. FastSAM-s was not run: its licence is AGPL, and the verdict is already clear from SAM's quality-to-cost ratio.
- **Tiny model.**
  - 2 folds instead of 5, as the task asked.
  - Conv positional encoding instead of 2-D RoPE (both are translation-equivariant).
  - Latency timed with torch on CPU rather than a numpy port.
  - F2 training took 28 min of wall time because of other agents' load (F1 took 15 min). Intrinsic cost is about 0.08 s per step on an idle MPS.
- **Role tracker.** Player, click-target and HUD only.

## Recommendation
1. **Do not ship a learned vision model for objects.** The pretrained ViT segmenter that helps (MobileSAM) is too slow on CPU and has no GPU room. The trained tiny ViT does not transfer to unseen sprite designs, which is what the 110 hidden games contain.
2. **Role facts need a better tracker, not better segments.** If perception work continues, put the effort into the behavioural tracker's rule: per-level candidates, co-motion grouping, handling several controllable pieces, and occlusion-robust identity. Then re-run `roles_eval.py` against the 80% gate. Plain colour CCs are good enough input.
3. **If a cheap segmentation upgrade is wanted,** lexicon `cc` mode (4 ms, deterministic, numpy+scipy) is the only deployable improvement, and it is small (+3 F1). Its main concrete win is click-target identity (73% → 83%). It would only be worth a screen as part of a click-target fact.
Recreate venv: uv venv --python 3.12 vision/venv && VIRTUAL_ENV=vision/venv uv pip install torch numpy scipy timm git+https://github.com/ChaoningZhang/MobileSAM.git
