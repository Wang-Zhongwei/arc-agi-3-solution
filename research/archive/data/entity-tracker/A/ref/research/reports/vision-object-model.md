# Vision models for object identification (ViT / SAM / detectors / tiny engine-trained net)

*2026-10-10. The user: "if we can use a hybrid vit model or smth that is good enough to identify objects and give it to the qwen model, that would be awesome" (after Kaggle discussion 746295, the lexicon view). Benchmarked against exact engine ground truth: 1,000 frames from all 25 public games rendered pixel-exact from the game source, plus 5,717 replayed steps for role tracking. Bench artifacts were in the session scratchpad (`vision/bench/`).*

## Verdict

**Don't ship a learned vision model.** MobileSAM zero-shot is the most accurate (+8.2 instance F1 over colour components) but takes 1.9–3.2 s and 5 GB per frame on CPU and merges tile maps; a tiny CNN-ViT trained on engine ground truth wins by 16–18 F1 on its training games but loses 3 F1 on held-out games (this recipe memorised public sprite designs). The lexicon `cc` segmenter is the best deployable one (4 ms, +3.5 F1, click target under the cursor right 83% vs 73%). **Perfect oracle segments do not improve player identification** (72% vs 74% correct by level end), so roles come from behaviour, not appearance. Next perception step: a behavioural entity tracker, gated on its offline Stage-0 (80% of levels with the player committed within 5 actions), with the spec revisions in the review below.

## Benchmark

Instance-level F1@IoU0.5, macro average over games; group-level (same-design tiles merged) in brackets. Each row is compared with a baseline run on the same frames.

| candidate | frames / games | F1@0.5 [group] | vs baseline on same frames | P / R | frag / shatter / merge | hard-subset recall (touch-diff / multicolour / tile games) | CPU ms per frame, 1 thread (median / p95) | gate |
|---|---|---|---|---|---|---|---|---|
| 1 Baseline colour CCs (`segment_layer`) | 1,000 / 25 | 0.467 [0.494] | n/a | 0.41 / 0.68 | 0.24 / 0.15 / 0.20 | 0.25 / 0.68 / 0.56 | 110 / 540 (max 2,057; earlier quiet run 32 median) | reference |
| 2 Lexicon heuristic, `cc` mode (frozen) | 1,000 / 25 | 0.502 [0.529] | +3.5 [+3.5]; held-out 10: +3.1 [+2.4] | 0.48 / 0.64 | 0.15 / 0.11 / 0.30 | 0.28 / 0.71 / 0.49 | 3.8 / 12.6 | FAIL (gain too small); best deployable segmenter |
| 2 Lexicon, `group` mode (competitor-style) | 1,000 / 25 | 0.395 [0.385] | -7.2 [-10.9] | 0.47 / 0.48 | 0.16 / 0.11 / 0.52 | 0.17 / 0.58 / 0.28 | ~9 | FAIL |
| 3 MobileSAM zero-shot, CPU (point-prompted) | 250 (every 4th) / 25 | 0.550 [0.587] | +8.2 [+9.1] (baseline 0.468 [0.496] on the same frames) | 0.59 / 0.60 | 0.19 / 0.10 / 0.33 | 0.26 / 0.69 / 0.34 | 3,165 / 7,123; 1,873 at 4 threads; 5.1 GB RSS | FAIL (40-60x over the latency budget, RAM) |
| 4 Tiny hybrid CNN-ViT (845k params, trained on engine GT), out-of-fold | 400 / 10 held-out (2 splits) | 0.365 [0.434] | -3.0 [-4.6] (baseline 0.395 [0.480]); in-fold +16 to +18 | 0.41 / 0.46 | 0.14 / 0.31 / 0.27 | 0.15 / 0.56 / 0.11 | 16.4 / 20.2 (torch CPU, 276 MB) | FAIL (does not generalise) |
| 5 Behavioural role tracker on colour CCs | 5,717 steps / 58 levels in 12 games | player correct at level end 74%; committed within 5 actions 48%; segment under a click = GT target 73% | oracle GT segments: 72% / 47% / 100%; lexicon 64% / 19% / 83%; tiny (held-out) 25% / 0% | n/a | n/a | HUD-change pixels P/R 0.40 / 0.40 | ~1 | FAIL (gate needs 80% within 5 actions) |

### Notes per candidate

- Setup: I added a full-trajectory ground-truth dump (bench/full, made by gt_full.py, which reuses gt_dump.py). It holds every replayed step's frame, instance map and HUD map: 5,717 frames, all pixel-exact matches. Segmentation metrics use the existing 1,000-frame set (40 per game). The full dump supplies the training data for candidate 4 and the action stream for candidate 5. Everything ran niced inside the shared lock. The machine was heavily shared (load average 7 to 45), so timings are noisy upper bounds.
- 1 Baseline (harness segment_layer): macro instance F1 0.467, group F1 0.494. Precision is the main weakness: 3.2 segments per GT object. Textured or multi-colour sprites shatter it: m0r0 0.04, bp35 0.06, re86 0.09, tu93 0.16. Pure-Python cost was 110 ms median and 2 s max per frame under load, because the containment pass is quadratic on noisy frames.
- 2 Lexicon: I reimplemented it from the discussion's description, because the competitor's code is not in competitors/ and no browser subagent was available to me. The competitor-style group mode (4-connected multi-colour blobs, then library and period cuts) badly over-merges sprites drawn on other sprites: F1 0.395. The frozen cc mode starts from colour components, then merges multi-colour designs that were seen isolated earlier in the same game. It scores +3.5 F1 overall and +3.1 on the 10 held-out games. That gain is from the 20 non-dev games (0.497 vs 0.452); on the 5 dev games it ties the baseline (0.524 vs 0.527). It runs in 4 ms, is deterministic and needs only numpy and scipy. Cell size is right on 79% of frames (86% excluding bp35/lf52); the competitor reported 93%. It fails on ar25, m0r0 and tu93. Period cutting made the dev games worse, so it is off. The biggest concrete win is click targets: the segment under a click is the real object 83% of the time vs 73%.
- 3 MobileSAM (Apache-2.0, TinyViT encoder): ran zero-shot in PyTorch on CPU (not ONNX), prompted with one point per colour component, on every 4th frame because of its cost. It is the most accurate candidate: +8.2 instance F1 and +9.1 group F1 over the baseline on the same 250 frames, mainly by grouping multi-colour and textured sprites (m0r0 0.04 to 0.35, bp35 0.06 to 0.48, re86 0.09 to 0.37). It merges tile maps (tile-game recall 0.55 to 0.34) and still misses the +10 gate. The encoder alone takes 0.55-0.65 s; with the decoder a frame takes 1.9-3.2 s with 5 GB RSS. The GPU has only about 1.2 GB free, so it is infeasible on Kaggle. I did not run FastSAM-s (AGPL) given this result.
- 4 Tiny hybrid CNN-ViT: 845k params, 16-colour one-hot plus same-colour-neighbour channels, convs, then 3 transformer blocks on 16x16 tokens (conv positional encoding rather than RoPE), then a U-Net decoder. Heads predict affinity at distances 1, 2 and 4, a background/sprite/HUD class, and clickable. Training: 4,000 AdamW steps on MPS, 30% synthetic frames, palette-permutation and D4 augmentation, two leave-5-games-out splits. On its training games it beats the baseline by 16-18 F1 points (0.617 and 0.696). On held-out games it loses: F1 split 0.455 vs 0.502, F2 split 0.274 vs 0.287, pooled -3.0 instance and -4.6 group; worst is bp35 at 0.04. It memorises public sprite designs and does not transfer to unseen ones, which is the key risk for the hidden games. It costs 16 ms on CPU with torch. Training wall time was 15 and 28 min per fold; the second was slowed by other jobs' load. I did not run the CNN-only or 2-frame ablations.
- 5 Behavioural role tracker (roles_eval.py), online and causal. A segment counts as the player if it moves on more than 30% of arrow presses and more than 60% of the time in the arrow's direction. With plain colour components the player is right by level end in 74% of 58 eligible levels, and committed within 5 actions in 48%. Oracle GT segments give the same (72% / 47%), so segmentation is not the bottleneck. The failures are in the rule: several controllable pieces (re86, ka59), co-moving sprites (sc25, m0r0), and names that change under occlusion. A ratio-based rule was worse (66% / 40%). Click-target identity does depend on segmentation: colour components 73%, lexicon 83%, oracle 100%. HUD detection from edge-strip change frequency is weak (P/R 0.40/0.40). I did not evaluate blocker or collectible roles, or the cross-frame identity metric (time box).
- Housekeeping: I deleted the scratch venv (789 MB); the command to recreate it is at the end of results.md. Kept: the tiny model weights (bench/weights/F1.pt and F2.pt, 3.3 MB each), MobileSAM weights (vision/sam_weights, 40 MB) and the full GT dump (16 MB). Nothing was installed into the repo venv, and no repo files were touched. The shared lock is managed by lock_broker.sh, which frees it whenever fewer than 3 heavy jobs run, so lock holders do overlap by design. That explains the earlier 'lock removed by another process' note.

### Best
No candidate passes the gate for a Kaggle arm, so I'm not recommending a learned vision model for objects. MobileSAM, the off-the-shelf ViT, is the most accurate (+8.2 instance F1, +9.1 group), but at 1.9-3.2 s and 5 GB per frame on CPU it can't run on Kaggle. The tiny ViT trained on engine ground truth scores below the plain colour-component baseline on games it has never seen (-3 F1), even though it wins by 16-18 points on the games it trained on. That is the wrong behaviour for 110 hidden games with new sprites. The best deployable segmenter is lexicon cc mode (4 ms, deterministic, +3.5 F1 overall and +3.1 on held-out games), but the gain is small; its clearest use is identifying click targets (83% vs 73%). The most useful next step is the behavioural role tracker (gap analysis Stack A) on the existing colour components. Perfect segments did not improve player identification (72% vs 74% correct by level end), so the work belongs in the tracking rule. Re-run roles_eval.py until it reaches the 80% within-5-actions gate before building a facts-in-opener arm.

## Recommendation (builder)

Short answer: don't ship a learned vision model. I found such models and tested the closest fits against exact engine ground truth: 1,000 frames from all 25 public games, rendered pixel-exact from the game source. None passes the gate, and the evidence says the problem isn't a visual one.

1. MobileSAM, a small off-the-shelf ViT (Apache-2.0, 40 MB), is the most accurate. Instance F1 is 0.550 against 0.468 for colour connected components on the same 250 frames (+8.2, still under the +10 gate). It mostly wins by grouping multi-colour and textured sprites (m0r0 0.04 to 0.35, bp35 0.06 to 0.48), but it merges tile maps. On CPU it costs 1.9-3.2 s per frame and 5 GB of RAM. The GPU has about 1.2 GB free, and MEMFRAC 0.97 already caused a CUDA OOM. It can't run on Kaggle.
2. The tiny hybrid CNN-ViT trained on engine ground truth (845k params, 16 ms on CPU) beats the baseline by 16-18 F1 on the games it trained on. On 10 held-out games it loses (-3.0 instance F1, -4.6 group F1). It memorises the public sprite designs, which is the wrong behaviour for 110 hidden games. So the honest answer is that nothing we can train on the public games generalises to unseen sprites.
3. Lexicon `cc` mode is the best deployable segmenter: about 4 ms, deterministic, numpy/scipy only, no weights. It scores +3.5 F1 overall and +3.1 on held-out games. Its clearest win is click targets: the segment under a click is the real object 83% of the time, against 73% for colour components.
4. The decisive finding: perfect oracle segments do not improve player identification (72% correct by level end against 74%; committed within 5 actions 47% against 48%). The errors come from the tracking rule: several controllable pieces (re86, ka59), sprites that move together (sc25, m0r0), and names that change under occlusion. Roles come from behaviour, not appearance. The model already names roles in its reasoning ('goal' 36%, 'wall' 31%, 'player' 17%); what it lacks is verified behavioural evidence.

What to build is the gap analysis's Stack A: a host-side behavioural entity tracker on the existing colour components, with lexicon-cc grouping and cell size folded in. It pushes short, evidence-counted facts into the opener; it is not an optional view.
- Group sprites by motion: components that move together under an action are merged into one entity. This repairs multi-colour shattering using behaviour, not appearance, so it carries over to unseen sprite designs.
- Track entities across frames by Hungarian matching, reusing frame_diff, so names survive occlusion.

Before any Kaggle compute, roles_eval.py has to clear its gate: 80% committed within 5 actions, and a low rate of wrong claims. Only then build the arm (`arm/entity-facts`).

If later evidence shows the touching same-colour case (10.1% of instances) is a measured failure, revisit a learned grouping net trained mainly on procedurally generated sprites. Until then, MobileSAM stays an offline oracle for benchmarking only.

For `arms.md`: log 'learned object segmenter (MobileSAM zero-shot / tiny engine-trained CNN-ViT)' under Didn't work (screen level, no Kaggle run). Log 'behavioural entity tracker facts (Stack A)' as the top perception item under Not yet implemented, gated on roles_eval.

### Integration spec (needs the revisions below)
Flag `ARC3_ENTITY_FACTS=1`, off by default. tests/test_flags_off_identity.py must still pass with it off. Document the flag in CONFIGURATION.md in the same PR.

Module: `inference/utils/entity_tracker.py`, pure numpy/scipy, runs host-side in the harness process. The sandbox runs `python -I -S` with no packages, so it cannot run there. It keeps per-level state and is updated once per executed action with (prev_frame, action, new_frame). It does three things:
- Segments with lexicon-cc, falling back to segment_layer.
- Matches components to entities by Hungarian assignment (reuse frame_diff), then merges components that keep moving together into one entity.
- Updates an evidence ledger for each entity.

Stable names: E1, E2, ... in order of first appearance on the level. A name is never reused within a level, and the ledger resets on level change.

Roles are stated only as observed behaviour, never in photo vocabulary. No 'key' or 'door'; goal inference stays with the model. Each fact is stated only once it has enough evidence, with counts:
- moves with arrows (n/m presses in the arrow's direction, step size in px and in cells);
- blocks Ex (Ex failed to move into it k times);
- vanished when Ex overlapped it (step s);
- changes every step (likely counter or timer, shown only once its precision is validated; HUD detection is currently P/R 0.40);
- click at (r,c) -> Ey changed (moved, recoloured or vanished);
- cell size N px (lattice R x C).
Tentative claims carry '?'. A claim that later evidence contradicts is reported as withdrawn on a new line, never rewritten in place.

Placement: the dynamic opener section, straight after the valid-actions line where facts-stack puts its lines, which is before `static_start` (tool_agent.py:5490). That keeps it safe for ARC3_OPENER_DEDUP. The text is computed once per step and stored, so a rebuilt opener is byte-identical. The block has a hard 250-token budget and is emitted only when something changed since the last opener.

The system prompt gets one constant clause, following the ARC3_AUTO_FRAME_DIFF pattern (tool_agent.py:2988-2995): 'Entity lines are harness-computed from your actions; counts are evidence; names are stable within a level.'

No sandbox attribute in v1, so the 'exposes only' lines (prompts.py:161, tool_agent.py:354-362, 5493-5496) stay unchanged. If a v2 adds `current_frame.entities`, those lines must be edited in the same commit; the lexicon view was used in 0 of 1,130 calls until that was fixed.

Example opener block:
```
Entities (harness-tracked, this level):
- E1 moves with ACTION1-4: 9/9 presses, 4 px = 1 cell per press; cells r5-5 c2-2; colours 9,12
- E3 blocked E1 3 times (E1 pressing LEFT at r5 c1)
- E4 vanished when E1 overlapped it (step 14)
- Cell size 4 px (16x16 lattice)
```
Logging: write the per-step ledger to the events jsonl so the viewer and Stage-0 scripts can audit every claim against the truth.

### Kaggle packaging
Nothing new to ship. No Kaggle dataset, no weights and no GPU memory. It needs numpy and scipy (both already in the Kaggle image; verify scipy is importable from the harness venv) plus about 300 lines of our own code (licence CC0/MIT-0). It runs on CPU in the harness process: lexicon-cc takes about 4 ms per frame and the tracker update about 1 ms, so roughly 5 ms per action. Over about 5,700 actions per 25 games that is under 30 s of total CPU, against 41-46 s median wall time per turn. That leaves the 1.2 GB of GPU headroom and the ~60 GiB of free RAM untouched. The arm should also log `os.cpu_count()` and the per-call tracker time, since our logs never print the CPU count (a competitor on the same GPU shape recorded 48).

For reference, the rejected learned options would have needed:
- MobileSAM: a 40 MB dataset, 1.9-3.2 s per frame on CPU and 5 GB RSS (not viable), or GPU memory we don't have.
- Tiny CNN-ViT: a 3.3 MB weights dataset plus torch or onnxruntime on CPU, 16 ms per frame. Packaging is easy, but it is rejected because it fails on held-out games.

### Expected effect
Small and uncertain. The prior is +0.5 to +2 points on the 25 public games over the 56.94 v4-user-combo-10x baseline, inside run-to-run noise. Confirming it will take at least 3 paired full runs.

Why it isn't zero:
- These are facts pushed into the opener, not an optional view. The competitor's lexicon view showed no score change partly because it went unused (0% of calls, then 12% after the prompt fix).
- Recognising walls, blockers and collectibles from behaviour feeds directly into goal inference, which is where most points are lost.
- The click-target gain (83% against 73%) helps the 11 click-moved games.

Why it's small:
- The binding limit is goal inference, and these facts only narrow it.
- The model already names 'player' and 'wall' itself in its reasoning.
- The tracker is only 74% right by level end today, so the gate exists to stop wrong facts from anchoring the model.

The gap analysis estimated +0.6..+2 (tracker) and +0.5..+1.5 (lattice), and those overlap, so don't add them.

Test plan:
- **Stage 0 (offline, free):**
  - roles_eval.py on the 5,717-step ground-truth replays, gated on: 80% committed within 5 actions; wrong committed claims at most 5%; blocker and vanish-on-contact precision of at least 90% on the levels where they apply.
  - Replay the tracker over every opener of v4-user-combo-10x. Check the mean and max of the block's tokens, that a rebuilt opener stays byte-identical, and count turns where the model's reasoning contradicted a correct fact. If the model was rarely confused about the facts the tracker would state, expect about zero gain and stop there.
- **Stage 1 (screen):** a 20-25-min panel of the 13 controlled-sprite games plus the 11 click games against a matched baseline, at matched wall-clock. Judge it on usage (share of reasoning or code that names E-ids or uses the facts, target above 30%; 'fill is not use'), level clears, and tokens per request (keep the increase under 3%). The prefix-cache rate must stay at about 89%.
- **Stage 2:** full 121-min runs, at least 3 repeats against the baseline, compared with `make significance` (paired per game). It counts as an alpha only if the repeats score consistently higher.

## Independent review

**PARTLY SUPPORTED; REVISE BEFORE ADOPTING. The headline answer holds up against the benchmark artifacts (/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision/bench/results.md and results/*.json, which I re-checked). Neither MobileSAM nor the tiny CNN-ViT should ship, and the per-game role numbers reproduce exactly: colour components 43/58 correct at level end and 28/58 committed within 5 actions; oracle segments 42/58 and 27/58; lexicon 37/58 and 11/58. The ~5,700 actions per run and 41 s median turn time also check out against v4-user-combo-10x v1 (1,211 turns, 4.7 actions per turn). Four things are wrong. (1) Some of the reasons given for rejecting MobileSAM are overstated or contradict the recommendation's own numbers. (2) The 'build Stack A' spec goes against its own benchmark and the gap analysis it cites: it picks a segmenter that hurts tracking, an opener channel close to the STATE_TRACKER dead end, a cell-size fact that is wrong about 21% of the time, and claims occlusion handling that frame_diff cannot do. (3) The tracker latency figures are assumed, not measured. (4) The test plan cannot confirm an effect of the stated +0.5 to +2. Keep the 'no learned segmenter' conclusion and log it under Didn't work. Rewrite the tracker spec before it goes into arms.md as the top perception item.**

- CONTRADICTS ITS OWN BENCHMARK: the integration spec says 'Segments with lexicon-cc, falling back to segment_layer', but lexicon segments make the tracker much worse. Player correct at level end drops from 43/58 to 37/58. Committed within 5 actions drops from 28/58 (48%) to 11/58 (19%), and is worse in 9 of 12 games. Median actions to commit rises from 3 to 25 (results/roles_lex.json vs roles_cc.json). results.md itself concludes 'plain colour CCs are good enough input' and suggests lexicon only for a click-target fact (73% -> 83%). The recommendation mentions only the lexicon's gains.
- CHANNEL REPEATS A NEAR-DEAD END: research/reports/harness-systems-gap.md designs G1 (the tracker this calls 'Stack A') as 'result-dict fields the model reads in code, not an opener block', plus one tail line of at most 40 tokens. An opener 'Entities' block is a harness state block in the user turn, rank 5 in that report's channel ranking. Its precedent, STATE_TRACKER, was cited in about 6% of requests (arms.md:215). v1 also has no sandbox attribute, so the model cannot use E-ids in code (rank 2 channel). Saying 'pushed facts, not an optional view' therefore does not support a non-zero effect or the 30% usage target. Per the Flash-Next memory, free text in the user turn has been ignored before.
- CELL-SIZE FACT IS UNSUPPORTED: the spec and its example ('Cell size 4 px (16x16 lattice)') state cell size as a fact. Our lexicon estimator is right on only 78.9% of frames (85.8% excluding bp35/lf52) and fails on ar25, m0r0 and tu93 (results.md section 1). The gap analysis says the specified lattice detector failed its Stage 0. The recommendation leaves out the 78.9% and implies the competitor's 93%. A roughly 21% error rate breaks its own 'wrong committed claims at most 5%' bar, and the cell coordinates in the other facts ('cells r5 c2') inherit the error.
- OCCLUSION CLAIM IS WRONG ABOUT frame_diff: 'Hungarian matching, reusing frame_diff, so names survive occlusion'. But ARC3-Inference/inference/utils/frame_diff.py:82-230 only runs Hungarian assignment within identical-hash groups (same colour and shape). An occluded or reshaped sprite is reported as disappeared plus appeared, which is exactly the occlusion failure the role benchmark found. The gap analysis also notes that 52% of auto-diff 'moved' entries are 2 px or less of noise.
- CO-MOTION GROUPING CAN MERGE WHOLE BOARDS: the gap analysis's G1 removes the best global translation (view_shift) before matching, and the spec drops that step. On a scrolling or custom camera, every component moves together under an arrow, so 'merge components that move together' collapses the board into one entity. The recommendation names motion grouping as the defence for custom-camera games when it is the weak point. sc25 and m0r0, where other sprites also move with the arrows, are the observed failures that co-motion merging could make worse. Separately, oracle segments do not improve the player rule (72% vs 74%), so a grouping improvement is not expected to lift player identification.
- TRACKER LATENCY IS ASSUMED, NOT MEASURED: '~1 ms tracker update' is really the scipy colour-component segmentation time in roles_eval.py, not a Hungarian match plus ledger update. Lexicon's 3.8 ms is a standalone figure; inside the tracker it measured 16-60 ms under load. The segment_layer fallback is 32-110 ms median, 540 p95 and 2,057 max. segtime.py times frame_diff, but its output was never reported. The '~5 ms per action, under 30 s per run' claim is therefore not real. It is still small against a 41 s median turn, but the number should be measured.
- MOBILESAM REJECTION IS OVERSTATED AND INTERNALLY INCONSISTENT: it says '5 GB of RAM ... can't run on Kaggle', yet the packaging section cites about 60 GiB of free RAM. A competitor's Kaggle boot log shows 176.9 GiB total and 48 CPUs (competitors/rellik13-milestone2/run/sglang_boot_result.json), minus our 95.4 GiB PLE table. CPU latency was measured on a loaded laptop at 1-4 threads against a self-set 50 ms per-frame budget. The task's budget is under ~1 s per frame. Run once per turn (about 1,211 turns per run, 41 s median) on a 48-core VM, it might fit; that is unmeasured, as are EdgeSAM, EfficientViT-SAM and ONNX/int8 variants. GPU headroom is stated as 1.2 GB, but v4-user-combo-10x v1 serve.log shows 0.52 GiB free during serving, so the GPU is even tighter. The valid reason to reject is value, not feasibility: oracle segments do not help the player rule, tile-map recall falls from 0.55 to 0.34, and the lexicon view showed no score effect.
- GENERALISATION NEGATIVE IS OVERCLAIMED: 'nothing we can train on the public games generalises to unseen sprites' rests on one 845k-param model trained for 4,000 steps on 2 folds. Its synthetic frames were built only from training sprites, and the CNN-only and 2-frame ablations were skipped. Fold F2 held-out is nearly tied (0.274 vs 0.287); F1 is -4.7. The evidence supports 'this recipe did not generalise', not a universal negative.
- 'THE PROBLEM ISN'T VISUAL' OVERREACHES: the evidence covers player identification (oracle = colour components) and click targets. On click targets segmentation does matter: 73% for colour components, 83% for lexicon, 100% for oracle. Goal, key and door roles, which often rest on visual matches (same design or colour), were not evaluated. Role words in the reasoning ('goal' 36%, 'wall' 31%, 'player' 17%) are word-presence counts and say nothing about whether the roles are correct.
- EXPECTED EFFECT AND POWER: +0.5 to +2 is borrowed from G1's estimate, which used a different channel and design (result-dict avatar plus view_shift, a tail line of at most 40 tokens). It is quoted against 56.94, the best of three baseline runs (56.94 / 52.70 / 49.09, mean 52.91, sd about 3.9). The gap analysis says two runs resolve only about +6 to +10, so 'at least 3 paired full runs' cannot confirm a +0.5 to +2 effect. The plan should say a score effect is likely undetectable and that the decision rests on Stage-0 and Stage-1 mechanism metrics.
- TOKEN BUDGET CONFLICTS WITH ITS OWN GATE: measured on v4-user-combo-10x v1, a request carries a mean of 12.7 retained openers and a 54.8k-token prompt. A 250-token block per opener adds about 3.2k tokens (about 5.9%), above the 'under 3%' Stage-1 gate. G1 budgeted 30-60 tokens per acting turn. The cap should be about 100 tokens.
- STAGE 0 IS IN-SAMPLE: the tracker rule's thresholds and failure diagnosis come from the same 58 levels in 12 games that the gate will re-score. That is the overfitting standard used to reject the CNN-ViT. Freeze the rules on a dev subset of games and gate on held-out games. The gate also drops two of G1's bars: 0 false avatars in click games, and ID stability of at least 90%.
- SMALLER ERRORS: (a) 'the touching same-colour case (10.1%)' understates it. 31.6% of instances touch a same-colour object; 10.1% touch one of a different design. The competitor's ~23% matches our macro figure of 23.4% (gt/summary_games.json). (b) 'Emitted only when something changed since the last opener' must re-emit the current facts after a trim or rollback. The repo has hit this bug class before (step-bar fix #100/#102). (c) The Stage-1 panel of '13 controlled + 11 click games' is 23 of the 25 games, since sc25 is in both, so it is barely cheaper than a full run. (d) scipy is not declared in pyproject.toml; it arrives only through tufa-arc-agi-framework. The recommendation's 'verify scipy is importable' check is needed.

## Literature and constraints (research stage)

Short answer: yes, these models exist, and the small ones would fit on Kaggle. But none of them is likely to give Qwen object facts it can't already get, and the main open problem (object roles) isn't a visual problem. I found no ARC-AGI-3 system that uses a learned object detector. Every one I found uses colour connected components, diffs or raw images. The strongest new result, VISTA (Kaiming He's group, 2026-10-01), deliberately uses no segmentation: raw 8x-upscaled images plus `inspect` (zoom into any region of any past frame) and `read_pixels` tools. With frontier models it went from 70.05 to 94.10 to 99.00 RHAE; GLM-5.3 Flash 320B scored 66.93. On 64x64 frames with 16 exact colours, exact algorithms already find masks and repeats perfectly. Off-the-shelf detectors and VLMs are trained on natural photos. COCO-trained YOLO matched nothing on low-res game art. SAM fails on low-texture line art. VLMs can't reliably count grid rows. In ARC-AGI-3, roles (player, key, wall, counter) come from behaviour, not looks.

Ranked by fit to our constraints (offline, CPU well under 1 s per frame, little VRAM, unseen sprites, licence allowed):

1. **Best fit: no download. A role tracker built from diffs, plus optionally a tiny grid-native net we train ourselves.** Roles come from behaviour: what moves when you press an action is the player; what blocks movement is a wall; what vanishes on contact is a collectible; HUD bars that change with every step are counters. These go into tool results as facts. That is the gap analysis's Stack A (controllable-entity tracker), with the lexicon view's cell-size and multi-colour cutting folded in. For the ~23% of objects that touch a same-colour neighbour, a small CNN or U-Net (under 1M params, native 64x64) could predict which cells belong to the same sprite. Training labels are free: the engine source for all 25 public games is in `competitors/polyphony-arc-3/vendor/arc/environment_files`, and each Sprite has pixels, x, y, layer, scale and rotation. VARC is the architectural template: an 18M ViT with 2x2 patches on a 64x64 canvas, trained from scratch on ARC, that generalises to unseen tasks. My estimate is milliseconds on CPU. The risk is generalisation to new sprite designs; palette permutation and procedurally generated sprites help.
2. **YOLO11n-seg (2.9M params, 6.2 MB, 24 ms on an M4 Air CPU) or FastSAM-s (11.8M, 23.9 MB, 58 ms).** Fast enough on CPU and cheap to fine-tune on those engine masks. Licence is AGPL-3.0, which the ARC rules appear to allow for third-party code. Zero-shot they will probably fail on pixel art, so this is option 1 with more weight.
3. **SAM-family point-prompted maskers: MobileSAM (10.1M, 40.7 MB), EfficientSAM-Ti, EdgeSAM (all Apache-2.0), SAM 2.1-tiny (38.9M, ~151 MB as ONNX, Apache-2.0).** The input is fixed at 1024x1024, so a 64x64 frame costs as much as a photo. MobileSAM's own demo is ~3 s on a Mac i5 CPU; segment-everything mode takes 23-24 s on an M4 Air. Too slow unless run on the GPU, which competes with SGLang. EfficientViT-SAM-L0 (35M, Apache-2.0, 762 images/s on A100) is GPU-only in practice. SAM 3 (848M, custom licence, gated weights, CUDA) is out.
4. **Open-vocabulary naming: OWLv2-B/16 (155M, Apache-2.0, ~0.4 s per image on CPU), Grounding DINO-tiny (172M, Apache-2.0, 331 ms p50 on an L4 GPU), YOLO-World-S (~13M, GPL-3.0), Florence-2 base/large (0.23B/0.77B, MIT, ~1-3 s per caption on CPU).** Their names come from photo vocabularies, so abstract sprites would get wrong labels such as "key" or "door", which could anchor the model. OWLv2's find-by-example detection is already done exactly by the harness's `shape_hash`. The closest "hybrid" design is Microsoft's OmniParser V2 (a YOLOv8 detector plus a fine-tuned Florence-2 captioner, 0.6 s/frame on A100). It only works because it was fine-tuned on its own labelled data, and we have no role labels for unseen games.
5. **Small VLMs: SmolVLM-256M/500M (Apache-2.0), Moondream 2 (1.9B, Apache-2.0, with a 0.5B variant), Qwen3-VL-2B (Apache-2.0); not Qwen2.5-VL-3B or PaliGemma 2 3B.** Qwen2.5-VL-3B's research licence probably isn't allowed; PaliGemma 2's Gemma licence is doubtful. All of these are weaker at seeing than Qwen3.8-Flash-Next, which already sees the same 10x image, and each caption is a multi-second extra model call.
6. **ARC research models, as ideas rather than drop-ins.**
   - ViTARC: 2.8M params, but trained per task for 6-10 GPU-hours, so not usable at runtime. Its useful idea is giving object indices as a positional input, with object indices from plain OpenCV contours.
   - Slot attention / Slot Abstractors: no ARC-AGI-3 evidence.
   - AXIOM: learns object identities online from pixels, but needs thousands of steps.
   - MarioNette: learns its sprite dictionary from videos of each game.
   - ARGA and Ferré's MDL work: symbolic object abstractions.

Constraint notes:
- The Kaggle box is probably a GCP g4-standard-48 (48 vCPU, 180 GB). This is inferred from 177 GB RAM in one local note and not verified, so it's worth logging `os.cpu_count()`.
- ARC rules allow third-party code under licences that permit public sharing (Apache-2.0, GPLv3, so AGPL should be fine) and pretrained models that are freely available. Gated SAM 3 and the research and Gemma licences are doubtful.

What I'd do: build Stack A's behavioural tracker as harness facts first. Add a tiny engine-trained grouping net only if the touching same-colour objects show up as a measured failure. Separately, VISTA's lossless frame memory plus `inspect`/`read_pixels` is a candidate arm, but its evidence is from frontier models only. `research-sweep.md` already dropped a bbox-cropped zoom diff image, so it needs a Qwen screen before any repeats.
I checked our side read-only, for a hybrid ViT-style object identifier that would feed Qwen.

**What the model gets today.** It sees a 640x640 board image (402 tokens), a diff image, and animation verdicts plus a short digest in the turn opener. Single-colour 4-connected segmentation and `frame_diff` exist only as python-callable helpers. Auto frame diff and level inventory were off in v4-user-combo-10x.

**How much the model uses them.** Across 2,734 python calls it reads `.ascii` row slices in 42%, `.segmentation` in 25%, `frame_diff` in 10% and animation data in 6%. The identity fields are almost never touched: adjacency 0.9%, hash 0.5%, children 3.5%. The model does infer roles in prose: "goal" appears in 36% of reasoning blocks, "wall" in 31%, "player" in 17%. The harness supplies no roles at all.

**Kaggle VM.**
- GPU: about 1.2 GB free at steady state (nvidia-smi 94.4/95.6 GB). MEMFRAC 0.97 already caused a CUDA OOM at about 90 min.
- RAM: 176.9 GiB, with about 60 GiB free during the run.
- CPU: 48 vCPU, a measured count from a competitor's boot log on the same GPU shape; our own logs never print it.
- So the perception model has to run on CPU, or fit in well under 1 GB of VRAM including its CUDA context.

**Where it would plug in.** It cannot run inside the sandbox, which is `python -I -S` with no packages. It must run host-side in the harness process.
- Text facts go in the turn opener's dynamic section: next to `_prepare_auto_diff` at tool_agent.py:5464, or straight after the valid-actions line where facts-stack puts its lines. Both are before `static_start` (tool_agent.py:5490), which keeps them safe for `ARC3_OPENER_DEDUP`.
- The tail of the action's tool result reaches the model at the same moment, because a turn ends as soon as an action executes (tool_agent.py:7837-7869).
- A mask image would go in `_build_user_message`.
- Any new python-side view must also edit the "exposes only" prompt lines (prompts.py:161, tool_agent.py:358). That is what kept the lexicon view unused.
- Output must be deterministic or stored once, so a rebuilt opener stays byte-identical.

**Latency budget.** A turn takes a median 41-46 s of wall time (about 1.5 requests and 4.7 actions per turn). Prefix-cache reuse is 89.5%. The current pure-Python segmentation takes 32 ms median and 253 ms worst case per 64x64 frame on a laptop. Running a CPU model under 1 s once per turn opener costs under 2.5% of stream time. Per action it would run about 5,700 times per 25 games.

The measurement scripts are in /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/vision/: usage.py, roles.py, latency.py, latency2.py and segtime.py.
I built the ground-truth set: 1,000 frames from all 25 public games, 40 per game, and every frame matches the recorded board. The script loads each game's source with arcengine 0.9.3 from the repo venv (I installed nothing), replays the actions recorded in runs/kaggle/v4-user-combo-10x/v1/artifacts/<game>_p0_events.jsonl.xz, and checks each rendered frame against the recorded board. All 5,717 replayed steps (25 initial resets plus 5,692 actions) matched pixel for pixel. Frame sampling keeps the first frame of each level and spreads the rest evenly over the replay, with no duplicate frames.

**How the per-pixel instance map works.** For each frame, every sprite's pixels are swapped for ID codes and the game's own camera._raw_render is called. This reproduces layer order, rotation, mirroring, scale, letterboxing and the custom cameras in s5i5, tu93 and wa30 exactly. HUD pixels come from wrapping Camera.render and diffing the frame before and after each RenderableUserDisplay.

**bp35 and lf52 needed an adapter.** They don't use arcengine sprites: they run a private scene graph that is drawn into a 64x64 buffer, and a UI display returns that buffer as the whole frame. The adapter deep-copies that scene, gives each drawn entity an ID-coded stand-in image, and calls the game's own drawing function on the copy, so the live game is untouched. A consistency check now passes on all 25 games: outside HUD pixels, the frame equals the scene render, and every instance pixel's colour belongs to its sprite.

**Semantic roles.** The sources almost never name roles: sprite names and tags are obfuscated random strings in essentially all 25 games. The only readable names are bp35 "player", lp85 "goal"/"button"/"tile", dc22 "bg"/"Bridge", ka59 "Enemy", sc25 "...-ui"/"action-ui"/"...-button" and tr87 "background". So there are no source-given player/key/door labels to train on. The usable labels are:
- the engine tag sys_click (clickable), found on 14 of 25 games;
- visible/collidable/layer/blocking flags;
- HUD membership;
- roles worked out from the replay: a sprite counts as controlled when it moves on more than 30% of arrow actions and in the arrow's direction more than 60% of the time.

Controlled sprites were found in 13 games (ar25, cn04, dc22, g50t, ka59, ls20, m0r0, re86, sc25, sk48, sp80, tu93, wa30). Click-moved sprites were found in 11 games (bp35, cd82, lf52, lp85, r11l, s5i5, sb26, sc25, su15, tn36, vc33). Recolouring, rotation and spawn/despawn are also recorded.

**Segmentation findings (26,590 visible instances).**
- Same-colour connected components match a sprite exactly for only 43.9% of instances; 31.0% are split into several components and 31.7% are merged with another sprite.
- 31.6% of instances touch a same-colour neighbour from another sprite, but most of that is tile maps (ls20 89%, ar25 81%). Only 10.1% touch a same-colour sprite of a different design, which is the genuinely ambiguous case; the competitor reported about 23%.
- 17.0% of sprites are multi-colour (cd82 88%, tu93 83%, re86 72%).

**Sprite designs per game** (distinct pixel arrays seen in the sampled frames): ar25 42, bp35 11, cd82 54, cn04 40, dc22 35, ft09 48, g50t 40, ka59 27, lf52 45, lp85 11, ls20 26, m0r0 11, r11l 58, re86 57, s5i5 82, sb26 46, sc25 47, sk48 11, sp80 12, su15 15, tn36 45, tr87 35, tu93 15, vc33 106, wa30 14. These over-count real types, because a sprite whose colours change mid-game gets a new hash each time.

**Caveats.**
- A sprite instance is not the same as a perceptual object: walls are often built from many same-design tiles (ls20 averages 98 instances per frame), so tiles probably need grouping by design and adjacency before this set is used as object labels.
- The labels cover only states the Kaggle agent actually reached.
- I did not check the Kaggle VM CPU count; it wasn't needed for this step.
- Once during a run, the shared lock directory was removed by another process before my rmdir. Lock users may be stepping on each other.