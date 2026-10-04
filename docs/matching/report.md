# Hungarian vs mutual unique nearest matching

**Current decision: keep Hungarian.** Mutual nearest below is an evaluated alternative, not the selected production matcher.

[Ten illustrated cases](visual_cases.md) · [Download the 10-page PDF](figures/matching_examples.pdf)

The replacement changes behavior materially. On this synthetic suite, Hungarian captures **231/271 supported transformed pairs (85.2%)**, versus **199/271 (73.4%)** for mutual unique nearest. Hungarian also makes **72 irrelevant matches**, versus **56**. Both classify every correctly paired object into the expected transform bucket.

These results do **not** justify removing Hungarian as an equivalent simplification. Mutual unique nearest loses matches even when the global minimum-distance assignment is unique. Hungarian can make more wrong assignments when identical objects cross or are replaced. The decision is to retain Hungarian for the pending API update. The frozen comparison remains available for later study; this is not a claim that Hungarian always recovers identity.

## What was compared

- **Mutual:** the frozen mutual-nearest `frame_diff` alternative tested before the decision to retain Hungarian. A pair must be each other's unique nearest valid candidate. Ties are rejected, and rejected candidates cannot fall through to resizing. This is a single pass, not repeated nearest-neighbor matching after each removal.
- **Hungarian:** the original `_hungarian` solver extracted verbatim from commit `64a98a8`, applied to the **same valid candidates** in the current diff. An adapter adds unmatched dummy columns and forbids invalid edges, maximizing valid match count first and minimizing total bbox-center Manhattan distance second. Equal-cost solutions are resolved by the solver's ordering, without ambiguity rejection.
- Both use the same segmentation, hashes, unchanged-node removal, mask verification, center tolerance, recolor/scaling rules, and resize fallback. Only the exact-stage selector changes. The resize fallback still uses mutual tie rejection in both variants. Different exact assignments can affect which nodes reach that fallback.
- This is a controlled assignment comparison, **not** a comparison against the entire historical diff API. The old API lacked the new reflection/scaling buckets, so using it wholesale would confound matching with transform support.
- The manual API is tested: `max_group_match=None`. Automatic group caps, prompt rendering, and cropping are outside this experiment; they have separate existing unit tests.

Frozen runtime sources are included so the experiment remains reproducible after production code changes. The Hungarian adapter was checked against exhaustive maximum-cardinality/minimum-cost enumeration for **100 small random candidate graphs**, including incomplete graphs and ties.

## Ground truth and confusion-matrix definitions

Each scene has generated objects with persistent **fixture identities**. These identities are used only for scoring and are never given to either matcher. Masks, colors, and positions generate actual grids; both grids are segmented by the same runtime. Assertions ensure each labeled foreground object becomes exactly one node, with no accidental merging or splitting.

The scoring unit is one ordered **before-object / after-object pair**, over the full foreground Cartesian product:

| Ground truth | Captured as a matched diff pair | Not captured |
|---|---|---|
| Same fixture identity, with a visible change | TP: transformed and captured | FN: transformed but not captured |
| Different identities, or same identity with no visible change | FP: irrelevant but captured | TN: irrelevant and not captured |

A wrong assignment can therefore cause both an FP for the wrong pair and an FN for the missed true pair. Merely returning `appeared`/`disappeared` does not count as capturing a transformation. Background nodes are passed to both matchers but excluded from scoring. The per-case artifact records their emitted-pair count separately.

The main matrix counts **all visible transformations**, including unsupported combinations. A second matrix excludes changed true pairs that cannot be expressed under the agreed single-change contract. It retains all negatives. An independent dense-matrix oracle determines contract support and expected bucket; it does not call the production classifier or use its hashes.

For symmetric shapes, a performed operation may produce no visible change, or a combination may be equivalent to one allowed transform. Ground truth uses the resulting observable mask/color/position, not an assumed recoverable action history. For example, rotation followed by reflection can still be a single reflection in D4. Correct pair but wrong bucket is audited separately: **zero cases for both methods**.

## Test coverage

Seed: `20261003`. **684 scenes**, **4,884 foreground pair decisions**; 646 changed true pairs and 4,238 negatives.

- **48 single-operation scenes:** movement, quarter-turn rotation, reflection, exact integer scaling, recoloring, and geometry change (`resized`), with stationary distractors.
- **456 combination scenes:** all 57 subsets of size 2–6 of those six operations, eight fixtures per subset. Execution order is geometry change → reflection → rotation → enlargement → recolor → translation. This covers every subset, not every permutation or repeated operation sequence.
- **56 parameter scenes:** all four reflection axes, 90/180/270° rotations, scale changes 1→2, 2→1, 2→3, 3→2, and center displacements around the one-pixel tolerance.
- **28 negative controls:** independent disappearance and appearance that visually resemble each transform, plus far-away unrelated geometry.
- **96 assignment/control scenes:** competing identical objects, locally tied but globally unique assignments, globally tied assignments, crossings, stationary duplicates, replacement, survivor plus birth/death, unchanged scenes, pure birth, and pure death.

Four base shapes exercise chiral and symmetric geometry, open concavity, and a hole. Combination scenes contain three foreground objects per frame. Assignment scenes use identical point sprites to isolate assignment from shape recognition. Complete object masks, positions, identities, predictions, misses, and false pairs are saved in [cases.jsonl](data/cases.jsonl).

## Confusion matrices: all visible transformations

### Mutual unique nearest

| Actual pair | Captured | Not captured |
|---|---:|---:|
| Transformed | **199 TP** | **447 FN** |
| Irrelevant / unchanged | **56 FP** | **4182 TN** |

Precision: **78.0%**. Recall: **30.8%**.

### Hungarian

| Actual pair | Captured | Not captured |
|---|---:|---:|
| Transformed | **231 TP** | **415 FN** |
| Irrelevant / unchanged | **72 FP** | **4166 TN** |

Precision: **76.2%**. Recall: **35.8%**.

The 375 out-of-contract positives consist of **369 combined changes** and **6 parameter cases outside the center tolerance**. Both methods intentionally leave all 375 unmatched. They explain much of the low recall above; replacing the assignment solver cannot add unsupported transform types.

## Confusion matrices: supported transformations only

Same negatives as above; the 375 out-of-contract positive pairs are excluded.

| Method | Transformed, captured (TP) | Transformed, not captured (FN) | Irrelevant, captured (FP) | Irrelevant, not captured (TN) | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| mutual | 199 | 72 | 56 | 4182 | 78.0% | 73.4% |
| hungarian | 231 | 40 | 72 | 4166 | 76.2% | 85.2% |

All supported isolated transforms and parameter variants are captured by both methods. The remaining supported-pair misses occur in the assignment stress suite. Hungarian gains **32 true captures** and **16 additional false captures** over mutual in the aggregate.

## Results by suite

Entries below use the all-transform definition. The high number of easy stationary/cross-identity negatives makes overall accuracy uninformative, so precision and recall are preferred.

| Suite | Method | TP | FN | FP | TN |
|---|---|---:|---:|---:|---:|
| single | mutual | 43 | 0 | 0 | 389 |
| single | hungarian | 43 | 0 | 0 | 389 |
| combination | mutual | 83 | 369 | 0 | 3652 |
| combination | hungarian | 83 | 369 | 0 | 3652 |
| parameters | mutual | 41 | 6 | 0 | 9 |
| parameters | hungarian | 41 | 6 | 0 | 9 |
| negative_controls | mutual | 0 | 0 | 24 | 4 |
| negative_controls | hungarian | 0 | 0 | 24 | 4 |
| assignment | mutual | 32 | 72 | 32 | 128 |
| assignment | hungarian | 64 | 40 | 48 | 112 |

In the assignment suite alone, mutual has **50.0% precision / 30.8% recall** and Hungarian has **57.1% precision / 61.5% recall**. The aggregate precision ordering differs because the other suites change the mix of true and false captures. These fixture proportions are deliberate stress coverage, not estimates of game-event frequency.

Pure birth and pure death controls pass **8/8 each**, for both methods. Across all scenes containing actual births/deaths, both capture **20/52 births and 20/52 deaths**. The rest are visually plausible replacement events incorrectly paired as transforms. Appearance/disappearance are unary events and are not mixed into the pair confusion matrices.

## Concrete cases

### Local competition: mutual both misses and mispairs

Case `local_competition_ordered/0`: two identical pixels remain on row 20. Fixture A moves from column 8 to 11; fixture B from 12 to 16.

| Distance in cells | After A: 11 | After B: 16 |
|---|---:|---:|
| Before A: 8 | 3 | 8 |
| Before B: 12 | 1 | 4 |

Mutual selects **before B → after A**, the unique nearest pair. A is then unpaired. Result: **0 TP, 2 FN, 1 FP, 1 TN**.

Hungarian selects **A → A and B → B**, total distance 7 instead of 9. Result: **2 TP, 0 FN, 0 FP, 2 TN**. Thus mutual uniqueness does not guarantee correct identity, and local competition can discard a unique global solution.

### A local tie can have a unique global solution

Case `local_tie_global_unique/0`: before columns 8 and 12, after columns 10 and 16. Both before objects are equally near column 10, so mutual returns no pairs. Hungarian returns the unique minimum-cost full assignment, 8→10 and 12→16. Mutual: **0 TP, 2 FN**. Hungarian: **2 TP, 0 FN**.

### The same images can imply different histories

`local_competition_crossed/0` has exactly the same rendered frames as the first example, but fixture A goes 8→16 and B goes 12→11. Mutual now gets one correct pair; Hungarian gets both identities wrong. Neither algorithm can recover this hidden history from the two images alone.

The negative replacement controls expose the same limit for **every transform bucket**: deleting one object and creating another in a geometrically valid transformed pose looks exactly like transforming the original. Both methods capture 24 such irrelevant pairs; the four far-away different-shape controls are correctly rejected. Hashes and global optimization cannot establish persistent identity by themselves.

### Global ties remain arbitrary under Hungarian

With before columns 8/12 and after columns 16/20, both complete assignments cost 16. Mirroring or transposing the scene can change which fixture identities align with the solver's deterministic tie choice. Hungarian's minimum cost does not imply unique correspondence. A possible follow-up is to reject globally ambiguous assignments while retaining globally unique ones; **that hybrid was not tested here**.

## Every transform and combination

Each tuple is **(TP, FN, FP, TN)**. “Unsupported” counts changed true pairs outside the single-change contract. “Supported” includes combinations whose resulting appearance is expressible by one bucket; it does not mean the original action sequence was recovered.

| Operations | Scenes | Supported | Unsupported | Mutual | Hungarian |
|---|---:|---:|---:|---|---|
| move | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| rotate | 8 | 5 | 0 | (5, 0, 0, 67) | (5, 0, 0, 67) |
| mirror | 8 | 6 | 0 | (6, 0, 0, 66) | (6, 0, 0, 66) |
| scale | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| recolor | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| move + rotate | 8 | 3 | 5 | (3, 5, 0, 64) | (3, 5, 0, 64) |
| move + mirror | 8 | 2 | 6 | (2, 6, 0, 64) | (2, 6, 0, 64) |
| move + scale | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + mirror | 8 | 4 | 0 | (4, 0, 0, 68) | (4, 0, 0, 68) |
| rotate + scale | 8 | 3 | 5 | (3, 5, 0, 64) | (3, 5, 0, 64) |
| rotate + recolor | 8 | 3 | 5 | (3, 5, 0, 64) | (3, 5, 0, 64) |
| rotate + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| mirror + scale | 8 | 2 | 6 | (2, 6, 0, 64) | (2, 6, 0, 64) |
| mirror + recolor | 8 | 2 | 6 | (2, 6, 0, 64) | (2, 6, 0, 64) |
| mirror + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| scale + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror | 8 | 2 | 6 | (2, 6, 0, 64) | (2, 6, 0, 64) |
| move + rotate + scale | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + scale | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + scale + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + mirror + scale | 8 | 3 | 5 | (3, 5, 0, 64) | (3, 5, 0, 64) |
| rotate + mirror + recolor | 8 | 3 | 5 | (3, 5, 0, 64) | (3, 5, 0, 64) |
| rotate + mirror + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| rotate + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + scale + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| rotate + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| mirror + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| mirror + scale + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| mirror + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + scale | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + scale + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + scale + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + mirror + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + mirror + scale + reshape | 8 | 8 | 0 | (8, 0, 0, 64) | (8, 0, 0, 64) |
| rotate + mirror + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| mirror + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + scale + recolor | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + scale + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + mirror + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| rotate + mirror + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |
| move + rotate + mirror + scale + recolor + reshape | 8 | 0 | 8 | (0, 8, 0, 64) | (0, 8, 0, 64) |

## Parameter, negative-control, and assignment detail

Each tuple remains **(TP, FN, FP, TN)**. Zero pair counts for birth-only/death-only are expected because one side of the Cartesian product is empty; unary results are reported above.

| Family | Scenes | Mutual | Hungarian |
|---|---:|---|---|
| `mirror_axis_-` | 4 | (3, 0, 0, 1) | (3, 0, 0, 1) |
| `mirror_axis_\|` | 4 | (2, 0, 0, 2) | (2, 0, 0, 2) |
| `mirror_axis_/` | 4 | (3, 0, 0, 1) | (3, 0, 0, 1) |
| `mirror_axis_\` | 4 | (3, 0, 0, 1) | (3, 0, 0, 1) |
| `rotation_90` | 4 | (3, 0, 0, 1) | (3, 0, 0, 1) |
| `rotation_180` | 4 | (2, 0, 0, 2) | (2, 0, 0, 2) |
| `rotation_270` | 4 | (3, 0, 0, 1) | (3, 0, 0, 1) |
| `scale_1_to_2` | 4 | (4, 0, 0, 0) | (4, 0, 0, 0) |
| `scale_2_to_1` | 4 | (4, 0, 0, 0) | (4, 0, 0, 0) |
| `scale_2_to_3` | 4 | (4, 0, 0, 0) | (4, 0, 0, 0) |
| `scale_3_to_2` | 4 | (4, 0, 0, 0) | (4, 0, 0, 0) |
| `center_tolerance_rotate` | 4 | (2, 2, 0, 0) | (2, 2, 0, 0) |
| `center_tolerance_mirror` | 4 | (2, 2, 0, 0) | (2, 2, 0, 0) |
| `center_tolerance_scale` | 4 | (2, 2, 0, 0) | (2, 2, 0, 0) |
| `unrelated_moved` | 4 | (0, 0, 4, 0) | (0, 0, 4, 0) |
| `unrelated_rotated` | 4 | (0, 0, 4, 0) | (0, 0, 4, 0) |
| `unrelated_mirrored` | 4 | (0, 0, 4, 0) | (0, 0, 4, 0) |
| `unrelated_scaled` | 4 | (0, 0, 4, 0) | (0, 0, 4, 0) |
| `unrelated_changed_color` | 4 | (0, 0, 4, 0) | (0, 0, 4, 0) |
| `unrelated_resized` | 4 | (0, 0, 4, 0) | (0, 0, 4, 0) |
| `unrelated_different_shape_far_away` | 4 | (0, 0, 0, 4) | (0, 0, 0, 4) |
| `local_competition_ordered` | 8 | (0, 16, 8, 8) | (16, 0, 0, 16) |
| `local_competition_crossed` | 8 | (8, 8, 0, 16) | (0, 16, 16, 0) |
| `local_tie_global_unique` | 8 | (0, 16, 0, 16) | (16, 0, 0, 16) |
| `global_tie` | 8 | (0, 16, 8, 8) | (8, 8, 8, 8) |
| `crossing` | 8 | (0, 16, 8, 8) | (0, 16, 16, 0) |
| `stationary_duplicate` | 8 | (8, 0, 0, 24) | (8, 0, 0, 24) |
| `identical_replacement` | 8 | (0, 0, 8, 0) | (0, 0, 8, 0) |
| `survivor_plus_death` | 8 | (8, 0, 0, 8) | (8, 0, 0, 8) |
| `survivor_plus_birth` | 8 | (8, 0, 0, 8) | (8, 0, 0, 8) |
| `all_unchanged` | 8 | (0, 0, 0, 32) | (0, 0, 0, 32) |
| `birth_only` | 8 | (0, 0, 0, 0) | (0, 0, 0, 0) |
| `death_only` | 8 | (0, 0, 0, 0) | (0, 0, 0, 0) |

## Limits and recommendation

This is a deterministic synthetic correctness/stress experiment, not an ARC score benchmark. Shapes, permutations, and pair decisions within a scene are correlated; no t-test, confidence interval, or claim of statistical superiority is warranted. The Cartesian-product negative count depends on how many distractors are supplied. Synthetic identity labels intentionally include histories that cannot be distinguished from two frames. There are no real traces, occlusion, segmentation merges/splits, crop-induced fragments, noisy pixels, or performance measurements in this study.

**Decision: retain Hungarian assignment.** Its removal is unnecessary for adding reflection, scaling, or the new diff schema. Keep the frozen comparison for later investigation of false matches and ambiguity. A hybrid with global ambiguity checks has not been implemented or validated.

## Reproduction and artifacts

From `ARC3-Inference/`:

```bash
.venv/bin/python ../docs/matching/scripts/run_study.py --self-test
.venv/bin/python ../docs/matching/scripts/run_study.py --check-production
```

The runner uses frozen local sources and the Python standard library. It does not edit production files or require third-party scientific packages. It regenerates the JSON/CSV artifacts; this report interprets the recorded run.

- [run_study.py](scripts/run_study.py): generator, independent oracle, Hungarian adapter, exhaustive adapter checks, scorer.
- [results.json](data/results.json): aggregate, suite, and family matrices, seed, and source SHA-256 values.
- [per_family.csv](data/per_family.csv): all families and both methods in tabular form.
- [cases.jsonl](data/cases.jsonl): every fixture and its per-method predictions, misses, and false pairs.
- [frame_diff_snapshot.py](snapshots/frame_diff_snapshot.py): mutual-nearest alternative frozen before this study.
- [segmentation_snapshot.py](snapshots/segmentation_snapshot.py): segmentation used by both methods.
- [hungarian_reference.py](snapshots/hungarian_reference.py): original solver from `64a98a8`.

Production HEAD when the study began: `64a98a8` plus the uncommitted proposed diff. The SHA-256 values identify the exact tested code, rather than attributing the working-tree implementation to HEAD.

| Source | SHA-256 |
|---|---|
| `scripts/run_study.py` | `f4fada0446d5699f5247098e32e6702d40b26c5a8de0483d32bdee81230b2913` |
| `snapshots/hungarian_reference.py` | `7b44cac4aee7a90bc208641cf6a963c4841df94251e24e2336935fc2f5d8ca7e` |
| `snapshots/frame_diff_snapshot.py` | `4ad531ab04d9fafb2b26995f9916ab3e934de8d77a237045f6c9801c6ce192a7` |
| `snapshots/segmentation_snapshot.py` | `7bcf29ade29cc9883185a99056d8fc74b46ff07b87f5746a6459acab34a7090d` |

Validation after retaining Hungarian: **88 focused API tests passed**, and the production diff returned exactly the frozen Hungarian result on **all 684 study scenes**. The production source hash is recorded in `data/results.json` under `production_replay`.
