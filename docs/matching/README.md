# Matching: keep Hungarian

**Decision:** retain Hungarian assignment for `moved`, `rotated`, `mirrored`, `scaled`, and `changed_color`. The new transform checks and return format do not require replacing it. `resized` continues to use its separate overlap-based fallback.

The comparison found a tradeoff: Hungarian recovered more true pairs, with more false matches. Mutual unique nearest remains a frozen experimental alternative for later study. Neither method can establish persistent identity from two ambiguous frames.

## Read first

1. [Ten visual cases](visual_cases.md) — actual grids, ground truth, both predictions, and correct/wrong/missed counts.
2. [Ten-page PDF](figures/matching_examples.pdf) — the same figures for zooming or sharing.
3. [Comparison report](report.md) — definitions, confusion matrices, all transform combinations, limitations, and the decision.

For a short introduction, see [candidate transforms](figures/intro/01_transform_candidates.png), [competing assignments](figures/intro/02_assignment_competition.png), and [identity ambiguity](figures/intro/03_identity_ambiguity.png).

## Folder layout

```text
README.md                   Start here; current decision and navigation
report.md                   Full 684-scene comparison
visual_cases.md             Annotated ten-case gallery
figures/
  matching_examples.pdf     Ten-page gallery
  cases/                    Ten cases, PNG and editable SVG
  intro/                    Three introductory diagrams, PNG and SVG
data/
  results.json              Aggregate/suite/family metrics and source hashes
  per_family.csv            Tabular results for every family and method
  cases.jsonl               Full fixtures, predictions, misses, and false pairs
snapshots/
  frame_diff_snapshot.py    Frozen mutual-nearest alternative used in the study
  segmentation_snapshot.py  Segmentation shared by both study methods
  hungarian_reference.py    Original solver extracted from commit 64a98a8
scripts/
  run_study.py              Reproduce results; optionally verify production
  make_gallery.py           Replay and draw the ten recorded cases
  make_examples.py          Draw the three introductory examples
```

Snapshots are evidence, not runtime modules. They intentionally retain the evaluated mutual-nearest implementation. Production uses `ARC3-Inference/inference/utils/frame_diff.py`.

## Reproduce

Run from `ARC3-Inference/`:

```bash
# Study runner: Python standard library only.
.venv/bin/python ../docs/matching/scripts/run_study.py --self-test
.venv/bin/python ../docs/matching/scripts/run_study.py --check-production

# Figures: matplotlib and NumPy, available in the local environment.
.venv/bin/python ../docs/matching/scripts/make_gallery.py
.venv/bin/python ../docs/matching/scripts/make_examples.py
```

`--check-production` also checks that the current production diff returns exactly the frozen study's Hungarian output for every scene, including background records. Omit it when reproducing the historical study after intentionally changing production behavior. Results record the production source hash when this check is used.

The gallery replays each selected scene before drawing it. Regeneration writes files only in this folder. The report and this decision note are maintained separately from generated results.
