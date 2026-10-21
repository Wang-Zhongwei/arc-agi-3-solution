# Vision object-model benchmark (2026-10-10)

Scripts and results behind `research/reports/vision-object-model.md`.

- `gt-tools/gt_dump.py`: renders pixel-exact frames plus per-pixel instance and HUD maps from the public games' engine source (`competitors/polyphony-arc-3/vendor/arc/environment_files`). `gt-tools/*.py` also holds the latency, usage and role probes used in the research stage.
- `bench/eval.py`: instance F1@IoU0.5, group F1, fragmentation, hard subsets. Candidates: `cand_baseline.py` (harness colour components), `cand_lexicon.py` (lexicon view, reimplemented from Kaggle discussion 746295), `cand_mobilesam.py`, `cand_tiny.py` + `tiny_model.py` / `train_tiny.py` (845k-param CNN-ViT trained on engine ground truth).
- `bench/roles_eval.py`: the online behavioural player/click-target tracker and its Stage-0 gate.
- `bench/results.md`, `bench/results/`: the numbers in the report.

Not kept: the frame dumps (~40 MB), MobileSAM weights (40 MB) and the tiny model weights; `gt_full.py` and `gt_dump.py` regenerate the dumps.
