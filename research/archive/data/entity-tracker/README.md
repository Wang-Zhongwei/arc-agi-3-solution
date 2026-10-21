# Behavioural entity tracker: offline Stage-0 (2026-10-10)

Two independent designs, each with a held-out split fixed before any code ran, scored once on held-out levels with frozen code (checksums in `A/FROZEN.sha256`). Ground truth comes from the engine replay in `research/data/vision-object-model/`.

| | Player committed within 5 actions (held-out) | Correct at level end | Wrong committed claims | Gate |
|---|---|---|---|---|
| A: evidence-accumulating entity ledger (`A/tracker.py`) | 18/32 (56%) vs baseline 17/32 | 26/32 vs 24/32 | 15.7% vs 11.7% (8.3% without sk48) | FAIL (bar: 80% and ≤5%) |
| B: action-conditioned diff model (`B/`) | 16/34 (47%) vs baseline 16/34 | 25/34 vs 28/34 | 15.9% roles GT, 2.2% arrow-driven GT | FAIL |

Both designs gained a lot on their dev games (A: within-5 69% vs 42%) and lost most of it on held-out games, so many rules were dev-specific. Kept: latency p95 3–6 ms, ≤100 tokens per action, view-shift removal, identity through RESET, selection-follow after the next press. Failure causes and fixes are in each design's `results/` notes: one-name ground truth for multi-piece games (use an extended GT), re86's SPACE marker jump, appearance priors on lookalikes, co-moving twins (report them as a set), click-selection rules. A retry needs fresh held-out games or leave-games-out cross-validation, since all 12 role-benchmark games have now been used.
