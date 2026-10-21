# Data behind `reports/mem-repair2-2026-10-15.md`

**Reads** (agent reports, condensed):
- `sp80-bp35.md`: the sp80 L2 and bp35 L2 stall reads.
- `m0r0-tn36.md`: the m0r0 L3 and tn36 L2 stall reads.
- `precision-and-reads.md`: the run-wide precision check of every memory line, plus the tu93/re86 and
  ka59/wa30/ls20 reads.

**Judgments:** `precision-judgments.jsonl` holds the 262 hand judgments from the precision check. Each row has the
line type, game, level, request, text, verdict, whether the model used it, and a note.

**Scripts** (`scripts/`). They read the runs listed in `runs_def.py`, which point at the local `kaggle-outputs/` and
`runs/kaggle/` directories.

| Script | What it computes |
|---|---|
| `decomp.py` | Points earned, lost to extra actions and lost to uncleared levels |
| `open.py` | Open levels and the points at stake |
| `pace.py` | Levels and tokens at 30/60/90/120 min |
| `park.py` | Idle gaps per game |
| `park_after_clear.py` | Idle gaps that begin right after a level clear |
| `serve_occ.py` | Server batch occupancy |
| `eval_variants.py` | Rules-filter variants on SF's labelled quotes and MR's judged quotes |
