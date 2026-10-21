# Archived notebooks

Exact copies of notebooks that ran on Kaggle, pulled with `kaggle kernels pull <kernel> -m`,
so a notable run can be inspected or re-pushed without rebuilding it.

## `v4-mem-repair-10x-v1.ipynb`: the best run on record (10 of 25 games won)

- **Kernel:** `mellamoa/v4-mem-repair-10x` version 1 (2026-10-14 18:36–20:38 UTC), with its
  `v4-mem-repair-10x-v1.kernel-metadata.json` (RTX Pro 6000, no internet, W4A16
  Qwen3.8-Flash-Next + albucino MTP drafter, `dfranzen/pennyroyal-v253` and
  `dfranzen/taaf-kaggle-source-bundle-copy`).
- **Score:** 54.69 on the 25 public games, 121-minute offline run, 116 levels, 5.03M tokens.
  **10 games fully won** (ar25 55m, ft09 46m, tr87 49m, sb26 77m, cd82 83m, vc33 94m, sc25 96m,
  lp85 104m, r11l 105m, cn04 110m), tying the most wins of any run (user-combo-10x v1). One run:
  paired test vs stall-fix-10x +2.85 (P 0.72), vs mem-unified-10x −1.40 (P 0.38). Not yet an
  alpha by the repeat rule. The run record is `runs/kaggle/v4-mem-repair-10x/v1`.
- **What it is:** `user-combo-10x` + the memory stack (PR #172 mem-unified, #173 stall-fix, #174
  mem-repair). Every flag is in `kaggle/arms.json` arm `mem-repair-10x` and documented in
  `ARC3-Inference/CONFIGURATION.md` (on `main` since PR #174 merged).
- **Built from:** branch `arm/mem-repair` (now tag `archive/arm/mem-repair`, merged into `main`) at commit `847ab5e3`, notebook fingerprint
  `b01c88d77f95b028` (recorded in `kaggle/runs.json` on that branch). Pulled with
  `kaggle kernels pull mellamoa/v4-mem-repair-10x -m`; its cells are identical to the local build.
- **Game- and time-agnostic check (2026-10-14):** no code branches on a game id, grid size,
  palette index, level count or action set; public game ids appear only in comments and in
  Franzen's local game list (`framework/kaggle.py`, not used by the notebook). No minutes or game
  counts reach the model. Thresholds calibrated on public games are listed in
  `research/reports/mem-repair-2026-10-14.md`.

## `v4-user-combo-10x-v1.ipynb`: the previous best run (56.94; 10 wins)

- **Kernel:** `mellamoa/v4-user-combo-10x` version 1 (2026-10-09 02:30–04:46 UTC), with its
  `v4-user-combo-10x-v1.kernel-metadata.json` (RTX Pro 6000, no internet, W4A16
  Qwen3.8-Flash-Next + albucino MTP drafter, `dfranzen/pennyroyal-v253` and
  `dfranzen/taaf-kaggle-source-bundle-copy`).
- **Score:** 56.94 on the 25 public games, 121-minute offline run, 123 levels. Two repeats of the
  same notebook on the snowiers account scored **52.70** and **49.09**: mean of three 52.91
  (sd 3.93) vs control 50.62. Not an alpha by the repeat rule; the reliable gain is
  throughput (harness ~705 vs 602 tok/s, +17%). Details: `research/archive/notes/arms.md`.
- **Built from:** branch `arm/user-combo` at commit `1fccfd13`, notebook fingerprint
  `399abe34aa243a07` (recorded in `kaggle/runs.json`).
- **Rebuild:** `python kaggle/build_notebook.py --arm user-combo-10x --submission` from the repo
  root, under Python 3.12, builds it as the submission kernel. Without `--submission` (or
  `--replica N`) the build is refused, because this fingerprint already ran. The rebuild at the
  merge into `main` gave the same fingerprint, and its cells are identical to this file. Python
  3.14 writes a different gzip header byte into the embedded blobs, so its build of the same
  notebook gets another fingerprint (`11d2c46673d51188`).

### Config (arm `user-combo-10x` in `kaggle/arms.json`)

Harness env, on top of the template's defaults (priority scheduling, `EXPOSE_UNDO=on`,
`LOCAL_ANALYZER_TOP_P=0.95`, `TOP_K=20`, max output 12k, 121 min, all 25 games):

| Flag | Value | What it does |
|---|---|---|
| `ARC3_MAX_ACTIVE_STREAMS` | 14 | 14 concurrent games (template: 10) |
| `ARC3_LEDGER_ON_TRIM` | 1 | keep a ledger of the model's tagged findings (`<world_model>`, `<hypothesis>`, `<level_note>`) and show it once at the head of the retained history after each context trim |
| `ARC3_ANIM_TRUTH` | 1 | Franzen's animation truth verdicts |
| `ARC3_ANIM_RING` | 8 | keep the last 8 animated actions, each with its full frame chain (`animation_for(n)`, `animations`) |
| `ARC3_ANIM_DIGEST` | 1 | Wang-style animation digest |
| `EXPOSE_RESET` | on | the model may RESET the level (zero-action RESET refused) |
| `LOCAL_ANALYZER_TEMPERATURE` | 0.65 | template: 0.7 |
| `LOCAL_ANALYZER_CONTEXT_WINDOW` | 98304 | template: 131072 |
| `ARC3_CONTEXT_DRAIN_TOKENS` | 43008 | template: 59392 |
| `ARC3_OPENER_DEDUP` | 1 | drop the repeated resume image and static opener text |

Serving (SGLang `CFG` overrides): `MAXREQ` 14, `CUDAGRAPH_MAXBS` 14,
`CUDAGRAPH_BS_EXTRA` [3, 5, 6, 11, 12, 13], `MAMBA_CACHE` 84, `CTX` 106496, `MEMFRAC` 0.96.
Stock 10 experts per token (no MoE routing override), speculative decoding with 3 steps and
acceptance 1.0, FP8 KV cache, chunked prefill 8192.
