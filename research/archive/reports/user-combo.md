# user-combo: the owner's combined run

2026-10-08. Arm `user-combo` in `kaggle/arms.json`, branch `arm/user-combo`. All 25 offline games, 121 minutes, W4A16 Qwen3.8-Flash-Next with the albucino MTP drafter.

**This arm changes seven things at once.** If it wins, the win belongs to the bundle, not to any one part. Attributing it needs the single-change arms (`ledger-on-trim`, `experts13-basenorm`, `expose-reset`, the animation layer arms) run against the same control.

## What it changes against the control

The control is the flags-off build of the same commit, which is `arm/throughput-stack`'s control (best on record 50.62, same-commit). It has the #4/#5 fixes and Franzen's scheduler. Every change below is behind a flag that the control leaves off.

| # | Change | Settings | Prior evidence |
|---|---|---|---|
| 1 | 14 streams instead of 10 | `ARC3_MAX_ACTIVE_STREAMS=14`; serve `MAXREQ` 14, `CUDAGRAPH_MAXBS` 14, CUDA graphs for every batch size 1-14 (`CUDAGRAPH_BS_EXTRA` 3, 5, 6, 11, 12, 13), `MAMBA_CACHE` 84 | `fz-streams14` on the old code |
| 2 | Less context per game, so 14 games fit the ~1.01M-token KV pool | `LOCAL_ANALYZER_CONTEXT_WINDOW=98304`, `ARC3_CONTEXT_DRAIN_TOKENS=43008`, serve `CTX` 106496, `MEMFRAC` 0.96 | fz-streams14's values; **placeholders**, see below |
| 3 | Memory ledger re-shown after each trim | `ARC3_LEDGER_ON_TRIM=1` | 52.56 / 49.14 vs 46.25 on the old code, not significant |
| 4 | Animation layers L1-L3 (combining Franzen's and Wang's animation) | `ARC3_ANIM_TRUTH=1`, `ARC3_ANIM_RING=8`, `ARC3_ANIM_DIGEST=1` | Design doc: 20% of NO-OP echoes are animated actions; no score yet |
| 5 | RESET exposed, with the zero-action guard | `EXPOSE_RESET=on` | Not run (#134) |
| 6 | Temperature 0.65 | `LOCAL_ANALYZER_TEMPERATURE=0.65` (template 0.7) | `temp06` |
| 7 | 13 experts per token, trained weights kept | serve `TEXT_CONFIG_OVERRIDE {"num_experts_per_tok": 13}`; serve_env `SGLANG_ARC_MOE_ROUTING=base_norm`, `SGLANG_ARC_MOE_BASE_K=10` | experts16 with default renormalisation: −7.6 |

Unchanged: Franzen's scheduler (sched-oracle and the D′/EEVDF tuning found nothing better), the template's Franzen animation (`ARC3_ANIMATION=1`, frames and timeline in the REPL, diff images), no animation images (`ARC3_ANIM_KEYFRAMES=0`), the throughput-stack levers off (tput-safe lost, 42.09 vs 50.62), `ARC3_OPENER_DEDUP` off (available from #129).

**KV settings (#2).** 14 streams × 128k does not fit the pool, so the window, drain, `CTX` and `MEMFRAC` are explicit in the arm. They are fz-streams14's proven values: window and drain scaled by 10/14, with `CTX` = window + 8k and the launcher's `MEMFRAC`. The parent sets them from the config-sweep-kv result before pushing. Keep `CTX` at or above the window.

## The animation layers in this arm
- **L1, truth.** An action that left the board area unchanged but whose animation changed cells and changed them back reads `ANIMATED (blink; board area unchanged; 2 frames, 1 cell changed and changed back - see animation_for(12))`, not `NO-OP`, in the echo and trace. It passes the batch no-op stop, does not arm the stale-state latch and is never filed as a known no-op.
  - Cost: about 12 tokens on about 57 actions per run.
  - Behaviour change to watch: batches of animated clicks now run on where the control stopped at the first one.
- **L2, ring of 8.** `animation_for(n)` and `animations` in Python. `last_animation_*` follow the newest animated action. The per-batch `animations` list holds only actions that hid something.
- **L3, digest.** A typed opener block (blink, tween with from>to and d(dr,dc), path), capped at 600 characters. In the loop simulation, 54 digests had a median of 234 characters (about 60 tokens) and a maximum of 378.

## Verification (no GPU)
- `tests/test_animation_layers.py`: each layer on the real sandbox and solver session.
- `tests/test_anim_loop.py`: the real game loop (`_HarnessGameSession.play()` → `ToolAgent.analyze()` → the real sandbox) under the notebook's `setup_env`, in 11 configurations including this arm's env with a forced trim, with `EXPOSE_RESET`, and with `ARC3_OPENER_DEDUP`. It checks:
  - every prompt extends the previous one unless history was evicted;
  - the system prompt and tools are static;
  - the digest lands once, only in fresh-turn openers, never on a resumption;
  - the ring survives a trim;
  - with the layers off, requests are byte-identical to `arm/throughput-stack`.
  - Ten mutants are all caught.
- `kaggle/tests/test_user_combo_arm.py`: every env, serve and serve_env key is traced to the code that reads it, through the built notebook.

## How to judge it
- Compare it with a same-batch control at matched wall-clock (121 minutes) using `make significance`.
- One run is a screen. The owner's rule needs repeated runs that consistently score higher before this counts as an alpha.
- Watch throughput: 13 experts costs compute (k=16 cost about 17%), and 14 streams change batch shapes, so check the serve_report tok/s and prefix-cache hit rate against the control.
