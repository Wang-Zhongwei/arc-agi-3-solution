# ARC-AGI-3: Daniel Franzen's Milestone 2 solution

> **Team fork notes.** See [This fork](#this-fork): the recommended run, our research in [`research/`](research/README.md) and every Kaggle run in [`runs/`](runs/README.md).

This repository contains my solution for the **ARC-AGI-3 Progress Prize #2**. **For a detailed explanation of the approach, changes, and experiments, see the [write-up](WRITEUP.md).**

My approach is based on [Tufa Labs' Duck harness](https://github.com/Tufalabs/duck-harness). Full credit to **Jeroen Cottaar and Tufa Labs** for the original solver, harness, and notebook.

My changes focus on getting more useful work from the model within the competition's compute budget: faster inference with a larger context, better prefix-cache retention, adaptive compute allocation across games, and improvements to the information and tools available to the agent.

## This fork

This fork works toward **10 of the 25 public games fully won in the 121-minute offline run**, a stand-in for
the 110 private games in Kaggle's 9-hour run. `main` is Franzen's solution plus our harness, memory, serving and
tooling work. Every feature we added is behind a default-off flag, except the #4 and #5 sandbox fixes (#52, #51).
With default settings, `main` is therefore Franzen plus those two fixes; an arm in
[`kaggle/arms.json`](kaggle/arms.json) turns features on.

Research and the result of every experiment are in [`research/`](research/README.md) (earlier work in
[`research/archive/`](research/archive/README.md)). Every Kaggle run's logs and outputs are in
[`runs/`](runs/README.md). Every flag is documented in
[`ARC3-Inference/CONFIGURATION.md`](ARC3-Inference/CONFIGURATION.md), and serving settings in
[`serving/CUSTOM.md`](serving/CUSTOM.md).

New experiments get their own `arm/<name>` branch and PR off `main`. The branches of earlier experiments were
closed on 2026-10-14. Their tips are kept as tags `archive/<branch>`, and their table is in
[`research/archive/readme-fork-section-2026-10-14.md`](research/archive/readme-fork-section-2026-10-14.md).

### Recommended full run: `mem-repair-10x`

The best run so far: **10 games won**, mean 54.69, 116 levels, 5.03M tokens (`mellamoa/v4-mem-repair-10x`
v1, 2026-10-14). The exact notebook is in [`kaggle/notebooks/`](kaggle/notebooks/README.md), and the report is
[`research/reports/mem-repair-2026-10-14.md`](research/reports/mem-repair-2026-10-14.md). It is one run, so
it is not yet an alpha by the repeat rule.

Also on `main` and **off by default** (2026-10-17): `ARC3_SCHED_FINAL_HOLD` (#180) and `ARC3_VICTORY_LAP` (#181).
Neither is in the recommended config; results in [`research/README.md`](research/README.md). Offline test runs now
wait for the model server before the games start (#182); submissions are unchanged.

On top of Franzen's notebook settings it turns on:

| Layer | Flags |
|---|---|
| 14 concurrent games, smaller window and drain | `ARC3_MAX_ACTIVE_STREAMS=14`, `LOCAL_ANALYZER_CONTEXT_WINDOW=98304`, `ARC3_CONTEXT_DRAIN_TOKENS=43008`; serve `MAXREQ` 14, `CUDAGRAPH_MAXBS` 14, `CUDAGRAPH_BS_EXTRA` [3, 5, 6, 11, 12, 13], `MAMBA_CACHE` 84, `CTX` 106496, `MEMFRAC` 0.96 |
| user-combo base | `ARC3_LEDGER_ON_TRIM`, `ARC3_ANIM_TRUTH`, `ARC3_ANIM_RING=8`, `ARC3_ANIM_DIGEST`, `EXPOSE_RESET=on`, `LOCAL_ANALYZER_TEMPERATURE=0.65`, `ARC3_OPENER_DEDUP` |
| Prompt facts | `ARC3_BUDGET_TRUTH` (no run clock shown to the model), `ARC3_PROMPT_HYGIENE`, `ARC3_IMAGE_GEOMETRY`, `ARC3_PREFER_TOOL_CALLS` |
| Memory (mem-unified) | `ARC3_MEM_SOLVED`, `ARC3_MEM_CARD`, `ARC3_MEM_EFFECTS`, `ARC3_MEM_SEEN`, `ARC3_MEMORY_PLACEMENT=trim`, `ARC3_MEM_REPLY_WINS`, `ARC3_MEM_DEBRIEF`, `ARC3_MEM_KEEP_EDGE`, `ARC3_MEM_EVERY_CALL`, `ARC3_MEM_LEGIBLE`, `ARC3_MEM_RECAP_ASK`, `ARC3_MEM_APPEND_COMMIT`, `ARC3_MEM_RECAP_CHARS=800`, `ARC3_MEM_RULES`, `ARC3_MEM_MOVED`, `ARC3_TOUCHING_SPLIT` |
| Stall facts (stall-fix) | `ARC3_ACTION_ERROR_RAISE`, `ARC3_ACTION_ALIASES`, `ARC3_MEM_COLOURS_FIRST`, `ARC3_DEATH_VERDICT`, `ARC3_UNDONE_FACT`, `ARC3_CLICK_RUNS` |
| Memory repairs (mem-repair) | `ARC3_RULES_FILTER_V2`, `ARC3_RULES_ACTION_TAG`, `ARC3_RETAINED_DELTA`, `ARC3_MOVED_PERSIST`, `ARC3_CLEARED_BOARDS`, `ARC3_THIS_LEVEL_AFTER_TRIM`, `ARC3_MEM_CONTACT_ROW`, `ARC3_CLICK_MISS`, `ARC3_MEM_CLICKED_ALSO`, `ARC3_EFFECT_MOVED_FIX` |

The model is the W4A16 Qwen3.8-Flash-Next with the albucino MTP drafter, with the stock 10 experts per token.
No code branches on a game's identity, and no minutes or game counts reach the model (audit in the report).

**Build it:**
1. Run `git fetch origin tag bundle-base`.
2. From the repo root, under Python 3.12, run
   `python kaggle/build_notebook.py --arm mem-repair-10x --submission`. This builds it as the submission
   kernel (`v4-submission`); add `--push` to push.
3. A plain `--arm mem-repair-10x` build is refused, because that notebook has already run.

**Earlier best runs:**
- `user-combo-10x`: 56.94 and 10 wins, but its repeats scored 52.70 and 49.09.
- `mem-unified-10x`: 56.09, 8 wins.

**Tools on `main`:**
- [`kaggle/`](kaggle/README.md): the Kaggle arm builder, run records, game panels and the live score timeline.
- [`analysis/`](analysis/README.md): the score timeline and the failure taxonomy.
- `tools/phone_viewer.py` and the Tufa run viewer (`make view`) for saved runs.
- The editable SGLang serving stack ([`serving/CUSTOM.md`](serving/CUSTOM.md)).

## Reproducing the solution

The [competition notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) is the intended reproduction path. It includes the harness patch and sets up the model and offline serving runtime on Kaggle. When copying it, select the **RTX Pro 6000** GPU manually.

The fork is based on upstream commit [`7652836`](https://github.com/Tufalabs/duck-harness/tree/7652836056c59e044f093e3c13ed7438c814169e). The source here includes both the features used in the submission and optional experiments. The published notebook's settings identify the competition configuration; enabling every option is not the intended setup.

For local use and development, see the [harness README](ARC3-Inference/README.md). The checked-in local configuration and Tufa deployment commands are separate from the competition notebook's serving setup.

For building the offline SGLang runtime and details of the serving patches, see the [serving README](serving/README.md).

## Main changes

- **Model and serving:** Qwen3.8-Flash-Next with an optimized SGLang runtime. Serving work makes long contexts practical while maintaining throughput, including changes to Mamba prefix-cache retention and checkpoint prefetching.
- **Context management:** calibrated text-token estimates, image-size-aware token accounting, and blockwise history trimming. Evicting a larger block at once lets the retained prefix be reused for more requests before the next trim.
- **Compute allocation:** bounded concurrent admission reduces cache pressure. Priority scheduling allocates compute using current progress, actions and tokens spent, and continuation value. Cached continuations receive priority; tail fade shifts emphasis toward immediate level completions near the deadline.
- **Perception and feedback:** larger board images, board-difference and death images, animation frames and timelines accessible in Python, and clearer action-result history.
- **Agent tools and prompts:** executable undo, optional reset, guidance for transferring knowledge across levels, and retention of eligible Python functions and supported imports.
- **Legibility (flag-gated, `ARC3_LEGIBILITY`):** harness text the model misread, restated as facts: segmentation nodes with a `bbox`, schema-complete animation lookups, readable retained-function source, truthful stale-state and RESET wording, and a turn-based statement of the step bar, plus smaller flag-only fixes ([configuration guide](ARC3-Inference/CONFIGURATION.md) §2).
- **Execution safeguards:** guards for stale-state actions and no-op batches, protection at level boundaries, and clearer tool output with action echo and optional middle truncation.

- **Makefile compatibility:** fixes for newer GNU Make versions, including how configuration-derived variables are evaluated.
- **Viewer fixes and improvements:** correct per-turn transcript display, visibility of retried attempts, clearer handling of pending transcripts and the latest board state, and per-step token-usage information.

The detailed option reference is available in the [configuration guide](ARC3-Inference/CONFIGURATION.md). The [write-up](WRITEUP.md) distinguishes features used in the competition from experiments with negative or inconclusive results.

## Repository layout

| Path | Contents |
| --- | --- |
| [ARC3-Inference/](ARC3-Inference/) | Modified solver, prompts, tools, scheduler, diagnostics, and viewer. |
| [serving/](serving/) | Offline SGLang wheelhouse builder, custom patches, and build instructions. |
| [kaggle/](kaggle/) | This fork's Kaggle arm builder: arm definitions, the notebook template, run records, push queue and output fetcher, the live score-timeline logger it adds to every notebook, and archived notebooks of notable runs (`kaggle/notebooks/`). |
| [analysis/](analysis/) | Offline analysis of saved runs: `score_timeline.py` (score over wall-clock, matched-minute comparisons, HTML overlay). |
| [research/](research/) | This fork's research notes (including the arms log), reports, and every workflow's agent reports. |
| [runs/](runs/) | Logs and outputs of our Kaggle kernel versions (xz-compressed), with `index.csv` and `submissions.csv`. |
| [tufa-arc-agi-framework/](tufa-arc-agi-framework/) | Tufa's bundled game execution framework. |
| [taaf-duck-harness-kaggle-share.ipynb](taaf-duck-harness-kaggle-share.ipynb) | Tufa's original bundled notebook; see the competition notebook above for this solution. |

## Inspect a saved run

The viewer does not require a model server or GPU. From the repository root, with `uv` installed:

```bash
cd ARC3-Inference
uv sync --locked
make view VIEW_RUN_DIR=/path/to/your/run VIEW_PORT=8011
```

Open `http://127.0.0.1:8011` to inspect the saved run's boards, actions, and model transcripts.

To browse all of our saved Kaggle runs (stored with `.xz` files) and read them on a phone over Tailscale:

```bash
cd ARC3-Inference
uv run inference-view --host <tailscale-ip> --port 8011 --runs-dir ../runs/kaggle
```

Then open `http://<tailscale-ip>:8011` on the phone. Use `--host 127.0.0.1` for the laptop only. See the [viewer section](ARC3-Inference/README.md#viewer) for the controls and the score-over-time panel.

## Acknowledgements

This work builds on Tufa's [Duck harness](https://github.com/Tufalabs/duck-harness) and [technical write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133), the Qwen models, and the SGLang serving framework.

The serving setup also builds on work by [John Pezzulli (Pennyroyal)](https://github.com/jpezzulli/sglang-rtxpro6000), [Mamy Ratsimbazafy](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo), and [Gabriel Olympie](https://github.com/gabrielolympie/sglang-flashnext-sm120), with model quantizations from RadixArk and Intel. Exact checkpoints, runtime versions, and patches are recorded in the competition notebook and its build scripts.
