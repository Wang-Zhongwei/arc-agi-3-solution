# Duck harness: ARC-AGI-3 Milestone 2 fork

This directory contains Daniel Franzen's modified version of [Tufa Labs' Duck
harness](https://github.com/Tufalabs/duck-harness), originally developed by
Jeroen Cottaar and Tufa Labs. See the [repository README](../README.md) for the
solution overview, attribution, and competition publication links.

For a detailed explanation of the approach, changes, and experiments, see the
[write-up](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md).

The harness connects TAAF's `Benchmark` / `GameAPI` to an OpenAI-compatible
model server and a Python tool for inspecting and interacting with games.
The competition setup uses SGLang; the bundled local server launcher uses
vLLM. OpenRouter is also supported.

This document covers setup, the model-facing interface, and run inspection.
See the separate [configuration guide](CONFIGURATION.md) for feature switches,
defaults, and their interactions. Use the [competition notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) for the exact
submission settings.

## Quick start

Run these commands from `ARC3-Inference/`. The project pins Python 3.12.12;
`uv` can provision it.

For the viewer, or a client using an already running model server:

```bash
uv sync --locked
```

To inspect a saved run without a GPU:

```bash
make view VIEW_RUN_DIR=/path/to/your/run VIEW_PORT=8011
```

Open `http://127.0.0.1:8011`.

For the bundled local vLLM workflow, install the server and development extras:

```bash
make install
make server
make interactive
```

`make install` downloads the pinned vLLM/Torch stack as well as the base
packages. The competition's SGLang runtime is configured separately by the
notebook; `make server` does not reproduce that serving setup.

For the offline SGLang wheelhouse builder and custom serving patches, see the
[serving README](../serving/README.md).

The Makefile includes compatibility fixes for newer GNU Make versions,
including evaluation of configuration-derived variables.

For an existing OpenAI-compatible server, set the model, endpoint, and context
window in a copy of `configs/inference.json`, then run:

```bash
CONFIG_PATH=/path/to/config.json make interactive
```

For the competition SGLang endpoint, the harness uses provider `vllm` as its
OpenAI-compatible request mode. That setting does not start a vLLM server.
The configured context window and output budget must fit the server's limits.

Run through OpenRouter instead:

```bash
export OPENROUTER_API_KEY=your-key
CONFIG_PATH=configs/inference.openrouter.json make interactive
```

## Agent interface

For each game, `HarnessSolver` provides the latest board, valid actions,
history, and the `python` tool. The model writes Python to inspect the state,
plan, and execute real game actions with `action(...)`.

### Frames and changes

| Name | Meaning |
| --- | --- |
| `current_frame` | Latest board, exposing `.ascii`, `.segmentation`, `.shape`, `.step`, and `.level`. `.shape` is `(rows, cols)`. |
| `previous_frame` | Board before the most recent recorded action, when available. |
| `history` | Recorded states and actions. `history[-1].frame` is the current, post-action board. |
| `transitions` | Individual executed actions, each with `.action`, `.before_frame`, `.after_frame`, and retained `.result` metadata. |
| `last_transition` | Most recent transition, or `None`. |
| `last_action_call_result` | Result of the latest model-issued `action(...)` call, including batch totals, executed/skipped actions, and stop reasons. |
| `frame_diff(before=None, after=None)` | Cell and component differences; defaults to comparing `previous_frame` with `current_frame`. |
| `valid_actions` | Currently advertised model-facing action names. |

Segmentation uses same-color, 4-connected components. Components and their
cross-frame matches are geometric descriptions, not guaranteed game-object
identities. The raw numeric grid is not exposed; `.ascii` provides the symbolic
grid and `.segmentation` provides objects, boundaries, containment, and adjacency.

`action(...)` returns `last_action_call_result` and refreshes the provided
runtime state before the next Python statement. Inspection-only calls retain
the last action-call result. Each transition's `.result` describes one action,
whereas `last_action_call_result` describes the whole model-issued call.

After death followed by automatic reset, the fatal action and reset are
separate transitions. `current_frame` shows the restarted level;
`last_action_call_result` still describes the call that caused death. Transition
metadata identifies harness-initiated actions with `automatic=True`.

### Controls

The standard names are `UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, and `MOUSE`.
Their effects depend on the game. Only use actions available in the current state.

```python
action(['LEFT'])
action([{'action': 'MOUSE', 'row': 4, 'col': 7}])
```

Mouse coordinates are zero-based: `row` increases downwards, `col` increases to
the right, and the origin is the top-left cell. Use board coordinates, not
coordinates in an upscaled image. Legacy `x` / `y` fields are rejected.

`UNDO` and model-initiated `RESET` can also be exposed when configured. Their
availability and behavior depend on the game. See the configuration guide for
these options and reset restrictions.

### Python state and retained functions

Each tool call uses a fresh Python namespace populated with the provided game
state. Ordinary variables do not persist. The tool supports an allowlist of
imports, printed output, and a final `result` value; the default execution
timeout is 30 seconds.

Optional function retention restores eligible model-defined functions and
supported import dependencies in later calls. Ordinary variables remain local
to each call. Retained functions use refreshed runtime globals.

When animation access is enabled, `last_animation_frames` and
`last_animation_timeline` expose intermediate frames and a compressed change
timeline for inspection from Python.

## Configuration

The local CLI uses JSON configuration and Make overrides. For the fork's
additional feature settings, see the [configuration guide](CONFIGURATION.md).

The main config is strict JSON. Comments are not supported.

- `configs/inference.json` is the default local-vLLM / Slurm config.
- `configs/inference.openrouter.json` uses OpenRouter.
- `configs/eval.json` selects runs for `make eval`.
- `configs/significance.json` selects score files for `make significance`.

Use a different config with:

```bash
CONFIG_PATH=/path/to/config.json make interactive
CONFIG_PATH=/path/to/config.json make sbatch
```

Useful sections in `configs/inference.json`:

- `shared.*`: model name, base URL, provider, and context window.
- `experiments.root_dir`: where timestamped run directories are written.
- `environment.*`: games, tags, passes, concurrency, and runtime limits.
- `deployment.*`: inline vs Slurm and source repos bundled into Slurm jobs.
- `deployment.slurm.*`: GPU, walltime, partition, local-server startup, and
  extra `sbatch` flags.
- Kaggle runs are configured by CLI/Make overrides because the notebook slug
  and source dataset are usually per run.
- `server.*`: vLLM model-serving settings.
- `analyzer.*`: duck sampling/tool settings. This key is still named
  `analyzer` for compatibility with existing code and configs.
- `chat.*`: direct chat probing with `make chat`.
- `viewer.port`: default viewer port.
- `multimodal.*`: image context for the current grid.

The bundled JSON files retain upstream model and deployment settings. Review
model availability, paths, concurrency, and hardware before using them; they
are not a specification of this fork's competition run.

## Running Games

Run inline in the current process:

```bash
make interactive
```

Submit to Slurm:

```bash
make sbatch
```

Name a run:

```bash
make sbatch RUN_NAME=baseline-qwen
```

Run one game or short-prefix:

```bash
make interactive GAME=taps N_PASSES=1 MAX_RUNTIME_MINUTES=10
```

Run the official tag set with a whole-experiment cap:

```bash
make sbatch GAME=[] GAME_TAGS=official MAX_EXPERIMENT_RUNTIME_MINUTES=360
```

Run the duck locally against TAAF's competition Arcade simulator:

```bash
make interactive \
  GAME=[] GAME_TAGS=official \
  SIMULATE_COMPETITION_ARCADE=true \
  COMPETITION_CLONE_RUNS=110 \
  N_PASSES=1
```

The simulator is inline-only. `COMPETITION_CLONE_RUNS=110` repeats the selected
official games with unique competition-safe IDs. This is useful for testing
the submission interface locally; it does not reproduce the hidden game set.

Common overrides:

- `GAME`: one game, comma-separated games, or a JSON list.
- `GAME_TAGS`: include tags such as `official`.
- `EXCLUDE_GAME_TAGS`: exclude tags.
- `N_PASSES`: TAAF passes per selected game.
- `CONCURRENT_JOBS`: TAAF concurrency. With Slurm local servers this is per
  GPU/server.
- `MAX_ACTIONS`: optional per-game action cap.
- `MAX_RUNTIME_MINUTES`: per-game wall-clock cap.
- `MAX_EXPERIMENT_RUNTIME_MINUTES` or `MAX_EXPERIMENT_RUNTIME_HOURS`: whole-run
  wall-clock budget. If the per-game cap is unset, the runner derives it from
  the number of games, passes, and effective concurrency.
- `EXPERIMENTS_DIR`: base directory for timestamped runs.
- `EXPERIMENT_DIR`: exact output directory for one run.

List resolved official games without running them:

```bash
uv run --no-sync inference-taaf-run --include-tags official --list-games
```

## Local vLLM

Start the server:

```bash
make server
```

Check or stop it:

```bash
make check-server
make stop-server
```

The default local base URL is `http://127.0.0.1:1234/v1`. `make server`
generates a local server API key unless `SERVER_REQUIRE_API_KEY=false`.

On cluster machines, the Makefile moves Hugging Face, Torch, Triton, and related
caches under `/shared/<user>` when that directory exists.

## Slurm Flow

`make sbatch` runs through TAAF's Slurm deployment. The job directory includes
the run config, generated Slurm script, dependency override file, benchmark
artifacts, diagnostics, and logs.

For local-vLLM Slurm runs, the solver starts local server processes inside the
allocation before the duck begins playing. Each run gets its own API key and
run-scoped localhost ports, so a run either talks to its own server or fails
fast. The server is started from the bundled `src/ARC3-Inference` snapshot in
the run directory, using the worker's per-run virtualenv.

`deployment.source_repos` is bundled into the Slurm job directory. The worker
uses a generated dependency override file so it installs those bundled repos
instead of fetching private dependencies from GitHub.

## Kaggle reproduction and upstream deployment

For this solution, use the competition notebook linked from the
[repository README](../README.md). It supplies the offline model-serving
runtime, source patch, and submission settings. Select the RTX Pro 6000 GPU
when copying the notebook.

The inherited `make kaggle-duck` workflow packages sources through TAAF and
publishes a notebook with its configured model and wheelhouse datasets. It is
a separate deployment path and does not automatically use the competition's
SGLang build or settings. The bundled
[`taaf-duck-harness-kaggle-share.ipynb`](../taaf-duck-harness-kaggle-share.ipynb)
is Tufa's original notebook.

## Run Artifacts

Each run writes a timestamped directory under `experiments.root_dir`, or under
`EXPERIMENTS_DIR` / `EXPERIMENT_DIR` when overridden.

Important files include:

- `run_config.json`: resolved games, passes, concurrency, runtime caps, model,
  Slurm settings, and hardware metadata.
- `benchmark.json`: saved TAAF benchmark and per-game `GameRun` state.
- `diagnostics.html`: TAAF diagnostics.
- `artifacts/*_viewer_data.json`: compact viewer payloads.
- `artifacts/*_events.jsonl`: append-only full viewer event sidecars.
- duck transcript HTML/text files linked from the viewer.
- `stdout.log` and `stderr.log` for Slurm jobs.
- `requests.jsonl` files when `analyzer.save_request_logs` is true.

## Viewer

Start the viewer for a saved run on the port from `configs/inference.json`:

```bash
make view VIEW_RUN_DIR=/path/to/your/run
```

Override the port:

```bash
make view VIEW_RUN_DIR=/path/to/your/run VIEW_PORT=8012
```

Point at a run root (clear the default single-run selection):

```bash
make view VIEW_RUN_DIR= VIEW_RUNS_DIR=/path/to/runs
```

Point at one exact run:

```bash
make view VIEW_RUN_DIR=/shared/arc_3_results/$USER/<run-name>
```

Browse every saved Kaggle run (`runs/kaggle/<kernel>/v<N>/`) straight from the
repo, without unpacking the `.xz` files:

```bash
make view VIEW_RUN_DIR= VIEW_RUNS_DIR=../runs/kaggle
```

Read runs on a phone over Tailscale: bind to the laptop's Tailscale IP (not
`0.0.0.0`; the viewer has no auth) and open `http://<tailscale-ip>:8011` on the phone:

```bash
make view VIEW_HOST=<tailscale-ip> VIEW_RUN_DIR= VIEW_RUNS_DIR=../runs/kaggle
# or directly:
uv run inference-view --host <tailscale-ip> --port 8011 --runs-dir ../runs/kaggle
```

The viewer shows run summaries, per-game progress, boards, actions, rewards,
level transitions, and the duck's transcript. What it does:

- **Run picker:** every run under `--runs-dir` (searched three levels deep, so
  `<kernel>/v<N>/` works), newest first by the run's own start time, each with its
  mean score from `summary.txt` (or `benchmark.json`). Folders holding only logs are
  skipped. Any file can be stored plain or as `<name>.xz`; it is decompressed in
  memory, and big games are parsed once and cached (a whole game's boards are about
  20 KB gzipped, so stepping needs no further requests).
- **Action player:** step through every action with its board: previous/next action,
  previous/next turn, jump to action N, play/pause with a speed control, and level
  chips that jump to the first board of each level. Keys: left/right arrows,
  `[`/`]`, space, Home/End. On a phone, swipe the board. Each action shows the turn
  (analysis step) that chose it, with that turn's thinking, tool calls and results;
  "Show full model window" loads the exact request context.
- **Score over time:** the official score (level score `min(115, (human/agent actions)^2 * 100)`,
  level `i` weighted by `i`), levels completed and generated tokens against
  **wall-clock time since run start**, for the game or the whole run (mean over games),
  with a marker at the viewed action. Tap the chart to jump. Built from `benchmark.json`.
- **Runs without viewer data** (panels that saved only `benchmark.json` plus
  `transcripts/`): the action list, levels and score-over-time come from
  `benchmark.json` and the reasoning from the transcripts; only the boards are missing.
- **Layout:** desktop shows the turn list beside the board; below 1100 px the page is one
  column with the board first; at phone width only the current turn's reasoning is shown
  (toggle "All turns"), the step controls are fixed at the bottom, and the board scales to
  the screen width. The position is kept in the URL (`#run=..&game=..&pos=..`), so a
  reload or bookmark returns to the same action.

JSON API (all GET): `/api/runs`, `/api/run?run=ID`, `/api/game?run=ID&index=I&lite=1`,
`/api/game-step?run=ID&index=I&step=S&lite=1`, `/api/frames?run=ID&index=I`,
`/api/timeline?run=ID&game=<game_id>_p<pass>`, and `/api/file?run=ID&path=P` for a run's
`solver_analysis/*.html`, `transcripts/*.txt` and `summary.txt`. Run ids are paths under
`--runs-dir`; anything outside it is refused.

This fork corrects per-turn transcript display, includes all recorded attempts
when an analysis step is retried, and distinguishes pending transcripts from
the latest board state. Per-step token usage is displayed when request logs
contain the corresponding usage data.

## Scoring

Score one run directory:

```bash
make score_run SCORE_RUN_DIR=/path/to/run
```

Evaluate runs from `configs/eval.json`:

```bash
make eval
```

Write a score file somewhere specific:

```bash
make score_run SCORE_RUN_DIR=/path/to/run SCORE_OUTPUT_PATH=docs/candidate-score.json
```

The scorer reads TAAF `benchmark.json`, uses persisted `final_score` values when
present, and otherwise asks TAAF's `GameRun` scorer to compute the score from
the saved state. It writes `evaluation.json` plus the lightweight `score.json`
format used by significance checks.

## Significance

Compare a candidate score file against a current best:

```bash
make significance \
  BASELINE_SCORE=docs/current-best-score.json \
  CANDIDATE_SCORE=docs/candidate-score.json
```

Or configure those paths in `configs/significance.json` and run:

```bash
make significance
```

The comparison aligns by `game_id`, averages repeated trials within each game,
and uses games as the paired unit. It checks runtime budget, hardware, dataset
metadata, and trial counts before reporting whether the candidate passes the
internal-highscore threshold:

```text
P(true_delta > 0 | results) >= 0.90
```

The output also includes win rate, a bootstrap 90% interval, and TAAF paired
test p-values as robustness checks.

## Trace Export

Export machine-readable per-episode duck traces:

```bash
make traces
```

For runs outside `runs/`, call the tool directly:

```bash
uv run --no-sync inference-traces --runs-dir /shared/arc_3_results/$USER
```

Traces are written in live-chat `messages` format. They preserve assistant
reasoning, tool calls, compact tool results, actions, scores, and level
transitions linked back to message indices.

## Useful Commands

- `make install`: create `.venv` and install base, server, and development dependencies.
- `make server`: start local vLLM.
- `make interactive`: run through TAAF inline deployment.
- `make sbatch`: submit through TAAF Slurm deployment.
- `make chat PROMPT="..."`: send a direct chat probe to the configured model.
- `make view`: serve the run viewer.
- `make score_run SCORE_RUN_DIR=...`: score one saved run.
- `make eval`: score runs selected by an eval config.
- `make significance`: compare two score files.
- `make traces`: export trace JSON.
- `make zip`: zip the local `runs/` directory.
- `make clean`: remove local `runs/` artifacts.

## Repo Map

- `inference/framework/run.py`: CLI entry point and TAAF deployment setup.
- `inference/framework/solver.py`: TAAF solver adapter, action execution,
  viewer events, transcripts, and local-server orchestration.
- `inference/agent/tool_agent.py`: OpenAI-compatible tool-calling duck.
- `inference/agent/python_tool_sandbox.py`: isolated Python tool runtime.
- `inference/agent/priority_scheduler.py`: priority scoring and continuation tables.
- `inference/utils/segmentation.py`: connected-component board segmentation.
- `inference/utils/frame_diff.py`: cell and component change descriptions.
- `inference/utils/animation.py`: intermediate frames and compressed timelines.
- `inference/utils/retained_functions.py`: supported function and import retention.
- `inference/tools/eval.py`: TAAF score export.
- `inference/tools/significance.py`: paired score comparison.
- `inference/tools/traces.py`: trace export.
- `viewer/`: local browser UI for saved runs.
