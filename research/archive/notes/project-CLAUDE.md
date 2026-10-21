# ARC-AGI-3 Kaggle Competition

Goal: maximize the full private score across all 110 games of the ARC-AGI-3 benchmark, within Kaggle's compute budget.

ARC-AGI-3 tests a model + harness on interactive games built on François Chollet's Core Knowledge Priors:
- **Objectness:** objects persist, have boundaries, and don't appear or vanish without cause.
- **Agentness:** some entities act with goals.
- **Geometry & topology:** shape, symmetry, spatial relations, containment.
- **Numbers & counting:** comparing quantities, sorting, simple arithmetic.

## The core idea
The model is the brain; the harness has to supply everything else a human player has:
- **Perception:** turn frames into something the model can reason about (images, diffs, segmentation).
- **Memory:** what was tried, what happened, and what carries over between levels.
- **Reasoning:** tools such as a Python sandbox for testing hypotheses.
- **Judgement:** when to explore, exploit, reset, or move on.

We **extend** Franzen's core (new perception, planning, memory and tools); we don't only tune it. Tuning is fine as an enabler that buys compute for extensions.

Efficiency is a hard constraint. Judge every change by score per unit of compute (tokens, wall-clock, cache reuse), not by score alone.

## Layout
- `daniel-franzen-sol/`: Daniel Franzen's Progress Prize #2 solution, our baseline. It's a fork of Tufa Labs' Duck harness.
  - `ARC3-Inference/`: the harness. `inference/agent/tool_agent.py` is the agent loop; `inference/framework/solver.py` and `run.py` handle orchestration; `inference/agent/prompts.py` holds the prompts; `viewer/` is the run viewer.
  - `serving/`: the SGLang runtime build and patches. `tufa-arc-agi-framework/`: the game execution framework.
  - `WRITEUP.md` says which features were used in the competition and which were failed experiments. Read it before changing behavior.
  - `ARC3-Inference/CONFIGURATION.md` is the option reference.
- `/Users/advaith/Documents/studying-stuff/arc-agi-3-v3/` (outside `experiments/`): the user's old Milestone 2 notebook work. Several variant folders each contain a `taaf-duck-harness-kaggle-share.ipynb` (`franzen*`, `final-c1`, `wang-duck-harness`, `tufalabs-duck-harness`, `gb10-wt/*`), plus `kaggle-outputs/` and `reports/`. Read-only reference; don't edit it.

## Commands (run in `daniel-franzen-sol/ARC3-Inference/`)
- `uv sync --locked`: install.
- `make view VIEW_RUN_DIR=<run> VIEW_PORT=8011`: inspect a saved run (no GPU needed).
- `make eval`, `make significance`, `make traces`: analyze runs.

## Arms (experiments)
An **arm** is one feature or idea we try. Every arm:
1. Gets its own branch off `main` in `daniel-franzen-sol/`, named `arm/<short-name>`. Never commit straight to `main`.
2. Is built in several small commits, one logical change each, with imperative messages (`Add death-image diff to prompt`). Keep refactors separate from behavior changes.
3. Gets its own PR from that branch. The PR description covers the hypothesis, what changed, how it was tested, and the results (score and compute cost).
4. Is logged in `arms.md` (in this folder) at three points: under **Not yet implemented** when it's an idea, and under **Worked** or **Didn't work** once results are in. Move entries; never delete them. The list keeps growing as the user and Claude brainstorm, so add new ideas as they come up.

## Memory files (in `arc-agi-v4/`, outside the repo so all branches share them)
- `arms.md`: the target, plus Worked / Didn't work / Not yet implemented (a ranked backlog; re-rank it as results come in). Read it before proposing an idea so we don't repeat a dead end.
- `scratchpad.md`: the user's own thinking, such as what might be an alpha, what definitely isn't, hunches and open questions. When the user shares a view like this, write it down in their words, with the date.

## Other reference files (in `arc-agi-v4/`)
- `research-sweep.md`: a ranked list of improvement ideas, checked against Franzen's code, Kaggle constraints and `arms.md`.
- `competitors/`: shallow clones of other contestants' code and pulled Kaggle notebooks. Read-only.
- Bottleneck issues are #2–#33 on GitHub, labelled `DEFINITELY` or `MAYBE`.

## Getting run information
- **Kaggle CLI** (installed and logged in): `kaggle competitions submissions arc-prize-2026-arc-agi-3` lists scores. `kaggle kernels status|logs|output <user>/<kernel>` gives a notebook run's status, logs and output files (use `-p <dir>` to choose where they download).
- **Computer use / browser:** for anything the CLI can't show (leaderboard, notebook UI, discussion threads), hand the task to a subagent with `model: "sonnet"` (Sonnet 5.5) that uses the Chrome tools, and have it report back the facts. Always use Sonnet for this, never the main model.
- **Run arms on Kaggle** with `daniel-franzen-sol/kaggle/` (see its README): `python kaggle/build_notebook.py --arm <arm> --push`, or `kaggle/push_queue.sh <arms...>`.
- Write each run's results into the matching arm in `arms.md`.

## Working rules
- **Kaggle model:** always test with the W4A16 Qwen3.8-Flash-Next (`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround`, plus the albucino MTP drafter), as in `kaggle/template-metadata.json`. Don't swap quantizations unless that is the arm.
- **Target:** 56–60 on the 25 public games in the 121-min offline run (see `arms.md`).
- **Keep the READMEs current as the project grows.** Any PR that adds or changes behaviour, tooling or how to run things must update the relevant docs in the same PR:
  - `daniel-franzen-sol/README.md`: the overview of what this fork adds beyond Franzen, and how to reproduce and run it.
  - The area READMEs: `kaggle/README.md`, `serving/CUSTOM.md`, `analysis/README.md`, and `ARC3-Inference/CONFIGURATION.md` for every new flag.
  - Revise stale sections; don't only append. When a PR merges to `main`, check the top-level README still describes what `main` actually does.
- **Laptop load:** run heavy local work (simulations, test suites, benchmarks) one at a time and at low priority (`nice`). Parallel heavy jobs throttled the user's laptop.
- Git: the repo is `daniel-franzen-sol/` (the parent folder is not a repo). Remote: `github.com/adsdemaybe/daniel-franzen-sol`.
- Prompt strings in `prompts.py` and the sandbox bootstrap are behavior. Edit them only on purpose.
- Before claiming an improvement, compare score files with `make significance BASELINE_SCORE=... CANDIDATE_SCORE=...`, which runs a paired per-game test; game scores are noisy.
- **An alpha is valid only if repeated runs consistently score higher** (user's rule). One run, or a 20-min panel, is a screen that decides whether to spend the repeats.
