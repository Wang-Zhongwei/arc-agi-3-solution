# Repository guidance

## Workflow

- Follow [commit rules](docs/rules/README.md) and the [A/B comparison guide](docs/rules/t-test.md). Work on your role's branch; document tested changes before opening a PR.
- Use Make targets from `ARC3-Inference/`, not `uv run ...` or `.venv/bin/...` directly. The repo lives under iCloud-synced `~/Documents`, which marks files in dot-folders such as `.venv` hidden; Python skips hidden `.pth` files, so the venv's editable installs stop importing. The Makefile's `UV_RUN` sets `PYTHONPATH` to both source trees; if no target fits, prefix the command with `PYTHONPATH=.:../tufa-arc-agi-framework/src`.
- Keep parameter definitions in [CONFIGURATION.md](ARC3-Inference/CONFIGURATION.md), role plans in `docs/roles/`, experiment reports in `docs/<theme>/`, and submission decisions in `docs/submission/`.

## Data sync and viewer

- Share completed run folders through Google Drive. Preserve `ARC3-Inference/runs/local/` (HPC runs) and `ARC3-Inference/runs/kaggle/<username>/<notebook-name>/vNN/` (Kaggle runs), including viewer artifacts, event sidecars, request logs, configs, and evaluations.
- Keep run artifacts out of Git. Reference exact notebook versions in reports.
- Run `make view PORT=8022` and open `http://127.0.0.1:8022`. Select one run with `VIEW_RUN_DIR=/path/to/run`; comparison pages require `evaluation.json`.

## Submit to Kaggle

- Use `make kaggle-duck CONFIG_PATH=configs/<experiment>.json` with the intended `KAGGLE_USERNAME`, `KAGGLE_KERNEL_SLUG`, and `KAGGLE_DATASET_REF`. Confirm the destination and effective settings before pushing.
- A version in a config path (e.g. `ARC3-Inference/configs/scheduling/v1.json`) is the Kaggle notebook version only. One experiment is one notebook and Kaggle numbers its versions on each push, so keep version numbers out of `kaggle.kernel_slug`, `kaggle.kernel_title` and `kaggle.dataset_slug`.
- Keep credentials outside Git. If using a configured alternate profile, prefix the command with `kaggle-as <profile>` and ensure private input datasets are shared with that account.
- Resolve both flat and namespaced Kaggle mounts for wheelhouses and datasets. Prefer this run's packaged TAAF source over a serving bundle's copy.
- Record the pushed version. `QUEUED` or `RUNNING` means submitted, not evaluated; report results only from completed, exact-version artifacts.

## Fetch from Kaggle

- From `ARC3-Inference/`, run `make pull-kaggle NOTEBOOK=[<username>/]<notebook-name> VERSION=<n>`. It saves that exact version's output and log to `runs/kaggle/<username>/<notebook-name>/vNN/`; the username defaults to your Kaggle credentials.
- Add `SOURCE=true` to also save the notebook to `kernel/` and the source snapshot it ran (`src/`, `benchmark_initial.pkl`, ...) into the same folder.
