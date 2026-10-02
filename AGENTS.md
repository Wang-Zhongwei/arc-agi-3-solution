# Repository guidance

## Workflow

- Follow [commit rules](docs/rules/README.md) and the [A/B comparison guide](docs/rules/t-test.md). Work on your role's branch; document tested changes before opening a PR.
- Use Make targets from `ARC3-Inference/`. If local editable imports fail, set `PYTHONPATH=.:../tufa-arc-agi-framework/src` before the command.
- Keep parameter definitions in [CONFIGURATION.md](ARC3-Inference/CONFIGURATION.md), role plans in `docs/roles/`, experiment reports in `docs/<theme>/`, and submission decisions in `docs/submission/`.

## Data sync and viewer

- Share completed run folders through Google Drive. Preserve `ARC3-Inference/runs/kaggle/<owner>/<notebook>/vNN/`, including viewer artifacts, event sidecars, request logs, configs, and evaluations.
- Keep run artifacts out of Git. Reference exact notebook versions in reports.
- Run `make view PORT=8022` and open `http://127.0.0.1:8022`. Select one run with `VIEW_RUN_DIR=/path/to/run`; comparison pages require `evaluation.json`.

## Submit to Kaggle

- Use `make kaggle-duck CONFIG_PATH=configs/<experiment>.json` with the intended `KAGGLE_USERNAME`, `KAGGLE_KERNEL_SLUG`, and `KAGGLE_DATASET_REF`. Confirm the destination and effective settings before pushing.
- Kaggle numbers notebook versions automatically; keep version numbers out of notebook slugs/titles and dataset slugs.
- Keep credentials outside Git. If using a configured alternate profile, prefix the command with `kaggle-as <profile>` and ensure private input datasets are shared with that account.
- Resolve both flat and namespaced Kaggle mounts for wheelhouses and datasets. Prefer this run's packaged TAAF source over a serving bundle's copy.
- Record the pushed version. `QUEUED` or `RUNNING` means submitted, not evaluated; report results only from completed, exact-version artifacts.
