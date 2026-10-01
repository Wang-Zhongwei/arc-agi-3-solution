# Repository guidance

## Versions and Kaggle notebook names
A version in a config path (e.g. `ARC3-Inference/configs/scheduling/v1.json`) means the Kaggle notebook version only. Never put it in `kaggle.kernel_slug`, `kaggle.kernel_title` or `kaggle.dataset_slug`: one experiment is one notebook, and Kaggle auto-numbers its versions on each push.

## Running commands
Use `make` targets (from `ARC3-Inference/`, e.g. `make kaggle-duck CONFIG_PATH=...`), not `uv run ...` or `.venv/bin/...` directly. The repo lives under iCloud-synced `~/Documents`, which keeps flagging files in dot-folders such as `.venv` as hidden; Python skips hidden `.pth` files, so the venv's editable installs of `inference` and `taaf` stop importing. The Makefile's `UV_RUN` sets `PYTHONPATH` to both source trees, so make targets are unaffected. If no target fits, use that same `PYTHONPATH` prefix.
