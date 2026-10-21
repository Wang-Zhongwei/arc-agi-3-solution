# Kaggle run logs and outputs

Every notebook version run on the team's Kaggle account (`mellamoa`) for ARC-AGI-3, with its execution log and all of its output files. The first snapshot was taken on 2026-10-05, and later runs are added as they finish. As of 2026-10-08, `index.csv` lists 126 versions of 94 kernels, and 124 of those versions have files here.

Not included: `notebook024611ef81`, a July notebook unrelated to ARC (language LoRA adapters). Also missing: `arc3-franzen-m2-exact-10-game-validation` and `duck-arm-mtp-off-drop-multi-token-prediction`, which the API no longer serves any version of, and `v4-dprime`, which was cancelled while queued and has an empty version 1 (it still has a row in `index.csv`).

## Layout

- `kaggle/<kernel>/v<N>/`: one kernel version, downloaded through the Kaggle API.
  - `<kernel>.log`: the notebook's execution log.
  - The output files, typically:
    - `benchmark.json`: per game, the action history with tokens and wall-clock time, levels and score.
    - `summary.txt` and `evaluation.json`: the scores.
    - `serve.log`: SGLang decode/prefill timings.
    - `transcripts/`: per-game model transcripts.
    - `*_requests.jsonl`: the full request payloads, on runs that logged them.
- `local/<folder>/`: runs downloaded earlier by hand from `arc-agi-3-v3/kaggle-outputs`. Only files that are not byte-identical to a file under `kaggle/` are kept. Those are mostly versions the API no longer serves, plus locally computed evaluations.
- `index.csv`: one row per kernel version or local folder, with file count, raw size and the score where a summary file states one.
  - `raw_mb` is the uncompressed size in MB (10^6 bytes) of the files committed here.
  - `score` is always a number or empty.
  - `v4-gguf-final` v1 was cancelled at about 61 min, so its 2.33 covers only the games that had played by then.
- `submissions.csv`: every leaderboard submission and its public score (`kaggle competitions submissions`).

Kernel names map to the experiments in [`research/notes/arms.md`](../research/notes/arms.md):
- `v4-*`: arms built from this repo by `kaggle/build_notebook.py`.
- `fz-*`: Franzen-based arms from the earlier notebook.
- `taaf-*` and `duck-*`: the older Wang/Duck harness.

## Compression

Files over 64 KB are stored as `<name>.xz` (xz -9), except `summary.txt` and `evaluation.json`, which stay plain so scores read on GitHub. 41.5 GB of raw output packs down to the size of this folder: request logs repeat the whole conversation in every request and shrink about 290×, transcripts and viewer pages about 490×. The per-token logprob logs (`*_signals.jsonl`, from the old harness's introspection runs) compress worst, about 6×. Unpack everything in place with:

```sh
./runs/unpack.sh            # or: find runs -name '*.xz' -exec xz -dk {} +
```

From Python you can read the files without unpacking: `lzma.open("runs/kaggle/fz-ucb/v1/benchmark.json.xz")`.

## Scoring a run

From `ARC3-Inference/`, after unpacking:

```sh
uv run python -m inference.tools.eval ../runs/kaggle/<kernel>/v<N> --score-output score.json
make significance BASELINE_SCORE=a.json CANDIDATE_SCORE=b.json
```

Per-game scores swing by 50–80 points between identical runs, so compare runs with the paired test, not by eye.
