# RunPod smoke test

`make runpod-duck` runs the notebook bundle that `make kaggle-duck` pushes, on a RunPod GPU pod instead of Kaggle. This smoke test checks that the bundle, setup commands and Flash Next serving work end to end outside Kaggle. It is not a Kaggle score, so keep it out of A/B comparisons.

## Run

- Config: [`configs/runpod-smoke/v1.json`](../../ARC3-Inference/configs/runpod-smoke/v1.json), which is `thinking-priority/v2` with `public_games` set to ft09 and vc33 and a 25-minute public budget. The share notebook, Pennyroyal serving and both Flash Next models are mapped to pinned Hugging Face revisions.
- Pod: `c7gq8re8ot95cg`, 1× RTX PRO 6000 Blackwell, secure cloud, $2.09/h. The run took about 31 minutes, roughly $1.10.
- Output: private Kaggle dataset `jonathanwang2022/runpod-smoke-rp-c7gq8re8ot95cg`, copied to `runs/runpod/jonathanwang2022/runpod-smoke/c7gq8re8ot95cg/`.

| Stage | Time |
|---|---|
| Staging (competition files, 6.6 GB Pennyroyal wheels from Kaggle, 181 GB model + drafter from Hugging Face, pip) | 324 s |
| Hugging Face model download (181 GB) | 113 s, about 1.6 GB/s |
| Kaggle wheelhouse download (6.6 GB) | 171 s, about 39 MB/s |
| Pennyroyal install + server ready | 583 s |
| Play | 14 m 42 s |

The notebook exited with code 0 and wrote `benchmark.json`, the diagnostics and the dummy `submission.parquet`, as on Kaggle.

| Game | Score | Levels | Actions |
|---|---|---|---|
| ft09 | 100.00 | 6/6 | 94 |
| vc33 | 20.93 | 3/7 | 110 |

## Problems found

1. Cloudflare in front of `rest.runpod.io` rejects Python's default User-Agent (403, error 1010). The client now sends its own.
2. Two pods sat `RUNNING` without ever starting their container. Six probe pods afterwards started within 32 s, so this was a host problem. `deploy()` now deletes a pod whose container has not started after `runpod.startup_timeout_s` (900 s).
3. The notebook's offline `pip install arc-agi` failed with `uninstall-no-record-file` over the image's apt-managed Python packages. The pod runner now pre-installs the competition's own `arc-agi` wheel with `--ignore-installed`.
4. Kaggle expands an uploaded `working.tar.gz` into the dataset's files, so `make pull-runpod` now accepts an already-expanded download.
