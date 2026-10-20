"""inference/rl/wandb_log.py: train_rft log rows become the same wandb metrics live and in a replay."""

import json

from inference.rl import wandb_log


class FakeRun:
    def __init__(self):
        self.rows, self.axes, self.summary = [], {}, {}

    def define_metric(self, name, step_metric=None):
        self.axes[name] = step_metric

    def log(self, metrics):
        self.rows.append(metrics)


ROWS = [
    {"t": 1.0, "event": "loaded", "args": {"lr": "0.0001"}},
    {"t": 2.0, "event": "arena", "cap_gb": 113.2, "limit_gb": 233.8, "ple_gb": 95.4, "process_gb": 3.2, "reserve_s": 39.8},
    {"t": 3.0, "event": "val", "epoch": 0, "loss": 0.31, "tokens": 417343, "samples": 16},
    {"t": 4.0, "event": "sample", "step": 0, "tokens": 100, "saved_tokens": 40, "loss": 0.3, "forward_s": 1.0,
     "backward_s": 2.0, "tokens_per_s": 33.3, "recompute_mode": "layer", "peak_gb": {"cuda:0": 90.0},
     "arena_used_gb": 50.0, "major_faults": [0, 3], "error": None, "wall_start": 1000.0},
    {"t": 5.0, "event": "sample", "step": 0, "tokens": 80, "error": "layer: CUDA out of memory"},
    {"t": 6.0, "event": "step", "step": 0, "epoch": 0, "lr": 1e-5, "grad_norm": 0.12, "loss": 0.3,
     "trained_tokens": 40, "seconds": 3.0},
    {"t": 7.0, "event": "val", "epoch": 1, "loss": 0.29, "tokens": 417343, "samples": 16},
]


def test_rows_map_to_metrics_on_their_own_axes():
    run = FakeRun()
    logger = wandb_log.WandbLogger(run)
    for row in ROWS:
        logger.event(row)
    assert run.axes["train/*"] == "train/step" and run.axes["sample/*"] == "sample/index"
    assert run.axes["val/*"] == "train/step" and run.axes["gpu/*"] == "gpu/elapsed_s"
    val0, sample1, sample2, step0, val1 = run.rows
    assert val0 == {"train/step": 0, "val/loss": 0.31, "val/tokens": 417343}
    assert sample1["sample/index"] == 1 and sample1["sample/layer"] == 1.0 and sample1["sample/peak_gb"] == 90.0
    assert sample1["sample/major_faults"] == 3 and sample1["sample/failed"] == 0.0
    assert sample2 == {"sample/index": 2, "sample/step": 0, "sample/tokens": 80, "sample/layer": 0.0, "sample/failed": 1.0}
    assert step0["train/step"] == 0 and step0["train/grad_norm"] == 0.12
    assert val1["train/step"] == 1 and val1["val/loss"] == 0.29  # after step 0: plotted at step 1
    assert run.summary["arena/cap_gb"] == 113.2


def test_replay_reads_a_run_folder_with_gpu_samples(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "train" / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    # nvidia-smi --format=csv,noheader: timestamp, utilization.gpu, power.draw, memory.used (UTC 1970: wall 1000 s)
    (tmp_path / "results" / "gpu.csv").write_text("1970/01/01 00:16:40.000, 97 %, 650.5 W, 120000 MiB\n"
                                                    "1970/01/01 00:16:41.000, 12 %, 150.0 W, 120000 MiB\nbad row\n")
    run = FakeRun()
    counts = wandb_log.replay(tmp_path, wandb_log.WandbLogger(run))
    assert counts == {"events": len(ROWS), "samples": 2, "gpu_rows": 2}
    gpu = [r for r in run.rows if "gpu/util" in r]
    assert gpu[0] == {"gpu/elapsed_s": 0.0, "gpu/util": 97.0, "gpu/power_w": 650.5, "gpu/memory_gb": 120000 / 1024}
    assert gpu[1]["gpu/elapsed_s"] == 1.0
