"""The CPU example (inference/grpo/cpu_example.py) runs the pipeline end to end on a tiny model: step 1 trains data
played by the same weights (log ratio ~0), the first sync carries the trainer's update into the server (the loop's
probe), and step 2 trains data played one update earlier."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")
pytest.importorskip("peft")

from inference.grpo import cpu_example  # noqa: E402


def test_the_example_runs_two_steps_on_policy_then_one_update_behind(tmp_path):
    out = cpu_example.main(["--steps", "2", "--max-turns", "4", "--work", str(tmp_path)])
    first, second = out["history"]
    assert first["log_ratio_abs_mean"] < 1e-5  # on-policy: only rounding between the patched and plain paths
    assert second["log_ratio_abs_mean"] > 10 * first["log_ratio_abs_mean"]  # one update behind
    assert out["probe"]["gap_before"] < 1e-4 and out["probe"]["change_corr"] > 0.9
