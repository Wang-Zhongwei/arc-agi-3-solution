"""The CPU example (inference/grpo/cpu_example.py) runs the pipeline end to end on a tiny model, through the loop's
build_job and the trainer's run_job: step 1's first minibatch trains data played by the same weights (log ratio ~0),
later minibatches drift one update each, the first sync carries the trainer's update into the server (the loop's
probe), and step 2 trains a wave played a step of updates earlier."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp")
pytest.importorskip("peft")

from inference.grpo import cpu_example  # noqa: E402


def test_the_example_runs_minibatches_on_policy_first_then_further_behind(tmp_path):
    out = cpu_example.main(["--steps", "2", "--max-turns", "4", "--work", str(tmp_path)])
    first, second = out["history"]
    assert len(first["grad_norms"]) == len(first["log_ratio_by_minibatch"]) >= 2
    on_policy = first["log_ratio_by_minibatch"][0]
    assert on_policy < 1e-5  # only rounding between the patched and plain paths
    assert first["log_ratio_by_minibatch"][-1] > 10 * on_policy  # updates later in the same step
    assert second["log_ratio_by_minibatch"][0] > 10 * on_policy  # a wave played before step 1's updates
    assert out["probe"]["gap_before"] < 1e-4 and out["probe"]["change_corr"] > 0.9
