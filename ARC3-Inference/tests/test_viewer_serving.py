import json
from pathlib import Path

from viewer.serving import serving_metrics
from viewer.server import _load_comparison_payload


PREFILL = (
    "[2026-10-01 12:00:00] Prefill batch, #new-token: 100, #cached-token: 900, "
    "full token usage: 0.25, #running-req: 0, input throughput (token/s): 1000.00\n"
)
DECODE = (
    "[2026-10-01 12:00:01] Decode batch, #running-req: 10, full token usage: 0.50, "
    "gen throughput (token/s): 800.00, #queue-req: 0\n"
)


def test_sglang_serve_log_recovers_missing_metrics(tmp_path: Path) -> None:
    (tmp_path / "serve.log").write_text(PREFILL + DECODE)
    metrics = serving_metrics(tmp_path, {"logs_found": 0, "sample_count": 0})
    server = metrics["per_server"][0]
    assert metrics["logs_found"] == 1
    assert server["sample_count"] == 2
    assert server["prompt_throughput_tokens_per_s"]["mean"] == 1000
    assert server["generation_throughput_tokens_per_s"]["mean"] == 800
    assert server["prefix_cache_hit_rate_pct"]["mean"] == 90
    assert server["gpu_kv_cache_usage_pct"]["mean"] == 37.5
    assert metrics["metric_labels"]["prefix_cache_hit_rate_pct"] == "Logged prefill cache reuse / %"


def test_cache_reuse_weights_tokens_and_missing_fields_stay_missing(tmp_path: Path) -> None:
    (tmp_path / "serve.log").write_text(
        PREFILL + "Prefill batch, #new-token: 9000, #cached-token: 0\n"
    )
    server = serving_metrics(tmp_path, None)["per_server"][0]
    assert server["prefix_cache_hit_rate_pct"]["mean"] == 9
    assert server["generation_throughput_tokens_per_s"] is None


def test_preserves_evaluated_metrics_and_handles_missing_logs(tmp_path: Path) -> None:
    recorded = {"sample_count": 4, "generation_throughput_tokens_per_s": {"mean": 123}}
    (tmp_path / "serve.log").write_text(DECODE)
    assert serving_metrics(tmp_path, recorded) is recorded
    assert serving_metrics(tmp_path / "absent", {}) == {}


def test_vllm_log_aliases_and_idle_samples(tmp_path: Path) -> None:
    line = (
        "Avg prompt throughput: 100 tokens/s, Avg generation throughput: 200 tokens/s, "
        "Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 25%, Prefix cache hit rate: 75%\n"
    )
    idle = (
        "Avg prompt throughput: 0 tokens/s, Avg generation throughput: 0 tokens/s, "
        "Running: 0 reqs, Waiting: 0 reqs\n"
    )
    (tmp_path / "server.log").write_text(line + idle)
    (tmp_path / "serve.log").write_text(line)
    metrics = serving_metrics(tmp_path, {})
    assert metrics["logs_found"] == 1
    assert metrics["per_server"][0]["generation_throughput_tokens_per_s"]["mean"] == 200


def test_changed_log_refreshes_metrics(tmp_path: Path) -> None:
    path = tmp_path / "serve.log"
    path.write_text(DECODE)
    assert serving_metrics(tmp_path, {})["sample_count"] == 1
    path.write_text(DECODE + DECODE.replace("800.00", "400.00"))
    metrics = serving_metrics(tmp_path, {})
    assert metrics["sample_count"] == 2
    assert metrics["per_server"][0]["generation_throughput_tokens_per_s"]["mean"] == 600


def test_comparison_recovers_logs_without_rewriting_evaluation(tmp_path: Path) -> None:
    run = tmp_path / "kaggle/alice/notebook/v01"
    run.mkdir(parents=True)
    score = {"score": 10, "server": {"logs_found": 0}}
    original = json.dumps(score)
    (run / "evaluation.json").write_text(original)
    (run / "serve.log").write_text(PREFILL + DECODE)
    payload = _load_comparison_payload(tmp_path)
    evaluation = payload["runs"]["kaggle/alice/notebook/v01"]["scores"]["public"]["score"]
    assert evaluation["score"] == 10
    assert evaluation["server"]["logs_found"] == 1
    assert (run / "evaluation.json").read_text() == original
