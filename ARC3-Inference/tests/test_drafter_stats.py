"""drafter_stats' log readers on synthetic vLLM metrics lines: the --until window and the timeline buckets."""
from inference.tools import drafter_stats

ENGINE = ("(APIServer pid=1) INFO 10-31 {t} [loggers.py:320] Engine 000: Avg prompt throughput: {p} tokens/s, "
          "Avg generation throughput: {g} tokens/s, Running: 14 reqs, Waiting: {w} reqs, GPU KV cache usage: {kv}%, "
          "Prefix cache hit rate: {hit}%, MM cache hit rate: 0.0%\n")
SPEC = ("(APIServer pid=1) INFO 10-31 {t} [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.00, "
        "Accepted throughput: 1.0 tokens/s, Drafted throughput: 1.0 tokens/s, Accepted: {a} tokens, Drafted: {d} tokens, "
        "Per-position acceptance rate: {pos}, Avg Draft acceptance rate: 50.0%\n")


def write_log(tmp_path):
    lines = [
        "(APIServer pid=1) INFO:     Application startup complete.\n",
        ENGINE.format(t="15:00:00", p=1000, g=900, w=1, kv=20.0, hit=80.0),
        SPEC.format(t="15:00:00", a=60, d=90, pos="0.800, 0.700, 0.500"),
        ENGINE.format(t="15:03:00", p=3000, g=700, w=3, kv=40.0, hit=90.0),
        SPEC.format(t="15:03:00", a=30, d=30, pos="1.000, 1.000, 1.000"),
        ENGINE.format(t="15:06:00", p=9000, g=200, w=2, kv=95.0, hit=85.0),
        SPEC.format(t="15:06:00", a=0, d=30, pos="0.000, 0.000, 0.000"),
    ]
    log = tmp_path / "vllm-openai-server.log"
    log.write_text("".join(lines))
    return log


def test_accept_until_keeps_the_first_minutes(tmp_path):
    log = write_log(tmp_path)
    whole = drafter_stats.acceptance(log)
    assert whole["lines"] == 3 and whole["length"] == round(1 + 90 / (150 / 3), 3)
    early = drafter_stats.acceptance(log, until=5)
    assert early["lines"] == 2 and early["length"] == round(1 + 90 / (120 / 3), 3)
    assert early["per_position"] == [0.9, 0.85, 0.75]


def test_timeline_buckets_by_minutes_after_the_first_metrics_line(tmp_path):
    rows = drafter_stats.timeline(write_log(tmp_path), bucket=5)
    assert [r["minutes"] for r in rows] == [[0, 5], [5, 10]]
    assert rows[0]["prompt"] == 2000 and rows[0]["gen"] == 800 and rows[0]["kv"] == 30.0 and rows[0]["kv_max"] == 40.0
    assert rows[0]["length"] == round(1 + 90 / (120 / 3), 3)
    assert rows[1]["prompt"] == 9000 and rows[1]["length"] == 1.0 and rows[1]["waiting"] == 2.0
