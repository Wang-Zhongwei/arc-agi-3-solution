"""Recover serving metrics from SGLang or vLLM logs that ``server_metrics`` cannot read.

Shared by the evaluator (``evaluation.json``) and the viewer (older score files).
"""
from __future__ import annotations

import re
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

_METRICS = (
    "prompt_throughput_tokens_per_s", "generation_throughput_tokens_per_s",
    "prefix_cache_hit_rate_pct", "gpu_kv_cache_usage_pct",
)
SERVER_LOG_NAMES = ("server.log", "serve.log", "vllm.log", "sglang.log", "vllm-openai-server.log")
_NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
_FIELDS = {
    "input": re.compile(r"input throughput \(token/s\):\s*" + _NUMBER),
    "generation": re.compile(r"gen throughput \(token/s\):\s*" + _NUMBER),
    "kv": re.compile(r"full token usage:\s*" + _NUMBER),
    "new": re.compile(r"#new-token:\s*(\d+)"),
    "cached": re.compile(r"#cached-token:\s*(\d+)"),
}
_VLLM_FIELDS = {
    "prompt_throughput_tokens_per_s": re.compile(r"Avg prompt throughput:\s*" + _NUMBER),
    "generation_throughput_tokens_per_s": re.compile(r"Avg generation throughput:\s*" + _NUMBER),
    "prefix_cache_hit_rate_pct": re.compile(r"Prefix cache hit rate:\s*" + _NUMBER),
    "gpu_kv_cache_usage_pct": re.compile(r"GPU KV cache usage:\s*" + _NUMBER),
}


def _value(pattern: re.Pattern, line: str) -> float | None:
    match = pattern.search(line)
    return float(match.group(1)) if match else None


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    deciles = statistics.quantiles(values, n=10, method="inclusive") if len(values) > 1 else [values[0]] * 9
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "p10": deciles[0], "p90": deciles[-1]}


@lru_cache(maxsize=32)
def _parse_log(path: str, mtime_ns: int, size: int) -> dict[str, Any] | None:
    # Stat arguments invalidate cached extraction when a log changes.
    values: dict[str, list[float]] = {key: [] for key in _METRICS}
    cached_tokens = new_tokens = samples = 0
    cache_reuse_percentages: list[float] = []
    backends: set[str] = set()
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        if "Avg prompt throughput:" in line:
            backends.add("vllm")
            samples += 1
            fields = {key: _value(pattern, line) for key, pattern in _VLLM_FIELDS.items()}
            running = re.search(r"Running:\s*(\d+)", line)
            waiting = re.search(r"Waiting:\s*(\d+)", line)
            active = any(fields[key] for key in _METRICS[:2]) or any(
                match and int(match.group(1)) for match in (running, waiting)
            )
            for key, value in fields.items():
                if value is not None and (key not in _METRICS[:2] or active):
                    values[key].append(value)
        elif "Prefill batch," in line or "Decode batch," in line:
            backends.add("sglang")
            samples += 1
            prefill = "Prefill batch," in line
            throughput = _value(_FIELDS["input" if prefill else "generation"], line)
            if throughput is not None:
                key = _METRICS[0 if prefill else 1]
                values[key].append(throughput)
            kv = _value(_FIELDS["kv"], line)
            if kv is not None:
                values[_METRICS[3]].append(100 * kv)
            if prefill:
                new = _value(_FIELDS["new"], line)
                cached = _value(_FIELDS["cached"], line)
                if new is not None and cached is not None:
                    new_tokens += int(new)
                    cached_tokens += int(cached)
                    if new + cached > 0:
                        cache_reuse_percentages.append(100 * cached / (cached + new))
    if not samples:
        return None
    result = {key: _stats(items) for key, items in values.items()}
    result.update(log=path, sample_count=samples, backend="/".join(sorted(backends)))
    if backends == {"sglang"}:
        total = cached_tokens + new_tokens
        # This is a ratio of logged token counts, not a mean of batch percentages
        # and not the sliding-window hit rate emitted by vLLM.
        cache_stats = _stats(cache_reuse_percentages)
        if cache_stats is not None:
            cache_stats["mean"] = 100 * cached_tokens / total
        result[_METRICS[2]] = cache_stats
        result["prefill_tokens"] = {"cached": cached_tokens, "new": new_tokens}
    return result


def recover_server_block(log_dirs: list[Path], recorded: dict[str, Any] | None) -> dict[str, Any]:
    """Keep populated metrics; otherwise recover them from each directory's serving log."""
    recorded = recorded or {}
    if recorded.get("sample_count") or any(recorded.get(key) for key in _METRICS):
        return recorded
    servers = []
    for directory in log_dirs:
        # Alternative filenames for the same server: do not count them twice.
        for name in SERVER_LOG_NAMES:
            path = directory / name
            try:
                stat = path.stat()
            except OSError:
                continue
            parsed = _parse_log(str(path), stat.st_mtime_ns, stat.st_size)
            if parsed:
                servers.append(parsed)
                break
    if not servers:
        return recorded
    block = {**recorded, "per_server": servers, "logs_found": len(servers),
             "sample_count": sum(item["sample_count"] for item in servers),
             "source": "serving logs", "aggregation": "per-server"}
    if len(servers) == 1:
        # One server: its figures are the run's, so expose them where vLLM blocks keep them.
        block.update({key: servers[0][key] for key in _METRICS if servers[0].get(key)})
    sglang = any("sglang" in item["backend"] for item in servers)
    if sglang:
        block["metric_labels"] = {
            _METRICS[2]: "Logged prefill cache reuse / %",
            _METRICS[3]: "KV token pool use / %",
        }
        block["notes"] = (
            "SGLang throughput uses means of logged prefill/decode samples, not whole-run rates. "
            "Prefill cache reuse is cached / (cached + new) logged tokens; "
            "its mean is token-weighted and P10/P90 describe individual prefill batches. "
            "KV use is the logged full-token-pool occupancy. "
            "These differ from vLLM interval throughput and sliding-window cache hit rates."
        )
    else:
        block["notes"] = "Recovered from vLLM logs; throughput averages active samples only."
    return block
