"""Kaggle helpers for the ARC3 duck harness."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from inference.utils.openai_compat import normalize_provider

# The vLLM runtime is the image-layer dataset built from the pinned vLLM main nightly by the
# pack-vllm-image-runtime skill. The dataset carries its runtime overlay, which the kernel applies
# through the manifest-pinned applier before starting the server. This is the rc3 runtime plus
# GDN RecoverSSM, which the server environment switches on, exact C14 decode graphs at 44
# and 52 tokens with the rc3 SM120 GEMM at M52, and the vLLM main fixes #58489 (PLE prefetch
# ids), #58961 (QSA profiling KV cache), #58430 (allocator split limit in profile_run), #58114
# (PLE metadata without a GPU sync), #58434 and #58784 (padded one-token prompt tails under
# MTP) and #58720 (indexed expert mapping in load_weights).
DEFAULT_VLLM_RUNTIME_DATASET_SOURCE = "lordhansolo/vllm-main-e975732-arc3"
# Our mirror of primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8 @ 07915ee, with the repo's
# mtp_nvfp4/ overlay already flattened into the root, so the NVFP4 MTP head is what loads.
# The served name has to stay equal to shared.model_name, which is what the agent asks for.
DEFAULT_QWEN_MODEL_SOURCE = "lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1"
DEFAULT_SERVED_MODEL_NAME = "primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8"
DEFAULT_VLLM_PORT = 1234
DEFAULT_VLLM_MAX_MODEL_LEN = 32768
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_VLLM_DTYPE = "bfloat16"
# Empty leaves --gpu-memory-utilization off the command line and vLLM on its own default.
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = ""
DEFAULT_VLLM_KV_CACHE_DTYPE = "auto"
DEFAULT_VLLM_INDEXER_KV_DTYPE = ""
DEFAULT_VLLM_MAMBA_SSM_CACHE_DTYPE = "auto"
DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_VLLM_ENABLE_PREFIX_CACHING = False
# Zero leaves vLLM's physical cache-block granularity in place. The Qwen3.8
# experimental runtime accepts a finer logical prefix-hash unit while keeping
# coarser physical Mamba pages, so 128 recovers almost all of MTP's boundary
# loss without changing the KV allocation geometry.
DEFAULT_VLLM_PREFIX_MATCH_UNIT = 0
# -1 leaves the runtime default (dense Mamba checkpoints) unchanged; 0 keeps
# only request-replay boundaries and positive values retain one boundary per
# interval. The launcher passes the supported serve CLI option explicitly.
DEFAULT_VLLM_PREFIX_CACHE_RETENTION_INTERVAL = -1
# Pins --kv-cache-memory-bytes; 0 lets vLLM size the KV cache from gpu_memory_utilization after
# profiling the real activation peak and the CUDA graph pool, which gave 5.14 GiB in v187. Do not
# take the pool size v187's startup line named as fully utilising the GPU (8.84 GiB). That figure
# is built from the steady activation peak it profiled (1.78 GiB), while FlashInfer autotune runs
# a dummy forward at max_num_batched_tokens that took 3.39 GiB, and it counts nothing for the PLE
# offload worker, which holds 0.70 GiB of the same GPU from a separate process. Pinning 8 GiB on
# those numbers ran the card out of memory during warmup with 40 MiB free. Weights and non-torch
# memory leave 11.4 GiB, so the real ceiling is near 6.5 GiB and 6 keeps a margin under it.
DEFAULT_VLLM_KV_CACHE_GIB = 6
# Prompt budget handed to the agent: max_model_len minus room for one generation.
VLLM_OUTPUT_RESERVE_TOKENS = 4096
# ARC-AGI-3 renders every frame as a 64x64 grid, which the agent upscales by MULTIMODAL_UPSCALE.
ARC_FRAME_SIDE = 64
# MTP draft tokens per decode step; 0 disables speculative decoding. The draft pays because
# 8-10 sequences leave the decode latency-bound.
DEFAULT_VLLM_MTP_TOKENS = 3
# Fallback for server.max_num_seqs when the config carries no vLLM server block. The KV pool holds
# 4-5 sequences at the 17-25k contexts the agent runs, so this cap only bounds the scheduler and
# raising it past the pool queues work instead of serving it. Every sequence also reserves one
# mamba page of the same pool, which a lower cap gives back. With MTP-3 the CUDA graph capture size
# is four tokens per sequence, so 8 captures up to 32, the geometry v187 started on.
DEFAULT_VLLM_MAX_NUM_SEQS = 8
# --quantization, passed explicitly so a checkpoint whose config says something else fails loudly
# instead of loading under a scheme nobody chose. RadixArk/Qwen3.8-Flash-Next-NVFP4 is ModelOpt
# NVFP4; the mixed NVFP4/FP8 checkpoints are compressed-tensors. The value comes from
# server.quantization, and the empty default is what makes clearing that key mean "no flag, let
# vLLM read the method out of config.json" rather than "fall back to a method chosen here".
DEFAULT_VLLM_QUANTIZATION = ""
# VLLM_GDN_DECODE_KERNEL. The image defaults to the CUDA kernel, which the FP8 linear-attention
# projections of the mixed checkpoints stall on under concurrency; those need "triton". Empty
# leaves the variable unset.
DEFAULT_VLLM_GDN_DECODE_KERNEL = ""
# Opt-in only. The rendered Kaggle setup script applies a second, runtime
# competition-mode veto before it exposes any profiler configuration to vLLM.
DEFAULT_VLLM_PROFILE = False
# server.cache_diagnostics: Lord's prefix-cache tracer, which hooks every cache lookup and
# eviction. True keeps it on in test runs (never in a competition rerun); false leaves it off
# for a clean throughput run, as matijaludvig's DHL/vcintel notebooks do.
DEFAULT_VLLM_CACHE_DIAGNOSTICS = True
# server.draft_vocab: a JSON file of token-id ranges, relative to ARC3-Inference/, that the
# e975732 rc3 runtime limits MTP draft proposals to. Empty keeps the full-vocabulary draft head.
DEFAULT_VLLM_DRAFT_VOCAB = ""
# kaggle.draft_model_source: a Kaggle model ref whose MTP head drafts for the target in place of
# the target checkpoint's own MTP layer. Empty keeps the target's head. The setup script mounts
# it next to the target and loads only its mtp.* tensors, lm_head and embed_tokens.
DEFAULT_VLLM_DRAFT_MODEL_SOURCE = ""
# server.draft_method: "mtp" drafts with an MTP head, the target's own or draft_model_source's.
# "dflash" serves draft_model_source as a DFlash/DFlash2 block drafter (vLLM method "dflash"),
# which reads the target's hidden states after its configured layers. The e975732 rc3 runtime
# cannot feed one to Qwen4Exp, so the setup script patches the unpacked runtime for it
# (patch_runtime_for_dflash). The target still verifies every draft, so outputs are unchanged.
DEFAULT_VLLM_DRAFT_METHOD = "mtp"
VLLM_DRAFT_METHODS = ("mtp", "dflash")
# kaggle.ple_fp8_model_source: a Kaggle model ref whose FP8 PLE table, stored with one global
# weight_scale, replaces the target's BF16 table. dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple
# carries RadixArk's table next to Intel's unchanged W4A16 body. The pinned host copy then takes
# about 48 instead of 95.4 GiB of the 175 GiB limit. Empty serves the target's own table. The
# setup script mounts it and links the target's files and the source's PLE files into one directory.
DEFAULT_VLLM_PLE_FP8_MODEL_SOURCE = ""
PROJECT_DIR = Path(__file__).resolve().parents[2]

# The 25 official ARC-AGI-3 games. The first 16 are the original Kaggle duck
# validation harness order; the remaining 9 complete the official tag set.
DUCK_HARNESS_PUBLIC_GAME_IDS: tuple[str, ...] = (
    "tn36-ef4dde99",
    "lf52-271a04aa",
    "cn04-2fe56bfb",
    "bp35-0a0ad940",
    "wa30-ee6fef47",
    "lp85-305b61c3",
    "r11l-495a7899",
    "tu93-0768757b",
    "sp80-589a99af",
    "m0r0-492f87ba",
    "vc33-5430563c",
    "ar25-0c556536",
    "ka59-38d34dbb",
    "sc25-635fd71a",
    "sk48-d8078629",
    "dc22-fdcac232",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ls20-9607627b",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "su15-1944f8ab",
    "tr87-cd924810",
)


@dataclass(frozen=True)
class DuckKaggleVllmConfig:
    """Kaggle-side vLLM/model configuration declared by ``HarnessSolver``."""

    runtime_dataset_source: str = DEFAULT_VLLM_RUNTIME_DATASET_SOURCE
    model_source: str = DEFAULT_QWEN_MODEL_SOURCE
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME
    vllm_port: int = DEFAULT_VLLM_PORT
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    kv_cache_gib: int = DEFAULT_VLLM_KV_CACHE_GIB
    mtp_tokens: int = DEFAULT_VLLM_MTP_TOKENS
    max_num_seqs: int = DEFAULT_VLLM_MAX_NUM_SEQS
    dtype: str = DEFAULT_VLLM_DTYPE
    gpu_memory_utilization: str = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    kv_cache_dtype: str = DEFAULT_VLLM_KV_CACHE_DTYPE
    indexer_kv_dtype: str = DEFAULT_VLLM_INDEXER_KV_DTYPE
    mamba_ssm_cache_dtype: str = DEFAULT_VLLM_MAMBA_SSM_CACHE_DTYPE
    max_num_batched_tokens: int = DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS
    enable_prefix_caching: bool = DEFAULT_VLLM_ENABLE_PREFIX_CACHING
    prefix_match_unit: int = DEFAULT_VLLM_PREFIX_MATCH_UNIT
    prefix_cache_retention_interval: int = DEFAULT_VLLM_PREFIX_CACHE_RETENTION_INTERVAL
    quantization: str = DEFAULT_VLLM_QUANTIZATION
    moe_backend: str = ""
    gdn_decode_kernel: str = DEFAULT_VLLM_GDN_DECODE_KERNEL
    profile: bool = DEFAULT_VLLM_PROFILE
    cache_diagnostics: bool = DEFAULT_VLLM_CACHE_DIAGNOSTICS
    draft_vocab: str = DEFAULT_VLLM_DRAFT_VOCAB
    prefetch_threads: int = 8
    draft_model_source: str = DEFAULT_VLLM_DRAFT_MODEL_SOURCE
    draft_method: str = DEFAULT_VLLM_DRAFT_METHOD
    ple_fp8_model_source: str = DEFAULT_VLLM_PLE_FP8_MODEL_SOURCE


def resolve_draft_vocab(path: str) -> str:
    """Return the draft-vocabulary path relative to ARC3-Inference/, or an empty string when unset.

    The deploy bundle snapshots ARC3-Inference/, so the setup script only needs this relative
    path to find the file under the mounted bundle. Embedding the JSON itself pushed the setup
    command past the 128 KiB limit Linux puts on one argument to /bin/sh. The file must lie
    inside ARC3-Inference/ and hold a non-empty ``token_ranges`` list, so a broken file fails at
    deploy time instead of at server start.
    """
    if not path:
        return ""
    vocab_path = (PROJECT_DIR / path).resolve()
    token_ranges = json.loads(vocab_path.read_text(encoding="utf-8")).get("token_ranges")
    if not isinstance(token_ranges, list) or not token_ranges:
        raise ValueError(f"{vocab_path} has no token_ranges.")

    return vocab_path.relative_to(PROJECT_DIR).as_posix()


def duck_kaggle_dataset_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    cfg = config or DuckKaggleVllmConfig()
    return [cfg.runtime_dataset_source]


def duck_kaggle_model_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    cfg = config or DuckKaggleVllmConfig()
    return [ref for ref in (cfg.model_source, cfg.draft_model_source, cfg.ple_fp8_model_source) if ref]


def duck_kaggle_setup_command(config: DuckKaggleVllmConfig | None = None) -> str:
    cfg = config or DuckKaggleVllmConfig()
    if cfg.prefetch_threads < 1:
        raise ValueError("vLLM prefetch_threads must be >= 1")
    runtime_owner, runtime_slug = _split_dataset_source(
        cfg.runtime_dataset_source,
        option_name="runtime_dataset_source",
    )
    model_source_parts = _split_model_source(cfg.model_source)
    draft_model_source_parts = (
        _split_model_source(cfg.draft_model_source) if cfg.draft_model_source else None
    )
    ple_fp8_model_source_parts = (
        _split_model_source(cfg.ple_fp8_model_source) if cfg.ple_fp8_model_source else None
    )
    if cfg.draft_method not in VLLM_DRAFT_METHODS:
        raise ValueError(f"vLLM draft_method must be one of {VLLM_DRAFT_METHODS}, got {cfg.draft_method!r}")
    if cfg.draft_method == "dflash" and not (cfg.draft_model_source and cfg.mtp_tokens > 0 and not cfg.draft_vocab):
        raise ValueError(
            "draft_method 'dflash' needs a draft_model_source, mtp_tokens > 0 and no draft_vocab "
            "(the draft vocabulary restricts the MTP head only)"
        )
    configured_context_window = os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW")
    analyzer_context_window = int(configured_context_window or cfg.max_model_len)
    prompt_window = int(cfg.max_model_len) - VLLM_OUTPUT_RESERVE_TOKENS
    if configured_context_window and analyzer_context_window > prompt_window:
        print(
            f"kaggle-duck: LOCAL_ANALYZER_CONTEXT_WINDOW {analyzer_context_window} clamped to "
            f"{prompt_window} (max_model_len {cfg.max_model_len} minus {VLLM_OUTPUT_RESERVE_TOKENS} for output)"
        )
    analyzer_context_window = min(analyzer_context_window, prompt_window)
    # Base URL / model are pinned to the local vLLM server below, so reject a
    # provider that disagrees (e.g. openrouter) — it would drop vLLM-only payload
    # field (chat_template_kwargs) against a vLLM endpoint.
    analyzer_provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")
    multimodal_upscale = os.environ.get("MULTIMODAL_UPSCALE", "4")
    if normalize_provider(analyzer_provider) != "vllm":
        raise ValueError(
            f"kaggle-duck runs a local vLLM server, so LOCAL_ANALYZER_PROVIDER must be "
            f"vLLM/OpenAI-compatible, got {analyzer_provider!r}."
        )
    replacements = {
        "__RUNTIME_OWNER__": repr(runtime_owner),
        "__RUNTIME_SLUG__": repr(runtime_slug),
        "__MODEL_SOURCE_PARTS__": repr(model_source_parts),
        "__DRAFT_MODEL_SOURCE_PARTS__": repr(draft_model_source_parts),
        "__VLLM_DRAFT_METHOD__": repr(cfg.draft_method),
        "__PLE_FP8_MODEL_SOURCE_PARTS__": repr(ple_fp8_model_source_parts),
        "__SERVED_MODEL_NAME__": repr(cfg.served_model_name),
        "__VLLM_PORT__": repr(int(cfg.vllm_port)),
        "__VLLM_MAX_MODEL_LEN__": repr(int(cfg.max_model_len)),
        # The launcher's Makefile exports LOCAL_ANALYZER_CONTEXT_WINDOW from
        # JSON shared.context_window (or analyzer.context_window). Embed it
        # here so the agent's prompt budget on Kaggle is the JSON value, not
        # vllm's max-model-len. Falls back to max_model_len if unset.
        "__ANALYZER_CONTEXT_WINDOW__": repr(analyzer_context_window),
        "__LOCAL_ANALYZER_TARGET_CONTEXT__": repr(os.environ.get("LOCAL_ANALYZER_TARGET_CONTEXT", "0")),
        "__VLLM_KV_CACHE_MEMORY_BYTES__": repr(int(cfg.kv_cache_gib) * 1024**3),
        "__VLLM_MTP_TOKENS__": repr(int(cfg.mtp_tokens)),
        "__VLLM_MAX_NUM_SEQS__": repr(int(cfg.max_num_seqs)),
        # Remaining JSON-driven analyzer/multimodal config: the launcher's
        # Makefile exports each from inference.json; embed the launcher value
        # so the rendered setup_env on Kaggle reflects JSON edits. Fallback
        # equals the historical hardcoded literal so direct kaggle.py callers
        # outside Make are unaffected.
        "__LOCAL_ANALYZER_PROVIDER__": repr(analyzer_provider),
        "__LOCAL_ANALYZER_APP_NAME__": repr(os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Kaggle Harness")),
        "__LOCAL_ANALYZER_MAX_OUTPUT__": repr(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", "0")),
        "__LOCAL_ANALYZER_TOOL_STEPS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_STEPS", "0")),
        "__LOCAL_ANALYZER_TOOL_TIMEOUT__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_TIMEOUT", "30")),
        "__LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", "1024")),
        "__LOCAL_ANALYZER_YIELD_SECONDS__": repr(os.environ.get("LOCAL_ANALYZER_YIELD_SECONDS", "120")),
        "__LOCAL_ANALYZER_TEMPERATURE__": repr(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "1.0")),
        "__LOCAL_ANALYZER_TOP_P__": repr(os.environ.get("LOCAL_ANALYZER_TOP_P", "0.95")),
        "__LOCAL_ANALYZER_TOP_K__": repr(os.environ.get("LOCAL_ANALYZER_TOP_K", "20")),
        "__LOCAL_ANALYZER_ENABLE_THINKING__": repr(os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING", "1")),
        "__LOCAL_ANALYZER_REASONING_EFFORT__": repr(os.environ.get("LOCAL_ANALYZER_REASONING_EFFORT", "xhigh")),
        "__MULTIMODAL_CONTEXT__": repr(os.environ.get("MULTIMODAL_CONTEXT", "current_grid")),
        "__MULTIMODAL_UPSCALE__": repr(multimodal_upscale),
        "__VLLM_IMAGE_MAX_PIXELS__": repr((ARC_FRAME_SIDE * int(multimodal_upscale)) ** 2),
        "__VLLM_TENSOR_PARALLEL_SIZE__": repr(int(cfg.tensor_parallel_size)),
        "__VLLM_GPU_MEMORY_UTILIZATION__": repr(str(cfg.gpu_memory_utilization)),
        "__VLLM_DTYPE__": repr(str(cfg.dtype)),
        "__VLLM_KV_CACHE_DTYPE__": repr(str(cfg.kv_cache_dtype)),
        "__VLLM_INDEXER_KV_DTYPE__": repr(str(cfg.indexer_kv_dtype)),
        "__VLLM_MAMBA_SSM_CACHE_DTYPE__": repr(str(cfg.mamba_ssm_cache_dtype)),
        "__VLLM_MAX_NUM_BATCHED_TOKENS__": repr(int(cfg.max_num_batched_tokens)),
        "__VLLM_ENABLE_PREFIX_CACHING__": repr(bool(cfg.enable_prefix_caching)),
        "__VLLM_PREFIX_MATCH_UNIT__": repr(int(cfg.prefix_match_unit)),
        "__VLLM_PREFIX_CACHE_RETENTION_INTERVAL__": repr(int(cfg.prefix_cache_retention_interval)),
        "__WATCHDOG_SCRIPT__": repr(_DUCK_VLLM_WATCHDOG_SCRIPT),
        "__CACHE_DIAGNOSTICS_SOURCE__": repr(Path(__file__).with_name("cache_diagnostics.py").read_text(encoding="utf-8")),
        "__VLLM_QUANTIZATION__": repr(str(cfg.quantization)),
        "__VLLM_MOE_BACKEND__": repr(str(cfg.moe_backend)),
        "__VLLM_GDN_DECODE_KERNEL__": repr(str(cfg.gdn_decode_kernel)),
        "__VLLM_CUDA_LAUNCH_BLOCKING__": repr(os.environ.get("KAGGLE_VLLM_CUDA_LAUNCH_BLOCKING", "false").strip().lower() in {"1", "true", "yes", "on"}),
        "__VLLM_PROFILE_REQUESTED__": repr(bool(cfg.profile)),
        "__VLLM_CACHE_DIAGNOSTICS_REQUESTED__": repr(bool(cfg.cache_diagnostics)),
        "__VLLM_DRAFT_VOCAB__": repr(resolve_draft_vocab(cfg.draft_vocab)),
        "__VLLM_PREFETCH_THREADS__": repr(int(cfg.prefetch_threads)),
        "__PROJECT_DIR_NAME__": repr(PROJECT_DIR.name),
    }
    script = _DUCK_VLLM_SETUP_SCRIPT
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return f"\"$PYTHON\" - <<'PYSETUP'\n{script}\nPYSETUP"


def duck_kaggle_teardown_command() -> str:
    return f"\"$PYTHON\" - <<'PYTEARDOWN'\n{_DUCK_VLLM_TEARDOWN_SCRIPT}\nPYTEARDOWN"


def _split_dataset_source(value: str, *, option_name: str) -> tuple[str, str]:
    parts = str(value or "").strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"{option_name} must be a Kaggle dataset ref in owner/slug format."
        )
    return parts[0], parts[1]


def _split_model_source(value: str) -> tuple[str, str, str, str, str]:
    """Split a Kaggle model ref into its owner/model/framework/instance/version parts.

    The mount path is resolved on Kaggle rather than here, because the layout
    varies between kernel images.
    """
    parts = str(value or "").strip().split("/")
    if len(parts) != 5 or not all(parts):
        raise ValueError(
            "model_source must be a Kaggle model ref in "
            "owner/model/framework/instance/version format."
        )
    owner, model, framework, instance, version = parts

    return owner, model, framework, instance, version


_DUCK_VLLM_WATCHDOG_SCRIPT = r"""'''Keep the vLLM OpenAI server alive for the whole harness run.

Started by the notebook setup step with the path of a JSON config holding the server command and
the server log, pid and ready-marker paths. The server is spawned as a child and awaited with a
blocking wait, so a dead server is noticed without polling. Every exit is logged with its code,
uptime and the tail of the server log. The server is restarted after a short pause, which lets the
engine process that the dying API server kills release the GPU and the port. It is not restarted
when it died before the setup step marked it ready, so a broken launch fails the setup step, and
not more than MAX_RESTARTS times, so a crash that comes back deterministically ends in a readable
"giving up" line instead of a restart loop for the rest of the run.

Each server start runs the GPU shard prefetcher from that start's log offset, so a restart never
mistakes the previous load's completed progress bar for its own. Prefetch reports are kept
separately for the initial start and every restart. The prefetch runs in a separate process,
with bounded termination waits so a blocked NFS read cannot hold up server restarts.
'''
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def is_test_run() -> bool:
    '''Return true only for an explicit noncompetition test run.'''

    return (
        os.environ.get('TAAF_RUN_AS_SUBMISSION', '').strip().lower() == '0'
        and os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'', '0', 'false'}
    )


MAX_RESTARTS = 5
RESTART_DELAY_SECONDS = 10
HOST_MEMORY_SAMPLE_INTERVAL_SECONDS = 15
GPU_SHARD_PREFETCH_WINDOW_BYTES = 8 * 1024**3
GPU_SHARD_PREFETCH_BLOCK_BYTES = 16 * 1024**2
GPU_SHARD_PREFETCH_POLL_SECONDS = 0.5
GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS = 2
PLE_SHARD_NAME_PREFIXES = ('ple-', 'model-ple-', 'model-plefp8-')
GPU_SHARD_PREFETCH_ABORT = threading.Event()
WATCHDOG_EXIT_DEFERRED = False
WATCHDOG_EXIT_REQUESTED = False

config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
GPU_SHARD_PREFETCH_THREADS = int(config.get('prefetch_threads', 8))
if GPU_SHARD_PREFETCH_THREADS < 1:
    raise ValueError('vLLM prefetch_threads must be >= 1')
server_command = config['command']
server_log = Path(config['server_log'])
server_pid = Path(config['server_pid'])
server_ready = Path(config['server_ready'])


def read_integer_file(path: str) -> int | None:
    '''Read a cgroup counter, returning None for unreadable or nonnumeric values.'''
    try:
        return int(Path(path).read_text(encoding='ascii').strip())
    except (OSError, ValueError):
        return None


def read_host_memory() -> dict[str, int | None]:
    '''Report available RAM, including the cgroup limit used by vLLM's loader.'''
    host_available_bytes = None
    for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        if line.startswith('MemAvailable:'):
            host_available_bytes = int(line.split()[1]) * 1024
            break
    if host_available_bytes is None:
        raise ValueError('MemAvailable is missing from /proc/meminfo')

    cgroup_limit_bytes = read_integer_file('/sys/fs/cgroup/memory.max')
    if cgroup_limit_bytes is not None:
        cgroup_usage_bytes = read_integer_file('/sys/fs/cgroup/memory.current')
    else:
        cgroup_limit_bytes = read_integer_file('/sys/fs/cgroup/memory/memory.limit_in_bytes')
        if cgroup_limit_bytes is not None and cgroup_limit_bytes >= 1 << 62:
            cgroup_limit_bytes = None
        cgroup_usage_bytes = (
            read_integer_file('/sys/fs/cgroup/memory/memory.usage_in_bytes')
            if cgroup_limit_bytes is not None else None
        )

    cgroup_available_bytes = None
    if cgroup_limit_bytes is not None:
        cgroup_available_bytes = (
            cgroup_limit_bytes
            if cgroup_usage_bytes is None else max(0, cgroup_limit_bytes - cgroup_usage_bytes)
        )
    available_bytes = (
        min(host_available_bytes, cgroup_available_bytes)
        if cgroup_available_bytes is not None else host_available_bytes
    )

    return {
        'available_bytes': available_bytes,
        'host_available_bytes': host_available_bytes,
        'cgroup_available_bytes': cgroup_available_bytes,
        'cgroup_limit_bytes': cgroup_limit_bytes,
        'cgroup_usage_bytes': cgroup_usage_bytes,
    }


# memory.stat fields kept with each sample: cgroup v2 names, then the v1 hierarchy totals.
CGROUP_MEMORY_STAT_FIELDS = (
    'anon', 'file', 'shmem', 'file_mapped', 'unevictable', 'kernel', 'sock',
    'total_rss', 'total_cache', 'total_shmem', 'total_mapped_file', 'total_unevictable',
)
# The largest processes by anonymous memory listed in each sample, next to the per-group sums.
HOST_MEMORY_TOP_PROCESSES = 8


def read_cgroup_memory_stat(cgroup_root: Path = Path('/sys/fs/cgroup')) -> dict[str, int]:
    '''Split the cgroup's usage into process memory, page cache and shared memory.'''
    for stat_path in (cgroup_root / 'memory.stat', cgroup_root / 'memory' / 'memory.stat'):
        try:
            lines = stat_path.read_text(encoding='ascii').splitlines()
        except OSError:
            continue
        fields = (line.split(' ', 1) for line in lines)
        return {key: int(value) for key, value in fields if key in CGROUP_MEMORY_STAT_FIELDS and value.isdigit()}

    return {}


def classify_process(cmdline: list[str], cwd: str) -> str:
    '''Name the role of a process: a vLLM process by its title, the harness by its notebook kernel.

    vLLM retitles its worker processes VLLM::EngineCore and VLLM::Worker; the API server keeps its
    `vllm ... serve` command line. Every game runs as a thread of the notebook kernel
    (colab_kernel_launcher on Kaggle), so the harness is one process. A python tool call runs in a
    fresh interpreter whose working directory is an rgb_python_tool_* temporary directory.
    '''
    joined = ' '.join(cmdline)
    if cmdline and cmdline[0].startswith('VLLM::'):
        return 'vllm_' + cmdline[0][len('VLLM::'):].split()[0].lower()
    if Path(cwd).name.startswith('rgb_python_tool_'):
        return 'sandbox'
    if 'vllm-watchdog.py' in joined:
        return 'prefetch' if '--prefetch' in cmdline else 'watchdog'
    if 'vllm' in joined:
        return 'vllm_other'
    if 'ipykernel' in joined or 'kernel_launcher' in joined:
        return 'harness'

    return 'other'


def read_process_memory(proc_root: Path = Path('/proc')) -> tuple[dict[str, dict[str, int]], list[dict]]:
    '''Sum resident memory per process role and list the largest processes.

    RssAnon is the process's own memory. RssFile and RssShmem count pages that other processes
    may map too, so their group sums can exceed what the cgroup holds. Kernel threads have no Rss
    lines and are skipped, as is a process that exits while it is read.
    '''
    groups: dict[str, dict[str, int]] = {}
    processes = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            rss = {}
            for line in (proc / 'status').read_text(encoding='utf-8', errors='replace').splitlines():
                key, _, value = line.partition(':')
                if key in ('RssAnon', 'RssFile', 'RssShmem'):
                    rss[key] = int(value.split()[0]) * 1024
            cmdline = [part for part in (proc / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0') if part]
        except (OSError, ValueError, IndexError):
            continue
        if not rss:
            continue
        try:
            cwd = os.readlink(proc / 'cwd')
        except OSError:
            cwd = ''
        group = classify_process(cmdline, cwd)
        totals = groups.setdefault(group, {'count': 0, 'anon_bytes': 0, 'file_bytes': 0, 'shmem_bytes': 0})
        totals['count'] += 1
        totals['anon_bytes'] += rss.get('RssAnon', 0)
        totals['file_bytes'] += rss.get('RssFile', 0)
        totals['shmem_bytes'] += rss.get('RssShmem', 0)
        processes.append({
            'pid': int(proc.name), 'group': group, 'anon_bytes': rss.get('RssAnon', 0),
            'cmd': ' '.join(cmdline).replace('\n', ' ')[:80],
        })
    processes.sort(key=lambda process: process['anon_bytes'], reverse=True)

    return groups, processes[:HOST_MEMORY_TOP_PROCESSES]


def read_shm_used_bytes(path: str = '/dev/shm') -> int | None:
    '''Bytes in /dev/shm, a tmpfs whose files count as shared memory and are never reclaimed.'''
    try:
        stats = os.statvfs(path)
    except OSError:
        return None

    return (stats.f_blocks - stats.f_bfree) * stats.f_frsize


def read_memory_breakdown() -> dict:
    '''What holds the RAM: the cgroup's memory.stat, per-role process memory and /dev/shm.'''
    groups, top_processes = read_process_memory()

    return {
        'cgroup_memory_stat': read_cgroup_memory_stat(),
        'process_groups': groups,
        'top_processes': top_processes,
        'dev_shm_used_bytes': read_shm_used_bytes(),
    }


def log_host_memory() -> None:
    '''Sample host RAM and what holds it throughout server startup, inference, and restarts.'''
    output_path = Path(config['host_memory_log'])
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as output:
            while True:
                record = {'ts': round(time.time(), 3), **read_host_memory(), **read_memory_breakdown()}
                output.write(json.dumps(record) + '\n')
                output.flush()
                time.sleep(HOST_MEMORY_SAMPLE_INTERVAL_SECONDS)
    except (OSError, ValueError) as error:
        log_event(f'host memory monitor stopped: {error!r}')


def log_event(message: str) -> None:
    print(f'{datetime.now().isoformat(timespec="seconds")} watchdog: {message}', flush=True)


def read_server_log_tail(lines: int = 40) -> str:
    if not server_log.exists():
        return ''

    return '\n'.join(server_log.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def build_natural_sort_key(path: Path) -> list:
    '''Split a file name into text and integer parts, the key vLLM's safetensors loader sorts by.'''
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', path.name)]


def list_loader_shards(model_path: Path) -> list[Path]:
    '''Return the checkpoint shards of the index in the order vLLM's safetensors loader reads them.'''
    index = json.loads((model_path / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    shards = [model_path / name for name in set(index['weight_map'].values())]

    return sorted(shards, key=build_natural_sort_key)


def read_file_into_page_cache(
    path: Path, stop_event: threading.Event, start: int, end: int
) -> dict:
    '''Read a disjoint file range into page cache in large blocks, until its end or a stop.

    The read stops early when its stop_event or GPU_SHARD_PREFETCH_ABORT is set. Both are checked
    between blocks, so a stop takes effect only after an in-flight read returns. Returns the
    monotonic start and end of the read and whether it was stopped early.
    '''
    started_at = time.monotonic()
    buffer = memoryview(bytearray(GPU_SHARD_PREFETCH_BLOCK_BYTES))
    with path.open('rb', buffering=0) as handle:
        handle.seek(start)
        remaining = end - start
        while remaining and not (stop_event.is_set() or GPU_SHARD_PREFETCH_ABORT.is_set()):
            count = handle.readinto(buffer[:min(len(buffer), remaining)])
            if not count:
                raise OSError(f'Checkpoint file shrank while prefetching {path}')
            remaining -= count

    return {'started_at': started_at, 'finished_at': time.monotonic(), 'stopped': remaining > 0}


def drop_file_from_page_cache(path: Path) -> None:
    '''Ask the kernel to evict a file's clean cached pages. The kernel treats this as a hint.

    An eviction that fails only leaves the pages in the cache, so it is reported and the prefetch
    carries on with a window that is smaller in practice than it is on paper.
    '''
    try:
        file_descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(file_descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        print(f'GPU shard prefetch could not evict {path.name}: {error!r}', flush=True)


class LoaderProgressFollower:
    '''Follow the completed-shard count of the target load's progress bar in the vLLM server log.

    The bar prints `| N/<shard_count> [` after the N-th shard, one update per line. The MTP draft's
    second bar has a different total, so it never matches. The unterminated tail of each read is
    kept and completed by the next one, so a line cut by a read is still counted. The count is -1
    until the bar's first line, which the loader prints once the pinned PLE table is allocated.
    '''

    def __init__(self, log_handle, shard_count: int) -> None:
        self._log_handle = log_handle
        self._pattern = re.compile(rf'\| (\d+)/{shard_count} \[')
        self._unterminated_text = ''
        self.loaded_shard_count = -1

    def update_loaded_shard_count(self) -> int:
        '''Read the new log text and return the highest completed count seen so far.'''
        complete_text, _, self._unterminated_text = (self._unterminated_text + self._log_handle.read()).rpartition('\n')
        for match in self._pattern.finditer(complete_text):
            self.loaded_shard_count = max(self.loaded_shard_count, int(match.group(1)))

        return self.loaded_shard_count


class GpuShardPrefetcher:
    '''Keep the GPU shards a bounded window ahead of vLLM's single-stream loader in the page cache.

    The loader reads the GPU shards through one NFS stream at 44-153 MiB/s, while the PLE shards
    already arrive at 615-923 MiB/s through its multi-threaded copy, so only the GPU shards are
    prefetched. vLLM's own prefetch (v264, v265) read ahead with no bound, and after the pinned PLE
    table the page cache holds about 20 GiB, so pages far ahead were evicted before the loader got
    there. As in SGLang's lookahead reader, threads read disjoint ranges of a shard. The shared
    pool limits total readers, and only the current shard and its immediate successor may be
    staged, within the byte window. Intel's model-ple-* shards are excluded too: their loader
    already copies them in parallel. Nothing is read before the first progress line, so the window never
    competes with the PLE allocation. From then on shards are submitted in load order while the
    bytes submitted and not yet passed by the loader fit the window, and the shard being loaded is
    always submitted.

    A shard the loader has passed gets its read stopped, and once that read has ended its pages
    are evicted. Evicting while the read still runs would let it refill the cache behind the loader.
    Pages read twice (the prefetch, then the loader) sit on the active LRU list and would
    otherwise push the window's unread pages out first.

    The loop returns once every GPU shard is passed and evicted, and writes a JSON report with the
    per-shard prefetch and loader times, relative to the loader's first progress line. If the bar
    never appears, nothing is read and no report is written. Shards the loader crossed inside one
    poll carry the timestamp of the next printed update, so complete_before_loader_reached is an
    upper bound and the report counts them. The log offset isolates each server start's progress.
    '''

    def __init__(self, shards: list[Path], server_log: Path, report_path: Path, log_offset: int = 0) -> None:
        self._shards = shards
        self._server_log = server_log
        self._report_path = report_path
        self._log_offset = log_offset
        self._gpu_shard_indices = [
            index for index, shard in enumerate(shards) if not shard.name.startswith(PLE_SHARD_NAME_PREFIXES)
        ]
        self._shard_sizes = {index: shards[index].stat().st_size for index in self._gpu_shard_indices}
        self._futures: dict[int, list[concurrent.futures.Future]] = {}
        self._stop_events: dict[int, threading.Event] = {}
        self._progress_reached_at: dict[int, float] = {}
        self._submitted_position = 0
        self._passed_position = 0
        self._dropped_position = 0

    def run(self) -> None:
        '''Follow the loader until every GPU shard is passed and evicted, then write the report.

        GPU_SHARD_PREFETCH_ABORT ends the loop as well as the reads, and a pass that ends early
        writes no report.
        '''
        while not self._server_log.exists():
            if GPU_SHARD_PREFETCH_ABORT.is_set():
                return
            time.sleep(GPU_SHARD_PREFETCH_POLL_SECONDS)
        with (
            self._server_log.open(encoding='utf-8', errors='replace') as log_handle,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=GPU_SHARD_PREFETCH_THREADS, thread_name_prefix='gpu-shard-prefetch'
            ) as executor,
        ):
            log_handle.seek(self._log_offset)
            follower = LoaderProgressFollower(log_handle, len(self._shards))
            while self._dropped_position < len(self._gpu_shard_indices) and not GPU_SHARD_PREFETCH_ABORT.is_set():
                loaded_shard_count = follower.update_loaded_shard_count()
                if loaded_shard_count >= 0:
                    self._record_progress(loaded_shard_count)
                    self._stop_passed_shards(loaded_shard_count)
                    self._drop_passed_shards()
                    self._submit_window(executor, loaded_shard_count)
                time.sleep(GPU_SHARD_PREFETCH_POLL_SECONDS)
        if self._dropped_position == len(self._gpu_shard_indices) and self._progress_reached_at:
            self._write_report()

    def _record_progress(self, loaded_shard_count: int) -> None:
        '''Stamp the first time each completed count was seen, including counts a poll skipped.

        tqdm drops the updates that fall inside its minimum interval, so counts it never printed
        carry the timestamp of the next printed one, which is later than the real crossing. Those
        shards show equal reached and passed times in the report.
        '''
        now = time.monotonic()
        for count in range(len(self._progress_reached_at), loaded_shard_count + 1):
            self._progress_reached_at[count] = now

    def _stop_passed_shards(self, loaded_shard_count: int) -> None:
        '''Stop the reads of GPU shards the loader has finished and cancel those still queued.'''
        while (
            self._passed_position < len(self._gpu_shard_indices)
            and self._gpu_shard_indices[self._passed_position] < loaded_shard_count
        ):
            shard_index = self._gpu_shard_indices[self._passed_position]
            if shard_index in self._futures:
                self._stop_events[shard_index].set()
                for future in self._futures[shard_index]:
                    future.cancel()
            self._passed_position += 1

    def _drop_passed_shards(self) -> None:
        '''Evict passed GPU shards in order, each only after its read has ended.'''
        while self._dropped_position < self._passed_position:
            shard_index = self._gpu_shard_indices[self._dropped_position]
            if any(not future.done() for future in self._futures.get(shard_index, [])):
                return
            drop_file_from_page_cache(self._shards[shard_index])
            self._dropped_position += 1

    def _submit_window(self, executor: concurrent.futures.ThreadPoolExecutor, loaded_shard_count: int) -> None:
        '''Submit the next GPU shards while the bytes still held in the page cache fit the window.

        The window counts from the first shard not yet evicted, not from the loader, so shards the
        loader has passed while their read winds down keep their bytes in the budget.
        '''
        self._submitted_position = max(self._submitted_position, self._passed_position)
        ahead_bytes = sum(
            self._shard_sizes[index]
            for index in self._gpu_shard_indices[self._dropped_position:self._submitted_position]
        )
        while (
            self._submitted_position < len(self._gpu_shard_indices)
            and self._gpu_shard_indices[self._submitted_position] <= loaded_shard_count + 1
        ) and (
            self._submitted_position == self._passed_position
            or ahead_bytes + self._shard_sizes[self._gpu_shard_indices[self._submitted_position]]
            <= GPU_SHARD_PREFETCH_WINDOW_BYTES
        ):
            shard_index = self._gpu_shard_indices[self._submitted_position]
            self._stop_events[shard_index] = threading.Event()
            size = self._shard_sizes[shard_index]
            blocks = (size + GPU_SHARD_PREFETCH_BLOCK_BYTES - 1) // GPU_SHARD_PREFETCH_BLOCK_BYTES
            readers = min(GPU_SHARD_PREFETCH_THREADS, max(1, blocks))
            self._futures[shard_index] = [
                executor.submit(
                    read_file_into_page_cache, self._shards[shard_index], self._stop_events[shard_index],
                    (blocks * reader // readers) * GPU_SHARD_PREFETCH_BLOCK_BYTES,
                    min(size, (blocks * (reader + 1) // readers) * GPU_SHARD_PREFETCH_BLOCK_BYTES),
                )
                for reader in range(readers)
            ]
            ahead_bytes += self._shard_sizes[shard_index]
            self._submitted_position += 1

    def _describe_shard(self, shard_index: int, loader_started_at: float) -> dict:
        '''Return one shard's report row, with times in seconds after the loader's first progress line.'''
        row = {
            'name': self._shards[shard_index].name,
            'gib': round(self._shard_sizes[shard_index] / 1024**3, 3),
            'loader_reached': round(self._progress_reached_at[shard_index] - loader_started_at, 1),
            'loader_passed': round(self._progress_reached_at[shard_index + 1] - loader_started_at, 1),
        }
        futures = self._futures.get(shard_index, [])
        completed = [future for future in futures if not future.cancelled()]
        if not completed:
            row['prefetch'] = 'not started'
        elif errors := [future.exception() for future in completed if future.exception() is not None]:
            row['prefetch'] = f'failed: {errors[0]!r}'
        else:
            reads = [future.result() for future in completed]
            stopped = len(completed) < len(futures) or any(read['stopped'] for read in reads)
            row['prefetch'] = 'stopped' if stopped else 'complete'
            row['prefetch_started'] = round(min(read['started_at'] for read in reads) - loader_started_at, 1)
            row['prefetch_finished'] = round(max(read['finished_at'] for read in reads) - loader_started_at, 1)

        return row

    def _write_report(self) -> None:
        '''Write the window settings and one row per GPU shard to the report file.'''
        loader_started_at = self._progress_reached_at[0]
        rows = [self._describe_shard(index, loader_started_at) for index in self._gpu_shard_indices]
        report = {
            'threads': GPU_SHARD_PREFETCH_THREADS,
            'reader_mode': 'parallel_ranges',
            'lookahead_shards': 1,
            'window_gib': GPU_SHARD_PREFETCH_WINDOW_BYTES / 1024**3,
            'gpu_shards': len(self._gpu_shard_indices),
            'gpu_shard_gib': round(sum(self._shard_sizes.values()) / 1024**3, 2),
            'other_shards': len(self._shards) - len(self._gpu_shard_indices),
            'complete_before_loader_reached': sum(
                row['prefetch'] == 'complete' and row['prefetch_finished'] <= row['loader_reached'] for row in rows
            ),
            'shards_crossed_within_one_poll': sum(row['loader_reached'] == row['loader_passed'] for row in rows),
            'shards': rows,
        }
        self._report_path.write_text(json.dumps(report, indent=1), encoding='utf-8')


def prefetch_gpu_shards(log_offset: int, report_path: Path) -> None:
    '''Build the GPU shard prefetcher for the mounted model and run it.

    It runs in a separate process, so a failure while reading the checkpoint index
    only ends the prefetch and leaves the server launch alone.
    '''
    GpuShardPrefetcher(
        list_loader_shards(Path(config['model_path'])), server_log, report_path, log_offset
    ).run()


def start_gpu_shard_prefetch(log_offset: int, report_path: Path) -> subprocess.Popen | None:
    '''Launch an isolated prefetch process, leaving the server running if spawning fails.'''
    try:

        return subprocess.Popen([
            sys.executable, __file__, sys.argv[1], '--prefetch', str(log_offset), str(report_path)
        ])
    except OSError as error:
        log_event(f'could not start GPU shard prefetch: {error!r}')

        return None


def stop_gpu_shard_prefetch(prefetch_process: subprocess.Popen) -> None:
    '''Terminate prefetch without waiting indefinitely for a blocked filesystem read.'''
    if prefetch_process.poll() is not None:

        return
    prefetch_process.terminate()
    try:
        prefetch_process.wait(timeout=GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS)

        return
    except subprocess.TimeoutExpired:
        log_event(f'GPU shard prefetch did not stop within {GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS}s; sending SIGKILL')
    prefetch_process.kill()
    try:
        prefetch_process.wait(timeout=GPU_SHARD_PREFETCH_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        log_event('GPU shard prefetch is still blocked after SIGKILL; continuing without waiting for it')


@contextmanager
def defer_watchdog_exit():
    '''Finish registering or stopping prefetch before honoring a watchdog exit request.'''
    global WATCHDOG_EXIT_DEFERRED
    WATCHDOG_EXIT_DEFERRED = True
    try:
        yield
    finally:
        WATCHDOG_EXIT_DEFERRED = False
        if WATCHDOG_EXIT_REQUESTED:
            raise SystemExit(0)


def exit_watchdog(signal_number: int, frame) -> None:
    '''Request shutdown once, deferring it while prefetch ownership changes or cleanup runs.'''
    global WATCHDOG_EXIT_REQUESTED
    if WATCHDOG_EXIT_REQUESTED:

        return
    WATCHDOG_EXIT_REQUESTED = True
    if not WATCHDOG_EXIT_DEFERRED:
        raise SystemExit(0)


def run_watchdog() -> None:
    '''Spawn the server, wait for it to exit, log the exit and start it again while that is allowed.'''
    restart_count = 0
    prefetch_process = None
    while True:
        with server_log.open('a' if restart_count else 'w', encoding='utf-8') as server_log_handle:
            if restart_count:
                server_log_handle.write(
                    f'\n==== watchdog restart {restart_count} at {datetime.now().isoformat(timespec="seconds")} ====\n'
                )
                server_log_handle.flush()
            log_offset = server_log_handle.tell()
            started_at = time.monotonic()
            server_process = subprocess.Popen(
                server_command, stdout=server_log_handle, stderr=subprocess.STDOUT, text=True
            )
            server_pid.write_text(str(server_process.pid), encoding='utf-8')
            log_event(f'vLLM server started (pid {server_process.pid}, restart {restart_count})')
            report_path = Path(config['shard_prefetch_report'])
            if restart_count:
                report_path = report_path.with_name(f'{report_path.stem}-restart-{restart_count}{report_path.suffix}')
            try:
                with defer_watchdog_exit():
                    if prefetch_process is None or prefetch_process.poll() is not None:
                        prefetch_process = start_gpu_shard_prefetch(log_offset, report_path)
                    else:
                        log_event('previous GPU shard prefetch is still exiting; skipping another prefetch')
                exit_code = server_process.wait()
                log_event(f'vLLM server exited with code {exit_code} after {time.monotonic() - started_at:.0f}s')
            finally:
                with defer_watchdog_exit():
                    if prefetch_process is not None:
                        stop_gpu_shard_prefetch(prefetch_process)
        log_event('last server log lines:\n' + read_server_log_tail())
        if not server_ready.exists():
            log_event('server died before it was ready; not restarting')
            sys.exit(1)
        if restart_count >= MAX_RESTARTS:
            log_event(f'giving up after {MAX_RESTARTS} restarts; the server stays down')
            sys.exit(1)
        restart_count += 1
        log_event(f'restarting vLLM server in {RESTART_DELAY_SECONDS}s (restart {restart_count} of {MAX_RESTARTS})')
        time.sleep(RESTART_DELAY_SECONDS)


if len(sys.argv) > 2 and sys.argv[2] == '--prefetch':
    prefetch_gpu_shards(int(sys.argv[3]), Path(sys.argv[4]))
    sys.exit(0)
signal.signal(signal.SIGTERM, exit_watchdog)
if is_test_run():
    threading.Thread(target=log_host_memory, name='host-memory-monitor', daemon=True).start()
run_watchdog()
"""

_DUCK_VLLM_SETUP_SCRIPT = r"""import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def is_test_run() -> bool:
    '''Return true only for an explicit noncompetition test run.'''

    return (
        os.environ.get('TAAF_RUN_AS_SUBMISSION', '').strip().lower() == '0'
        and os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower() in {'', '0', 'false'}
    )


RUNTIME_OWNER = __RUNTIME_OWNER__
RUNTIME_SLUG = __RUNTIME_SLUG__
MODEL_SOURCE_PARTS = __MODEL_SOURCE_PARTS__
# A separate checkpoint whose MTP head drafts for the target, or None for the target's own head.
DRAFT_MODEL_SOURCE_PARTS = __DRAFT_MODEL_SOURCE_PARTS__
# 'mtp' assembles that checkpoint's MTP head; 'dflash' serves it as a DFlash block drafter.
VLLM_DRAFT_METHOD = __VLLM_DRAFT_METHOD__
# A checkpoint whose FP8 PLE table replaces the target's, or None for the target's own table.
PLE_FP8_MODEL_SOURCE_PARTS = __PLE_FP8_MODEL_SOURCE_PARTS__
SERVED_MODEL_NAME = __SERVED_MODEL_NAME__
VLLM_HOST = '127.0.0.1'
VLLM_PORT = __VLLM_PORT__
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_SERVER_URL = f'http://{VLLM_HOST}:{VLLM_PORT}'
VLLM_MAX_MODEL_LEN = __VLLM_MAX_MODEL_LEN__
ANALYZER_CONTEXT_WINDOW = __ANALYZER_CONTEXT_WINDOW__
VLLM_TENSOR_PARALLEL_SIZE = __VLLM_TENSOR_PARALLEL_SIZE__
VLLM_GPU_MEMORY_UTILIZATION = __VLLM_GPU_MEMORY_UTILIZATION__
VLLM_KV_CACHE_MEMORY_BYTES = __VLLM_KV_CACHE_MEMORY_BYTES__
VLLM_MTP_TOKENS = __VLLM_MTP_TOKENS__
VLLM_CUDAGRAPH_CAPTURE_STEP = 8
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'
VLLM_SERVER_READY = WORKING_DIR / 'vllm-openai-server.ready'
VLLM_WATCHDOG_SCRIPT = WORKING_DIR / 'vllm-watchdog.py'
VLLM_WATCHDOG_CONFIG = WORKING_DIR / 'vllm-watchdog.json'
VLLM_WATCHDOG_LOG = WORKING_DIR / 'vllm-watchdog.log'
VLLM_WATCHDOG_PID = WORKING_DIR / 'vllm-watchdog.pid'
VLLM_SHARD_PREFETCH_REPORT = WORKING_DIR / 'vllm-shard-prefetch.json'
WATCHDOG_SCRIPT_TEXT = __WATCHDOG_SCRIPT__
CACHE_DIAGNOSTICS_SOURCE = __CACHE_DIAGNOSTICS_SOURCE__
VLLM_QUANTIZATION = __VLLM_QUANTIZATION__
VLLM_MOE_BACKEND = __VLLM_MOE_BACKEND__
VLLM_GDN_DECODE_KERNEL = __VLLM_GDN_DECODE_KERNEL__
VLLM_CUDA_LAUNCH_BLOCKING = __VLLM_CUDA_LAUNCH_BLOCKING__
VLLM_PROFILE_REQUESTED = __VLLM_PROFILE_REQUESTED__
VLLM_CACHE_DIAGNOSTICS_REQUESTED = __VLLM_CACHE_DIAGNOSTICS_REQUESTED__
VLLM_PROFILE_DIR = WORKING_DIR / 'vllm-profile'
# The image-layer runtime unpacks to ~19.5 GB. /kaggle/working is a 20 GB volume, while
# `df` inside a kernel on the RTX PRO 6000 host reported ~1.1 TB free on the root overlay,
# so the unpack goes to /tmp and the teardown removes it.
IMAGE_RUNTIME_ROOT = Path(os.environ.get('TAAF_IMAGE_RUNTIME_ROOT', '/tmp/vllm-image-runtime'))
IMAGE_RUNTIME_CACHE_ROOT = IMAGE_RUNTIME_ROOT.with_name(IMAGE_RUNTIME_ROOT.name + '-cache')
# The server.draft_vocab path relative to the bundled ARC3-Inference/ snapshot; empty keeps the
# full-vocabulary MTP draft head.
VLLM_DRAFT_VOCAB = __VLLM_DRAFT_VOCAB__
VLLM_PREFETCH_THREADS = __VLLM_PREFETCH_THREADS__
VLLM_DRAFT_VOCAB_FILE = (
    Path(os.environ['TAAF_KAGGLE_BUNDLE_DIR']) / 'src' / __PROJECT_DIR_NAME__ / VLLM_DRAFT_VOCAB
    if VLLM_DRAFT_VOCAB
    else None
)
if VLLM_DRAFT_VOCAB_FILE is not None and not VLLM_DRAFT_VOCAB_FILE.is_file():
    raise FileNotFoundError(f'Missing draft vocabulary: {VLLM_DRAFT_VOCAB_FILE}')
RUNTIME_MANIFEST_FILE_NAME = 'runtime-manifest.json'
VLLM_MAX_NUM_BATCHED_TOKENS = __VLLM_MAX_NUM_BATCHED_TOKENS__
VLLM_DTYPE = __VLLM_DTYPE__
VLLM_KV_CACHE_DTYPE = __VLLM_KV_CACHE_DTYPE__
VLLM_INDEXER_KV_DTYPE = __VLLM_INDEXER_KV_DTYPE__
VLLM_MAMBA_SSM_CACHE_DTYPE = __VLLM_MAMBA_SSM_CACHE_DTYPE__
VLLM_ENABLE_PREFIX_CACHING = __VLLM_ENABLE_PREFIX_CACHING__
VLLM_PREFIX_MATCH_UNIT = __VLLM_PREFIX_MATCH_UNIT__
VLLM_PREFIX_CACHE_RETENTION_INTERVAL = __VLLM_PREFIX_CACHE_RETENTION_INTERVAL__
# Pixel area of the agent's grid image, embedded from MULTIMODAL_UPSCALE at build time.
VLLM_IMAGE_MAX_PIXELS = __VLLM_IMAGE_MAX_PIXELS__
FINE_PREFIX_CACHE_RUNTIME_IDENTITY = {
    'image': 'vllm/vllm-openai:nightly-e9757321527ca1ecd514c07c1418dd2c53da3d19',
    'amd64_manifest_digest': 'sha256:e0eee5c5506bea9bfe350f7d99b07dc49e37d42647a128c2a57ff184551fba10',
    'vllm_version': '0.29.1rc1.dev573+ge97573215',
}
FINE_PREFIX_CACHE_PATCH_IDENTITY = {
    'artifact': 'arc3_vllm_main_e975732_arc3_overlay.tar.blob',
    'applier': 'apply_vllm_main_e975732_arc3.py',
    'overlay_sha256': 'e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb',
    'vllm_commit': 'e9757321527ca1ecd514c07c1418dd2c53da3d19',
}
FINE_PREFIX_CACHE_REQUIRED_OVERLAYS = {
    'vllm/v1/core/kv_cache_coordinator.py': {
        'stock_sha256': '7fa4065d19021e77b3424d24e1b088c530ed70b90c5ed9bc52cf4b7581d5fc48',
        'sha256': '7494e9f76d4a6b0bb469e80816e4052fb52171bc63b34799c3b9588092bc243e',
    },
    'vllm/v1/core/kv_cache_utils.py': {
        'stock_sha256': 'd359221ef91a94f7e570220f083979297c297227c1ad1b8ccdbec76cc24565b7',
        'sha256': '816aea33eaa28a15b8d739be602a877aa0be1a8ec83853cdc92412dcdd137b4e',
    },
    'vllm/v1/core/single_type_kv_cache_manager.py': {
        'stock_sha256': '6ebd4ddb5210d500b52b300dfaf462e2e92f58a4ee0e15691a860549dc086d68',
        'sha256': '97a0a4518c757cbe48e0d13d7a1890f48bc0db777004a3d66a8eb9ea73ba2405',
    },
    'vllm/v1/worker/gpu/model_states/mamba_hybrid.py': {
        'stock_sha256': '6ea89adcb39762b533f0c5b765a66e6f942cbde90208a5d13a758ef86a4ac6f1',
        'sha256': '8cd9eac0da31add7685ba7f23dfc2dcd7d6d422470fa98281aa73f90d5a8408d',
    },
    'vllm/v1/worker/gpu_model_runner.py': {
        'stock_sha256': '95dbd301feae580913938cafe4151da7af2cbd5d3f37fa645fab62a8cd1b54ef',
        'sha256': 'a2d3709963d463256b8b62ecb5264bce10e1b133d552ab10afdd31f68b42d1b2',
    },
    'vllm/v1/worker/mamba_utils.py': {
        'stock_sha256': '28c8a9bb07a56e3612e245ccc48cd7f29d85181dbe5e4cf8940c6bcd086df856',
        'sha256': 'b5e30c8ab2343340f478c9c8ea94b9369649c74044cd42ea11b246dc84e11243',
    },
}
# Embedded from KAGGLE_VLLM_MAX_NUM_SEQS at build time, which the launcher fills from
# server.max_num_seqs, the way the KV pin and the MTP tokens are filled.
VLLM_MAX_NUM_SEQS = __VLLM_MAX_NUM_SEQS__
# The deadline covers layer unpack, host PLE allocation, both target and MTP weight loads,
# engine init and CUDA graph capture. The mixed checkpoint's 167.7 GiB safetensors set is larger
# than the free RAM left after allocating the BF16 PLE table, so an NFS cold start can take about
# 30 minutes. Keep enough margin for Kaggle storage variance while still detecting a stuck start.
VLLM_SERVER_READY_TIMEOUT_SECONDS = 3600

GPU_NAME_PATTERNS = {'rtx-pro-6000': ('rtx pro 6000',), 'h100': ('h100',), 'l4': ('l4',)}


def taaf_kaggle_input_paths() -> dict[str, Path]:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError('TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.')
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    for dataset_path in (Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug):
        if dataset_path.exists():
            return dataset_path
    return Path('/kaggle/input') / slug


def resolve_kaggle_model_path(parts: tuple[str, str, str, str, str]) -> Path:
    owner, model, framework, instance, version = parts
    candidates = []
    for framework_name in dict.fromkeys((framework.lower(), framework)):
        candidates.append(Path('/kaggle/input/models') / owner / model / framework_name / instance / version)
        candidates.append(Path('/kaggle/input') / model / framework_name / instance / version)
    for model_path in candidates:
        if model_path.exists():
            return model_path

    return candidates[0]


def describe_kaggle_input_tree(max_depth: int = 5) -> str:
    root = Path('/kaggle/input')
    if not root.exists():
        return 'No /kaggle/input mount is present.'
    lines = []
    frontier = [root]
    for _ in range(max_depth):
        children = []
        for directory in frontier:
            for entry in sorted(directory.iterdir()):
                if entry.is_dir():
                    lines.append('  ' + str(entry))
                    children.append(entry)
        frontier = children

    return 'Directories under /kaggle/input:\n' + '\n'.join(lines)


RUNTIME_DATASET = resolve_kaggle_dataset_path(RUNTIME_OWNER, RUNTIME_SLUG)
MOUNTED_MODEL_PATH = resolve_kaggle_model_path(MODEL_SOURCE_PARTS)
# Symlinks to the target's files with its PLE table swapped for the FP8 one; copies no weights.
PLE_FP8_TARGET_DIR = Path('/tmp/vllm-target-fp8-ple')
MODEL_PATH = PLE_FP8_TARGET_DIR if PLE_FP8_MODEL_SOURCE_PARTS else MOUNTED_MODEL_PATH
# Symlinks into the mounted draft checkpoint plus a filtered index; assembling it copies no weights.
DRAFT_MODEL_DIR = Path('/tmp/vllm-mtp-draft')


def assert_expected_cuda_gpu() -> None:
    if not Path('/kaggle/input').exists():
        return
    assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    assert result.returncode == 0, f'nvidia-smi failed: {result.stderr.strip()}'
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert gpu_names, 'nvidia-smi did not report any CUDA GPUs.'
    expected_gpu_type = os.getenv('KAGGLE_GPU_TYPE', 'rtx-pro-6000').strip().lower()
    expected_count = os.getenv('KAGGLE_GPU_COUNT', '1')
    if expected_count.isdigit():
        assert len(gpu_names) == int(expected_count), f'Expected {expected_count} CUDA GPU(s), found {gpu_names}'
    patterns = GPU_NAME_PATTERNS.get(expected_gpu_type, (expected_gpu_type.replace('-', ' '),))
    mismatched = [name for name in gpu_names if not any(pattern in name.lower() for pattern in patterns)]
    assert not mismatched, f'Expected GPU type {expected_gpu_type!r}, found {gpu_names}'
    print(f'CUDA GPU check passed for {expected_gpu_type} x{expected_count}: {gpu_names}', flush=True)


def prepend_path_entries(existing: str, entries: list[Path]) -> str:
    joined = os.pathsep.join([str(entry) for entry in entries] + ([existing] if existing else []))

    return joined


def read_runtime_manifest() -> dict:
    manifest_path = RUNTIME_DATASET / RUNTIME_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing image runtime manifest: {manifest_path}')

    return json.loads(manifest_path.read_text(encoding='utf-8'))


def build_runtime_python_paths(manifest: dict) -> list[Path]:
    '''PYTHONPATH entries; the cutlass DSL .pth of the image is not processed on PYTHONPATH, so add its target.'''
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']

    return [dist_packages / 'nvidia_cutlass_dsl' / 'dsl_packages', dist_packages]


def resolve_runtime_cuda_home(manifest: dict) -> Path:
    cuda_home = IMAGE_RUNTIME_ROOT / Path(manifest['nvcc']).parent.parent

    return cuda_home


def build_runtime_library_paths(manifest: dict) -> list[Path]:
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    nvidia_libs = sorted(dist_packages.glob('nvidia/*/lib')) + sorted(dist_packages.glob('nvidia/cu13/*/lib'))

    return [
        resolve_runtime_cuda_home(manifest) / 'targets' / 'x86_64-linux' / 'lib',
        dist_packages / 'torch' / 'lib',
        *nvidia_libs,
        Path('/usr/local/nvidia/lib64'),
    ]


def build_image_runtime_env(manifest: dict) -> dict[str, str]:
    '''Build the vLLM server environment for the extracted image runtime.

    With the b12x MoE backend the b12x micro path is disabled, so every batch runs through
    the dynamic kernel. vLLM marks padding rows with expert id -1, and the b12x 1.3.0 micro
    kernel (up to 6 tokens at top-10) indexes weights with that id, which crashes with an
    illegal memory access. The dynamic kernel skips such rows.

    A configured draft vocabulary is read in place from the mounted bundle and handed to the
    e975732 rc3 runtime through VLLM_QWEN4_EXP_DRAFT_VOCAB; without one the variable is removed.

    VLLM_ARC3_GDN_RECOVERSSM=1 makes the GDN layers verify MTP drafts without per-draft state
    blocks and commit the accepted tokens after sampling, which frees 9 KV blocks per request.
    It is only a default, so an A/B launcher that sets the variable to 0 keeps the stock path.
    '''
    env = os.environ.copy()
    cuda_home = resolve_runtime_cuda_home(manifest)
    cache_paths = {
        'VLLM_CACHE_ROOT': IMAGE_RUNTIME_CACHE_ROOT / 'vllm',
        'TRITON_CACHE_DIR': IMAGE_RUNTIME_CACHE_ROOT / 'triton',
        'FLASHINFER_WORKSPACE_BASE': IMAGE_RUNTIME_CACHE_ROOT / 'flashinfer',
        'B12X_COMPILE_CACHE_DIR': IMAGE_RUNTIME_CACHE_ROOT / 'b12x' / 'compile',
        'TMPDIR': IMAGE_RUNTIME_CACHE_ROOT / 'tmp',
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env.pop('PYTORCH_ALLOC_CONF', None)
    # These legacy variables are either deprecated or no longer consumed by the pinned runtime.
    # Their supported equivalents are passed by build_vllm_server_command.
    env.pop('VLLM_PLE_CPU_OFFLOAD', None)
    env.pop('VLLM_PLE_OFFLOAD_READY_TIMEOUT', None)
    env.pop('VLLM_PREFIX_CACHE_RETENTION_INTERVAL', None)
    env['PYTHONPATH'] = prepend_path_entries(env.get('PYTHONPATH', ''), build_runtime_python_paths(manifest))
    env['PATH'] = prepend_path_entries(env.get('PATH', ''), [cuda_home / 'bin', IMAGE_RUNTIME_ROOT / 'usr' / 'local' / 'bin'])
    env['LD_LIBRARY_PATH'] = prepend_path_entries(
        env.get('LD_LIBRARY_PATH', ''), [path for path in build_runtime_library_paths(manifest) if path.is_dir()]
    )
    env.update({key: str(path) for key, path in cache_paths.items()})
    env.update(
        {
            'USE_TF': '0',
            'TRANSFORMERS_NO_TF': '1',
            'TRANSFORMERS_NO_TORCHVISION': '1',
            'VLLM_NO_USAGE_STATS': '1',
            'CUDA_HOME': str(cuda_home),
            'CUDACXX': str(cuda_home / 'bin' / 'nvcc'),
            'TORCH_CUDA_ARCH_LIST': '12.0',
            'VLLM_ENABLE_CUDA_COMPATIBILITY': '0',
            'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
            'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False',
            'HF_HUB_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    env.setdefault('VLLM_ARC3_GDN_RECOVERSSM', '1')
    if VLLM_GDN_DECODE_KERNEL:
        env['VLLM_GDN_DECODE_KERNEL'] = VLLM_GDN_DECODE_KERNEL
    if VLLM_MOE_BACKEND == 'b12x':
        env['B12X_MICRO_DYNAMIC_CUTOVER_PAIRS'] = '0'
    if VLLM_CUDA_LAUNCH_BLOCKING:
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    env.pop('VLLM_QWEN4_EXP_DRAFT_VOCAB', None)
    if VLLM_DRAFT_VOCAB_FILE is not None:
        env['VLLM_QWEN4_EXP_DRAFT_VOCAB'] = str(VLLM_DRAFT_VOCAB_FILE)

    return env


def verify_layer_blob(layer: dict) -> Path:
    '''Kaggle already guarantees dataset integrity, so only the recorded size is checked here.'''
    blob = RUNTIME_DATASET / layer['file']
    if not blob.exists():
        raise FileNotFoundError(f'Missing runtime layer blob: {blob}')
    if blob.stat().st_size != int(layer['size']):
        raise RuntimeError(f'Runtime layer {layer["index"]} does not match its manifest size: {blob.name}')

    return blob


def clear_opaque_whiteout_targets(layer: dict) -> None:
    for whiteout in layer.get('whiteouts', []):
        if whiteout['kind'] == 'opaque' and whiteout['target'] not in ('', '.'):
            shutil.rmtree(IMAGE_RUNTIME_ROOT / whiteout['target'], ignore_errors=True)


def remove_whiteout_markers_and_targets(layer: dict) -> None:
    for whiteout in layer.get('whiteouts', []):
        (IMAGE_RUNTIME_ROOT / whiteout['marker']).unlink(missing_ok=True)
        if whiteout['kind'] == 'file':
            target = IMAGE_RUNTIME_ROOT / whiteout['target']
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)


def extract_layer(blob: Path) -> None:
    '''pigz, when the image ships it, moves reading, checksumming and writing off the single inflate thread
    (roughly 1.2-1.5x over gzip); the .pyc files stay in so the server does not recompile vLLM and torch.'''
    decompressor = ['--use-compress-program=pigz'] if shutil.which('pigz') else ['-z']
    subprocess.run(
        ['tar', '-xf', str(blob), '-C', str(IMAGE_RUNTIME_ROOT), '--no-same-owner', *decompressor],
        check=True,
    )


def apply_runtime_patch(manifest: dict) -> None:
    patch = manifest.get('patch')
    if not patch:
        print('Image runtime ships no patch', flush=True)

        return
    overlay_files = patch.get('overlay_files')
    if not isinstance(overlay_files, list) or not overlay_files:
        raise RuntimeError('Runtime patch requires a non-empty overlay_files manifest')
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    result = subprocess.run(
        [sys.executable, str(RUNTIME_DATASET / patch['applier']), str(dist_packages)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f'Runtime patch failed:\n{result.stderr.strip()}')
    print(f'Runtime patch {patch["artifact"]}: {result.stdout.strip()}', flush=True)


INC_PLE_ANCHOR = (
    '        if not isinstance(quant_config, Fp8Config):\n'
    '            raise NotImplementedError(\n'
    '                "Qwen4Exp PLE embedding does not support quantization config "'
)


def route_inc_ple_to_unquantized(manifest: dict) -> None:
    '''Let the runtime's PLE embedding load an Intel AutoRound (INC) checkpoint.

    The PLE embedding accepts only the NVFP4, ModelOpt and FP8 configs of the mixed
    checkpoints and raises for INC, whose PLE table is unquantized BF16, as in
    dfranzen/intel-qwen3.8-flash-next-w4a16-autoround. INCConfig is routed to the
    unquantized PLE method, as matijaludvig/arc3-vcintel-25x1 does; other quantization
    configs take the same path as before. A runtime without the anchor fails only when
    the checkpoint is AutoRound, which would otherwise fail at weight load.
    '''
    ple_py = (IMAGE_RUNTIME_ROOT / manifest['dist_packages'] / 'vllm' / 'models'
              / 'qwen4_exp' / 'nvidia' / 'ngram_embedding.py')
    source = ple_py.read_text(encoding='utf-8') if ple_py.is_file() else ''
    if source.count(INC_PLE_ANCHOR) == 1:
        ple_py.write_text(source.replace(
            INC_PLE_ANCHOR,
            '        if type(quant_config).__name__ == "INCConfig":\n'
            '            return Qwen4ExpPLEUnquantizedEmbeddingMethod()\n' + INC_PLE_ANCHOR,
        ), encoding='utf-8')
        print(f'Routed INC to the unquantized PLE embedding in {ple_py}', flush=True)
        return
    config = json.loads((MODEL_PATH / 'config.json').read_text(encoding='utf-8'))
    method = str((config.get('quantization_config') or {}).get('quant_method', '')).lower()
    if method in {'auto-round', 'auto_round', 'inc'}:
        raise RuntimeError(
            f'{method} checkpoint needs the INC PLE route, but its anchor appears '
            f'{source.count(INC_PLE_ANCHOR)} times in {ple_py}'
        )


# Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight in the e975732 runtime.
EXACT_PINNED_PLE_ANCHOR = (
    '        return torch.empty(\n'
    '            num_embeddings,\n'
    '            embedding_dim,\n'
    '            dtype=dtype,\n'
    '            device="cpu",\n'
    '            pin_memory=True,\n'
    '        )\n'
)
EXACT_PINNED_PLE_CALL = '        return _allocate_exact_pinned_cpu_weight(num_embeddings, embedding_dim, dtype)\n'
EXACT_PINNED_PLE_HELPERS = '''

def _clear_cuda_runtime_error() -> bool:
    # A failed cudaHostRegister stays the runtime's last error, which torch's next kernel launch
    # check would raise. Read it back through the libcudart this process already loaded.
    import ctypes

    with open("/proc/self/maps", encoding="utf-8") as maps:
        paths = sorted({line.split()[-1] for line in maps if "libcudart.so" in line})
    for path in paths:
        try:
            ctypes.CDLL(path).cudaGetLastError()
        except (OSError, AttributeError):
            continue
        return True
    return False


def _allocate_exact_pinned_cpu_weight(num_embeddings, embedding_dim, dtype):
    # pin_memory=True takes torch's CachingHostAllocator, which rounds every block up to the next
    # power of two, so the 95.4 GiB BF16 table held 128 GiB of host RAM. This page-locks a CPU
    # tensor of exactly the table's size instead: Mapped keeps the UVA device view and Portable
    # makes it valid in every context. The table lives as long as the worker and is never
    # unregistered.
    weight = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu")
    nbytes = weight.numel() * weight.element_size()
    if nbytes:
        cuda_host_register_portable_mapped = 3
        error = torch.cuda.cudart().cudaHostRegister(
            weight.data_ptr(), nbytes, cuda_host_register_portable_mapped
        )
        if int(error) == 0:
            logger.info("PLE table pinned at its exact size: %.2f GiB", nbytes / 2**30)
            return weight
        if not _clear_cuda_runtime_error():
            raise RuntimeError(f"cudaHostRegister of the PLE table failed ({error}) and the error could not be cleared")
        logger.warning(
            "cudaHostRegister of the %.2f GiB PLE table failed (%s); using the power-of-two pinned allocator",
            nbytes / 2**30,
            error,
        )
        del weight
    return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu", pin_memory=True)
'''


def pin_ple_table_at_exact_size(manifest: dict) -> None:
    '''Pin the host PLE table at its own size rather than the next power of two.

    torch's pinned allocator rounded the 95.4 GiB BF16 table up to 128 GiB (porting-advaith v3:
    130 GiB of VLLM::Worker shmem with the embedding), a third of it unused. The runtime's
    allocation is replaced by a plain CPU tensor registered with cudaHostRegister, which saves
    about 33 GiB of host RAM for BF16 and 16 GiB for FP8. The table's values and the UVA lookup
    are unchanged. A runtime without the anchor keeps its allocator, and so does a registration
    that fails at load, with a warning in the server log.
    '''
    ple_py = (IMAGE_RUNTIME_ROOT / manifest['dist_packages'] / 'vllm' / 'models'
              / 'qwen4_exp' / 'nvidia' / 'ngram_embedding.py')
    source = ple_py.read_text(encoding='utf-8') if ple_py.is_file() else ''
    if source.count(EXACT_PINNED_PLE_ANCHOR) != 1:
        print(
            f'PLE table keeps the power-of-two pinned allocator: its anchor appears '
            f'{source.count(EXACT_PINNED_PLE_ANCHOR)} times in {ple_py}',
            flush=True,
        )
        return
    ple_py.write_text(
        source.replace(EXACT_PINNED_PLE_ANCHOR, EXACT_PINNED_PLE_CALL) + EXACT_PINNED_PLE_HELPERS, encoding='utf-8'
    )
    print(f'PLE table pinned at its exact size in {ple_py}', flush=True)


# Where the e975732 rc3 runtime picks the FP8 PLE method from config.ple_embedding_dtype. Under
# INCConfig that check comes before the unquantized route above, which the FP8 table needs.
FP8_PLE_RUNTIME_MARKERS = (
    '        if embedding_dtype == "float8_e4m3fn":\n'
    '            return Qwen4ExpPLEFp8EmbeddingMethod()\n',
    'getattr(config, "ple_embedding_dtype", None)',
)


def check_runtime_reads_fp8_ple(manifest: dict) -> None:
    '''Fail before the load when the runtime would not read the swapped PLE table as FP8.

    A runtime that ignores ple_embedding_dtype keeps the BF16 method under INC and copies the FP8
    codes into a BF16 table without their global scale, which loads and serves wrong embeddings.
    '''
    if PLE_FP8_MODEL_SOURCE_PARTS is None:
        return
    ple_py = (IMAGE_RUNTIME_ROOT / manifest['dist_packages'] / 'vllm' / 'models'
              / 'qwen4_exp' / 'nvidia' / 'ngram_embedding.py')
    source = ple_py.read_text(encoding='utf-8') if ple_py.is_file() else ''
    missing = [marker for marker in FP8_PLE_RUNTIME_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(f'{ple_py} does not select the FP8 PLE method from ple_embedding_dtype: {missing}')


# The tensors of the PLE n-gram table: its row shards and, in an FP8 table, the global weight_scale.
PLE_TABLE_TENSOR_MARKER = '.ple_embedding.ngram_embedding.'


def assemble_ple_fp8_target() -> None:
    '''Link the target with its BF16 PLE table replaced by the FP8 checkpoint's into PLE_FP8_TARGET_DIR.

    dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple holds RadixArk's FP8 table (ten
    model-plefp8-* files, 128 row shards and one global weight_scale) next to Intel's W4A16 body,
    while the target keeps its table in BF16 files of its own (dfranzen's Intel copy: 22
    model-ple-* files). The directory links every file of the mounted target except its index,
    config.json and the files of its table, then the files of the source's table. The index maps
    the target's other tensors as before and the table to the source, and config.json is the
    target's with text_config.ple_embedding_dtype set, which makes the runtime allocate the pinned
    table in FP8. Only the table changes, so every GPU tensor is the target's. A source table that
    lacks a row shard of the target's, has no scale, or shares a file with other tensors fails here.
    '''
    if PLE_FP8_MODEL_SOURCE_PARTS is None:
        return
    source = resolve_kaggle_model_path(PLE_FP8_MODEL_SOURCE_PARTS)
    index_name = 'model.safetensors.index.json'
    if not (source / index_name).is_file():
        raise FileNotFoundError(f'FP8 PLE checkpoint not mounted at {source}\n' + describe_kaggle_input_tree())
    target_map = json.loads((MOUNTED_MODEL_PATH / index_name).read_text(encoding='utf-8'))['weight_map']
    source_map = json.loads((source / index_name).read_text(encoding='utf-8'))['weight_map']
    source_dtype = json.loads((source / 'config.json').read_text(encoding='utf-8'))['text_config'].get('ple_embedding_dtype')
    if source_dtype != 'float8_e4m3fn':
        raise RuntimeError(f'{source} declares ple_embedding_dtype {source_dtype!r}, not float8_e4m3fn')

    def table_files(weight_map: dict, checkpoint: Path) -> set[str]:
        files = {shard for name, shard in weight_map.items() if PLE_TABLE_TENSOR_MARKER in name}
        shared = sorted({shard for name, shard in weight_map.items() if PLE_TABLE_TENSOR_MARKER not in name} & files)
        if shared:
            raise RuntimeError(f'PLE files of {checkpoint} also hold other tensors: {shared}')

        return files

    target_table = {name for name in target_map if PLE_TABLE_TENSOR_MARKER in name}
    source_table = {name: shard for name, shard in source_map.items() if PLE_TABLE_TENSOR_MARKER in name}
    missing = sorted(target_table - set(source_table))
    scales = [name for name in source_table if name.endswith('.weight_scale')]
    unexpected = sorted(set(source_table) - target_table - set(scales))
    if missing or not scales or unexpected:
        raise RuntimeError(
            f'{source} cannot replace the PLE table of {MOUNTED_MODEL_PATH}: missing {missing[:3]} '
            f'({len(missing)}), scales {scales}, unexpected {unexpected[:3]}'
        )
    target_files = table_files(target_map, MOUNTED_MODEL_PATH)
    source_files = table_files(source_map, source)
    shutil.rmtree(PLE_FP8_TARGET_DIR, ignore_errors=True)
    PLE_FP8_TARGET_DIR.mkdir(parents=True)
    for entry in MOUNTED_MODEL_PATH.iterdir():
        if entry.is_file() and entry.name not in target_files | {index_name, 'config.json'}:
            (PLE_FP8_TARGET_DIR / entry.name).symlink_to(entry)
    for shard in sorted(source_files):
        if (PLE_FP8_TARGET_DIR / shard).exists():
            raise RuntimeError(f'{shard} of {source} clashes with a file of {MOUNTED_MODEL_PATH}')
        (PLE_FP8_TARGET_DIR / shard).symlink_to(source / shard)
    weight_map = {name: shard for name, shard in target_map.items() if name not in target_table}
    weight_map.update(source_table)
    (PLE_FP8_TARGET_DIR / index_name).write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}), encoding='utf-8')
    config = json.loads((MOUNTED_MODEL_PATH / 'config.json').read_text(encoding='utf-8'))
    config['text_config']['ple_embedding_dtype'] = source_dtype
    (PLE_FP8_TARGET_DIR / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(
        f'FP8 PLE target at {PLE_FP8_TARGET_DIR}: {MOUNTED_MODEL_PATH} without {len(target_files)} BF16 '
        f'PLE files, plus {len(source_table)} FP8 PLE tensors from {sorted(source_files)} of {source}',
        flush=True,
    )


# What the MTP drafter loads from a draft checkpoint besides its mtp.* layer: the LM head and the
# input embedding. The rest of that checkpoint is a second copy of the target's layers and stays unread.
DRAFT_SHARED_TENSORS = ('lm_head.weight', 'model.language_model.embed_tokens.weight')


def assemble_mtp_draft() -> Path | None:
    '''Build a draft model directory that holds only the draft checkpoint's MTP head.

    matijaludvig/arc3-vcintel-lorddraft-25x1 drafts for the Intel W4A16 AutoRound target with the
    NVFP4 MTP head of lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8 instead of Intel's BF16 MTP
    layer. Each of the draft passes then reads a quarter of the expert bytes, and the KV pool gets
    what the BF16 layer held. The directory links the checkpoint's config and tokenizer files and
    the shards that hold the kept tensors, and its index lists only those tensors, so vLLM loads
    the head without the target layers. A checkpoint with no MTP head fails here instead of
    starting a server without a drafter.
    '''
    if DRAFT_MODEL_SOURCE_PARTS is None:
        return None
    source = resolve_kaggle_model_path(DRAFT_MODEL_SOURCE_PARTS)
    index_path = source / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise FileNotFoundError(f'MTP draft checkpoint not mounted at {source}\n' + describe_kaggle_input_tree())
    weight_map = json.loads(index_path.read_text(encoding='utf-8'))['weight_map']
    kept = {name: shard for name, shard in weight_map.items()
            if name.startswith('mtp.') or name in DRAFT_SHARED_TENSORS}
    mtp_count = sum(name.startswith('mtp.') for name in kept)
    missing = [name for name in DRAFT_SHARED_TENSORS if name not in kept]
    if not mtp_count or missing:
        raise RuntimeError(f'{source} has {mtp_count} mtp.* tensors and lacks {missing}, so it cannot draft')
    shards = sorted(set(kept.values()))
    shutil.rmtree(DRAFT_MODEL_DIR, ignore_errors=True)
    DRAFT_MODEL_DIR.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.is_file() and not entry.name.endswith('.safetensors') and entry.name != index_path.name:
            (DRAFT_MODEL_DIR / entry.name).symlink_to(entry)
    for shard in shards:
        (DRAFT_MODEL_DIR / shard).symlink_to(source / shard)
    (DRAFT_MODEL_DIR / index_path.name).write_text(
        json.dumps({'metadata': {}, 'weight_map': kept}), encoding='utf-8')
    print(f'MTP draft at {DRAFT_MODEL_DIR}: {mtp_count} mtp.* tensors from {shards} of {source}', flush=True)

    return DRAFT_MODEL_DIR


# DFlash on the e975732 rc3 runtime: (relative path, anchor, replacement). Each anchor must occur
# exactly once, otherwise the setup fails before the load instead of serving without a drafter.
DFLASH_RUNTIME_PATCHES = (
    # Qwen4Exp's config check admits only its MTP and n-gram proposers.
    (
        'vllm/model_executor/models/config.py',
        '            "ngram_gpu",\n        }:\n',
        '            "ngram_gpu",\n            "dflash",\n        }:\n',
    ),
    # The V2 runner feeds the drafter the target's MTP buffer whenever the target has the getter,
    # but Qwen4Exp allocates that buffer only for method "mtp" and returns None otherwise.
    (
        'vllm/v1/worker/gpu/model_runner.py',
        '                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]\n',
        '                if pre_hc_hidden_states is not None:\n'
        '                    spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]\n',
    ),
    (
        'vllm/v1/worker/gpu/model_runner.py',
        '                spec_hidden_states = pre_hc_hidden_states[: draft_hidden_states.size(0)]\n',
        '                if pre_hc_hidden_states is not None:\n'
        '                    spec_hidden_states = pre_hc_hidden_states[: draft_hidden_states.size(0)]\n',
    ),
    # Qwen4ExpModel returns the target's hidden states after the requested layers. vLLM's aux id L
    # is the output of layer L - 1. Lord's layers defer their hyper-connection combine to the next
    # layer, so the completed multi-stream state is combined here (hc_combine is pure and its
    # result feeds nothing else, so the target's own computation is unchanged) and averaged over
    # the streams, the 2560-wide feature DFlash2 was trained on.
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        '    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n'
        '        return self.embed_tokens(input_ids)\n',
        '    aux_hidden_state_layers: tuple[int, ...] = ()\n\n'
        '    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n'
        '        return self.embed_tokens(input_ids)\n',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        '        block_output = None\n        injection = None\n        last_layer = None\n',
        '        block_output = None\n        injection = None\n        last_layer = None\n'
        '        aux_hidden_states: list[torch.Tensor] = []\n',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        '                hidden_states = hidden_states + deepstack_embed\n\n'
        '        if not get_pp_group().is_last_rank:\n',
        '                hidden_states = hidden_states + deepstack_embed\n'
        '            if layer_idx + 1 in self.aux_hidden_state_layers:\n'
        '                completed = (\n'
        '                    hidden_states\n'
        '                    if block_output is None\n'
        '                    else layer.mlp_hyper_connection.combine(\n'
        '                        hidden_states, block_output, injection\n'
        '                    )\n'
        '                )\n'
        '                aux_hidden_states.append(\n'
        '                    completed.unflatten(\n'
        '                        -1, (self.config.hc_count, self.config.hidden_size)\n'
        '                    ).mean(dim=-2)\n'
        '                )\n\n'
        '        if not get_pp_group().is_last_rank:\n',
    ),
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        '            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)\n'
        '        return sample_hidden_states\n',
        '            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)\n'
        '        if self.aux_hidden_state_layers:\n'
        '            return sample_hidden_states, aux_hidden_states\n'
        '        return sample_hidden_states\n',
    ),
    # SupportsEagle3 is a runtime-checked protocol: the runner asks the target for these members
    # before it requests auxiliary hidden states.
    (
        'vllm/models/qwen4_exp/nvidia/model.py',
        '    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:\n'
        '        return self.language_model.get_mtp_target_hidden_states()\n',
        '    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:\n'
        '        return self.language_model.get_mtp_target_hidden_states()\n\n'
        '    supports_eagle3 = True\n'
        '    has_own_lm_head = False\n'
        '    has_own_embed_tokens = False\n\n'
        '    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:\n'
        '        layers = tuple(sorted(set(layers)))\n'
        '        self.language_model.model.aux_hidden_state_layers = layers\n'
        '        logger.info(\n'
        '            "Qwen4Exp DFlash taps: outputs of layers %s, mean over HC streams",\n'
        '            [layer - 1 for layer in layers],\n'
        '        )\n\n'
        '    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:\n'
        '        num_layers = len(self.language_model.model.layers)\n'
        '        return (2, num_layers // 2, num_layers - 3)\n',
    ),
)


def patch_runtime_for_dflash(manifest: dict) -> None:
    '''Let the e975732 rc3 runtime serve a DFlash drafter for Qwen4Exp (vllm#56088).

    The runtime has vLLM's DFlash2 speculator, but Qwen4Exp rejects the method, the V2 runner
    dereferences the MTP buffer that Qwen4Exp allocates only for MTP, and the model never returns
    the auxiliary hidden states the drafter reads. DFLASH_RUNTIME_PATCHES fixes all three, and
    an anchor that does not match exactly once fails the setup. MTP runs leave the runtime as is.
    '''
    if VLLM_DRAFT_METHOD != 'dflash':
        return
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    sources = {}
    for relative_path, anchor, replacement in DFLASH_RUNTIME_PATCHES:
        path = dist_packages / relative_path
        source = sources.get(path) or (path.read_text(encoding='utf-8') if path.is_file() else '')
        if source.count(anchor) != 1:
            raise RuntimeError(f'DFlash runtime patch anchor appears {source.count(anchor)} times in {path}:\n{anchor}')
        sources[path] = source.replace(anchor, replacement)
    for path, source in sources.items():
        path.write_text(source, encoding='utf-8')
    print(f'Patched the runtime for DFlash drafting: {len(DFLASH_RUNTIME_PATCHES)} edits in {len(sources)} files', flush=True)


def resolve_dflash_draft() -> Path:
    '''Return the mounted DFlash checkpoint, which vLLM loads as it is.

    It ships no embedding or LM head; the DFlash loader binds the target's.
    '''
    source = resolve_kaggle_model_path(DRAFT_MODEL_SOURCE_PARTS)
    config_path = source / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'DFlash draft checkpoint not mounted at {source}\n' + describe_kaggle_input_tree())
    config = json.loads(config_path.read_text(encoding='utf-8'))
    dflash_config = config.get('dflash_config') or {}
    print(
        f'DFlash draft at {source}: {config.get("architectures")}, target layers '
        f'{dflash_config.get("target_layer_ids")}, block {dflash_config.get("block_size")}, '
        f'{VLLM_MTP_TOKENS} draft tokens',
        flush=True,
    )
    return source


def install_image_runtime() -> dict:
    '''Unpack the image layers in index order, replay whiteouts and apply the runtime overlay.'''
    manifest = read_runtime_manifest()
    validate_prefix_cache_runtime(manifest)
    shutil.rmtree(IMAGE_RUNTIME_ROOT, ignore_errors=True)
    IMAGE_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    layers = sorted(manifest['selected_layers'], key=lambda layer: int(layer['index']))
    started = time.monotonic()
    for layer in layers:
        blob = verify_layer_blob(layer)
        clear_opaque_whiteout_targets(layer)
        extract_layer(blob)
        remove_whiteout_markers_and_targets(layer)
        print(f'Extracted runtime layer {layer["index"]} ({blob.name}) after {time.monotonic() - started:.0f}s', flush=True)
    apply_runtime_patch(manifest)
    print(
        f'Image runtime ready at {IMAGE_RUNTIME_ROOT}: vLLM {manifest["vllm_version"]}, torch {manifest["torch_version"]}',
        flush=True,
    )

    return manifest


def validate_prefix_cache_runtime(manifest: dict) -> None:
    '''Fail closed when fine-grained Qwen3.8 caching lacks its Mamba/MTP safety overlays.'''
    if not VLLM_ENABLE_PREFIX_CACHING or VLLM_PREFIX_MATCH_UNIT <= 0:

        return

    identity_mismatches = sorted(
        key
        for key, expected_value in FINE_PREFIX_CACHE_RUNTIME_IDENTITY.items()
        if manifest.get(key) != expected_value
    )
    if identity_mismatches:
        raise RuntimeError(
            'Fine-grained Qwen3.8 prefix caching requires the pinned vLLM runtime identity; '
            'mismatched fields: ' + ', '.join(identity_mismatches)
        )

    patch = manifest.get('patch') or {}
    patch_identity_mismatches = sorted(
        key
        for key, expected_value in FINE_PREFIX_CACHE_PATCH_IDENTITY.items()
        if patch.get(key) != expected_value
    )
    if patch_identity_mismatches:
        raise RuntimeError(
            'Fine-grained Qwen3.8 prefix caching requires the pinned runtime patch identity; '
            'mismatched fields: ' + ', '.join(patch_identity_mismatches)
        )

    overlays = patch.get('overlay_files', [])
    overlay_targets = [overlay.get('target') for overlay in overlays]
    overlays_by_target = {
        overlay.get('target'): overlay
        for overlay in overlays
    }
    missing_targets = sorted(
        set(FINE_PREFIX_CACHE_REQUIRED_OVERLAYS) - set(overlay_targets)
    )
    duplicate_targets = sorted(
        target
        for target in FINE_PREFIX_CACHE_REQUIRED_OVERLAYS
        if overlay_targets.count(target) > 1
    )
    mismatched_targets = sorted(
        target
        for target, expected_hashes in FINE_PREFIX_CACHE_REQUIRED_OVERLAYS.items()
        if target in overlays_by_target
        and any(
            overlays_by_target[target].get(hash_name) != expected_hash
            for hash_name, expected_hash in expected_hashes.items()
        )
    )
    problems = []
    if missing_targets:
        problems.append('missing targets: ' + ', '.join(missing_targets))
    if duplicate_targets:
        problems.append('duplicate targets: ' + ', '.join(duplicate_targets))
    if mismatched_targets:
        problems.append('mismatched overlay hashes: ' + ', '.join(mismatched_targets))
    if problems:
        raise RuntimeError(
            'Fine-grained Qwen3.8 prefix caching requires the validated Mamba/MTP runtime overlays; '
            + '; '.join(problems)
        )


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def request_no_content(url: str, timeout: int = 30) -> None:
    request = urllib.request.Request(url, data=b'', method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f'Unexpected HTTP {response.status} from {url}')


def tail_log(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ''
    return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def wait_for_vllm_server(watchdog_process: subprocess.Popen, timeout_seconds: int) -> None:
    '''Poll the models endpoint until the server answers.

    The watchdog exits when the server dies before it was marked ready, and polling the watchdog
    reaps it, so a broken launch raises at once instead of running out the timeout.
    '''
    deadline = time.monotonic() + timeout_seconds
    url = f'{VLLM_BASE_URL}/models'
    while time.monotonic() < deadline:
        if watchdog_process.poll() is not None:
            raise RuntimeError(
                f'vLLM watchdog exited with code {watchdog_process.returncode} before the server was ready.\n'
                f'{tail_log(VLLM_WATCHDOG_LOG)}'
            )
        try:
            models = request_json(url, timeout=5)
            print('vLLM server ready:', models, flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f'Timed out waiting for vLLM server at {url}.\nLast server log lines:\n{tail_log(VLLM_SERVER_LOG)}')


def build_vllm_server_command(draft_model_dir: Path | None = None) -> list[str]:
    '''The Qwen3.8-Flash-Next launch on the packed image runtime, as verified on Kaggle in v187.

    Prefix caching follows the config. With it enabled, the hybrid model uses
    mamba_cache_mode "align" and the runtime aligns Mamba checkpoints to the resolved
    attention block size. The patched runtime is required for correct scheduler and
    worker alignment; saved checkpoints share the KV pool with active requests.

    The quantization method, the sequence cap and the KV cache dtype come from the config, so all
    three reach this command. The method is passed explicitly rather than left to vLLM's detection,
    so a checkpoint whose config.json disagrees fails at startup instead of loading under a scheme
    nobody picked. The cap has to follow the KV pool rather than the worker count, because a cap
    above what the pool holds only queues the surplus in the scheduler. The experimental runtime
    adds FP8 KV support to QSA attention. Auto is passed as no flag at all; FP8 is enabled only
    when explicitly selected in the config. The MTP head is the model's own drafter unless the
    config names a draft checkpoint, whose assembled head is then the draft model. The pinned KV cache size is optional; zero lets vLLM size the pool
    from gpu_memory_utilization after profiling the activation peak.

    A decode step carries one token per sequence plus its MTP drafts, and vLLM captures CUDA
    graphs at 1, 2 and 4 tokens and then at every multiple of 8 below 256. A capture size off that
    ladder is truncated down to the last size it captured, so a full batch whose step lands above
    it decodes with no graph at all. That is what nine sequences did in v216, a 36-token step
    against a largest graph of 32. The size is rounded up to the next multiple of 8 instead, and a
    step that falls between two sizes pads into the graph above it.

    With a draft vocabulary the MTP drafter picks its greedy proposals through
    use_local_argmax_reduction, which the e975732 rc3 runtime serves from the listed LM-head rows
    only. The target still verifies against the full head, so the list changes draft acceptance
    and never the generated text.

    embed_tokens in --cpu-offload-params keeps the 1.18 GiB input embedding table in pinned host
    memory, next to the PLE table, and the KV pool gets those bytes. The e975732 rc3 runtime reads
    only the requested rows over PCIe, so embeddings stay bit-identical, and the MTP drafter shares
    the table. The table sits outside the --cpu-offload-gb budget, which stays 0, so vLLM's generic
    weight offloader never starts.

    vLLM profiles the vision encoder on the largest image the preprocessor accepts, 4096x4096
    or 16384 tokens, and in v257 that one dummy set the whole 2.19 GiB activation peak, which the
    KV pool gives up. The agent only ever sends its grid image, so max_pixels caps the
    preprocessor at that image's area. The profile then encodes as many grid images as one
    max_num_batched_tokens chunk holds, the worst case a real step can meet, while a grid image is
    processed exactly as before. Video is disabled, otherwise its 12288-token dummy becomes the profiled item.
    '''
    max_num_seqs = VLLM_MAX_NUM_SEQS
    decode_step_tokens = max_num_seqs * (1 + VLLM_MTP_TOKENS)
    cudagraph_capture_size = (
        math.ceil(decode_step_tokens / VLLM_CUDAGRAPH_CAPTURE_STEP) * VLLM_CUDAGRAPH_CAPTURE_STEP
    )
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.cli.main',
        'serve',
        str(MODEL_PATH),
        '--served-model-name',
        SERVED_MODEL_NAME,
        '--host',
        VLLM_HOST,
        '--port',
        str(VLLM_PORT),
        '--load-format',
        'safetensors',
        '--dtype',
        VLLM_DTYPE,
        '--tensor-parallel-size',
        str(VLLM_TENSOR_PARALLEL_SIZE),
        '--distributed-executor-backend',
        'mp',
        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
        '--max-num-seqs',
        str(max_num_seqs),
        '--max-num-batched-tokens',
        str(VLLM_MAX_NUM_BATCHED_TOKENS),
        '--async-scheduling',
        '--enable-chunked-prefill',
        '--max-cudagraph-capture-size',
        str(cudagraph_capture_size),
        '--enable-prefix-caching' if VLLM_ENABLE_PREFIX_CACHING else '--no-enable-prefix-caching',
        '--enable-auto-tool-choice',
        '--tool-call-parser',
        'qwen3_coder',
        '--reasoning-parser',
        'qwen3',
        '--generation-config',
        'vllm',
        '--engram-config',
        json.dumps({'cpu_offload': True}),
        '--cpu-offload-params',
        'embed_tokens',
        '--mm-processor-kwargs',
        json.dumps({'max_pixels': VLLM_IMAGE_MAX_PIXELS}),
        '--limit-mm-per-prompt',
        json.dumps({'video': 0}),
        '--default-chat-template-kwargs',
        '{"preserve_thinking": true, "reasoning_effort": "xhigh"}',
    ]
    if VLLM_QUANTIZATION:
        cmd += ['--quantization', VLLM_QUANTIZATION]
    if VLLM_MOE_BACKEND:
        cmd += ['--moe-backend', VLLM_MOE_BACKEND]
    if VLLM_ENABLE_PREFIX_CACHING and VLLM_PREFIX_MATCH_UNIT > 0:
        cmd += ['--prefix-match-unit', str(VLLM_PREFIX_MATCH_UNIT)]
    if VLLM_ENABLE_PREFIX_CACHING and VLLM_PREFIX_CACHE_RETENTION_INTERVAL >= 0:
        cmd += [
            '--prefix-cache-retention-interval',
            str(VLLM_PREFIX_CACHE_RETENTION_INTERVAL),
        ]
    if VLLM_MTP_TOKENS > 0:
        speculative_config = {'method': VLLM_DRAFT_METHOD, 'num_speculative_tokens': VLLM_MTP_TOKENS}
        if draft_model_dir is not None:
            speculative_config['model'] = str(draft_model_dir)
        if VLLM_DRAFT_VOCAB and VLLM_DRAFT_METHOD == 'mtp':
            speculative_config['use_local_argmax_reduction'] = True
        cmd += ['--speculative-config', json.dumps(speculative_config)]
    if VLLM_GPU_MEMORY_UTILIZATION:
        cmd += ['--gpu-memory-utilization', VLLM_GPU_MEMORY_UTILIZATION]
    if VLLM_KV_CACHE_MEMORY_BYTES > 0:
        cmd += ['--kv-cache-memory-bytes', str(VLLM_KV_CACHE_MEMORY_BYTES)]
    if VLLM_KV_CACHE_DTYPE and VLLM_KV_CACHE_DTYPE != 'auto':
        cmd += ['--kv-cache-dtype', VLLM_KV_CACHE_DTYPE]
    if VLLM_INDEXER_KV_DTYPE:
        cmd += ['--attention-config', json.dumps({'indexer_kv_dtype': VLLM_INDEXER_KV_DTYPE})]
    if VLLM_MAMBA_SSM_CACHE_DTYPE and VLLM_MAMBA_SSM_CACHE_DTYPE != 'auto':
        cmd += ['--mamba-ssm-cache-dtype', VLLM_MAMBA_SSM_CACHE_DTYPE]
    chat_template = MODEL_PATH / 'chat_template.jinja'
    if chat_template.exists():
        cmd += ['--chat-template', str(chat_template)]

    return cmd


def configure_cache_diagnostics(manifest: dict, server_env: dict, cmd: list[str]) -> None:
    '''Install observational hooks only in test mode, and only when the config asks for them.'''
    namespace = {'__name__': 'arc3_cache_diagnostics'}
    exec(CACHE_DIAGNOSTICS_SOURCE, namespace)
    enabled = VLLM_CACHE_DIAGNOSTICS_REQUESTED and namespace['is_test_run']()
    server_env['ARC3_CACHE_DIAGNOSTICS'] = '1' if enabled else '0'
    server_env.pop('ARC3_CACHE_DIAGNOSTICS_DIR', None)
    if not enabled:
        return
    dist_packages = IMAGE_RUNTIME_ROOT / manifest['dist_packages']
    (dist_packages / 'arc3_cache_diagnostics.py').write_text(CACHE_DIAGNOSTICS_SOURCE, encoding='utf-8')
    metadata = dist_packages / 'arc3_cache_diagnostics-1.0.dist-info'
    metadata.mkdir(exist_ok=True)
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: arc3-cache-diagnostics\nVersion: 1.0\n')
    (metadata / 'entry_points.txt').write_text(
        '[vllm.general_plugins]\narc3_cache_diagnostics = arc3_cache_diagnostics:install_cache_diagnostics\n'
    )
    if 'VLLM_PLUGINS' in server_env:
        server_env['VLLM_PLUGINS'] += ',arc3_cache_diagnostics'
    server_env['ARC3_CACHE_DIAGNOSTICS_DIR'] = str(WORKING_DIR / 'metrics')
    cmd += ['--kv-cache-metrics', '--kv-cache-metrics-sample', '0.1']


def should_enable_vllm_profiler() -> bool:
    '''Fail closed: a request is honored only in an explicit non-competition test run.'''

    return VLLM_PROFILE_REQUESTED and is_test_run()


def configure_vllm_profiler(cmd: list[str]) -> None:
    '''Add the bounded worker profiler without touching production commands.'''
    if not should_enable_vllm_profiler():
        return
    VLLM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    cmd += [
        '--profiler-config',
        json.dumps({
            'profiler': 'torch',
            'torch_profiler_dir': str(VLLM_PROFILE_DIR),
            'torch_profiler_with_stack': False,
            'torch_profiler_use_gzip': True,
            'torch_profiler_dump_cuda_time_total': True,
            'torch_profiler_record_shapes': True,
            'torch_profiler_with_memory': False,
            'ignore_frontend': True,
            # The pinned WorkerProfiler starts on count == delay before
            # execute_model, so 6 skips exactly 5 iterations. It also stops on
            # profiled_count > max before execute_model, making 20 the number
            # of model iterations captured despite the internal counter reaching 21.
            'delay_iterations': 6,
            'max_iterations': 20,
        }),
    ]


def start_vllm_profiler() -> None:
    '''Start profiling after startup and smoke traffic, immediately before ARC requests.'''
    if not should_enable_vllm_profiler():
        return
    request_no_content(f'{VLLM_SERVER_URL}/start_profile', timeout=30)
    print(
        'vLLM profiler armed: skip 5 engine iterations, capture 20; '
        f'traces will be written to {VLLM_PROFILE_DIR}',
        flush=True,
    )


def build_watchdog_config(cmd: list[str]) -> dict:
    '''Configure the server watchdog and test-only host RAM sampling.'''
    watchdog_config = {
        'command': cmd,
        'server_log': str(VLLM_SERVER_LOG),
        'server_pid': str(VLLM_SERVER_PID),
        'server_ready': str(VLLM_SERVER_READY),
        'model_path': str(MODEL_PATH),
        'shard_prefetch_report': str(VLLM_SHARD_PREFETCH_REPORT),
        'prefetch_threads': VLLM_PREFETCH_THREADS,
    }
    if is_test_run():
        watchdog_config['host_memory_log'] = str(WORKING_DIR / 'metrics' / 'host_memory.jsonl')

    return watchdog_config


def start_vllm_server() -> None:
    '''Unpack the image runtime and launch the vLLM OpenAI server against the mounted model.

    The server is not spawned directly but through a watchdog process that becomes its parent.
    A CUDA illegal memory access poisons the engine, and the API server process exits in the same
    second, so the watchdog notices through a blocking wait on its child with no polling at all. It
    logs the exit and the tail of the server log to VLLM_WATCHDOG_LOG and starts the server again.
    The harness retries a failed request every second until its deadline, so games resume on their
    own once the new server binds the port. A server that dies before VLLM_SERVER_READY exists is
    not restarted, so a broken launch still fails this setup step quickly, and the watchdog gives up
    after a fixed number of restarts so a deterministic crash does not loop for the rest of the run.

    A separate process prefetches the GPU shards a bounded window ahead of each weight load
    (GpuShardPrefetcher), including restarts. It follows only the current start's server log and
    finishes long before the server is ready. If the server dies first, the watchdog terminates
    prefetch with bounded waits before restarting, so a blocked NFS read cannot delay recovery
    indefinitely.
    '''
    manifest = install_image_runtime()
    route_inc_ple_to_unquantized(manifest)
    pin_ple_table_at_exact_size(manifest)
    check_runtime_reads_fp8_ple(manifest)
    patch_runtime_for_dflash(manifest)
    draft_model_dir = resolve_dflash_draft() if VLLM_DRAFT_METHOD == 'dflash' else assemble_mtp_draft()
    server_env = build_image_runtime_env(manifest)
    cmd = build_vllm_server_command(draft_model_dir)
    configure_cache_diagnostics(manifest, server_env, cmd)
    configure_vllm_profiler(cmd)
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_LOG.unlink(missing_ok=True)
    VLLM_SHARD_PREFETCH_REPORT.unlink(missing_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    VLLM_SERVER_READY.unlink(missing_ok=True)
    if os.environ.get('TAAF_RUN_AS_SUBMISSION') == '1':
        cmd += ['--disable-log-stats', '--disable-uvicorn-access-log']
    else:
        cmd += ['--enable-prompt-tokens-details']

    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    VLLM_WATCHDOG_SCRIPT.write_text(WATCHDOG_SCRIPT_TEXT, encoding='utf-8')
    VLLM_WATCHDOG_CONFIG.write_text(json.dumps(build_watchdog_config(cmd)), encoding='utf-8')
    watchdog_log_handle = VLLM_WATCHDOG_LOG.open('w', encoding='utf-8')
    watchdog_process = subprocess.Popen(
        [sys.executable, str(VLLM_WATCHDOG_SCRIPT), str(VLLM_WATCHDOG_CONFIG)],
        env=server_env,
        stdout=watchdog_log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    VLLM_WATCHDOG_PID.write_text(str(watchdog_process.pid), encoding='utf-8')
    wait_for_vllm_server(watchdog_process, VLLM_SERVER_READY_TIMEOUT_SECONDS)
    VLLM_SERVER_READY.write_text(datetime.now().isoformat(timespec='seconds'), encoding='utf-8')


def print_vllm_backend_summary() -> None:
    markers = (
        'attention backend',
        'attentionbackendenum',
        'kv cache layout',
        'kernel for',
        'block fp8',
        'prefill kernel',
        'gpu kv cache size',
        'maximum concurrency',
    )
    log_lines = VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace').splitlines()
    print('\n' + '=' * 88, flush=True)
    print('VLLM ATTENTION BACKEND / KV CACHE SUMMARY', flush=True)
    for line in log_lines:
        if any(marker in line.lower() for marker in markers):
            print(line, flush=True)
    print('=' * 88 + '\n', flush=True)


def run_vllm_api_smoke_test() -> None:
    payload = {
        'model': SERVED_MODEL_NAME,
        'messages': [{'role': 'user', 'content': 'Answer in one short sentence: what is 2 + 2?'}],
        'temperature': 0.0,
        'max_tokens': 96,
        'chat_template_kwargs': {'enable_thinking': False},
    }
    response = request_json(f'{VLLM_BASE_URL}/chat/completions', payload=payload, timeout=120)
    generated = response['choices'][0]['message'].get('content', '').strip()
    print('\n' + '=' * 88, flush=True)
    print('VLLM OPENAI SERVER QWEN SMOKE TEST REAL MODEL OUTPUT', flush=True)
    print('Generated:', generated, flush=True)
    print('=' * 88 + '\n', flush=True)


print(f'vLLM runtime dataset path: {RUNTIME_DATASET}', flush=True)
print(f'Qwen model path: {MODEL_PATH}', flush=True)
assert_expected_cuda_gpu()
missing = [str(path) for path in (RUNTIME_DATASET, MOUNTED_MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError(
        'Missing attached input path(s): ' + ', '.join(missing) + '\n' + describe_kaggle_input_tree()
    )
assemble_ple_fp8_target()
start_vllm_server()
print_vllm_backend_summary()
if os.environ.get('TAAF_RUN_AS_SUBMISSION') == '1':
    print('Skipping vLLM smoke test (submission mode)', flush=True)
else:
    run_vllm_api_smoke_test()
start_vllm_profiler()
setup_env = {
    'ARC3_CACHE_DIAGNOSTICS': '1' if is_test_run() else '0',
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'OPENAI_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': __LOCAL_ANALYZER_APP_NAME__,
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_TARGET_CONTEXT': __LOCAL_ANALYZER_TARGET_CONTEXT__,
    'LOCAL_ANALYZER_MAX_OUTPUT': __LOCAL_ANALYZER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_TOOL_STEPS': __LOCAL_ANALYZER_TOOL_STEPS__,
    'LOCAL_ANALYZER_TOOL_TIMEOUT': __LOCAL_ANALYZER_TOOL_TIMEOUT__,
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': __LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__,
    'LOCAL_ANALYZER_YIELD_SECONDS': __LOCAL_ANALYZER_YIELD_SECONDS__,
    'LOCAL_ANALYZER_TEMPERATURE': __LOCAL_ANALYZER_TEMPERATURE__,
    'LOCAL_ANALYZER_TOP_P': __LOCAL_ANALYZER_TOP_P__,
    'LOCAL_ANALYZER_TOP_K': __LOCAL_ANALYZER_TOP_K__,
    'LOCAL_ANALYZER_ENABLE_THINKING': __LOCAL_ANALYZER_ENABLE_THINKING__,
    'LOCAL_ANALYZER_REASONING_EFFORT': __LOCAL_ANALYZER_REASONING_EFFORT__,
    'MULTIMODAL_CONTEXT': __MULTIMODAL_CONTEXT__,
    'MULTIMODAL_UPSCALE': __MULTIMODAL_UPSCALE__,
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
"""

_DUCK_VLLM_TEARDOWN_SCRIPT = r"""import os
import shutil
import signal
import time
from pathlib import Path

WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])


def stop_process(pid_path: Path, name: str) -> None:
    '''Terminate the process recorded in pid_path, escalating to SIGKILL after 30 seconds.'''
    if not pid_path.exists():
        return
    try:
        pid = int(pid_path.read_text(encoding='utf-8').strip())
        print(f'Stopping {name}', flush=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        print(f'Could not stop {name} cleanly: {exc!r}', flush=True)
    pid_path.unlink(missing_ok=True)


stop_process(WORKING_DIR / 'vllm-watchdog.pid', 'vLLM watchdog')
stop_process(WORKING_DIR / 'vllm-openai-server.pid', 'vLLM server')
shutil.rmtree('/tmp/vllm-mtp-draft', ignore_errors=True)
shutil.rmtree('/tmp/vllm-target-fp8-ple', ignore_errors=True)
image_runtime_root = Path(os.environ.get('TAAF_IMAGE_RUNTIME_ROOT', '/tmp/vllm-image-runtime'))
for runtime_path in (image_runtime_root, image_runtime_root.with_name(image_runtime_root.name + '-cache')):
    if runtime_path.exists():
        shutil.rmtree(runtime_path, ignore_errors=True)
        print(f'Removed image runtime files at {runtime_path}', flush=True)
"""
