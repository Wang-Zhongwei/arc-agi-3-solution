"""Kaggle helpers for the ARC3 duck harness."""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from inference.utils.openai_compat import normalize_provider

DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
DEFAULT_QWEN_MODEL_DATASET_SOURCE = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
DEFAULT_SERVED_MODEL_NAME = "vrfai/Qwen3.6-27B-FP8"
DEFAULT_VLLM_PORT = 1234
DEFAULT_VLLM_MAX_MODEL_LEN = 65536
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_WHEELHOUSE_STAMP_TEXT = "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n"

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

    wheelhouse_dataset_source: str = DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE
    model_dataset_source: str = DEFAULT_QWEN_MODEL_DATASET_SOURCE
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME
    vllm_port: int = DEFAULT_VLLM_PORT
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    wheelhouse_stamp_text: str = DEFAULT_WHEELHOUSE_STAMP_TEXT


def duck_kaggle_dataset_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    cfg = config or DuckKaggleVllmConfig()
    return [cfg.wheelhouse_dataset_source, cfg.model_dataset_source]


# Endpoint keys belong to whichever server the notebook starts.
_SERVING_KEYS = frozenset({
    "LOCAL_ANALYZER_BASE_URL", "OPENAI_BASE_URL", "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER", "LOCAL_ANALYZER_MODEL_ID", "INFERENCE_ANALYZER_MODEL",
})


def harness_env() -> dict[str, str]:
    """Harness settings from the config's `harness_env` section.

    The Makefile exports the section as ARC3_HARNESS_ENV_JSON. Any harness
    variable is accepted (ARC3_*, LOCAL_ANALYZER_*, MULTIMODAL_*, ...) except
    the endpoint keys, which the serving setup pins. Values are stringified
    like any env var.
    """
    raw = os.environ.get("ARC3_HARNESS_ENV_JSON", "").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("harness_env must be a JSON object")
    bad = sorted(str(key) for key in data if str(key) in _SERVING_KEYS)
    if bad:
        raise ValueError(f"harness_env cannot set serving endpoint keys {bad}")
    return {
        str(key): ("1" if value is True else "0" if value is False else str(value))
        for key, value in data.items() if value is not None
    }


# Analyzer and image settings the Makefile exports from the config's analyzer,
# multimodal and shared sections: the same set the vLLM setup script embeds.
_CONFIG_ANALYZER_KEYS = (
    "LOCAL_ANALYZER_APP_NAME", "LOCAL_ANALYZER_CONTEXT_WINDOW", "LOCAL_ANALYZER_MAX_OUTPUT",
    "LOCAL_ANALYZER_THINKING_TOKEN_BUDGET", "LOCAL_ANALYZER_TOOL_STEPS",
    "LOCAL_ANALYZER_TOOL_TIMEOUT", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS",
    "LOCAL_ANALYZER_YIELD_SECONDS", "LOCAL_ANALYZER_TEMPERATURE", "LOCAL_ANALYZER_TOP_P",
    "LOCAL_ANALYZER_TOP_K", "LOCAL_ANALYZER_ENABLE_THINKING",
    "MULTIMODAL_CONTEXT", "MULTIMODAL_UPSCALE",
)


def config_analyzer_env() -> dict[str, str]:
    """Config analyzer/multimodal settings as the Makefile exported them.

    Without these a Pennyroyal run falls back to the agent's own defaults: no
    board images and a 32K context window (new-baseline v1).
    """
    return {key: os.environ[key] for key in _CONFIG_ANALYZER_KEYS if os.environ.get(key, "").strip()}


def apply_harness_env() -> None:
    """Local runs: config settings fill in what the shell did not set."""
    for key, value in harness_env().items():
        os.environ.setdefault(key, value)


def kaggle_serving_config() -> dict:
    """The config's `kaggle` section, exported by the Makefile as ARC3_KAGGLE_JSON."""
    raw = os.environ.get("ARC3_KAGGLE_JSON", "").strip()
    data = json.loads(raw) if raw else {}
    if not isinstance(data, dict):
        raise ValueError("kaggle config section must be a JSON object")
    serving = data.get("serving", "vllm")
    if serving not in ("vllm", "pennyroyal", "lordhansolo"):
        raise ValueError(f"kaggle.serving must be 'vllm', 'pennyroyal', or 'lordhansolo', got {serving!r}")
    if serving in ("pennyroyal", "lordhansolo"):
        for key in ("wheelhouse_dataset_source", "model_source"):
            if not data.get(key):
                raise ValueError(f"kaggle.serving={serving} requires kaggle.{key}")
    return {**data, "serving": serving}


def pennyroyal_dataset_sources() -> list[str]:
    return [kaggle_serving_config()["wheelhouse_dataset_source"]]


def pennyroyal_model_sources() -> list[str]:
    cfg = kaggle_serving_config()
    return [ref for ref in (cfg["model_source"], cfg.get("draft_model_source")) if ref]


def pennyroyal_setup_command() -> str:
    """Start Flash Next on Pennyroyal SGLang, then publish endpoint + harness env."""
    cfg = kaggle_serving_config()
    spec = {
        key: cfg[key] for key in (
            "wheelhouse_dataset_source", "model_source", "draft_model_source",
            "served_model_name", "port", "server_startup_timeout_s", "sglang",
        ) if key in cfg
    }
    # harness_env wins over the analyzer/multimodal sections, as on vLLM.
    spec["env"] = {**config_analyzer_env(), **harness_env()}
    launcher = "$TAAF_KAGGLE_BUNDLE_DIR/src/ARC3-Inference/inference/framework/pennyroyal_launch.py"
    return f'ARC3_PENNYROYAL_SPEC={shlex.quote(json.dumps(spec))} "$PYTHON" "{launcher}"'


def pennyroyal_teardown_command() -> str:
    return (
        'pid_file="$TAAF_KAGGLE_WORKING_DIR/sglang-server.pid"; '
        '[ -f "$pid_file" ] && kill "$(cat "$pid_file")" 2>/dev/null; rm -f "$pid_file"; true'
    )


def lordhansolo_setup_command() -> str:
    """Use Lord's pinned serving runtime with this config's harness settings."""
    from inference.framework.lordhansolo import kaggle as runtime

    cfg = kaggle_serving_config()
    runtime.PROJECT_DIR = Path(__file__).resolve().parents[2]
    server = runtime.DuckKaggleVllmConfig(
        runtime_dataset_source=cfg["wheelhouse_dataset_source"],
        model_source=cfg["model_source"],
        served_model_name=cfg.get("served_model_name", "flashnext"),
        vllm_port=int(cfg.get("port", 8001)),
        **cfg.get("vllm", {}),
    )
    command = runtime.duck_kaggle_setup_command(server)
    # Lord's source hardcodes xhigh reasoning in the server template. Keep this
    # harness's template settings when swapping engines.
    original = '\'{"preserve_thinking": true, "reasoning_effort": "xhigh"}\''
    template = json.dumps(cfg.get("default_chat_template_kwargs", {"preserve_thinking": True}))
    if command.count(original) != 1:
        raise ValueError("Lord Hansolo launcher template changed; review the adapter")
    command = command.replace(original, repr(template))
    # The upstream launcher hands off its own analyzer defaults. Override them
    # with our complete harness_env before the benchmark imports the agent.
    marker = "setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])"
    if command.count(marker) != 1:
        raise ValueError("Lord Hansolo launcher handoff changed; review the adapter")
    overrides = harness_env()
    overrides.setdefault(
        "LOCAL_ANALYZER_THINKING_TOKEN_BUDGET",
        os.environ.get("LOCAL_ANALYZER_THINKING_TOKEN_BUDGET", "0"),
    )
    overrides["ARC3_KAGGLE_PRIVATE_WAVES"] = str(cfg.get("private_waves", 1))
    overrides["ARC3_KAGGLE_PRIVATE_RUNTIME_MINUTES"] = str(cfg.get("private_runtime_minutes", 532))
    handoff = f"setup_env.update({overrides!r})\n"
    for key in ("LOCAL_ANALYZER_REASONING_EFFORT", "LOCAL_ANALYZER_TARGET_CONTEXT"):
        if key not in overrides:
            handoff += f"setup_env.pop({key!r}, None)\n"
    # Preserve the configured prompt window even if the upstream renderer would
    # clamp it. Reject a server window that cannot hold a full response instead.
    context = int(overrides.get("LOCAL_ANALYZER_CONTEXT_WINDOW", os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW", 131072)))
    output = int(overrides.get("LOCAL_ANALYZER_MAX_OUTPUT", os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", 12288)))
    if context + output > server.max_model_len:
        raise ValueError("Lord Hansolo max_model_len must fit the harness context window plus output budget")
    handoff += f"setup_env['LOCAL_ANALYZER_CONTEXT_WINDOW'] = {str(context)!r}\n"
    return command.replace(marker, handoff + marker)


def lordhansolo_teardown_command() -> str:
    from inference.framework.lordhansolo import kaggle as runtime

    return runtime.duck_kaggle_teardown_command()


def duck_kaggle_setup_command(config: DuckKaggleVllmConfig | None = None) -> str:
    cfg = config or DuckKaggleVllmConfig()
    wheelhouse_owner, wheelhouse_slug = _split_dataset_source(
        cfg.wheelhouse_dataset_source,
        option_name="wheelhouse_dataset_source",
    )
    model_owner, model_slug = _split_dataset_source(
        cfg.model_dataset_source,
        option_name="model_dataset_source",
    )
    # Base URL / model are pinned to the local vLLM server below, so reject a
    # provider that disagrees (e.g. openrouter) — it would drop vLLM-only payload
    # fields (top_k, chat_template_kwargs) against a vLLM endpoint.
    analyzer_provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")
    if normalize_provider(analyzer_provider) != "vllm":
        raise ValueError(
            f"kaggle-duck runs a local vLLM server, so LOCAL_ANALYZER_PROVIDER must be "
            f"vLLM/OpenAI-compatible, got {analyzer_provider!r}."
        )
    replacements = {
        "__WHEELHOUSE_OWNER__": repr(wheelhouse_owner),
        "__WHEELHOUSE_SLUG__": repr(wheelhouse_slug),
        "__MODEL_OWNER__": repr(model_owner),
        "__MODEL_SLUG__": repr(model_slug),
        "__SERVED_MODEL_NAME__": repr(cfg.served_model_name),
        "__VLLM_PORT__": repr(int(cfg.vllm_port)),
        "__VLLM_MAX_MODEL_LEN__": repr(int(cfg.max_model_len)),
        # The launcher's Makefile exports LOCAL_ANALYZER_CONTEXT_WINDOW from
        # JSON shared.context_window (or analyzer.context_window). Embed it
        # here so the agent's prompt budget on Kaggle is the JSON value, not
        # vllm's max-model-len. Falls back to max_model_len if unset.
        "__ANALYZER_CONTEXT_WINDOW__": repr(
            int(os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW") or cfg.max_model_len)
        ),
        # Remaining JSON-driven analyzer/multimodal config: the launcher's
        # Makefile exports each from inference.json; embed the launcher value
        # so the rendered setup_env on Kaggle reflects JSON edits. Fallback
        # equals the historical hardcoded literal so direct kaggle.py callers
        # outside Make are unaffected.
        "__LOCAL_ANALYZER_PROVIDER__": repr(analyzer_provider),
        "__LOCAL_ANALYZER_APP_NAME__": repr(os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Kaggle Harness")),
        "__LOCAL_ANALYZER_MAX_OUTPUT__": repr(os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", "0")),
        "__LOCAL_ANALYZER_THINKING_TOKEN_BUDGET__": repr(
            os.environ.get("LOCAL_ANALYZER_THINKING_TOKEN_BUDGET", "0")
        ),
        "__LOCAL_ANALYZER_TOOL_STEPS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_STEPS", "0")),
        "__LOCAL_ANALYZER_TOOL_TIMEOUT__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_TIMEOUT", "30")),
        "__LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__": repr(os.environ.get("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", "1024")),
        "__LOCAL_ANALYZER_YIELD_SECONDS__": repr(os.environ.get("LOCAL_ANALYZER_YIELD_SECONDS", "60")),
        "__LOCAL_ANALYZER_TEMPERATURE__": repr(os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "0.6")),
        "__LOCAL_ANALYZER_TOP_P__": repr(os.environ.get("LOCAL_ANALYZER_TOP_P", "0.95")),
        "__LOCAL_ANALYZER_TOP_K__": repr(os.environ.get("LOCAL_ANALYZER_TOP_K", "20")),
        "__LOCAL_ANALYZER_ENABLE_THINKING__": repr(os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING", "1")),
        "__MULTIMODAL_CONTEXT__": repr(os.environ.get("MULTIMODAL_CONTEXT", "current_grid")),
        "__MULTIMODAL_UPSCALE__": repr(os.environ.get("MULTIMODAL_UPSCALE", "4")),
        "__VLLM_TENSOR_PARALLEL_SIZE__": repr(int(cfg.tensor_parallel_size)),
        "__WHEELHOUSE_STAMP_TEXT__": repr(cfg.wheelhouse_stamp_text),
        "__HARNESS_ENV__": repr(harness_env()),
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


_DUCK_VLLM_SETUP_SCRIPT = r"""import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WHEELHOUSE_OWNER = __WHEELHOUSE_OWNER__
WHEELHOUSE_SLUG = __WHEELHOUSE_SLUG__
MODEL_OWNER = __MODEL_OWNER__
MODEL_SLUG = __MODEL_SLUG__
SERVED_MODEL_NAME = __SERVED_MODEL_NAME__
VLLM_HOST = '127.0.0.1'
VLLM_PORT = __VLLM_PORT__
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_MAX_MODEL_LEN = __VLLM_MAX_MODEL_LEN__
ANALYZER_CONTEXT_WINDOW = __ANALYZER_CONTEXT_WINDOW__
VLLM_TENSOR_PARALLEL_SIZE = __VLLM_TENSOR_PARALLEL_SIZE__
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
SITE_PACKAGES = WORKING_DIR / 'vllm-site-packages'
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'
INSTALL_STAMP = SITE_PACKAGES / f'.{WHEELHOUSE_SLUG}'
STAMP_TEXT = __WHEELHOUSE_STAMP_TEXT__

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


WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)
MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)


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


def vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(SITE_PACKAGES) if not existing else f'{SITE_PACKAGES}{os.pathsep}{existing}'
    env.update(
        {
            'USE_TF': '0',
            'TRANSFORMERS_NO_TF': '1',
            'TRANSFORMERS_NO_TORCHVISION': '1',
            'VLLM_NO_USAGE_STATS': '1',
        }
    )
    return env


def cached_install_is_usable() -> bool:
    if not INSTALL_STAMP.exists() or INSTALL_STAMP.read_text(encoding='utf-8') != STAMP_TEXT:
        return False
    result = subprocess.run(
        [sys.executable, '-c', "import vllm, torch; print(f'Cached vLLM {vllm.__version__}, torch {torch.__version__}')"],
        env=vllm_env(),
        text=True,
    )
    return result.returncode == 0


def install_vllm_wheelhouse() -> None:
    requirements = WHEELHOUSE / 'requirements.lock'
    if not requirements.exists():
        raise FileNotFoundError(f'Missing wheelhouse lock file: {requirements}')
    if cached_install_is_usable():
        print(f'Using cached vLLM target install at {SITE_PACKAGES}', flush=True)
        return
    shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--no-index',
        '--find-links',
        str(WHEELHOUSE),
        '--requirement',
        str(requirements),
        '--target',
        str(SITE_PACKAGES),
        '--upgrade',
        '--ignore-installed',
        '--only-binary',
        ':all:',
        '--no-compile',
        '--disable-pip-version-check',
        '--no-warn-conflicts',
    ]
    print('Installing vLLM wheelhouse into', SITE_PACKAGES, flush=True)
    subprocess.run(cmd, check=True)
    INSTALL_STAMP.write_text(STAMP_TEXT, encoding='utf-8')


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def tail_server_log(lines: int = 80) -> str:
    if not VLLM_SERVER_LOG.exists():
        return ''
    return '\n'.join(VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])


def wait_for_vllm_server(timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f'{VLLM_BASE_URL}/models'
    while time.monotonic() < deadline:
        if VLLM_SERVER_PID.exists():
            try:
                os.kill(int(VLLM_SERVER_PID.read_text().strip()), 0)
            except OSError as exc:
                raise RuntimeError(f'vLLM server process is not alive: {exc}\n{tail_server_log()}') from exc
        try:
            models = request_json(url, timeout=5)
            print('vLLM server ready:', models, flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f'Timed out waiting for vLLM server at {url}.\nLast server log lines:\n{tail_server_log()}')


def start_vllm_server() -> None:
    install_vllm_wheelhouse()
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open('w', encoding='utf-8')
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        str(MODEL_PATH),
        '--served-model-name',
        SERVED_MODEL_NAME,
        '--host',
        VLLM_HOST,
        '--port',
        str(VLLM_PORT),
        '--tensor-parallel-size',
        str(VLLM_TENSOR_PARALLEL_SIZE),
        '--enable-auto-tool-choice',
        '--tool-call-parser',
        'qwen3_coder',
        '--generation-config',
        'vllm',
        '--enable-prefix-caching',
        '--default-chat-template-kwargs',
        '{"preserve_thinking": true}',
        '--reasoning-parser',
        'qwen3',
        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
    ]
    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=vllm_env(), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    VLLM_SERVER_PID.write_text(str(process.pid), encoding='utf-8')
    wait_for_vllm_server()


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


print(f'vLLM wheelhouse path: {WHEELHOUSE}', flush=True)
print(f'Qwen model path: {MODEL_PATH}', flush=True)
assert_expected_cuda_gpu()
missing = [str(path) for path in (WHEELHOUSE, MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError('Missing attached dataset path(s): ' + ', '.join(missing))
start_vllm_server()
run_vllm_api_smoke_test()
setup_env = {
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'PYTHONPATH': str(SITE_PACKAGES) + os.pathsep + os.environ.get('PYTHONPATH', ''),
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'OPENAI_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': __LOCAL_ANALYZER_APP_NAME__,
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_MAX_OUTPUT': __LOCAL_ANALYZER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_THINKING_TOKEN_BUDGET': __LOCAL_ANALYZER_THINKING_TOKEN_BUDGET__,
    'LOCAL_ANALYZER_TOOL_STEPS': __LOCAL_ANALYZER_TOOL_STEPS__,
    'LOCAL_ANALYZER_TOOL_TIMEOUT': __LOCAL_ANALYZER_TOOL_TIMEOUT__,
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': __LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__,
    'LOCAL_ANALYZER_YIELD_SECONDS': __LOCAL_ANALYZER_YIELD_SECONDS__,
    'LOCAL_ANALYZER_TEMPERATURE': __LOCAL_ANALYZER_TEMPERATURE__,
    'LOCAL_ANALYZER_TOP_P': __LOCAL_ANALYZER_TOP_P__,
    'LOCAL_ANALYZER_TOP_K': __LOCAL_ANALYZER_TOP_K__,
    'LOCAL_ANALYZER_ENABLE_THINKING': __LOCAL_ANALYZER_ENABLE_THINKING__,
    'MULTIMODAL_CONTEXT': __MULTIMODAL_CONTEXT__,
    'MULTIMODAL_UPSCALE': __MULTIMODAL_UPSCALE__,
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
# ARC3_* harness settings from the launcher config's harness_env section.
setup_env.update(__HARNESS_ENV__)
print('Harness env: ' + json.dumps(__HARNESS_ENV__, sort_keys=True), flush=True)
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
pid_path = WORKING_DIR / 'vllm-openai-server.pid'
site_packages = WORKING_DIR / 'vllm-site-packages'
if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding='utf-8').strip())
        print('Stopping vLLM server', flush=True)
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
        print(f'Could not stop vLLM server cleanly: {exc!r}', flush=True)
    pid_path.unlink(missing_ok=True)
shutil.rmtree(site_packages, ignore_errors=True)
print(f'Removed temporary vLLM install at {site_packages}', flush=True)
"""
