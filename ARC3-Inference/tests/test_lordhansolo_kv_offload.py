"""int4-kv v4: the CPU KV tier's launcher, reload probe and stats, and the SM120 GEMM row extension."""
import ast
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from inference.framework import kaggle
from inference.framework.lordhansolo import kaggle as lord

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/int4-kv/v4.json"
OFFLOAD = Path(lord.__file__).resolve().parent / "kv_offload"
V3_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 100]
GEMM_SOURCE = '''GEMM_ROWS = (40, 44, 48, 52, 56, 64)
LOG = "ARC3 rc3 SM120 GEMM enabled for target M=40/44/48/52/56/64; first decoder"
'''
GEMM_PATH = "dist/vllm/models/qwen4_exp/nvidia/arc3_sm120_gemm.py"


def _render(monkeypatch, tmp_path, config):
    monkeypatch.setenv("ARC3_KAGGLE_JSON", json.dumps(config["kaggle"]))
    monkeypatch.setenv("ARC3_HARNESS_ENV_JSON", json.dumps(config["harness_env"]))
    monkeypatch.setenv("LOCAL_ANALYZER_CONTEXT_WINDOW", str(config["shared"]["context_window"]))
    monkeypatch.setenv("LOCAL_ANALYZER_MAX_OUTPUT", "12288")
    monkeypatch.setenv("TAAF_KAGGLE_WORKING_DIR", str(tmp_path))
    monkeypatch.setenv("TAAF_KAGGLE_BUNDLE_DIR", str(tmp_path))
    monkeypatch.setenv("TAAF_KAGGLE_SETUP_ENV", str(tmp_path / "setup_env.json"))
    monkeypatch.setenv("TAAF_RUN_AS_SUBMISSION", "0")
    source_root = tmp_path / "src"
    source_root.mkdir(exist_ok=True)
    if not (source_root / "ARC3-Inference").exists():
        (source_root / "ARC3-Inference").symlink_to(ROOT)
    return kaggle.lordhansolo_setup_command()


@pytest.fixture
def setup(monkeypatch, tmp_path):
    command = _render(monkeypatch, tmp_path, json.loads(CONFIG.read_text()))
    source = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
    ast.parse(source)
    namespace = {}
    exec(source.split("print(f'vLLM runtime dataset path:", 1)[0], namespace)
    return command, source, namespace


def test_config_shape():
    config = json.loads(CONFIG.read_text())
    vllm, env = config["kaggle"]["vllm"], config["harness_env"]
    step = vllm["max_num_seqs"] * (1 + vllm["mtp_tokens"])
    assert step in vllm["cudagraph_capture_sizes"]
    assert vllm["kv_offload_fallback_seqs"] * (1 + vllm["mtp_tokens"]) in vllm["cudagraph_capture_sizes"]
    assert env["ARC3_MAX_ACTIVE_STREAMS"] == vllm["max_num_seqs"]
    assert env["ARC3_TOKEN_EXACT_HISTORY"] == "1" and vllm["token_exact_history"] is True
    assert env["ARC3_NORMALIZE_HISTORY"] == "0"
    assert 1664 % vllm["prefix_match_unit"] == 0


def test_setup_command_fits_one_argument(setup):
    # Linux caps one argument at 131,072 bytes, and the notebook adds a few hundred; keep 4 KiB spare.
    assert len(setup[0].encode()) < 124 * 1024


def test_server_command_has_the_tier_once(setup):
    cmd = setup[2]["build_vllm_server_command"]()
    assert cmd.count("--kv-offloading-size") == 1
    assert cmd[cmd.index("--kv-offloading-size") + 1] == "40.0"
    assert cmd[cmd.index("--kv-offloading-backend") + 1] == "native"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "27"


def test_fallback_command_is_v3_without_the_tier(setup):
    namespace = setup[2]
    fallback = namespace["without_kv_offload"](namespace["build_vllm_server_command"]())
    assert not any("offloading" in arg for arg in fallback)
    assert fallback[fallback.index("--max-num-seqs") + 1] == "25"
    assert fallback[fallback.index("--max-cudagraph-capture-size") + 1] == "100"
    assert json.loads(fallback[fallback.index("--compilation-config") + 1])["cudagraph_capture_sizes"] == V3_SIZES


def test_setup_order_and_late_environment(setup):
    source = setup[1]
    start = source.index("def start_vllm_server() -> None:")
    positions = [source.index(step, start) for step in (
        "install_int4_kv(manifest)", "install_token_exact_history(manifest)", "install_kv_offload(manifest)",
        "check_int4_kv_kernels(server_env)", "extend_sm120_gemm_rows(manifest, server_env)", "launch_vllm_watchdog(cmd",
    )]
    assert positions == sorted(positions)
    assert source.index("    run_kv_offload_probe()\n    try:\n        run_token_exact_probe()") > 0
    assert source.index("relaunch_without_kv_offload(f'smoke test failed") < source.index("    run_kv_offload_probe()\n")
    # The fallback's stream count must beat the config's value in the handoff.
    assert source.index("setup_env.update(") < source.index("existing_setup_env.update(setup_env)") < source.index("existing_setup_env.update(LATE_SETUP_ENV)")


def test_watchdog_switches_command_on_first_restart():
    ast.parse(lord._DUCK_VLLM_WATCHDOG_SCRIPT)
    assert "server_command = config['fallback_command']" in lord._DUCK_VLLM_WATCHDOG_SCRIPT


def test_tier_requires_a_fallback(monkeypatch, tmp_path):
    config = json.loads(CONFIG.read_text())
    config["kaggle"]["vllm"]["kv_offload_fallback_seqs"] = 0
    with pytest.raises(ValueError, match="kv_offload_fallback_seqs"):
        _render(monkeypatch, tmp_path, config)


def test_candidate_rows_must_have_graphs(monkeypatch, tmp_path):
    config = json.loads(CONFIG.read_text())
    config["kaggle"]["vllm"]["sm120_gemm_candidate_rows"] = [92]
    with pytest.raises(ValueError, match="sm120_gemm_candidate_rows"):
        _render(monkeypatch, tmp_path, config)


@pytest.mark.parametrize("harness", ["0", "1"])
def test_token_exact_flags_must_agree(monkeypatch, tmp_path, harness):
    config = json.loads(CONFIG.read_text())
    config["harness_env"]["ARC3_TOKEN_EXACT_HISTORY"] = harness
    config["kaggle"]["vllm"]["token_exact_history"] = harness == "0"
    with pytest.raises(ValueError, match="must agree"):
        _render(monkeypatch, tmp_path, config)


def test_runtime_patches_install_once(setup, tmp_path):
    namespace = setup[2]
    dist = tmp_path / "rt" / "dist"
    (dist / "vllm").mkdir(parents=True)
    patches = json.loads((OFFLOAD / "runtime_patches.json").read_text())
    for patch in patches:
        path = dist / patch["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(edit["anchor"] for edit in patch["replacements"]))
    namespace["IMAGE_RUNTIME_ROOT"] = tmp_path / "rt"
    namespace["KV_OFFLOAD_DIR"] = OFFLOAD
    namespace["install_kv_offload"]({"dist_packages": "dist"})
    assert (dist / "vllm" / "arc3_offload_stats.py").is_file()
    for patch in patches:
        text = (dist / patch["path"]).read_text()
        assert all(edit["replacement"] in text for edit in patch["replacements"])
    with pytest.raises(RuntimeError, match="anchor appears 0 times"):
        namespace["install_kv_offload"]({"dist_packages": "dist"})


class _Result:
    def __init__(self, returncode=0, stdout=""):
        self.returncode, self.stdout = returncode, stdout


def _extend(namespace, tmp_path, monkeypatch, run, source=GEMM_SOURCE):
    path = tmp_path / "rt" / GEMM_PATH
    path.parent.mkdir(parents=True)
    path.write_text(source)
    namespace["IMAGE_RUNTIME_ROOT"] = tmp_path / "rt"
    monkeypatch.setattr(subprocess, "run", run)
    namespace["extend_sm120_gemm_rows"]({"dist_packages": "dist"}, {})
    return path.read_text()


def test_gemm_rows_take_only_the_accepted(setup, tmp_path, monkeypatch):
    text = _extend(setup[2], tmp_path, monkeypatch, lambda *a, **k: _Result(0, "SM120_GEMM_ACCEPTED [108, 96]\n"))
    assert "GEMM_ROWS = (40, 44, 48, 52, 56, 64, 96, 108)\n" in text
    assert "target M=40/44/48/52/56/64/96/108;" in text
    compile(text, "gemm", "exec")


@pytest.mark.parametrize("result", [_Result(0, "SM120_GEMM_ACCEPTED []\n"), _Result(1, "SM120_GEMM_ACCEPTED [108]\n"), _Result(1, "ImportError\n")])
def test_gemm_rows_unchanged_on_any_rejection(setup, tmp_path, monkeypatch, result):
    assert _extend(setup[2], tmp_path, monkeypatch, lambda *a, **k: result) == GEMM_SOURCE


def test_gemm_rows_unchanged_on_timeout(setup, tmp_path, monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("preflight", 600)
    assert _extend(setup[2], tmp_path, monkeypatch, run) == GEMM_SOURCE


def test_gemm_rows_skip_the_preflight_when_the_anchor_moved(setup, tmp_path, monkeypatch):
    def run(*args, **kwargs):
        raise AssertionError("no preflight")
    moved = "GEMM_ROWS = (1, 2)\n"
    assert _extend(setup[2], tmp_path, monkeypatch, run, moved) == moved


def test_probe_timeout_falls_back(setup, monkeypatch):
    namespace = setup[2]
    reasons = []
    namespace["relaunch_without_kv_offload"] = reasons.append
    namespace["KV_OFFLOAD_RELAUNCH"]["fallback"] = ["x"]
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 900)
    monkeypatch.setattr(subprocess, "run", run)
    namespace["run_kv_offload_probe"]()
    assert reasons and "timed out" in reasons[0]


@pytest.mark.parametrize("code,falls_back", [(0, False), (3, False), (1, True)])
def test_probe_exit_codes(setup, monkeypatch, code, falls_back):
    namespace = setup[2]
    reasons = []
    namespace["relaunch_without_kv_offload"] = reasons.append
    namespace["KV_OFFLOAD_RELAUNCH"]["fallback"] = ["x"]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result(code, "out"))
    namespace["run_kv_offload_probe"]()
    assert bool(reasons) == falls_back


def test_probe_skipped_without_a_tier(setup, monkeypatch):
    def run(*args, **kwargs):
        raise AssertionError("no probe")
    monkeypatch.setattr(subprocess, "run", run)
    setup[2]["KV_OFFLOAD_RELAUNCH"].clear()
    setup[2]["run_kv_offload_probe"]()


class _FakeServer:
    """A vLLM stand-in: the request after the prefix-cache reset gets `reload`, others a GPU hit."""

    def __init__(self, reload_top=None, reload_cached=11_800, reset_results=(True,), reload_status=200, gpu_cached=11_800, status_before_reset=200):
        fake = self
        self.was_reset = False
        self.resets = list(reset_results)
        self.queries = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.startswith("/reset_prefix_cache"):
                    fake.queries.append(self.path)
                    fake.was_reset = fake.resets.pop(0) if fake.resets else True
                    return self._send(200, {"success": fake.was_reset})
                if not fake.was_reset and status_before_reset != 200:
                    return self._send(status_before_reset, {})
                if fake.was_reset and reload_status != 200:
                    return self._send(reload_status, {})
                cached = reload_cached if fake.was_reset else gpu_cached
                top = (reload_top if fake.was_reset and reload_top else None) or [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -2.3}]
                self._send(200, {
                    "choices": [{"token_ids": [1], "logprobs": {"content": [{"token": top[0]["token"], "top_logprobs": top}]}}],
                    "usage": {"prompt_tokens": 12_000, "prompt_tokens_details": {"cached_tokens": cached}},
                })

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()

    def run(self):
        url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return subprocess.run(
            [sys.executable, "-I", str(OFFLOAD / "probe.py"), url, "m", "data:image/png;base64,AA"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
        )


def test_probe_passes_on_identical_reload():
    with _FakeServer() as fake:
        result = fake.run()
    assert result.returncode == 0 and "KV_OFFLOAD_PROBE_PASSED" in result.stdout
    assert "reset_external=false" in fake.queries[0] and "reset_running_requests=false" in fake.queries[0]


def test_probe_fails_on_a_different_distribution():
    with _FakeServer(reload_top=[{"token": "b", "logprob": -0.1}, {"token": "a", "logprob": -2.3}]) as fake:
        result = fake.run()
    assert result.returncode == 1 and "next-token distributions differ" in result.stdout


def test_probe_passes_a_swap_in_the_tail():
    # int4-kv v4's real values: a healthy reload that the old set comparison failed as an infinite gap.
    def top(items):
        return [{"token": token, "logprob": logprob} for token, logprob in items]
    v4_gpu = top([("We", -0.014230557717382908), ("The", -4.514230728149414), ("Okay", -6.514230728149414), ("Got", -7.514230728149414), ("I", -8.139230728149414)])
    v4_tier = top([("We", -0.012726283632218838), ("The", -4.762726306915283), ("Okay", -6.137726306915283), ("Got", -7.012726306915283), ("User", -7.887726306915283)])
    spec = __import__("importlib.util").util.spec_from_file_location("probe", OFFLOAD / "probe.py")
    probe = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    def served(items):
        return {"tokens": ["We"], "ids": [1], "prompt_tokens": 12402, "cached_tokens": 12160,
                "steps": [{item["token"]: probe.math.exp(item["logprob"]) for item in items}]}
    assert probe.judge(served(v4_tier), served(v4_gpu), served(v4_gpu)) is None
    assert probe.distance(served(v4_tier), served(v4_gpu)) < 0.005


def test_probe_fails_on_a_shallow_tier_hit():
    with _FakeServer(reload_cached=3_328) as fake:
        result = fake.run()
    assert result.returncode == 1 and "too shallow" in result.stdout


def test_probe_fails_when_the_reload_request_errors():
    with _FakeServer(reload_status=500) as fake:
        result = fake.run()
    assert result.returncode == 1 and "reload request failed" in result.stdout


def test_probe_retries_a_held_reset():
    with _FakeServer(reset_results=(False, True)) as fake:
        result = fake.run()
    assert result.returncode == 0 and len(fake.queries) == 2


def test_probe_fails_when_the_server_errors_before_the_reload():
    with _FakeServer(status_before_reset=500) as fake:
        result = fake.run()
    assert result.returncode == 1 and "before the reload" in result.stdout


def test_probe_is_inconclusive_when_a_route_is_missing():
    with _FakeServer(status_before_reset=404) as fake:
        result = fake.run()
    assert result.returncode == 3 and "KV_OFFLOAD_PROBE_INCONCLUSIVE" in result.stdout


def test_copy_check_timeout_means_no_tier(setup, monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("check", 120)
    monkeypatch.setattr(subprocess, "run", run)
    assert setup[2]["kv_offload_copy_available"]({}) is False


def test_probe_is_inconclusive_when_gpu_hits_are_shallow():
    with _FakeServer(gpu_cached=0) as fake:
        result = fake.run()
    assert result.returncode == 3 and "KV_OFFLOAD_PROBE_INCONCLUSIVE" in result.stdout


def test_stats_line(monkeypatch, capsys):
    spec = __import__("importlib.util").util.spec_from_file_location("arc3_offload_stats", OFFLOAD / "arc3_offload_stats.py")
    stats = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    import dataclasses

    @dataclasses.dataclass
    class Boundary:
        stored: int = 7

    class Pool:
        def get_num_free_blocks(self):
            return 11

    manager = type("Manager", (), {"cpu_block_pool": Pool(), "boundary_store_stats": Boundary()})()
    clock = [100.0]
    monkeypatch.setattr(stats.time, "monotonic", lambda: clock[0])
    stats._last_report = 100.0
    stats.on_step(manager, [1, 2, 3], ["r"], [4])
    clock[0] = 100.5
    stats.on_load_done("r")
    assert capsys.readouterr().out == ""
    clock[0] = 161.0
    stats.on_step(manager, [], [], [5, 6])
    line = capsys.readouterr().out
    assert "loads=1 load_blocks=3 stored_blocks=3 free_cpu_blocks=11 load_ready_ms_p50=500 load_ready_ms_p95=500" in line
    assert "'stored': 7" in line
