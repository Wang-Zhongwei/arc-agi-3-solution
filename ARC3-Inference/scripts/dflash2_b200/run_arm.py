"""One arm on one B200: the Kaggle setup step as rendered for that run, then the replay bench.

    python3 run_arm.py ARM OUT_DIR BENCH_ARGS...

setup_<ARM>.py holds the rendered Kaggle setup script up to its main flow (make_arms.py). This runs
the same main flow minus the RTX PRO 6000 GPU-name check: unpack Lord's image layers, overlay e3a6fe0d,
PLE fixes, the DFlash patches (DFlash arms), the exact server command under the watchdog, and the
smoke test. The only runtime change is TORCH_CUDA_ARCH_LIST 10.0 for JIT kernels on the B200.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
arm, out = sys.argv[1], Path(sys.argv[2])
bench_args = sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
os.environ['TAAF_KAGGLE_WORKING_DIR'] = str(out)
os.environ['TAAF_KAGGLE_SETUP_ENV'] = str(out / 'taaf_setup_env.json')

namespace = {'__name__': f'setup_{arm}'}
exec(compile((HERE / f'setup_{arm}.py').read_text(), f'setup_{arm}.py', 'exec'), namespace)
kaggle_env = namespace['build_image_runtime_env']


def build_b200_env(manifest):
    env = kaggle_env(manifest)
    env['TORCH_CUDA_ARCH_LIST'] = '10.0'
    return env


namespace['build_image_runtime_env'] = build_b200_env
started = time.time()
print(f'[{arm}] runtime dataset {namespace["RUNTIME_DATASET"]}, model {namespace["MODEL_PATH"]}', flush=True)
try:
    namespace['assemble_ple_fp8_target']()
    namespace['start_vllm_server']()
    print(f'[{arm}] server ready after {time.time() - started:.0f} s', flush=True)
    namespace['print_vllm_backend_summary']()
    namespace['run_vllm_api_smoke_test']()
    url = f'http://127.0.0.1:{namespace["VLLM_PORT"]}/v1'
    bench = subprocess.run(
        [sys.executable, str(HERE / 'bench.py'), '--url', url, '--out', str(out / 'bench'), '--label', arm, *bench_args]
    )
    print(f'[{arm}] bench exit {bench.returncode} after {time.time() - started:.0f} s', flush=True)
finally:
    (out / 'arm_done.json').write_text(json.dumps({'arm': arm, 'seconds': round(time.time() - started)}))
    # the watchdog's SIGTERM handler stops the server it supervises; the server pid is the fallback
    for pid_file in (namespace['VLLM_WATCHDOG_PID'], namespace['VLLM_SERVER_PID']):
        try:
            os.kill(int(Path(pid_file).read_text().strip()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        time.sleep(10)
