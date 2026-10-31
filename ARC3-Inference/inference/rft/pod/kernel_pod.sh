#!/bin/bash
# Kernel pod (no checkpoint): CUDA kernel tests, then real-size single-layer benchmarks
# (bench_layer.py) for each variant in $BENCHES. Upload inference/ and tests/ to $WORK, then:
#   ssh ... 'cd /workspace/rl; nohup bash inference/rft/pod/kernel_pod.sh > kernel_pod.log 2>&1 < /dev/null &' < /dev/null
# $WORK/DONE or $WORK/FAILED marks the end; results land in $WORK/results/.
set -uo pipefail

WORK=${WORK:-/workspace/rl}
VENV=${VENV:-/workspace/venv}
TOKENS=${TOKENS:-65536}
# NAME:ARGS pairs, ARGS with commas for spaces
BENCHES=${BENCHES:-"chunked:--qsa,chunked fused:--qsa,fused"}
cd "$WORK"
mkdir -p results
rm -f DONE FAILED
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; echo "$*" > FAILED; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH

log "phase 1: python packages"
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || fail "venv"
export PATH="$VENV/bin:$PATH"
if ! python -c "import transformers, peft, auto_round, fla, pytest; assert transformers.__version__ == '5.20.1'" 2>/dev/null; then
    python -m pip install -q -U pip
    python -m pip install -q -r inference/rft/pod/requirements-pod.txt || fail "pip install"  # tested versions
    if python -c "import torch, sys; sys.exit(torch.cuda.get_device_capability()[0] != 9)"; then
        python -m pip install -q tilelang==0.1.15 || fail "tilelang"  # fla's GDN backward on Hopper (fla #640)
    fi
fi
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name(0))" \
    | tee results/env.txt || fail "no GPU"

log "phase 2: CUDA tests (kernels, compiled layer)"
PYTHONPATH=. python -m pytest -q -rs tests/test_qsa_kernel.py tests/test_compile_hc.py tests/test_qsa_fused_model.py tests/test_grouped_gemm.py tests/test_prefix_stop_grad_cuda.py > results/pytest_kernel.log 2>&1
status=$?
log "pytest exit $status: $(tail -1 results/pytest_kernel.log)"
[ $status -eq 0 ] || grep -E "^(FAILED|E  )" results/pytest_kernel.log | head -20  # benchmarks still run

log "phase 3: single-layer benchmarks at $TOKENS tokens"
for bench in $BENCHES; do
    name=${bench%%:*}
    args=${bench#*:}
    log "bench $name: ${args//,/ }"
    PYTHONPATH=. python inference/rft/bench_layer.py --tokens "$TOKENS" --profile --out "results/bench_$name.json" \
        ${args//,/ } > "results/bench_$name.log" 2>&1 || log "bench $name failed: $(tail -3 results/bench_$name.log)"
    grep -v '^\s*$' "results/bench_$name.log" | grep '^{' || true
done

if [ $status -eq 0 ]; then date -u > DONE; else echo "pytest exit $status" > FAILED; fi
log "all done"
