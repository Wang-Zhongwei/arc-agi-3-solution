#!/bin/bash
# One-shot LoRA smoke test on a GPU pod: install, fetch the checkpoint, run, collect.
#
# Upload this directory (inference/rl + a transcript) to $WORK and launch once:
#   ssh ... 'cd /workspace/smoke; nohup bash inference/rl/pod/smoke_pod.sh > smoke_pod.log 2>&1 < /dev/null &' < /dev/null
# Every phase is skipped when already done, so a relaunch resumes. The model is
# loaded once by a job server (smoke_lora --serve); more jobs can be dropped into
# $WORK/jobs as NAME.job while it runs (KEEP_SERVER=1 keeps it up after the
# default jobs). JOBS_FILE replaces the default jobs: one "NAME JSON" per line, run in
# order. Results land in $WORK/results/; $WORK/DONE or $WORK/FAILED marks the end.
set -uo pipefail

WORK=${WORK:-/workspace/smoke}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
TRANSCRIPT=${TRANSCRIPT:-$WORK/game.txt}
LENGTHS=${LENGTHS:-32768o,65536o}  # o = offload checkpoints to host memory
BACKEND=${BACKEND:-auto}                         # AutoRound kernel backend
KEEP_SERVER=${KEEP_SERVER:-0}
JOBS_FILE=${JOBS_FILE:-}                         # "NAME JSON" per line; default: the two smoke jobs below
VENV=${VENV:-/workspace/venv}
JOBS=$WORK/jobs
cd "$WORK"
mkdir -p results "$JOBS"
rm -f DONE FAILED
echo $$ > run.pid  # stop with: kill $(cat run.pid); never pkill -f, which also matches the ssh command
# the loader's many small copies fragment the caching allocator (OOM at 67 GiB used, 73 GiB reserved-free)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# fla's GDN backward needs tilelang on Hopper with torch 2.8's Triton 3.4 (fla #640), and tilelang JIT-compiles with nvcc
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; echo "$*" > FAILED; exit 1; }

log "phase 1: python packages"
# the image's system Python is externally managed (PEP 668): a venv that still sees its torch
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || fail "venv"
export PATH="$VENV/bin:$PATH"
hopper=$(python -c "import torch; print(int(torch.cuda.get_device_capability()[0] == 9))" 2>/dev/null || echo 0)
need="import transformers, peft, auto_round, fla$([ "$hopper" = 1 ] && echo ", tilelang"); assert transformers.__version__ == '5.20.1'"
if ! python -c "$need" 2>/dev/null; then
    python -m pip install -q -U pip
    # tilelang only for fla's GDN backward on Hopper (fla #640); WHEELS: a directory of wheels for hosts without PyPI
    python -m pip install -q ${WHEELS:+--no-index --find-links "$WHEELS"} "transformers==5.20.1" peft accelerate \
        auto-round flash-linear-attention hf_transfer pillow numpy torchvision $([ "$hopper" = 1 ] && echo tilelang) \
        || fail "pip install"
fi
if [ -d tests ] && [ ! -e results/pytest.log ]; then  # the CUDA-only tests before the 5-minute load
    # only those: with fla installed, transformers sends the CPU tiny-model tests' GDN layers to
    # fla's Triton kernels, which reject CPU tensors. The CPU tests run locally and in gpu_tests.sh
    python -c "import pytest" 2>/dev/null || python -m pip install -q pytest
    PYTHONPATH=. python -m pytest -q -rs tests/test_host_offload.py -k cuda > results/pytest.log 2>&1 \
        || fail "unit tests, see results/pytest.log"
    log "unit tests: $(tail -1 results/pytest.log)"
fi
python - <<'PY' | tee results/env.txt
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)
for m in ("auto_round", "fla", "tilelang", "kernels", "causal_conv1d"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "not importable:", e)
PY

log "phase 2: checkpoint $MODEL_REPO"
start=$(date +%s)
# 52 s on US-NC-1 but 665 s on US-CO-1: pin the pod's data center or use a network volume
hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --exclude "*.bak" > results/download.log 2>&1 || fail "download"
log "checkpoint ready in $(( $(date +%s) - start ))s: $(du -sh "$MODEL_DIR" | cut -f1)"

log "phase 3: model server (loads once, ~5 min)"
# no --use-kernels: the hub kernels it pulls (activation, mamba-ssm) are inference kernels, and kernels 0.17
# refuses them when the Hub's publisher-trust lookup fails; torch's short-conv fallback is fine for training
rm -f "$JOBS/STOP"
if ! { [ -s server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; }; then
    nohup python -m inference.rl.smoke_lora --model "$MODEL_DIR" --transcript "$TRANSCRIPT" \
        --tools inference/rl/data/python_tool.json --backend "$BACKEND" --serve "$JOBS" --out results/server.json ${SERVER_ARGS:-} \
        > results/server.log 2>&1 < /dev/null &
    echo $! > server.pid
fi
submit() {  # name, job json; skipped when already queued, running or done
    local name=$1
    if [ -e "$JOBS/$name.job" ] || [ -e "$JOBS/$name.running" ] || [ -e "$JOBS/$name.done" ]; then return 0; fi
    echo "$2" > "$JOBS/$name.tmp" && mv "$JOBS/$name.tmp" "$JOBS/$name.job"
}
if [ -n "$JOBS_FILE" ]; then
    while read -r name job; do
        [ -n "$name" ] && [ "${name#\#}" = "$name" ] && submit "$name" "$job" && last=$name
    done < "$JOBS_FILE"
else
    # the longest real request first (cd82, ~110K tokens): fail fast if it does not fit
    submit 01_longest '{"lengths": "130000o", "repeats": 1, "transcript": "game_long.txt"}'
    submit 02_speed "{\"lengths\": \"$LENGTHS\", \"repeats\": 1}"
    last=02_speed
fi

log "phase 4: jobs (GPU utilization and power sampled every second into results/gpu.csv)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used --format=csv,noheader \
    -l 1 > results/gpu.csv 2>/dev/null &
sampler=$!
while [ ! -e "$JOBS/$last.done" ]; do
    kill -0 "$(cat server.pid)" 2>/dev/null || { kill $sampler; fail "model server exited, see results/server.log"; }
    sleep 20
done
kill $sampler
cp "$JOBS"/*.result.json results/
if [ "$KEEP_SERVER" != 1 ]; then
    touch "$JOBS/STOP"
fi
log "all done (server $( [ "$KEEP_SERVER" = 1 ] && echo "still serving $JOBS" || echo stopping))"
date -u > DONE
