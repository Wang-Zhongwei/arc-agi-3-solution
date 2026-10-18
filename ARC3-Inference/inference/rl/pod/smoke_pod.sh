#!/bin/bash
# One-shot LoRA smoke test on a GPU pod: install, fetch the checkpoint, run, collect.
#
# Upload this directory (inference/rl + a transcript) to $WORK and launch once:
#   ssh ... 'cd /workspace/smoke; nohup bash inference/rl/pod/smoke_pod.sh > smoke_pod.log 2>&1 < /dev/null &' < /dev/null
# Every phase is skipped when already done, so a relaunch resumes. The model is
# loaded once by a job server (smoke_lora --serve); more jobs can be dropped into
# $WORK/jobs as NAME.job while it runs (KEEP_SERVER=1 keeps it up after the
# default jobs). Results land in $WORK/results/; $WORK/DONE or $WORK/FAILED marks the end.
set -uo pipefail

WORK=${WORK:-/workspace/smoke}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
TRANSCRIPT=${TRANSCRIPT:-$WORK/game.txt}
LENGTHS=${LENGTHS:-32768o,65536o,100000o}  # o = offload checkpoints to host memory
BACKEND=${BACKEND:-auto}                         # AutoRound kernel backend
KEEP_SERVER=${KEEP_SERVER:-0}
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
if ! python -c "import transformers, peft, auto_round, fla, tilelang; assert transformers.__version__ == '5.20.1'" 2>/dev/null; then
    python -m pip install -q -U pip
    python -m pip install -q "transformers==5.20.1" peft accelerate auto-round kernels flash-linear-attention \
        tilelang hf_transfer pillow numpy torchvision || fail "pip install"
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
rm -f "$JOBS/STOP"
if ! { [ -s server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; }; then
    nohup python -m inference.rl.smoke_lora --model "$MODEL_DIR" --transcript "$TRANSCRIPT" \
        --tools inference/rl/data/python_tool.json --backend "$BACKEND" --serve "$JOBS" --out results/server.json \
        > results/server.log 2>&1 < /dev/null &
    echo $! > server.pid
fi
submit() {  # name, job json; skipped when already queued, running or done
    local name=$1
    if [ -e "$JOBS/$name.job" ] || [ -e "$JOBS/$name.running" ] || [ -e "$JOBS/$name.done" ]; then return 0; fi
    echo "$2" > "$JOBS/$name.tmp" && mv "$JOBS/$name.tmp" "$JOBS/$name.job"
}
submit 01_profile '{"lengths": "16384", "repeats": 1, "profile": true}'
submit 02_base "{\"lengths\": \"$LENGTHS\", \"repeats\": 1}"
submit 03_shared_expert '{"lengths": "32768o", "repeats": 1, "adapter": "shared_expert"}'

log "phase 4: jobs"
while [ ! -e "$JOBS/03_shared_expert.done" ]; do
    kill -0 "$(cat server.pid)" 2>/dev/null || fail "model server exited, see results/server.log"
    sleep 20
done
cp "$JOBS"/*.result.json results/
if [ "$KEEP_SERVER" != 1 ]; then
    touch "$JOBS/STOP"
fi
log "all done (server $( [ "$KEEP_SERVER" = 1 ] && echo "still serving $JOBS" || echo stopping))"
date -u > DONE
