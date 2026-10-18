#!/bin/bash
# One-shot LoRA smoke test on a GPU pod: install, fetch the checkpoint, run, collect.
#
# Upload this directory (inference/rl + a transcript) to $WORK and launch once:
#   nohup bash inference/rl/pod/smoke_pod.sh > smoke_pod.log 2>&1 &
# Every phase is skipped when already done, so a relaunch resumes. Results land
# in $WORK/results/; $WORK/DONE or $WORK/FAILED marks the end.
set -uo pipefail

WORK=${WORK:-/workspace/smoke}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
TRANSCRIPT=${TRANSCRIPT:-$WORK/game.txt}
LENGTHS=${LENGTHS:-32768,32768o,65536o,100000o}  # o = offload checkpoints to host memory
cd "$WORK"
mkdir -p results
rm -f DONE FAILED
echo $$ > run.pid  # stop with: kill $(cat run.pid); never pkill -f, which also matches the ssh command
VENV=${VENV:-/workspace/venv}
# the loader's many small copies fragment the caching allocator (OOM at 67 GiB used, 73 GiB reserved-free)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; echo "$*" > FAILED; exit 1; }

log "phase 1: python packages"
# the image's system Python is externally managed (PEP 668): a venv that still sees its torch
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || fail "venv"
export PATH="$VENV/bin:$PATH"
if ! python -c "import transformers, peft, auto_round, fla; assert transformers.__version__ == '5.20.1'" 2>/dev/null; then
    python -m pip install -q -U pip
    python -m pip install -q "transformers==5.20.1" peft accelerate auto-round kernels flash-linear-attention \
        hf_transfer pillow numpy torchvision || fail "pip install"
fi
python - <<'PY' | tee results/env.txt
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)
for m in ("auto_round", "fla", "kernels", "causal_conv1d"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "not importable:", e)
PY

log "phase 2: checkpoint $MODEL_REPO"
start=$(date +%s)
hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --exclude "*.bak" > results/download.log 2>&1 || fail "download"
log "checkpoint ready in $(( $(date +%s) - start ))s: $(du -sh "$MODEL_DIR" | cut -f1)"

log "phase 3: smoke runs"
run() {  # name, extra args
    local name=$1; shift
    if [ -s "results/$name.json" ] && python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if 'after_optimizer_step_gb' in d else 1)" "results/$name.json"; then
        log "$name already complete"; return 0
    fi
    log "run $name: $*"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader > "results/$name.gpu_before.txt"
    python -m inference.rl.smoke_lora --model "$MODEL_DIR" --transcript "$TRANSCRIPT" \
        --tools inference/rl/data/python_tool.json --lengths "$LENGTHS" --out "results/$name.json" "$@" \
        > "results/$name.log" 2>&1 < /dev/null
    local status=$?
    log "$name exited $status"; tail -5 "results/$name.log"
    return $status
}
# default kernels first; if loading or the backward fails, retry on AutoRound's plain torch backend
run auto || run torch_backend --backend torch || fail "both backends failed, see results/*.log"
# shared expert LoRA on top, if the base run worked
LENGTHS=100000o run shared_expert --shared-expert || log "shared_expert variant failed (base result still valid)"

log "all done"
date -u > DONE
