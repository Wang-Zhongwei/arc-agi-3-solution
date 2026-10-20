#!/bin/bash
# One-shot RFT training on an H200: network check, packages, checkpoint, the CUDA tests that decide the training
# flags (a failed test switches off only its feature, as make_jobs.py does), train_rft (restarted from `latest`
# if it dies), then FULL_DONE. Nothing waits for a person. The pod stops itself (POD_END=delete: deletes) MAX_MINUTES after start, or
# GRACE_MINUTES after the end; the local watcher (watch_pod.sh) normally pulls the results and stops it first.
#
# Upload inference/, tests/ and data/ (rft_data's output) to $WORK, then:
#   ssh ... 'cd /workspace/rft; nohup bash inference/rl/pod/train_pod.sh > train_pod.log 2>&1 < /dev/null &' < /dev/null
# TRAIN_ARGS adds train_rft arguments (a rehearsal: "--epochs 1 --limit 4 --batch 2").
set -uo pipefail

WORK=${WORK:-/workspace/rft}
VENV=${VENV:-/workspace/venv}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
MAX_MINUTES=${MAX_MINUTES:-480}
TRAIN_HOURS=${TRAIN_HOURS:-7}  # train_rft saves and stops after this long
GRACE_MINUTES=${GRACE_MINUTES:-20}
POD_END=${POD_END:-stop}  # what ends the pod (timers, abort): stop (GPU billing ends, the pod and its volume stay) or delete
MIN_HF_MBPS=${MIN_HF_MBPS:-40}
TRAIN_ARGS=${TRAIN_ARGS:-}
cd "$WORK"
mkdir -p results train
rm -f FULL_DONE ABORTED
# an ssh session lacks the container's RUNPOD_* variables (pod id and a pod-scoped API key for runpodctl)
for var in RUNPOD_POD_ID RUNPOD_API_KEY; do
    [ -n "${!var:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$var=")" 2>/dev/null || true
done
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
self_stop() {
    log "ending this pod ($1): $POD_END"
    if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1  # it reads a config file, not the env
        if [ "$POD_END" = delete ]; then
            runpodctl remove pod "$RUNPOD_POD_ID" || log "runpodctl failed"
        else
            runpodctl stop pod "$RUNPOD_POD_ID" || log "runpodctl failed"
        fi
    else
        log "no runpodctl or RUNPOD_POD_ID: stopping the processes only"
        pkill -f "inference.rl.train_rft" || true
    fi
}
abort() { log "ABORTED: $*"; echo "$*" > ABORTED; self_stop "abort"; exit 1; }
( sleep $((MAX_MINUTES * 60)); echo "time limit" > ABORTED; self_stop "time limit ${MAX_MINUTES} min" ) &
echo $! > deadman.pid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH
[ -s data/train.jsonl ] || abort "no data/train.jsonl"

log "phase 0: network"
shard=$(python3 - <<'PY'
import json, urllib.request
req = urllib.request.Request("https://huggingface.co/api/models/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound",
                             headers={"User-Agent": "Mozilla/5.0"})
print(next(s["rfilename"] for s in json.load(urllib.request.urlopen(req, timeout=20))["siblings"]
           if s["rfilename"].endswith(".safetensors")))
PY
) || abort "Hugging Face API unreachable"
hf_bps=$(curl -sSL -m 12 -o /dev/null -w "%{speed_download}" "https://huggingface.co/$MODEL_REPO/resolve/main/$shard" || true)
hf_mbps=$(python3 -c "print(int(float('${hf_bps:-0}') / 1e6))")
pypi=$(curl -sS -m 8 -o /dev/null -w "%{http_code}" https://pypi.org/simple/transformers/ || echo 000)
log "HF ${hf_mbps} MB/s, PyPI HTTP ${pypi}, $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), host RAM $(free -g | awk '/Mem:/{print $2}') GB"
[ "$hf_mbps" -ge "$MIN_HF_MBPS" ] || abort "Hugging Face at ${hf_mbps} MB/s (< ${MIN_HF_MBPS})"
if [ "$pypi" != "200" ] && [ ! -d wheels ]; then abort "no PyPI and no wheels/"; fi

log "phase 1: packages"
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || abort "venv"
export PATH="$VENV/bin:$PATH"
hopper=$(python -c "import torch; print(int(torch.cuda.get_device_capability()[0] == 9))" 2>/dev/null || echo 0)
pkgs="transformers==5.20.1 peft accelerate auto-round flash-linear-attention hf_transfer pytest pillow numpy torchvision"
[ "$hopper" = 1 ] && pkgs="$pkgs tilelang"
if [ "$pypi" = "200" ]; then
    python -m pip install -q $pkgs || abort "pip install"
else
    python -m pip install -q --no-index --no-deps wheels/*.whl || abort "wheel install"
fi
python -c "import torch, triton, transformers, peft, fla; print('torch', torch.__version__, 'triton', triton.__version__, transformers.__version__, 'peft', peft.__version__, torch.cuda.get_device_name(0))" | tee results/env.txt || abort "imports"
# fla 0.5.2 with Triton 3.4 on sm100 (B200): the GDN backward's pipelined configs fail to compile (see full_run.sh)
python - <<'PY' | tee -a results/env.txt
import pathlib, torch
if torch.cuda.get_device_capability()[0] == 10:
    import fla.ops.common.chunk_delta_h as m
    path = pathlib.Path(m.__file__)
    old = "for num_stages in ([2, 3, 4] if check_shared_mem('ampere') else [1])"
    new = "for num_stages in ([2, 3, 4] if check_shared_mem('ampere') and not IS_NVIDIA_BLACKWELL else [1])"
    text = path.read_text()
    print("fla sm100 patch:", "applied" if text.count(old) == 1 else "not found (fla changed?)")
    path.write_text(text.replace(old, new))
PY

log "phase 1b: checkpoint download in the background"
HF_HUB_ENABLE_HF_TRANSFER=1 nohup hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --exclude "*.bak" > results/download.log 2>&1 < /dev/null &
download=$!

log "phase 2: CUDA tests"
runtest() {  # name, pytest args
    local name=$1; shift
    timeout 1200 env PYTHONPATH=.:tests python -m pytest -q -p no:logging "$@" > "results/pytest_$name.log" 2>&1
    echo $? > "results/pytest_$name.exit"
    log "  $name: exit $(cat results/pytest_$name.exit), $(tail -1 results/pytest_$name.log)"
}
passed() { [ "$(cat "results/pytest_$1.exit" 2>/dev/null)" = 0 ]; }
runtest qsa_kernel tests/test_qsa_kernel.py
runtest compile_layer "tests/test_compile_hc.py::test_compiled_layer_is_as_accurate_as_eager[plain]"
runtest fast_gdn "tests/test_compile_hc.py::test_compiled_layer_is_as_accurate_as_eager[fast_gdn]"
runtest compile_lora "tests/test_compile_hc.py::test_compiled_lora_matches_peft"
runtest fused_model tests/test_qsa_fused_model.py
runtest grouped tests/test_grouped_gemm.py
runtest prefix_cuda "tests/test_prefix_stop_grad_cuda.py::test_prefix_stop_grad_cuda_matches_the_cache_path[torch_gdn]"
runtest prefix_cuda_fast "tests/test_prefix_stop_grad_cuda.py::test_prefix_stop_grad_cuda_matches_the_cache_path[fast_gdn]"
flags=""
passed qsa_kernel || flags="$flags --no-qsa-kernel --no-compile-qsa"
passed fused_model || flags="$flags --no-compile-qsa"
passed compile_layer || flags="$flags --no-compile-layers"
passed fast_gdn || flags="$flags --no-fast-gdn"
passed compile_lora || flags="$flags --no-fast-lora"
if ! passed prefix_cuda; then
    flags="$flags --no-prefix-stop-grad"  # slower (about 1.4x at these prefixes) but still training
elif ! passed prefix_cuda_fast; then
    flags="$flags --no-fast-gdn"  # prefix stop-grad is worth more than fast_gdn
fi
log "train_rft flags:${flags:- (all features on)}"
echo "$flags" > results/train_flags.txt

log "phase 3: wait for the checkpoint, then train"
wait $download || abort "checkpoint download failed"
log "checkpoint: $(du -sh "$MODEL_DIR" | cut -f1)"
# GPU utilization and power every second, to line up with each sample's wall_start/wall_end in train_log.jsonl
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used --format=csv,noheader -l 1 \
    > results/gpu.csv 2>/dev/null &
sampler=$!
for attempt in 1 2 3; do
    resume=$([ "$attempt" -gt 1 ] && echo --resume)
    PYTHONPATH=. python -m inference.rl.train_rft --model "$MODEL_DIR" --data data --out train --max-hours "$TRAIN_HOURS" \
        $flags $resume $TRAIN_ARGS >> results/train.log 2>&1
    code=$?
    log "train_rft attempt $attempt exited $code: $(grep -v '^\s*$' results/train.log | tail -1 | cut -c1-200)"
    [ -e train/TRAIN_DONE ] && break
    grep -q '"event": "time_limit"' train/train_log.jsonl 2>/dev/null && break
done
kill $sampler 2>/dev/null
cp train/train_log.jsonl results/ 2>/dev/null
[ -e train/TRAIN_DONE ] && log "TRAIN_DONE" || log "training did not finish; the last saved adapter is in train/"
date -u > FULL_DONE
( sleep $((GRACE_MINUTES * 60)); self_stop "grace period after FULL_DONE" ) &
log "FULL_DONE (the pod ends itself ($POD_END) in ${GRACE_MINUTES} min unless the watcher does first)"
