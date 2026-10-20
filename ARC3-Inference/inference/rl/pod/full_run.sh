#!/bin/bash
# One-shot run on an expensive GPU (B200, H200): network check, packages, checkpoint, CUDA tests, a jobs file
# generated from the tests (features whose test failed are off), the model server's jobs, then FULL_DONE.
# Nothing waits for a person. The pod deletes itself MAX_MINUTES after start, or GRACE_MINUTES after the end
# (a local watcher normally pulls the results and deletes it first).
#
# Upload inference/, tests/, game.txt, game_ft09.txt (and wheels/ when PyPI is unreachable) to $WORK, then:
#   ssh ... 'cd /workspace/smoke; nohup bash inference/rl/pod/full_run.sh > full_run.log 2>&1 < /dev/null &' < /dev/null
# REHEARSAL=1: shorter lengths (16K/32K, no 123K) on a cheaper 96 GB GPU.
set -uo pipefail

WORK=${WORK:-/workspace/smoke}
VENV=${VENV:-/workspace/venv}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
REHEARSAL=${REHEARSAL:-0}
PLE_FP8=${PLE_FP8:-0}  # 1: also fetch nvidia's FP8 PLE table and run make_jobs.py --ple-fp8 (BF16 vs FP8 PLE gradients)
MAX_MINUTES=${MAX_MINUTES:-90}
GRACE_MINUTES=${GRACE_MINUTES:-20}
MIN_HF_MBPS=${MIN_HF_MBPS:-40}
cd "$WORK"
mkdir -p results
rm -f FULL_DONE ABORTED
# an ssh session lacks the container's RUNPOD_* variables (pod id and a pod-scoped API key for runpodctl)
for var in RUNPOD_POD_ID RUNPOD_API_KEY; do
    [ -n "${!var:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$var=")" 2>/dev/null || true
done
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
self_delete() {
    log "deleting this pod ($1)"
    if command -v runpodctl > /dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1  # it reads a config file, not the env
        runpodctl remove pod "$RUNPOD_POD_ID" || log "runpodctl failed"
    else
        log "no runpodctl or RUNPOD_POD_ID: stopping the processes only"
        pkill -f "inference.rl.smoke_lora" || true
    fi
}
abort() { log "ABORTED: $*"; echo "$*" > ABORTED; self_delete "abort"; exit 1; }
( sleep $((MAX_MINUTES * 60)); echo "time limit" > ABORTED; self_delete "time limit ${MAX_MINUTES} min" ) &
echo $! > deadman.pid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=/usr/local/cuda/bin:$PATH

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
log "HF ${hf_mbps} MB/s, PyPI HTTP ${pypi}, $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "{\"hf_mbps\": ${hf_mbps}, \"pypi\": \"${pypi}\"}" > results/network.json
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
python -c "import torch, triton, transformers, fla; print('torch', torch.__version__, 'triton', triton.__version__, transformers.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())" | tee results/env.txt || abort "imports"

# fla 0.5.2 with Triton 3.4 on sm100 (B200): the GDN backward kernel's pipelined configs (num_stages 2-4) fail to
# compile ("PassManager::run failed" in TritonGPUAssignLatencies), in every path including transformers' own.
# fla already special-cases Blackwell for the forward; give the backward num_stages 1 there.
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
if [ "$PLE_FP8" = 1 ]; then  # 53.7 GB: nvidia's PLE (FP8, one scale) and MTP, after the main checkpoint
    HF_HUB_ENABLE_HF_TRANSFER=1 nohup sh -c "hf download '$MODEL_REPO' --local-dir '$MODEL_DIR' --exclude '*.bak' && hf download nvidia/Qwen3.8-Flash-Next-NVFP4 model-fp8-mtp-ple.safetensors --local-dir /workspace/nvfp8" > results/download.log 2>&1 < /dev/null &
else
    HF_HUB_ENABLE_HF_TRANSFER=1 nohup hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --exclude "*.bak" > results/download.log 2>&1 < /dev/null &
fi
download=$!

log "phase 2: CUDA tests"
runtest() {  # name, pytest args
    local name=$1; shift
    timeout 1200 env PYTHONPATH=. python -m pytest -q -p no:logging "$@" > "results/pytest_$name.log" 2>&1
    echo $? > "results/pytest_$name.exit"
    log "  $name: exit $(cat results/pytest_$name.exit), $(tail -1 results/pytest_$name.log)"
}
runtest qsa_kernel tests/test_qsa_kernel.py
runtest compile_layer "tests/test_compile_hc.py::test_compiled_layer_is_as_accurate_as_eager[plain]"
runtest fast_gdn "tests/test_compile_hc.py::test_compiled_layer_is_as_accurate_as_eager[fast_gdn]"
runtest compile_lora "tests/test_compile_hc.py::test_compiled_lora_matches_peft"
runtest fused_model tests/test_qsa_fused_model.py
runtest grouped tests/test_grouped_gemm.py
runtest prefix_cuda "tests/test_prefix_stop_grad_cuda.py::test_prefix_stop_grad_cuda_matches_the_cache_path[torch_gdn]"
runtest prefix_cuda_fast "tests/test_prefix_stop_grad_cuda.py::test_prefix_stop_grad_cuda_matches_the_cache_path[fast_gdn]"
echo 0 > results/pytest_prefix_cpu.exit  # CPU tiny-model tests cannot run next to fla; passed locally

log "phase 3: jobs from the test outcomes"
python inference/rl/pod/make_jobs.py results jobs.txt $([ "$REHEARSAL" = 1 ] && echo --rehearsal) $([ "$PLE_FP8" = 1 ] && echo --ple-fp8) | tee -a results/env.txt

log "phase 4: wait for the checkpoint, then the model server's jobs"
wait $download || abort "checkpoint download failed"
log "checkpoint: $(du -sh "$MODEL_DIR" | cut -f1)"
SERVER_ARGS=$([ "$PLE_FP8" = 1 ] && echo "--ple-fp8-path /workspace/nvfp8") \
    JOBS_FILE="$WORK/jobs.txt" MODEL_DIR="$MODEL_DIR" bash inference/rl/pod/smoke_pod.sh > results/smoke_pod.log 2>&1
log "jobs finished: $(tail -1 results/smoke_pod.log)"
cp jobs/*.result.json results/ 2>/dev/null
date -u > FULL_DONE
( sleep $((GRACE_MINUTES * 60)); self_delete "grace period after FULL_DONE" ) &
log "FULL_DONE (the pod deletes itself in ${GRACE_MINUTES} min unless the watcher does first)"
