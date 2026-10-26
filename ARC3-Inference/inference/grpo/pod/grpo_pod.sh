#!/bin/bash
# GRPO on one 4xH200 pod (docs/RL/grpo-pipeline.md): vLLM on GPUs 0-1 (TP=2, 64 sequences, MTP 3, processed logprobs,
# the weight-sync extension), two trainers on GPUs 2 and 3 (LoRA data parallel), the harness and the loop on the CPU.
# The image is the vLLM nightly (torch 2.14); the trainers run in their own venv with the tested torch 2.8 stack.
# Nothing waits for a person: the loop's gates abort a broken setup in the first waves (token-exact sequences,
# trainer/vLLM agreement at the base weights, the first sync carrying the update). Results go to results/, then
# FULL_DONE; the pod ends itself (POD_END=stop|delete) GRACE_MINUTES after the end, or at MAX_HOURS.
#   upload repo.tgz (ARC3-Inference + tufa-arc-agi-framework + docs/RL/data, no runs/ or .venv) and env.tgz
#   (environment_files/) to $WORK, then ssh ... 'cd /workspace/grpo; nohup bash grpo_pod.sh > run.log 2>&1 < /dev/null &'
#   WORKDIR=/workspace/grpo RUNLOG=run.log PULL="results run.log" bash inference/rft/pod/watch_pod.sh POD HOST PORT DEST
set -uo pipefail

WORK=${WORK:-/workspace/grpo}
REPO=${REPO:-/workspace/repo}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
TRAINER_VENV=${TRAINER_VENV:-/workspace/trainer-venv}
CONFIG=${CONFIG:-configs/grpo/v1.json}
STEPS=${STEPS:-30}
LR=${LR:-1e-5}
KEEP=${KEEP:-6}
MINIBATCHES=${MINIBATCHES:-4}  # optimizer steps per wave (decided 2026-10-26)
OPTIMIZER=${OPTIMIZER:-adamw}
MAX_HOURS=${MAX_HOURS:-12}
ARENA_MAX_GB=${ARENA_MAX_GB:-200}  # per trainer: 130K tokens in "layer" recompute need ~127 GB
WANDB_PROJECT=${WANDB_PROJECT:-}
POD_END=${POD_END:-stop}
GRACE_MINUTES=${GRACE_MINUTES:-5}
LOG_WAIT_SECONDS=${LOG_WAIT_SECONDS:-180}
SERVER_READY_MINUTES=${SERVER_READY_MINUTES:-30}
PORT=8000
cd "$WORK"
mkdir -p results
rm -f FULL_DONE ABORTED
for var in RUNPOD_POD_ID RUNPOD_API_KEY; do  # an ssh session lacks the container's RUNPOD_* variables
    [ -n "${!var:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$var=")" 2>/dev/null || true
done
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
stop_all() {
    touch "$WORK/STOP" "$WORK/jobs/STOP" 2>/dev/null
    [ -s trainer0.pid ] && kill "$(cat trainer0.pid)" 2>/dev/null
    [ -s trainer1.pid ] && kill "$(cat trainer1.pid)" 2>/dev/null
    [ -s loop.pid ] && kill "$(cat loop.pid)" 2>/dev/null
    [ -s server.pid ] && kill "$(cat server.pid)" 2>/dev/null
    sleep 10
    pkill -9 -f "[v]llm serve" || true
}
self_stop() {
    log "ending this pod ($1): $POD_END"
    stop_all
    [ -n "${RUNPOD_POD_ID:-}" ] || { log "no RUNPOD_POD_ID: only the processes were stopped"; return; }
    if command -v runpodctl > /dev/null; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1
        if [ "$POD_END" = delete ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi && return
    fi
    if [ "$POD_END" = delete ]; then m=DELETE; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
    else m=POST; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"; fi
    log "REST $m: $(curl -s -o /dev/null -w '%{http_code}' -X $m -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" -A Mozilla/5.0 "$u")"
}
collect() {  # small things the watcher pulls; the waves stay on the volume
    mkdir -p results/loop
    cp -f loop_log.jsonl loop_state.json probe.json train_rank*.jsonl results/loop/ 2>/dev/null
    cp -rf jobs results/loop/ 2>/dev/null
    rm -rf results/loop/ckpt && mkdir -p results/loop/ckpt && cp -rf ckpt/latest results/loop/ckpt/ 2>/dev/null
    for d in waves/local/*_wave_*; do
        [ -d "$d" ] || continue
        mkdir -p "results/loop/waves/$(basename "$d")"
        cp -f "$d/rewards.json" "$d/evaluation.json" "results/loop/waves/$(basename "$d")/" 2>/dev/null
    done
}
abort() { log "ABORTED: $*"; echo "$*" > ABORTED; collect; stop_all; sleep "$LOG_WAIT_SECONDS"; self_stop abort; exit 1; }
( sleep $((MAX_HOURS * 3600)); touch "$WORK/STOP"; sleep 1800; echo "time limit" > ABORTED; collect; self_stop "time limit ${MAX_HOURS} h" ) &
echo $! > timer.pid

log "phase 0: machine"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee results/gpus.txt
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H200)" -ge 4 ] || abort "need 4 H200: $(cat results/gpus.txt)"
{ nproc | sed 's/^/vcpus: /'; free -g; df -h /workspace; } | tee results/machine.txt
command -v runpodctl > /dev/null || { curl -sSL -m 120 -o /usr/local/bin/runpodctl \
    https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl; } \
    || log "no runpodctl: self-stop falls back to REST"
for f in repo.tgz env.tgz; do [ -s "$f" ] || abort "missing $f"; done
# --no-same-owner: as root, tar restores the Mac's uid and the /workspace volume refuses chown
mkdir -p "$REPO" env && tar --no-same-owner -xzf repo.tgz -C "$REPO" && tar --no-same-owner -xzf env.tgz -C env || abort "unpacking"
ENV_DIR=$WORK/env/environment_files
[ "$(find "$ENV_DIR" -name metadata.json | wc -l)" -ge 8 ] || abort "no game files in $ENV_DIR"
[ -s "$REPO/docs/RL/data/thinking_per_action.json" ] || abort "missing docs/RL/data/thinking_per_action.json in repo.tgz"

log "phase 1: model, harness venv and trainer venv, in parallel"
(
    if [ ! -e "$MODEL_DIR/DOWNLOAD_DONE" ]; then
        pip install -q hf_transfer > /dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1
        start=$(date +%s)
        (command -v hf > /dev/null && hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" --exclude "*.bak" > results/download.log 2>&1) \
            || huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR" > results/download.log 2>&1 || exit 1
        echo "downloaded $(du -sh "$MODEL_DIR" | cut -f1) in $(( $(date +%s) - start )) s" >> results/download.log
        date -u > "$MODEL_DIR/DOWNLOAD_DONE"
    fi
) &
download=$!
(
    # the trainers' tested stack (torch 2.8, requirements-pod.txt, tilelang for fla's GDN backward on Hopper), not
    # the image's torch 2.14
    if [ ! -e "$TRAINER_VENV/INSTALL_DONE" ]; then
        python3 -m venv "$TRAINER_VENV" && "$TRAINER_VENV/bin/python" -m pip install -q --upgrade pip \
            && "$TRAINER_VENV/bin/python" -m pip install -q "torch==2.8.0" "torchvision==0.23.0" \
            && "$TRAINER_VENV/bin/python" -m pip install -q -r "$REPO/ARC3-Inference/inference/rft/pod/requirements-pod.txt" \
                "tilelang==0.1.15" safetensors $( [ -n "$WANDB_PROJECT" ] && echo wandb ) \
            && date -u > "$TRAINER_VENV/INSTALL_DONE"
    fi
) > results/trainer_venv.log 2>&1 &
trainer_venv=$!
pip install -q py-spy > /dev/null 2>&1 || true
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > results/uv_install.log 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
(cd "$REPO/ARC3-Inference" && uv sync --frozen > "$WORK/results/uv_sync.log" 2>&1) || abort "uv sync: $(tail -3 results/uv_sync.log)"
[ -n "$WANDB_PROJECT" ] && (cd "$REPO/ARC3-Inference" && uv pip install -q wandb > /dev/null 2>&1)
log "harness venv: $(cd "$REPO/ARC3-Inference" && uv run --no-sync python -c 'import arcengine, arc_agi, taaf; print("ok")' 2>&1 | tail -1)"
wait $trainer_venv || abort "trainer venv: $(tail -5 results/trainer_venv.log)"
"$TRAINER_VENV/bin/python" -c "import torch, transformers, peft, fla, tilelang; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" \
    | tee results/trainer_env.txt || abort "trainer imports: $(tail -5 results/trainer_venv.log)"
# tilelang JIT-compiles fla's GDN backward kernels on Hopper: it needs nvcc
NVCC_HOME=/usr/local/cuda
if [ ! -x "$NVCC_HOME/bin/nvcc" ]; then
    "$TRAINER_VENV/bin/python" -m pip install -q "nvidia-cuda-nvcc-cu12==12.8.*" >> results/trainer_venv.log 2>&1
    NVCC_HOME=$("$TRAINER_VENV/bin/python" -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null)
fi
[ -x "$NVCC_HOME/bin/nvcc" ] || abort "no nvcc for tilelang"
log "nvcc: $("$NVCC_HOME/bin/nvcc" --version | tail -1)"
wait $download || abort "download: $(tail -2 results/download.log)"
log "$(tail -1 results/download.log)"

log "phase 2: server on GPUs 0-1, the trainers' CUDA tests on GPU 2 meanwhile"
mkdir -p results/server
# the trainers simulate both FP8 caches (train_grpo --kv-fp8, its default); BF16 caches here need --no-kv-fp8 there
cmd="vllm serve $MODEL_DIR --served-model-name flashnext --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 2 --max-num-seqs 64 --max-model-len 147072 --max-num-batched-tokens 8192 \
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --attention-config '{\"indexer_kv_dtype\": \"fp8\"}' \
    --mamba-ssm-cache-dtype bfloat16 --gpu-memory-utilization 0.92 --async-scheduling \
    --enable-prefix-caching --prefix-match-unit 64 \
    --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3, \"draft_lm_head_quantization\": \"fp8\"}' \
    --logprobs-mode processed_logprobs --engram-config '{\"cpu_offload\": true}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --chat-template $MODEL_DIR/chat_template.jinja \
    --default-chat-template-kwargs '{\"preserve_thinking\": true}' --generation-config vllm \
    --mm-processor-kwargs '{\"max_pixels\": 589824}' --limit-mm-per-prompt '{\"video\": 0}' \
    --enable-prompt-tokens-details --worker-extension-cls inference.grpo.vllm_sync.WeightSyncExtension"
echo "$cmd" > results/server/args.txt
CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 PYTHONPATH="$REPO/ARC3-Inference" \
    bash -c "exec $cmd" > results/server/server.log 2>&1 &
echo $! > server.pid
runtest() {  # name, pytest args: the trainer's CUDA tests, as train_pod.sh
    local name=$1; shift
    (cd "$REPO/ARC3-Inference" && CUDA_VISIBLE_DEVICES=2 CUDA_HOME=$NVCC_HOME timeout 1200 env PYTHONPATH=.:tests \
        "$TRAINER_VENV/bin/python" -m pytest -q -p no:logging "$@") > "results/pytest_$name.log" 2>&1
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
runtest grpo tests/test_grpo_loss.py tests/test_grpo_train.py
passed grpo || abort "GRPO trainer tests failed on the GPU host: $(tail -5 results/pytest_grpo.log)"
flags=""
passed qsa_kernel || flags="$flags --no-qsa-kernel --no-compile-qsa"
passed fused_model || flags="$flags --no-compile-qsa"
passed compile_layer || flags="$flags --no-compile-layers"
passed fast_gdn || flags="$flags --no-fast-gdn"
passed compile_lora || flags="$flags --no-fast-lora"
log "trainer flags:${flags:- (all features on)}"
deadline=$(( $(date +%s) + SERVER_READY_MINUTES * 60 ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)" = 200 ]; do
    kill -0 "$(cat server.pid)" 2> /dev/null || abort "server exited: $(grep -E 'Error|error' results/server/server.log | tail -3)"
    [ "$(date +%s)" -lt $deadline ] || abort "server not ready in ${SERVER_READY_MINUTES} min"
    sleep 10
done
log "server ready"
( while true; do  # server counters every 30 s
    printf '{"t": %s, "metrics": %s}\n' "$(date +%s.%N)" \
        "$(curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|prompt_tokens_total|generation_tokens_total|spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_total|prefix_cache_queries_total|prefix_cache_hits_total|request_success_total)' \
            | python3 -c 'import json, sys, re, collections
v = collections.defaultdict(float)
for line in sys.stdin:
    m = re.match(r"^vllm:(\w+)(\{[^}]*\})? (\S+)$", line.strip())
    if m: v[m[1]] += float(m[3])
print(json.dumps(v))')" >> results/metrics.jsonl
    sleep 30
done ) &
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 > results/gpu.csv &

log "phase 3: trainers on GPUs 2 and 3"
# host memory left for the two pinned arenas: what is available now (server up, its PLE table pinned) minus the
# trainers' shared PLE page cache (~103 GB), their processes and the harness (~60 GB)
available=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
arena=$(python3 -c "print(max(0, min($ARENA_MAX_GB, ($available - 103 - 60) // 2)))")
log "host memory available ${available} GB -> arena ${arena} GB per trainer"
[ "$arena" -ge 120 ] || abort "only ${arena} GB of host arena per trainer (130K-token episodes need ~127)"
mkdir -p "$WORK/jobs"
for rank in 0 1; do
    (cd "$REPO/ARC3-Inference" && CUDA_VISIBLE_DEVICES=$((rank + 2)) CUDA_HOME=$NVCC_HOME RANK=$rank WORLD_SIZE=2 \
        MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
        nohup "$TRAINER_VENV/bin/python" -m inference.grpo.train_grpo --model "$MODEL_DIR" --work "$WORK" \
        --arena-gb "$arena" --optimizer "$OPTIMIZER" --resume $flags > "$WORK/results/trainer$rank.log" 2>&1 < /dev/null &
     echo $! > "$WORK/trainer$rank.pid")
done
deadline=$(( $(date +%s) + 20 * 60 ))
until grep -q '"event": "loaded"' "$WORK/train_rank0.jsonl" 2>/dev/null && grep -q '"event": "loaded"' "$WORK/train_rank1.jsonl" 2>/dev/null; do
    for rank in 0 1; do kill -0 "$(cat trainer$rank.pid)" 2>/dev/null || abort "trainer $rank exited: $(tail -5 results/trainer$rank.log)"; done
    [ "$(date +%s)" -lt $deadline ] || abort "trainers not loaded in 20 min"
    sleep 15
done
log "trainers loaded"

log "phase 4: the loop ($STEPS steps)"
wandb_args=()
[ -n "$WANDB_PROJECT" ] && wandb_args=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "grpo-${RUNPOD_POD_ID:-local}")
(cd "$REPO/ARC3-Inference" && nohup uv run --no-sync python -m inference.grpo.loop --work "$WORK" --config "$CONFIG" \
    --environments-dir "$ENV_DIR" --base-url "http://127.0.0.1:$PORT/v1" --steps "$STEPS" --lr "$LR" --keep "$KEEP" --minibatches "$MINIBATCHES" \
    --tau "$REPO/docs/RL/data/thinking_per_action.json" "${wandb_args[@]}" > "$WORK/results/loop.log" 2>&1 < /dev/null &
 echo $! > "$WORK/loop.pid")
while kill -0 "$(cat loop.pid)" 2>/dev/null; do
    for rank in 0 1; do kill -0 "$(cat trainer$rank.pid)" 2>/dev/null || abort "trainer $rank died: $(tail -5 results/trainer$rank.log)"; done
    kill -0 "$(cat server.pid)" 2>/dev/null || abort "server died: $(tail -5 results/server/server.log)"
    collect
    sleep 60
done
grep -q '"event": "end"' loop_log.jsonl || abort "loop ended without finishing: $(tail -5 results/loop.log)"
log "loop done: $(tail -1 loop_log.jsonl | cut -c1-300)"
touch "$WORK/jobs/STOP"
collect
date -u > FULL_DONE
log "FULL_DONE"
( sleep $((GRACE_MINUTES * 60)); self_stop "${GRACE_MINUTES} min after FULL_DONE" ) &
