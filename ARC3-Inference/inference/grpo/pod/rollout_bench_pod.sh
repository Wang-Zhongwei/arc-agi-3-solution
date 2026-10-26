#!/bin/bash
# One GRPO-shaped rollout step on a 2xH200 pod (docs/RL/h200-rollout-serving.md, "First benchmark"): the upstream
# vLLM image serves Intel W4A16 at TP=2 with 40 sequences, and bench_rollout.py replays 5 games x 8 rollouts to
# their end. Nothing waits for a person: startup failures fall back automatically (prefix_match_unit, draft head
# quantization, memory), results go to results/<job>/, then FULL_DONE. The pod stops itself MAX_MINUTES after
# the start or GRACE_MINUTES after the end; the local watcher normally pulls the results and stops it first.
#   scp rollout_bench_pod.sh bench_rollout.py replay.jsonl to $WORK, then
#   ssh ... 'cd /workspace/bench; nohup bash rollout_bench_pod.sh > run.log 2>&1 < /dev/null &' < /dev/null
#   WORKDIR=/workspace/bench RUNLOG=run.log PULL="results run.log" bash inference/rft/pod/watch_pod.sh POD HOST PORT DEST
set -uo pipefail

WORK=${WORK:-/workspace/bench}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
JOBS=${JOBS:-A}  # A: TP=2, 40 sequences; B: A plus --enable-hc-sp
GAMES=${GAMES:-cn04,sc25,ka59,s5i5,dc22}
COPIES=${COPIES:-8}
MAX_MINUTES=${MAX_MINUTES:-90}
GRACE_MINUTES=${GRACE_MINUTES:-2}
LOG_WAIT_SECONDS=${LOG_WAIT_SECONDS:-180}  # an abort waits this long so the watcher can pull the logs
POD_END=${POD_END:-delete}
SERVER_READY_MINUTES=${SERVER_READY_MINUTES:-40}
MIN_HF_MBPS=${MIN_HF_MBPS:-40}
PORT=8000
cd "$WORK"
mkdir -p results
rm -f FULL_DONE ABORTED
for var in RUNPOD_POD_ID RUNPOD_API_KEY; do  # an ssh session lacks the container's RUNPOD_* variables
    [ -n "${!var:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$var=")" 2>/dev/null || true
done
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
self_stop() {
    log "ending this pod ($1): $POD_END"
    pkill -f "vllm serve" || true
    [ -n "${RUNPOD_POD_ID:-}" ] || { log "no RUNPOD_POD_ID: only the server was stopped"; return; }
    if command -v runpodctl > /dev/null; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1
        if [ "$POD_END" = delete ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi && return
    fi
    if [ "$POD_END" = delete ]; then m=DELETE; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
    else m=POST; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"; fi
    log "REST $m: $(curl -s -o /dev/null -w '%{http_code}' -X $m -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" -A Mozilla/5.0 "$u")"
}
abort() { log "ABORTED: $*"; echo "$*" > ABORTED; pkill -f "vllm serve"; sleep "$LOG_WAIT_SECONDS"; self_stop abort; exit 1; }
( sleep $((MAX_MINUTES * 60)); echo "time limit" > ABORTED; pkill -f "vllm serve"; sleep "$LOG_WAIT_SECONDS"; self_stop "time limit ${MAX_MINUTES} min" ) &

log "phase 0: machine"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee results/gpus.txt
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H200)" -ge 2 ] || abort "need 2 H200: $(cat results/gpus.txt)"
nvidia-smi topo -m > results/topo.txt 2>&1
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" | tee results/versions.txt \
    || abort "vllm does not import"
free -g | tee results/host_memory.txt
command -v runpodctl > /dev/null || { curl -sSL -m 120 -o /usr/local/bin/runpodctl \
    https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl; } \
    || log "no runpodctl: self-stop falls back to REST"
for f in bench_rollout.py replay.jsonl; do [ -s "$f" ] || abort "missing $f"; done

log "phase 1: model"
if [ ! -e "$MODEL_DIR/DOWNLOAD_DONE" ]; then
    probe=$(curl -sSL -m 15 -o /dev/null -w "%{speed_download}" -A Mozilla/5.0 \
        "https://huggingface.co/$MODEL_REPO/resolve/main/model-00003-of-00017.safetensors" || echo 0)
    mbps=$(python3 -c "print(int(float('${probe:-0}') / 1e6))")
    log "Hugging Face probe: ${mbps} MB/s"
    [ "$mbps" -ge "$MIN_HF_MBPS" ] || abort "Hugging Face at ${mbps} MB/s is below ${MIN_HF_MBPS}"
    pip install -q hf_transfer > /dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1
    start=$(date +%s)
    (command -v hf > /dev/null && hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" > results/download.log 2>&1) \
        || huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR" > results/download.log 2>&1 \
        || abort "download: $(tail -2 results/download.log)"
    log "downloaded $(du -sh "$MODEL_DIR" | cut -f1) in $(( $(date +%s) - start )) s"
    date -u > "$MODEL_DIR/DOWNLOAD_DONE"
fi

server_args() {  # $1 job; reads UTIL, PMU, SPEC
    local extra=()
    [ "$1" = B ] && extra+=(--enable-hc-sp)
    [ -n "$PMU" ] && extra+=(--prefix-match-unit "$PMU")
    echo vllm serve "$MODEL_DIR" --served-model-name flashnext --host 127.0.0.1 --port $PORT \
        --tensor-parallel-size 2 --max-num-seqs 40 --max-model-len 147072 --max-num-batched-tokens 8192 \
        --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --attention-config "'{\"indexer_kv_dtype\": \"fp8\"}'" \
        --mamba-ssm-cache-dtype bfloat16 --gpu-memory-utilization "$UTIL" --async-scheduling \
        --enable-prefix-caching "${extra[@]}" --speculative-config "'$SPEC'" \
        --logprobs-mode processed_logprobs --engram-config "'{\"cpu_offload\": true}'" \
        --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
        --chat-template "$MODEL_DIR/chat_template.jinja" \
        --default-chat-template-kwargs "'{\"preserve_thinking\": true}'" --generation-config vllm \
        --mm-processor-kwargs "'{\"max_pixels\": 589824}'" --limit-mm-per-prompt "'{\"video\": 0}'" \
        --enable-prompt-tokens-details
}

start_server() {  # $1 job, $2 attempt; returns 0 when ready, 1 when the server exited, 2 on timeout
    local dir=results/$1 cmd
    cmd=$(server_args "$1")
    echo "$cmd" > "$dir/server_args.$2.txt"
    bash -c "exec $cmd" > "$dir/server.$2.log" 2>&1 &
    SERVER_PID=$!
    local deadline=$(( $(date +%s) + SERVER_READY_MINUTES * 60 ))
    while [ "$(date +%s)" -lt $deadline ]; do
        kill -0 $SERVER_PID 2> /dev/null || return 1
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)" = 200 ] && return 0
        sleep 10
    done
    return 2
}

stop_server() {
    kill "$SERVER_PID" 2> /dev/null
    for _ in $(seq 60); do kill -0 "$SERVER_PID" 2> /dev/null || return; sleep 2; done
    kill -9 "$SERVER_PID" 2> /dev/null; pkill -9 -f "vllm serve" || true; sleep 5
}

for job in $JOBS; do
    dir=results/$job
    mkdir -p "$dir"
    UTIL=0.92 PMU=64 SPEC='{"method": "mtp", "num_speculative_tokens": 3, "draft_lm_head_quantization": "fp8"}'
    ready=0
    for attempt in 1 2 3 4; do
        log "job $job: server attempt $attempt (util $UTIL, prefix_match_unit ${PMU:-default}, spec $SPEC)"
        start=$(date +%s)
        start_server "$job" "$attempt"; rc=$?
        if [ $rc = 0 ]; then ready=1; log "job $job: server ready in $(( $(date +%s) - start )) s"; break; fi
        slog=$dir/server.$attempt.log
        stop_server
        if [ $rc = 2 ]; then log "job $job: not ready in ${SERVER_READY_MINUTES} min"; break; fi
        if grep -q "Invalid prefix_match_unit" "$slog"; then
            log "fallback: $(grep -o 'Got group block sizes=[^.]*' "$slog" | head -1); prefix_match_unit unset"; PMU=""
        elif grep -v "non-default args" "$slog" | grep -q "draft_lm_head_quantization"; then  # not the args echo
            log "fallback: draft head quantization rejected; BF16 draft head"
            SPEC='{"method": "mtp", "num_speculative_tokens": 3}'
        elif grep -qiE "out of memory|OutOfMemoryError" "$slog"; then
            log "fallback: out of memory at startup; util 0.88"; UTIL=0.88
        else
            log "job $job: server failed: $(grep -E 'Error|error' "$slog" | tail -3)"; break
        fi
    done
    if [ $ready != 1 ]; then echo "server failed" > "$dir/FAILED"; continue; fi
    grep -E "Using V2 Model Runner|kv cache group sizes|GPU KV cache size|Maximum concurrency|MoE backend|GDN|SpecDecoding|Capturing|cudagraph" \
        "$dir/server.$attempt.log" | head -40 > "$dir/server_summary.txt"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 > "$dir/gpu.csv" &
    smi=$!
    log "job $job: benchmark ($GAMES x $COPIES)"
    python3 bench_rollout.py --replay replay.jsonl --out "$dir/bench" --games "$GAMES" --copies "$COPIES" \
        --url http://127.0.0.1:$PORT/v1 --model flashnext --label "tp2_c40_$job" \
        --step-timeout 2700 --request-timeout 1800 > "$dir/bench.log" 2>&1 || log "job $job: benchmark exited $?"
    kill $smi 2> /dev/null
    tail -1 "$dir/bench.log"
    stop_server
    cp "$dir/server.$attempt.log" "$dir/server.log"
done

date -u > FULL_DONE
log "FULL_DONE"
( sleep $((GRACE_MINUTES * 60)); self_stop "${GRACE_MINUTES} min after FULL_DONE" ) &
