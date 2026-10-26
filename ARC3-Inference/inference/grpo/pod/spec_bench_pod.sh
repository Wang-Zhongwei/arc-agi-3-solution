#!/bin/bash
# MTP draft count and --enable-hc-sp on a 2xH200 pod: the upstream vLLM image serves Intel W4A16 at TP=2 with 64
# sequences, once per configuration (JOBS: kN = N draft tokens, spN = N draft tokens + --enable-hc-sp). Per
# configuration: a temperature-0 output check (8 start requests), then bench_rollout.py's full-batch window
# (8 games x 8 attempts, each from its first request with >= 40K tokens of context). The PROFILE_JOB server also
# records a torch profile of 30 engine iterations after its window (profile_summary.py: target forward vs drafter
# passes, r). Nothing waits for a person: startup failures fall back (prefix_match_unit, draft head quantization,
# memory) or skip the configuration; results go to results/<job>/, then FULL_DONE. The pod ends itself
# MAX_MINUTES after the start or GRACE_MINUTES after the end; the local watcher normally pulls and ends it first.
#   scp spec_bench_pod.sh bench_rollout.py profile_summary.py replay.jsonl to $WORK (profile_summary.py under
#   inference/grpo/), then
#   ssh ... 'cd /workspace/bench; nohup bash spec_bench_pod.sh > run.log 2>&1 < /dev/null &' < /dev/null
#   WORKDIR=/workspace/bench RUNLOG=run.log PULL="results run.log" POD_END=delete bash inference/rft/pod/watch_pod.sh POD HOST PORT DEST
set -uo pipefail

WORK=${WORK:-/workspace/bench}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
JOBS=${JOBS:-k3 k4 k5 sp3 sp4}
PROFILE_JOB=${PROFILE_JOB:-k3}
GAMES=${GAMES:-cn04,dc22,ka59,s5i5,g50t,ls20,r11l,sp80}
COPIES=${COPIES:-8}
CONCURRENCY=${CONCURRENCY:-64}
START_TOKENS=${START_TOKENS:-40000}
MIN_AFTER=${MIN_AFTER:-20000}
WINDOW=${WINDOW:-240}
WARMUP_MAX=${WARMUP_MAX:-240}
GREEDY=${GREEDY:-8}
GREEDY_TOKENS=${GREEDY_TOKENS:-256}
MAX_MINUTES=${MAX_MINUTES:-120}
GRACE_MINUTES=${GRACE_MINUTES:-3}
LOG_WAIT_SECONDS=${LOG_WAIT_SECONDS:-180}  # an abort waits this long so the watcher can pull the logs
POD_END=${POD_END:-delete}
SERVER_READY_MINUTES=${SERVER_READY_MINUTES:-30}
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
python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" | tee results/versions.txt \
    || abort "vllm does not import"
command -v runpodctl > /dev/null || { curl -sSL -m 120 -o /usr/local/bin/runpodctl \
    https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl; } \
    || log "no runpodctl: self-stop falls back to REST"
for f in bench_rollout.py inference/grpo/profile_summary.py replay.jsonl; do [ -s "$f" ] || abort "missing $f"; done

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

server_args() {  # $1 job; reads UTIL, PMU, QUANT
    local job=$1 extra=() spec k
    k=${job//[!0-9]/}  # k3 -> 3, sp4 -> 4
    spec="{\"method\": \"mtp\", \"num_speculative_tokens\": $k$([ -n "$QUANT" ] && echo ", \"draft_lm_head_quantization\": \"$QUANT\"")}"
    [[ $job == sp* ]] && extra+=(--enable-hc-sp)
    [ -n "$PMU" ] && extra+=(--prefix-match-unit "$PMU")
    if [ "$job" = "$PROFILE_JOB" ]; then
        extra+=(--profiler-config "'{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$WORK/results/$job/profile\", \"torch_profiler_with_stack\": false, \"ignore_frontend\": true, \"max_iterations\": 30}'")
    fi
    echo vllm serve "$MODEL_DIR" --served-model-name flashnext --host 127.0.0.1 --port $PORT \
        --tensor-parallel-size 2 --max-num-seqs "$CONCURRENCY" --max-model-len 147072 --max-num-batched-tokens 8192 \
        --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --attention-config "'{\"indexer_kv_dtype\": \"fp8\"}'" \
        --mamba-ssm-cache-dtype bfloat16 --gpu-memory-utilization "$UTIL" --async-scheduling \
        --enable-prefix-caching "${extra[@]}" --speculative-config "'$spec'" \
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

bench() {  # $1 job, $2 output dir, then bench_rollout.py arguments
    local job=$1 out=$2; shift 2
    python3 bench_rollout.py --replay replay.jsonl --out "$out" --games "$GAMES" --copies "$COPIES" \
        --url http://127.0.0.1:$PORT/v1 --model flashnext --label "$job" \
        --start-tokens "$START_TOKENS" --min-after "$MIN_AFTER" --request-timeout 1800 "$@"
}

PMU=64 QUANT=fp8  # carried across jobs: a fallback found once applies to the rest
for job in $JOBS; do
    dir=results/$job
    mkdir -p "$dir"
    UTIL=0.92
    ready=0
    for attempt in 1 2 3 4; do
        log "job $job: server attempt $attempt (util $UTIL, prefix_match_unit ${PMU:-default}, draft head ${QUANT:-bf16})"
        start=$(date +%s)
        start_server "$job" "$attempt"; rc=$?
        if [ $rc = 0 ]; then ready=1; log "job $job: server ready in $(( $(date +%s) - start )) s"; break; fi
        slog=$dir/server.$attempt.log
        stop_server
        if [ $rc = 2 ]; then log "job $job: not ready in ${SERVER_READY_MINUTES} min"; break; fi
        if grep -q "Invalid prefix_match_unit" "$slog"; then
            log "fallback: prefix_match_unit unset"; PMU=""
        elif grep -v "non-default args" "$slog" | grep -q "draft_lm_head_quantization"; then  # not the args echo
            log "fallback: BF16 draft head"; QUANT=""
        elif grep -qiE "out of memory|OutOfMemoryError" "$slog"; then
            log "fallback: util 0.88"; UTIL=0.88
        else
            log "job $job: server failed: $(grep -E 'Error|error|unrecognized' "$slog" | tail -3)"; break
        fi
    done
    if [ $ready != 1 ]; then echo "server failed" > "$dir/FAILED"; continue; fi
    grep -E "Using V2 Model Runner|kv cache group|GPU KV cache size|Maximum concurrency|MoE backend|SpecDecoding|cudagraph|hc_sp|sequence parallel" \
        "$dir/server.$attempt.log" | head -40 > "$dir/server_summary.txt"
    log "job $job: temperature-0 check ($GREEDY requests x $GREEDY_TOKENS tokens)"
    bench "$job" "$dir/greedy" --greedy "$GREEDY" --greedy-tokens "$GREEDY_TOKENS" > "$dir/greedy.log" 2>&1 \
        || log "job $job: greedy check exited $?"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 5 > "$dir/gpu.csv" &
    smi=$!
    profile=()
    [ "$job" = "$PROFILE_JOB" ] && profile=(--profile-iterations 30 --profile-seconds 25)
    log "job $job: full-batch window ($GAMES x $COPIES from ${START_TOKENS} tokens, ${WINDOW} s)"
    bench "$job" "$dir/bench" --window "$WINDOW" --warmup-max "$WARMUP_MAX" "${profile[@]}" > "$dir/bench.log" 2>&1 \
        || log "job $job: benchmark exited $?"
    kill $smi 2> /dev/null
    python3 -c "import json; w = json.load(open('$dir/bench/summary.json')).get('window', {}); print('window', w)" 2>&1 | tail -1
    stop_server
    cp "$dir/server.$attempt.log" "$dir/server.log"
    if [ "$job" = "$PROFILE_JOB" ] && [ -d "$dir/profile" ]; then
        PYTHONPATH=. python3 -m inference.grpo.profile_summary "$dir/profile" > "$dir/profile_summary.json" 2> "$dir/profile_summary.err" \
            || log "profile summary failed: $(tail -2 "$dir/profile_summary.err")"
        log "profile: $(du -sh "$dir/profile" | cut -f1), $(python3 -c "import json; print([(d['file'][:40], d.get('r_per_small_graph'), d.get('small_graphs_per_target_forward')) for d in json.load(open('$dir/profile_summary.json'))])" 2>&1 | tail -1)"
    fi
done

log "temperature-0 outputs against $PROFILE_JOB"
python3 - "$PROFILE_JOB" $JOBS > results/greedy_compare.json <<'PY'
import json, os, sys
base_job, jobs = sys.argv[1], sys.argv[2:]
def load(job):
    path = f"results/{job}/greedy/greedy.json"
    return {r["slot"]: r for r in json.load(open(path))} if os.path.exists(path) else None
base = load(base_job)
out = {}
for job in jobs:
    other = load(job)
    if base is None or other is None:
        out[job] = None
        continue
    rows = []
    for slot, b in base.items():
        o = other.get(slot, {})
        bt, ot = b.get("text", ""), o.get("text", "")
        common = next((i for i, (x, y) in enumerate(zip(bt, ot)) if x != y), min(len(bt), len(ot)))
        rows.append({"slot": slot, "identical": bt == ot, "common_chars": common, "chars": len(bt)})
    out[job] = {"identical": sum(r["identical"] for r in rows), "of": len(rows),
                "min_common_chars": min(r["common_chars"] for r in rows), "rows": rows}
print(json.dumps(out, indent=1))
PY
log "$(python3 -c "import json; d = json.load(open('results/greedy_compare.json')); print({j: (v['identical'], v['of'], v['min_common_chars']) if v else None for j, v in d.items()})" 2>&1 | tail -1)"

date -u > FULL_DONE
log "FULL_DONE"
( sleep $((GRACE_MINUTES * 60)); self_stop "${GRACE_MINUTES} min after FULL_DONE" ) &
