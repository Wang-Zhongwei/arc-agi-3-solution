#!/bin/bash
# The real rollout environment on a 2xH200 pod: the upstream vLLM image serves Intel W4A16 at TP=2 (64 sequences,
# MTP 3), and the Kaggle harness plays the games inline against it (configs/rollout-throughput/v1.json: 8 games
# x 9 copies = 72 games, 30 minutes each). Tool time, game time, the harness's CPU and its full-history requests are
# all real, unlike bench_rollout.py's replay. Recorded: the server's /metrics every 10 s, GPU and per-process CPU,
# a torch profile of 30 engine iterations at RUN_PROFILE_MINUTE, py-spy samples of the API server and engine core,
# and the harness's own run folder. Nothing waits for a person; results go to results/, then FULL_DONE. The pod
# ends itself MAX_MINUTES after the start or GRACE_MINUTES after the end.
#   upload repo.tgz (ARC3-Inference + tufa-arc-agi-framework, no runs/ or .venv) and env.tgz (environment_files/)
#   to $WORK, then ssh ... 'cd /workspace/bench; nohup bash rollout_env_pod.sh > run.log 2>&1 < /dev/null &' < /dev/null
#   WORKDIR=/workspace/bench RUNLOG=run.log PULL="results run.log" POD_END=delete bash inference/rft/pod/watch_pod.sh POD HOST PORT DEST
set -uo pipefail

WORK=${WORK:-/workspace/bench}
REPO=${REPO:-/workspace/repo}
MODEL_DIR=${MODEL_DIR:-/workspace/w4a16}
MODEL_REPO=${MODEL_REPO:-Intel/Qwen3.8-Flash-Next-W4A16-AutoRound}
CONFIG=${CONFIG:-configs/rollout-throughput/v1.json}
CONCURRENCY=${CONCURRENCY:-64}
RUN_PROFILE_MINUTE=${RUN_PROFILE_MINUTE:-15}
MAX_MINUTES=${MAX_MINUTES:-75}
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
    pkill -f "[v]llm serve" || true
    [ -n "${RUNPOD_POD_ID:-}" ] || { log "no RUNPOD_POD_ID: only the server was stopped"; return; }
    if command -v runpodctl > /dev/null; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1
        if [ "$POD_END" = delete ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi && return
    fi
    if [ "$POD_END" = delete ]; then m=DELETE; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
    else m=POST; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"; fi
    log "REST $m: $(curl -s -o /dev/null -w '%{http_code}' -X $m -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" -A Mozilla/5.0 "$u")"
}
abort() { log "ABORTED: $*"; echo "$*" > ABORTED; pkill -f "[v]llm serve"; sleep "$LOG_WAIT_SECONDS"; self_stop abort; exit 1; }
( sleep $((MAX_MINUTES * 60)); echo "time limit" > ABORTED; pkill -f "[v]llm serve"; sleep "$LOG_WAIT_SECONDS"; self_stop "time limit ${MAX_MINUTES} min" ) &

log "phase 0: machine"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee results/gpus.txt
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H200)" -ge 2 ] || abort "need 2 H200: $(cat results/gpus.txt)"
nproc | sed 's/^/vcpus: /' | tee results/machine.txt; free -g | tee -a results/machine.txt
command -v runpodctl > /dev/null || { curl -sSL -m 120 -o /usr/local/bin/runpodctl \
    https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 && chmod +x /usr/local/bin/runpodctl; } \
    || log "no runpodctl: self-stop falls back to REST"
for f in repo.tgz env.tgz inference/grpo/profile_summary.py; do [ -s "$f" ] || abort "missing $f"; done
# --no-same-owner: as root, tar restores the Mac's uid and the /workspace volume refuses chown, so tar fails
mkdir -p "$REPO" env && tar --no-same-owner -xzf repo.tgz -C "$REPO" && tar --no-same-owner -xzf env.tgz -C env || abort "unpacking"
ENV_DIR=$WORK/env/environment_files
[ "$(find "$ENV_DIR" -name metadata.json | wc -l)" -ge 8 ] || abort "no game files in $ENV_DIR"

log "phase 1: model download and the harness's venv, in parallel"
(
    if [ ! -e "$MODEL_DIR/DOWNLOAD_DONE" ]; then
        pip install -q hf_transfer > /dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1
        start=$(date +%s)
        (command -v hf > /dev/null && hf download "$MODEL_REPO" --local-dir "$MODEL_DIR" > results/download.log 2>&1) \
            || huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR" > results/download.log 2>&1 || exit 1
        echo "downloaded $(du -sh "$MODEL_DIR" | cut -f1) in $(( $(date +%s) - start )) s" >> results/download.log
        date -u > "$MODEL_DIR/DOWNLOAD_DONE"
    fi
) &
download=$!
pip install -q py-spy > /dev/null 2>&1 || log "no py-spy"
command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > results/uv_install.log 2>&1; }
export PATH="$HOME/.local/bin:$PATH"
(cd "$REPO/ARC3-Inference" && uv sync --frozen > "$WORK/results/uv_sync.log" 2>&1) || abort "uv sync: $(tail -3 results/uv_sync.log)"
log "harness venv: $(cd "$REPO/ARC3-Inference" && uv run --no-sync python -c 'import arcengine, arc_agi, taaf; print("arcengine", arcengine.__version__ if hasattr(arcengine, "__version__") else "?", "ok")' 2>&1 | tail -1)"
wait $download || abort "download: $(tail -2 results/download.log)"
log "$(tail -1 results/download.log)"

log "phase 2: server (TP=2, $CONCURRENCY sequences, MTP 3)"
mkdir -p results/server results/profile
cmd="vllm serve $MODEL_DIR --served-model-name flashnext --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 2 --max-num-seqs $CONCURRENCY --max-model-len 147072 --max-num-batched-tokens 8192 \
    --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --attention-config '{\"indexer_kv_dtype\": \"fp8\"}' \
    --mamba-ssm-cache-dtype bfloat16 --gpu-memory-utilization 0.92 --async-scheduling \
    --enable-prefix-caching --prefix-match-unit 64 \
    --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 3, \"draft_lm_head_quantization\": \"fp8\"}' \
    --logprobs-mode processed_logprobs --engram-config '{\"cpu_offload\": true}' \
    --profiler-config '{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$WORK/results/profile\", \"torch_profiler_with_stack\": false, \"ignore_frontend\": true, \"max_iterations\": 30}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --chat-template $MODEL_DIR/chat_template.jinja \
    --default-chat-template-kwargs '{\"preserve_thinking\": true}' --generation-config vllm \
    --mm-processor-kwargs '{\"max_pixels\": 589824}' --limit-mm-per-prompt '{\"video\": 0}' \
    --enable-prompt-tokens-details"
echo "$cmd" > results/server/args.txt
bash -c "exec $cmd" > results/server/server.log 2>&1 &
SERVER_PID=$!
deadline=$(( $(date +%s) + SERVER_READY_MINUTES * 60 ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models)" = 200 ]; do
    kill -0 $SERVER_PID 2> /dev/null || abort "server exited: $(grep -E 'Error|error' results/server/server.log | tail -3)"
    [ "$(date +%s)" -lt $deadline ] || abort "server not ready in ${SERVER_READY_MINUTES} min"
    sleep 10
done
log "server ready"

log "phase 3: samplers"
( while true; do  # server counters every 10 s, one JSON line each
    printf '{"t": %s, "metrics": %s}\n' "$(date +%s.%N)" \
        "$(curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|prompt_tokens_total|generation_tokens_total|spec_decode_num_drafts_total|spec_decode_num_accepted_tokens_total|prefix_cache_queries_total|prefix_cache_hits_total|request_success_total)' \
            | python3 -c 'import json, sys, re, collections
v = collections.defaultdict(float)
for line in sys.stdin:
    m = re.match(r"^vllm:(\w+)(\{[^}]*\})? (\S+)$", line.strip())
    if m: v[m[1]] += float(m[3])
print(json.dumps(v))')" >> results/metrics.jsonl
    sleep 10
done ) &
sampler=$!
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 5 > results/gpu.csv &
smi=$!
( while true; do  # whole-machine CPU and the busiest processes every 30 s
    { date -u +%H:%M:%S; top -b -n 1 -w 250 | head -40; } >> results/top.log
    sleep 30
done ) &
toplog=$!
(   # mid-run: the torch profile of 30 engine iterations, then py-spy on the API server and the engine core
    sleep $((RUN_PROFILE_MINUTE * 60))
    curl -s -X POST -m 900 http://127.0.0.1:$PORT/start_profile > /dev/null
    sleep 20
    curl -s -X POST -m 900 http://127.0.0.1:$PORT/stop_profile > /dev/null
    echo "profiled at $(date -u +%H:%M:%S)" > results/profile/STAMP
    if command -v py-spy > /dev/null; then
        for name in "APIServer" "EngineCore"; do
            pid=$(pgrep -f "$name" | head -1)
            [ -n "$pid" ] || pid=$(ps -eo pid,comm | awk -v n="$name" '$2 ~ n {print $1; exit}')
            [ -n "$pid" ] && timeout 60 py-spy record --pid "$pid" --duration 30 --rate 50 --format speedscope \
                -o "results/pyspy_$name.json" > "results/pyspy_$name.log" 2>&1
        done
    fi
) &
profiler=$!

log "phase 4: the harness ($CONFIG), against the server"
cd "$REPO/ARC3-Inference"
start=$(date +%s)
make interactive CONFIG_PATH="$CONFIG" ENVIRONMENTS_DIR="$ENV_DIR" EXPERIMENTS_DIR="$WORK/results/harness" \
    RUN_NAME=rollout-throughput LOCAL_ANALYZER_BASE_URL=http://127.0.0.1:$PORT/v1 > "$WORK/results/harness.log" 2>&1 \
    || log "harness exited $?"
cd "$WORK"
log "harness done in $(( $(date +%s) - start )) s: $(grep -E 'total tokens|generated tokens/sec|duration' results/harness.log | tr '\n' ' ')"
kill $sampler $smi $toplog 2> /dev/null
wait $profiler 2> /dev/null
kill $SERVER_PID 2> /dev/null; sleep 20; pkill -9 -f "[v]llm serve" || true
PYTHONPATH=. python3 -m inference.grpo.profile_summary results/profile > results/profile_summary.json 2> results/profile_summary.err \
    || log "profile summary failed: $(tail -2 results/profile_summary.err)"
# keep the transcripts but compress them (they are most of the bytes)
find results/harness -name '*.txt' -path '*transcripts*' -size +1M -exec gzip -q {} + 2> /dev/null || true
date -u > FULL_DONE
log "FULL_DONE"
( sleep $((GRACE_MINUTES * 60)); self_stop "${GRACE_MINUTES} min after FULL_DONE" ) &
