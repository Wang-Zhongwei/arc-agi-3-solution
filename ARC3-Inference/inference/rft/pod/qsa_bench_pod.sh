#!/bin/bash
# QSA kernel tests and the dK/dV time split (bench_qsa.py) on a light GPU pod: no checkpoint, the image's torch
# and Triton. Upload inference/ and tests/ to $WORK, then:
#   ssh ... 'cd /workspace/qsa; nohup bash inference/rft/pod/qsa_bench_pod.sh > run.log 2>&1 < /dev/null &' < /dev/null
#   WORKDIR=/workspace/qsa RUNLOG=run.log PULL="results run.log" POD_END=delete \
#       bash inference/rft/pod/watch_pod.sh POD HOST PORT DEST
# FULL_DONE or ABORTED marks the end. The pod also ends itself (POD_END) GRACE_MINUTES after that, or MAX_MINUTES
# after the start, in case the watcher is gone.
set -uo pipefail

WORK=${WORK:-/workspace/qsa}
VENV=${VENV:-/workspace/venv}
MAX_MINUTES=${MAX_MINUTES:-40}
GRACE_MINUTES=${GRACE_MINUTES:-5}
LOG_WAIT_SECONDS=${LOG_WAIT_SECONDS:-120}  # an abort waits this long so the watcher can pull the logs
POD_END=${POD_END:-delete}
cd "$WORK"
mkdir -p results
rm -f FULL_DONE ABORTED
for var in RUNPOD_POD_ID RUNPOD_API_KEY; do  # an ssh session lacks the container's RUNPOD_* variables
    [ -n "${!var:-}" ] || export "$(tr '\0' '\n' < /proc/1/environ | grep "^$var=")" 2>/dev/null || true
done
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
self_stop() {
    log "ending this pod ($1): $POD_END"
    [ -n "${RUNPOD_POD_ID:-}" ] || { log "no RUNPOD_POD_ID: nothing to stop"; return; }
    if command -v runpodctl > /dev/null; then
        runpodctl config --apiKey "${RUNPOD_API_KEY:-}" > /dev/null 2>&1
        if [ "$POD_END" = delete ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi && return
    fi
    if [ "$POD_END" = delete ]; then m=DELETE; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID"
    else m=POST; u="https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"; fi
    log "REST $m: $(curl -s -o /dev/null -w '%{http_code}' -X $m -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" -A Mozilla/5.0 "$u")"
}
abort() { log "ABORTED: $*"; echo "$*" > ABORTED; sleep "$LOG_WAIT_SECONDS"; self_stop abort; exit 1; }
( sleep $((MAX_MINUTES * 60)); echo "time limit" > ABORTED; sleep "$LOG_WAIT_SECONDS"; self_stop "time limit ${MAX_MINUTES} min" ) &

log "phase 0: machine"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader | tee results/gpu.txt
# the image's system Python is externally managed (PEP 668): a venv that still sees its torch and Triton
[ -x "$VENV/bin/python" ] || python3 -m venv --system-site-packages "$VENV" || abort "venv"
export PATH="$VENV/bin:$PATH"
python -c "import pytest" 2> /dev/null || python -m pip install -q pytest || abort "pip install pytest"
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name(0))" \
    | tee results/env.txt || abort "no torch, Triton or GPU"

log "phase 1: kernel tests against the FP32 dense reference"
PYTHONPATH=. python -m pytest -q -rs tests/test_qsa_kernel.py tests/test_qsa_union.py > results/pytest.log 2>&1
status=$?
log "pytest exit $status: $(tail -1 results/pytest.log)"
[ $status -eq 0 ] || echo "pytest exit $status" > results/TESTS_FAILED

log "phase 2: benchmark"
PYTHONPATH=. python -m inference.rft.bench_qsa > results/bench_qsa.json 2> results/bench_qsa.log \
    || log "benchmark failed: $(tail -3 results/bench_qsa.log)"
cat results/bench_qsa.json

date -u > FULL_DONE
log "FULL_DONE"
( sleep $((GRACE_MINUTES * 60)); self_stop "${GRACE_MINUTES} min after FULL_DONE" ) &
