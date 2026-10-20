#!/bin/bash
# Local watcher for full_run.sh / train_pod.sh: when the pod writes FULL_DONE or ABORTED, pull its results and
# end it: POD_END=stop (default: GPU billing ends, the pod and its volume stay) or POD_END=delete.
#   bash inference/rl/pod/watch_pod.sh POD_ID HOST PORT DEST_DIR
# RUNPOD_API_KEY comes from the repo's .env (never printed). Exits when the pod is stopped, gone or unreachable.
# NO_DELETE=1 pulls the results only (to test the pod's own self-stop afterwards).
# WORKDIR, RUNLOG and PULL (paths under WORKDIR) default to full_run.sh's; SYNC_MINUTES > 0 also pulls PULL that
# often while the run goes on (train_pod.sh: the adapters saved so far survive a pod that dies).
set -uo pipefail
ID=$1 HOST=$2 PORT=$3 DEST=$4
WORKDIR=${WORKDIR:-/workspace/smoke}
RUNLOG=${RUNLOG:-full_run.log}
PULL=${PULL:-results jobs jobs.txt full_run.log}
SYNC_MINUTES=${SYNC_MINUTES:-0}
ENV_FILE=${ENV_FILE:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/.env}
KEY=$(grep -E '^RUNPOD_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
SSH="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p $PORT root@$HOST"
mkdir -p "$DEST"
pull() {
    $SSH "cd $WORKDIR && tar czf /tmp/pull.tgz $PULL 2>/dev/null" < /dev/null
    scp -q -o StrictHostKeyChecking=no -P "$PORT" "root@$HOST:/tmp/pull.tgz" "$DEST/" \
        && tar xzf "$DEST/pull.tgz" -C "$DEST" && rm "$DEST/pull.tgz"
}
misses=0 pull_failures=0 last_sync=$(date +%s)
while true; do
    state=$($SSH "cd $WORKDIR; for f in FULL_DONE ABORTED; do [ -e \$f ] && echo \$f; done; tail -1 $RUNLOG" < /dev/null 2>/dev/null)
    if [ -z "$state" ]; then
        misses=$((misses + 1)); [ $misses -ge 20 ] && { echo "pod $ID unreachable for 10 min: stopping the watch"; exit 1; }
    else
        misses=0
        if echo "$state" | grep -qE "^(FULL_DONE|ABORTED)$"; then
            if ! pull; then
                # keep the pod running (and its results) for another try; after 5 failures stop it anyway (billing)
                pull_failures=$((pull_failures + 1))
                echo "pod $ID: pulling the results failed ($pull_failures)"
                if [ $pull_failures -lt 5 ]; then sleep 30; continue; fi
            fi
            code=skipped
            if [ "${NO_DELETE:-0}" != 1 ] && [ "${POD_END:-stop}" = delete ]; then
                code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
                    -H "Authorization: Bearer $KEY" -A "Mozilla/5.0" "https://rest.runpod.io/v1/pods/$ID")
            elif [ "${NO_DELETE:-0}" != 1 ]; then
                code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
                    -H "Authorization: Bearer $KEY" -A "Mozilla/5.0" "https://rest.runpod.io/v1/pods/$ID/stop")
            fi
            echo "pod $ID: $(echo "$state" | grep -E '^(FULL_DONE|ABORTED)$'); results in $DEST; ${POD_END:-stop} HTTP $code"
            exit 0
        fi
        if [ "$SYNC_MINUTES" -gt 0 ] && [ $(( $(date +%s) - last_sync )) -ge $((SYNC_MINUTES * 60)) ]; then
            pull && echo "[$(date -u +%H:%M:%S)] synced: $(echo "$state" | tail -1 | cut -c1-120)"
            last_sync=$(date +%s)
        fi
    fi
    sleep 30
done
