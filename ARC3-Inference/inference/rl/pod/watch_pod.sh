#!/bin/bash
# Local watcher for full_run.sh: when the pod writes FULL_DONE or ABORTED, pull its results and delete it.
#   bash inference/rl/pod/watch_pod.sh POD_ID HOST PORT DEST_DIR
# RUNPOD_API_KEY comes from the repo's .env (never printed). Exits when the pod is gone or deleted.
# NO_DELETE=1 pulls the results only (to test the pod's own self-delete afterwards).
set -uo pipefail
ID=$1 HOST=$2 PORT=$3 DEST=$4
ENV_FILE=${ENV_FILE:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/.env}
KEY=$(grep -E '^RUNPOD_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
SSH="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p $PORT root@$HOST"
mkdir -p "$DEST"
misses=0 pull_failures=0
while true; do
    state=$($SSH 'cd /workspace/smoke; for f in FULL_DONE ABORTED; do [ -e $f ] && echo $f; done; tail -1 full_run.log' < /dev/null 2>/dev/null)
    if [ -z "$state" ]; then
        misses=$((misses + 1)); [ $misses -ge 20 ] && { echo "pod $ID unreachable for 10 min: stopping the watch"; exit 1; }
    else
        misses=0
        if echo "$state" | grep -qE "^(FULL_DONE|ABORTED)$"; then
            $SSH 'cd /workspace/smoke && tar czf /tmp/full_run.tgz results jobs jobs.txt full_run.log 2>/dev/null' < /dev/null
            if ! { scp -q -o StrictHostKeyChecking=no -P "$PORT" "root@$HOST:/tmp/full_run.tgz" "$DEST/" \
                   && tar xzf "$DEST/full_run.tgz" -C "$DEST" && rm "$DEST/full_run.tgz"; }; then
                # keep the pod (and its results) for another try; after 5 failures delete it anyway (billing)
                pull_failures=$((pull_failures + 1))
                echo "pod $ID: pulling the results failed ($pull_failures)"
                if [ $pull_failures -lt 5 ]; then sleep 30; continue; fi
            fi
            code=skipped
            [ "${NO_DELETE:-0}" = 1 ] || code=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
                -H "Authorization: Bearer $KEY" -A "Mozilla/5.0" "https://rest.runpod.io/v1/pods/$ID")
            echo "pod $ID: $(echo "$state" | grep -E '^(FULL_DONE|ABORTED)$'); results in $DEST; delete HTTP $code"
            exit 0
        fi
    fi
    sleep 30
done
