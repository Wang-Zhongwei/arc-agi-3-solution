#!/bin/zsh
# Push arms one at a time as GPU slots free up, never starting two within
# STAGGER seconds: two simultaneous cold starts reading the ~190 GB model from
# the input mount left both servers hung (fz-arms v1/v2), while a lone start
# was healthy. Set NEW_BATCH_ARM=<arm> to start a new batch with that arm.
# <arm>@<N> pushes run N of an arm that already ran (build_notebook.py --replica N).
# Set GATE_PASSED="<arm> ..." to push gated arms whose gate holds (--gate-passed).
# Usage: kaggle/push_queue.sh <arm>[@N] [<arm>[@N] ...]   (run from the repo root)
STAGGER=${STAGGER:-1200}
cd "${0:A:h}/.." || exit 1
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" KAGGLE_USERNAME=mellamoa
first=1
for item in "$@"; do
  arm=${item%@*}; replica=1; [[ $item == *@* ]] && replica=${item##*@}
  [ $first = 1 ] || { echo "$(date -u +%H:%M) staggering ${STAGGER}s before $item"; sleep $STAGGER; }
  first=0
  while true; do
    flags=(); [ "$arm" = "${NEW_BATCH_ARM:-}" ] && flags=(--new-batch)
    [ "$replica" = 1 ] || flags+=(--replica "$replica")
    [[ " ${GATE_PASSED:-} " == *" $arm "* ]] && flags+=(--gate-passed)
    out=$(python3 kaggle/build_notebook.py --arm "$arm" --push $flags 2>&1)
    if echo "$out" | grep -q "^recorded"; then
      v=$(echo "$out" | grep -o 'version [0-9]*' | tail -1)
      git add kaggle/runs.json && git commit -q -m "runs: $item as v4-$arm $v"
      echo "$(date -u +%H:%M) pushed $item: $v"; break
    fi
    if echo "$out" | grep -q "not building a duplicate"; then echo "$(date -u +%H:%M) $item already ran; skipping"; break; fi
    if ! echo "$out" | grep -qi "Maximum batch GPU session"; then
      echo "$(date -u +%H:%M) $item: unexpected failure, stopping queue:"; echo "$out"; exit 1
    fi
    echo "$(date -u +%H:%M) $item waiting for a free GPU slot"
    sleep 600
  done
done
echo "queue done"
