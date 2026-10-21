#!/bin/bash
S=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad
PY=/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/.venv/bin/python
D=/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle/v4-user-combo-10x/v1/artifacts
for f in $D/*_p0_events.jsonl.xz; do
  g=$(basename $f | cut -c1-4)
  [ -f $S/vision/bench/full/$g.npz ] && continue
  nice -n 15 $PY $S/vision/bench/gt_full.py --game $g --events $f --envdir $S/wt/openrouter/runs/local/environment_files --out $S/vision/bench/full 2>&1 | tail -2 | cut -c1-200
done
