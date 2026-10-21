#!/bin/zsh
cd /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/replay-sim/analysis
R=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/sched_results
PYTHONPATH=../ARC3-Inference nice -n 10 /Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/.venv/bin/python screen_policies.py \
  --never 0.4 --reps 40 --cache $R/cache.json --out $R/gap_screen.jsonl --policies $(cat $R/gap_names.txt)
echo EXIT $?
