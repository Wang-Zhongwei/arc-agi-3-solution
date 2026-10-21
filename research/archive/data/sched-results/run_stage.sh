#!/bin/zsh
# usage: run_stage.sh <out.jsonl> <reps> <extra screen args...> -- <names...>
# runs ONE screen_policies.py invocation, niced; never in parallel
cd /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/replay-sim/analysis
R=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/sched_results
out=$1; reps=$2; shift 2
extra=(); while [[ $1 != "--" ]]; do extra+=$1; shift; done; shift
PYTHONPATH=../ARC3-Inference nice -n 10 /Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/.venv/bin/python screen_policies.py \
  --reps $reps --cache $R/cache.json --out $R/$out $extra --policies "$@"
echo EXIT $?
