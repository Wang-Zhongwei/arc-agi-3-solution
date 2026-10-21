#!/bin/zsh
S=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/sched_results/run_stage.sh
$S misc.jsonl 40 --never 0.4 -- 'm43-fresh:x10,thr0' 'm43-fresh:x20,thr0' 'm43-fresh:x40,thr0' 'm43-fresh:x20,thr4'
$S misc.jsonl 40 --never 0.4 --penalty 0 -- 'm43-fresh:x20,thr0'
$S sens.jsonl 100 --never 0 --censored-actions observed -- 'dl-hRg:pn0.4' 'dl-hQ:pn0' 'aff-v1:nofresh' 'dl-hQ:pn0.4' 'dprime-float'
cd /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/replay-sim/analysis
PYTHONPATH=../ARC3-Inference nice -n 10 /Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/.venv/bin/python /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/sched_results/trace_top.py
echo ALLDONE
