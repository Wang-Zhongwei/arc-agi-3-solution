#!/bin/zsh
# sequential, one simulation process at a time
S=/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/sched_results/run_stage.sh
TOP3=('dl-hRg:pn0.4' 'dl-hQ:pn0' 'aff-v1:nofresh')
# 1. finals at penalty 0 (dl-hRg:pn0.4, aff-v1:nofresh, dl-hQ:pn0.4, end-fox-km:e0.3 already have 200 reps)
for nv in 0 0.4 0.8; do $S final.jsonl 200 --never $nv -- 'dl-hQ:pn0'; done
# 2. penalty 3000 at every never (top 3 + the two endgame-freeze configs, which are defined at 3000)
for nv in 0 0.4 0.8; do
  if [[ $nv == 0.4 ]]; then names=('dl-hQ:pn0'); else names=($TOP3); fi
  $S final_p3000.jsonl 200 --never $nv --penalty 3000 -- $names
  $S final_p3000.jsonl 200 --never $nv -- 'end-srpt-km:e0.3,f0.03,p3000' 'end-fox-km:e0.3,f0.03,p3000'
done
# 3. sensitivity: action model with no clearability trace, and a misfit KM tail
for nv in 0.4 0.8; do $S sens.jsonl 100 --never $nv --censored-actions observed -- $TOP3 'dl-hQ:pn0.4' 'dprime-float'; done
for rs in 2 0.5; do $S sens.jsonl 100 --never 0.4 --resid-scale $rs -- $TOP3 'dl-hQ:pn0.4' 'dprime-float'; done
# 4. M43 without its switch penalty (separates the penalty from the policy)
$S misc.jsonl 40 --never 0.4 --penalty 0 -- 'm43:x20,thr0' 'm43:x20,thr4'
$S misc.jsonl 40 --never 0.4 --censored-actions observed -- 'm43:x20,thr0' 'm43:x20,thr4'
echo ALLDONE
