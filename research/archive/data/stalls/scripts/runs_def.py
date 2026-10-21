"""Wins and time-to-win (bench minutes from model-ready) per game, across full 10x runs."""
import json, sys
R = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/runs/kaggle'
K = '/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs'
RUNS = {
    'A': f'{R}/v4-pf-solved-10x/v1', 'B': f'{R}/v4-pf-keep-10x/v1', 'comb': f'{R}/v4-pf-keep-v2-all-10x/v1',
    'C': f'{R}/v4-pf-solved-cap8k-10x/v1-snowiers', 'keep': f'{R}/v4-mem-recap-keep-10x/v1',
    'ms1': f'{R}/v4-mem-solved-10x/v1', 'ms2': f'{R}/v4-mem-solved-10x/v2-snowiers',
    'uc1': f'{R}/v4-user-combo-10x/v1', 'uc2': f'{R}/v4-user-combo-10x/v2-snowiers', 'uc3': f'{R}/v4-user-combo-10x/v3-snowiers',
    'U': f'{K}/v4-mem-unified-10x/v1', 'Ucore': f'{K}/v4-mem-unified-core-10x/v1', 'SF': f'{K}/v4-stall-fix-10x/v1',
}
