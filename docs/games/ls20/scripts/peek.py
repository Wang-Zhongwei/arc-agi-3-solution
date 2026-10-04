"""peek.py <run_rel> <pass> <action_lo> <action_hi> [maxchars]  -> thinking + tool errors of analysis steps in range."""
import os, json, re, sys
from pathlib import Path

R = Path(os.environ.get('ARC3_RUNS_DIR', Path(__file__).resolve().parents[4] / 'ARC3-Inference/runs')) / 'kaggle'
rel, pas, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 600
rows = [json.loads(l) for l in open(R / rel / f'artifacts/ls20-9607627b_p{pas}_events.jsonl')]
for r in rows:
    if r.get('type') == 'action' and lo <= r['action_num'] <= hi and (r.get('game_over') or r.get('level_completed')):
        print(f"### action {r['action_num']} level={r['level']} game_over={r.get('game_over')} level_completed={r.get('level_completed')}")
    if r.get('type') != 'analysis' or not (lo <= r['action_num'] <= hi):
        continue
    t = r['transcript'].split('[USER PROMPT]', 1)[-1]
    th = re.findall(r'\[THINKING\]\n(.*?)(?=\n\[[A-Z][A-Z _:]+\])', t, re.S)
    errs = re.findall(r'^(\w+(?:Error|Exception): .*)$', t, re.M)
    acts = re.findall(r'action\((\[[^\]]*\])', t)
    txt = ' | '.join(x.strip().replace('\n', ' ') for x in th)
    print(f"--- step {r['analysis_step']} @action {r['action_num']} lvl {r['level']}: {txt[:mx]}")
    if errs: print('    ERR:', errs[:3])
    if acts: print('    ACT:', [a[:80] for a in acts[:3]])
