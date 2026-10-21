import sys, os, json, importlib.util, io, contextlib
WD='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/memv2-analysis/work/m0r0-492f87ba'
os.chdir(WD)
src3=open('sim3.py').read()
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('sim.py').read().split("if __name__")[0])
    exec(src3.split("def solve_from")[0].replace("exec(open('sim.py').read().split(\"if __name__\")[0])",""))
SEL0=0
exec("def solve_from"+src3.split("def solve_from")[1])
W_,H_,walls,haz,blocks0,P0,Q0,doors,plates=parse(6)
path=solve_from(6,P0,Q0)
print('analyst path from (start, block selected):',len(path),' '.join(path))
GAME='/Users/advaith/Documents/studying-stuff/experiments/arc-agi-3/data/environment_files/m0r0/492f87ba/m0r0.py'
R='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/runs-out/mem-v2-10x'
spec=importlib.util.spec_from_file_location('m0r0src', GAME); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from arcengine import ActionInput, GameAction
g=mod.M0r0()
b=json.load(open(R+'/benchmark.json'))
run=[x for x in b['game_runs'] if x['game_id'].startswith('m0r0')][0]
g.perform_action(ActionInput(id=GameAction.RESET))
N=int(sys.argv[1]) if len(sys.argv)>1 else 276
for h in run['history'][:N]:
    a=h['action']; fd=g.perform_action(ActionInput(id=GameAction[a['id']] if a['id']!='RESET' else GameAction.RESET,data=a.get('data') or {}))
blk=g.current_level.get_sprites_by_name('mosdlc')[0]
print('after',N,'levels',fd.levels_completed,'selected',g.jnyxgyktgcv is not None, 'block at',blk.x,blk.y)
MV={'U':GameAction.ACTION1,'D':GameAction.ACTION2,'L':GameAction.ACTION3,'R':GameAction.ACTION4}
for i,a in enumerate(path):
    if a in MV: fd=g.perform_action(ActionInput(id=MV[a]))
    elif a.startswith('C') and a!='C-':
        blk=g.current_level.get_sprites_by_name('mosdlc')[0]
        fd=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':4*blk.x+7,'y':4*blk.y+7}))
    else:
        fd=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':7,'y':7}))
    if fd.levels_completed>5 or fd.state.name!='NOT_FINISHED':
        print('after',i+1,'actions:',fd.levels_completed,fd.state); break
print('final levels',fd.levels_completed,fd.state)
