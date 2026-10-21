exec(open('/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/memv2-analysis/work/verify/cn04/fin.py').read().split("print('selected'")[0])
import numpy as np
def tips(g,pref):
    s=[s for s in g.current_level.get_sprites() if s.name.startswith(pref)][0]
    o=g.hlxyvcmpk[s.name]; cur=s.pixels.copy(); s.pixels=o; r=s.render(); s.pixels=cur
    return s.x,s.y,s.rotation,[((s.x+x,s.y+y),int(r[y,x])) for y in range(r.shape[0]) for x in range(r.shape[1]) if r[y,x] in (8,13)]
g=replay(98)
def owner(g,gx,gy):
    for sp in g.current_level.get_sprites():
        if g.ixutchviko(sp,gx,gy,tqpvgsnhag=True) and sp.is_visible: return sp.name
p=[s for s in g.current_level.get_sprites() if s.name.startswith('0004')][0]
r=p.render()
cells=[(p.x+x,p.y+y) for y in range(r.shape[0]) for x in range(r.shape[1]) if r[y,x]>=0 and owner(g,p.x+x,p.y+y)==p.name]
print('p-clickable cells',cells)
gx,gy=cells[0]
CL=[(dx,dy) for dx in range(64) for dy in range(64) if g.camera.display_to_grid(dx,dy)==(gx,gy)][0]
print('click',CL)
for k in range(4):
    g=replay(98)
    g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x':CL[0],'y':CL[1]}))
    print('sel', g.xseexqzst.name)
    for _ in range(k): g.perform_action(ActionInput(id=GameAction.ACTION5))
    print(k, tips(g,'0004'))
print('Y', tips(g,'0002'))
names={'U':GameAction.ACTION1,'D':GameAction.ACTION2,'L':GameAction.ACTION3,'R':GameAction.ACTION4,'S':GameAction.ACTION5}
for seq in ['SSLUUU','SSUUUL']:
    g=replay(98)
    fd=g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x':CL[0],'y':CL[1]}))
    for i,ch in enumerate(seq):
        fd=g.perform_action(ActionInput(id=names[ch]))
        print(seq, i, ch, fd.levels_completed, fd.state)
