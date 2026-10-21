import json,re,sys
from collections import defaultdict
path=sys.argv[1]
ev=[json.loads(l) for l in open(path)]
ev=[e for e in ev if e.get('type') in ('initial','action')]
def board(e):
    b=e['board_ascii']; 
    return b.split('\n') if isinstance(b,str) else b
def comps(g, colors='MpY'):
    seen=set(); out=[]
    for r in range(10,63):
        for c in range(64):
            ch=g[r][c]
            if ch in colors and (r,c) not in seen:
                st=[(r,c)]; seen.add((r,c)); cells=[]
                while st:
                    a,b=st.pop(); cells.append((a,b))
                    for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x,y=a+da,b+db
                        if 10<=x<63 and 0<=y<64 and (x,y) not in seen and g[x][y]==ch:
                            seen.add((x,y)); st.append((x,y))
                rs=[x for x,_ in cells]; cs=[y for _,y in cells]
                out.append((ch,min(rs),min(cs),max(rs),max(cs),len(cells)))
    return out
def sparkle(g):
    pts=[(r,c) for r in range(10,63) for c in range(64) if g[r][c]=='P']
    if not pts: return None
    return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))
rows=[]
for k in range(1,len(ev)):
    a=ev[k]; prev=ev[k-1]
    m=re.match(r'MOUSE\(row=(\d+), col=(\d+)\)', a.get('action_display') or '')
    if not m or a.get('level')!=prev.get('level'): continue
    cr,cc=int(m.group(1)),int(m.group(2))
    g0=board(prev); g1=board(a)
    P0=comps(g0); P1=set(comps(g1))
    s0=sparkle(g0); s1=sparkle(g1)
    for p in P0:
        ch,r0,c0,r1,c1,n=p
        if n not in (4,9,16): continue
        stayed = p in P1
        dtl=max(abs(cr-r0),abs(cc-c0))
        rr=min(max(cr,r0),r1); cc2=min(max(cc,c0),c1)
        dnear=max(abs(cr-rr),abs(cc-cc2))
        cen=((r0+r1)/2,(c0+c1)/2); dcen=max(abs(cr-cen[0]),abs(cc-cen[1]))
        eaten = s1 is not None and abs(s1[0]-cen[0])<4 and abs(s1[1]-cen[1])<4
        rows.append((a['level'],a['action_num'],(cr,cc),ch,n,(r0,c0),dtl,dnear,round(dcen,1),'stay' if stayed else 'MOVE', 'nearS' if eaten else ''))
    if s0 and s1:
        ds=max(abs(cr-s0[0]),abs(cc-s0[1])); mv=((s1[0]-s0[0]),(s1[1]-s0[1]))
        rows.append((a['level'],a['action_num'],(cr,cc),'SPARK',round(ds,1),'move',tuple(round(x,1) for x in mv), round(max(abs(mv[0]),abs(mv[1])),1)))
for r in rows: print(r)
