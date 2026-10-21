import json,glob,re,sys
run=sys.argv[1]
TL=re.compile(r'<this_level>\n(.*?)\n</this_level>',re.S)
NAMES={"W": "white", "w": "light gray", "g": "gray", "G": "dark gray", "c": "charcoal", "B": "black","M": "magenta", "P": "pink", "R": "red", "b": "blue", "S": "sky blue", "Y": "yellow","O": "orange", "r": "dark red", "N": "light green", "p": "purple"}
KEYS=('UP','DOWN','LEFT','RIGHT','SPACE','UNDO')
shown=0
for f in sorted(glob.glob(f'{run}/*.req.jsonl')):
    g=f.split('/')[-1][:4]
    acts=[json.loads(l) for l in open(f.replace('.req.','.act.'))]
    lvl_before={}; L=1
    for a in acts: lvl_before[a['n']]=L; L=a['level']
    for l in open(f):
        r=json.loads(l); nxt=int(r['action'] or 1)
        for m in r['new']:
            if m['role']!='user' or m['idx']!=r['n_msgs']-1: continue
            for b in TL.findall(m['text']):
                lines=b.split('\n'); lv=int(re.match(r'Level (\d+)',lines[0]).group(1))
                done=[a for a in acts if a['n']<nxt and lvl_before[a['n']]==lv]
                tried={a['a'] for a in done if a['a'] in KEYS}
                clicked={NAMES.get(a['clickcol']) for a in done if a['clickcol']}
                for ln in lines:
                    if ln.startswith('Not tried on this level:'):
                        u=set(x.strip(' .') for x in ln.split(':',1)[1].split(','))
                        if u&tried and shown<4:
                            shown+=1; print('== UNTRIED',g,r['i'],'next action',nxt,u&tried,[ (a['n'],a['a']) for a in done if a['a'] in u&tried][:5]); print(lines[0])
                    if ln.startswith('Colours on the board not clicked'):
                        u=set(re.sub(r' \(new on this level;[^)]*\)','',x).strip(' .') for x in ln.split(':',1)[1].split(', '))
                        if u&clicked and shown<8:
                            shown+=1; print('== COLOUR',g,r['i'],'next',nxt,u&clicked,[(a['n'],a['a'],a['clickcol'],a['changed']) for a in done if NAMES.get(a['clickcol']) in u&clicked][:4]); print(lines[0]); print(ln[:300])
