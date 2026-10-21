import json,sys,re,glob
G='/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/memv2-analysis/games'
game=sys.argv[1]; calls=list(map(int,sys.argv[2].split(',')))
md=open(glob.glob(f'{G}/{game}*.md')[0]).read().split('\n')
cur=None; ct={}
for line in md:
    m=re.match(r'## step (\d+) \| action (\d+) \| (\S+)',line)
    if m: cur=(int(m.group(1)),int(m.group(2)),m.group(3))
    m=re.match(r'### thinking \(call (\d+)\)',line)
    if m: ct[int(m.group(1))]=cur
for c in calls:
    for k in range(c-1,c+3): print(game,k,ct.get(k))
    print()
