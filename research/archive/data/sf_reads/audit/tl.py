import json,glob,re,sys
from collections import Counter,defaultdict
run=sys.argv[1]
TL=re.compile(r'<this_level>\n(.*?)\n</this_level>',re.S)
NAMES={"W": "white", "w": "light gray", "g": "gray", "G": "dark gray", "c": "charcoal", "B": "black","M": "magenta", "P": "pink", "R": "red", "b": "blue", "S": "sky blue", "Y": "yellow","O": "orange", "r": "dark red", "N": "light green", "p": "purple"}
KEYS=('UP','DOWN','LEFT','RIGHT','SPACE','UNDO')
rowtype=Counter(); nblk=0; skipped=Counter(); skipped_blocks=0; wrong_untried=0; wrong_col=0; ex=defaultdict(list)
lens=[]; extra_len=[]; budget_part=[]
per_game=Counter()
for f in sorted(glob.glob(f'{run}/*.req.jsonl')):
    g=f.split('/')[-1][:4]
    acts=[json.loads(l) for l in open(f.replace('.req.','.act.'))]
    lvl_before={}; L=1
    for a in acts: lvl_before[a['n']]=L; L=a['level']
    for l in open(f):
        r=json.loads(l)
        nxt=int(r['action'] or 1)
        for m in r['new']:
            if m['role']!='user' or m['idx']!=r['n_msgs']-1: continue
            for b in TL.findall(m['text']):
                nblk+=1; lines=b.split('\n')
                lv=int(re.match(r'Level (\d+)',lines[0]).group(1))
                done=[a for a in acts if a['n']<nxt and lvl_before[a['n']]==lv]
                tried={a['a'] for a in done if a['a'] in KEYS}
                clicked={NAMES.get(a['clickcol']) for a in done if a['clickcol']}
                shownkeys=set(); shownclick=set(); untried=set(); notclicked=set(); nclickrows=0
                budget=0
                for ln in lines[1:]:
                    k=None
                    mm=re.match(r'(UP|DOWN|LEFT|RIGHT|SPACE|UNDO) \(',ln)
                    if mm: k='key'; shownkeys.add(mm.group(1))
                    elif ln.startswith('Clicks on '):
                        k='click'; nclickrows+=1; shownclick.add(re.match(r'Clicks on (.*?) cells',ln).group(1))
                    elif ln.startswith('Not tried on this level:'):
                        k='untried'; untried=set(x.strip(' .') for x in ln.split(':',1)[1].split(','))
                    elif ln.startswith('Colours on the board not clicked'):
                        k='colours'; notclicked=set(re.sub(r' \(new on this level;[^)]*\)','',x).strip(' .') for x in ln.split(':',1)[1].split(', ') )
                    elif re.match(r'Level \d+: attempt \d+ ended in GAME OVER',ln): k='hazard'
                    elif ln.startswith('The current board was seen before'): k='seen'
                    elif ln.startswith('Your note'): k='note'
                    elif ln.startswith('Rules you confirmed'): k='rules'
                    elif ln.startswith('Moved since this level started'): k='moved'
                    else: k='other:'+ln[:40]
                    rowtype[k]+=1
                    if k in ('key','click','untried'): budget+=len(ln)+1
                budget_part.append(budget)
                lens.append(len(b))
                miss_keys=tried-shownkeys
                miss_click=(clicked-shownclick-{None}) if nclickrows<6 else set()
                if miss_keys or miss_click:
                    skipped_blocks+=1; per_game[g]+=1
                    for k in miss_keys: skipped['key']+=1
                    for k in miss_click: skipped['click']+=1
                    if len(ex[g])<2: ex[g].append((r['i'],sorted(miss_keys),sorted(miss_click),b))
                bad=untried & tried
                if bad: wrong_untried+=1
                badc=notclicked & clicked
                if badc: wrong_col+=1
print(run,'blocks',nblk)
print('row types',rowtype.most_common(20))
print('blocks with a tried key/clicked colour missing its row:',skipped_blocks, dict(skipped), dict(per_game))
print('untried row lists a tried key:',wrong_untried,' colours row lists a clicked colour:',wrong_col)
import statistics as st
print('effect-rows chars (key+click+untried) median',st.median(budget_part),'n>600',sum(1 for x in budget_part if x>600))
json.dump({g:v for g,v in ex.items()},open(f'{run}_tl_skips.json','w'),indent=1)
