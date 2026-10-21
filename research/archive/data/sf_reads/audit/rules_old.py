import json, glob, re, os, sys
from collections import defaultdict, Counter
run=sys.argv[1]
TL=re.compile(r'<this_level>\n(.*?)\n</this_level>',re.S)
LV=re.compile(r'Level (\d+) so far')
rules={}  # (g, lvl, text) -> dict
for f in sorted(glob.glob(f'{run}/*.req.jsonl')):
    g=os.path.basename(f)[:4]
    for l in open(f):
        r=json.loads(l)
        for m in r['new']:
            if m['role']!='user': continue
            for blk in TL.findall(m['text']):
                lv=LV.search(blk); lv=int(lv.group(1)) if lv else None
                for line in blk.split('\n'):
                    if line.startswith('Rules you confirmed on this level: '):
                        body=line[len('Rules you confirmed on this level: '):]
                        for q in re.findall(r'"(.*?)"(?: \| |$)',body):
                            k=(g,lv,q)
                            d=rules.setdefault(k,{'g':g,'lvl':lv,'rule':q,'src':'this','first':r['i'],'last':r['i'],'n':0})
                            d['last']=r['i']; d['n']+=1
        if r.get('pl'):
            cur=None
            for line in r['pl'].split('\n'):
                m2=re.match(r'Level (\d+): cleared',line)
                if m2: cur=int(m2.group(1))
                if line.startswith('  Rules you confirmed: '):
                    for q in re.findall(r'"(.*?)"(?: \| |$)',line[len('  Rules you confirmed: '):]):
                        k=(g,cur,q)
                        if k in rules:
                            rules[k].setdefault('prev_first',r['i'])
                        else:
                            rules[k]={'g':g,'lvl':cur,'rule':q,'src':'prev_only','first':r['i'],'last':r['i'],'n':0,'prev_first':r['i']}
json.dump(list(rules.values()),open(f'{run}_rules_old.json','w'),indent=0,ensure_ascii=False)
c=Counter(d['g'] for d in rules.values())
print(len(rules),'distinct rules', dict(c))
print(Counter(d['src'] for d in rules.values()))
