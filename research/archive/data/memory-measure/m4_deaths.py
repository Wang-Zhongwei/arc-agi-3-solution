import json,glob,os,re,collections,statistics as st
from align import *
BURN_PRE=re.compile(r"(burn|trigger (a |the )?reset|waste (the|my|some|a few)|let (it|the bar|the budget) (die|deplete|run out|expire)|let it die|intentional|deliberately|end the attempt|force (a )?reset|just die|kill (it|myself|the)|suicide|reset (the level )?(now|early)|restart the level)",re.I)
BURN_POST=re.compile(r"(intentional|as intended|\(intended\)|that's what i wanted|on purpose|as planned|✓ that's|deliberate)",re.I)
BUDGET=re.compile(r"(budget (depletion|death|ran out|exhausted|is gone)|bar (was |is )?(fully )?(depleted|empty|0|zero|gone)|depleted|ran out of (steps|moves|budget)|out of (steps|moves|budget)|timer (ran|expired)|death by (budget|bar|timer)|bar hit 0|bar reached 0)",re.I)
NOTBUD=re.compile(r"(not (fully )?depleted|wasn't depleted|was not depleted|budget was not|NOT depleted|bar (had|was at|showed) \d+|not the budget|isn't the budget)",re.I)
def classify(pre,post):
    pre=pre[-900:]; post=post[:1200]
    notb=bool(NOTBUD.search(post))
    bud=bool(BUDGET.search(post)) and not notb
    intent=bool(BURN_PRE.search(pre)) or bool(BURN_POST.search(post))
    if bud and intent: return 'budget_intentional_burn'
    if bud: return 'budget_unintended'
    if notb: return 'action_caused'
    return 'action_caused_or_unclear'
def deaths_for(run):
    out=[]
    for p in sorted(glob.glob(f'parsed/{run}__*.json')):
        b=os.path.basename(p)
        if '__req__' in b or b.startswith('ev__'): continue
        g=b.split('__')[1][:-5]
        T=json.load(open(p)); calls=[(ti,t,c) for ti,t in enumerate(T) for c in t['calls']]
        for idx,(ti,t,c) in enumerate(calls):
            if '-> GAME OVER' in c['result']:
                th=''
                for j in range(idx+1,min(idx+4,len(calls))):
                    if calls[j][2]['thinking']: th=calls[j][2]['thinking']; break
                # number of actions in this call batch
                m=re.search(r'batch of (\d+) actions?, (\d+) executed',c['result'])
                nb=int(m.group(2)) if m else None
                out.append(dict(run=run,game=g[:4],level=t['level'],action=t['action'],cls=classify(c['thinking'],th),batch=nb,pre=c['thinking'][-300:],post=th[:350],code=c['code'][:200]))
    return out
if __name__=='__main__':
    allr={}
    for run in ['fz-ledger-on-trim','fz-ledger-on-trim-r2','fz-arms-v4-control']:
        D=deaths_for(run); allr[run]=D
        cnt=collections.Counter(d['cls'] for d in D)
        print(run,len(D),dict(cnt),'by game',dict(collections.Counter(d['game'] for d in D)))
    json.dump(allr,open('m4_deaths.json','w'))
