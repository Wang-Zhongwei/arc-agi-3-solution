import json,collections
from align import *
def run_game(g):
    G=load(g); E=G['E']
    # build sequence of attempts: list of lists of action dicts with before/after
    attempts=[]; cur=[]; level=E[0]['level']; state=E[0]['ih']; start_state={level:state}
    att_level=level
    seq=[]  # flat with attempt id
    aid=0
    for e in E[1:]:
        if e['action_display']=='RESET':
            aid+=1; state=e['ih']; level=e['level']; continue
        before=state
        a={'aid':aid,'level':level,'disp':e['action_display'],'before':before,'after':e['ih'],'n':e['action_num'],'go':bool(e['game_over']),'lvlup':bool(e['level_completed']),'call':G['step_call'].get(e['analysis_step'])}
        seq.append(a)
        if e['level_completed']:
            aid+=1; level=e['level']; state=e['ih']
        elif e['game_over']:
            state=e['ih']
        else:
            state=e['ih']
    return seq,G
tot=collections.Counter(); per={}
for g in games():
    seq,G=run_game(g)
    c=collections.Counter()
    seen_pair={}; 
    # per attempt visit index for loops
    visits=collections.defaultdict(dict)  # aid -> state -> last idx in attempt order
    attempt_idx=collections.Counter()
    loop_cover=collections.defaultdict(set)
    closure_next=[]
    for i,a in enumerate(seq):
        c['actions']+=1
        key=(a['level'],a['before'],a['disp'])
        if key in seen_pair:
            p=seen_pair[key]
            c['pair_repeat']+=1
            if p['aid']==a['aid']: c['pair_repeat_same_attempt']+=1
            else: c['pair_repeat_replay_other_attempt']+=1
            if p['call']==a['call']: c['pair_repeat_same_call']+=1
        else:
            seen_pair[key]=a
        # loops within attempt
        aid=a['aid']; ai=attempt_idx[aid]; attempt_idx[aid]+=1
        if ai==0: visits[aid][a['before']]=-1
        if a['after']!=a['before'] and not a['lvlup']:
            if a['after'] in visits[aid]:
                s=visits[aid][a['after']]
                c['loop_closures']+=1
                loop_cover[aid].update(range(s+1,ai+1))
                c['loop_len_sum']+=ai-s
                closure_next.append((aid,i))
            visits[aid][a['after']]=ai
    c['loop_covered_actions']=sum(len(v) for v in loop_cover.values())
    # next action after closure: repeat of pair previously taken from that state?
    # reconstruct pair history set per attempt
    hist=collections.defaultdict(set)
    for i,a in enumerate(seq):
        hist[(a['aid'],a['before'])].add(a['disp'])
    # evaluate
    first=set()
    seen_in_attempt=collections.defaultdict(set)
    for i,a in enumerate(seq):
        seen_in_attempt[(a['aid'],a['before'])]
    for (aid,i) in closure_next:
        if i+1<len(seq) and seq[i+1]['aid']==aid:
            nxt=seq[i+1]
            prior=set(x['disp'] for x in seq[:i+1] if x['aid']==aid and x['before']==nxt['before'])
            c['closure_next_total']+=1
            c['closure_next_repeat']+= (nxt['disp'] in prior)
    per[g]=dict(c); tot.update(c)
    print(g[:4],dict(c))
print('TOTAL',dict(tot))
json.dump(per,open('m3b.json','w'))
