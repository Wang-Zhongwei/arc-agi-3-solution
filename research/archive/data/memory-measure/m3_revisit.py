import json,collections,sys
from align import *
def analyze(g):
    G=load(g); E=G['E']; calls=G['calls']
    seen_state=collections.defaultdict(set)     # level -> states visited (after-states incl. initial)
    seen_pair={}                                # (level,before,display) -> (action_idx, after, call)
    seen_before=collections.defaultdict(dict)   # (level,before) -> {display: idx}
    attempt_states=set()
    acts=[]
    prev=E[0]; level=E[0]['level']; cur=E[0]['ih']; seen_state[level].add(cur); attempt_states={cur}
    pending_reset=False
    for e in E[1:]:
        if e['action_display']=='RESET':
            level=e['level']; cur=e['ih']; seen_state[level].add(cur); attempt_states={cur}; continue
        k=G['step_call'].get(e['analysis_step'])
        before=cur; lvl=level
        after=e['ih']
        lvlup=bool(e['level_completed'])
        noop=(after==before and not lvlup and not e['game_over'])
        a={'n':e['action_num'],'level':lvl,'disp':e['action_display'],'call':k,'noop':noop,'go':bool(e['game_over']),'lvlup':lvlup,'before':before,'after':after}
        if not noop and not lvlup:
            a['revisit_any']= after in seen_state[lvl]
            a['revisit_attempt']= after in attempt_states
        key=(lvl,before,e['action_display'])
        if key in seen_pair:
            j,aft_prev,kprev=seen_pair[key]
            a['pair_repeat']=True
            a['same_outcome']=(aft_prev==after)
            a['prev_noop']=(aft_prev==before)
            a['same_call']=(kprev==k)
            if k is not None and kprev is not None and kprev!=k:
                a['forgotten']= (retained(G,kprev,k)==False)
                a['trim_between']= any(kprev< t <=k for t in G['trims'])
        a['before_visited']=(lvl,before) in seen_before
        if a['before_visited']:
            a['new_action_at_visited']= e['action_display'] not in seen_before[(lvl,before)]
        seen_pair.setdefault(key,(e['action_num'],after,k))
        seen_before[(lvl,before)][e['action_display']]=e['action_num']
        acts.append(a)
        # advance
        if lvlup:
            level=e['level']; cur=after; seen_state[level].add(cur); attempt_states={cur}
        elif e['game_over']:
            cur=after  # next event is RESET which re-sets
        else:
            cur=after; seen_state[lvl].add(cur); attempt_states.add(cur)
    return acts,G
if __name__=='__main__':
    allres={}
    tot=collections.Counter()
    for g in games():
        acts,G=analyze(g)
        c=collections.Counter()
        for a in acts:
            c['actions']+=1
            c['noop']+=a['noop']
            if not a['noop'] and not a['lvlup'] and not a['go']:
                c['moves']+=1
                c['revisit_any']+=bool(a.get('revisit_any'))
                c['revisit_attempt']+=bool(a.get('revisit_attempt'))
            if a.get('pair_repeat'):
                c['pair_repeat']+=1
                c['pair_repeat_same_outcome']+=bool(a['same_outcome'])
                c['pair_repeat_prev_noop']+=bool(a['prev_noop'])
                c['pair_repeat_same_call']+=bool(a['same_call'])
                if 'forgotten' in a:
                    c['pair_repeat_cross_call']+=1
                    c['pair_repeat_forgotten']+=bool(a['forgotten'])
                    c['pair_repeat_trim_between']+=bool(a['trim_between'])
                    c['pair_repeat_forgotten_nonnoop']+=bool(a['forgotten'] and not a['prev_noop'])
                    c['pair_repeat_cross_nonnoop']+=bool(not a['prev_noop'])
            if a['before_visited']:
                c['at_visited']+=1; c['at_visited_new_action']+=bool(a['new_action_at_visited'])
        allres[g]=dict(c); tot.update(c)
        print(g[:4],dict(c))
    print('TOTAL',dict(tot))
    json.dump(allres,open('m3_revisit.json','w'))
