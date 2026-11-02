"""For each passing candidate: useful lines caught with reader's info type; top-game concentration (does it pass without
its largest game?); evidence mutations on useful rule lines; overlap within a phase."""
import sys,importlib.util,json,collections as C
sys.dont_write_bytecode=True
sys.path.insert(0,'/Users/wang/.claude/jobs/910e124b/tmp/scripts')
sys.path.insert(0,'/Users/wang/.claude/jobs/910e124b/tmp/verify/noise/critic')
import noise_harness as H
ND='/Users/wang/.claude/jobs/910e124b/tmp/verify/noise/'
def load(n):
    s=importlib.util.spec_from_file_location(n,ND+n+'.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V={p:H.views(p) for p in H.PH}
D=json.load(open(ND+'dev_lines_v02.json'))
INFO={p:{x['id']:x for x in D[p]} for p in D}
MUT={'+coord_aside':lambda s:s.rstrip()+' (seen at (12,30))','+step_aside':lambda s:s.rstrip()+' (at step 40)',
     '+bracket_tag':lambda s:s.replace('✓ (confirmed)','[confirmed]').replace('(confirmed)','[confirmed]').replace('✓ confirmed','[confirmed]') ,
     'now_in_main':lambda s:s.replace('Confirmed:','Confirmed: now,',1) if 'Confirmed:' in s else 'Now '+s}
def audit(mod,names,ph):
    m=load(mod); rd=H.PH[ph]
    for nm in names:
        p,fn=m.CANDIDATES[nm]
        hit=[r for r,l in V[ph] if fn(l)]
        c=C.Counter(H.lab(r,rd) for r in hit); n=len(hit)
        by=C.Counter(r['game'] for r in hit)
        top,tn=by.most_common(1)[0] if by else ('',0)
        rest=[r for r in hit if r['game']!=top]
        cr=C.Counter(H.lab(r,rd) for r in rest); nr=len(rest)
        pr=(cr['noise']+cr['bad'])/nr if nr else 0
        ok=nr>=4 and len({r['game'] for r in rest})>=3 and pr>=.7
        rules=[(r,l) for r,l in V[ph] if H.lab(r,rd)=='useful' and INFO[ph][r['id']]['usefulness']=='game_rule']
        mres={}
        for k,f in MUT.items():
            mres[k]=sum(1 for r,l in rules if fn(dict(l,text=f(l['text']))))
        base=sum(1 for r,l in rules if fn(l))
        u=[r['id']+':'+INFO[ph][r['id']]['usefulness'] for r in hit if H.lab(r,rd)=='useful']
        print(f"{mod[5:]}.{nm:26s} {ph:7s} n{n:3d} U{c['useful']:2d} B{c['bad']:2d} N{c['noise']:2d} top {top}={tn} | w/o top n{nr} {pr*100:.0f}% {'pass' if ok else 'FAIL'} | game_rule base {base}/{len(rules)} mut {mres} | U: {u}")
T={'cand_snapshot':{'current':['SNAP_CORE','now_coord_nocause','locative_nocause','step_strict','lead_coord'],'past':['SNAP_CORE','SNAP_PAST','locative_nocause','coord_no_event','value_dump']},
 'cand_fragment':{'current':['core','bare_ref','letter_label','heading_colon','label_snapshot','short_verbless','core_or_snapshot','core_or_snap_or_verbless'],'past':['core','bare_ref','letter_label','label_snapshot','sym_dense','other_level_norule','core_or_sym','core_or_snapshot','core_or_snap_or_sym']},
 'cand_plan_progress':{'current':['moment','loc_strict','self_check','bare_obj','next_act','step_main','heading','union_current','union_current_no_self'],'past':['moment','loc_wide','step_any','self_check','bare_obj','layout_check','next_act','other_level','union_past','union_past_no_self']},
 'cand_past_stale_redundant':{'current':['symbol_literal','episode_trace','no_claim','trailing_colon_only','current_union'],'past':['past_union','static_coords_keeprule','static_coords','layout_noun','layout_wide','symbol_literal','episode_trace','no_claim','RED_contain_50']}}
if __name__=='__main__':
    for mod,ph in T.items():
        for p,names in ph.items(): audit(mod,names,p)
    # who catches what, by line
    for ph in H.PH:
        rd=H.PH[ph]; rows=C.defaultdict(list)
        for mod,pp in T.items():
            m=load(mod)
            for nm in pp[ph]:
                fn=m.CANDIDATES[nm][1]
                for r,l in V[ph]:
                    if fn(l): rows[r['id']].append(mod[5:8]+'.'+nm)
        print('\n=== coverage',ph,'lines hit by anything',len(rows))
        jt=0
        for r,l in V[ph]:
            lb=H.lab(r,rd)
            if lb!='useful': jt+=1
        print('junk total',jt,'junk touched',sum(1 for r,l in V[ph] if H.lab(r,rd)!='useful' and r['id'] in rows),'useful touched',sum(1 for r,l in V[ph] if H.lab(r,rd)=='useful' and r['id'] in rows))
        print('useful touched by:')
        for r,l in V[ph]:
            if H.lab(r,rd)=='useful' and r['id'] in rows:
                print(' ',r['id'],len(rows[r['id']]),INFO[ph][r['id']]['usefulness'],'|',l['text'][:110],'|',','.join(rows[r['id']][:6]))
        print('junk NOT touched by anything:')
        for r,l in V[ph]:
            if H.lab(r,rd)!='useful' and r['id'] not in rows:
                print(' ',H.lab(r,rd),r['id'],'|',l['text'][:150])
