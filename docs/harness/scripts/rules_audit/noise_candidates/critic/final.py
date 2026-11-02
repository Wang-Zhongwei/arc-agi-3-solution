import sys,importlib.util,collections as C,json
sys.path.insert(0,'/Users/wang/.claude/jobs/910e124b/tmp/scripts'); sys.path.insert(0,'critic')
import noise_harness as H, adv as A
m=A.load('cand_shortlist')
EXTRA=[("Clicking (3,4) turns the tile blue ✓ confirmed.","input+coord"),("Confirmed: pressing ACTION5 at (2,2) flips the row.","input"),
("Confirmed: a tile at (2,2) turns blue on the second visit.","no input, -s verb"),
("Confirmed: the gate at (6,6) opens only on the second touch.","no input, -s verb"),
("Confirmed: stepping on (4,4) kills the player.","no input, -s verb"),
("Confirmed: the budget bar is at row 63 and it is the move counter.","HUD meaning"),
("Goal confirmed: the crate at (5,5) goes to the target at (9,9).","goal"),
("Hazard confirmed at (8,8): anything entering it dies.","hazard"),
("Confirmed: walls at col 5 block the beam.","block"),
("Confirmed: the player is at (1,1) at the start of every level.","reset invariant")]
SENT=A.ADV+EXTRA
V={p:H.views(p) for p in H.PH}
D=json.load(open('dev_lines_v02.json')); INFO={p:{x['id']:x for x in D[p]} for p in D}
MUT={'+coord_aside':lambda s:s.rstrip()+' (seen at (12,30))','+step_aside':lambda s:s.rstrip()+' (at step 40)','+bracket_tag':lambda s:s.replace('✓ (confirmed)','[confirmed]').replace('(confirmed)','[confirmed]').replace('✓ confirmed','[confirmed]'),
 'now_prefix':lambda s:'Now I see: '+s}
for nm,(p,fn) in m.CANDIDATES.items():
    print('\n',nm,'adv hits',sum(1 for s,_ in SENT if fn(A.line(s))),'/',len(SENT))
    for s,k in SENT:
        if fn(A.line(s)): print('    ',k,'|',s)
    for ph,rd in H.PH.items():
        if p not in (ph,'both'): continue
        hit=[r for r,l in V[ph] if fn(l)]
        by=C.Counter(r['game'] for r in hit); top,tn=by.most_common(1)[0]
        rest=[r for r in hit if r['game']!=top]; cr=C.Counter(H.lab(r,rd) for r in rest); nr=len(rest)
        pr=(cr['noise']+cr['bad'])/nr if nr else 0
        ok=nr>=4 and len({r['game'] for r in rest})>=3 and pr>=.7
        rules=[(r,l) for r,l in V[ph] if H.lab(r,rd)=='useful' and INFO[ph][r['id']]['usefulness']=='game_rule']
        mu={k:sum(1 for r,l in rules if fn(dict(l,text=f(l['text'])))) for k,f in MUT.items()}
        print('   ',ph,'top',top,tn,'w/o top n',nr,'%.0f%%'%(100*pr),'pass' if ok else 'FAIL','| game_rule',sum(1 for r,l in rules if fn(l)),'/',len(rules),mu)
# incremental contribution and residual
for ph,rd,order in (('current','same',['fragment','snapshot']),('past','later',['fragment','snapshot','dead_coords','dead_layout','dead_literal'])):
    seen=set()
    for nm in order:
        fn=m.CANDIDATES[nm][1]; h={r['id'] for r,l in V[ph] if fn(l)}
        print(ph,nm,'adds',len(h-seen),'of',len(h)); seen|=h
    uni=m.CANDIDATES['current_rule' if ph=='current' else 'past_rule'][1]
    rest=[(r,l) for r,l in V[ph] if not uni(l)]
    c=C.Counter(H.lab(r,rd) for r,l in rest); print(ph,'remaining',len(rest),dict(c),'junk %.1f%%'%(100*(c['noise']+c['bad'])/len(rest)))
    for r,l in rest:
        if H.lab(r,rd)!='useful': print('    ',H.lab(r,rd),r['id'],l['text'][:120])
# overlap of survivors with designers' top unions
for ph,rd,refs in (('current','same',[('cand_snapshot','SNAP_CORE'),('cand_fragment','core'),('cand_plan_progress','union_current_no_self'),('cand_past_stale_redundant','current_union')]),('past','later',[('cand_snapshot','SNAP_PAST'),('cand_fragment','core_or_sym'),('cand_plan_progress','union_past_no_self'),('cand_past_stale_redundant','past_union')])):
    uni=m.CANDIDATES['current_rule' if ph=='current' else 'past_rule'][1]; me={r['id'] for r,l in V[ph] if uni(l)}
    for mod,nm in refs:
        f=A.load(mod).CANDIDATES[nm][1]; o={r['id'] for r,l in V[ph] if f(l)}
        print(ph,nm,'n',len(o),'shared',len(o&me),'only theirs',sorted(o-me),'only mine',len(me-o))
