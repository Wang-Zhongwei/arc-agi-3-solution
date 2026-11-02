import sys,collections as C
sys.path.insert(0,'/Users/wang/.claude/jobs/910e124b/tmp/scripts'); sys.path.insert(0,'critic')
import noise_harness as H, adv as A
m=A.load('cand_shortlist'); V=H.views('past')
for nm,f in {'past w/o layout,literal':lambda l:m.current_rule(l) or m.dead_coords(l),'past w/o dead_*':m.current_rule}.items():
    h=[r for r,l in V if f(l)]; c=C.Counter(H.lab(r,'later') for r in h); print(nm,len(h),dict(c),len({r['game'] for r in h}))
# counts: how many distinct games, per-concept word-list size; designers' parts that add nothing
fr=A.load('cand_fragment'); sn=A.load('cand_snapshot'); pl=A.load('cand_plan_progress')
for ph,rd in H.PH.items():
    v=H.views(ph)
    br={r['id'] for r,l in v if fr.bare_ref(l)}; rest={r['id'] for r,l in v if fr.coord_open(l) or fr.letter_label(l) or fr.heading(l)}
    print(ph,'bare_ref',sorted(br),'adds beyond other core parts',sorted(br-rest))
    bo={r['id'] for r,l in v if pl.bare_obj(l)}; print(ph,'bare_obj',sorted(bo))
    vd={r['id'] for r,l in v if sn.dump(l)}; print(ph,'value_dump',sorted(vd))
    ns={r['id'] for r,l in v if fr.sym_dense(l)}; print(ph,'sym_dense',sorted(ns))
