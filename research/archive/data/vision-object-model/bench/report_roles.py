import json, sys, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
print("| segments | games w/ eligible levels | eligible levels | player correct at level end | committed <=5 actions | median actions-to-commit | click: segment = GT target (IoU>=0.5) | HUD-change px P/R | seg ms/frame |")
print("|---|---|---|---|---|---|---|---|---|")
for name in sys.argv[1:]:
    p = R / f"roles_{name}.json"
    if not p.exists(): continue
    r = json.load(open(p))
    if len(sys.argv) > 1 and "--games" in name: pass
    L = [l for g in r.values() for l in g["levels"]]
    games = sum(1 for g in r.values() if g["levels"])
    com = [l["commit"] for l in L if l["commit"]]
    on = sum(g["click"].get("on_sprite", 0) for g in r.values()); i5 = sum(g["click"].get("iou50", 0) for g in r.values())
    H = [h for g in r.values() for h in g["hud"]]
    inter = sum(h["inter"] for h in H); pred = sum(h["pred"] for h in H); gtc = sum(h["gt_changing"] for h in H)
    ms = st.median([g["ms"] for g in r.values()])
    print(f"| {name} | {games} | {len(L)} | {sum(l['final_ok'] for l in L)}/{len(L)} ({sum(l['final_ok'] for l in L)/max(len(L),1):.0%}) | "
          f"{sum(1 for c in com if c <= 5)}/{len(L)} ({sum(1 for c in com if c <= 5)/max(len(L),1):.0%}) | {st.median(com) if com else float('nan'):.0f} | "
          f"{i5}/{on} ({i5/max(on,1):.0%}) | {inter/max(pred,1):.2f}/{inter/max(gtc,1):.2f} | {ms:.1f} |")
