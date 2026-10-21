"""Summaries for results.md: macro-over-games metrics per candidate and game subset."""
import json, sys, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = HERE / "results"
F1 = "ls20,cd82,bp35,sp80,vc33".split(","); F2 = "ar25,tu93,lf52,cn04,r11l".split(",")
HELD = F1 + F2
DEV = "dc22,g50t,sb26,sc25,sk48".split(",")


def summ(res, games, lvl="inst"):
    gs = [g for g in games if g in res]
    m = lambda k: st.mean(res[g][lvl][k] for g in gs)
    pooled = lambda t: (sum(res[g][lvl]["hard"][t][0] for g in gs) / max(1, sum(res[g][lvl]["hard"][t][1] for g in gs)))
    ms = [res[g]["ms"]["median"] for g in gs]; p95 = [res[g]["ms"]["p95"] for g in gs]
    v = [res[g]["names"]["v"] for g in gs if res[g]["names"]["n"]]
    pooledF = (lambda tp, gt, pr: 2 * tp / max(gt + pr, 1))(sum(res[g][lvl]["tp50"] for g in gs), sum(res[g][lvl]["gt"] for g in gs), sum(res[g][lvl]["pred"] for g in gs))
    return dict(n=len(gs), F50=m("F50"), P50=m("P50"), R50=m("R50"), pooledF=pooledF, R90=m("R90"), exact=m("exact"), frag=m("frag"), shatter=m("shatter"),
                merge=m("merge"), overseg=m("overseg"), ari=m("ari"), vi=m("vi"), leak=m("leak"),
                touchdiff=pooled("touchdiff"), multicolour=pooled("multicolour"), scaled=pooled("scaled"), tilegame=pooled("tilegame"),
                V=st.mean(v) if v else float("nan"), ms=st.median(ms), p95=max(p95),
                cell=st.mean(res[g].get("cell_acc", float("nan")) for g in gs))


def load(name):
    p = R / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def row(label, res, games, lvl="inst"):
    s = summ(res, games, lvl)
    return (f"| {label} | {s['n']} | {s['F50']:.3f} | {s['P50']:.2f}/{s['R50']:.2f} | {s['R90']:.2f} | {s['exact']:.2f} | {s['frag']:.2f} | {s['shatter']:.2f} | {s['merge']:.2f} | "
            f"{s['overseg']:.2f} | {s['ari']:.2f} | {s['touchdiff']:.2f} | {s['multicolour']:.2f} | {s['scaled']:.2f} | {s['tilegame']:.2f} | {s['V']:.2f} | {s['ms']:.0f} |")


HDR = ("| candidate | games | F1@0.5 (macro) | P/R@0.5 | R@0.9 | exact | frag | shatter | merge | segs/GT | ARI | hard: touch-diff R | multicolour R | scaled R | tile-game R | names V | ms median |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        p = spec.split(":"); name = p[0]; games = p[1] if len(p) > 1 else "all"; lvl = p[2] if len(p) > 2 else "inst"
        res = load(name)
        gs = {"all": sorted(res), "held": HELD, "f1": F1, "f2": F2, "dev": DEV, "nondev": [g for g in sorted(res) if g not in DEV]}.get(games, games.split("+"))
        print(row(f"{name} [{games},{lvl}]", res, gs, lvl))
