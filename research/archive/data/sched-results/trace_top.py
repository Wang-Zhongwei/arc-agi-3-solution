"""How dl-hRg:pn0.4 and dl-hQ:pn0 hand out slots vs D' (never 0.4, 40 reps)."""
import collections, csv, os, random, statistics as st, sys
sys.path.insert(0, os.getcwd())
from fit_levels import kaplan_meier
from replay_scheduler import load_trajectories, make_dprime, make_game, simulate
from policies import candidate_policies
rows = list(csv.DictReader(open('levels.csv')))
curve = kaplan_meier([(int(r["tokens"]), int(r["cleared"])) for r in rows])
eff = [float(r["efficiency"]) for r in rows if r["cleared"] == "1"]
tr = load_trajectories('levels.csv')
reg = candidate_policies(rows)
pols = {"dprime": (make_dprime(), {"fresh_first": False}), "dl-hRg:pn0.4": reg["dl-hRg:pn0.4"],
        "dl-hQ:pn0": reg["dl-hQ:pn0"], "aff-v1:nofresh": reg["aff-v1:nofresh"]}
REPS = 40
agg = {n: collections.Counter() for n in pols}
gain = {n: collections.defaultdict(float) for n in pols}
late = {n: collections.Counter() for n in pols}
started, never_started_unclear, scores, levels_by_depth = ({n: [] for n in pols} for _ in range(4))
for rep in range(REPS):
    rng = random.Random(rep); sample = [rng.choice(tr) for _ in range(110)]
    for n, (pol, ex) in pols.items():
        games = [make_game(t, curve, eff, random.Random(rep * 1000 + i), 0.4) for i, t in enumerate(sample)]
        trace = []
        scores[n].append(simulate(games, pol, slots=10, rate=61, minutes=532, quantum=25000, trace=trace, **ex))
        for q in trace:
            agg[n][q["kind"]] += 1; gain[n][q["kind"]] += q["gain"]
            if q["t"] >= 0.8: late[n][q["kind"]] += 1
            agg[n]["L%d" % min(q["level"], 5)] += 1
        started[n].append(sum(g.started for g in games) / len(games))
        # tokens burnt on levels that can never clear
        never_started_unclear[n].append(sum(g.spent_on_level for g in games
                                            if g.level < len(g.levels) and g.levels[g.level].cost == float("inf")) / 1e6)
for n in pols:
    tot = sum(agg[n][k] for k in ("start", "switch", "repick", "hold"))
    lt = sum(late[n].values())
    print(f"{n:16} score {st.mean(scores[n]):.2f} quanta {tot/REPS:.0f} "
          + " ".join(f"{k} {agg[n][k]/tot:.1%} ({gain[n][k]/max(1,agg[n][k]):.3f}/q)" for k in ("start", "switch", "repick"))
          + f" | late(>=80%) switch {late[n]['switch']/max(1,lt):.1%} | started {st.mean(started[n]):.1%}"
          + f" | quanta on L1/L2/L3/L4/L5+ " + "/".join(f"{agg[n]['L%d'%d]/tot:.0%}" for d in range(1, 6))
          + f" | M tokens stuck on unclearable at end {st.mean(never_started_unclear[n]):.2f}")
