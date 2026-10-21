# sim-v2-calibrate-validate-explore: build

Workflow run `wf_15abaf47-5b6`, phase Build. Calibrate a continuous-time scheduler simulator from real runs, validate it against held-out real policies (UCB, D-prime), then explore annealed round-robin to D-prime/UCB schedulers

---

I've built simulator v2 and committed it on branch `arm/sim-v2` in six small commits (f6aa5b1 … 4fb8423). All 27 unit tests pass, both with plain `python3` and under pytest. Nothing was pushed, and the `daniel-franzen-sol` working tree wasn't touched.

The main finding is that the simulator reproduces every real run once it is given that run's own level costs. On that basis, almost none of D′'s real 7.5-point loss is the scheduler's doing.

**Design** (`analysis/sim2.py`, standard library only, about 15 ms per 25-game replicate):
- **Clock:** every game's clock starts at the benchmark start and stops at `minutes × 60`. Waiting at the gate counts against it, and requests still running at the deadline are cut and their tokens lost. `--server-ready` reproduces the startup race.
- **Gate:** an exact port of `_PriorityGate`:
  - A game hands over only at a context trim, and may win its slot straight back.
  - With the competition's refreshed queue, waiting games are re-priced at every pump.
  - Franzen, easy-first and UCB start every fresh game in the 2M − 100·index band; D′ prices fresh games with its own formula, or uses the 2M band as `dprime-ff`.
  - Initial arrivals get a few places of jitter. This matches how the real first waves are reordered, and it decides ties between equal D′ fresh prices.
- **Requests:**
  - Each request prefills its new tokens (or the whole retained prompt after a trim), then decodes. Decoding streams share the server at the measured per-stream rate for the current number of streams.
  - Rates are token-weighted from the server logs, with separate, slower curves for the 12- and 14-slot configurations.
  - Completion size, prompt growth and turn boundaries are drawn together from real consecutive request pairs.
  - Client pauses come from a fitted model: prompt size, first request of a turn, actions so far, a per-game effect and real residuals.
- **Context:** a game trims when its next prompt passes the window minus the 12k output reserve, drops to the drain level, and pays a re-prefill.
- **Level costs:** each game-level is drawn independently per replicate, from a Kaplan-Meier estimate over that game-level's attempts across runs (unfinished attempts count as censored).
  - Where a game-level has fewer than 3 attempts, the pooled attempts of all games at that level join in, rescaled to the game's difficulty; unreached levels use the pool alone.
  - Each draw is one real (tokens, actions) clear, so efficiency comes with it.
  - A game's own earlier levels are never used. In the data, consecutive levels' costs correlate at only 0.05.
  - `--ledger on/off` picks the ledger or no-ledger runs.
  - `--replay-run` keeps one run's actual costs instead of sampling.
- **Scorer:** the same formula `inference.tools.eval` reports. A test confirms it reproduces every game's `final_score` in all 18 runs.
- **Policies:** Franzen, easy-first and D′ call the harness's own `priority_scheduler.py` with the competition settings (fade 0.2, D′ 0.4). UCB is ported from ff0630c. FIFO is included as a reference.
- **Paired comparisons:** level draws, arrival order and each game's request stream are seeded per game, so policies are compared on the same draws.
- **Calibration addition:** `calibrate.py` now also writes `calibration/games_meta.json` (baselines for every level, and each run's dispatch order).

**Validation** (`validate_sim2.py`, each of the 10 full runs simulated under its own settings):
- **Replaying each run's own costs:** mean error +0.34 points, RMSE 0.99. Switch-outs, games never re-admitted, start waves (Franzen's 10 / 15–20 / 30–34 min; D′'s last start at 72–78 min), total tokens (within 2%) and tokens per active second all match.
- **Sampling costs from the other runs:** bias +0.40, RMSE 3.3. The z-scores have mean −0.10 and sd 1.02, so the simulated run-to-run spread is about right.
- **The one outlier is the D′ run (v4-stack-v1):** sampled prediction 51.1 ± 3.1 against an actual 43.32 (z = −2.5), while replaying its own costs gives 43.28.
- **On that run's own costs, Franzen would have scored 44.1.** So the scheduler explains under 1 of the 7.5 points. The rest is which levels that run failed to clear: tr87 L2, tn36 L3, m0r0 L3 and cn04 L2, early levels of short, high-value games.
- **Luck or harness change:** one run can't tell whether that was bad luck or some other change in the v4 stack.
- **4-slot panels:** control 5.13 against 4.84 simulated, easy-first 7.51 against 6.12. The D′ panel's 10.77 against 5.71 rests on ft09 winning a tie at admission, which the simulator reproduces in 38% of replicates.

**First policy results** (difference from Franzen, standard error in brackets):

| Setting | Franzen | D′ | UCB | easy-first |
|---|---|---|---|---|
| 25 games / 10 slots / 121 min, ledger costs | 49.2 | +0.13 (0.11) | +0.10 | −0.14 |
| 25 games / 10 slots / 121 min, no-ledger costs | 44.6 | +3.26 (0.17) | −0.94 | +0.73 |
| 110 games / 532 min, ledger costs | 42.4 | +1.57 (0.13) | +0.10 | +0.02 |
| 110 games / 532 min, no-ledger costs | 39.2 | +3.21 (0.16) | −1.80 | +0.83 |

The D′ gap depends on how stuck levels are modelled: +0.1 to +1.1 on ledger costs and +2.6 to +3.9 on no-ledger costs. D′ gains on 6- and 7-level games and loses on 9- and 10-level games.

**How to run** (from `analysis/`, no venv needed):
```
nice -n 10 python3 sim2.py --policy franzen dprime dprime-ff ucb easy-first --ledger on --reps 200
nice -n 10 python3 sim2.py --policy franzen dprime --n-games 110 --minutes 532 --reps 40 --quiet
nice -n 10 python3 sim2.py --policy franzen --games-from-run v4-panel-control --slots 4 --minutes 20
nice -n 10 python3 sim2.py --policy dprime --replay-run v4-stack-v1 --exclude-run v4-stack-v1 --ledger on
nice -n 10 python3 validate_sim2.py --reps 40
python3 test_sim2.py
```
The output gives score mean, sd and se; a per-game table; start-time stats; tokens (and tokens lost at the deadline); switch-outs; and paired differences against the first policy listed. Other options include `--games`, `--ctx`/`--drain`, `--speed-scale` (0.854 reproduces the 16-experts run), `--fade`, `--ucb-c`, `--tail`, `--prior-weight`, `--prior-max-n` and `--json`.

**Limits:**
- Level costs are assumed not to depend on the scheduler (trims and switch-outs don't change what a level costs); nothing tests this.
- The 110-game set is the 25 public games plus copies.
- The 12- and 14-slot speed curves come from one run each.
- Pause times are fitted on one ledger run (r2) plus the short panels.

The design, validation tables and limits are written up in `analysis/README.md`. Total simulation CPU was a few minutes, run one job at a time under `nice`.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/wt/sim-v2/analysis/`:
- sim2.py
- test_sim2.py
- validate_sim2.py
- README.md
- calibrate.py
- calibration/games_meta.json
