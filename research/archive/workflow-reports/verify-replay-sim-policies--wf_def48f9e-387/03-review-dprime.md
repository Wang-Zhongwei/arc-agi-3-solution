# verify-replay-sim-policies: review:dprime

Workflow run `wf_def48f9e-387`, phase Review. Sonnet reviewers check the replay simulator models easy-first, D-prime and Franzen faithfully

---

**D' lens: no bug found that changes the Franzen vs D' ranking.** The D' inputs and fade match the harness. The mismatches below are small, and the flip risks I found shrink the D' margin without reversing it.

Sim D' minus Franzen is about +4.9 (110 games, 10 slots, 532 min, 6–12 reps). My checks:
- **Quantum 8k–100k:** +5.5 to +6.5.
- **Switch cost 600–2000 tokens:** +5.1 to +5.6.
- **Action-proxy scale ×0.25–×2:** +3.7 to +6.2.
- **Level costs ×1–×4:** +3 to +5.7. At ×2, Franzen is about 28, which matches the LB scale, and the margin is still about +5.

**Checked and matching:**
- `make_dprime` (replay_scheduler.py:216-226) builds the snapshot with the same field order as the harness (tool_agent.py:6475).
- level is `g.level+1`, tokens are `spent_on_level`, and `level_cost` is the mean cleared-level cost (the harness tracks the same at tool_agent.py:3916-3927).
- The fade is `remaining/0.4` clipped to 1, as in tool_agent.py:2035-2039 and `d_prime_priority`.
- D' has no streak hold, and the arm has no easy boost (arms.json).
- Fresh snapshots are `(1, 0, 0, …)` with raw n, as in solver.py:629.
- Ties break by arrival order in both.

**Concrete mismatches (all minor):**
1. **Initial admission.** The harness admits the first 10 `acquire` callers in arrival order with no ranking (`_enqueue_and_wait` pumps while `free>0`, tool_agent.py:2094-2105). I confirmed this with the real `_PriorityGate`. The sim ranks all 110 games for the 10 starting slots (replay_scheduler.py:260-261, with `fresh_first=False` at :334), so it starts the shortest games. This would favor D'. Measured effect with arrival-order starts: +0.01.
2. **Stale fade clock.** `ctx["remaining"]` is set at :267 before the quantum, but `pick()` at :294 runs after it, so the clock is stale by up to 1.3% of the run. Updating it live moves D' by +0.3 and Franzen by +0.2, so the margin changes by about +0.1. This slightly understates D'.
3. **Hidden-information leak.** `_level_actions` returns 50.0 for an unclearable level even at zero spend (:186). D' then ranks unclearable fresh games lower, which the harness cannot do. It only matters with `--never>0`. At the default 0 there are no infinite-cost levels (0 of 2200 games). At `--never 0.5` the leak inflates the margin by about 0.4 (+2.28 vs +1.87 with it removed).
4. **`total_levels` handling.**
   - The sim passes raw n everywhere. The harness passes raw n for fresh games (solver.py:629) but clamps to [6,10] at trim time (tool_agent.py:6430, 6461-6462).
   - `d_prime_value` clamps only the normalisation (priority_scheduler.py:215-216) and uses raw n for remaining levels and the tail (:229-230).
   - The replay data only has n in 6–10, so this is invisible in the sim. A hidden game with n<6 would get a nonzero tail on its last level at trim but zero at fresh admission.
   - If session metadata is unavailable, the harness falls back to n=10 (tool_agent.py:6433-6434). D' would then lose its fewer-levels-first preference after the first trim.

**Modelling simplifications that could shrink or flip the ranking:**
- **`--never` default 0 (replay_scheduler.py:309-310).** Every stuck level eventually clears. D' keeps its full tail B for stuck games until the fade, because B is not multiplied by chance (priority_scheduler.py:231). So a patient policy gains. The margin by `never` value:

  | `never` | 0 | 0.3 | 0.5 | 0.7 | 0.9 |
  |---|---|---|---|---|---|
  | D' minus Franzen | +4.9 | +3.4 | +2.3 | +1.4 | +1.0 |

  Offline calibration cannot identify it. At 25 games and 121 min, Franzen scores 46.4, 45.6 and 44.6 for `never` 0, 0.5 and 0.9, against a real control of 46.25. The `--validate` flag the help text mentions does not exist in this copy. At level costs ×2 the margin stays near +5 for every `never`, so I found no flip.
- **Fade fraction is the dominant lever.** D' minus Franzen is +4.9 at fade 0.4, +1.8 at 0.2, and −7.3 with no fade. Franzen at fade 0.4 gains only +0.2. The sim assumes the full 532 min is generation time. The real clock is `min(start+532 min, soft deadline)` (solver.py:619-626), and it includes server startup.
- **Synthetic deep levels** (replay_scheduler.py:107-108). Levels past the observed ones are resampled from the game's own cleared costs. Under D' this makes up about 9.8% of cleared levels, against 6.3% for Franzen. That mildly favors D', which wins 47 games per run to Franzen's 37. Cost by level index in the data is flat, so I cannot say which way it biases the real run.
- **D' plus easy-first.** The boost multiplies B·phi too (priority_scheduler.py:242-244), so a stuck boosted game never decays. At `--never 1.0` it collapses to −17.5 against Franzen. This is a property of the harness code, not a sim bug, and the notebook D' arm does not set EASY_FIRST.

I found no infinite loops or starvation bugs in small-scale runs (5 games on 10 slots, 1 slot with a 1k quantum, 5 min, `--never 1.0`). D' leaves about 3.5 games unstarted and about 19 stuck at 0 levels. That follows from the policy.
