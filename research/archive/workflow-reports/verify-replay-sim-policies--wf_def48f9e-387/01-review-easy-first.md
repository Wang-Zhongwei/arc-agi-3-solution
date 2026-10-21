# verify-replay-sim-policies: review:easy-first

Workflow run `wf_def48f9e-387`, phase Review. Sonnet reviewers check the replay simulator models easy-first, D-prime and Franzen faithfully

---

Easy-first is modelled correctly in the mechanics. I found no mismatches in what the boost counts, what it multiplies, the streak formula or fresh-first admission. There are no infinite loops or scoring errors, and the full 10-slot token budget is always used. The issues are one fidelity gap and a few simplifications that matter for the ranking. Paired-rep numbers below are mean differences vs franzen over 40 reps.

**Checked and matching the harness**
- **Boost inputs:** `_cleared()` (replay_scheduler.py:189) gives levels cleared and the sum of their generated-token costs. The harness uses `_easy_cleared_levels` and `_easy_cleared_tokens` from the per-level generated-token delta (tool_agent.py:3916-3927). The boost then multiplies the final value in `priority_value` (priority_scheduler.py:179) and `d_prime_priority` (:244), as in the harness (tool_agent.py:6470-6475).
- **Streak/hold inputs:** `easy_hold` passes the same three arguments as `_maybe_handover` (tool_agent.py:6488). `level_tokens` is tokens on the current level since level start, which is `g.spent_on_level`. Holding skips the handover and the slot stays occupied (replay_scheduler.py:289 vs tool_agent.py:6498).
- **Fresh-first:** the harness gives unstarted games a fixed top priority with no snapshot (solver.py:640), and the sim's `fresh[0]` matches. Ties break by arrival order in both.
- **Fade:** `remaining/fraction` matches the harness's `remaining/(fraction*(deadline-start))` (tool_agent.py:2084-2087).

**1. The `--never 0.0` default hides a real D'+easy-first failure** (replay_scheduler.py:309, 342). The help text points to a `--validate` flag that no longer exists, so the default is uncalibrated and means every stuck level eventually clears. In the harness, `value *= boost` (priority_scheduler.py:244) also scales D''s `B*phi` tail term, which never decays when a level stalls. A game that cleared two levels at 8k each and has sat 300k tokens on level 3 scores 31,558 against 17,000 for a fresh game, and 16,296 without the boost. It keeps winning the slot back until phi falls to about 0.5, and several such games clog slots.

| `--never` | dprime+easy-first vs franzen |
|---|---|
| 0.0 | +3.26 |
| 0.4 | -7.26 (3/40 reps better) |
| 0.8 | -11.7 (0/8) |

- **Concrete effect:** at 0.8, one game consumed a whole slot budget (1.95M tokens).
- **Direction:** the default overstates D'+easy-first. This is faithful reproduction of harness behaviour, not a sim bug, and D'+easy-first is not an arm in kaggle/arms.json. Franzen plus boost is not affected because its `chance` term goes to 0 on a stall (-0.36 at never 0.4).

**2. The easy-first components have opposite signs, and the net is within noise.** At never 0, boost-only is -0.67, hold-only is +0.24 and combined is -0.54, with paired sd about 1.0 and 40 reps (se about 0.15). The 3-rep run gave easy-first -1.6. Changing the quantum to 10k, 15k, 25k, 40k or 60k flips the sign of easy-first vs franzen across 5-rep runs (+0.2 to -0.6). Conclusion: easy-first vs franzen is not resolved below about 100 reps.

**3. Simplifications that could shift the easy-first result**
- **Fixed quantum:** `quantum` is a uniform 25k generated tokens (replay_scheduler.py:308), while the harness trims on context growth (128Ki window, drain to about 57Ki). The sim also checks the hold only at quantum boundaries, so the effective stall cutoff is rounded up by up to one quantum. The harness's first trim also comes much later than later ones, which lengthens the fresh-first phase, and a held game blocks admission during it.
- **Synthetic levels (replay_scheduler.py:106-108):** 412 of 1281 levels (32%) are synthetic, drawn from the game's own earlier costs, including the sampled tail cost. A game's history is therefore more predictive of its future than the data supports, which is the signal boost and hold use. Replacing them with pooled draws moved easy-first vs franzen from -0.46 to -0.04 (30 reps, se about 0.2), and shifted absolute scores by about +1. The sign of the bias is not clear-cut.
- **No re-prefill cost:** `switch_cost` applies only to `stall_swap`. A yielded game must re-prefill its whole context when it returns, which holding avoids, so the sim biases against hold (and the hold-only gain of about +0.24 is likely understated). The harness comment calls yielding at a trim free, which is only partly true.

**4. Minor: stale fade clock** (replay_scheduler.py:267 vs 254, 294). `ctx["remaining"]` is set at the start of a quantum but used at the end-of-quantum pick. The fade is therefore up to 25k/1.95M = 1.3% of the run late, about 6% of the 0.2 window and 3% of the 0.4 window. It applies equally to all arms, so the effect on the comparison is negligible. Setting `remaining` from `clocks[s]` just before `pick()` would fix it.

**5. Idle-slot latch** (replay_scheduler.py:295-296). If `pick()` returns None, the slot is set to `inf` and is never refilled when other games re-enter the queue. This is harmless at 110 games, since every policy used the full 19.47M tokens, but it would distort runs with `--games` of 10 or fewer.

Files are in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol`: `analysis/replay_scheduler.py` and `ARC3-Inference/inference/agent/priority_scheduler.py`. My scratch scripts are in the scratchpad directory.
