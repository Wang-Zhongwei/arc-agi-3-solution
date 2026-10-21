# verify-replay-sim-policies: review:mechanics

Workflow run `wf_def48f9e-387`, phase Review. Sonnet reviewers check the replay simulator models easy-first, D-prime and Franzen faithfully

---

**Verdict:** the policy modelling matches the harness. There is no simulator bug, no infinite loop and no scoring error that changes the comparison. The one real issue is that the default `--never 0.0` hides a starvation failure of `dprime+easy-first`.

**Verified matches**
- **Fade window:** `_fade` (`replay_scheduler.py:194-198`) equals `remaining / (fraction * run duration)`. That is the same as `_snapshot_priority` at `tool_agent.py:2035-2039` (`window = fraction * (deadline - start)`), and both clip to [0, 1].
- **Config:** `franzen` (fade 0.2, remaining-tail, normalization, pace 1) and `dprime` (fade 0.4) match the competition and arm configs. They call the same `priority_value` and `d_prime_priority` code.
- **Franzen baseline:** `franzen-nofade` is the pre-fade baseline. The comparison base is the right one. Fade is worth +2 to +3.6 points over no fade in my 12-rep runs.
- **Streak and start order:** the hold uses the harness defaults (`tool_agent.py:6488-6503`). Fresh-first for Franzen and D' ranking of unstarted games match `solver.py:627-632`.

**Hold `continue` path (`replay_scheduler.py:289-290`)**
- It cannot loop forever and does not skip budget checks. The `continue` returns to the top-of-loop `clocks[s] >= budget` break, and every inner iteration advances the clock.
- I stress-tested always-hold with quantum 500 and 25000, with `never` at 0 and 1, and with 5 games on 10 slots. All terminated instantly.
- A game held at the budget is no longer re-queued, so the last-quantum slot "theft" artifact disappears for hold policies. This is under 1.3% of the run and has no measurable effect.

**Concrete issues**

1. **Ranking flip hidden by `--never 0.0` (the default).**
   - Evidence: `replay_scheduler.py:99` and `:309-310`. The `--validate` flag named in the help text does not exist, so the default is uncalibrated. With `never=0`, every stuck level eventually clears at a finite cost.
   - `d_prime_priority` multiplies the whole value, including the B tail, by `easy_boost` (up to 4x). Its chance term floors at 0.1. A stuck game that once cleared levels cheaply therefore always outranks a fresh game and holds a slot indefinitely.
   - This is the harness's own behaviour (`priority_scheduler.py`, `d_prime_priority`), so it is a real risk the simulator fails to show.

   | `never` | `dprime` vs franzen | `dprime+easy-first` vs franzen |
   |---|---|---|
   | 0.0 | +5.1 | +3.8 |
   | 0.3 | +3.3 | -2.6 |
   | 0.5 | +2.0 | -6.6 |
   | 0.7 | +1.3 | -9.8 |

   - At `never=0.5`, `dprime+easy-first` and `dprime+boost` burn 7.7M of 19.5M tokens on unclearable levels and leave about 21 games never started. One game alone consumed 1.93M tokens, a full slot budget. Franzen burns 3.06M and plain D' 4.0M.
   - Bias direction: `never=0` overstates D' and badly overstates `dprime+easy-first`. Rerun with `--never 0.3` to `0.7` before deciding.

2. **Stale `ctx["remaining"]` (`replay_scheduler.py:267`).**
   - It is set from the clock at the start of a quantum but used at the pick at the end (`:294`), so it lags by one quantum (1.3% of the run). Franzen's 0.2 window means a 6% error in the fade fraction, and D''s 0.4 window about 3%.
   - Setting it at pick time moves franzen by +0.17 (48.07 to 48.24) and dprime by +0.04. This is negligible, and slightly pessimistic for franzen.

3. **Initial slot assignment under D'.**
   - Harness: the first 10 threads to call `acquire` pop immediately (`tool_agent.py:2095,2104-2119`), so the first 10 slots go in arrival order. Simulator: it ranks all 110 games by D' and starts the lowest-level-count games (`:250,260-261`).
   - I tested arrival-order start. D' moved from +5.12 to +5.13, so it is negligible.

**Modelling simplifications that could flip the ranking or mislead**
- **Easy-first vs franzen is unresolved noise.** Mean difference by quantum:

  | Quantum | Easy-first vs franzen |
  |---|---|
  | 12.5k | -0.09 |
  | 25k | -0.51 (5/12 better) |
  | 50k | +0.50 (10/12 better) |

  A 3-rep gap of -1.6 is not a finding. The fixed 25k-token quantum is not the harness trim cadence, which is turn and context driven (`CONTEXT_DRAIN_TOKENS` 58*1024, `HISTORY_TURN_DRAIN` 30). Hold frequency and handover frequency in the harness vary per game.
- **No cost for switching games (`:291-295`).** Only `stall_swap` pays `switch_cost` (`:286`). I added 3000 tokens (about 50 s) per switch to a different game:

  | Policy | No switch cost | With switch cost |
  |---|---|---|
  | franzen | 48.0 | 45.2 |
  | round-robin | 44.5 | 40.1 |
  | dprime | 53.1 | 51.4 |
  | easy-first | 47.5 | 45.4 |

  - D' repicks the same game 53% of the time, Franzen 42%, and easy-first holds 33% of trims. Ignoring switch cost understates D' and easy-first relative to Franzen and round-robin.
  - D' stays ahead (+5.1 to +6.2). Easy-first moves from -0.5 to +0.2, which is within noise.
- **Absolute level is optimistic.** Simulated franzen scores 45 to 48 against the real 27.89 (`WRITEUP.md:5`). The replay set also has a median cleared-level cost of 22k, below the 30k easy reference. Most games therefore get a boost above 1 and the boost discriminates little. A harder hidden set could behave differently.
- **Lockstep slots and fixed rate.** Slots start in lockstep and picks run in slot order. ±50% quantum jitter did not change the ranking: dprime +5.9, easy-first +0.2. Time is token-equivalent at a constant 61 tok/s per slot.
- **Scoring cap (`:72-73`, pre-existing).** Caps the summed efficiency at the cleared weight, not per level, and efficiencies reach 1.15. It applies equally to every policy.
- **Stuck-level actions.** A stuck level has its actions frozen at 50 (`:186`), so stuck games are not demoted by the action term. This only matters when `never` is above 0.

Files:
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/analysis/replay_scheduler.py`
- Probe scripts: `probe.py` through `probe5.py` in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad`.
