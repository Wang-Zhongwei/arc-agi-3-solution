# Stall allocation

Source: `runs/kaggle/_external/matijaludvig-spn-rrw/bundle_full/spn-rrw-bundle/`
`src/ARC3-Inference/inference/framework/solver.py`, specifically
`_HarnessGameSession.current_cap_seconds` and `HarnessSolver.clock_extension_allowed`.

Stall was adapted as a continuous clock policy for keeping games with recent
level progress running. It avoids the pause/resume ranking used by rolling
extension and UCB.

The [allocation reference](../../ARC3-Inference/docs/allocation.md) owns parameter definitions,
defaults, constraints, and timing semantics. The table below records provenance
from the teammate implementation; it is not a second parameter specification.

| Reference knob | Harness setting |
| --- | --- |
| `clock_base_s` | `allocation.stall_base_seconds` |
| `clock_guarantee_s` | `environment.max_runtime_minutes` (converted to seconds) |
| `clock_stall_s` | `allocation.stall_timeout_seconds` |
| `clock_quantum_s` | `allocation.quantum_seconds` |
| `clock_max_s` | `allocation.stall_max_seconds` |
| `clock_extend_min_levels` | `allocation.stall_extend_min_levels` |

The integration uses the existing global serving budget and soft deadline. Its
reservation check is deliberately conservative: all queued games reserve their
full guarantee, even when the remaining shared budget cannot cover every game.
In that case no extra extensions are admitted. The reference instead reserves a
truncated entitlement near the deadline. Reservation checks and cap increments
are atomic here, so concurrent workers cannot spend the same spare time.

For recorded fields, see [diagnostics](../../ARC3-Inference/docs/allocation.md#diagnostics-and-implementation).

Short local GPU test (one H100, six official games, two worker slots):

```sh
make sbatch CONFIG_PATH=configs/inference.json RUN_NAME=stall-policy-smoke \
  ALLOCATION_POLICY=stall GAME=ls20,ft09,vc33,cn04,lp85,sp80 \
  N_PASSES=1 CONCURRENT_JOBS=2 DEPLOYMENT_SLURM_GPU_COUNT=1 \
  MAX_RUNTIME_MINUTES=2 STALL_BASE_SECONDS=45 STALL_TIMEOUT_SECONDS=60 \
  EXTENSION_QUANTUM_SECONDS=30 STALL_MAX_SECONDS=240 \
  SERVING_BUDGET_MINUTES=6 DEPLOYMENT_SLURM_TIME=00:30:00
```

This is a plumbing smoke test, not evidence of a score improvement. Model startup
is outside the six-minute serving clock. Short timers are intended to exercise
retirement promptly; deterministic tests cover progress and extension decisions.
