# UCB retry recovery without abandoning games

2026-09-17. Follow-up implementation on `fix/ucb-retry-and-disk-usage`.
The initial failure-count cutoff was removed: every game remains eligible after
request failures, subject to the existing scheduler budgets.

The public SPN log identifies vLLM `0.28.1rc1.dev417+g5690b02c0`.
Inspection of that exact upstream commit confirms:

- `MultiModalProcessorSenderCache.get_and_update_item` returns `None` for image
  data on an API-process cache hit.
- `MultiModalReceiverCache.get_and_update_item` raises `MultiModalCacheMissError`
  when the engine receives that reference but has no cached tensor.
- The commit already attempts stale-entry invalidation on cache misses.
  The reported repeated 400/500 sequence shows that retrying alone did not
  recover those runs. Restoring rejected history was a separate harness bug.

The fix retains the reduced history after failed retries, so a cache-miss error
cannot restore the oversized prompt that caused the preceding rejection. The
multimodal processor cache remains enabled to avoid adding repeated image
preprocessing and transfer overhead. Behavior is documented in
[runtime reliability](../../ARC3-Inference/docs/runtime-reliability.md).

The precise estimator/server token-count discrepancy is not yet established.
The overflow warning now records the server error body alongside our estimate
to support that investigation. This does not claim that context overflows can
never occur; they must recover without poisoning image delivery.

Validation covers generated argv, SPN setup/teardown, history rollback, recovery
after more than five failures, and continued eligibility across scheduler resume.
GPU/Kaggle validation and a throughput measurement remain outstanding.
