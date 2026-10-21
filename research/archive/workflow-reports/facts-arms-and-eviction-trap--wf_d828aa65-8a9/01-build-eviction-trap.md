# facts-arms-and-eviction-trap: build:eviction-trap

Workflow run `wf_d828aa65-8a9`, phase Build. Revise attention-prompt from advice to harness-computed steering facts, build indicator-notes and budget-alert arms, investigate and file the memory pin/eviction trap issue; review, fix, integrate, PRs

---

Filed issue #58 with label MAYBE. Neither current stack-v2 nor memory-v2 as designed counts memory placement as an eviction, so I made no fix branch and nothing was committed. The trap is latent: it fires only if memory text is later stripped from old openers, or if Fable's pinned log is combined with our ledger-on-trim.

**Issue:** #58, https://github.com/adsdemaybe/daniel-franzen-sol/issues/58 (labels MAYBE, bottleneck)

**(1) How a handover is decided** (arm/stack-v2 @ d55df75, `ARC3-Inference/inference/`)
- **Slot held for the whole game:** `framework/solver.py:625` calls `gate.acquire` at game start and `:634` calls `release` at the end. The only other slot event is `_PriorityGate.handover` (`agent/tool_agent.py:2109-2119`), which releases and re-queues in one locked step.
- **Request order:** every request runs trim (6825), then `_apply_due_ledger` (6826), then `_maybe_handover` (6827).
- **`_maybe_handover`** (6454-6499) does nothing unless `_context_was_trimmed` is set. When it is set, it clears the flag, drops the game to the bottom priority band (6494-6496) and calls `gate.handover` (6499).
- **What counts as "history evicted":** only `_note_history_evicted` (6375-6380) sets that flag; it also sets `_has_evicted` and `_ledger_due`. Its callers:
  - the trimmer's message-count check `len(history) != len(messages) - 1` (6626-6627);
  - `_persistent_history_messages` at every turn end (7273), through `not trimmed` (6406-6408) and `history != original_history` (6430-6431). The second check fires on any change between what was sent and what is stored.
  - `_force_reduce_messages` (6642);
  - the rolling-summary replace path (6239-6247). That is Fable's B005 journal design; it is off in Kaggle.
- **Effect of one flagged eviction:** a handover, the bottom band, and `_ledger_due`. `_apply_due_ledger` (6382-6393) then rewrites the head message, which forces a full re-prefill.
- **Cost in our own run** (`fz-ledger-on-trim-r2`, 5 games):

  | Turns | Count | Median | Mean | Over 4 min |
  |---|---|---|---|---|
  | Start with a trim (handover) | 13 | 283 s | 1,084 s | 7 |
  | All others | 173 | 42 s | 66 s | 5 |

**(2) Is each placement counted as an eviction?** I ran a scripted model through the real `analyze()` with a fake gate, 31 requests, in a scaled-down Kaggle setup (14k window, 4k drain). I also scanned real request logs.
- **(a) Ledger-on-trim today: not affected.**
  - Simulation: 9 handovers, every one after a real drop (8 with the ledger off; the extra one is a real trim caused by the ledger's own tokens). No prefix break went unflagged and there were no back-to-back full re-prefills.
  - Real logs (400 requests): 15 full re-prefills, all at a turn's first request and all with the old head dropped. No head rewrite came from the ledger alone, there were no mid-history rewrites, and none of 378 append-only requests missed the cache.
- **(b) memory-v2 `trim` placement (3,900-char card): safe.** 11 handovers, all after real drops. One caveat: the card is added after the budget check. With drain set to 0 it causes back-to-back trims (27 vs 24 handovers). Kaggle's drain is 58k, so this does not happen there.
- **(c) memory-v2 `delta` placement, kept verbatim: safe.** 9 handovers, all after real drops.
  - Unsafe if deltas are later stripped from older openers: 21 of 31 requests re-prefill without the scheduler knowing.
  - If that strip is counted as an eviction: 30 handovers instead of 9. This is Fable's trap.
- **(d) Fable-style pinned pair:**
  - Appended and restored after the check with the ledger off: safe (8 handovers).
  - Refreshing it in mid-history: one unpriced full re-prefill.
  - **Combined with ledger-on-trim: 27 handovers instead of 9, with 16 of them on no drop at all.** This holds whether the restore goes before or after the check. The ledger merges `<memory>` into the restored pin, Fable's content match misses it, the pair is inserted again, and the turn-end compare at 6430 counts that as an eviction. It repeats every turn (up to 3 pin copies and 3 `<memory>` blocks per request).

**(3) Fix status:** no branch. My scripts are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/eviction-check/`: `real_scan.py`, `turn_waits.py`, `sim/evict_sim.py` and `sim_results2.jsonl`. The issue body is in the same folder as `issue_body.md`.

**(4) Text to inject into the memory-v2 implementation prompt** (line numbers are stack-v2's `tool_agent.py`):

EVICTION TRAP (issue #58): memory text must never count as a history eviction. On Kaggle (10 slots, 25 games) a game loses its GPU slot only when _note_history_evicted (L6375) sets _context_was_trimmed; _maybe_handover (L6454, called at L6827) then calls gate.handover. Measured: turns after a trim take a median of 283 s vs 42 s. Fable's history-replacing journal was counted as an eviction: 30% of game-minutes became pauses, and the score fell 28.65 to 10.49.
Rules:
1. Only a real drop of already-sent messages may reach _note_history_evicted or its checks: the trimmer's count compare (L6626) and the persist compare `history != original_history` (L6430). Never insert, restore, merge, refresh or strip memory text in a way either check can see.
2. Deltas and seen-before lines go only into the NEW opener (_build_user_message). Compose them once and persist them verbatim. Never edit or strip them in older openers; a strip measured 21/31 unpriced re-prefills, or 30 handovers vs 9 when counted.
3. The head card goes only through _apply_due_ledger (L6826): after the trim (L6825), before _maybe_handover (L6827), only when _ledger_due is set. It must not re-render between trims, and nothing but _note_history_evicted may set _ledger_due.
4. No pinned user/assistant pairs and no mid-history removal. If a pin is ever added, dedupe it by a marker, never by content: _with_ledger_head merges <memory> into the head, a content dedupe then re-inserts the pin every turn (27 handovers vs 9).
5. Reserve the card's tokens in the trim (extra_safety_tokens=card tokens) so the card can never push a request over budget into a back-to-back trim.
6. On any exception, leave messages unchanged.
REQUIRED TEST tests/test_memory_eviction.py: one niced subprocess per config, tests/memory_sim.py KAGGLE_ENV, about 31 turns.
- Monkeypatch tool_agent._priority_gate with a fake gate whose handover() records len(requests_seen), the next request's index.
- Wrap the agent's _drop_oldest_history_block, _keep_recent_history_turns, _drop_until_first_user_message and _force_reduce_messages to record len(requests_seen) whenever they actually remove messages.
- Let d be the first differing message index between consecutive requests. Assert:
  (1) every handover index is a recorded drop index;
  (2) d < len(prev) implies a handover at that index, so between trims each request is a message-wise prefix of the next;
  (3) back-to-back full re-prefills (d <= 1) are no more than with flags off (0 under KAGGLE_ENV);
  (4) each request holds at most one <memory> block and one copy of any pinned message.
- Configs: all flags off; ARC3_LEDGER_ON_TRIM=1; ARC3_MEMORY_PLACEMENT=trim, delta and both, each with the ARC3_MEM_* flags on.
- Negative control: a variant that strips the previous opener's delta and calls _note_history_evicted must FAIL assertion (1).
Baseline on stack-v2: flags off gives 8 handovers and ledger-on-trim gives 9, each after a real drop.
