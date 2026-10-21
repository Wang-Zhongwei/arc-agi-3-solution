# animation-digest-arm: review

Workflow run `wf_fc3cd4e3-111`, phase Review. Implement the merged Wang+Franzen animation mechanism on arm/animation-digest behind default-off flags, then review and fix

---

Verdict: 5 defects, none touching flags-off behaviour. Flags-off is byte-identical to `arm/kaggle-builder`, and all 51 tests pass with the given command.

**What I checked and found clean**
- **Flags-off identity.** I ran base (`git archive` of `arm/kaggle-builder`) and head side by side. Across 5 env configs, a scripted session gave identical tool output, step summaries, `animation_record`, history results and openers (images hashed), with the batch no-op guard both on and off. The 5 configs were `ARC3_ANIMATION` alone, timeline off, image modes, and `ARC3_ANIM_RING`/`GATE`/`KEYFRAMES` set without the master flag. A separate snapshot-script comparison of 7 scenarios against base was also identical.
- **Crashes.** I fuzzed `overlay_trails` and `build_animation_digest` over 3000 random chains: ragged, resized, 1x1 and 64x64 grids, 0 to 40 frames, all gates, budgets 0 to 5000. No exceptions, and the per-digest cap held every time. Level win, game-over inside a batch, RESET, refused batch and no-frames cases also ran without crashing.
- **Prefix cache.** The digest and keyframes only go into the new tail user message, and a rebuilt opener reuses the cached text without re-charging. Nothing earlier is rewritten.
- **Hot path.** On 64x64 boards, with 2, 15 and 111 states plus a worst case of 2000 random cell changes per state, the extra cost is small:

  | Item | Cost |
  |---|---|
  | `_animation_ring_record` | 2 to 94 ms |
  | digest, 1 record | 1 to 109 ms |
  | digest, 3 records | 2 to 221 ms |

  `_collect_animation_records` is O(N) per `action()` call in the number of records collected so far.

**Defects, most severe first**

1. **Signatures shown in a level-completing batch are filed under the new level.**
   - Where: `agent/tool_agent.py:5850-5872` (`_animation_digest_lines`).
   - The level check uses `summary["level"]`, which is the post-action level, and `shown` is then written into the reset seen-set.
   - Repro: `action(['SPACE','LEFT'])` where LEFT completes the level. `_anim_seen_sigs` ends as `{sigSPACE: 1}` with `_anim_seen_level == 2`. The next level-2 `SPACE` then prints "repeats the animation of action 1 (same signature)". This contradicts the design's "same animation on a new level is worth showing again" and the CONFIGURATION claim "reset on level change".
   - Fix: only add signatures whose `record["level"] == summary["level"]` to `_anim_seen_sigs`, or key the seen-set by `(level, signature)`.

2. **A repeat pointer can name an original that the digest never showed.**
   - Where: `utils/animation.py:766-806` (`first_in_call`, then pointers, then `chosen`).
   - The pointer's "original" is the first same-signature record in `full`, but `chosen` is only first, peak and last. If that original is not chosen, the digest says "repeats the animation of action 2" with no description of action 2 anywhere, and the signature is not added to `shown`, so the next turn treats it as novel again.
   - Repro: a 14-action batch of SPACE, 11 flash clicks and 2 sweep clicks printed "action 12 repeats the animation of action 2 ... as do 9 earlier", with the action-2 header absent and "(+1 more)".
   - Fix: only emit a pointer when `original` is in `chosen` or already in `seen`. Otherwise count it in `omitted`, or add the first record per signature to `chosen`.

3. **`animations` rides into the model-visible action result, duplicating `animation` and growing without bound.**
   - Where: `agent/tool_agent.py:5777` (passthrough list) and `framework/solver.py:1259`.
   - It is placed before `stop_reason`, `stop_detail` and the timing keys, so truncation cuts those first.
   - Each entry is about 125 characters, roughly 40 tokens. A single animated action's result now carries it twice. A 30-action animated batch printed with `print(r)` was cut inside `animations` at the default 1024-token tool-output budget, so `stop_reason` was lost.
   - It also appears in `last_action_call_result` and in the auto `result` when the snippet prints nothing.
   - Fix: keep `animations` agent-side. Stash `raw_payload["animations"]` in `_handle_action` and have `_summarize_step_sequence` read it from there. If it must stay in the result, drop it when there is only one animated action, cap it, and emit it after `stop_*`.

4. **Pointers to `animation_for(n)` and "`animations` lists the readable ones" can name actions that were evicted from the ring.**
   - Where: `framework/solver.py:131` (default `ANIM_RING_SIZE` = 1) with the digest tail in `utils/animation.py:851-870`.
   - With `ARC3_ANIM_DIGEST=1` and `ARC3_ANIM_RING` unset, a 3-action batch printed a digest naming action 1, but `animations` was `[3]` and `animation_for(1)` returned "No animation is kept".
   - It also happens with ring=8 when a batch has more than 8 animated actions and the first or peak action is chosen. The system-prompt addendum says recent animated actions "stay readable".
   - Fix: default the ring to at least 8 when the digest is on. Also only emit `animation_for(n)` for actions still in the ring. The agent can read `ARC3_ANIM_RING`, or `animation_for` can fall back to the agent's collected records.

5. **A subsampled record's digest points at the wrong numbering.**
   - Where: `utils/animation.py:851-854` (`only_last and timeline` tail).
   - S-numbers refer to the stored 32-entry chain, but the tail says "step k = last_animation_frames[k-1]" and "Exact cells: last_animation_timeline", which use the full chain.
   - Repro: 61 states printed "31 states ... subsampled from 61", then the tail above, while `last_animation_frames` has 61 entries.
   - Fix: when `summary.frames > stored`, use the `animation_for(n)` tail, which does index the stored states.

**Nits**
- CONFIGURATION says keyframes are "never stripped". An opt-in `ARC3_HISTORY_IMAGE_KEEP` still strips them to a placeholder, like every other image.
- "Capped at 32 states" is really 32 chain entries including S0, so 31 states.
