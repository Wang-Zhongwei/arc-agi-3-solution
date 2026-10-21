# animation-digest-arm: fix

Workflow run `wf_fc3cd4e3-111`, phase Fix. Implement the merged Wang+Franzen animation mechanism on arm/animation-digest behind default-off flags, then review and fix

---

I reproduced all 5 confirmed defects on `arm/animation-digest` before changing anything, and fixed all of them; none was rejected. The two nits are fixed too. The branch was not pushed and no other branch was touched. The full suite now passes, 59 tests. The flags-off identity test still passes.

**Per finding**

1. **Fixed** (4474adf). Repro confirmed: after `action(['SPACE','LEFT'])` the seen-set held SPACE's signature under level 2, and the next level-2 SPACE printed as a repeat. In `_animation_digest_lines`, a shown signature is now added only if its record ran on the summary's level. New test: `test_level_completing_batch_does_not_file_old_level_signatures`.

2. **Fixed** (5429b6a). Repro confirmed: the 14-action batch printed "action 12 repeats the animation of action 2" with action 2 never described. A pointer is now kept only if its original is in `seen` or is shown in full in this digest. Otherwise the repeat is counted with the omitted actions and its signature stays novel for the next turn. New test: `test_digest_pointer_never_names_an_unshown_original`.

3. **Fixed** (3153448). I took the reviewer's fallback rather than keeping the list agent-side only, because the design plan calls for every animated action of a batch to appear in the action result.
   - It is left out when it would only repeat `animation`, meaning the one animated action was the last one executed.
   - It keeps at most the newest 8 entries, with `animations_omitted` counting the rest.
   - It is now the last key in the result, after `stop_*`, the timing keys and `error`, so a truncated print loses it first.
   - `_summarize_step_sequence` now reads the full list from an agent-side stash of the raw payloads (`_anim_call_animations`).
   - Three new tests cover this, and `CONFIGURATION.md` is updated.

4. **Fixed** (9db968e). Repro confirmed: with the ring at 1, `animation_for(1)` returned "No animation is kept".
   - `ARC3_ANIM_RING` now defaults to 8. The ring is only filled when the digest is on, so flags-off behaviour does not change.
   - After each python call the agent asks the ring which actions it still holds, without spending an action, and passes those numbers to `build_animation_digest`.
   - `animation_for(n)` is now named only for actions still in the ring. Pointers to evicted actions end without it, and the footer says "Action 1's frames are no longer kept."
   - Two new tests cover this, and `CONFIGURATION.md` is updated.

5. **Fixed** (6962a38). Repro confirmed: the digest said "31 states, subsampled from 61" while `last_animation_frames` had 61 entries. A subsampled entry now gets the `animation_for(n)` footer instead. I checked that `animation_for(1)` returns 31 frames and 31 timeline steps, which match S1 to S31. New test: `test_subsampled_last_action_points_at_the_stored_states`.

**Nits** (88440bd): `CONFIGURATION.md` and the keyframe docstring now say an `ARC3_HISTORY_IMAGE_KEEP` window strips keyframes like any other image. The cap is described as 32 boards including S0, i.e. 31 states.

I also ran 1,500 random digests of up to 15 actions through the builder. None crashed or went over its character budget, every `animation_for(n)` it printed was for an action still in the ring, and every repeat pointer named an action shown earlier or in the same digest.

**git log arm/kaggle-builder..arm/animation-digest --oneline**
```
88440bd Correct the keyframe stripping and stored-frame cap wording
6962a38 Point a subsampled digest entry at the stored states
9db968e Keep 8 animations by default and point only at ones still kept
3153448 Bound the action result's animations list and put it last
5429b6a Point a digest repeat only at an original the model can find
4474adf File digest signatures under the level their action ran on
fa0e48e List coherent animation trails before scattered ones
04ddf63 Document the animation digest flags and add its Kaggle arms
5d58289 Attach animation keyframe images with the digest
6e69231 Describe the animation digest in the system prompt
ab30566 Append an animation digest to the turn opener
e8c2b8b Expose recent animations to the python sandbox
afee76a Exempt hiding animations from the batch no-op stop
3240195 Keep a ring of recent animation records in the solver
da76334 Pin what the model sees with the animation digest flags off
0bb9d3b Add animation digest helpers to utils/animation
```

**pytest:** `59 passed, 14 warnings in 3.18s`

Files changed:
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/agent/tool_agent.py`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/utils/animation.py`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/framework/solver.py`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/CONFIGURATION.md`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/tests/test_animation_digest.py`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/tests/test_animation_opener.py`
- `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/tests/test_animation_sandbox.py`
