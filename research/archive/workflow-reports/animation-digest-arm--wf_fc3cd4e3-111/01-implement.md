# animation-digest-arm: implement

Workflow run `wf_fc3cd4e3-111`, phase Implement. Implement the merged Wang+Franzen animation mechanism on arm/animation-digest behind default-off flags, then review and fix

---

Branch: `arm/animation-digest` (created from `arm/kaggle-builder`; nothing was pushed, nothing went to Kaggle and no other branch was touched). All seven scope items are in. Lens B (`object_moves`) is left as a follow-up, as asked.

Commit log (`git log --oneline arm/kaggle-builder..HEAD`; every commit ends with the two trailer lines):
```
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

pytest, run with the given command (imports resolve to `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/inference/__init__.py`):
```
51 passed, 14 warnings in 2.76s
```
The 14 warnings are matplotlib/pyparsing deprecations. There were no tests on this branch before, so all 51 are new.

**Flags-off identity test:**
- **How it works.** I extracted the `arm/kaggle-builder` code (4d63980) with `git archive` and recorded what the model sees in four configurations, one of them the competition config. That covers the system prompt, three full turn messages (images hashed) and the sandbox globals and proxy methods.
- **What it checks.** The current code must reproduce all of it byte for byte. It also passes when the dependent knobs are set but `ARC3_ANIM_DIGEST` is off, and when `ARC3_ANIM_DIGEST=1` is set without `ARC3_ANIMATION`.
- **Guard.** One extra test confirms the flag really changes the system prompt, so the identity tests can't pass by accident.
- **Payloads.** A separate test checks that with the digest off the solver's action results gain no keys and the ring stays empty.

**Decisions I made:**
- **`.items()`/`.values()` are gated** behind `ARC3_ANIM_DIGEST`. Adding them unconditionally could change outputs: a snippet that raised AttributeError would now succeed, and `hasattr` checks would flip. With the flag on, the proxies use a subclass that has them.
- **The ring is filled only when the digest is on.** `ARC3_ANIM_RING` does nothing without it, so flags-off runs don't pay for the extra work.
- **Where the digest gets its frames.** The agent collects every animated action's record from each `action()` call, so a long batch can't push the first animated action out of the ring. To bound memory it keeps frames for at most 16 records per call: the first and newest of each signature, plus the first, peak and newest overall.
- **Repeat pointers point at the new action**, e.g. `animation_for(215)`, not at the original (212). The original may have dropped out of the ring.
- **Budgets.** If less than 120 characters remain under the per-game cap, no digest is added and only the existing animation line stays. When over the per-digest cap, trail lines are dropped first, then pointers, then later actions.
- **An opener rebuilt for the same step** (a retried turn) reuses the same digest text and keyframes without charging the game cap again.
- **An invalid `ARC3_ANIM_DIGEST_GATE` fails when the agent starts**, the same way the image-mode settings do.
- **Keyframes come from one action only**: the one shown in full with the most transient cells. They show intermediate boards only, since S0 and the final board are already in front of the model.
- **Coherent trails are listed before scattered ones** (more than three look-alike blobs in one state), so noise doesn't use up the three-trail limit.
- **Level-completing actions stay eligible** for the digest, as they are in Franzen's single record.
- **The tests and fixture travel inside the Kaggle notebook patch**, because the builder diffs all of `ARC3-Inference`. Both arms build locally, and the `anim-digest` notebook is 353 KB, well under the 950 KB limit.

**Not done, or limits:**
- **Lens B** (rigid-motion lines from `frame_diff` with the background check) is not built. Lens A reports a slide as two trails, for example `Y 2x2` then `Y 2x1`, rather than "moved by +3".
- **Long animations.** When an animation has more than 32 states, the S-numbers and the `animation_for` timeline refer to the 32 stored states. `last_animation_timeline` still uses the full chain, so in that case the two numberings differ. The digest and the `animation_for` result both say when this happens.
- **No real-game measurement yet.** On a synthetic 64x64, 112-state animation the extra work was about 100 ms, but I didn't measure real boards or Qwen token counts.

Files are in `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/`:
- `ARC3-Inference/inference/utils/animation.py`
- `ARC3-Inference/inference/framework/solver.py`
- `ARC3-Inference/inference/agent/tool_agent.py`
- `ARC3-Inference/inference/agent/python_tool_sandbox.py`
- `ARC3-Inference/inference/agent/vision_context.py`
- `ARC3-Inference/inference/agent/prompts.py`
- `ARC3-Inference/CONFIGURATION.md`
- `kaggle/arms.json`
- `ARC3-Inference/tests/` (`test_animation_digest.py`, `test_animation_ring.py`, `test_animation_sandbox.py`, `test_animation_opener.py`, `test_flags_off_identity.py`, `flags_off_snapshot.py`, `solver_fakes.py`, `fixtures/flags_off_snapshot.json`)
