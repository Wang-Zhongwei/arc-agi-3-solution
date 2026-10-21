# Harness systems gap: 13 candidate systems after 3 skeptic passes

Date: 2026-10-09. Input: 13 merged candidate systems (G1-G13). Each was checked by three skeptic lenses (model-fit, evidence, systems-cost); a lens either refutes it or not and gives a fix. Stack under test: user-combo-10x (56.94 / 52.70) vs control 50.62. Model: Qwen3.8-Flash-Next W4A16 + MTP. Real-run sd is ~1.7-4, so +3 needs ~6 paired runs per arm.

## 0. Ranking rule and scores

Score = E x p / C, where:

- E = realistic expected points if the system ships. This is the skeptic-consensus midpoint, not the proposer's number.
- p = P(passes its no-GPU gate) x P(the model uses the channel).
- C = engineering days, including the Stage 0 replay.

Order: survivors (<=1 refutation) first, then survived-with-fix (concrete, cheap reduced form), then rejected.

| rank | system (form to build) | refutations | status | E (pts) | p | C (days) | score |
|---|---|---|---|---|---|---|---|
| 1 | G1-core: avatar + view_shift in result dict, change-only tail line | 1/3 | survived | 1.5 | 0.5 x 0.6 = 0.30 | 2.5 | 0.18 |
| 2 | G2: colour-only lattice view (`cells`/`tile_of`), redesigned estimator | 1/3 | survived (detector failed its spec'd Stage 0) | 1.0 | 0.3 x 0.5 = 0.15 | 2.5 | 0.06 |
| 3 | G9-lite: level-start function catalogue + clearing-function pin + guarded auto-watch (rider on sandbox-robust #76) | 3/3 full form | survived-with-fix | 0.3 | 0.7 x 0.6 = 0.42 | 0.75 | 0.17 |
| 4 | G3-lite: object-level surprise line at the existing batch-noop/stale stop (no DSL) | 3/3 full form | survived-with-fix | 0.75 | 0.5 x 0.6 = 0.30 | 1.5 | 0.15 |
| 5 | G8-reduced: computed trim card (colour/shape_hash win signature, predicates, counters), gated on the memory-v2 solved-levels run | 2/3 | survived-with-fix | 1.5 | 0.4 x 0.6 = 0.24 | 3 | 0.12 |
| 6 | G10-lite: alias retry only when 0 actions executed (sub-flag of #76) | 2/3 | survived-with-fix | 0.15 | 0.4 x 1.0 = 0.40 | 0.5 | 0.12 |
| 7 | G5-lite: harness executes the model's own REQUIRED `probe` on an inspect-streak trip | 2/3 | survived-with-fix | 1.0 | 0.3 x 0.9 x 0.6 = 0.16 | 2 | 0.08 |
| 8 | G13-lite: offline `analysis/signals.py` + `make screen` + preregister.md template | 2/3 | survived-with-fix (infrastructure) | 0 direct | n/a | 1 | 0 (enabler) |
| - | G4 expect-and-halt | 3/3 | rejected | - | - | - | - |
| - | G6 experiment log (as a new arm) | 3/3 | rejected: already built as memory-v2 | - | - | - | - |
| - | G7 forced restart with facts kept | 3/3 | rejected | - | - | - | - |
| - | G11 inspect budget yield | 3/3 | rejected | - | - | - | - |
| - | G12 level-start affordance sweep | 3/3 | rejected | - | - | - | - |

Sum of the realistic E over everything ranked: ~+6, if every gate passed and the effects added up. No single system plausibly reaches +3 by itself. Each one counts only as a near-free rider, so the unit to screen is a stack (section 2).

## 1. Ranked systems

### Rank 1. G1-core: controllable-entity tracker, cut to the near-free core (survived, 1/3)

- **Mechanism.**
  - Chain frame_diff's pairwise Hungarian match (utils/frame_diff.py:82-230) into stable per-level ids (E1..En) with colour, shape_hash and bbox.
  - Avatar = the id whose displacement is a consistent function of the keyboard action over >=3 effectful presses, with no competing id.
  - A confidence count drops on any contradiction. Below 0.8 the field is `None` (silent).
  - Remove the best global translation before matching, so a scrolling board is reported as one `view_shift` (dr,dc), not N moved objects.
  - Dropped (skeptic consensus): occupancy grid, `.bfs`, `entities`/`ego`/`scene.graph` views, ARC3_FIRST_LOOK, the `events` list and `blocked_edges_learned` in the result dict.
- **Channel.**
  - `avatar` {id, bbox, cell, moved} and `view_shift` on every `transitions[i].result` and `last_action_call_result`. The model reads these dicts in code: gameplay_changed in 35 bp35 turns, animation in 24, last_action_call_result in 61 sc25 turns.
  - One tail line only when something changed (<=40 tokens), e.g. `avatar 3x3 blue (12,30)->(12,36); view shift 0`.
  - One static system-prompt sentence (<=80 cached tokens) naming the fields.
- **Cache safety.** Safe. Tail-only and deterministic (sorted ids, no clocks). Result keys live on the Python side. Requests are byte-identical with the flag off.
- **Expected points.**
  - Proposer: +2 to +4. Skeptics: -1 to +2, realistic ~+1.5 if Stage 0 passes.
  - Arithmetic: 18 keyboard/mixed games score 40.9-56.8 vs 63.8-75.7 for click games. One extra level on an unfinished game is worth 0.6-1.2 mean points.
  - Transcripts support a wrong map as the cause of a lost level only in g50t (parser bugs + clock; L6 = +0.86). bp35 was analysis paralysis, and Makarov finds 83% of stuck forks had already tried the decisive action.
  - So realistically 1-2 conversions per run, about +0.6-2.
  - Token side: removes some of the 3-5 position-helper rewrites per game (`bbox` rebuilt 6,791 times in r2).
- **Cost.**
  - Tokens: 30-60 per acting turn when something changed, 0 otherwise. CPU <50 ms per transition.
  - Engineering: ~1.5 days for the core plus 1 day of Stage 0.
- **First falsifiable test (Stage 0, no GPU).** Replay the tracker over every transition in the 7 full-run histories.
  - PASS (all needed):
    - a unique avatar within 10 effectful actions in >=70% of keyboard/mixed levels
    - precision >=90% at confidence >=0.8, and fewer than 5% of batches confident-but-wrong
    - 0 false avatars in click games
    - id stable for >=90% of actions to the level end
    - the counterfactual check (below) holds
  - Counterfactual (evidence lens): on stalled levels, count turns where the model's own stated player cell or blocked map was wrong AND the tracker was right and confident. Need >=3 such levels per run.
  - KILL: any bar missed. A confident wrong line is worse than none.
  - Stage 1 (25-min panel): adoption = code reads of `result['avatar']` / `['view_shift']`, counted the way gameplay_changed reads are counted. KILL at <=6%, the STATE_TRACKER precedent (2/33). Mechanism check: model-written position-wrapper defs (safe/cells/bb/pos) per game drop >=50% vs panel-control.
- **Closest dead end.**
  - state-manager / STATE_TRACKER (inconclusive): a harness state block in the user turn, cited in ~6% of requests.
  - research-sweep dropped `effects()`/`track()` as per-call helpers.
  - ARC3_AUTO_FRAME_DIFF: 52% of 'moved' entries are <=2 px noise.
  - How G1 differs: persistent identity plus contingency counts give one tracked object, not a change list. It goes into result-dict fields the model reads in code, not an opener block.
  - The 6% precedent is the main risk, so it is the kill bar.

### Rank 2. G2: lattice / tile-map service, colour-only, with a new estimator (survived, 1/3; the specified detector failed)

- **Mechanism.**
  - At level start, estimate tile pitch and offset. If accepted, expose a cell map: dominant colour, purity, tile class, `.at(px)`, `.crop(r0,c0,r1,c1)` with a ruler, and `tile_of(cell)`.
  - Passability and `path()` come only after G1 ships and is validated.
  - **The boundary-autocorrelation estimator must be replaced.** Two skeptics measured it on the v4-user-combo-10x/v1 level-start boards:
    - At >=0.85 on both axes it accepts only cn04 (1.00), sp80 (1.00) and tu93 (0.95-0.99).
    - It rejects every payoff game: g50t 0.34-0.51 (L5 at pitch 6: 0.28), bp35 0.37-0.38, lf52 0.41-0.46, sk48 0.42-0.52, ls20 0.64-0.78, dc22 0.40-0.72, wa30 0.45-0.71.
    - A +-1-tolerant variant accepts 4/25 (sp80, m0r0, vc33, wa30). g50t comes out at 7 px on L1 and at 6 px (0.68) on L6.
  - Replace it with modal same-colour-component bbox size and/or run-length histograms of wall-colour segments. Reuse the lattice-node code in arm/attention-prompt salient_facts.py:199-201.
- **Channel.**
  - Advertised by signature plus one worked example in the tool description, exactly like frame_diff.
  - One fact line at level start (~25-30 tokens).
  - Adoption precedent is measured on v1 (2,713 unique python calls):
    - `.segmentation`: 681 calls (25.1%), >=3 uses in 25/25 games
    - `frame_diff(`: 284 calls (10.5%), >=3 uses in 23/25 games
    - the model already uses cell-size constants or `//k` in ~11/25 games
  - death_ledger's 0 uses is not a counterexample: it was never advertised.
- **Cache safety.** Safe. Static tool-description text plus a line in the level-start opener.
- **Expected points.**
  - Evidence lens: +0.5 to +1.5. The sim-slope token conversion is dropped. It keeps the g50t-class level loss (+0.86) and parser-version waste in tile games (4-5 defs x 3-8k tokens).
  - Conditional on the estimator working, ~+1.
- **Cost.**
  - Tokens: ~25-30 per level start, 40-80 cached.
  - Engineering: 1.5-3 days (estimator + view + line). Passability is deferred.
- **First falsifiable test (Stage 0, no GPU).** Run the new estimator on the level-start boards in `runs/kaggle/*/artifacts/*_events.jsonl.xz` and hand-label ~50 frames.
  - PASS (all needed):
    - accepted on >=8/25 games, including >=4 of {g50t, ls20, wa30, lf52, sk48, dc22, bp35}
    - agrees with the model's own pitch on the ~11 games it derived one in (ar25/cn04/tu93/re86/ka59 = 3, sp80 = 3-4, m0r0 = 4-5, g50t = 6, lf52 = 6, ft09 = 8, bp35 = 6/10)
    - 0 false lattices on free-form games
    - silent when the bp35-style camera offset moves
  - Note (systems-cost lens): the model's own g50t '6 px' is itself approximate. Hand labels outrank it.
  - KILL: fail, and then ship only the Tier-4 subset (frame.crop with ruler + bbox/center on nodes, ~0.5 day) inside Stack A.
  - Stage 1: KILL if `cells`/`tile_of` appear in <10% of python calls on accepted games (the frame_diff floor), or if `.ascii` crop calls do not fall.
- **Closest dead end.**
  - numpy-grid port (model read `.grid` in 2/17 ar25 calls), Tier 4 scene card (+0.5-1), and trimmed perception helpers (2/5).
  - How G2 differs: a tile-semantic view on an existing code habit (`//k`, classify/cm/cm2), not a raw array or a prose card.

### Rank 3. G9-lite: function catalogue + clearing-function pin + guarded auto-watch (survived-with-fix; full form refuted 3/3)

- **Mechanism.**
  - On top of arm/sandbox-robust (#76: PERSIST_VARS, import shims, `inspect.getsource` shim at python_tool_sandbox.py:318-349):
    - At level start, append a catalogue to the existing retained-function opener line: <=12 entries, clearing functions first, each with signature + first docstring line + calls/last-outcome. Add the full source of <=2 clearing functions only if it fits in 300 tokens.
    - On a sibling define, report 'retained foo2 (supersedes foo)'.
  - Auto-watch: up to 4 retained one-argument functions, with all of these true:
    - the source contains no `action(` call
    - the last recorded call bound the argument to a FrameView
    - the result was <=80 chars
    - it runs under a 2 s cap
  - Their results go into the action result as `watch: {...}`.
  - Measured on v1: 1-argument functions are 29/44 (g50t), 34/69 (tr87), 13/22 (bp35), and include action wrappers `safe(seq)` and `run(seq, tag)`. Hence the guard.
  - Dropped: lib.test, macro, `lib()`/`source()` (the #76 shim covers it), snapshot diff (tautological: siblings are written to differ), dependency inlining (subsumed by PERSIST_VARS and re-creates the REG6 drift).
- **Channel.** The opener retained-function line (read today) plus result-dict `watch` (result-dict fields are read in code). No opt-in.
- **Cache safety.** Safe. Opener tail and tool-result tail only.
- **Expected points.**
  - 0 to +0.5. Error-tax arithmetic (NameError/ImportError carry-over 2.2% of calls x 1.2k tokens) belongs mostly to #76.
  - tr87-style cross-level reuse: 123 function names were called on a later level than they were defined, in 23/25 games on v1.
  - 624 of 1,085 functions were never reused (57%), 260 same-name redefinitions, 184 numeric siblings, 306 'unavailable dependencies' refusals in 2,706 results (11%, which #76 targets).
- **Cost.** ~0.5-1 day. <=300 tokens per level start, 15-30 per batch for watch.
- **First falsifiable test.**
  - Stage 0: from the 9 runs' logs, need >=1 eligible guarded one-argument helper per keyboard game.
  - Panel: 0 watch-triggered action executions or timeouts; watch lines in >=50% of keyboard-game action results; catalogue entries quoted in the same turn.
  - KILL: model-written per-step wrapper count (safe/bar loops) does not fall, which would make watch additive noise; or any watch-triggered action.
- **Closest dead end.** LHS save_module + frozen-observation tests (research-sweep dropped: opt-in, 2-3 days, low payoff), starter helper library (dropped), persistent kernel (dropped: <1% tokens). How G9-lite differs: it is harness-pushed at the one channel already read, with zero opt-in.

### Rank 4. G3-lite: surprise line at the existing stop (survived-with-fix; DSL induction refuted 3/3)

- **Mechanism.**
  - Per action id, a constant-displacement table for the largest moving component, from compute_frame_diff's `_hungarian` (or from G1's avatar if present).
  - Interior-masked with `_interior_gameplay_changed`, so the HUD bar never counts as a residual.
  - Only when the existing ARC3_BATCH_NOOP_BLOCK / stale-state stop fires (solver.py:1019-1047), add one object-level line to `stop_detail`, e.g. `expected E1 (4,5)->(4,6) on LEFT (5/5 before); stayed; target cell colour F`.
  - Log zero-surprise and surprise counters to events.jsonl only.
  - Dropped: transition DSL, decision-list preconditions, new halts, carry-over across levels, the `rules` view.
- **Channel.** stop_reason / stop_detail plus the opener's 'stopped before' sentence. These lines are quoted the same turn (bp35 read stop_reason in 6 code turns; g50t's 13 stale-state stops were acted on).
- **Cache safety.** Safe. Tail-only and edge-triggered.
- **Expected points.**
  - +0 to +1.5. The g50t losses were already halted by the existing stop. The increment is the explanation, and only where the model's map is wrong in harness pixel coordinates.
  - It does not fix the model's own cells() +1 column offset.
- **Cost.** ~1-1.5 days. ~30-40 tokens per stop. Stop fires per control run: no_op 19, game_over ~9, stale_state 3.
- **First falsifiable test (Stage 0 replay, 7 runs).**
  - Prospective exactness of constant-displacement vs a nothing-changes predictor per action id.
  - False-surprise rate from matcher noise <10% of actions (52% of AUTO_FRAME_DIFF 'moved' entries are <=2 px).
  - Line would fire on >=70% of the batches the stale-state stop halted in g50t/wa30/lf52.
  - KILL: false-surprise >=10% or coverage <40%.
  - Panel: >=30% of post-line turns quote it (KILL <10%), and no rise in inspect-only turns.
- **Closest dead end.**
  - ARC3_STALE_STATE_BLOCK (worked: no-op only, no diff) and ARC3_AUTO_FRAME_DIFF (unrun, noisy).
  - research-sweep dropped prediction-checked batches (97% of loss is incomplete levels).
  - DF-WM check_sim (29.30 = noise).
  - How G3-lite differs: it adds no halt and no model prediction; it explains an existing halt.
  - Full DSL stays in the backlog, conditional on G1 ids plus >=70% of keyboard levels having a >=0.90 prospective-exact rule.

### Rank 5. G8-reduced: computed trim card (survived-with-fix, 2/3)

- **Step 0 (zero build).** Run the already-built memory-v2 `solved-levels` (ARC3_MEM_SOLVED=1, ARC3_MEM_REPLY_WINS=1, ledger on) on user-combo-10x. This measures 'computed facts vs harvested 200-char sentences' with no new code.
- **Mechanism (if step 0 >= stack and Stage 0 passes).** At level_completed, compute G1-free content:
  - clearing action and target cell
  - last <=5 actions
  - vanished/appeared shape_hash classes
  - border-excluded HUD cells that moved
  - a predicate from a small DSL: count(colour)==0/==k, cell-set(A)==cell-set(B), region-pattern equality, HUD cell==target
  - monotone interior per-colour / per-shape_hash counters, with the budget bar masked
- **Design bug to fix first.** After level_completed, `current_frame` is the NEXT level's first board (solver.py:1255-1262). A predicate 'true on the clearing frame, false on earlier frames' will trivially separate two layouts. Fit it on the pre-clear frame plus clearing action, or on animation-chain frames, and check whether the chain contains the completed level's end state.
- **What is shown.**
  - The card replaces the auto-harvested notes in the trim-time `<memory>` block (100/108 notes are cut at 200 chars).
  - An edge-triggered goal-check line (<=25 tokens) on levels >=2.
  - Dropped: sandbox globals (except possibly `source(fn)`, which the #76 shim covers), the free-text per-level ask, level-start opener placement (debrief objection: it duplicates retained history), and per-action counters (anchoring risk; the model already fixates on the bar in 11/25 games).
- **Channel.** The ledger-on-trim slot, the only broad local win (52.56/49.14 vs 46.25, not yet significant). The `<memory>` block was referenced 45x in g50t.
- **Cache safety.** Safe. It reuses `_with_ledger_head` (tool_agent.py:143-178 on arm/user-combo). Re-run the eviction guard (#58/#61) with the ~300-token block.
- **Expected points.**
  - Proposer: +2 to +3. Evidence lens: +0 to +1.5. The ledger 'decay' is controls catching up, and near-win losses were slot time (04-research-throughput-games §2).
  - Realistic ~+1 to +1.5, from the L2 conditional clear rate (86%; ~3.5 L2 failures per run).
- **Cost.**
  - Tokens: 200-300 per trim (5-8 trims per game), ~25 per goal-check change.
  - Engineering: 2-3 days on top of step 0, plus 1 day of Stage 0.
- **First falsifiable test (Stage 0, no GPU, budget bar masked, predicate on the pre-clear frame).**
  - >=40% of clears have a terminal monotone counter
  - >=60% have a separating predicate
  - >=40% of consecutive-clear pairs have the re-bound predicate false at the next level's start and true at its clear
  - predicate true on <=5% of non-clearing frames
  - KILL: any bar missed, or memory-v2 solved-levels < stack.
  - Full-run mechanism gate: card tokens quoted in >=20-30% of shown turns (KILL if 0 in >20/25 games), signature hand-audit >=80%, wrong counter quoted as the goal in <=2 games.
- **Closest dead end.**
  - Level-completion debrief (dropped: duplicates history), memory-v2 MEM_SOLVED (unrun), ARC3_LEVEL_INVENTORY (dropped: 175-3983-cell start diffs), Tufa cross-level notes (0/989).
  - How G8-reduced differs: computed and verified facts in the proven trim slot, no model ask.

### Rank 6. G10-lite: alias retry with 0 actions executed (survived-with-fix, 2/3)

- **Mechanism.**
  - When a python call fails with a view-field AttributeError (`.seg` -> `.segmentation`) or a harness-dict KeyError in an alias map built from the 9-run error corpus (56 view AttributeErrors, 28/83 KeyErrors on harness keys), and `host_action_results` is empty, rewrite and re-run once inside the call. Report `corrected .seg -> .segmentation and re-ran`.
  - Dropped:
    - auto-restore by re-executing archived source (re-evaluates against the current board, re-creates the REG6 drift, bypasses #76's literal-only reader)
    - retry after actions: 45/136 errored calls on v1 had already executed actions before the traceback
    - syntax pre-check (no round trip saved)
    - batch-safety advisory
- **Channel.** Tool-result line, ~10-30 tokens.
- **Cache safety.** Safe. Tail only.
- **Expected points.** ~+0.1 to +0.3. ~10-40 rescued events per run x ~1.2k tokens = 0.3-1% of tokens. It is insurance, not detectable in score.
- **Cost.** ~0.5 day as a commit on arm/sandbox-robust.
- **First falsifiable test.**
  - Re-run the 17,490-call corpus through #76 with flags on.
  - Build only if residual NameError + AttributeError + KeyError is >=1% of calls.
  - Replay with the retry: KILL on any changed stdout/actions for calls that succeeded, or any retry of a snippet that executed actions.
  - Panel: error calls per 100 python calls fall by >=1 (KILL otherwise).
- **Closest dead end.** Persistent per-game kernel (dropped: <2% of calls, <1% of tokens) and api-schema-errors (<=1%). How G10-lite differs: it fixes harness-API misuse inside the call and touches no state.

### Rank 7. G5-lite: harness executes the model's own probe (survived-with-fix, 2/3)

- **Mechanism.**
  - Static REQUIRED python-tool parameter `probe` ('<=3 engine actions' or 'none:<reason>').
  - The harness executes the model's own proposed probe only when the breaker trips:
    - K=3 inspect-only calls at an unchanged interior hash within one analysis step, or 1 inspect-only call after a death diagnosis
    - AND the level is past the population median clear cost
    - AND the reasoning-chars-per-action EMA is above ~4x the run median (bp35 13.3k vs tr87 1.3k)
  - Never: RESET, death-ledger pairs, fatal prefixes, or levels whose acting turns average >=5 executed actions (won profile: tr87 8.3, g50t 5.9).
  - Recorded as automatic=True, plus one fact line.
  - Dropped: harness-chosen 'EIG' chooser (on hashed boards every pair is untried, so it reduces to a random action), the two-keyed coverage table, class ids.
- **Why the extra gates.** Model-fit replay of plain K=3 on v1 request logs:
  - tr87 (won 6/6): 14 trips, 0.17 per call, max streak 15
  - bp35: 11 trips, 0.12 per call
  - g50t: 17 trips, 0.10 per call
  - sc25: 2 trips, 0.02 per call
  - Half of all trips land in cleared games, and tr87 has no UNDO and cipher state.
  - Pool (from the G11 replay): 117 streaks of >=4 non-acting calls = 1.83M tokens (34% of 5.30M); 894k tokens generated after the 4th call.
- **Channel.** REQUIRED tool parameter (89-93% fill precedent) plus a harness-executed transition plus a tail fact line.
- **Cache safety.** Safe if the schema is fixed from game start and the line is tail-only.
- **Expected points.** 0 to +2, sign uncertain. bp35's untested alternatives were multi-step click plans; the 83%-already-tried fact argues against single-step novelty.
- **Cost.** ~1.5-2 days. Reuses thinking-digest v2 parameter plumbing and the action path. 10-30 generated tokens per call.
- **First falsifiable test (Stage 0, no GPU).**
  - On the stuck-vs-success fork data and the 7 runs: share of trips (with the new gates) landing on levels later cleared. KILL if >40%.
  - Share of forks whose decisive divergence is a <=3-action probe from the current board. KILL if <50%.
  - Panel:
    - fill >=85%; KILL if 'none:' is >50% of trip-time values
    - inspect-only calls in the 3 calls after a probe vs before; KILL if it rises
    - probe deaths <=1 per run
- **Closest dead end.**
  - Kinetic rule 8 'no-action circuit breaker' (research-sweep dropped, no gain; v8 27.28 = noise), Franzen guards (refusals mislead: agentfix), hypothesis-loop (filled, tied).
  - How G5-lite differs: it is the first arm that executes an action on the model's behalf, and the model authors the probe.
  - Fallback if the gates fail: a fact-only 'untried here: ...' line in the memory-v2 seen-state component, with no action.

### Rank 8. G13-lite: offline signals + `make screen` (survived-with-fix, 2/3; 0 direct points)

- **Mechanism.**
  - `analysis/signals.py` derives a per-level and per-turn signal table from existing `*_requests.jsonl.xz` + `*_events.jsonl.xz` + `benchmark.json` + `score_timeline.jsonl` across all 108 saved runs. Signals: inspect-only streak, reasoning chars per executed action, distinct-board rate, exact-repeat share, deaths, deliberate vs automatic RESET, spend vs KM clear-cost quantile, wall-clock age.
  - `make screen` wraps `make significance` + the matched-minute table + signals.
  - preregister.md template emitted by `kaggle/build_notebook.py --arm`.
  - Dropped: live shadow priorities and epsilon handovers. sched-hidden closed the scheduling question (best +0.25), and ~4-5 randomised handovers per run with IPW weights of 10 cannot resolve anything.
- **Channel / cache.** None (offline).
- **Expected points.** 0 direct. It runs the Stage 0 gates of G1/G3/G5/G7/G8 from one place.
- **Cost.** ~1 day. Goes to main under the research-data-on-main rule.
- **Test.** Reproduces the transcript reader's tr87/bp35/g50t/sc25 numbers and failure_taxonomy's per-run totals.
- **Closest dead end.** failure_taxonomy (post-hoc, per-report), sched-hidden pre-registration (one-off). How G13-lite differs: one entry point over every saved run.

## 2. Stacks (batch rule: up to 5 features, screened together)

Order matters: Stack B is nearly all built code, so screen it first.

### Stack B: memory and code facts (build ~1.5 days; screen first)

Features:

1. sandbox-robust #76 (ARC3_SANDBOX_PERSIST_VARS + SHIMS). Built, unrun.
2. memory-v2 `solved-levels` (ARC3_MEM_SOLVED, REPLY_WINS=1). Built, unrun. This is G8 step 0.
3. memory-v2 `action-effects` + `seen-state` (ARC3_MEM_EFFECTS, ARC3_MEM_SEEN), plus the G6 delta patch:
   - 'repeat #k (steps a, b)' suffix
   - death run-length clause on the hazard line, worded as 'board interior matched' so hidden-state mechanics are not claimed as proven no-ops
4. G9-lite catalogue + clearing-function pin + guarded watch.
5. G10-lite alias retry.

- **Gate before GPU:** exact (interior hash, action) repeats on never-cleared levels >=5% of their actions across the 7 runs (memory-v2 recorders).
- **Screen:** one 25-min panel for mechanism metrics:
  - quote rate of seen/solved lines
  - error calls per 100 python calls
  - 'unavailable dependencies' refusals -60%
  - 0 watch-triggered actions
- **Then:** 2 full runs on user-combo-10x at matched wall-clock. Keep if >= stack + 3 with exact repeats <=6% (the existing memory-v2 keep rule). KILL at < stack - 3.

### Stack A: perception facts for keyboard/mixed games (build ~5-6 days incl. Stage 0)

Features:

1. G1-core avatar + view_shift.
2. G2 colour-only lattice, or the Tier-4 fallback (crop with ruler + bbox/center/h/w on segmentation nodes) if the new estimator fails Stage 0.
3. bbox on segmentation nodes (needed by 1 and 2; also removes the 6,791 bbox rebuilds).
4. G3-lite surprise line at the existing stop (uses G1's avatar if present).
5. ARC3_AUTO_FRAME_DIFF narrowed to moves of >2 px (built, unrun), or the G9 watch if it was not shipped in B.

- **Shared Stage 0:** one replay over the 7 runs runs the G1 precision/counterfactual, G2 acceptance and G3 false-surprise gates (via G13-lite).
- **Screen:** 25-min panel on 18 keyboard/mixed games:
  - code reads of `avatar`/`view_shift`/`cells`: KILL if <=6% (STATE_TRACKER) / <10% (frame_diff floor)
  - position-wrapper defs per game: -50%
  - stale-state halted batches per 100 actions: down
- **Then:** 2 full runs. Keep if >= stack + 3 and the keyboard/mixed subset gains >= +4 at matched time with click games flat.
- **Attribution:** give each feature its own ARC3_* flag so a later ablation can attribute.

### Stack C: stall response, evidence first (build ~3 days; only if the gates pass)

Features:

1. G13-lite offline signals / `make screen`.
2. Stall-signal logging into events.jsonl / PrioritySnapshot with no prompt change (the surviving part of G7, 0 tokens, byte-identical).
3. G5-lite model-authored probe, gated as in rank 7.
4. G8-reduced trim card, if Stage 0 passes and Stack B's solved-levels run is >= stack.

- **Run C only if:**
  - the G5 Stage 0 gates pass (trips on later-cleared levels <=40%, single-step decisive divergence >=50%), and
  - G7's replayed trip rule gives <=2 false trips per run with >=5 never-cleared hits and >=20 on-level actions at trip time.
- Otherwise C reduces to items 1-2 (logging only).

## 3. Rejected (do not re-propose without new evidence)

| system | killing facts | what survives elsewhere |
|---|---|---|
| G4 expect-and-halt (REQUIRED `expect` + `hyp`, registry, per-action halt) | See G4 note 1 below | Diff-only 'actually' line (= ARC3_AUTO_FRAME_DIFF, Stack A item 5). The registry-in-trim-card idea is untested but rides on G8 if anywhere. |
| G6 experiment log (as a new arm) | See G6 note 2 below | The delta patch inside Stack B item 3. |
| G7 harness-forced restart with facts kept | See G7 note 3 below | Signal logging (Stack C item 2). A model-callable goto on level-path #66 only if a replayed trip rule passes. |
| G11 inspect-budget yield | See G11 note 4 below | The trigger moves to G5-lite. A true `gate.handover()` variant is a scheduler arm capped at +1.1-1.9. |
| G12 level-start affordance sweep | See G12 note 5 below | A zero-action per-key table from the model's own transitions = memory-v2 action-effects carried across levels (Stack B item 3). |
| Sub-parts cut from survivors | See note 6 below | - |

**1. G4 expect-and-halt.**

- The near-equivalent ran twice on this exact checkpoint: hypothesis-loop v1 bec652f / v2 fd15523.
- It had a fixed grammar, was graded by numpy against the board, and filed refutations to the ledger.
- Results:
  - fill 93% (14/15); 43% CONFIRMED
  - verdict line referenced in 5/14 reasoning blocks; ledgered refutations in 0/27
  - one refuted belief re-predicted the very next turn
  - levels tied (ar25 1v1, sk48 0v0) or lost (b1 ar25 1 vs 2)
  - ar25 acting turns 4 vs 10 (16-min probe spiral after a WRONG)
  - completion tokens per request +22-33%
- A REQUIRED parameter buys <=7% more fill.
- The per-action halt duplicates the stale-state stop, and a single end-state expectation is false after action 1 of a batch.

**2. G6 experiment log.** Already built on arm/memory-v2 (#64, unrun):

- ARC3_MEM_SEEN covers board seen before, outcome per action, and Untried.
- ARC3_MEM_EFFECTS covers outcome classes including 'anim only'.
- hazard_line carries attempt length (memory_facts.py:51, 818, 883-884, 914).
- Replay: fires on 13.8% of turns, ~900 tokens per game.
- Exact repeats are 9% of actions; all action inefficiency is worth 2.63 points (skarin).

**3. G7 forced restart.**

- The proposal's own Stage 1(a) gate fails on the two 10x runs:
  - deliberate RESET at >64k spend: 3/15 cleared within 20k tokens (0.20; 0.25 censored-excluded) vs continuation 27/140 = 0.193 → 1.04-1.30x, below the 1.5x bar
  - a systems-cost skeptic got 5/11 = 1.9x on v1 alone, but those were record-and-replay resets with a plan in hand
- Heavy resetters never cleared: su15 L4 x3, wa30 L2 x5, lf52 L2 x4.
- Draining context to ledger + frame = fresh-eyes reset (didn't work).
- The +12.08 ceiling is imputed. At the hidden ratio it is +0.9..+2.1 imputed and -2..-6 observed.
- Past 64k, levels clear about half the time (12 cleared / 9 never on v1), so false trips run ~50%.

**4. G11 inspect-budget yield.**

- A yield does not release the slot. `_maybe_handover` returns unless `_context_was_trimmed` (tool_agent.py:6372-6382); solver.py:719-728 continues the same game.
- LOCAL_ANALYZER_YIELD_TOKENS=2048 already yields after 1-2 calls:
  - in 97/117 streaks of >=4 the model had already had a yield before its 4th call
  - in 66/117 the 4th call ended in a yield anyway
  - 428 yields sit inside the streaks
- The remaining fact line duplicates 'Your last python snippet executed NO actions', which the model received 428 times in these streaks without acting.
- Resume-prompt variants: 'no consistent gain' (Franzen s6).

**5. G12 level-start affordance sweep.**

- Scoring is quadratic per level, (base/actions)^2, weighted by level index.
- Re-scored stack runs:
  - +k probes per cleared level: v1 -1.19/-1.22, v2 -1.34/-1.39, control -0.92
  - +10 per level (probe + UNDO): -4.25 / -3.97
  - L1 only: -0.01 to -0.14
- Total actions per run are 5,278-6,451, not 13k.
- UNDO is valid in 6/25 games, of which only 4 are keyboard games. 7 games are click-only and unprobed.
- First 5 actions already cover all keys in only 1-8% of keyboard levels, yet L2 failures are 'not startup failures'.
- The '12.5% humans reset' AutumnBench stat was already struck in research-sweep.

**6. Sub-parts cut from survivors.**

- G1: occupancy/`.bfs`, `entities`/`ego` views, ARC3_FIRST_LOOK ('known death' is vacuous at level 1), `events` and `blocked_edges_learned` in the result dict (they blow the 3,072-token output budget in 15-24-action batches).
- G2: boundary-autocorrelation estimator.
- G3: transition DSL with preconditions and a new halt (8.5-10.5 days; Sergei: exact-on-observed simulators no better than nothing-changes; Makarov 83% goal judgement).
- G5: harness EIG chooser.
- G8: sandbox globals, free-text per-level ask (0 reply text in 340 tool-calling replies), level-start placement, per-action counters.
- G9: lib.test, macro, snapshot diff, dependency inlining.
- G10: auto-restore re-exec, retry after actions.
- G13: epsilon handovers, live shadow priorities.

## 4. Cross-cutting facts these verdicts rest on

- Channel ranking on this Qwen, by observed use:
  1. REQUIRED tool parameter (89-93% fill, but fill is not score: hypothesis-loop and thinking-digest v2 were both tied or unscored)
  2. tool-result dict fields read in code (gameplay_changed, animation, stop_reason, last_action_call_result)
  3. turn and opener fact lines quoted the same turn (NO-OP, stopped before, GAME OVER)
  4. advertised sandbox helpers (segmentation 25%, frame_diff 10.5% of calls)
  5. harness state blocks in the user turn (STATE_TRACKER 6%)
  6. unadvertised globals (death_ledger 0, register_predictor 0/2,762)
- The measured stall shape is over-inspection, not action starvation: 34% of generated tokens sit in >=4-call inspect-only streaks, and 40.7% of tokens go to non-acting python calls. Every system that inserts an unexplained event risks more inspection (bp35: 6 inspect-only turns after one death).
- Goal judgement, not dynamics, dominates stuck forks (Makarov 83%; T4, snippet only). Perception systems (G1/G2/G3) only address the wrong-map subset (g50t clearly; bp35 partly, via camera offset).
- Power: two runs resolve only ~+6-10. Every item here is <=+2 alone, so stacks are the unit, judged on mechanism metrics in the panel and on score only across >=2 (ideally ~6) paired runs.

## 5. What this does not cover

- No new GPU evidence. Every number is from existing runs, replays and transcripts (mostly v4-user-combo-10x v1/v2 and 7 full-run histories), plus skeptic measurements on those logs. Only 4 games were read as full transcripts (tr87, bp35, g50t, sc25), so per-game failure shares generalise from a small sample.
- p and E in the ranking are judgement calls built from skeptic ranges, not fitted. The score column orders the list; it is not a forecast. Near-free items (G9/G10) rank high on the ratio but are individually undetectable.
- Hidden-set transfer is unmodelled. 110 games / 10 slots / 532 min may change click/keyboard mix, lattice prevalence and level counts. G1/G2/G3 pay off only on keyboard/mixed tile games, whose hidden share is unknown.
- Interactions inside a stack (e.g. Stack B's seen-state lines plus G9 watch plus G1 tail lines all growing the tail) are not costed jointly. The eviction guard and tool-output-budget truncation need a combined check.
- Two skeptic numbers conflict and were not re-checked: the G7 RESET ratio (1.04-1.30x across both runs vs 1.9x on v1 alone), and the G9 one-argument-function counts (used both to support and to restrict auto-watch).
- Frontier-model evidence (Retrodict, BED-LLM, Battleship, Go-Explore, DreamCoder, Voyager) was treated as inspiration only, per the Qwen-local rule. Makarov's 83% is from a discussion snippet whose page body was not fetchable.
- Throughput, quantization, scheduling and prompt-advice levers were out of scope (fact 3, fact 4, and the advice dead ends). Scheduling appears only where a candidate system drifted into it (G11, G13).
- arms.md is not updated by this report. The 13 systems, their verdicts and the rejected list still need entries under Not yet implemented / Didn't work.
