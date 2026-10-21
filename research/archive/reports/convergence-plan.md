# Convergence plan: lab variants, rounds and approvals

2026-10-11. This answers the user's request: "set up the fix up this by testing different versions of the prompts + other systems that help the model converge onto the right path."

The problem comes from the memv2 reasoning audit (`research/reports/memv2-reasoning-audit.md` on main). The model states a correct idea and then drops it: 143 times in 25 games, and the dropped idea was verified right in 137. What triggers the reversals:
- a misread crop or tool output, about 36%;
- untested doubt, about 22%;
- a rule lost when the context was trimmed, about 14%;
- its own buggy search saying "impossible", about 10%;
- a confounded test, about 10%.

Related failures:
- 29–36 replies per run say they will record a note and don't;
- the code doesn't match the reasoning (21% of 279 mismatches);
- the model states a cheap deciding test and then runs something else;
- 31 replies hit the 12,288-token cap;
- loops with no cap;
- wrong budget models;
- the board before a winning move read as the win frame.

## 0. What the user approves

| Item | Amount | When |
|---|---|---|
| OpenRouter spend | At most **$1.00** of the ~$4 left after round 1, enforced by the ledger's `--budget`. At least $3 stays in reserve. **No top-up is requested now.** | Round 2, 10-12 |
| Kaggle GPU for the lab | About **15 GPU-h** in 7 sessions: L0 0.5, L1 2.5, L2 2.5, L3 2.75, L4 2.5, L5 2.5, RO 2 | 10-12 → 10-19 |
| Kaggle GPU for confirmation | **3 paired 121-min runs ≈ 14 GPU-h**. Another 14 GPU-h only if the first 3 pairs are inconclusive. | 10-20 → 10-30 |
| A private Kaggle dataset | `mellamoa/state-lab-capsules`: about 50–100 MB of our own logged requests, shared with snowiers | 10-12 |
| Bars fixed in advance | Gate A (a variant enters the stack) and Gate B (claim and keep), §5. Signed off before L1, because stopping rules must not move after data arrives. | before 10-13 |
| Decided later, only if their condition holds | (a) A **$10 OpenRouter top-up**, asked only if the L1 calibration passes (§4.3). (b) One extra reflective (GEPA) generation, 2.5 GPU-h, only if fewer than 2 fixes survive L3. | after L1 / after L3 |

The lab displaces about 6–7 full runs of quota. Both accounts together have about 11 GPU-h a day, roughly 240 GPU-h until 11-02, which the other arms share.

## 1. What the plan rests on

1. **The verdict model has to be our W4A16 build.** OpenRouter serves Alibaba's Qwen3.8 Flash at unknown precision.
   - In the precursor it reasoned 12–17× less at one state and 3–5× more at another, and it read a wall colour differently.
   - Proxies have failed before: the D′ simulator said +1.25 and the real run came out −7.5.
   - So OpenRouter is used to debug edits and parsing, and for the small thinking-off gates.
   - It becomes a screen only if L1 shows it tracks W4A16 (§4.3).
2. **What our model uses is a harness-computed fact at the point of use.** Lessons from our runs:
   - Free-text advice is ignored: the thinking guide drew 0 of 28 lines, the hats appeared in 2 of 2,643 messages, and rellik13's advice blocks went from 6.42 to 4.16.
   - Restructuring the prompt cost 15.8 points.
   - Forced formats lost points.
   - Prompt versions that are advice phrased as facts (C9, C10, C11) therefore rank last and get W4A16 cells only.
3. **Mine states from the base the fixes will ship on**, which is stack-2h/stack-3h, on top of user-combo-10x.
   - mem-v2-10x (43.81) is a Didn't-work config. Its memory section is different, and its colour tallies and `how:` lines don't exist in the shipping base.
   - mem-v2 states are kept because they are the audited ground truth. user-combo-10x v1 and the first stack-2h run supply states in the shipping format.
4. **"Misread" is four different problems:**
   - a row read by hand (C2);
   - a filter that hid the evidence (`ARC3_BEFORE_AFTER`, already in stack-3h);
   - a parser bug (nothing proposed);
   - the board before the win (round 1's `win_snapshot`).

   Each fix is scored only on its own kind of state.
5. **A single call on a frozen state underestimates effects that build up over a game:** notes written as a habit, rules that survive several trims, follow-through after a guard. Those fixes also need branch rollouts (§4.8) before full runs.
6. **No thinking-off on the full conversation.** Side calls that reuse the conversation close the thought with an assistant pre-fill (`<think>\n\n</think>\n\n`, `continue_final_message`), the path `ARC3_DEBRIEF_FALLBACK` already uses on stack-3h. Only self-contained side prompts (C5's classifier) may set `enable_thinking: false`.

## 2. Lab extensions to build first (branch `tooling/state-lab`, one commit each)

The round-1 lab (`analysis/state_lab/`) handles tail edits, two exec guards and OpenRouter sampling. Before any new variant it needs the following:

| # | Extension | Spec |
|---|---|---|
| E1 | **New cell types** in `variants.py` / `run.py` | **`system`:** (old, new) replacements on `messages[0]`. It refuses if any `old` doesn't occur exactly once, and sets `breaks_prefix=True`.<br>**`all_messages`:** a string replacement in every message, with `breaks_prefix=True`.<br>**`tool_rewrite`:** parse the newest tool result's JSON, transform named fields (`stdout`), and re-serialise with the harness's own serialiser. It refuses unless the unedited round trip is byte-identical.<br>**`two_step`:** after `execute()` under a policy, if a guard with `followup: true` fired, sample once more on `payload.messages + [assistant: content, reasoning_content, tool_calls of the sample] + [tool: the refusal JSON execute() produced]`. This is exactly what the harness sends next, because a refused call doesn't end the turn (`tool_agent.py` breaks only on `step_executed`). Execute again and grade the final outcome.<br>**`side_call`:** build a side request from the payload, sample it with the cell's parameters, post-process, then edit the main payload with the result.<br>**`continuation`:** take an existing sample whose reasoning exceeds B tokens and send `payload.messages + [assistant: "<think>\n" + reasoning[:B tokens] + CLOSE + "\n</think>\n\n"]` with `continue_final_message: true`, `add_generation_prompt: false`, `max_tokens` 2,048. |
| E2 | **Run flags per source run** | Replace the hard-coded `RUN_ENV` with the env of the run that produced the state: the template plus `kaggle/runs.json`'s env. Use one worker process per source run, because flags are read at import time. A state passes only if `validate.py` re-executes the logged reply byte-identically under that env. |
| E3 | **SGLang backend** for `sample.py` | `--backend sglang --base-url http://127.0.0.1:<port>/v1`.<br>Send the logged payload verbatim, **including `chat_template_kwargs`** (it is our server), with the run's sampling: T 0.65, top_p 0.95, top_k 20, `max_tokens` 12,288, and the logged `reasoning_effort` if it was set.<br>Parse the reply with the harness's own response and tool-call parser (`ARC3_TOOLCALL_TEXT_FALLBACK` on).<br>Send a per-request `seed` only if the L0 seed check passes. |
| E4 | **State capsules** | `python -m analysis.state_lab.state capsule <run> <game> <request>` writes one JSON per state:<br>- the run id, game id and engine version;<br>- the request payload as sent;<br>- the replay batches with the auto-reset points;<br>- the retained-function sources;<br>- the frozen clock;<br>- the verification hashes (board, image, step line, function names).<br>`rebuild_from_capsule()` must reproduce every hash on Kaggle; a state that fails is dropped and logged. |
| E5 | **0/1/2 grader** in `grade.py` | **0, damage:**<br>- RESET executed on a state marked `winnable: true`;<br>- GAME OVER;<br>- the state's pre-registered reversal action prefix executed;<br>- `finish_reason == length` with no tool call;<br>- in a `two_step` cell, the follow-up re-sends the refused action and it is one of the above.<br>**2, success:** the state's pre-registered success checks, using the round-1 check vocabulary.<br>**1:** anything else.<br>Primary metric: success rate. Harm metric: damage rate. |
| E6 | **Racing controller** | `run.py --race race.json` applies §4.5 after every batch of 4 samples per cell, and writes `race.jsonl`. |
| E7 | **Rollout mode** | `python -m analysis.state_lab.rollout`: restore a capsule into a real `ToolAgent` plus `_HarnessGameSession`, with the payload's messages as history, then run the real solver loop with the variant's flags. Stop on a level clear, a GAME OVER limit or 25 min. Log cleared, actions, tokens and guard firings. Retained functions are restored; other sandbox globals are not logged, so a rollout can diverge from the original. |
| E8 | **Lab notebook kind** | `python kaggle/build_notebook.py --lab <session.json> [--push]` (§4.3). |

Tests: extend `analysis/state_lab/tests/` with a fake-sampler test per cell type and a capsule round-trip test. Document every new piece in `analysis/state_lab/README.md` and `kaggle/README.md` in the same PR.

## 3. Catalogue

### 3.1 Which candidate targets which trigger

| Trigger (audit share) | Candidates |
|---|---|
| Misread own crop or tool output (36%) | C2 row runs (hand reads); `win_snapshot` (round 1, pre-win board); `BEFORE_AFTER` is already in the stack (filters) |
| Untested doubt (22%) | C8 reversal fact; C9 hyp_fact; C3 open-test lines |
| Rule lost at a trim (14%) | C1 trim harvest; C7 mem_wording |
| Own search says "impossible" (10%) | C11 search_check; C8 when it negates a stated claim |
| Confounded test (10%) | C4 no-op scoping |
| Announced a note, wrote none (~29×/run) | C7 note receipt |
| Code doesn't match reasoning (21%), plan drift (11%) | C5 plan↔code hold; C10 deciding_test |
| Cheap deciding test stated, not run | C3 open tests; C10 |
| Premature RESET | C3 (round-1 guard + open tests; resample vote) |
| Reply cut at 12,288 tokens (31) | C6 thinking budget |
| Loops with no cap; wrong budget model | `loop_guard`, `budget_time` (round 1) |

### 3.2 Shared definitions

These are used by several specs. Regexes are Python, case-insensitive unless marked.

```
HEDGE   = r"\b(maybe|might|perhaps|probably|possibly|could be|unclear|not sure|seems?|guess|wonder)\b|\?"
IMPOSS  = r"\b(impossible|unsolvable|can'?t|cannot|dead[ -]end|no solution|not possible|stuck)\b"
STRICT  = r"✓|\b(confirmed|verified|so the rule is|this proves|definitely)\b"
NEG     = r"\b(not|no longer|doesn'?t|isn'?t|wasn'?t|wrong|instead|actually|rather than|impossible|can'?t|cannot)\b"
KEYS    = r"\b(UP|DOWN|LEFT|RIGHT|SPACE|UNDO|ACTION[1-7])\b"          # case-sensitive
CLICK   = r"\(\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)"
TAG_RE  = r"<(world_model|hypothesis|level_note)\b"
ANNOUNCE_RE: as in convergence/prompt_variants.py (record/save/note ... memory|world model|notes; "Let me record/note/save")
ROW_RE  = r"(?<![A-Za-z])[WwgGcBMPRbSYOrNp]{6,64}(?![A-Za-z])"       # case-sensitive; ARC_COLOR_LEGEND symbols
```

- **Steps.** The model never sees call numbers. Every harness line therefore labels earlier events by the **step** shown in that call's opener ("Current state: step N, level L").
- **Requests.** The audit's "call N" is request N − 1.
- **Exact strings.** All prompt `old` strings below were checked against `prompts.py` and `solver.py` on main and `origin/arm/stack-3h`. The full text of every prompt-version edit is in `convergence/prompt_variants.py`; copy it into the lab as part of E1.

### 3.3 Fix variants (ranked)

Each entry gives:
- the flag it would ship as;
- what the harness does, with the exact text;
- the lab cell;
- the states;
- the grader;
- the compute cost;
- its own gate, the step before the race.

Every flag is off by default. "Bank" means states mined by §4.4's detectors.

---

#### C1. Pre-trim verbatim harvest: `ARC3_TRIM_HARVEST=1`

**Harness behaviour.**
- **When it fires.** Once per trim, at the request where the trimmer is about to drop turns, before the trim and before the slot handover. It must pass the eviction guard (#61). It is skipped when the game has less than 90 s left or when history plus 600 tokens does not fit the window.
- **The side request.**
  - Messages: the untrimmed history exactly as the next main request would carry it before trimming (so the prefix is cached), plus one user message:
    ```
    Older turns of this conversation are about to be dropped. Copy, word for word, up to 8 sentences you wrote above that state a confirmed rule, the goal, a step size, a step budget or a cause of death, one per line. Copy only; write nothing else.
    ```
  - The assistant turn is pre-filled with `<think>\n\n</think>\n\n` (`continue_final_message`), using the `ARC3_DEBRIEF_FALLBACK` helper. `max_tokens` 400, temperature 0.
- **Checking each returned line.**
  - Strip bullets and quotes; the line must be at least 30 characters.
  - It must be a whitespace-normalised exact substring of an assistant `reasoning_content` or `content` in this conversation.
  - Drop lines that match `HEDGE` or `IMPOSS`.
  - Dedupe against lines already kept (normalised equality, or token Jaccard ≥ 0.8).
  - Keep at most 8 new lines and 1,200 characters.
- **Store.**
  - Per level, at most 12 lines; the oldest go first.
  - Each line is labelled by the step of the call it was copied from.
- **Rendering.**
  - Stack base (`ARC3_MEM_EVERY_CALL`): inside `<this_level>`, under the header:
    ```
    Your own words from turns that were dropped (copied by the harness):
    - (step 37) UP moves the block 3 cells.
    ```
  - On a level commit, the lines move into that level's `<previous_levels>` entry after the recap, as `Your words: ...`.
  - user-combo base (`ARC3_LEDGER_ON_TRIM`): at the end of the ledger head block.

**Lab.**
- **Stage 1, the harvest-quality gate.** A `side_call` cell at the trim before each trim-loss reversal. The candidate points come from `ana/retrieve.py`: ka59 c39, m0r0 c157, tu93 c67, wa30 c120, cd82 c86, g50t c103, lf52 c153, su15 c100, bp35 c32, sp80 c109 and r11l c54. s5i5 is excluded because no rule sentence exists. Points in held-out games move out, which leaves about 9.
  - The side request is built from the last pre-trim request, plus its logged reply and its tool result.
  - Run it on OpenRouter with reasoning off (R2, about $0.3), then repeat on W4A16 with the closed thought (L2, 9 × N=2).
  - Measures:
    - recall of the lost rule within the 8 lines, with the rule regex pre-registered from `games.json` `true_rules`;
    - the verbatim rate before filtering;
    - the wrong-claim rate after filtering (a blind Claude judge against `true_rules`).
  - **Gate:** the rule is quoted at ≥ 5 of 9 points, ≥ 80% of lines are verbatim, and ≤ 20% of kept lines are wrong.
- **Stage 2, the race.** At each reversal request, insert the W4A16-verified quotes.
  - On stack-format states this is a tail edit (`<this_level>`).
  - On mem-v2 and user-combo states the quotes go into the memory head block, which breaks the prefix there.
  - N per the race. Success: the reasoning uses the rule (state regex) and the executed actions don't contradict it.
- **Stage 3:** branch rollouts (§4.8), because the real effect is a rule surviving several trims.

**Cost.** About 140 trims per run × (cached prefix + ~110 prompt tokens + ≤ 400 decode) ≈ 56k decode, plus at most 300 cached tokens per later request on the level. That is about 1% of compute.

**Risk.** It can cement a wrong "confirmed" claim: cd82's "7 poses" cost about 14 actions, and ft09's "M2 inert ✓" was a misread. Hence the `IMPOSS` filter and the rollout check.

---

#### C2. Indexed runs on printed board rows: `ARC3_ROW_RUNS=1`

**Harness behaviour.** The post-processor runs on the python call's `stdout`, after the harness's echo lines are added and after the tool-output truncation.
- **Which lines.** On each line not written by the harness (lines starting `[action]` and the like are skipped), take the longest `ROW_RE` match.
  - Annotate it only if it has 2–16 runs.
  - At most 24 annotated lines per result.
  - Annotate only while the result stays under `LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS` (3,072), so the annotation never causes extra truncation.
- **The annotation**, appended to the line after two spaces:
  - If the string occurs exactly once in `current_frame.ascii` (all rows, all offsets), give absolute columns:
    `BpppppppppOB  [row 22: B14 p15-23 O24 B25]`
  - Otherwise give 0-based positions within the printed string:
    `BpppppppppOB  [runs: B0 p1-9 O10 B11]`
  - A run is written `{symbol}{a}` when a == b, else `{symbol}{a}-{b}`.

**Lab.** A `tool_rewrite` of the newest tool result's `stdout`.
- **Target states (hand reads):** cd82 r29, ar25 r42, tu93 r95, ka59 r81 and lp85 r62 (r11l r16 if r11l is not held out), plus bank states from the detector "the last tool result printed ROW_RE rows and the next reasoning cites a column, width or count".
- **Comparison cells on the same states:**
  - `crop_numbers`: a `system` cell, `CROPS_OLD → CROPS_NEW` in `prompt_variants.py`. This is the advice form.
  - `before_after`: a `tool_rewrite` whose new tool result is re-rendered with `ARC3_BEFORE_AFTER=1`. Rebuild the state at request k−1, re-execute reply k−1's logged code with the flag on, and substitute that tool result. This applies only on mem-v2 and user-combo states whose previous code executed actions; on stack states it is already on.
  - If round 1's `crops` variant already annotates the model's own prints, keep it as the comparison and test C2 only as the always-on version.
- **Grader.** Success = the reasoning states the pre-registered correct number (regex) and not the wrong one. Engine outcome and damage as usual.

**Offline gate (free).** Run the annotator over all 2,719 distinct logged tool results (468 of them have board-like rows).
- Rows annotated, and tokens added per run: must be < 1.5% of prefill.
- The unique-match share.
- False annotations on HUD or legend strings, in 50 drawn by hand: ≤ 10%.
- Coverage: for each hand-read misread in the audit, does the annotation carry the correct number? It must for ≥ 4 of 6.

**Cost.** About 150k prefill tokens per run (about 1%). Tail only, so the cache is safe.

---

#### C3. RESET decision family: `ARC3_RESET_CONFIRM=1` (A) and `ARC3_RESET_VOTE=2` (B)

**A: guard plus open tests.** This extends round 1's `untried_reset_guard`. When shipped, it adds lines to the existing `EXPOSE_RESET` refusal path (`reset_no_actions`, #134); it is not a second guard.
- **When it refuses.** The first RESET requested on a board is refused, and nothing is spent, when the level has had at least 1 action since its start or last RESET and either Untried is non-empty or an open test exists. A RESET sent again executes.
- **Refusal text.** This is round 1's text with one added sentence, so the only new thing tested is the quoted tests:
  ```
  RESET was NOT executed and no action was spent: the first RESET of a snippet asks for confirmation. From this exact board, actions never tried: {untried}. Tests you proposed on this level and never ran: (step 57) "if it fails, try rot 0"; (step 61) "...". Issue RESET again to restart the level.
  ```
  The tests sentence is left out when there are none.
- **Open-test extraction.**
  - Look at sentences in the assistant reasoning of this level that are visible in the payload, plus the current reply, that match:
    ```
    TEST_RE = r"\b(?:let me|let's|I(?:'ll| will)|next,? I(?:'ll| will)|if (?:it|that|this) (?:fails|doesn't work),?\s*(?:I(?:'ll| will)\s*)?)(?:test|try|check|press|click|run)\b"
    ```
  - Keep only sentences that name an action (`KEYS`, or `CLICK` coordinates) other than RESET.
  - A test counts as **run** if a later executed action in this level matches one of its keys, or is a click within Chebyshev distance 2 of a named coordinate.
  - Quote the 3 most recent tests not yet run, at most 160 characters each, labelled with their step.

**B: resample vote.**
- When a reply's first `action()` batch contains RESET (found by dry-running the code with `action()` stubbed to record and stop), draw 2 more samples of the same request (the prefix is cached) and dry-run each the same way.
- If at least 1 of the 2 also starts with RESET, the original reply runs.
- Otherwise the first draft without RESET replaces the reply: its reasoning, code and tool call go into history, and it runs.
- Every vote is logged.

**Lab.**
- **A** is a `two_step` cell on the pre-RESET states, about 9:
  - lf52 r167, m0r0 r184, cn04 r73, the ka59 L2 RESET, 4 × sc25 (r90, r95 and two from the bank), and su15 r47;
  - plus a positive control, wa30 L3, where the step budget forces the RESET;
  - plus counter-states from the bank: a RESET with nothing untried, followed by a clear within 30 actions.
  - **Pass:** at least 30% relative fewer executed RESETs on the wrong-RESET states; RESET kept in at least 60% of samples on the forced state; success not lower.
- **B** is estimated for free first.
  - Per state, p = the share of control samples whose first batch has RESET. Then P(RESET runs under the vote) = p·(1 − (1 − p)²).
  - Also bootstrap it, using the control samples themselves as the drafts.
  - It goes to a live cell only if the estimate shows at least 30% relative fewer RESETs on at least 3 states.

**Cost.**
- A: about 20 follow-up calls per run, about 80k decode.
- B: 2 drafts × about 20–26 RESET replies × ~5k tokens ≈ 0.2–0.26M decode, about 4–5%. B must clear a bigger effect to pay for that.

---

#### C4. No-op scoping: `ARC3_NOOP_SCOPED=1` (a) and `ARC3_NOOP_CONTEXT=1` (b)

**(a) Wording.** Replace these strings in `solver.py` `_action_verdict` (line 306/308 on stack-3h) and in `tool_agent.py` (lines 4641/4643/6393):
- `NO-OP (executed, no change inside the board area)` → `NO-OP from this board (executed, no change inside the board area)`
- `NO-OP (executed, changed nothing)` → `NO-OP from this board (executed, changed nothing)`
- `executed but produced no change inside the board area.` → `executed but produced no change inside the board area from the board it was taken on.`

Lab: an `all_messages` cell, which breaks the prefix.

**(b) A fact computed from this level's record.**
- **When it fires.** When a call's results include an executed no-op (`gameplay_changed` False) of key K, at most 2 distinct keys per call.
- **What it counts.** Over this level's earlier history entries:
  - changed(K) = how many distinct board interiors K changed the board from;
  - noop(K) = how many it did nothing from.
- **The key for clicks.** For MOUSE, K is the segmentation component under the clicked cell in the before frame, as (shape_hash, colour), falling back to the exact cell.
- **Output.** Only if changed(K) ≥ 1, add a key `"noop_context"` to the tool result:
  ```
  LEFT changed the board from 4 other boards on this level (last: step 31) and was a no-op from 2.
  A click on this red object changed the board from 3 other boards on this level (last: step 18) and was a no-op from 1.
  ```

Lab: a tail edit with `needs_state`, at the request right after the no-op call.

**States.**
- Targets: cd82 r14, r21, r30–32, r59, r85, r88; ka59 r46; dc22 r15; m0r0 r156 and r184; lf52 r167; sp80 r21 (if not held out).
- Counter-states from the bank: a no-op on a key that works elsewhere, where the model correctly moved on and the level was later cleared without K from that board (ls20, tu93, ka59 wall bumps).

**Grader.**
- Success = the reply re-tests K from another board, or keeps the movement, orbit or push model (state regex), without asserting the over-general claim (`r"(no|cannot|can't).{0,20}(slot|move|push)|disallowed|disabled"`).
- Damage as usual.

**Cost.** (a) about 0; (b) about 30 tokens per fire, tail only.

---

#### C5. Plan↔code hold: `ARC3_PLAN_CODE_CHECK=1` (behind a classifier gate)

**Harness behaviour.**
- **The dry run.** Dry-run the reply's code in a forked sandbox, with `action()` stubbed to record its first batch and stop. The cap is 10 s; on a timeout the check is skipped.
- **When it fires.** When the first batch has 2 or more actions, contains RESET or UNDO, or `action()` sits inside a for/while loop (by the AST). It is skipped when the reasoning is shorter than 200 characters.
- **The side request.** It is self-contained: one user message, `enable_thinking: false` (allowed, since no prefix is shared), `max_tokens` 80, temperature 0:
  ```
  Here is the end of a game-playing agent's reasoning, the Python code it then wrote, and the exact list of game actions that code is about to send.

  REASONING (end):
  <<<{last 1,200 characters of the reasoning}>>>

  CODE:
  <<<{code}>>>

  ACTIONS ABOUT TO RUN: {actions, e.g. RIGHT, RIGHT, RIGHT, RIGHT}

  Does the action list carry out the plan stated at the end of the reasoning? Answer with JSON only: {"match": true or false, "plan_actions": "...", "code_actions": "..."}
  ```
- **On a mismatch** the call is not run and nothing is spent. It stops with `stop_reason: "plan_mismatch"` (the `KnownNoOpActionError` pattern), and the tool result reads:
  ```
  Not executed (nothing was spent): the plan at the end of your reasoning is {plan_actions} ("{last plan sentence, ≤120 chars}"), but this action() call sends {code_actions}. If the code is what you want, send the same action() call again and it will run.
  ```
  An identical batch sent again in the same turn bypasses the check.

**Gate first.** The classifier is scored on two sets:
- the 90 labelled rows in `games.json` (59 code↔reasoning mismatches plus 31 plan drifts);
- 150 random acting replies with batches of 2 or more as controls.

It runs on OpenRouter with reasoning off (about 300 calls, about $0.12, in R2), then again on W4A16 in L3 (240 calls).

**Gate:** precision ≥ 0.7, recall ≥ 0.4, and false holds ≤ 3% on the controls, on both models.

**Lab.** A `two_step` cell (classifier → hold → follow-up) on lp85, ar25 L1, g50t, ar25 L3, sb26 r19 and tn36 r53, plus bank states. Success = the executed actions match the pre-registered plan.

**Cost.** About 780 side calls per run, about 0.94M uncached prefill (about 72 s of serial prefill per run) and 47k decode. It adds 1–2 s of latency per acting call.

---

#### C6. Thinking budget: `ARC3_THINK_BUDGET=8192` (variant 6144), with `cap_fact` as a cell

**Harness behaviour.** The main request's `max_tokens` becomes B. If `finish_reason == "length"` and there is no tool call:
- **Still thinking** (no `</think>`): send a continuation with `continue_final_message`. The assistant content is `"<think>\n" + reasoning + CLOSE + "\n</think>\n\n"`, `max_tokens` 2,048, where:
  ```
  CLOSE = "\n\nI have thought about this long enough; I will now write the code for my current best plan and check its result."
  ```
- **Reasoning done but the call cut off:** continue from the closed reasoning plus the partial content, `max_tokens` 2,048.

Reuse the `ARC3_DEBRIEF_FALLBACK` helper.

**Lab (W4A16 only).**
- States: the cut-off requests s5i5 r20 and r98, tu93 r89, sk48 r26, sc25 r90, wa30 r126, ka59 r63 and sp80 r107, plus about 4 more cut-offs and 12 replies over 8k tokens, all from logs.
- Cells:
  - control, N=4;
  - `think_8k` and `think_6k`, both `continuation` cells built **from the control samples themselves** (only the samples that passed B need a continuation);
  - `cap_fact`, a `system` cell: `NO_RATION_LINE → + PREFER_TOOL_CALLS_LINE` and `TIME_LIMIT_OLD → + CAP_FACT` from `prompt_variants.py`, N=4.

**Pass:**
- on the cut-off states, the rate of replies that act rises by at least 25 points and success is not lower;
- on the >8k states, success is non-inferior (margin −0.10);
- decode tokens per sample fall by at least 20%.

**Cost.** None added; it saves about 0.3–0.6M tokens per run.

**Risk.** It can cut long correct searches (tu93, sb26), so track actions per cleared level in the rollouts.

---

#### C7. Note wording and note receipt: `ARC3_MEM_WORDING=1` (a) and `ARC3_NOTE_RECEIPT=1` (b)

**(a) Wording, stack base only.** Append one sentence to `MEMORY_LAYOUT_NOTES_LINE`:
- old: `... Your notes appear in \`<this_level>\` while you play the level and move into \`<previous_levels>\` when it is cleared.\n`
- new: `... when it is cleared. They stay when older turns are dropped, so findings from dropped turns are not lost.\n`

On a memory-v2 base, the edits are `MEM_HEAD_OLD → MEM_HEAD_NEW` and `MEM_SCOPE_OLD → MEM_SCOPE_NEW` in `prompt_variants.py`. On user-combo-10x there is no change, because it already has the wording that drew 34–58 notes.

Lab: a `system` cell, but **it is judged in rollouts** (§4.8: true notes per level). A single call is biased toward null when the history holds no tags.

**(b) Receipt.**
- When a reply's reasoning matches `ANNOUNCE_RE` and there is no `TAG_RE` match in its reasoning, content or code, its tool result gets a key `"memory"` with:
  ```
  No note was recorded from this reply: only a line inside a <world_model>, <hypothesis> or <level_note> tag is kept as a note.
  ```
- It fires on about 1.8% of replies in mem-v2. Measure the stack base rate with the detector on stack-2h logs before racing.

Lab: a tail edit at the request after the announcing reply: bp35 r20 and r24, g50t r61 and r72, plus bank states. Success = the next reply writes a tag whose content is true (blind Claude judge against `true_rules`).

**Fallback if (b) fails:** a closed-thought side call that writes at most 3 tags from about 2k characters of the announcing reply. Gate it on the 36 announcement points, about $0.02.

---

#### C8. Reversal fact: `ARC3_REVERSAL_FACT=1` (after a free offline gate)

**Harness behaviour.**
- **Claim store, per level.** Assistant sentences that match `STRICT` and match neither `HEDGE` nor `IMPOSS`, plus C1's quotes. Each is stored with its step and its content tokens: lowercase words of 3 letters or more not in a 100-word stopword list, plus all numbers and `KEYS` names.
- **Detector.** On the current reply's reasoning, find sentences that match `NEG` and share at least 2 content tokens with a stored claim from an earlier call. Keep the one with the most overlap.
- **Output.** Add a key `"reversal"` to this reply's tool result. Version 1 has no hold.
  ```
  At step 41 you wrote "<claim, ≤160 chars>". This reply says "<negation, ≤160 chars>". Since step 41: 0 actions executed, 0 boards not seen before.
  ```
  The counts are computed from history and not judged.

**Offline gate (free).**
- Recall against the 132 located reversal quotes (`ana/located.json`).
- Fires per run over all 25 transcripts.
- Precision on 40 random fires, judged against `true_rules`.
- **Gate:** recall ≥ 30%, at most 1 fire per 10 calls, and the earlier claim right in ≥ 60% of fires.

**Lab.** A tail edit at the request after the reversing reply. For a reversal at the audit's call N, that is request N: ka59 r39, wa30 r120, g50t r100 and r107, tn36 r62, cd82 r86 and ft09 r36. Also should-switch counter-states from the bank: a hedge followed by a plan change, where the new plan cleared within 20 actions.

**Grader.** Success = the reply goes back to the claimed plan, or runs the action that settles it. On should-switch states, success = it keeps the switch.

**Risk.** It can set a wrong earlier claim against a correct correction. Precision decides this one.

---

#### C9. hyp_fact: a `system` cell, with `hyp_advice` only as a conditional control

**Edit.** Insert after `WORLD_MODEL_FREE_ADDENDUM`'s line (`WM_FREE → WM_FREE + HYP_FACT` in `prompt_variants.py`):
```
- The game is deterministic, so a plan you have worked out is refuted only by an action result or a frame that contradicts it. A worry that the plan might be blocked or might fail is not such evidence, and an untested alternative has no more support than the plan it would replace: run the action that settles the worry.
```

**Lab.** On W4A16 only:
- the doubt states ar25 r30, sb26 r8, r11l r29, tu93 r15, vc33 r42 and ft09 r17;
- the bank's should-switch states;
- N per the race.

`hyp_advice`, the length-matched exhortation in the same slot, runs only if hyp_fact passes, to attribute the effect. Round 1's `advice` cell is the general advice control: if `advice` ≈ control on W4A16, demote C9, C10 and C11 further.

---

#### C10. deciding_test: a tail edit of the newest opener

**Edit.** Add this line to the newest turn message's first text part, immediately before `<this_level>` on stack states (since `<this_level>` must stay last) and as the last text line otherwise:
```
When your evidence supports a choice, call `action(actions)` with the sequence it supports. If you have named an action whose result decides between two ideas, run that action next, before anything else; otherwise batch the route you have.
```

When shipped, it goes in every opener, outside the `ARC3_OPENER_DEDUP` static block, at about 45 tokens per opener (about 126k cached prefill tokens per run). The lab tests only the newest opener, which is an approximation.

**States:** sb26 r19, dc22 r167, cn04 r63, cd82 r96, tn36 r53, m0r0 r184 and lf52 r167, plus should-switch states.

**Grader:** success = the first executed action is the pre-registered deciding action, checked automatically in the engine.

---

#### C11. search_check, new sentence only: a `system` cell

**Edit.** Add a new bullet after `FRAME_DIFF_LAST` (`prompts.py:299`):
```
- `transitions` holds every action taken so far with its before and after frames, so a step function or simulator can be replayed on them: one that does not reproduce them is wrong, and is the first thing to fix before doubting the goal or a rule you have already confirmed.
```

`transitions` is built from the full history with no cap (`python_tool_sandbox.py:500–506`). The STEP_VERIFICATION bullet itself is not re-tested, because Franzen measured `ARC3_STEP_VERIFICATION_HINT` as no gain.

**States:** sk48 r68, ar25 r80, s5i5 r57, vc33 r95 and ls20 r11, plus bank states from the detector "`IMPOSS` in the reasoning followed by a plan change or a RESET" (13 of the 19 code RESETs in mem-v2 come with impossibility talk).

**Grader:** success = the code replays its step function or simulator on `transitions`, or tests the doubted rule, without abandoning the goal.

---

### 3.4 Round-1 variants carried forward

| Variant | What happens next |
|---|---|
| `win_snapshot` | L1 calibration at N=4, then raced in L2 on the pre-win misread states (6 games). |
| `untried_reset_guard` | It becomes C3-A's base cell. Its OpenRouter control samples give the free p(RESET) estimate for C3-B. |
| `budget_time` | Raced in L2 on the 4 budget states plus wa30 and sc25 from the bank. |
| `crops` | The comparison cell inside C2. |
| `loop_guard` | Scored **for free**: re-execute every control sample's code with the guard on. A `two_step` follow-up only where it fired. Pass: on the loop states (tr87's 98 LEFTs, s5i5's ~29 blinks), wasted actions fall by at least 50%, with no lost success on states where the model was on track. |
| `memory_layout` | Already in stack-3h's base, so it is not a candidate. Its round-1 result is recorded only. |
| `advice` | The control for "advice is ignored"; it sets C9–C11's priority. |

### 3.5 Dropped or folded

- **`clean`.** Expected null on a cached prefix, and removing Franzen's duplicate lines hurt before (A3 dedup 5.25; hats −15.8). Only the missing `\n` at `prompts.py:102` is kept, as a one-line fix inside whichever prompt arm ships.
- **`facts_first`.** It reorders Franzen's game-over opener with no evidence that position matters, and C3 covers the "Untried" target.
- **`crop_numbers_tool`.** Folded into C2's advice comparison.
- **The full search_check bullet.** Already tested by Franzen (no gain); only C11's new sentence is kept.
- **Announced-note completion.** C7's fallback.
- **The standalone open-test ledger and the RESET quorum.** Folded into C3. The ledger fires only at RESET guards, because "no solution" almost never appears in stdout (1 of 2,755).
- **The claim↔output check.** It adds about 17% prefill and would fail on character counting, which C2 covers.
- **Best-of-k at level start.** Adds 10–30% decode, and a vote amplifies the most common wrong choice (m0r0 L6 gave 3 of 3 identical samples). It can be measured for free from the control samples if needed.
- **Full GEPA.** Replaced by the light reflective loop in §4.6. One full generation only if fewer than 2 fixes survive L3.

## 4. Rounds

### 4.1 Timeline (day 0 = 10-11)

| Round | Dates | Where | What | Samples | Cost |
|---|---|---|---|---|---|
| R1 | running | OpenRouter | Round 1: about 15 states × 8 cells × N≈8. Used to debug graders, get OpenRouter control rates and hints of effects; **not for verdicts on single variants** (each variant has only 2–4 target states, power ≈ 0.46). | ~1,000 | $6 (already approved) |
| R1.5 | 10-11 → 10-12 | Laptop under `nice` | E1–E8; the bank (§4.4); the free analyses (§4.2) | 0 | $0 |
| R2 | 10-12 | OpenRouter | Debug the new cell types; C5 classifier gate; C1 harvest stage 1 | ~350 short calls | **≤ $1.00** |
| L0 | 10-12 | Kaggle | Smoke: the 15-min smoke bench + 50 lab samples + the seed check | 70 | 0.5 GPU-h |
| L1 | 10-13 | Kaggle | Calibration and controls | ~1,050 | 2.5 GPU-h |
| L2 | 10-14 | Kaggle | Race C1–C4 plus the round-1 carry-overs | ~1,200 | 2.5 GPU-h |
| Reflect 1 | 10-14 night | Claude | ≤ 6 new transforms | 0 | $0 |
| L3 | 10-15 | Kaggle | Race C5–C11; reflect-1 minibatch gates | ~1,200 | 2.75 GPU-h |
| Reflect 2 | 10-15 night | Claude | ≤ 6 new transforms | 0 | $0 |
| Build | 10-16 | Laptop | Winners as flags: `arm/<fix>` branches and PRs, then `stack-2h-conv` = stack-2h plus the winners | 0 | $0 |
| L4 | 10-17 | Kaggle | Stacking: all-on plus leave-one-out; reflect-2 minibatch gates | ~1,000 | 2.5 GPU-h |
| L5 | 10-18 | Kaggle | Held-out confirmation, per winner and for the stack | ~1,100 | 2.5 GPU-h |
| RO | 10-19 | Kaggle | Branch rollouts | 52 rollouts | 2 GPU-h |
| F1 | 10-20 → 10-25 | Kaggle | 3 pairs: `stack-2h-conv` vs `stack-2h`, 121 min, same batch | 6 runs | ~14 GPU-h |
| F2 | 10-26 → 10-30 | Kaggle | 3 more pairs **only if** 0.7 < P(Δ>0) < 0.95 | 6 runs | ~14 GPU-h |
| Ship | 10-31 → 11-02 | | If Gate B passes, the fixes go into the 3-hour stack and the submission config | | |

Sessions L0–RO run on **mellamoa** by default (6 h/day; it owns mem-v2-10x and user-combo-10x v1). Each session is staggered at least 20 min from any other push, because two cold starts at once hung both servers. A session can move to snowiers if mellamoa's day is used; the capsule dataset is shared with it.

Throughput assumed: about 12 min of startup, then about 400–620 samples per GPU-hour at 5–8k output tokens, with decode at 900–1,000 tok/s over 14 streams. If L0 measures less, every session shrinks pro rata, in the priority order of §3.3.

### 4.2 R1.5 and R2 (offline, then OpenRouter ≤ $1)

**Free offline work (R1.5):**
1. Copy the 57 seed states with their pre-written criteria (`convergence/lab_states.json`) into `analysis/state_lab/states/bank.json`, converting them to the 0/1/2 schema.
2. Download `mellamoa/v4-user-combo-10x` v1 (56.94) with `kaggle kernels output ... -p kaggle-outputs/v4-user-combo-10x` for mining.
3. Run the detectors (§4.4) over user-combo-10x v1, level-path-10x and mem-v2-10x.
4. Build capsules and validate every state (E2/E4).
5. Validate the graders: the logged Kaggle reply must score 0 or 1 on every reversal state, and the known-correct action must score 2.
6. The C2 offline gate.
7. C8's offline gate.
8. p(RESET) from R1's control samples at `lf52_L5_reset` and the other RESET states, giving C3-B's estimate.
9. Re-execute R1's control samples under `loop_guard`, for free.
10. Replay R1's samples through the racing rule, to see what it would have dropped and when.
11. Re-run `search/power2.py` with R1's real per-state control rates in place of the assumed mixture, and re-check that N=8/16 is enough.

**R2 on OpenRouter, a hard cap of $1.00 via `--budget`:**

| Use | Calls | Est. cost |
|---|---|---|
| Debug each new cell type (system, all_messages, tool_rewrite, two_step, side_call) on 1–2 states, N=2, about 10 variants | ~40 main + follow-ups | ~$0.25 |
| C5 classifier gate (reasoning off, self-contained) | ~300 | ~$0.12 |
| C1 harvest stage 1 (reasoning off, ~9 points × N=2; a cache miss on each first call) | ~18 | ~$0.30 |
| Spare for re-runs after fixes | — | ~$0.30 |

This leaves at least $3 in reserve: smoke tests of reflective transforms while the GPU is busy (about $0.02–0.03 per cell), and emergencies. **No increase is requested.** After L1, if OpenRouter passes the calibration in §4.3, the user may choose a $10 top-up (about 400 more N=8 tail cells) to screen reflective transforms off-GPU. Otherwise it stays a debugging tool.

### 4.3 The Kaggle lab notebook (E8)

- **Build:** `python kaggle/build_notebook.py --lab kaggle/lab/<session>.json --push [--account snowiers]`. The kernel is `<owner>/v4-lab-<session>`, recorded in `runs.json` as `"kind": "lab"`.
- **Server.**
  - The same setup and serving cells as `user-combo-10x`, which is also stack-3h's serve block: MAXREQ 14, CUDAGRAPH_MAXBS 14, MAMBA_CACHE 84, CTX 106496, MEMFRAC 0.96, and the W4A16 model plus the albucino MTP drafter from `template-metadata.json`.
  - The server is therefore byte-identical to the runs the states came from.
- **What is mounted.**
  - `dataset_sources += ["mellamoa/state-lab-capsules"]` (private, version pinned in the session file, shared with snowiers). These are the E4 capsules, built and validated locally: about 400 KB per state, about 50–100 MB for about 150 states.
  - The engines come from `competition_sources` (`arc-prize-2026-arc-agi-3`), the same versions the runs used.
  - Optional, for mining in the notebook: `kernel_sources += ["mellamoa/v4-user-combo-10x"]`, which mounts that kernel's output at `/kaggle/input/v4-user-combo-10x/` (`*_p0_requests.jsonl`, events). Kaggle mounts the **latest** version's output, so the session file records that version number, and only kernels of the same account (or shared ones) can be mounted. The capsule dataset is the default because it pins exactly and works from either account.
- **Lab cell.**
  - It installs the harness at the lab commit with the arms' patch mechanism, and loads the lab code, transforms, graders and race config as a gz+base64 blob, as the bench blob does.
  - An asyncio runner keeps 14 requests in flight, ordered by state. For each (state, cell), sample 1 goes out alone to warm the radix cache, then the rest.
  - Per-sample `seed` = hash(state, i), shared across cells (common random numbers), **only if L0's seed check passed**.
  - Sampled code is executed in a `ProcessPool` of `cpu_count − 1` workers, with the harness's 30-s limit.
  - Follow-ups (`two_step`, `continuation`) queue with priority.
  - Racing is evaluated after every batch.
- **Outputs.**
  - Written to `/kaggle/working/lab/` and flushed per sample, so a timeout keeps the data: `samples.jsonl`, `executions.jsonl`, `grades.jsonl`, `race.jsonl`, `serve.log`, `lab_summary.md`.
  - Sampling stops at the session budget minus 15 min.
  - Fetch them with `kaggle/fetch_version_output.py`, and push logs to main (the research-data rule).
- **L0 smoke.**
  - The existing 15-min smoke bench.
  - 50 lab samples on 5 states, with all cell types exercised.
  - The seed check: 10 prompts × 2 identical requests with the same seed, using common random numbers only if 10 of 10 are identical.
  - Check that the parsed tool calls equal those the harness parser gives on the logged replies.
- **L1, calibration and controls (about 1,050 samples):**
  - (a) R1's matrix on W4A16 at N=4, without its control column: about 15 × 7 × 4 = 420.
  - (b) Controls at N=8 on every train and validation bank state (about 75 states, R1's 15 included): about 600.
  - **Decisions at the end of L1:**
    1. **The OpenRouter kill rule.** OpenRouter stays a screen only if the Spearman correlation of R1's variant effects between the two models is ≥ 0.6 **and** control rates agree within ±0.2 on ≥ 70% of states.
    2. **Sort states** by W4A16 control success: 0.2–0.8 is a "sampling accident", where decoding systems such as C3-B can help; ≤ 0.1 is "systematic", where only information edits and guards can; ≥ 0.9 means the state serves as an on-track or counter-state.
    3. Fix any grader that disagrees with a blind read of 20 samples.
    4. Fix the race list for L2.
- **L2 (about 1,200 samples):**
  - C1 stage-1 recheck on W4A16, then C1 stage 2;
  - C2 with its `crop_numbers` and `before_after` comparison cells;
  - C3-A (`two_step`);
  - C4 (a) and (b);
  - `win_snapshot`, `budget_time`, and the `loop_guard` follow-ups.
- **L3 (about 1,200 samples):**
  - the C5 classifier on W4A16 (240 calls), then C5 `two_step` if its gate passes;
  - C6 (controls N=4 on about 24 states, the continuations, and `cap_fact`, about 0.8 GPU-h because the samples are long);
  - C7(b);
  - C8 if its offline gate passed;
  - C9, C10 and C11;
  - the reflect-1 minibatch gates (6 × 12 = 72).
- **L4 (about 1,000 samples).**
  - Stacking: one all-on cell on every train and validation state, including counter- and on-track states (about 75 × N=8 = 600).
  - One leave-one-out cell per winner, on that winner's relevant states only (about 4 × 20 × N=4 = 320).
  - The reflect-2 minibatch gates (about 72).
- **L5 (about 1,100 samples).** Held-out confirmation (§4.7).

### 4.4 State bank (built in R1.5, refreshed after each new run)

- **Size and split by game.**
  - **Held-out games**, drawn deterministically and pre-registered now: the first 5 games in `sorted(games, key=lambda g: sha256(("heldout-2026-10-11:" + g).encode()).hexdigest())` **that round 1 did not sample**. The order is r11l, tn36, dc22, sp80, sk48, tu93, cn04, ar25, …
  - The other 20 games split into about 16 train and 4 validation games.
  - Targets: about 60 train, 15 validation and 20 held-out states, plus **held-out B**, about 20 fresh states mined from the first stack-2h run's logs in the shipping format.
- **Sources:**
  - the audit's 143 episodes and 279 mismatch rows (`research/data/memv2-reasoning-audit/games.json`);
  - the 57 seed states;
  - the detectors below, run over user-combo-10x v1, level-path-10x and, later, stack-2h.
- **Detectors**, each emitting (run, game, request, trigger class, relevant variants):
  1. RESET executed while `Untried` was non-empty;
  2. code reading `history[k-1]` after `level_completed`;
  3. 20 or more identical actions in one call with no board change;
  4. `finish_reason == length`;
  5. `ANNOUNCE_RE` with no `TAG_RE` in the reply;
  6. `IMPOSS` followed by a plan change or a RESET;
  7. a no-op on a key that changed the board elsewhere on the level;
  8. printed `ROW_RE` rows followed by reasoning that cites a column or count;
  9. `STRICT` claims later negated (C8's detector);
  10. a dry-run batch of 2 or more actions (C5).
- **Cut points:** two per episode, one at the request where the trigger enters and one at the request where the reversal happens. Precursor (e) showed that at cn04's reversal request the idea "does not reliably come back".
- **Counter-states:** at least 4 per fix family where the intervention would be wrong (RESET was right, the switch was right, the crop was read right, the key really was blocked), plus about 6 on-track states (a reply followed by a level clear within 10 actions).
- **Criteria:** pre-registered per state in the 0/1/2 schema, including `winnable` and the reversal action prefix. They are committed before each session; mined states get criteria written by a Claude subagent and are spot-checked by a second agent.

### 4.5 Racing, statistic and stacking

- **Per variant:** only its pre-registered target states (at least 8), plus its counter-states and the on-track states.
  - One control cell per state, shared by every variant.
  - Controls are extended from N=8 to N=16 only on states that are being raced.
- **Statistic:** for success on the target states,
  `z = Σ_s (x_v/n_v − x_c/n_c) / sqrt( Σ_s p̄_s(1−p̄_s)(1/n_v + 1/n_c) )`,
  where p̄_s is the pooled success rate on state s. This is a paired z stratified by state, as `search/race.py` uses.
- **Race:** batches of 4 samples per cell.
  - Drop a variant at n = 4, 8 or 12 if z < −0.5, 0 or 0.6 respectively.
  - Accept early if z > 2.33 at n ≥ 8.
  - At n = 16, keep it if z > 1.28.
  - In simulation (`race.py`, 8 states, mostly-floor control rates) this costs about 106 / 152 / 182 samples for a harmful, null or +1.0-logit variant, against 256 at a fixed N=16. It keeps about 94% of real winners and lets about 10% of nulls through, which is why the held-out stage exists.
- **Harm check, on the counter-states and on-track states pooled:** success may drop by at most 0.10 and damage may rise by at most 0.05, both as one-sided 80% bounds.
- **Stacking:**
  - all-on must beat control on target states (z > 1.28) and be non-inferior on counter- and on-track states (the same margins);
  - drop any component whose leave-one-out cell beats all-on;
  - before predicting the stack's effect, shrink each winner's estimate with a hierarchical Beta-Binomial (the winner's curse).

### 4.6 Reflective-proposal loop (light, inside the sessions already planned)

1. After L2 and after L3, a Claude subagent receives:
   - every "systematic" state, where all cells have success ≤ 0.25;
   - the newest opener and tool result;
   - the last 2 reasoning excerpts;
   - up to 4 failing samples (reasoning ±1,500 characters around the reversal sentence, the code, and the executed tool result);
   - the grader's feedback;
   - the state's `true_rules` note.
2. It proposes **at most 6 transforms**, each a Python function `transform(payload, ctx) -> payload` or a guard policy, written to `analysis/state_lab/transforms/r<k>_<name>.py`, with the states it predicts will flip. **The grammar** allows only:
   - a fact line computed from `ctx.state` (history, frames, transitions, retained functions);
   - a change to a guard's text;
   - a Franzen-style wording edit of ≤ 40 tokens.

   It does not allow advice sentences, forced formats, new required parameters or restructuring.
3. **Lint (CPU):** an AST whitelist of readable attributes. It forbids reading `games.json`, audit files, later request rows and engine internals, and it flags string literals with state-specific coordinates. A second Claude agent reviews each transform for leaked answers. Held-out games are never shown to the proposer.
4. **Minibatch gate** in the next session: 3 of its predicted-flip states × 4 samples, against the shared control. It must gain at least 2 successes out of 12, with no extra damage. Survivors join the race on their own target states, under the same rules.
5. **Calibration:** log each proposer's predicted flips against what happened.
6. **Full GEPA**, with a Pareto pool, merges and several generations, runs only if fewer than 2 fixes survive L3: one generation, about 2.5 GPU-h, which needs the user's go-ahead.

### 4.7 Held-out validation (L5)

- **Per winner:** its held-out target states, 4–6 from the held-out games plus held-out B. Control and variant cells, **fresh samples** at N=16, about 100–190 samples per winner.
- **Stack:** all-on against control on all 20 held-out states, N=16, which is 640 samples, with the control shared with the per-winner cells where the states overlap.
- **Bar:** z > 1.645 on held-out targets, the §4.5 harm check on held-out counter-states, and the predicted effect direction on held-out B, the shipping format. Held-out states are used once; a variant that fails is not re-tuned and re-tested on them.

### 4.8 Branch rollouts (RO)

- **States:** 12–14 decisive ones, for example sp80 L4 (3-action finish), cn04 L2 (7), m0r0 L6 (39), lf52 L5 (54), ka59 L2, wa30, tu93, g50t, plus 4 from stack-2h.
- **Design:** 2 arms (stack-2h flags against stack-2h-conv flags) × 2 repeats, 25 min each, run in 4 waves of 14 streams: 52 rollouts, about 2 GPU-h.
- **Outcome:** the level cleared, the actions and tokens to clear, guard firings, and notes written and how many are true.
- **Bar:** the paired clear rate is not lower; median actions-to-clear are not worse by more than 10%; and for C1 and C7, true harvested quotes or notes appear in at least half of the stack-arm rollouts that reach a trim.

### 4.9 Full runs (F1, F2)

- `stack-2h-conv` (stack-2h's env and serve plus the winners' flags) against `stack-2h` itself, built with `--new-batch` at the same commit. Each pair is pushed together, one on each account, at 121 min.
- After 3 pairs, compare with `make significance BASELINE_SCORE=... CANDIDATE_SCORE=...` and at matched wall-clock, per the score timeline.
- Add 3 pairs if 0.7 < P(Δ>0) < 0.95.

## 5. Bars

### Gate A: a variant enters the 3-hour stack (all must hold)

1. **Lab race on W4A16:** kept by §4.5, with z > 1.28 at n=16 or z > 2.33 early, on at least 8 target states.
2. **No harm:** on counter-states plus on-track states, the success drop is ≤ 0.10 and the damage rise is ≤ 0.05 (one-sided 80% bounds).
3. **Held-out:** z > 1.645 on fresh held-out samples, the harm check passed again, and the effect direction holds on held-out B (the stack-2h format).
4. **Multi-turn fixes** (C1, C3, C7, and any guard whose value depends on the follow-through): the rollout bar in §4.8.
5. **Compute:** estimated from fire rates over the 25 logged games, at most +2% prefill and +3% decode per run. Anything larger (C3-B, C5) must show a held-out effect at least twice the size of the cheaper alternative that targets the same trigger.
6. **Engineering:**
   - a flag that is off by default, a `CONFIGURATION.md` row, and unit tests;
   - `tests/stack_loop_sim.py` shows no prefix breaks beyond the ones the variant declares;
   - the 15-min smoke shows it fires with no errors;
   - its own `arm/<name>` branch and PR, and an `arms.md` entry.

### Gate B: the fixes count as an alpha and stay in the 3-hour stack (the user's rule)

- `stack-2h-conv` beats `stack-2h` in **at least 3 of 3 pairs** (or at least 5 of 6 after F2), **and** `make significance` gives P(Δ>0) ≥ 0.95.
- The score per unit of compute (tokens, wall-clock) is not worse than stack-2h's.
- Otherwise each fix's lab result goes to **Didn't work** or **Inconclusive** in `arms.md`, and the stack ships without it.

## 6. Risks and kill switches

- **The lab doesn't transfer to score:** this is why rollouts and full runs follow the race, and why panels aren't used (r = 0.21 at minute 25).
- **Optional stopping:** the thresholds are fixed in §4.5 before L1 and never changed mid-session.
- **Overfitting:** held-out games, the lint plus a second reviewer, single use of the held-out states, and the proposer's predictions are logged.
- **Graders that reward doing nothing:** counter-states and on-track states, plus the damage metric.
- **A capsule that diverges on Kaggle:** the state is dropped and logged. If more than 10% of states drop, stop and fix before sampling.
- **Kaggle hangs or quota runs short:** one cold start at a time; outputs are written incrementally; sessions shrink in §3.3's priority order.
- **Time:** if L3 ends with no Gate-A candidates, skip L4–RO, record the results in `arms.md`, and give the quota back to the other arms.

## 7. Files

- This plan: `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad/convergence/plan.md`
- Exact prompt-edit strings and regexes: `.../convergence/prompt_variants.py`
- The 57 seed states with criteria: `.../convergence/lab_states.json`
- Measurements: `.../convergence/ana/measurements.txt` and `.../convergence/ana/located.json` (132 located reversal quotes)
- Power, racing and cost models: `.../convergence/search/power.py`, `power2.py`, `race.py` and `cost.py`
- Round-1 lab code: `.../scratchpad/wt/state-lab/analysis/state_lab/` (branch `tooling/state-lab`)
- Lab fidelity on mem-v2-10x: `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs/replays/state-lab/validation-mem-v2-10x.json`
