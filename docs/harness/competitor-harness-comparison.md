# Milestone 2 harnesses vs ours

Question: the three Milestone 2 winners run the same Duck harness, model and
roughly the same serving as we do, yet our score has plateaued while our
inference is faster. What did they change in the *harness* that we have not
tried? Serving is out of scope here.

Source: the comparison thread
[Comparing the Milestone 2 winning solutions](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792)
(Tong Hui Kang, 2026-10-01, all 18 comments read), the three write-ups and their
code, cloned read-only into `~/Documents/arc3/reference-harnesses/`:

| Folder | Source | Commit | Public LB |
|---|---|---|---:|
| `dfranzen/` | [da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution) | `10882e3` | 27.89 |
| `tonghuikang-mirror/kaggle/lordhansolo/` | [tonghuikang/daniel-franzen-arc-agi-3](https://github.com/tonghuikang/daniel-franzen-arc-agi-3) (lordhansolo's source bundle) | `f472820` | 23.84 |
| `sirikilohit/` | [LohitSiriki/arc-agi-3-milestone2-solution](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution) | `af10520` | 22.53 |

Our best is 30.78 (four-waves v4 at `92dece7`).

## Where they stand relative to us

- **dfranzen** is our base. Our `main` already contains his final commit
  `10882e3` (`git log main..franzen/main` is empty), so his harness has nothing
  we lack.
- **sirikilohit** ported *our* harness (sandbox, segmentation, prompts; credited
  as "Wang") at 13.40, then added fixes and memory on top. Their write-up says
  the last jump to 22.53 was FP8 KV plus longer history, with no prompt change.
- **lordhansolo** rewrote the tool agent (1,745 lines vs 2,063 stock) with a
  queryable game record, tested saved code and a stall audit. He states that
  none of these changes was A/B tested.

## What our own A/B runs already say

One rule fits every harness result we and they have:

- **Returning evidence the agent has lost helps.** Our level-up board image:
  +5.77 (p = 0.37). sirikilohit's solved-level memory: "the only text addition
  that helped (+2.8)", which is the public-LB gap between two consecutive single
  submissions (7.14 → 9.96); the same write-up reports one notebook scoring 5.02,
  5.50 and 6.42 on resubmission, so it is one noisy run. Their longer history:
  14.49 → 22.53.
- **Adding generic advice hurts.** Our `early-level-recovery` progress check
  (once, levels 1–2 only, at 32k generated tokens): −11.62 vs its baseline on the
  same 15 games (p = 0.06, worse in 9 of 15); the v2 rerun −6.25 (p = 0.21, worse
  in 10). `simple-event-prompts`: −2.96
  (p = 0.61). sirikilohit: more prompt text 6.42 → 4.16, a level-start advice
  block 13.40 → 9.98.

The ideas below are ranked by that rule. All p values are paired t-tests over
games; with 25 games and one pass, under about 8 points is not detectable.

## Where the score is lost

From the 108k model calls in [`../reasoning/data/hmm_calls.csv.gz`](../reasoning/data/hmm_calls.csv.gz):
a cleared level costs a median 20k generated tokens (p75 40k; 13% above 60k),
but the level a game ends on has already used a median 63k (80k in recent
runs). The plateau is games stuck on one level, so the useful ideas are the ones
that change what the agent knows while stuck.

## Ideas, ranked

| # | Idea | From | We have | Type | Cost |
|---|---|---|---|---|---|
| 1 | Solved-level memory | sirikilohit M85 | no (`Cross-level notes` exists but memory sections are off) | evidence | small |
| 2 | Game-over ledger | lordhansolo | partly (death frame, BAR RULE; no counts) | evidence | small |
| 3 | Per-action `plan` record | lordhansolo | no | evidence | medium |
| 4 | Scoring facts in the system prompt | lordhansolo | no | facts | tiny |
| 5 | Saved code with regression tests | lordhansolo | partly (`ARC3_PERSISTENT_FUNCTIONS=game`, no tests) | tooling | large |
| 6 | Animation readings (tween/path/blink) | lordhansolo | frames only (`ARC3_ANIMATION=1`) | evidence | medium |
| 7 | Strategy audit on stall | lordhansolo | a simpler nudge (`early-level-recovery`) lost: −11.6 / −6.3 | advice | low |

### 1. Solved-level memory

[`cells/42_patch_cell_v16_8_m85_solved_level_memory.py`](https://github.com/LohitSiriki/arc-agi-3-milestone2-solution/blob/af105202327dcf35bd670a50b2558adab12e8dd3/cells/42_patch_cell_v16_8_m85_solved_level_memory.py).
After a level-up, the next opener asks the model to state, in its reply,
"Rule for level N:" with the goal, what each control did and the winning plan
(≤ 5 lines). The harness captures it from the visible text or the reasoning and
pins one block at the end of the system prompt for the rest of the game: per
solved level, the rule, the action count, the last 30 winning actions and the
last 900 characters of the clearing turn's reasoning (rule only for levels older
than the newest three).

Their evidence (transcripts of two local runs, not an A/B): every level a run lost (cd82 L2, dc22 L2, s5i5 L3, sb26 L4) was
entered without the previous level's rule and spent 60–100+ minutes re-deriving
it (cd82 L2: 122 actions vs 8). Our harness keeps history across levels, but a
trim at 118k cuts back to 58k, after which the previous level's turns are gone;
our level-up board shows the solved *board*, not the rule or the winning moves.

Proposed flag `ARC3_SOLVED_LEVEL_MEMORY=1`. Pinning changes the prompt prefix
once per level-up, a re-prefill like one trim. Measure: generated tokens per
cleared level on levels ≥ 2, and actions per cleared level (score efficiency).

### 2. Game-over ledger

lordhansolo adds one line to the turn opener after a death, and a `game_overs`
runtime list:

> Game overs on this level so far: 3. Each ended a run of this many executed
> actions: 24, 24, 24. The newest one's board is in `game_overs`.

Equal counts identify a fixed action budget; differing counts point at something
on the board. It is a fact the harness already has, not advice. Our death
message explains the bar rule in prose but never gives the counts across
attempts. Proposed flag `ARC3_GAMEOVER_LEDGER=1`.

### 3. Per-action `plan` record

The `python` tool takes a one-line `plan` beside `code`; it is stored on every
action the call executes and readable as `history[i].plan` after the turn has
been trimmed. It returns the agent's own intent for each action, the piece of
evidence that trimming destroys first. Proposed flag `ARC3_ACTION_PLAN=1`; it
changes the tool schema, so the model must learn to fill it (lordhansolo
describes it in the tool reference).

### 4. Scoring facts

lordhansolo's system prompt states facts our prompt omits: every game is
solvable; an uncleared level scores zero; among cleared levels fewer actions
score higher, up to a limit; later levels weigh more. They are facts, but they
can shift behavior (fewer probing actions), so the sign is uncertain. Tiny to
add; test only inside a bundle.

### 5–6. Larger changes, lower priority

- **Tested saved code.** `save_module(name, source, tests)` reruns earlier tests
  on frozen observations and rejects a save that breaks them
  ([`python_workspace.py`](https://github.com/tonghuikang/daniel-franzen-arc-agi-3/blob/main/kaggle/lordhansolo/dataset-taaf-kaggle-source/src/ARC3-Inference/inference/agent/python_workspace.py)).
  Our retained functions carry code but not evidence that it still fits the
  game. Large change, unmeasured anywhere.
- **Animation readings.** A one-line reading of each animation (`tween` from/to,
  `path` points, `blink` cycles, `other` bbox), with the moving sprite named by
  its segmentation hash. sirikilohit measured that scalar animation counts did
  not help (11.55 vs 12.04) and switched to showing frames; we already expose
  frames, so this mainly saves the agent code.

### 7. Strategy audit (low priority)

How it works (`inference/framework/solver.py`, `prompts.py` in his bundle):

- **Trigger:** wall-clock. The interval is 25% of the game's initial runtime
  limit, about 16 minutes at his 65.3-minute per-game cap. The timer starts with
  the game; before each analyzer turn, if the interval has passed, that turn's
  user prompt gets `STRATEGY_AUDIT_PROMPT` appended after the state lines.
- **Repeat and reset:** after an audited turn the timer restarts from that
  turn's start, so it fires every ~16 minutes while the level stays uncleared.
  A level clear restarts it; deaths and the model's claims do not.
- **Text, gated in three tiers:** (1) if recent observations match predictions
  and a concrete plan is advancing, state the evidence and continue; (2) a
  search whose rules were checked against real observations may continue, but
  confidence, passing self-written tests or a growing search count are not
  evidence; (3) otherwise separate observed facts from assumptions, pick one
  uncertain assumption and an alternative, check existing transitions first, and
  only then run one short discriminating experiment, with the hypothesis and
  predicted outcome in the call's `plan`, each action's result checked before the
  next, and the counterexample added to the saved module's tests.

It leans on his `plan` field, tested modules and queryable history, so it is
not the same as our `early-level-recovery` nudge, which fired once, only on
levels 1–2, at 32k generated tokens, with a short generic text. That arm lost in
both runs, which argues against generic nudges rather than against his full
design. Our priority gate pauses games, so a port would need a token clock
instead of wall-clock time. Test it only after ideas 1–3 exist, since its text
depends on them. sirikilohit also reports that a "ledger for proven and refuted
assumptions" was never used.

## Small cleanup

With `ARC3_MEMORY_SECTIONS=off`, the death message still ends "Record the
diagnosed cause in your world model" (`tool_agent.py`, game-over block), and a
later line says "record the correction in your world model". Our general
guidance asks the model to keep a world model in its reasoning, so this is
ambiguous rather than broken, but sirikilohit's lesson is that removing
misleading lines paid off. Worth rewording when touching that block.

## Suggested combinations

Effects under 8 points are invisible in one 25-game pass, so bundle by type and
judge each bundle on a lower-variance metric as well as the score:

1. **Evidence bundle:** solved-level memory + game-over ledger (+ `plan` record
   once implemented). Same rule as the level-up board, the one recent change that
   helped. Secondary metrics: tokens and actions per cleared level on levels ≥ 2;
   clears after 40k tokens on a level.
2. **Facts bundle**, only if 1 holds up: add the scoring facts.
3. Tested saved code only after 1 and 2, since it is the largest change.

Leave out generic advice prompts. The strategy audit, if tried at all, comes
after the `plan` record exists.
