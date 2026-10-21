# memory-main: memory-v2 and level-path on the user-combo-10x stack

Branch `arm/memory-main`. This is step 1 of section 5 in
[graph-exploration-bandits-memory.md](graph-exploration-bandits-memory.md): the user asked for a
memory that stores the actions taken, not only ledgers. memory-v2 (#64) and level-path (#66)
already do this. Both were written on stack-v2, about 247 commits behind main, and neither has a
scored run on the best stack.

Main's own memory fields are nearly empty in practice. In `v4-user-combo-10x` v1, 1.1% of replies
carry a memory tag, the refuted list is empty in all 79 cards, and 82% of card lines are clipped
level notes. All 16 python calls that held a tag failed with a SyntaxError.

## What was ported

`origin/main` (7f8195bd), then two merge commits, not rebases:

1. **`arm/memory-v2`** (e0e9c5ac), with its #104 audit fixes:
   - 867a17f3 `ARC3_MEM_REPLY_WINS`
   - e08727bb delta-placement wording
   - 76b83895 stack-v2's think-fold under `ARC3_MEM_SOLVED` alone
   - bb38b9b4 / e0e9c5ac run records
2. **`arm/level-path`** (91ebe378), with its #106 audit fixes:
   - 3896c24f / 91ebe378 `level_path_before_death`
   - 9e4c2362 the solved line uses the route on record when the winning walk keeps an UNDO

`arm/memory-xml` (#65) is not brought in. It and `arm/memory-v3` (#67) were checked for fixes to
shared code. Their fixes (#105, #107) touch only the item store and the XML parser. memory-v3
merged level-path before the #106 fixes, so level-path itself is the newer source.

### Conflicts and how each was resolved

Each conflict keeps main's behaviour for everything outside the memory flags.

| File | Resolution |
|---|---|
| `tool_agent.py` (turn opener) | Main marks where the opener's static tail starts (`static_start`, for `ARC3_OPENER_DEDUP=static`). memory-v2 adds the per-turn Seen-before and `<memory_update>` lines after the valid-actions line. The memory lines go first, then `static_start`, so the per-turn lines stay out of the block that dedup treats as identical every turn. |
| `prompts.py` `memory_addendum` | memory-v2's #104 fix picks the solved line by placement (card form or `solved L<N>:` delta form). level-path picks it by `ARC3_MEM_PATH` (route wording). Both choices are kept. A new `MEMORY_SOLVED_ROUTE_DELTA_LINE` covers path on in placement delta. Every other flag combination keeps its branch's exact text. |
| `tests/memory_sim.py` | `MEMORY_V2_FLAGS` lists `ARC3_MEM_REPLY_WINS` (memory-v2) and `MEMORY_V3_FLAGS` (level-path), so the flags-off test clears both. |
| `tests/test_ledger_on_trim.py` | Both sides appended tests; both kept. |
| `kaggle/build_notebook.py` | Main's duplicate check kept as is. memory-v2's extra `"stopped"` exemption is not carried over. Main's check covers other branches' runs and submissions. |
| `kaggle/arms.json`, `kaggle/runs.json` | Main's entries unchanged; memory-v2's arms and its two run records appended. |
| `README.md`, `kaggle/README.md`, `CONFIGURATION.md` | Main's text, plus level-path's memory-arms section and both source-map rows. |

## Changes on top of the port

- **`ARC3_MEM_CODE_TAGS`** (step 3). Tags written inside python code used to be recorded and
  stripped only with `ARC3_MEM_CARD`.
  - Unset, the flag is now on whenever any memory flag is on: `ARC3_MEM_SOLVED`, `_CARD`,
    `_EFFECTS`, `_SEEN` or `ARC3_MEM_PATH`.
  - `1` turns it on for any ledger, including `ARC3_LEDGER_ON_TRIM` alone. `0` turns it off.
  - With every memory flag unset, the code runs as written, as on main. That keeps
    `user-combo-10x` byte-identical. This is the one deliberate departure from "whenever any
    ledger is on" in the report.
- **`ARC3_MEM_REPLY_WINS=1`** (step 2) is set in all three new arms. The reasoning is absorbed
  before the reply, so a tag drafted in the thinking no longer replaces the reply's final one.
- **Arms** (`kaggle/arms.json`). Each takes `user-combo-10x`'s env and serve settings, runs
  121 min on all games, and adds:

| Arm | Adds | Fingerprint (built locally, `--new-batch`, not pushed) |
|---|---|---|
| `mem-solved-10x` | `ARC3_MEM_SOLVED=1`, `ARC3_MEMORY_PLACEMENT=both`, `ARC3_MEM_REPLY_WINS=1` | `c8c717ef65f4a6b9` |
| `mem-v2-10x` | `ARC3_MEM_SOLVED`, `_CARD`, `_EFFECTS`, `_SEEN` = 1, placement `both`, `ARC3_MEM_REPLY_WINS=1` | `bda9ea68f047658c` |
| `level-path-10x` | `mem-v2-10x` + `ARC3_MEM_PATH=1` | `6e7866a32751dc03` |

## Tests

All tests run in the shared lock under `nice`.

- **Flags off equals main.** These pass unchanged, with no golden regenerated:
  - main's `test_flags_off_identity.py` and `test_flags_off_requests.py`;
  - the branches' `test_memory_flags_off_identity.py` (stack-v2 golden, plus the
    level-path-off golden from memory-v2);
  - `test_anim_loop.py` and `test_opener_dedup_loop.py`.

  The stack-v2 golden still holds because main changed nothing behind flags that are off.
- **New `tests/test_memory_main_loop.py`.** It uses `anim_loop_sim`: the real solver session,
  `analyze()`, the sandbox and the priority gate, with every request rendered through the Qwen
  chat template. It runs each arm's env from `arms.json` under the Kaggle `setup_env`, in a 30k
  window so history is trimmed, with a model RESET.
  - With memory flags off, user-combo-10x's requests and engine actions are byte-identical to
    7f8195bd, in both the tight and the 96k window. The memory knobs alone change nothing.
  - With each arm on:
    - every prompt extends the previous one unless history was evicted;
    - the system prompt and tools never change;
    - the game plays through;
    - the `<solved>`, `<actions>`, `<hazard>` and route text reach the model;
    - a bare tag in code no longer fails the call and reaches the card;
    - with the card, the reply's hypothesis wins over the reasoning's draft.
  - Setting `ARC3_MEM_REPLY_WINS=0` or `ARC3_MEM_CODE_TAGS=0` makes the matching check fail.
  - `anim_loop_sim` gains `ANIM_SIM_MEMORY_TAGS`. Unset, the sim is unchanged.
- **#58 eviction guard.** `eviction_sim` gains `combo-10x` and one `combo-<arm>` config per arm.
  All pass every check.
  - At the plain configs' 14k window, `combo-level-path-10x` trimmed twice in a row once
    (request 6).
  - The cause is size, not a memory bug. This stack's system prompt plus first opener is about
    8.1k tokens, so 14k leaves about 4k for history.
  - The combo configs therefore run at 18k with the same drain share and still trim five times.
    In the arms' 96k window, a double trim would need one turn to add more than the 42k drain.
- The memory suites from both branches pass: `memory_sim`, `test_memory_placement`,
  `test_memory_eviction`, `test_memory_facts`, `test_level_path` and `test_memory_wording`.

## Pre-registered keep rule

Judge on full 121-min runs only. A 25-min panel cannot see the card: the first trim, and with it
the first card, comes around request 65 of about 203 per game.

Measure each arm against `user-combo-10x` (repeats 56.94, 52.70, 49.09). Also compare
`level-path-10x` with `mem-v2-10x`.

- **Score.** Higher than `user-combo-10x` by `make significance BASELINE_SCORE=...
  CANDIDATE_SCORE=...` (paired per game) over repeats. The user's rule is that repeated runs must
  score higher consistently. One run is a screen that decides whether to spend the repeats.
- **Loops.** The share of actions on net-zero loops falls by at least 3 points.
- **Repeats.** Exact (board, action) repeats are at most 6% of actions.
- **Route use** (`level-path-10x`). `level_path` appears in at least 1% of executing snippets.
  Drop the arm if it is under 1% and loops and repeats do not change.

The expected effects are small next to a run sd of 1.7–4. The usage rules (loop share,
`level_path` share, repeats) should give a clear signal within 1–2 runs. A score claim needs
several paired repeats. Run `mem-solved-10x` first. The solved-level record is the part that
could matter more: it is the closest thing to rellik13's +2.8.

## Results

| Run | Score | Levels | Note |
|---|---|---|---|
| `user-combo-10x` v1/v2/v3 | 56.94 / 52.70 / 49.09 | | the control |
| `level-path-10x` | 48.31 | 104 | ahead at 25-30 min (10.09 vs 8.07-9.58), behind from 60 min (28.60 vs 32.59-41.34) |
| `mem-v2-10x` | 43.81 | 106 | tn36 3.57, cn04 3.09, ka59 0.50, ls20 12.39 |

Serving, cache hit (90.2%), trims and prompt size match the control, so memory cost no
throughput. One run each: no score claim either way.

## Audit (2026-10-10)

The user: "memory didn't seem to improve ... I think the implementation isn't correct". Four
audits (facts, harm, usage, design) were deduplicated and each claim re-checked on the logs or
with a test. Scratch scripts and outputs: `scratchpad/memaudit/` and `scratchpad/memaudit/fix/`
of the session (not in the repo).

**Verdict.** The implementation computes what it was designed to compute: an offline-engine
replay matched all 5,220 (level-path) and 5,763 (mem-v2) boards, and re-running the
`FactRecorder` on the events reproduced `*_level_path.jsonl` exactly. Wrong facts are rare and
do not line up with the losing games. The design is what is wrong: the card stores harness
facts (paths, counts) and none of what the model loses at a trim (the goal, the verified rules),
the one model sentence it keeps (`how`) is the wrong sentence, and the model almost stopped
writing its own notes. Of the six biggest losses, four are within run-to-run noise (a control
repeat stalled at the same place) and two had other causes (ls20 scheduler waits after trims;
ka59 a malformed click that returned silently).

### Confirmed and fixed (flag-gated; the recorded arms are unchanged)

| # | Bug | Evidence (re-checked) | Fix |
|---|---|---|---|
| 1 | **The card holds no goal or rules; the model stopped writing notes** (high) | Model tags: user-combo-10x v1 58 in 31 messages (1.13%); mem-v2-10x 10 in 6 (0.22%); level-path-10x 4 in 2 (0.07%) of 2,767. Level-path cards by chars: 40.7% solved routes, 53.4% action counts, 1.9% model notes. tn36: the L1 rule ("the tray of 5 T-glyphs is a counter", requests line 26) was gone after the first trim (line 52); the card held only the click path and the model doubted the rule (line 60). | `ARC3_MEM_DEBRIEF`: the model's own recap from its first reasoning after a clear rides on the solved line as `debrief:`; model text comes first on the card and is cut last; notes carry `L<N>:`; the prompt restores the ledger's "holding what you recorded" framing and states the debrief. |
| 2 | **`how` is the clearing turn's last plan** (medium) | All 25 `how` notes in level-path-10x listed; 0 describe the route. m0r0 L3 "74 acts [...+54] \| how: 5 moves: UP,UP,UP,RIGHT,RIGHT ✓"; g50t L1 "LEFT, DOWN×5..." on route [R4 S D7 R5]; ka59 L1's `how` drove a wrong goal revision (requests lines 126-130). | `ARC3_MEM_DEBRIEF` drops `how` (and the level note) from the solved line. |
| 3 | **Placement `both` repeats solved lines** (medium) | `dup_check.py`: 668 of 2,014 card requests (level-path) and 839 of 2,032 (mem-v2) hold the same solved level in the card and in a delta; 257 and 216 head messages hold both. | The corrected arm uses `ARC3_MEMORY_PLACEMENT=trim`. |
| 4 | **Solved paths drop edge-only actions** (medium) | Offline-engine replay of every clear from its attempt start: ar25 L3 (47 actions) does not clear; 103 of 104 level-path paths and 106 of 106 mem-v2 paths do. (The facts audit's g50t L4 failure in mem-v2 did not reproduce.) | `ARC3_MEM_KEEP_EDGE`: keep edge-only and anim-only actions and cut no loop across one: 104 of 104 and 106 of 106 clear in exactly their length, paths +1.1 and +2.1 actions on average. |
| 5 | **Route lines cost ~635 tokens per request and are only printed** (medium) | `level_path` in 22 of 997 executing snippets, passed to `action()` in 0; route and Seen-before text in request snapshots 4.01M chars (level-path) vs 0.33M (mem-v2). | The corrected arm leaves `ARC3_MEM_PATH` off (the pre-registered drop rule's usage condition). |
| 6 | Card truncation drops model notes before harness rows (low; would bite once debriefs exist) | `render_card`'s loop | Debrief card cuts action rows, then paths, then notes, debriefs last. |
| 7 | Model notes carry no level (low) | `_absorb_card` | `L<N>: ` label in debrief mode. |

The extractor finds a recap at 24, 28 and 31 of 96, 100 and 113 level-ups of level-path-10x,
mem-v2-10x and user-combo-10x v1 (19-20 of 25 games), from replies the model wrote without
being told the recap is kept; with the prompt line it may write more.

### Confirmed, not fixed

- The head card's `<actions level=N>` is a previous level's in 423 of 2,014 card requests (21%,
  re-counted). Redrawing the card at a level change would break the cached prefix; the table is
  labelled with its level, and in the debrief card it comes last and is cut first. No case of harm
  was found.
- Solved paths are cut at 64 chars (31 of 96 lines; 22% of route actions never shown), and the
  hidden part cannot be retrieved. Low.
- The effect table counts clicks by colour, so objects of one colour merge (tn36 line 110, su15
  line 150). Low.
- Moves inside the 4-cell edge band read "no board change" (ka59 #86-89). Low; the corrected arm
  has no route echo.
- 3 of 179 Seen-before "you tried" claims are wrong (dc22). Low.
- Not memory: the sandbox's `action()` returns silently on a malformed action. ka59 line 104 ran
  `step({'action':'MOUSE','row':44,'col':19})` through a helper that wraps its argument as an
  action name; the result came back `'ok': True, 'gp': None`, no click ran, and the model concluded
  "clicking doesn't transfer control" (line 108). All three control runs cleared ka59 L2 with
  4-9 clicks. Fixing it changes main's behaviour, so it belongs in its own arm.

### Corrections to the audit inputs

- m0r0 has 19-20 unique Seen-before lines; 498 counted the same lines re-sent in history.
- `ARC3_MEM_PATH`'s "(N in the winning attempt)" never appeared (0 of 104 clears): level-path
  changed no solved line relative to mem-v2.

### The corrected arm and its read-out

`mem-v2-fixed-10x` = `user-combo-10x` + `ARC3_MEM_SOLVED`, `_CARD`, `_EFFECTS`, `_SEEN`,
`ARC3_MEMORY_PLACEMENT=trim`, `ARC3_MEM_REPLY_WINS`, `ARC3_MEM_DEBRIEF`, `ARC3_MEM_KEEP_EDGE`.
The loop sim (`tests/test_memory_main_loop.py`) runs it through the real game loop and the
chat template: every request extends the previous one between trims, the system prompt is
static, the recap reaches the card, no `how`, no deltas, no route lines.

Screens on its first full run, before any score claim:
- a `debrief:` on solved lines in at least 15 of 25 games, and model tags at or above the
  control's rate (1.1% of messages);
- the same throughput, trims and cache hit as `user-combo-10x`;
- then paired repeats against `user-combo-10x` with `make significance`.

### Follow-up: the user's layout

The user asked for the memory in every request, the previous levels first, and legible lines
(2026-10-10). That is [memory-layout.md](memory-layout.md): `ARC3_MEM_EVERY_CALL`,
`ARC3_MEM_LEGIBLE`, `ARC3_LEVEL_COMPACT`, the recap precision fix, and the `mem-layout-10x` arm.
