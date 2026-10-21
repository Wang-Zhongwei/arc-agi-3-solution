# Harness ideas harvest: what other harnesses have that ours lacks (2026-10-08)

Owner (2026-10-08): "use safe and good ideas from other harnesses to improve ours", and "i can allow you to make slight prompt changes (similar to how Daniel Franzen made them.) i will analyze the differences in prompts later today side by side and see what else 'world model'-like we can add." Owner's rule for this harvest: the model is a **local** Qwen3.8-Flash-Next (W4A16, one RTX PRO 6000, offline, fixed wall-clock), so what worked for frontier models may not transfer.

Read-only research; nothing was run on Kaggle or committed. The companion file `prompts-side-by-side.md` quotes every harness's effective prompt verbatim.

## Short answer

- **The ideas that are safe and backed by evidence on this model are all of one kind.** They either make our existing text true (honesty fixes), or they hand the model a fact the harness already computed in the newest turn message. Every idea that needs the model to write structured prose every turn showed about zero uptake on this model, in every harness we can measure:
  - Wang `World model:` prefix: 2 of 1,442 reasoning blocks.
  - `register_predictor`: 0 of 2,762 calls.
  - Tycho builder: 0 calls in 10 games.
  - Logprob feedback: referenced in 0 of 1,442 blocks.
  - Our model-written tags: 0.9–2.1% of calls.
- **Two exceptions show that the model does write when it is asked once, at a defining moment, or through a tool parameter:**
  - rellik13's one-off "Rule for level N:" ask at a level clear: 27 of 28 levels answered (tier 2).
  - The GB10 `prediction` tool parameter: 93% filled (tier 1).
- **Four ADOPT items** (safe, with tier 1–2 evidence):
  1. A prompt-honesty pass on top of PR #74.
  2. rellik13's solved-level rule capture, placed so it is cache-safe, on top of memory-v2.
  3. Segmentation-API additions and errors that teach the API, folded into sandbox-v3.
  4. A run-insurance bundle.
- **Everything else is TRY or SKIP.** No public harness has an isolated, repeated A/B for any prompt idea. The tier-2 numbers are single Kaggle submissions that bundle several changes; identical Franzen code ranges about 20–34 on the LB.

**New findings from this pass:**
1. **Our GAME OVER block tells the model twice to write to a world model that does not exist** under `ARC3_MEMORY_SECTIONS=off`:
   - "Record the diagnosed cause in your world model." (`M/agent/tool_agent.py:4992`)
   - "...record the correction in your world model instead of retesting it." (`:5043`)

   Franzen removed exactly this kind of dangling instruction from the opener. He measured that the broken version dropped the share of replies carrying any assistant text from 76% to 43% (his comment at `M/agent/tool_agent.py:5111-5122`). No open arm touches these two lines.
2. **The `restart-wording` backlog item in `arms.md` misquotes its source.** rellik13 shipped `"Level restarted."`, not "level restarted (attempt N)" (`R/cells/40_*.py:434`). It fixed Tufa's `"The game is over."`, which our harness already replaced with a long, accurate GAME OVER block. So for us that item is moot as written.
3. **UNDO naming is not a gap.** We already expose `UNDO` with `UNDO_INFO_ADDENDUM` (`M/agent/prompts.py:105-107`).
4. **Several public forks' tuning flags never took effect.** They set flags in the "Customization hook" cell, which runs after the benchmark unpickle has already imported `tool_agent`. So module-level reads were frozen before the hook ran: `ARC3_AUTO_FRAME_DIFF`, `YIELD_SECONDS`, `TOOL_TIMEOUT` and the failure/backoff settings (sovereign-agent v1). Their scores say nothing about those flags.
5. **Our opener repeats a static instruction block of about 500 tokens.** It appears 5–34 times (median about 15) in each game's final context in r2, about 10% of context (tier 1 count; issue #17).
6. **Completion dominates action efficiency.** Franzen's 4×25 artifact loses 50.88 points to unfinished levels and only 2.63 to action inefficiency (trajectory-audit notebook, tier 2). Arms whose main effect is "fewer actions" are capped near 2–3 points.

## Method, keys and legend

**Sources** (all read; the per-source working notes were made by four read-only research passes):
- **Ours:** `daniel-franzen-sol` `main`, with the Kaggle env from `kaggle/template.ipynb` cell 4. The rendered system prompt matches a real `v4-panel-control` request byte for byte (18,619 chars).
- **Tufa:** Duck (`competitors/duck-harness`, plus `bundle-base`).
- **Other harnesses:** Lord Han Solo (the v305 bundle that scored 23.84), rellik13/sirikilohit (22.53), Fable F001, Tycho (our port and run), the owner's Wang `final-c1` / GB10 branches, and all 28 pulled notebooks in `competitors/kaggle/` (DF-WM, sovereign, kinetic, m2-top, win-v2, sigeward, Scott Le Grand's agentfix, ...).
- **GitHub competitor repos:** prime-agent, polyphony, dream-team, solve-arc-agi-3, ARC3-solution, the official ARC-AGI-3-Agents, tong-m2-comparison.
- **Our own docs and history:** `harness-feature-matrix.md`, `fable-vs-ours.md`, the design docs (animation-combo, tycho-world-model-port, sandbox-v3), `leaderboard-35-45.md`, `arms.md`.

**Path keys:**

| Key | Path |
|---|---|
| `M/` | `daniel-franzen-sol/ARC3-Inference/inference/` (main) |
| `R/` | `competitors/rellik13-milestone2/` (`cells/NN_*.py` are notebook cells) |
| `HS/` | `competitors/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/inference/` |
| `HSETUP:n` | line n of the decoded `setup_commands.json[0]` |
| `T/` | `tycho-port/tycho/` |
| `W/` | `arc-agi-3-v3/final-c1/ARC3-Inference/inference/` @`bfe3a78` |
| `F/` | the Fable layer `fable_layer_core.py` (leoprovorov notebook) |
| `K/<nb> cN` | `competitors/kaggle/<nb>`, code cell N (0-based) |
| `C/` | `competitors/` |

**Evidence tiers** (owner's rule):
- **T1:** our own runs on this model. Includes the owner's Wang-era Kaggle and GB10 runs and our Tycho port run, all on Flash-Next.
- **T2:** other teams on Qwen3.x-Flash-Next or the Duck/Franzen harness: Lord Han Solo, rellik13, Fable, DF-WM, the Franzen forks.
- **T3:** comparable open ~30B models (polyphony and solve-arc-agi-3 on Qwen3.6-27B).
- **T4:** frontier models or another domain (prime-agent on Opus 5, dream-team, Tycho's paper, StochasticGoose CNN). Inspiration only.

Anything resting only on T4 cannot be ADOPT.

**Safety fields:**
- **Flag:** can it be default-off with flags-off requests byte-identical to `main`?
- **Cache:**
  - *static*: system or tool text that never changes in a game;
  - *tail*: new text only in the newest turn message or tool result;
  - *trim*: shown only at a history trim, riding the re-prefill the trim already pays;
  - *breaks*: edits earlier messages or the system prompt mid-game.
- **Tok:** prompt tokens per turn (chars/4).
- **Neg:** resemblance to a known negative. The negatives are:
  - fixed advice text on every prompt (rellik13 6.42 → 4.16; a level-start advice block 13.40 → 9.98);
  - summaries;
  - text world-model sections (Franzen's largest gain was turning his off);
  - consensus sampling;
  - images above 10×;
  - an executable world model with replay validation (Makarov; DF-WM 29.30 ≈ noise; Tycho port 3.53);
  - extra blocking guards;
  - scratchpad tags and thinking guides;
  - D′ (−7.5 real), UCB (43.84), the strategy audit (44.10), prompt dedup by rewriting history (A3 5.25).

**Verdicts:**
- **ADOPT:** safe, with T1–T2 evidence; build it.
- **TRY:** safe, but the evidence is weak or only process-level; screen it, then run ≥2 paired full runs.
- **SKIP:** with the reason.

---

## 1. Prompt honesty and wording (static or tail text)

### P1. Make the GAME OVER block's world-model lines true
- **Source:** ours, `M/agent/tool_agent.py:4992` and `:5043`. The model for the fix is Franzen's own fix to the opener (comment `:5111-5122`, code `:5123-5126`).
- **What:** under `ARC3_MEMORY_SECTIONS=off` there is no world model to record into.
  - **Fix:** drop the clause, or point it at a real carrier when one exists. With the ledger arms (#45, #56, #64), "record the cause as a `<hypothesis status="refuted">` tag" names the actual memory channel at the exact moment a hypothesis was falsified. That is the same "ask once at a defining moment" pattern that got 27/28 compliance in rellik13's M85.
  - With #76's carried variables, "in a python variable" would also be true.
- **Evidence:** T2, Franzen on this model and harness: a dangling world-model instruction in the opener cut the share of replies with text from 76% to 43%. For the death block itself, none.
- **Safety:**
  - **Flag:** yes.
  - **Cache:** tail; it only appears after deaths.
  - **Tok:** −10 or +5.
  - **Neg:** no; it removes a false instruction.
- **Overlap:** none (#74 doesn't touch it). It pairs with #45/#56 ledger-on-trim.
- **Verdict: ADOPT** (fold into P2's arm).

### P2. Finish the step-bar honesty fix (extends PR #74 budget-truth)
- **Sources:**
  - rellik13 v18.1g, `R/cells/40_*.py:498-508`: deletes the whole clause "and whether the last action affected gameplay state or only HUD elements such as countdown bars" (#74 removes only "such as countdown bars").
  - Han Solo's budget sentence, `HS/agent/prompts.py:58`: "It is a readout of that budget rather than a puzzle element, every executed action counts against it whether or not the board changed, and exhausting it ends the level in a game over."
  - Tycho's generalisation, `T/prompts/actor.system.j2:29`: "Some cells may be HUD or interface state (budgets, lives, timers, selected tools, progress indicators, validation lights, status bars); treat them as observations of game state, not decoration, and infer their role from transitions and terminal evidence."
- **What:** one consistent story about the bar. Today our prompt says both "do not get distracted by it" (`M/agent/prompts.py:84`) and "count remaining budget into your plans" (`:226`).
- **Evidence:**
  - **T1 diagnosis:** the model calls the bar a real-time timer in 12 of 25 games in each of r1, r2 and control.
  - **T2:** rellik13's "harness fixes" bundle went 13.40 → 14.49 (one submission; bundled with 4 other changes).
- **Safety:**
  - **Flag:** a sub-flag of `ARC3_BUDGET_TRUTH`.
  - **Cache:** static.
  - **Tok:** net −10 to +40 cached.
  - **Neg:** no; it corrects wrong text.
- **Overlap:** #74 (core), #60 budget-alert (facts about the bar).
- **Verdict: ADOPT** as a second commit on #74. Keep Tycho's "lives, selected tools, validation lights" generalisation; it also covers sk48-style tray marks, which #74 doesn't.

### P3. True facts about turn mechanics we never state
- **Sources:**
  - Han Solo `HS/agent/prompts.py:53`: "The turn is over as soon as a `python` call executes an action, so anything you want to do after acting must live inside that same snippet." Ours already behaves this way (the tool loop breaks after `step_executed`, `M/agent/tool_agent.py:~7093-7105`) but never says so.
  - Han Solo `HS/agent/tool_agent.py:695-699`: state line "actions executed since this level last started = N".
- **Evidence:** T2, shipped, unmeasured. It is related to the failure-taxonomy finding (T1) that 40.7% of tokens go to python calls that never act; knowing that acting ends the turn changes how a snippet should be written.
- **Safety:**
  - **Flag:** yes.
  - **Cache:** static / tail.
  - **Tok:** +25 static, +8 per turn.
  - **Neg:** no (facts).
- **Overlap:** #60 (bar facts use the action count), #75 cutoff-signal (same "tell the model what the harness did" family).
- **Verdict: TRY** (ride along with P1/P2).

### P4. rellik13's "Level restarted." and UNDO naming
- **Source:**
  - `R/cells/40_*.py:430-451`: `M102_OLD = "The game is over."` → `M102_NEW = "Level restarted."`.
  - `R/cells/40_*.py:339-341`: `ACTION7` → `UNDO`.
- **Evidence:** T2, inside the +1.09 bundle. His motivation: 452 game-overs in six runs; the model reasoned as if the game had ended ("the harness says 'The game is over' but the image shows the player back at start position").
- **Ours:** both are already covered.
  - Our GAME OVER text says the attempt failed, the level restarted and the run is not finished (`M/agent/tool_agent.py:4969-4993`).
  - UNDO is exposed and named (`EXPOSE_UNDO=on`, `UNDO_INFO_ADDENDUM`).
- **Verdict: SKIP as written** (already true in ours). Correct the `arms.md` backlog quote. The live question is P5.

### P5. Short factual death text instead of our ~550-token GAME OVER / BAR RULE block
- **Sources:**
  - rellik13's one-liner (above).
  - Han Solo's death sentence, `HS/agent/tool_agent.py:684-688`: "The last of those actions caused a game over, and the harness then executed a RESET, counted as one more action. The level is back at its starting board ... The board at the game over is `previous_frame` this turn and `game_overs[-1].frame` until the next game over."
- **Evidence:**
  - T2 for short text (bundled).
  - **Against:** Franzen lists "detailed game-over guidance" under what worked (T2, same model and harness, qualitative).
- **Safety:**
  - **Flag:** yes.
  - **Cache:** tail.
  - **Tok:** about −450 per post-death turn (≈24 deaths per run).
  - **Neg:** it removes advice, but contradicts a Franzen positive.
- **Verdict: TRY, low.** Run it only after M1 (the facts line). Don't replace the block before the facts that would replace it exist.

### P6. Remove the static instruction block from every opener (issue #17)
- **Sources:**
  - rellik13 M90's 5-line turn message, `R/cells/40_*.py:64` (static content moved to `TURN_PROTOCOL_ADDENDUM`).
  - M24's measurement, `R/cells/21_*.py:41-42`: "34.2% of every prompt are EXACT DUPLICATE LINES".
- **What:** our opener repeats about 12 static lines ("Only tool: `python`. It receives ...", "Keep tool output compact...", the tool-call format line, ...). Nearly all of them are already in our system prompt (§2 of the side-by-side). Dropping them from **new** openers shrinks context growth by about 500 tokens per turn without editing any old message.
- **Evidence:**
  - **T1:** about 500 tokens × a median of 15 copies per final context (r2 prompt logs), about 10% of context. Trim re-prefills are 48% of uncached tokens (issue #10), so fewer trims save real wall-clock.
  - **T2:** part of rellik13's +3.44 bundle.
  - **Against (T1/T2):** the old-harness A3 "prompt dedup" scored 5.25 (it rewrote history; this doesn't), "token-efficiency" (33% smaller) gave no speed-up, and Franzen's shorter resume prompts gave "no consistent gain". He also tuned these lines ("clarified repeated openers" is in his worked list).
- **Safety:**
  - **Flag:** yes.
  - **Cache:** tail-safe (old turns unchanged).
  - **Tok:** −500 per turn.
  - **Neg:** mixed.
- **Overlap:** issue #17, issue #21 (duplicate opener on yield), backlog `yield-6k`.
- **Verdict: TRY.** Keep the dynamic lines and the closing "When ready, call `action(actions)`..." line; drop only what the system prompt already says verbatim. Judge it by trims per game and tokens per cleared level first, then score.

### P7. Scoring explanation, "every game is solvable", "full certainty is not required", "never give up"
- **Sources:**
  - Han Solo `HS/agent/prompts.py:51, 56-57, 156, 160`.
  - dream-team `scoring.jinja`.
  - Tycho `actor.system.j2:15`.
- **Evidence:** T1 negative: the Wang scoring guide b4 scored 11.12 vs 17.74; spend-actions 10.92 vs 13.65.
- **Verdict: SKIP.** One possible exception: the single-sentence solvability prior ("every level is solvable by human players"), tested only together with W2.

### P8. Han Solo's turn-message magnitude-reading guide
- **Source:** `HS/agent/prompts.py:164`: "0 means identical..., a handful of cells is most often a timer or step-bar tick, ...".
- **Evidence:** T2, unmeasured.
- **Neg:** static advice.
- **Verdict: SKIP alone.** If M3 ships, its numbers make this guide unnecessary.

---

## 2. "World model"-like memory and prompting

### W1. Solved-level rule capture: "Rule for level N:" (rellik13 M85)
- **Source:** `R/cells/42_*.py:26-32, 87-104, 114-125, 150-189`. Ask text at `:180-182`:
  > LEVEL %d SOLVED. In your reply this turn, before acting, write one short paragraph that starts exactly with 'Rule for level %d:' -- the goal, what each control/action did, and the plan that won (at most 5 lines). It is pinned for every later level.
- **What:**
  - At the first turn after a clear, the harness records the level's action count, its last 30 actions and the last 900 chars of the clearing turn's reasoning. It asks once (at most twice) for the rule, captures it from the visible text or the reasoning, and keeps a SOLVED LEVELS block for the game.
  - The newest 3 levels keep actions and reasoning; older levels keep only the rule.
- **Evidence:**
  - **T2:** +2.8 (7.14 → 9.96), "the only text addition that helped" (`R/WRITEUP.md:66-71`; one submission).
  - **Compliance (T2):** 27 of 28 levels in his run logs got a rule (vs 0.9–2.1% for our free-form tags).
  - **Motivation (T2):** in all 4 levels v16.7 lost, the agent re-derived the previous rule (cd82 L2: 122 actions vs 8). Tufa's "Cross-level notes" field was filled in 0 of 989 level≥2 prompts.
  - **Our analogue (T1):** ledger-on-trim, 52.56 / 49.14 vs 46.25. A real example is quoted in the side-by-side §rellik13.
- **Safety:**
  - **Flag:** yes.
  - **Cache: breaks as built.** He rewrites the end of the system prompt 2–3 times per level, each a full re-prefill of 58–118k tokens for us. **Cache-safe placement:**
    - the ask goes in the level-start opener (tail);
    - the captured rule is stored in memory-v2's `<solved>` record;
    - it is shown in the ledger-on-trim head block (trim) and, optionally, once more in the level-start opener of the next level (tail).
  - **Tok:** about 70 for the ask, 1–2 times per level; the rule is ≤5 lines.
  - **Neg:** partly a model-written world model, but written once per level at the moment of success and frozen. It is not a per-turn section.
- **Overlap:** memory-v2 `ARC3_MEM_SOLVED` (#64) has the harness-facts half. **The ask/capture half is missing everywhere in our arms.** It must obey the #58 eviction rules (#61 guard).
- **Verdict: ADOPT** as `ARC3_MEM_SOLVED_RULE` on top of #64. Also add Tycho's "winning transition" fact: the changed region of the terminal transition plus the winning action (`T/prompts/builder.system.j2:26`, "infer it from what changed at the moments a level ended").

### W2. Wang `GAME_PRIORS` static addendum
- **Source:** `W/agent/prompts.py:201-210` (1,372 chars; also ported verbatim by rellik13 M90, `R/cells/40_*.py:14`). Key lines:
  > "Priors are the hypotheses you test first, not conclusions." / "Action priors worth considering early: the arrow actions usually move or steer something ...; `SPACE` often rotates, switches, or cycles ...; `MOUSE` often selects or toggles ..." / "Work from mechanics outward to the goal ... form a candidate, test it cheaply, revise." / "The goal is sometimes legible before the mechanics are ..."
- **Evidence:**
  - **T2:** inside rellik13's "Wang's harness design" submission, 9.96 → 13.40 (bundled with Wang's sandbox, segmentation and turn protocol; his same-notebook spread was 5.02–6.42).
  - **T1:** on in every Wang run, never isolated.
- **Safety:**
  - **Flag:** yes.
  - **Cache:** static.
  - **Tok:** about 340 cached, 0 per turn.
  - **Neg:** static advice. rellik13's measured harm came from per-turn and level-start text at a 37k context, which crowded out history. Cached static text doesn't do that, though "made it doubt rules it had right" could still apply.
- **Overlap:** none. Franzen's "Action meanings" section (`ACTION_INFO_ADDENDUM`) already says "Test its effect rather than assuming".
- **Verdict: TRY.** It is the most "world model"-flavoured static prompt with any T2 support. Screen on the panel, then 2 full runs.

### W3. A `prediction` tool parameter graded by the harness (GB10 hypothesis-loop)
- **Source:** `gb10-wt/hypothesis-loop` @`fd15523` (`prompts.py:206-218`, `prediction_check.py`). The parameter text: "What the board will show once this call's actions finish ... Checked forms, joined with ';': `no_change`, `<color> moves <up|down|left|right> [cells]`, `cell <row>,<col> becomes <color>`, `level_complete`."
  - The verdict line opens the next turn: `prediction WRONG: expected level_complete, actually the level did not change; white moved left 3 (...)`.
- **Evidence:**
  - **T1 (GB10, 2 games):** 93% compliance; grammar parsed 100%; 3 of 9 right. The "actually ..." diff pointed the model at the right mechanic on ar25 and sk48. Levels tied control (arms.md: inconclusive).
  - It is the only predict-and-verify loop this model demonstrably fills.
- **Safety:**
  - **Flag:** yes.
  - **Cache:** static schema plus a tail verdict. The GB10 version stripped the nudge from history, which breaks the cache; don't.
  - **Tok:** about 220 cached, about 40 per acting turn.
  - **Neg:** partial (a world-model check), but through a tool parameter and a harness verdict, not prose.
- **Overlap:** memory-xml (#65): verdicts could set hypothesis status from the harness instead of the model's claim. The "actually ..." half duplicates M3.
- **Verdict: TRY** (after #65 shows adoption, and without the "Revise the belief..." nudge).

### W4. Han Solo `plan` parameter stored on every executed action
- **Source:** `HS/agent/tool_agent.py:116, 128-135, 734-741`. Schema: "One line naming what the actions in this call are meant to achieve and what result would confirm it. It is stored on every action it executes and stays readable from `history` ... Keep it under 500 characters." Readable as `history[i].plan` after the turn leaves context.
- **Evidence:** T2, unmeasured ("cannot confirm"). The T1 analogue is hypothesis-loop (93% compliance, inconclusive).
- **Safety:**
  - **Flag:** yes.
  - **Cache:** static.
  - **Tok:** +120 cached, +20–100 generated per acting call.
  - **Neg:** low.
- **Overlap:** memory-xml (#65) is the same carrier with more structure.
- **Verdict: TRY, low.** Pick one tool-parameter carrier: #65, W3 or W4.

### W5. Death-time and trim-time facts for the ledger: "ruled out" and "clicks tried → effect"
- **Sources:**
  - polyphony's compaction handoff, `C/polyphony-arc-3/arc_hs/hs_compaction.py:76-112`: "The biggest losses under the old design were dims 4 & 5 (ruled-out directions + already-clicked coords)".
  - dream-team's "Ruled-Out Hypotheses cumulative ledger" (`reflection_directive.jinja`).
  - Han Solo's world-model bullet (`HS/agent/prompts.py:151`), which adds "which hypotheses the observations have ruled out".
- **Evidence:** T3, anecdotal (code comments); T4.
- **Safety:**
  - **Flag:** yes.
  - **Cache:** trim (it rides ledger-on-trim).
  - **Tok:** 0 per turn.
  - **Neg:** none, if recorded by the harness. A model-written summary would be the summaries negative.
- **Overlap:** memory-v2 `ARC3_MEM_EFFECTS` / `ARC3_MEM_SEEN` (#64) and level-path (#66) record most of this. P1's refuted-tag pointer is the cheap model-side half.
- **Verdict: TRY** (a refinement inside #64, not a new arm).

### W6. Tycho's "no visible change is not proof of no effect" as a fact
- **Source:** `T/prompts/actor.user.j2:39-41`.
- **Evidence:** T2: of Fable's journal "no effect" claims, all 15 that were retried later changed the board (Fable HB p.84). T1: 9% of r2 actions are exact (board, action) repeats.
- **Fact form:** "no change inside the board area from LEFT on this board; LEFT changed the board on 3 other boards this level".
- **Safety:** tail; about 30 tokens on no-op turns only.
- **Overlap:** memory-v2 E3 wording.
- **Verdict: TRY** (wording refinement in #64).

### W7. Per-turn world-model protocols and rewritten memory sections
- **Sources:**
  - Wang `TURN_PROTOCOL` (`W/agent/prompts.py:150-157`; also rellik13 M90).
  - Tufa / Franzen memory sections (off in ours).
  - official ARC-AGI-3-Agents MultiModal memory and LangGraph journal.
  - polyphony `hs.md` world-model file.
  - dream-team step-log "Prediction Check".
  - prime-agent `[PLAN]` briefing (`C/arc-agi-3-prime-agent/AGENTS.md:13`).
- **Evidence:**
  - T1: `World model:` in 2 of 1,442 reasoning blocks.
  - T2: Franzen's largest gain in this area was turning his sections off.
  - T1: thinking guide 0/28; scratchpad zero tags.
- **Verdict: SKIP.**

### W8. Executable world models: simulator, replay check, search, verified modules, builder
- **Sources:**
  - DF-WM `check_sim`/`search`/`act_checked` (`K/df-wm c2`).
  - Tycho builder (`T/prompts/builder.system.j2`).
  - Han Solo `save_module` with regression tests (`HS/agent/python_workspace.py`).
  - Wang `register_predictor`/`mechanics()` (`W/agent/prompts.py:112-113`).
  - polyphony Heuristic System, dream-team simulator, solve-arc-agi-3 ADR-0010.
- **Evidence:**
  - T2: DF-WM 29.30, within Franzen noise; Makarov "largely ignored ... at or below baseline".
  - T1: Tycho builder 0 calls; `register_predictor` 0 of 2,762.
  - DF-WM bug: `PredictionMismatchError` subclasses `Exception`, so a model's `try/except` swallows the divergence stop.
- **Verdict: SKIP.** The only open path is the cheap adoption screen in `tycho-world-model-port-design.md` §5.0.

### W9. Level-start mechanics catalogue / knowledge pack (kinetic) and episodic or audit blocks (m2-top)
- **Sources:** `K/kinetic-learning-by-playing-v8 c3` (`ARC3_X_CATALOG`, ~900 tokens per level start); `K/arc-agi-3-m2-top-submission c16` (a fixed "Mechanics observed and validated." string on every turn, plus 4 escalating RESET audits).
- **Evidence:** T2, 29.68 and 31.93, single submissions within noise. m2-top was never validated offline.
- **Neg:** fixed advice, playbook, audit (T1 audit60k 44.10).
- **Verdict: SKIP.** Kinetic's own transcript statistics are worth citing: never-cleared levels spent 76% of calls without acting, against 55% for cleared ones.

### W10. Logprob uncertainty feedback ("where you were least certain")
- **Source:** `W/agent/introspection.py:140-190`.
- **Evidence:** T1: shown on 97% of turns and referenced in 0 of 1,442 reasoning blocks. This settles the arms.md "did R0 ever run" question: it ran and was ignored.
- **Verdict: SKIP.**

---

## 3. Perception: harness-computed facts in the newest turn

### M1. Game-over run-length fact line plus a `game_overs` sandbox variable
- **Sources:**
  - Han Solo `HS/agent/tool_agent.py:151-165` and `HS/agent/prompts.py:76-77`: "Game overs on this level so far: {n}. Each ended a run of this many executed actions: {a, b, ...}. The newest one's board is in `game_overs`."
  - The same line as kinetic's `ARC3_X_GAMEOVER_MEMORY` (`K/kinetic-learning-by-playing-v8 c3`, off in their run).
- **What:** equal run lengths point to a fixed step budget, different lengths to a board hazard. The harness states the fact and lets the model infer.
- **Evidence:** T2, shipped and unmeasured. T1 motivation: died attempts hold 14.5% of actions (failure taxonomy).
- **Safety:**
  - **Flag:** yes.
  - **Cache:** tail.
  - **Tok:** about 25–50, only on levels with deaths.
  - **Neg:** Franzen's death-ledger *guard* had "no clear benefit". This is a fact line, not a guard.
- **Overlap:** #60 budget-alert (bar deaths) and memory-v2 `<hazard>` lines (#64). Check for duplication with both.
- **Verdict: TRY (high).**

### M2. Fatal-transition text diff
- **Source:** Tycho `T/agent/agent.py:1521-1538` ("Diff from pre-fatal frame to observed GAME_OVER frame").
- **What:** reuse M3's formatter inside Franzen's existing, off `ARC3_GAMEOVER_DIFF`.
- **Evidence:** none.
- **Safety:** tail; about 50–150 tokens per death.
- **Verdict: TRY (rides M3).**

### M3. Exact per-turn change text: one carrier, in Tycho's lossless run-length format
- **Sources:**
  - Tycho `T/workspace/workspace.py:97-124` and `T/prompts/actor.user.j2:42-45`. Example:
    ```
    34 cells changed; region [rows 0-19, cols 8-59]
      9->12  rows 16-19, cols 28-31
    ```
  - Han Solo's region line, `HS/agent/tool_agent.py:661-684`: "those actions changed {c} of {t} board cells. The change spans {k} connected regions: {bbox: count, ...}", with net-zero and animation-only variants.
- **Evidence:**
  - **T1 adoption:** given a choice, our model turned off grid text but kept this diff in 9 of 10 Tycho games (130/154 openers; median 183 chars).
  - **T1 GB10:** harness-computed "what moved" text cleared ar25 L2 in 5 of 5 arm runs vs 2 of 8 controls (pooled post hoc, p≈0.03, one game).
  - **Against:** rellik13's per-action "what changed" notes didn't help (T2), and Franzen's auto-inserted animation *timeline* gave no clear gain (T2, a different feature). No score A/B exists anywhere. Franzen's own `ARC3_AUTO_FRAME_DIFF` is off and never evaluated.
- **Safety:**
  - **Flag:** yes, `off|runs|summary`.
  - **Cache:** tail.
  - **Tok:** about 50 median, capped about 100.
  - **Neg:** no (facts).
- **Overlap:** sandbox-v3 `ARC3_SANDBOX_ECHO_DELTA` (rank 2), `ARC3_AUTO_FRAME_DIFF`, backlog #3/#4 (numpy-perception and frame-diff text), the diff image. **Pick one carrier per arm;** stacking duplicates text.
- **Verdict: TRY (high).** It is the only lever with repeated positive *process* evidence on this model.

### M4. Animation gate: Tycho's eventful-animation classifier
- **Source:** `T/harness/animation_evidence.py:255-380`. Categories include `retracting_transient`, `viewport_or_reveal` and `nonredundant_intermediate`. Han Solo's typed `blink|tween|path` summary is the alternative (`HS/agent/animation.py:37, 163`).
- **Evidence:** T1: it passed only 10 animations across 10 games, so it is selective. No score.
- **Safety:** CPU only; a one-line hint when eventful.
- **Neg:** only if it becomes images or a long timeline. Don't port the LLM captioner.
- **Overlap:** animation-combo-design L1/L3 (PR #40 rebase). This is the missing "novel" rule for L3.
- **Verdict: TRY** (inside anim-combo, not separately).

### M5. Mid-action frame images (rellik13 M76) and contact sheets (Scott Le Grand)
- **Sources:** `R/cells/29_*.py:42-201` (≤4 frames, chosen by information lost); `K/taaf-flashnext-sheetu12b-0922 c10`.
- **Evidence:** T2, bundled. Franzen: composite animation images "hurt some games".
- **Tok:** at our 10×, about 400 per image.
- **Verdict: SKIP** (at most anim-combo L4, last).

### M6. Untried actions and objects on this level; the last-click target
- **Sources:**
  - Scott Le Grand agentfix F7 "UNTESTED on this level: ..." (`K/taaf-flashnext-sheetu12b-0922 c10`; off in his runs).
  - Fable "Valid but not pressed on this level" (`F/fable_layer_core.py:433-439`).
  - dream-team critic "under-tested available actions" (`C/dream-team/.../critic.jinja:57`) and click marker (`config.py:67`).
  - StochasticGoose novelty bias (`C/ARC3-solution/custom_agents/action.py:246-253`).
- **Evidence:** none at T1–T2 (F7 was off). T4 inspiration. Gedouluhui's click-candidate hints gave +28% on a ~1.4 base (T2/T3, uncontrolled).
- **Safety:** tail; about 15 tokens.
- **Overlap:** memory-v2 already prints untried actions per seen board and untried colours per level.
- **Verdict: TRY, low** (inside #64 if anything).

### M7. Edge-triggered no-progress fact
- **Source:** win-v2 `K/arc-agi3-win-v2 c7, c13`: after K=15 consecutive executed actions with `gameplay_changed == false` on a level, one line in the next opener only, re-armed by progress.
- **Evidence:** none.
- **What to keep:** the factual half only, "15 actions in a row on this level changed nothing inside the board area". Drop "Do not repeat the same plan."
- **Tok:** about 25 per streak.
- **Neg:** audit-like, but rare and factual.
- **Verdict: TRY, low.**

### M8. Never strip all history images
- **Source:** Scott Le Grand's agentfix notes (`K/taaf-flashnext-sheetu12b-0922 c10`; `K/arc3-b32-reap-v030-kv14-c12-ctx48-r1 c18`).
- **Evidence (T2, Duck + Flash-Next):** stripping all history images gave +51% reasoning per request and −29% actions per 2 h; image-blind scored 0.44 vs 1.11.
- **What it means for us:** if `ARC3_HISTORY_IMAGE_KEEP` (issue #19) is ever tried, keep ≥2 and strip only at trim time. polyphony strips every request (`C/polyphony-arc-3/compat/qwen_policy.py:46-153`), which is cache-hostile.
- **Verdict: guidance, not an arm.**

---

## 4. Tool and sandbox design

### S1. Segmentation and frame-view API additions, plus errors that name the right field
- **Sources:**
  - Han Solo `HS/agent/python_tool_sandbox.py:184-196` and `HS/agent/prompts.py:85-87, 100-101`: node `bbox`, `mask` (`#`/`.` rows, None above 400 cells), `pixel_count`; frame `.rows` and `.crop(r0,c0,r1,c1)` with a ruler.
  - rellik13's Wang sandbox (`R/cells/40_*.py:13`, sandbox source 150-182): `bbox`/`area`, and `KeyError`s that list the valid fields.
- **Evidence:**
  - **T2:** "198 code-error recoveries in 25 games" from missing fields (rellik13 notebook md cell 26).
  - **T1:** "bbox" appears 6,791 times in r2 transcripts (the model rebuilds it from `boundary`). Type/Index/Value/Key/Syntax errors are 627 errors and 2.7% of tokens; 28 of 83 KeyErrors are harness dict keys (failure taxonomy).
- **Safety:**
  - **Flag:** yes.
  - **Cache:** static (one line in the segmentation bullet and the tool description).
  - **Tok:** about 60 cached; saves generated tokens.
  - **Neg:** no.
- **Overlap:** sandbox-v3 rank 1 (`ARC3_SANDBOX_ERRORS`) and rank 3 (`ARC3_SANDBOX_HELPERS`), #76, issue #20, backlog `api-schema-errors`. **New detail:** put `bbox` on the segmentation nodes themselves rather than as a helper.
- **Verdict: ADOPT** (into sandbox-v3).

### S2. Model-chosen batch stop: `action(actions, until=...)`
- **Source:** Wang `final-c1` HEAD `2a92428`, `W/agent/prompts.py:168-181` and `python_tool_sandbox.py:736-760, 1059ff`.
- **What:** `until='unchanged'` would key on `gameplay_changed`; `until=fn` is the general form.
- **Evidence:** never run. The related "batch 5–20 actions" advice lost on T1 (spend-actions 10.92 vs 13.65).
- **Safety:**
  - **Flag:** yes.
  - **Cache:** static (about 60 tokens in the tool doc).
  - **Neg:** it is the model's own stop, not a harness block. Drop the advice text.
- **Verdict: TRY, low** (screen adoption: count `until=` uses on a panel).

### S3. Small sandbox items
- **Safe click pixel per segmentation node.** Polyphony `object_tools.py:133, 169-180`: the member cell nearest the centroid, so clicks on hollow shapes hit the object. No evidence; 0 tokens. **TRY, low.**
- **Fixed-cell histories helper.** prime-agent `AGENTS.md:7`: per-cell change counts this level, showing static scene vs HUD. No evidence. **TRY, low.**
- **Signature + first-docstring catalog of retained functions.** Han Solo `HS/agent/tool_agent.py:168-218`. Ours lists signatures only. **TRY, low.**
- **Tool results omit fields the turn message already reports.** Han Solo `HS/agent/tool_agent.py:107-114, 985-994`. Our action echo repeats `valid_actions`, `state` and the clock fields in every result (the clock goes under #74). −30–60 tokens per result. **TRY, low.**
- **`memo` dict (kinetic `ARC3_X_MEMO`) and `store` (Wang).** Covered by #76 and sandbox-v3 rank 4. **SKIP (covered).**
- **Sandbox pipes closed after exit.** DF-WM `python_tool_sandbox.py` `_wait_for_process_exit`. Ours never calls `close()` on the Popen streams (`M/agent/python_tool_sandbox.py:791-803, 820-890`). CPython's refcounting probably closes them when the function returns, so this is likely benign. **TRY as a check:** count open fds in one long local loop. Fold into sandbox-v3 `HARDEN` if it leaks.

---

## 5. Guards

### G1. Extra guards
- **Sources:** guards from level 1 plus repeat guards (m2-top, sovereign); a bootstrap single-action phase (prime-agent `game_protocol_guard.py:59-69`); a hard batch cap of 20 (prime-agent); a batch-size cap while a rule is unconfirmed (Fable).
- **Evidence:**
  - **T2:** Franzen found extra guards gave "no clear benefit".
  - **T2 (Scott Le Grand, Duck + Flash-Next):** guard refusals "made cn04/g50t misread the harness as the game", so he switched to execute-and-tag.
- **Verdict: SKIP.**

### G2. Model RESET allowed but not advertised
- **Source:** rellik13 M99, `R/cells/40_*.py:357-401`.
- **What:** RESET is refused inside a batch or straight after another RESET.
- **Evidence:** T2: 21.5% of analysis steps came after the model itself declared the level stuck; no score attribution. Han Solo removed model RESET entirely; dream-team has "RESET discipline" (T4).
- **Safety:** ours has `EXPOSE_RESET` (off). 0 tokens if not advertised.
- **Verdict: TRY, low.**

---

## 6. Scheduling and turn budget

- **Han Solo time-based yield** (`HS/agent/tool_agent.py:117, 1474-1479`; 150 s × min(1 + consecutive yields, 3)) and **rellik13 M44a, no yield and unlimited calls** (`R/cells/25_*.py:8-32`; "49.1% of calls execute no action; 88% of those ... cut off by YIELD_SECONDS=60", T2).
  - Already backlog `yield-time` / `yield-6k` (T1, corrected in issue #130: a yield's own opener is about 5% of prefill, and yield-6k removes about 3%).
  - Franzen lists his 2,048-token yield as worked, so this reverses one of his findings.
  - **TRY** (already queued).
- **Han Solo drain to 81.5k** (`HS/agent/tool_agent.py:1291-1367`; HSETUP:828). Already backlog `drain-80k`. **TRY.**
- **UCB / one-pool UCB** (rellik13 M72, Wang): T1 43.84. **SKIP.**
- **D′** (affectify, arc3-dprime-guarded, amatlas): T1 −7.5. **SKIP.**
- **Lottery scheduler** (duck-plus-ours): no evidence; scheduler sims mislead. **SKIP.**
- **Strategy audit** (Han Solo `HS/framework/solver.py:239-250`) and **escalating stuck prompts / fresh restarts** (polyphony `agent.py:404-414`, m2-top). T1 audit60k 44.10; rellik13 lists restarts as a failure. **SKIP.**
- **Per-game token cap** (sovereign: `max_generated_tokens_per_game=750000`) and **per-level action caps** (polyphony 750). Franzen's C factor already deprioritises stalled levels. **SKIP.**
- **Points per action by actions spent on the level** (rellik13 `R/cells/21_*.py:409-419`; decay ≈ actions^-1.73 over 2,443 attempts, T2). A **prior for sim-v2**, not an arm.

## 7. Serving and throughput (the owner's "win 2 more games via throughput" goal)

### V1. ARC-tuned speculative vocabulary
- **Source:** Han Solo `C/kaggle-datasets/lordhansolo-taaf-kaggle-source/src/ARC3-Inference/configs/draft_vocab_32k.json`, HSETUP:62. It holds 23,853 token ids actually generated in 125 Flash-Next ARC transcripts, padded to 32k. Ours uses the generic 64k FR-Spec hot map.
- **Evidence:** T2: acceptance about 2.9 vs Franzen's 2.7, but on a different stack (vLLM, NVFP4/FP8).
- **What:** build the FR-Spec map from our own request logs.
- **Safety:** lossless (verification is unchanged); serving-only; profile-scoped.
- **Verdict: TRY** (bench C-arm in `serving-bench`, judged on serve_report tok/s and accept length). Sim-v2: about +2.2–2.8 points per +10% throughput, an upper bound.

### V2. KV-room levers
- **Sources:**
  - Han Solo `--cpu-offload-params embed_tokens` (HSETUP:531, 589-590), +1.18 GiB KV.
  - Capping vision-encoder profiling at the grid image's pixel area (HSETUP:80, 591-592), "the whole 2.19 GiB activation peak, which the KV pool gives up" on vLLM.
  - rellik13 48 GB hicache host tier plus FP8 PLE (`R/cells/11_*.py:179-197`; `R/cells/10_*.py:71-108`).
- **Evidence:** T2, mostly concurrency-driven (1,135 / 1,159 tok/s at 14–16 streams). Against: our streams14 gave +15% tok/s for +0.13 points (T1). Hicache could still cut our yield and trim re-prefills (trim re-prefills mostly; a yield's own opener is only about 5% of prefill, issue #130).
- **Verdict: TRY** (backlog `kv-pool-audit`): first check whether Pennyroyal reserves for the maximum image size and supports embed offload, then bench.

### V3. Concurrency (14–16 streams)
- **Evidence:** T1: streams14 46.38 (noise) and ledger-streams12 48.08. **SKIP** (already tested).

## 8. Robustness (no score claim; protects a rerun from a 0)

### R1. Reasoning-render boot check
- **Source:** rellik13 `R/cells/11_*.py:452-468`. At boot, send the same assistant message with and without `reasoning_content` and require a difference of ≥200 tokens; otherwise abort.
- **Why:** his v14.57 rendered every past think block empty without anyone noticing. Franzen: removing thinking from history was "a large regression".
- **Verdict: ADOPT** (into serve_report or notebook startup; 0 tokens).

### R2. Server supervision
- Han Solo's watchdog, restarting up to 5 times (HSETUP:47).
- sovereign's relaunch without MTP when the server dies (`K/arc-prize-2026-sovereign-agent c12`).
- sigeward's graceful spec-asset fallback (`K/sigeward-milestone-2-solution c13`).
- action7-shadow's owned-process watchdog (`K/arc-agi-3-stock-taaf-action7-shadow c19`).
- These cover issue #26. **ADOPT** (insurance, default-off in Save & Run).

### R3. Soft end before the 9 h kill, and auto-reset retries
- Soft end: sigeward 120 s (`c21`), DF-WM 900 s (`K/df-wm c20`). This is backlog `rerun-soft-end` and issue #27.
- Auto-reset retries: Han Solo `HS/framework/solver.py:885-920` ("Retrying is never worse than giving up"); rellik13 M102 retry 4× with backoff. These cover issue #29.
- **ADOPT** (insurance).

### R4. Lesser robustness items
- Progress-aware SSE transport (action7-shadow inactive cells 49–56), solver-side `/v1/tokenize` counts (rellik13 M79) and emergency context repair (rellik13 `R/cells/36`).
- Ours already has calibrated estimates and overflow recovery.
- **SKIP / low.**

---

## 9. Already covered or in flight (no new detail)

| Idea seen elsewhere | Covered by |
|---|---|
| Hide the run clock (Han Solo, Fable) | PR #74 |
| Persist sandbox variables (kinetic memo, Wang store, rellik13 M44b) | PR #76, sandbox-v3 rank 4 |
| Indicator / far-change notes, step-bar alert (Fable) | PRs #59, #60, #62 |
| Solved-level facts, action effects, seen-state, hazards (rellik13 M85 facts half, Fable LEVEL LOG, sheetu12b F7) | memory-v2 #64 |
| Shortest route on record | level-path #66 |
| Memory written through a tool parameter (Tycho beliefs file, 7/10 games, T1) | memory-xml #65 (supporting evidence) |
| Ledger shown after trims (Wang memory ledger) | ledger-on-trim #45 / stack-v2 #56 |
| Animation ring, digest, truthful verdicts (rellik13 M111, Han Solo classifier, Wang frames) | animation-combo-design / PR #40 |
| Static reasoning effort (kinetic `ARC3_X_REASONING_EFFORT`, Han Solo xhigh) | PR #73 |
| Incremental state file (rellik13 M86) | PR #43 (not merged) |
| Retained-code memory (rellik13 M59b, not installed in his run) | Franzen's persistent functions, #51 |
| Level-start inventory (polyphony `on_new_level_v1.txt`) | Franzen `ARC3_LEVEL_INVENTORY` (off; the harness-fact version) |

**Open PRs, and how this harvest bears on each:**

| PR | Arm | Ideas here that extend or overlap it |
|---|---|---|
| #40 | animation-digest | M4 (Tycho's eventful gate as the "novel" rule for L3), M5 (skip images) |
| #45 / #56 | ledger-on-trim / stack-v2 | P1 (point the death text at a refuted tag), W1 (rule shown at trim), W5 |
| #57 | attention-prompt (`Salient:` facts) | M6 (untried actions/objects would join the same facts line, not a new line) |
| #59 / #60 / #62 | indicator-notes / budget-alert / facts-stack | M1 and M2 (game-over facts sit next to the step-bar line; check for duplication), P3 (actions since level start) |
| #64 | memory-v2 | W1 (adds the rule ask/capture), W5, W6 (no-op caveat wording), M6 |
| #65 | memory-xml | W3 (graded prediction as hypothesis status), W4 (`plan`), Tycho beliefs-file adoption (7/10 games) as supporting evidence |
| #66 | level-path | W5 (ruled-out routes) |
| #67 | memory-v3 | if W1 and W3 land, they stack on #67 rather than on #64 alone |
| #72 | reasoning-scaffold | the closest open arm to the thinking-guide negative (W7). This harvest found no evidence for or against a ~10-token prefill. |
| #73 | reasoning-effort | Han Solo sends `reasoning_effort=xhigh` per request, which is the template default; kinetic's `ARC3_X_REASONING_EFFORT` is the same idea, off in their run |
| #74 | budget-truth | P2 (finish the clause deletion, Han Solo's sentence, Tycho's HUD generalisation), P1 rides along |
| #75 | cutoff-signal | P3 (same "tell the model what the harness did" family) |
| #76 | sandbox-robust | S1 (on top of it via sandbox-v3), kinetic memo / Wang store (covered) |

Not resolvable from what we have:
- **nihilisticneuralnet "retained reasoning":** its code is in the dataset `nihilisticneuralnet/120926arc`, which isn't pulled. Franzen already replays reasoning.
- **richardcsaky arena hero v2:** a sha-pinned tarball, not local. The #6 LB position is tied to his GPTQ-ARC expert overlay (backlog `gptq-arc-experts`).

---

## 10. Ranked top 10 to build next

Ranked mainly on T1–T2 evidence and safety, most confident first. Every item is default-off and prefix-cache safe as designed. The cost is per turn, unless stated otherwise.

| # | Build | Verdict | Evidence (tier) | Cost | Extends |
|---|---|---|---|---|---|
| 1 | **Prompt-honesty v2.** P1 (fix the two dangling "record ... in your world model" lines; with the ledger, point them at a `<hypothesis status="refuted">` tag), P2 (delete the whole "only HUD elements such as countdown bars" clause; Han Solo's budget sentence; Tycho's HUD-as-state generalisation) and P3 (state that acting ends the turn; actions since level start). | **ADOPT** | T1: bar called a timer in 12/25 games per run. T2: Franzen's dangling world-model line 76% → 43% replies with text; rellik13 fixes bundle +1.09 | static, net ≈0; tail +8 | PR #74 |
| 2 | **Solved-level rule capture** (`ARC3_MEM_SOLVED_RULE`): a one-off "Rule for level N:" ask in the level-start opener, captured from text or reasoning, shown via memory-v2's `<solved>` record at trim and the next level start, plus the winning-transition region | **ADOPT** | T2: +2.8 single submission; 27/28 compliance; 0/989 cross-level notes without it. T1 mechanism: ledger-on-trim +4.6 mean | ≈70 tok × 1–2 per level | #64 (+#61 guard) |
| 3 | **Segmentation API**: node `bbox`/`mask`, `.crop()`, KeyErrors naming valid fields | **ADOPT** | T1: "bbox" 6,791× in r2; 627 errors = 2.7% of tokens. T2: 198 recoveries | ≈60 cached; saves output | sandbox-v3 ranks 1/3, #76 |
| 4 | **Run insurance**: reasoning-render boot check (R1), server supervision with no-MTP fallback (R2), soft end + auto-reset retries (R3) | **ADOPT** (no score claim) | T2: silent empty-reasoning regression (rellik13); our own OOM and hang history | 0 | issues #26, #27, #29 |
| 5 | **Game-over facts**: run-length line + `game_overs` variable (M1), Tycho-format fatal diff in `ARC3_GAMEOVER_DIFF` (M2) | TRY (high) | T2 shipped (Han Solo), unmeasured. T1: died attempts hold 14.5% of actions | 25–150 tok after deaths only | #60, #64 |
| 6 | **Exact change text, one carrier** (M3: Tycho run-length format, capped ≈100 tok) | TRY (high) | T1: kept in 9/10 games when optional; GB10 5/5 vs 2/8 (one game). No score A/B | ≈50 tok | sandbox-v3 ECHO_DELTA, `ARC3_AUTO_FRAME_DIFF`; backlog #3/#4 |
| 7 | **Leaner openers** (P6: drop static lines already in the system prompt from new openers) | TRY | T1: ≈500 tok × median 15 copies ≈10% of context; trims are 48% of uncached prefill. T2: bundle +3.44. Against: Franzen's shorter resumes "no consistent gain" | −500 tok | issues #17, #21 |
| 8 | **ARC-tuned FR-Spec map** from our transcripts (V1), plus a KV-room audit (V2) | TRY (bench-gated, lossless) | T2: acceptance 2.9 vs 2.7 (other stack). Sim-v2: +2.2–2.8 points per +10% tok/s (upper bound) | 0 | serving-bench #50, backlog `kv-pool-audit` |
| 9 | **Wang `GAME_PRIORS`** static addendum (W2), optionally with the one-sentence solvability prior | TRY | T2: inside rellik13's +3.44 bundle; T1: on in every Wang run, never isolated | ≈340 cached, 0 per turn | none |
| 10 | **Harness-graded `prediction` parameter** (W3), as the verdict source for memory-xml hypothesis items | TRY | T1 GB10: 93% compliance, levels tied control; the only predict-verify loop this model fills | ≈220 cached + ≈40 per acting turn | #65 |

**How to run them, given Kaggle quota.**
- **Batch A (prompt text only):** items 1 + 2 + 3, built on `stack-v2` so they inherit the ledger. All three target "true facts and memory the model will actually use", and none adds per-turn text. A loss sends us to the single arms.
- **Separate arms:** items 5 + 6 as a "facts" arm against `facts-stack` (#62), item 7 alone (its mechanism is throughput), and items 9 and 10 alone (each has a negative prior).
- **Merge without a score A/B:** item 4 is insurance. Item 8 is gated by a bench.
- **Judging:** every score claim needs ≥2 paired full 121-min runs and `make significance` (owner's rule). A 20-min panel is a screen only.

**"World model"-like elements for the owner's prompt review.** Ranked by how likely this model is to use them:
1. Harness facts at the moment they matter: items 2, 5 and 6.
2. A tool parameter the harness grades: item 10.
3. A one-off ask at a defining moment: the level-clear rule in item 2, and the refutation pointer at a death in item 1.
4. Static priors framed as hypotheses: item 9.

Per-turn world-model sections, builders, simulators and protocols all measured at about zero uptake on this model (§2 W7, W8, W10).

## Side effects to clean up

The Fable/Tycho/Wang research pass rendered Tycho's templates with `uv run` inside `tycho-port/`. That created an untracked `tycho-port/uv.lock` and re-synced `tycho-port/.venv` (4 packages removed, 4 added). The permission check blocked deleting `uv.lock`, so the owner should remove it. If the dev/test extras are needed, re-sync the venv.
