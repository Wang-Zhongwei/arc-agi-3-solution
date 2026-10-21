Digest of ARC-AGI-3_handbook_en.pdf, 90 pages, read in full. All text was extracted with `pdftotext`. Figure pages 32–40, 58 and 79–82 were also read as images through the Read tool. The page-tagged text is at `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare/pages.txt`.

Tags:
- **[V]** verified: I checked it against our code or files, or redid the arithmetic.
- **[T]** their claim, unverified: a figure from their own Kaggle logs, cited by PDF page ("p.").
- **[I]** inferred by me.

"nb cell N" means a cell of the THEIRS notebook, as tagged in `narrative.md`. Their run names:
- **B003** is their unchanged Franzen baseline: 15 hard public games, 72 min per game, 10 slots.
- **B005** is the same with the "journal replaces history" change plus a budget line.

## 1. Quantified findings

**Setup and scoring**
- Budget per game is about 181k generated tokens (626 tok/s × 31,800 s ÷ 110 games). p.4 and nb cell 10 [T]. The arithmetic gives 180,971 [V].
- A copy of Franzen's notebook scored 31.73 on the leaderboard. Resubmitting the same notebook moves it about ±4 points. p.5 [T]
- B003 scored 28.65 with 46 of 111 levels. Another participant's run on the same games scored 29.1 with 49 levels. p.5 and p.80 [T]
- Scoring rule: a level scores min(1.15, (h/a)²), weighted by level index, and the game score is capped by the cleared weight share. Twice the human action count earns 25% of the level. UNDO counts as a scored action. nb cell 7 [T]
- Token-score elasticity, from 13 points in the "sovereign agent" notebook: above 1.5 below about 70k tokens per game, and 0.27 / 0.24 near the real budget. nb cells 12–13 [T]

**Chapter 1 (sk48 level 1, B003)**
- The system prompt is 18,636 characters, about 5.5k tokens. The first request was 5,490 tokens, of which the frame image was 402. Answers average about 2k tokens (100 to 12k on this level). p.12 [T]
- Run totals: 660 turns, 1,362 model calls, about 2 calls and 4 actions per turn. p.12 and p.47 [T]
- sk48 waited 17 minutes for its first slot. After a trim (75 → 24 messages) one step waited 29 minutes. p.14, p.16 [T]
- Result: 68 actions, 33 calls, 101,551 tokens, level not cleared. The model was writing text about 29% of its 72 minutes. p.13, p.16 [T]
- The model read the budget bar as a real-time clock because `time_remaining_seconds` is in every action result. Time words appear in its reasoning in 11 of 15 games. p.15 [T]
- Ten copies of sk48 level 1, 48 min each, no queue: 9 of 10 cleared it, using 21k to 136k tokens. The tenth used 170k and failed. Mean 1.5 levels per copy. p.17 [T]

**Chapter 2 (Fable on the same level)**
- They report 97.4% on the public games (better of two runs) for $1,750 across 50 runs of 25 games, and 18 percentage points lower without the game log. p.18 [T]
- sk48: all 8 of 8 levels in 650 actions. Level 1 took 32 actions against a human baseline of 61. p.18 [T]
- Actions per round: 1, 3, 3, 6, 1, 3, 3, 3, 6, 3. Each round makes 4 to 7 tool calls. Fable has no token or time limit (2,400 s per call) and a cap of 500 actions per game. p.19, p.21 [T]

**Chapter 3 (learning from traces)**
- Why the baseline got stuck, across 13 levels read by hand:
  - The goal hypothesis was not revisited (sk48, bp35).
  - A rule was inferred from a few observations and never tested (sk48, ka59, lf52).
  - The budget bar was noticed late (tn36, wa30, su15).
  - The model's own world model was buggy (ls20, sp80).
  - Geometry confusion (bp35).
  - p.26 [T]
- 15 of 39 tool errors were "name not found", because sandbox variables do not survive between calls. p.28, p.51 [T]
- Fisher test on 10 vs 10 copies: 0 vs 5 gives p≈0.03, 2 vs 8 p≈0.02, 1 vs 6 p≈0.06, 2 vs 5 p≈0.35. p.31 [T]

**Chapter 4 (time and tokens)**
- Server load averaged 9.1 of 10 slots: 10 busy 46% of the time, 9 busy 32%, 8 busy 14%. p.32 [T]
- Throughput by concurrency: 582 / 691 / 737 / 773 / 798 tok/s at 6 / 7 / 8 / 9 / 10 requests. Keeping all 10 busy would add 3%; going from 8 to 10 slots adds 8%. p.32–33 [T]
- The run generated 2,695,985 tokens, which they give as 618 tok/s. The server log shows 770–800 tok/s at 9–10 requests. p.33 [T]
- Per game: 5 min (7%) waiting for the first slot, 38 min (53%) generating, 28 min (39%) other. Tokens per game ranged from 102k (sk48) to 277k (dc22). p.34 [T]
- Trims: 53 in total, 46 of them between turns. Waiting after those 46 trims was at least 275 of 1,080 game-minutes (25%). 13 trims caused waits of 4 minutes or more, up to 40. p.35 [T]
  - Stuck games waited longest: g50t 40, wa30 34, su15 30, sk48 29 minutes. Games making progress waited 1–4 minutes in total.
  - Stuck games received 102k–158k tokens; progressing games 224k–277k.
- Idle slots were 8.6% of the time. The server read 6.1M new prompt tokens against 2.7M generated, with 93.8% of prompt tokens served from cache. p.36 [T]
- Removing every pause would give about +3–10% tokens, which is under 1 leaderboard point (+10% tokens ≈ +1 point). p.36 [T]
- 31% of all tokens went to levels never cleared; in stuck games 56–100%. p.37 [T]
- Clear rate by human solution length, over 114 levels: ≤30 actions 31/31; 31–60 87% (n=38); 61–100 68% (22); 101–150 54% (13); >150 60% (10). Level 1 is cleared 96% of the time. p.38 [T]
- How well each feature separates cleared from uncleared levels (AUC): human solution length 0.80, Fable's action count 0.78, level number 0.71. p.38 [T]
- sirikilohit's changes, each measured once: +2.8 model/tuning, +2.6 SGLang plus more parallel games, +0.7 larger context, +2.8 memory of cleared levels, +3.4 Wang's design, +1.1 harness fixes, +8.0 FP8 cache plus longer history. His noise: the same notebook scored 5.02 / 5.50 / 6.42. Fig 4.6, p.39 [T]

**Chapters 6–7 (what the model reads and sees)**
- Tool output is capped at about 3,072 tokens (keeps 2/3 head and 1/3 tail); 30 s per call. p.50, p.52 [T]
- A frame image costs 402 tokens. Two images per turn add about 800 tokens. Images are never removed from history because `ARC3_HISTORY_IMAGE_KEEP` is unset. p.56 [T]
- What the model's code used, out of 1,334 calls: letters 24% (321), regions 32% (421), `frame_diff` 10% (139), animation 8% (112). p.53–57 [T]
- These settings are off in their run: `ARC3_AUTO_FRAME_DIFF`, `ARC3_ANIMATION_IMAGE`, `ARC3_MEMORY_SECTIONS`, `ARC3_PREFER_TOOL_CALLS`, `ARC3_OPENER_COMMIT_HINT`. p.50, p.57 [T]

**Chapter 8 (memory)**
- Window: 131,072 − 12,288 output − 512 margin = 118,272 tokens maximum per request. A trim drops history to about 58,880 tokens (`ARC3_CONTEXT_DRAIN_TOKENS` = 59,392). p.58–59 [T]
- sk48 window size: 13k at request 5, 44k at request 15, 107k at request 28, then 52k after the trim. p.59 [T]
- The model's own reasoning fills 54% of the window before a trim (58k of 107k) and 63% after it, because `preserve_thinking` is on. p.59 [T]
- 38 images had piled up by the trim: about 15k tokens, 14% of the window. Code results took about 9k. p.59 [T]
- Trims happen 3–4 times per game. The first comes at turn 11–32 (median 18). The 150-assistant-turn cap is never reached. p.59 [T]

**Chapter 9 (scheduling)**
- Priority = (A + B) × C:
  - A = level index × (25/(25+a))², where a is actions spent on the level.
  - B = 8 / 7 / 5 / 0 depending on levels remaining.
  - C = 0.25·0.5^((a/115)²) + 0.75·0.5^((t/62000)²), where t is tokens spent on the level.
  - The value of future levels fades over the last 20% of game time.
  - p.63 [T]. The 115 and 62000 constants match our code: `tool_agent.py:2173,2177` and `priority_scheduler.py:83` [V].
- C falls to 0.60 at 60k tokens, 0.27 at 100k, 0.09 at 150k, 0.03 at 200k, assuming about 1,030 tokens per action. p.64 [T]
- First-slot waits: sk48 17, tn36 16, m0r0 16, bp35 15, cn04 12 minutes. p.62 [T]
- There was never a run without priorities, so whether they help is unknown. p.64–65 [T]

**Chapter 10 (guards)**
- Guards fired rarely: 4 batch no-op stops, 42 stale-state blocks (27 of them in lf52), 2 terminal-state errors. p.66–67 [T]
- The no-op-repeat and death-repeat guards are off. p.67 [T]
- Action mix out of 2,624: mouse 804, right 464, left 409, down 400, up 393, space 127, undo 20 (0.8%), reset 7. p.68 [T]

**Chapter 11 (the journal design)**
- The journal triggers at 30k tokens on a level: 9 of 10 cleared levels in easy games fit within 30k, while a cleared hard level averages about 58k. It is limited to 4,096 tokens and skipped if under 500 characters. p.73–74 [T]
- Their budget-bar detector found the bar in 304 of 316 checks across the 25 public games. p.75 [T]

**Chapter 12 (the first test, B003 vs B005)**
- Score 28.65 → 10.49. Levels 46 → 23. Actions 2,624 → 1,467. Tokens 2.70M → 2.56M. p.78 [T]
- Up to the first journal the runs match (440 vs 488 actions). After it, 1,027 vs 2,136. p.79 [T]
- Level 2 cleared in 6 of 13 games vs 11 of 14 (p≈0.12). Level 3 in 2 of 6 vs 9 of 11 (p≈0.11). p.79 [T]
- Turns after a journal look normal: 3.9 vs 4.5 actions per turn, 11.7k vs 11.8k characters of reasoning. p.80 [T]
- There were 65 journals of about 1,400 tokens each. All 27 pauses over 4 minutes came right after a journal: 323 of 1,084 game-minutes (30%). The baseline had 10 such pauses, 187 of 1,043 minutes (18%). p.80–81 [T]
- Levels that passed 60k tokens: 11 of 18 cleared vs 2 of 17 (p≈0.005). Share of tokens on never-cleared levels rose from 31% to 68%. p.81–82 [T]
- On the 9 regressed levels the baseline used a median of 54k tokens to clear; the journal run used 129k without clearing. p.82 [T]
- Active minutes fell 11% (856 → 761), but levels per 100 active minutes fell 44% (5.4 → 3.0). So the queue explains only a small part of the loss. Pause minutes correlate with levels lost at r = 0.64. p.82 [T]
- The journal had 28 "action X has no effect" entries. In 15 cases the same action later changed the board; 13 were never retried; none was confirmed. Budget deaths were 6 vs 6. p.84 [T]
- In the baseline, 27 of 37 levels (73%) that passed 30k tokens were still cleared, and 11 of 18 (61%) that passed 60k. p.86 [T]
- sk48 ten-copy baseline: 9 of 10 cleared level 1, mean 1.5 levels, SD 0.97, level 2 cleared by 4 of 9. New success rule: mean ≥ 2.4 levels counts as a benefit, ≤ 0.6 as harm. p.86 [T]. The standard error of the difference is 0.43 [V].

**Notebook claims about their new layer**
- Lost marks correlate with level-1 tokens at ρ = 0.93 (p = 0.0004, n = 10). nb cell 25 [T]
- All 83 indicator events on sk48 were tray marks. Over 40 recorded runs: 202 events in 5,832 actions. nb cell 39 [T]
- In copy 9, UNDO was 127 of 317 actions, so the bar emptied at 0.19 cells per action against 0.32 per non-UNDO action. nb cells 33, 35 [T]
- With 10 copies the smallest detectable difference is 0.87 levels; with 30 it is 0.5. nb cell 38 [T]. 0.87 checks out [V].
- The log costs about 0.6–1k tokens. nb cell 30 [T]

## 2. Where Fable and the open harness separate on sk48 level 1

- **Same start:** Fable's rounds 1–7 (17 actions) match the baseline's turns 1–14 (about 8 minutes, 23 actions): both find the shooter, the beam, grab-and-pull, and the tray mark. The author calls this "not a difference". p.23 [T]
- **The split is Fable's round 8 against the model's turn 15 (minute 27).** Both see the tray's 2-column gaps.
  - The model reasons "exact tray spacing is not achievable → the tray is NOT the goal state". It then comes back to the right goal but builds a 35-action plan on an untested "blocked-pull" step, which it flags itself as "Risk". It never writes down "blocks link by a push" as a separate hypothesis, so it never tries the cheap probe. p.15–17, p.24 [T]
  - Fable writes "push red into green to link them" and tests it in 3 actions. p.23–24 [T]
- **The five habits:**
  1. Small probes. Fable sends 1–3 actions per round while a rule is unproven, 6 when several questions can be answered at once. The model sent 13 planned actions on turn 17. p.24 [T]
  2. Every round's note records the outcome, and wrong rules are corrected in place ("FAILED blue link… blocked press = link", round 10). p.23–24 [T]
  3. The budget is written down by round 3–5: "1 px per 3 actions, 64 px ≈ 192 actions". The baseline noticed tn36's bar only with 20 of 61 cells left. p.22, p.24 [T]
  4. The whole log sits outside the conversation, and Fable searches it with grep. p.24 [T]
  5. It reads the legend every round (rounds 4, 7, 9). nb cell 16 [T]
- The author's own reading: the error is in how hypotheses are handled, not in vision or code, and more tokens would not have fixed it. p.17 [I], theirs

## 3. Changes proposed or tested, with results

- **Tested:**
  - The journal that replaces history (P1) plus the budget line (P2): 46 → 23 levels, 28.65 → 10.49, harmful. They name three causes: the replacement counts as an eviction and gives up the slot; journal wording such as "REFUTED" and forced "ALTERNATIVES" turns state-specific results into general rules; and the model's own history is deleted. They did not separate the shares. p.78–84 [T]
  - The budget line on its own cannot be separated from the journal (budget deaths 6 vs 6). p.84 [T]
  - The Fable layer — indicator notes from 15k tokens, a pinned LEVEL LOG from 30k (refreshed every 40k), budget alert at ≤ 25% or ≤ 12 actions, five protocol lines — has only been checked with a scripted model: openers unchanged with the flag off, and the log comes back to the head after trims. **No score with the real model or on the leaderboard is reported anywhere.** nb cells 15, 39 [T]
- **Prepared but not run with the real model:**
  - Variant B: the journal is appended and pinned, and the slot is kept. p.29, p.76 [T]
  - A state-conditioned "level note" without the words "refuted" or "impossible", which lists each key's tried states. p.76, p.85 [T]
  - Hiding the game clock. p.31, p.76 [T]
  - Variant C, pinning the rules of cleared levels (a copy of sirikilohit's +2.8), is not implemented. p.30 [T]
- **Ideas, untested:**
  - Keep sandbox variables between calls (15 of 39 tool errors). p.28
  - A structural cap on batch size while a rule is unconfirmed. p.28
  - A harness-computed "key × board-state" line built from the no-op guard's state hash. p.57, p.86
  - Soften or remove the token-based demotion; first-come-first-served queue; turn on `ARC3_DIAG_CONCURRENCY=1`. p.65
  - Shorten the system prompt, and drop its contradictory "ignore the bar" line (p.49 against "count remaining budget"). p.28, p.49
  - Their estimated chance that the note works is about one third. p.86 [T]

## 4. Against our facts

- **Throughput gains did not convert to score: confirmed.**
  - The server is nearly saturated (9.1 of 10 slots); removing every pause gives +3–10% tokens, under 1 point; elasticity near the budget is 0.24–0.27. p.33, p.36, nb cell 13 [T]
  - This matches our streams14 run: +15% tok/s, 46.38 vs 46.25 (`arms.md:89`, `arms.md:255`) [V].
  - One caveat: sirikilohit's +2.6 from SGLang plus more parallel games, a single run. p.39 [T]
- **Per-game swings of 50–80 points: consistent, not measured directly.**
  - Ten sk48 copies spread level-1 cost from 21k to 170k tokens, mean 1.5 levels, SD 0.97. p.17, p.86 [T]
  - The aggregate still reproduces across runs (28.65 vs 29.1). p.80 [T]
  - They never report per-game swings in points; their answer to noise is copies of one level plus a minimum-detectable-effect calculation (nb cell 36). [I]
- **Context trims ~118k → ~59k: confirmed exactly.** 118,272 trigger, about 58,880 target (drain 59,392); sk48 went 107k → 52k. p.59 [T]. Our own docs say about 57Ki (`arms.md:37`) [V].
- **Slot handover only at trims: confirmed, with a nuance.** p.62 rule 3, p.81 [T]
  - Our code hands the slot over only when `_context_was_trimmed` is set: `tool_agent.py:6625-6629`, `6690-6700`. Line numbers are from the checked-out branch `arm/animation-digest` [V].
  - **Any** change to history sets that flag — turn-cap eviction, control pruning, removing images — not only the token trimmer (`tool_agent.py:6664-6667`) [V]. That is how journal replacement caused 27 long pauses (30% of game time, p.80).
  - Their measured cost of handover in a 15-game / 10-slot setup: at least 25% of game time waiting after trims, concentrated on stuck games. p.35 [T]
- **Ledger-on-trim +3 to +6: consistent; they never test it directly.**
  - Their design rules point the same way: add facts at the moment history is lost and never replace history. sirikilohit got +8.0 from more history and +2.8 from a pinned rule (p.39), and the replacement journal lost half the levels (p.78). [T]
  - Their LEVEL LOG is re-pinned at the head after trims — the same moment our ledger appears — but it has no score. nb cell 30 [T]
  - **Risk for our ledger:** model-written "no effect" or "refuted" entries were wrong 15 out of 15 times when retested (p.84 [T]). Ledger content should carry the board state or be computed by the harness. [I]
- **Model rarely writes ledger tags (2.7–3.9 per 100 calls): confirmed in direction.**
  - sirikilohit: "the agent did not use" his journal; notes "must be written by compulsion". p.28, p.39 [T]
  - `ARC3_MEMORY_SECTIONS` is off because section parsing failed. p.60 [T]
  - Their fix is a log the harness computes from the frames ("nothing in it is the model's own claim"). nb cell 30 [T]
  - This exact rate is not in `arms.md`; it comes from the task brief. [I]
- **14% of moves land on an already-seen board: no matching number.**
  - Related: 15 of 28 "no effect" claims later proved state-dependent (p.84); `ARC3_NOOP_REPEAT_GUARD` is off (p.67); they propose a harness-computed key × state line (p.57, p.86). [T]
  - UNDO was 0.8% of baseline actions but 127 of 317 in one sk48 copy. p.68, nb cell 33 [T]
- **46% of deaths intentional: no matching number.**
  - Related: B005 had 6 budget deaths vs 6 for B003 (p.84); after a death the harness auto-resets and sends a "BAR RULE" message (p.46); death guards are off (p.67). [T]
  - Their data neither supports nor contradicts the figure. [I]
- **Extra risk relevant to our 121-min offline benchmark** [I]: their 72-minute test showed real-clock pressure ("I should hurry" in 11 of 15 games), because the model mistakes `time_remaining_seconds` for the budget bar (p.15, p.31). Our 25-game / 121-min offline runs probably show the model the same field, which would make offline results less like the 532-minute hidden run.

**Bottom line:** I found no evidence that THEIRS beats our notebook.
- The handbook reports no score for the Fable layer — only scripted-model checks (nb cell 39). [T]
- Their only measured result for a change is negative (10.49 vs 28.65). p.78 [T]
- What they have that we may not:
  - The ten-copies-of-one-level test protocol.
  - The level log the harness computes and re-pins after trims.
  - Indicator notes and the budget alert.
  - Hiding the game clock.
  - A diagnosis of the turn-15 hypothesis error.
- Untested in their work — and whether we already have any of these I did not check [I]. Their 15-hard-game baseline is 28.65. p.5
