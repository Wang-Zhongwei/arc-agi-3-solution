**Data card: memory measurement on the Franzen Duck harness**

All numbers are measured. Scripts and cached JSON are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/memory-measure/` (`m1`–`m6*.py`, `parsed/`).

- **Data sources.** Only r2 (`fz-ledger-on-trim-r2`) has request logs and boards, so exact trims and board hashes are r2-only. r1 (`fz-ledger-on-trim`) and the control (`fz-arms-v4-control`) use transcripts plus `benchmark.json`.
- **Ledger replay.** I replayed the real `MemoryLedger` code on the transcripts. It reproduced the actual r2 ledger blocks exactly at 79 of 85 trims.
- **Trim prediction for r1 and control.** Trims are predicted from transcript text size. That prediction is within ±1 call of the real trim only about 55% of the time, so r1 and control post-trim numbers are rough.
- **Turn definition.** "Turn" below means one model call.

**(1) Ledger tags**

| | r1 | r2 |
|---|---|---|
| Model calls / actions | 2,414 / 5,278 | 2,246 / 5,654 |
| Tags absorbed | 66 | 88 |
| Tags per 100 calls | 2.7 | 3.9 |
| Tags per 100 actions | 1.25 | 1.56 |
| world_model per 100 calls | 1.12 | 2.63 |
| hypothesis per 100 calls | 0.95 | 0.62 |
| refuted hypotheses (count) | 4 | 2 |
| level_note per 100 calls | 0.66 | 0.67 |
| Calls with at least one tag | 22 (0.9%) | 48 (2.1%) |
| Games with zero tags | 18/25 | 16/25 |
| Tags written inside python code (lost) | 26 | 16 |

- **Skew.** tu93 in r2 alone wrote 40 of the 88 tags. Without it, r2 is 2.3 per 100 calls. In the 16 game-runs that wrote any tag, the median was 6 tags.
- **Writing channel.** Only 0.9% (r1) and 2.5% (r2) of calls contain any assistant text, and the control has 0.5%. Generated text is 88% thinking, 11% code and 0.2% assistant text.
- **Intent without writing.** The model says it will record memory in 56 (r1) and 52 (r2) calls but writes a tag in the same call only 23% of the time. Example: "Let me record the world model in memory first (important, since history trimming may occur)." with no tag. Another: "The memory tags in the code caused a syntax error. I need to put memory tags in my reply text."
- **Repeats and contradictions (154 tags).**
  - There is 1 near-duplicate (word overlap ≥0.6) and 0 confirmed-versus-refuted flips.
  - Two tags are explicit "Correction:" or "Fix:", and 8 carry self-revision wording.
  - Example: tu93 r2 L8 says "blocked moves are free 'wait' turns". L9 says "a blocked-direction move … FREEZES the whole world … waiting is not possible".
  - The model-written tag channel is not noisy; it is nearly empty.
- **Ledger blocks shown at the 85 r2 trims.**
  - Size: mean 1,107 chars (about 370 tokens), median 977, p90 2,367, max 3,685.
  - Composition: 4.8 entries per block. 60% of entries are harness-extracted fragments from the model's thinking (`note_think`), and 87% of level_notes are such fragments.
  - Fragment quality: all of them are cut off at 200 chars, and 61% start with filler such as So, Then, Hmm or But.
  - 68% of blocks (58 of 85) contain no model-written content. In r1, simulated, it is 87% (71 of 82).
  - Verbatim fragment shown to the model (ft09): `level 2: hmm, that doesn't work. That would be a beautiful interpretation: the lens shows the required colors literally...`
- **Whether the ledger gets used.** In the next 3 calls after a trim, the thinking shares a 6-word phrase with the ledger in 18% of trims.
- **Cap truncation.** The 200-char entry cap truncates 88% of world_model tags (median 332 chars, p90 529), 55% of level_notes, and 14% of hypotheses (median 129 chars).

**(2) Re-derivation after a trim (r2, 85 trims)**

- **Window left after a trim.** Trims come about every 21 calls. Afterwards about 14 calls (median 35 actions) stay visible. In 46% of trims no earlier-level evidence is left in the window.
- **Measured cost of trims.**
  - In the 3 calls after a trim, re-derive, "I don't remember" or "from scratch" language appears in 15.5% of calls, against 10.8% elsewhere. That is about 0.13 extra events per trim.
  - The cost is about 53k tokens (1.2% of the run) and about 30 extra repeated actions (0.5%).
  - "What do the actions do" never recurs.
- **Post-trim thinking length.** Post-trim thinking is 6.4k chars per call against a 5.1k baseline. But the 3 calls before a trim are longer still (7.6k), so long calls cause the trim rather than the reverse.
- **Rough comparison across runs.** Re-derive language after a trim versus baseline is 4.6% vs 4.6% (r1), 9.0% vs 5.0% (r2) and 10.2% vs 4.7% (control). The standard error is about 1.7 points, so ledger and control cannot be told apart.
- **Function redefinition.** About 10–12% of calls redefine an existing retained function, in all three runs.
- **Verbatim examples.**
  - r11l: "I don't remember exactly. My notes said 'moved W from [28,38] to [54,22]'…"
  - m0r0: "my earlier bfs2/solve5 machinery (which I lost)"
  - sc25: "Let me recall L4's head… I don't remember exactly."

**(3) Revisits (r2 boards, 5,630 actions, 4-cell border cropped)**

- **No-ops.** 5.9% of actions are no-ops. Arrow keys are about 4%, SPACE 7%, MOUSE 11%.
- **Revisits.**
  - 14.2% of moves land on a board already seen in the same level (11.7% within the same attempt). The per-game maximum is 31% (lf52).
  - 27% of actions lie on net-zero loops.
- **Exact (state, action) repeats.**
  - 504 repeats, 9.0% of actions, 95% with the same outcome. 74% are inside one attempt and 26% are replays after a death.
  - Only 8 of the 360 repeats across calls (2.2%) happened after the earlier occurrence had been trimmed. The other 97.8% repeated something still in the window.
- **What the model does at a revisit.**
  - 19% of actions start from a board already visited, and 47% of those repeat an action already taken from it.
  - After a loop closes, the next action is a previously tried one 56% of the time (338 of 598).
  - The model mentions the revisit in its next call 45% of the time, against 33% for the same words at baseline.
- **Redundant calls.** Calls where every action is a known pair are only 1% of generated text. Calls with only no-ops or known pairs are 7.3%.

**(4) Deaths**

- **Run endings.** 17 of 25 games end as `gave_up` at about 7,260 s of wall-clock (r1, r2); the control is 18 of 25. None ends by death. Score is the sum of cleared-level indices over the total, for example 5/6 levels gives 71.43 = 15/21, so actions are not penalised and the limit is time and tokens.
- **Death counts.** There are 168 deaths across 8 runs (r1 22, r2 24, control 20), concentrated in tu93 (5–9 per run), sc25, wa30 and bp35.
- **Causes.** I labelled 54 deaths by hand from the transcripts.

| Cause | Share |
|---|---|
| Deliberate budget burn | 30% |
| Deliberate hazard probe | 17% |
| Unintended hazard | 43% |
| Unintended budget run-out | 11% |

  - 46% are intentional, usually to force a reset. Example: ka59 "Let me burn about 10 more moves to trigger a reset."
  - Per run, intentional deaths are 9 of 18 (r1), 11 of 21 (r2) and 5 of 15 (control).
- **Cost (r2).** A dead attempt averages 43.5 actions (median 36). Dead attempts hold 17.8% of actions, 15.2% of generated text and 13.4% of calls.
- **Repeated deaths.**
  - Of 47 same-level consecutive death pairs, 23% share the fatal action, 19% share at least half the action prefix, 1 is an identical sequence, and 11% share the last 3 actions.
  - Example: tu93 r2 L4 died at actions 72 and 76 from the same trap.
  - The same first-death site recurs across runs: bp35 L2 in 8 of 8 runs, tu93 L2 in 8 of 8.

**(5) Level clears (8 runs, 854 cleared levels; ledger runs 224)**

- **Length.** Median 29 actions per cleared level (p25 16, p75 53, p90 73), at 0.66 of the human baseline.
- **Waste inside a level (r2).**
  - Of the 3,520 final-attempt actions on 104 cleared levels, 78.5% are needed once loops are removed (median 21).
  - 20% of the actions on cleared levels were spent in earlier failed attempts.
  - A compact run-length action string for the winning path has a median of 75 chars (p90 153, max 239), roughly 25–55 tokens.
- **Reuse across levels (664 consecutive cleared pairs).**
  - The next level uses fewer actions in 34% of pairs (L1 to L2: 24%, median ×1.7, against a human baseline ×1.5).
  - It uses fewer tokens in 42%, and a lower ratio to the human baseline in 50%.
  - Median calls before the first action are 3 on both L1 and L2+. First-call text mentions reuse in 44–50% of later levels against 0–4% on L1.
- **Final level.** The unfinished final level takes 28.7% of tokens and 18.8% of actions in the ledger runs.

**Caps suggestions**

- **Entry cap.** Raise the world_model entry cap from 200 to about 400 chars, which covers 73% (550 covers 91%). Keep hypotheses at 200–300.
- **Count caps.** The current counts are enough; the busy tu93 run needed 35 world_model entries against a cap of 6.
- **Block size.** About 1.5k chars (≈500 tokens), against the current 1.1k mean.
- **Pinned solved-level record.** About 160 chars per level, under 1.5k chars for 9 levels.
- **Action-effect table.**
  - Keyboard games have a median of 4 non-click actions, so a table is about 6 rows of 90 chars (about 540 chars).
  - Click games (17 of 25) have a median of 27 distinct click targets (p75 66, max 88); group them by object, about 12 rows (about 1.1k chars).
- **Every-turn cost.** A 370-token block is about 3% of a turn's new tokens (images are about 8.8k). If it is kept byte-identical in history for cache safety, it adds about 5k tokens per window (about 4%).

**Which component targets the largest measured loss**

Trim forgetting, which is what ledger-on-trim and a pinned record target, is the smallest loss: about 1–2% of tokens and 0.5% of actions. The ledger content is also 60–87% harness fragments, not model writing.

Ranked by what is measured:
1. Dead attempts cost 15% of tokens and 18% of actions, and 54% of deaths are unintended.
2. Visible-but-unused evidence is the largest loss: 9% exact repeats, 27% of actions on loops, and 97.8% of repeats still in the window.
3. Reuse across levels is weak: 34% of next levels are cheaper.

So the largest measured loss is knowledge that is already visible but unused, and a harness-built action-effect table shown every turn, with fatal-action marks and a "been here before" line, targets it directly. Model-written channels, the ledger-on-trim notes and the hypothesis ledger, address the smallest losses and are rarely written.

The ledger-run gain over the control (52.56 and 49.14 against about 45.3 across 5 control runs) is not explained by trim recovery.