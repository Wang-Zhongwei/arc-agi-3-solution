# fable-layer-vs-ours: measure:our-runs

Workflow run `wf_c4d3dccd-2d7`, phase Measure. Compare leoprovorov Fable & Astra notebook (handbook + Fable layer on Franzen) with our stack: read, measure claims on our runs, compare, skeptic-verify

---

# Fable claims tested on our recorded runs

## Scope and method

- **Boards used.** Frames are only stored in `artifacts/*_events.jsonl`, for six runs: the full runs `fz-ledger-on-trim-r2` and `v4-stack-v1`, plus the 10-game panels `v4-panel-control`, `-dprime`, `-easy-first` and `fz-experts8`. That is 13,040 non-RESET actions. `benchmark.json` holds actions only, no boards, so the other four full runs (control, temp06, audit60k, ledger-on-trim) only contribute to the token and UNDO counts. [verified]
- **Detector code.** Fable's `fable_layer_core.py` ran unmodified. It was fed exactly as `_fl_sync` does (fable_layer_patch.py:80-98): one Ledger per level, the clearing action fed as a no-op, RESET clears marks, and the noisy cutoff above 6 boxes.
- **Bar detection.** `detect_budget` ran at every analysis call on the attempt frames, built as `_fl_attempt_frames` builds them.
- **CPU.** About 35 s in total, all under `nice -n 15`, one process at a time.

Scripts and outputs are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/fable-compare/`:
- `frames_pass.py` and `frames_pass_<run>.json`
- `claims_analysis.py`, `claims_analysis.out`, `claims_analysis.json`
- `extra_summary.py`, `death_intent.py`, `render_samples.py` and `render_samples.out`
- `fp_samples/sheet1.png`, `fp_samples/sheet2.png`
- `sim_eviction_flags.py`

## (1) Indicator detector

**Event rate is about double theirs.** [verified]
- Ours: 824 Ledger events over 13,040 actions, which is 6.32 per 100 actions.
- Theirs: 202 over 5,832, which is 3.46 per 100 (narrative.md:274).
- r2: 377 events in 5,630 actions (6.70/100).
- stack-v1: 359 in 5,732 (6.26/100).
- Panels: 2.8–6.7/100.

**Which games fire** (r2 + stack-v1 pooled): 22 of 25. The detector is silent on cn04, ft09 and g50t, and vc33 gives only 1 event. [verified]

| Game | Events per 100 actions | Note |
|---|---|---|
| sb26 | 32.9 | 94 of 95 events are MOUSE events at the click point |
| r11l | 28.3 | |
| sp80 | 17.2 | |
| cd82 | 9.9 | |
| su15 | 9.4 | |
| dc22 | 9.3 | |
| ka59 | 8.4 | |
| re86 | 8.1 | |
| sk48 | 4.3 | |
| others | ≤3.8 | |

**Busy boards are not really silent.** The noisy cutoff only acts after 7 distinct boxes, and every event before that has already been emitted. 25 of 129 levels (r2) and 31 of 121 (stack-v1) went noisy. sb26 still produced 95 events with 14 of 16 levels noisy. [verified]

**The token gate passes about a third.** In the r2 replay with Fable's gates, 128 of 377 events (34%) became note lines, spread over 18 games (`replay_fz-ledger-on-trim-r2.json`). [verified]

**sk48 tray claim holds.** Their claim: all 83 events on 10 sk48 copies are tray marks in rows 58–59 (narrative.md:273). Ours: all 19 sk48 events across 6 copies are in rows 58–59 — box (58,28)-(59,29) ×15 and (58,34)-(59,35) ×4. [verified]

**False-positive proxies over all 824 events.** [verified]
- 152 of 449 MOUSE events (34%) are within 2 cells of the click.
- Only 15% lie within 6 cells of the frame edge, where HUDs and trays usually sit.
- 479 of 594 distinct boxes are seen only once.

**Ten-event visual sample** (`fp_samples/sheet*.png`, crops in `render_samples.out`; one event per game, seed 11). These labels are my visual judgement. [inferred]

| Label | Count | Events |
|---|---|---|
| Real HUD-type indicator | 2 | ls20 lives counter, rows 61-62 cols 62-63, red→grey as the bar refilled; sc25 right-column state stack, orange→grey after a keypad click |
| Far change in game state, informative but not a tray or legend | 4 | tn36 display-panel icon vanished after a button click; wa30 slot border; dc22 sliding bar segment; m0r0 block colour flip (kind "lost") |
| Not an indicator | 4 | r11l clicked object highlighted (click_gap 0); sb26 piece placed at the click; cd82 palette selection underline 3 cells from the click; re86 leftover piece of the moving red outline |

The note text ("Legends, trays and counters often report progress" / "A mark you earned may have been lost") would mislead in about 4 of 10 cases and overstate in another 4.

**Lost marks vs tokens on the level.** Their claim is ρ=0.93, n=10 copies of sk48 level 1 (narrative.md:203). Our results, joining 330 (run × game × level) units to levels.csv: [verified]

| Subset | n | Spearman ρ (lost, tokens) | Other |
|---|---|---|---|
| sk48 level 1, our copies | 6 | 0.80 (permutation p=0.068) | Partial ρ given actions 0.76. Same direction as theirs, not significant. |
| All units pooled | 330 | 0.26 (p=0.0002) | ρ(lost, actions) = 0.14; ρ(tokens, actions) = 0.64; partial ρ given actions = 0.22 |
| Two full runs only | 250 | 0.26 | |

- **Within game the sign is mixed:** sk48 0.79, m0r0 0.79, tn36 0.70, r11l 0.50, but tu93 0.04, ka59 0.04, sp80 −0.15, lp85 −0.20.
- **Levels with at least one lost mark** (n=52): median 43.8k tokens and 75% cleared. Levels with none: 21.9k tokens and 82% cleared.
- **Most lost events are in dc22** (22 of 97, the 4×4 tiles), then sp80 (13).

The association is real but weak overall. The strong ρ is specific to sk48-like trays, and the causal direction is untested. [inferred]

## (2) Token gates (sim-v2 levels.csv)

Tokens here are generated tokens on the level, deaths included. Six named full runs pooled: 748 level attempts, 27.47M tokens. [verified]

| Gate | Share of level attempts |
|---|---|
| ≥15k (notes start) | 67.4% |
| ≥30k (first pin) | 44.1% |
| ≥70k (first refresh) | 15.0% |
| ≥110k (second refresh) | 4.8% |

- **Per run, ≥30k:** 41.1% (ledger-on-trim) to 48.8% (control). All 11 full runs give 66.2 / 43.7 / 15.7 / 5.5%.
- **Tokens the pin could touch:** 42.9% of all generated tokens are spent on levels after they pass 30k. After 15k it is 65.1%.
  - 53% of those post-30k tokens are on levels never cleared (still being played when the 121-minute run ended).
  - So about 20% of all tokens are post-30k play on levels that do clear, and about 23% are on final stuck levels. [verified arithmetic]
- **Their 73% figure** ("27 of 37 baseline levels past 30k cleared", handbook p.86):
  - Ours past 30k: 233 of 330 cleared (70.6%).
  - Past 60k: 63 of 144 (43.8%), against their 11 of 18 (61%, same page). Their setting differs (15 hard games, 72 minutes).
- **Cleared levels:** 64% are cleared under 30k tokens (median 21.3k). Their figure for easy games is 90% (handbook p.74).
- **Pin count from levels.csv:** about 485 pins plus refreshes across the 6 runs, about 81 per run. The direct r2 replay gave 71. [verified]

## (3) Step budget

**Bars are everywhere.** `detect_budget` finds a bar in some analysis call in all 25 games. Most are row 63 or row 0 with cap 64; sk48 and sb26 use the row-53 divider. Not every one was checked by eye. [verified detection; individual correctness not checked]

**Deaths in the two full runs: 44 GAME_OVERs.** [verified]

| Cause (frame-based) | Count | Share | Detail |
|---|---|---|---|
| Budget: bar empty, or at most max(1, 2×rate) cells left before the fatal action | 17 | 39% | wa30 5, vc33 3, sc25 2, tr87 2, ka59 2, ft09, su15, tn36 |
| Action death with the bar well short of empty | 21 | 48% | tu93 13 of them (hazards, 29–61 cells left); bp35 4; su15 2 |
| No bar detected | 6 | 14% | ls20, whose bar refills; sc25, whose column changes colour |

- **Corrected budget share is about 45%.** The model's post-death words call 3 of the 6 "no bar" deaths budget deaths (sc25 ×2, ls20). Fable's monotone detector misses bars that refill or change colour. That gives about 20 of 44. [verified text]
- **All six board runs:** 61 deaths; 19 budget, 36 action, 6 no bar. [verified]
- **Intent behind the 17 budget deaths** (m4_deaths.py regexes plus my reading of the next turn's text): 5 are intentional burns used as a reset (ft09, su15, tr87, wa30 ×2), 4 are unintended (wa30 L2, sc25 L5, tn36 L3, vc33 L6), and 8 are unclear. The regex is noisy; wa30 L4 is misclassified. [inferred]

**When the model notices the bar** (THINKING/ASSISTANT/TOOL CALL text only; `BAR_RE` in frames_pass.py):
- **Regex precision:** 12 of 12 random matched calls and 15 of 15 random first mentions are about the step budget, by my reading. [inferred]
- **Mentions are near-universal and early:**
  - 193 of 226 levels with a detected bar (85%) have a mention.
  - 120 of those come before the detector could even see the bar (at most 3 actions into the attempt).
  - The other 73 come at a median of about 91% of the bar left.
  - Only 3 first mentions came at 50% or less left. [verified]
- **Deaths:** on 32 of 34 bar levels with a death, the model mentioned the bar before the first death. [verified]
- **Fable's low-budget alert** (≤25% left or ≤12 actions) would fire on 35 levels. On 34 of them the model had already mentioned the bar at the same call or earlier. [verified]
- **Levels with no mention** (33) are short (median 17 actions against 37) and had 1 death between them. [verified]
- **Their tn36 L2 anecdote** ("noticed only at 20 of 61 cells", narrative.md:239): in both our runs, tn36 L2 is mentioned before the detector could see the bar (call 9 in r2, call 6 in stack-v1). [verified]
  - The F001 baseline uses the same system prompt as ours. Our stack-v2 prompt already says "watch for a bar at the grid border that shrinks each action, and count remaining budget into your plans" (prompts.py:247).
  - The same prompt also says "do not get distracted by it" (prompts.py:105).
  - The control prompt (`sys_control.txt`) contains the same line. [verified]
- **What remains open:** mentioning the bar is not the same as counting it accurately. Fable's precise "N actions left" could still help on the 4–7 unintended budget deaths. [inferred]

**UNDO usage.** [verified counts]
- Six full runs: 159 UNDO of 32,318 actions (0.49%), from `benchmark.json` ACTION7 counts.
- r2 has 2 and stack-v1 has 11. Only su15, sk48 (65, all in control), lf52, bp35 and sb26 ever use it.
- Fable's UNDO caveat (their copy 9 pressed UNDO 127 times in 317 actions, narrative.md:243) is rare in our runs.
- Their "UNDO doesn't use the bar" is not universal: the bar line changed on 9 of 22 UNDOs taken while a bar was detected, including 4 of 4 in lf52. The meaning of those changes is inferred.

## (4) Do a pinned pair or our ledger head count as an eviction and trigger a handover?

**No.** Neither one raises the flag by itself. [verified code + sim]

Line numbers below are in `wt/stack-v2/ARC3-Inference/inference/agent/tool_agent.py`.

- **The handover only runs when the flag is set.** `_maybe_handover` returns unless `_context_was_trimmed` is set (6462-6464); only then does it call `gate.handover` (6499).
- **Only `_note_history_evicted` (6375-6380) sets the flag.** It is called in four places:
  1. `_trim_messages_for_context`, when the message count drops (6626).
  2. `_force_reduce_messages` (6642).
  3. `_persistent_history_messages`, when the trim result is empty (6406) or the persisted history differs from the request's (6430).
  4. Rolling-summary replacement (6239-6247). This is off: `ARC3_SUMMARY_REPLACES_HISTORY` defaults to False (1156), `ARC3_SUMMARY_INTERVAL_TOKENS` defaults to 0 (1036), and kaggle/template.ipynb cell 4 sets neither.
- **Tail-appended pair (Fable pin):** it is part of `self._history_messages` before the request is built. `original_history = messages[1:]` therefore already contains it, and the check at 6430 compares equal. No flag.
- **Our ledger head:**
  - `_apply_due_ledger` (6382-6393) runs only if `_ledger_due` is set, and only `_note_history_evicted` sets it (6380).
  - It runs in the same request iteration as the handover, between trim and `_maybe_handover` (6825-6827). So it rides on a handover the trim already caused.
  - The rewritten `messages` is what gets persisted (7273). The head is therefore byte-stable afterwards and never re-flags.
- **Simulation** on the real stack-v2 functions with the competition turn caps (150/30/coalesce), 40 turns:

| Scenario | Handovers in 40 turns |
|---|---|
| Plain | 0 |
| Tail pin pair | 0 |
| Ledger-on-trim | 5 (exactly one per flagged trim; `[mem]` head persists with no further flags) |
| Ledger + pin | 6 (the pin's tokens make trims arrive one turn earlier; still one handover per trim) |

- **Their lesson 3 is about history replacement,** which flags the eviction explicitly (6247 is the analogue in our code). It does not apply to tail pins or to our ledger. The opposite risk remains: Fable's refresh, and a head restore with nothing dropped, change the prefix with no flag. That means a full re-prefill while the game keeps its slot (B3/B5 in the earlier report).
- **Context for "27 pauses over 4 min" (handbook p.80, narrative.md:227):** our plain trim handovers already cause this.
  - From calibration intervals.csv, waits over 4 minutes: control 41 (about 1,208 min), ledger-on-trim 45, r2 40 (about 1,100 min; exact method: 45 switch-outs totalling 66,764 s), stack-v1 34. [verified]
  - That is about 45–50% of holding-plus-waiting minutes, because 25 games share 10 slots. Any feature that adds flagged evictions would multiply these waits. [inferred]
