# MR memory-line precision audit (agent report, 2026-10-15; data in S/audit/tmp-precision/)

2,630 requests; instance = first appearance of a line text at (game, level); 262 hand judgments (judgments.jsonl), board claims checked
by script against events. C/W/M/S/N = correct/wrong/misleading/stale/noise.

| Line | Shown (inst/games; stalled) | Sampled (stalled) | C/W/M/S/N % | Used |
|---|---|---|---|---|
| Rules quoted this level | 370/25; 167 | 40 (32) | 48/8/20/2/18 (+5 unverified) | 5% | 12 of 32 stalled samples bad vs 0 of 8 normal |
| Frozen rules in previous_levels | 196/23; 44 | 16 (4) | 25/0/12/0/62 | 0% |
| Not-touched contact row | 671 rows (2,811 objects)/25; 228 | 36 (28) | 78/3/6/0/14 | 6% | harm = what the 4-cell minimum leaves out (su15 1-cell dots, re86 1-cell goals); 54% rows cut ("N more") |
| Hit background | 17/9; 11 | 17 (11) | 59/0/29/0/12 | 65% | su15's 4 lines all on 13-frame animations |
| Moved since level start | 127/12; 38 | 26 (6) | 42/4/8/0/38 (+8 unverified) | 12% | g50t HUD headline, ka59 wrong identity, dc22 checker holes |
| Hidden since / Appeared at + boxes | 18/2; 10 | 11 (7) | 91/9/0/0/0 | 100% |
| Touching | 4/3; 1 | 4 | 75/0/0/0/25 | 75% |
| Also clicked | 38/11; 26 | 16 (11) | 25/0/0/0/75 | 12% |
| Budget bar | 7/3; 4 | 7 | 100 | 43% |
| Attempt GAME OVER | 7/3; 4 | 7 | 100 | 0% |
| Undone action | 72/10; 27 | 24 (11) | 100 | 79% |
| Dead click runs | 3/2; 0 | 3 | 100 | 33% |
| Effect counts | 1,737/25; 863 | 10 (all checked by script) | 100 | 3-4% |
| Not tried | 270/19; 77 | 10 | 0/0/0/0/100 | 0% |
| Board seen before | 130/21; 72 | 10 (all by script) | 100 | 4% |
| Recaps | 107/25; 23 | 25 (12) | 88/0/0/0/12 | ~8% |

Pattern: lines about the action just taken are used 43-79%; <this_level> rows 0-4%.
Top repairs: (1) rules filter V3: reject negation/impossibility words (goal is not, not the goal, never, impossible, cannot, can't) even
when marked; carried "So" conclusions with requires/required/must/critical; fragments starting lowercase, "✓" or "("; "possible:",
"Also possible", "consistent with" (fixes 7/14 bad sampled, loses 1/23 correct; run-wide drops 93/566 quotes). Next: recency tie-break.
(2) contact row header "objects of 4+ cells" + "Smaller marks (1-3 cells): sky blue x8 at (37,18) ..." (positions when <= 8 marks).
(3) hit background only when no interior animation; name the object with the nearest cell, any size (fixes su15 x4, sb26).
(4) moved-line headline: skip background-colour and edge-band objects (fixes 8/26, loses 0/11). (5) frozen rules: keep quotes with
<= 1 number, no plan words, passing (1) (drops 9/12 bad, keeps 4/4). Budget bar counts-only: REJECTED (all 7 correct, 2 used well).
Optional: drop "Not tried" (never referenced) and "Also clicked" (75% noise).

# tu93 / re86 read (agent, 2026-10-15; scripts S/audit/tmp-tu93-re86/)
tu93: MR L8 34 work-min vs SF 10 / Ucore 9 / keep 11 (own facing-reader bug + wrong pursuit recap); L9 19 min, 0 actions: the L5/L6
recap "avoid the orange's current and next cell" is false (every winning L9 route lands once on an orange's previous cell); cited at i142.
Park: 23 min right after L1 cleared (i=12, cached 0, no trim): committing L1's memory hands the slot over because
ARC3_MEM_APPEND_NO_HANDOVER is off; SF/keep/Ucore had the same park, uc runs did not. MR needed ~3-12 more minutes to win tu93.
re86: L1-L6 work 46 min (fastest of MR/uc2/SF); 37 min of parks (L2 +13 after L1 commit, L4 +24 trim handover) are the whole gap.
L7 38 min open: L4/L5 recap "parking the center on a block recolors" (uc2's: "recolor when the object touches the fill"); contact row
listed covered pad-frame fragments (misled ~4 min). Proposed: arrow-key mover facts (trail relation k presses earlier; contact outcomes),
recolour-event line instead of a count.

# ka59 L4 / wa30 L3 / ls20 L5 read (agent, 2026-10-15; scripts S/audit/tmp-ka59-wa30-ls20/)
Not parking (8, 0, 6 min). ka59 L4 (40.7 min open, one push short): L2 recap "purple blocks moves and pushes" contradicted MR's own
L2 evidence (pushes cross purple); rules line quoted a refuted "horizontal moves only" to i103; contact row dropped an untouched twin.
wa30 L3 (39.7 open): RIGHT no-op at action 199 (the body would have entered the wall) misread as a carry rule for the rest of the
level (~45%); rules line quoted an untested "carrying through the hole works" for 30 requests; contact row listed the player's own
parts and the recoloured grabbed block. ls20 L5 (53 open): empty bar restarts the level (lost lives) without GAME OVER; <this_level>
said "no deaths"; L4 recap's bar rate stale (2 vs 4 cells/action); rules line frozen 70 requests on conveyor sentences.
Moved-objects line never fires on keyboard games (arrow moves excluded by design). Fix priority: (1) soft restarts as deaths with a
bar reading; (2) blocked-by fact on no-op arrow presses (also m0r0); (3) per-colour passability facts.
