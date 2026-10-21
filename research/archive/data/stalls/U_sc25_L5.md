# sc25 L5 stall in run U (mem-unified-10x)

## 1. The level and how the fast runs won it
- Board: a 4x4 two-colour cap starts boxed in an isolated cross. The goal ("head") is at the top of a right-hand tube, which a dark-red cork and an orange block plug.
- Left region: two magenta sockets (a red core above a 2-wide charcoal gate; an orange core bottom-left) and a small purple 2x2 pad.
- A 3x3 keypad takes three reference patterns:
  - plus toggles the cap between 4x4 and 2x2 (engine `sc25.py` ~line 2118: scale 1↔2);
  - yellow-L teleports the cap to the ring-marked pad matching its current size (4x4 → tube bottom, 2x2 → purple pad);
  - magenta column fires a beam in the last-move direction; reaching a socket deletes the socket and all cells of its core colour.
- comb's winning route (10 min, 52 actions, 30k tokens): L, UP x4, column (nothing); plus, L → purple pad; LEFT x6, UP, column → red cleared; DOWN x4, LEFT x3, column → orange cleared; RIGHT, UP, **plus (grow)**, L → tube bottom, UP x6. uc1/uc2/B/A/uc3: same skeleton, 11–17 min.
- Key insight: plus is a toggle and the teleport is size-matched. comb i=64: "Fire the plus pattern now… should become 2x2. Fire it again: does it become 4x4?"; i=70: "The plus pattern GREW the block back to 4x4!… So the plus pattern toggles the size (shrink ↔ grow)."

## 2. The stall timeline
L5 ran from ~1174 s (i=45, action 90, 57.5k tokens) to ~6860 s (i=100, action 201, 166.2k): 95 min, 56 requests, ~109k completion tokens, 113 actions, two RESETs.
- P1 i=45–59 (1174–1650 s, 25.8k tok): normal discovery. Shrank, teleported to the purple pad, saw a second L do nothing; by i=59 the right theory: "the teleport picks the marker whose size matches the cap… maybe the PLUS pattern toggles the cap's size (shrink ↔ grow)."
- P2 i=60–64 (to 1851 s, 6.5k): cleared both obstacles (orange at action 113, red at 123); i=64: "the right tube is FULLY CLEAR". 13 actions from winning (plus, L, UP x6).
- Park 1: 1851→2986 s (19 min) while i=64 was in flight.
- P3 i=64–73 (to 3496 s, 30.3k): the slip. To test grow it typed the wrong clicks, actions 127–130: (50,25),(50,35),(55,30),(60,30) instead of (50,30),(55,25),(55,35),(60,30). i=65: "The PLUS match did nothing… the pattern is 'active' but no effect… So the plus doesn't grow the cap." The leftover lit cells spoiled its next L (131–133); it fixed them, L sent the cap back to the purple pad, beam into black to test a "paint" idea; i=71 was a 12,288-token budget-bar ramble that hit the length cap with no tool call; then RESET (action 144).
- Park 2: 3546→4684 s (19 min).
- P4 i=74–84 (17.6k): repeated experiments (teleport 4x4 cap to tube, beam from inside the tube, L again, push the blocks, click the purple pad). i=82: "So the cap has never grown: the shrink is permanent." At i=84 it even saw the bad input ("clicks (50,25),(50,35),(55,30),(60,30) → that's a different pattern (maybe a 'T')… None grew the cap") and did not re-test.
- P5 i=85–92 (23.2k; park 3 5257→6641 s, 23 min): theorising ("closest seat"), settled on "the 'grow' hypothesis (the button) is the key untested idea" (i=90) — right action, wrong reason.
- P6 i=92–100 (5.3k): RESET, shrink, L, LEFT x8 over the button, plus. i=97: "THE GROW WORKED!" → beams, L, UP x6, win.
About 34 of the 95 minutes were active, about 61 parked.

## 3. The decisive gap
U had the fast runs' full theory by i=59 and cleared both obstacles by action 123. Only a mis-typed click set caused a false negative, which it held from i=65 to i=92 (~71k tokens). The contradicting evidence was in its context: the correct plus clicks in its own `<previous_levels>` L4 route and its own L5 actions 90–93; no animation line in the turn message after 127–130 (every real match had one); four keypad cells left lit (a real match resets the keypad); at i=84 it explicitly noticed the clicks were "a different pattern". It never connected these to "my test was invalid". The harness made it worse: `<this_level>` said "Clicks on gray cells (20 clicks on 7 cells): changed cell colours 20 times", so the dead click run looked like the effective ones.

## 4. Classification (shares of ~109k tokens)
- (c) execution slip feeding (b) a wrong belief held against evidence: ~65% (i=65–92), including (d) loop: ~6 teleport re-tests from 3 positions, repeated history scans.
- normal discovery: ~30% (P1–P2, the same as comb's whole level).
- (c) python errors (bad `act()` signature, `inspect` import): <1%.
- (e) starvation: 61 of 95 wall-clock minutes (costs no tokens; parks follow from the low progress rate).
- (a)/(f): not the cause.

## 5. Proposed intervention: click-run outcome ledger (fact, not advice)
- Compute: split actions into runs of consecutive MOUSE clicks. A run is effective if its final click changed cells outside the clicked cells or played a multi-frame animation. Keep effective runs with a one-line effect from the object diff. When a run of >= 2 clicks is not effective, print it next to the effective runs, plus the cells it left lit.
- Fires: a non-effective click run when an effective run of similar size exists; here at i=65 (t≈2986 s, after action 130) and again at i=66.
- Text: "Click run (actions 127-130) (50,25)(50,35)(55,30)(60,30): changed only the 4 clicked cells, which stay lit; nothing else changed and no animation. Click runs on this game that changed the board: (50,30)(55,25)(55,35)(60,30) at 90-93 → b/S object 16→4 cells; (50,25)(50,30)(55,30) at 94-96 → b/S object moved rows 15-16 → 39-40; (50,30)(55,30)(60,30) at 111-113 → orange cells removed. Your run differs from the first in 2 cells."
- Likely effect: the correct plus at i=65 grows the cap in place (engine offset search fits rows 41-44 cols 15-18), then L, UP x6: clear at ~52 min instead of 114, saving ~70k tokens and both later parks.
- Overlap: none (moved/touching/chaser irrelevant; the rules ledger quoted unrelated rules). Closest in spirit to "own confirmed rules quoted" (3→9): the model's own earlier successful input against its failed one. Generic for click-panel games.
Temp dumps: scratchpad `u_l5_b.txt`, `u_l5_c.txt`.
