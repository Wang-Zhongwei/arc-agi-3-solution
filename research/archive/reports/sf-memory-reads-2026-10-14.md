# What the memory did in the stall-fix run (2026-10-14)

**Question (the user, after `stall-fix-10x` scored 51.85 with 8 wins):** "optimizing the prompts and improving upon
memory is making it better ... if we can solve 9 or 10 games in that same time period, it would help. analyze the
output for stall fix."

**Answer:** in every stall we read, the harness-written memory was **wrong, stale or noisy, or dropped a correct
fact exactly when it mattered**:
- The "Rules you confirmed" line is mostly not rules, and the model treats it as verified by the harness.
- A good fact (the moved-objects line) vanished when its marker was hidden.
- The board saved for a cleared level is the board before the winning move.
- After a trim, `<this_level>` can be missing altogether.
- The colours row is click-only and tags the model's own recolourings.

None of the six stall-fix flags slowed a level. Where a fast run cleared the same level, its edge was its own
exploration, not better memory. So the memory lever here is **repair**: stop the memory misleading the model, and
keep the facts that worked. More lines is not the lever.

Eight reads were done, each against the fastest runs on the same level, plus a run-wide audit.
Scripts are in `research/data/sf_reads/`.

## 1. The reads
| Level (SF) | Time / tokens | Fastest other run | What decided it | Memory finding |
|---|---|---|---|---|
| tn36 L2 (open) | 118 min / 136k | uc1 4 min | The demo's legend boxes were clicked only at 75 min, and their code was never copied to the editable grid | The colours row sent a click to the wrong yellow object and could never point at the boxes. Undone lines were correct but redundant. A rule "move the plug down 16 rows" stayed ~45 min, false. |
| vc33 L4 | 87 min / 160k | U 6 min | 28 near-idle requests re-read earlier levels' "winning boards" | **`history`'s last frame of a cleared level is the board BEFORE the clearing click**, so the model judged its correct recap "hallucinated". `<this_level>` was absent from i58 to i71 after trims. Three wrong sentences about L1-L3 were filed as L4 rules. |
| sc25 L3 | 35 min / 41k | ms2 4 min | Fired the beam at the wall instead of the gate; misread the step size | The death verdict was correct but redundant. 23.5 of the 35 min were parked. |
| sc25 L6 (open) | 63 min / 81k | B 10 min | The ring-size link was lost after a shrink | **The moved-objects line was correct and decisive at i89-91, then vanished from i92** because the marker was hidden. A refuted rule stayed "confirmed" i87-i101, and a "may" hypothesis was promoted. |
| tr87 L2 | 82 min / 135k | uc1 3 min | Rotated glyphs were compared as exact bitmaps | The level-1 recap omitted "rotation-invariant", which U's and Ucore's recaps had. A plan and a wrong rule were shown as confirmed rules. |
| m0r0 L3 | 46 min / 83k | comb 22 min | Probes misread; the level-2 recap was misleading | A refuted rule was quoted after its refutation. A finished plan then waited 20.6 min for a slot. |
| m0r0 L5 (open) | 46 min / 130k | ms2 9 min | Never stepped on the purple pressure plate; held "blocks are walls" untested | The colours row is click-framed: it led to 7 gate clicks and can't say "never touched". A hypothesis was quoted as a confirmed rule. |
| su15 L4 (open) | 79 min / 179k | U 4 min | Imprecise click-reach rules cost three attempts | The level-3 recap stated the wrong reach metric. 34 of 35 "new colour" tags were the model's own merge products. |
| cn04 L2 / L4 | 35 + 40 min | U 13, A 3 min | L2: a correct "click did nothing" fact was doubted (the click hit the gap in a U-shape). L4: a bug in the model's rotation helper. | All 51 "new colour" tags were selection recolourings. 26 of the 75 min were parked. |

## 2. Run-wide audit (all 25 games)
- **Rules line (512 distinct entries, hand-labelled):**

  | Class | Share |
  |---|---|
  | Real mechanics | 36% |
  | Trivia | 24% |
  | Plans | 17% |
  | Hypotheses stated as fact | 15% (30% of impressions on stalled levels) |
  | About the model's own code | 4% |
  | Echoes | 3% |

  - 23 entries were later contradicted, and in 7 cases both sides were shown at once.
  - The model calls the line "the harness note".
  - A stricter filter (no coordinates or ranges, fragments, echoes, plans, hedges, or unmarked "never/only/must"
    conclusions) raises mechanics to 46% and keeps 88% of them.
- **`<this_level>`:**
  - Median length 764 characters.
  - 152 of 1,237 blocks silently drop a clicked colour.
  - 44% of arrow presses that moved something are reported as "changed cell colours".
- **`<previous_levels>`:**
  - 7 of 10 spot-checked recaps are accurate.
  - Contradictory frozen rules appear in sp80 L2, ka59 L4 and re86.
  - Earlier levels are referenced in 42% of requests, so stale content there is costly.
- **Re-sent text:**
  - The retained-functions list is 37% of a turn message and is identical to the previous one 69% of the time
    (~5M re-sent tokens per run).
  - The model's own old reasoning is 72M of the 158M prompt tokens.
- **What the model cites:** the undone line (44%) and the touching line (46%) are cited often. Rules, colours,
  untried and effect rows are cited in 0-2% of the requests where they appear.

## 3. Not a memory problem: parks after trims
In every read, the first request after a context trim waited for a slot:

| Level | Minutes parked |
|---|---|
| tn36 L2 | ~80 of 118 |
| tr87 L2 | 42 of 82 |
| sc25 L6 | ~41 of 63 |
| vc33 L4 | 36.5 of 86 |
| cn04 L2 + L4 | 26 of 75 |
| sc25 L3 | 23.5 of 35 |
| m0r0 L3 | 20.6 of 46 |

- The server ran 13-14 requests at once for most of the run, so the parks hand GPU time to other games (zero-sum).
- The cause is the priority gate, which releases a game's slot at a trim (`tool_agent.py` `_PriorityGate`).
- Earlier scheduling studies bounded reallocation at about +4.9 points (oracle). Final-level hold measured
  +0.04-0.10 full clears in simulation.
- Recorded here for completeness. It is out of scope for a memory arm.

## 4. What we build next (`arm/mem-repair`, each change behind its own flag)
- **Rules:**
  - A stricter filter.
  - An action number on each quoted rule.
  - The retained-functions list sent only when it changes.
- **Facts:**
  - Moved and box lines persist when a marker is hidden, with hidden and appeared entries.
  - `cleared_boards[N]`, plus a fact saying history's last frame is pre-clear.
  - `<this_level>` re-rendered after a trim that removed it.
- **Objects:**
  - An object-level "not touched yet" row replacing the click-only colours row.
  - A missed-click line (a click on background inside an object's outline).
  - Clicked colours are never dropped.
  - A fix for the moved-vs-recoloured effect wording, if small.

Spec: `research/data/sf_reads/SPEC_mem_repair.md`.
