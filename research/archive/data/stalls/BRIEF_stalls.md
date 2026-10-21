# Brief: why do single levels stall? (2026-10-14)

Goal: 10 of 25 games fully won in the 2-hour Kaggle run. In the last two runs (U = mem-unified-10x, 56.09, 8 wins;
Ucore = mem-unified-core-10x, 51.15, 8 wins) wins were decided by single-level stalls: one level eating 70-100
minutes of wall-clock (100-135k tokens) on a level other runs clear in 5-30 minutes. We need a GENERIC harness
change (the private set has 110 unseen games) that shortens stalls, to test in one final full run tonight.

Already established (don't re-derive):
- A plain restart (wipe the level's conversation, start fresh) LOSES in simulation: after a 60k-token trigger,
  continuing clears 57% of stalled levels, a fresh attempt within the remaining effort ~12%.
- Stalled games also run slowly in tokens/minute (the scheduler deprioritises stalled games): ~1.2-1.4k tok/min
  vs ~3k for a busy stream.
- Earlier failures: a strategy-audit ADVICE prompt every 60k tokens (no gain); forced RESETs (no gain);
  required tool params like "prediction" (filled, not used).
- Model: Qwen3.8-Flash-Next W4A16 (small, local). Harness-computed FACTS in tool results / turn messages can work;
  ADVICE text is ignored or hurts; optional structure is ignored. State-lab evidence (resampling the next reply
  at frozen stuck states, k/20 right moves, control -> with fact): "moved objects since level start" 0 -> 16;
  "touching same-colour objects listed separately" 5 -> 10; "your own confirmed rules quoted" 3 -> 9;
  "the chaser stands where you were 2 moves ago" 1 -> 5; re-probe / untried-action / action-count / clock facts: no effect.
- U had these flags on: memory every call (`<previous_levels>` with each cleared level's route + the model's recap),
  confirmed-rules ledger, moved-objects line, touching-split line. Ucore had the memory + rules but NOT moved/touching.

## Data (read-only)
Compact extracts: `lab2/extract/<run>/<game>.requests.jsonl` (one row per model request, 0-based `i`):
  i, step, action (number of the NEXT game action), level (level at request time), within_turn, n_msgs,
  last_role, last_text (newest message: tool result or turn message, <=6000 chars), reasoning (model's thinking
  for its reply), content, code (python it ran), usage (prompt/completion tokens), finish.
`lab2/extract/<run>/<game>.actions.jsonl`: one row per game action: n, a, level (level AFTER the action), step,
  cleared, over (GAME OVER), changed (board changed), batch.
Raw logs (big; parse with scripts, never cat): run dirs are in RUNS in `lab2/extract.py`.
  `<dir>/artifacts/<gid>_p0_events.jsonl` has every action's board (64x64 ints) and analysis transcripts with
  HH:MM:SS timestamps; `<dir>/score_timeline.jsonl` has per-game levels/tokens vs `t_bench_s` (budget 7260 s).
Other runs' extracts exist for comparison (A, B, C, comb, keep, ms1, ms2, pf, uc1, uc2, uc3; not every game).
Earlier reports on these games: `lab2/report_{m0r0,tu93,sc25,r11l,vc33}.md` (they cover later deciding levels).

## What to return (your final message IS the report; <= 900 words, plain markdown)
1. The level: mechanics and the winning route (from a fast-clearing run's actions), and the key insight the
   fast runs had. Quote the fast run's reasoning at its turning point (request index).
2. The stall, as a timeline: level start/end (wall s, requests, tokens), and phases. For each phase: what the model
   believed, what it was doing (e.g. re-running the same experiment, long thinking without acting, python errors,
   dying repeatedly, executing a wrong plan, waiting/starved), quoting short reasoning snippets with request index.
3. The decisive gap: what the model was missing or got wrong. Was the needed information visible in its context
   (board, diff, tool result)? Did evidence contradict its belief, and did it notice? When it finally cleared,
   what changed (quote the turning point)?
4. Classify: (a) misperception a harness fact could fix (say which fact, computed how), (b) wrong goal/mechanic
   hypothesis held despite contradicting evidence, (c) execution/tooling failure (python, coordinates, planning
   code), (d) loop: repeating tried experiments, (e) starvation/time only, (f) needs an insight no generic fact gives.
   Several can apply; give rough shares of the stall's tokens.
5. A generic, harness-computable intervention that would have shortened THIS stall, with the exact text/behaviour,
   when it fires (a detectable signal at that moment, not hindsight), and the request index where it would have
   fired. Say whether one of the existing facts (moved, touching, own rules, chaser) already covers it.
Constraints: read-only (never edit repos or run dirs, do not write files); run python under `nice -n 15`,
one heavy job at a time; do not execute game engine code.
