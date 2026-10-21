# Brief: what in the prompt and memory decides a stall? (stall-fix run, 2026-10-14)

Goal (the user's): 9-10 of 25 games fully won in the 2-hour Kaggle run. The user's view: "optimizing the prompts and
improving upon memory is making it better". We want GENERIC prompt/memory changes (the private set has 110 unseen
games) that make the model clear hard levels faster, to stack on the current arm.

## The run you are reading
SF = `stall-fix-10x` (51.85, 8 wins). It is `mem-unified-10x` (U, 56.09, 8 wins) plus six default-off flags:
- `ARC3_ACTION_ERROR_RAISE` + `ARC3_ACTION_ALIASES`: a refused `action()` call prints `[action] NOT EXECUTED: ...` and
  raises; U/D/L/R work as arrows. (0 firings in SF.)
- `ARC3_MEM_COLOURS_FIRST`: the "Colours on the board not clicked on this level" row comes first in `<this_level>`, is
  never cut, and names colours new on this level with their location. ALSO: the 700-char `<this_level>` budget loop
  now SKIPS a row that does not fit (`continue`) instead of stopping (`break`), so later short rows now appear where
  they used to be dropped. This changes `<this_level>` content in every game, including keyboard games.
- `ARC3_DEATH_VERDICT`: the GAME OVER message gives a computed budget-bar reading and the attempt's undone actions.
- `ARC3_UNDONE_FACT`: a turn-message line "Action N X was undone by the game: it animated, then the board returned to
  how it was (kth undone action in this attempt; cells that changed during the animation: rows .., cols ..)".
- `ARC3_CLICK_RUNS`: a dead multi-click run shown beside earlier effective runs of the same length.
U's memory (also on in SF): `<previous_levels>` with each cleared level's route + the model's recap; `<this_level>`
(per-action-type effect counts, untried actions, "board seen before", "Rules you confirmed on this level: ..." quoting
the model's own earlier statements, moved-objects and touching lines); rules from previous levels.

## Established (don't re-derive)
- A plain restart loses in simulation. Stalls are not loops of identical experiments. Each full run has ~30 stalled
  levels (>= 60k tokens or >= 45 min); which hard levels stall varies by run. Earlier single-run stall reads found
  false negatives from invalid/misread tests; SF fixed those specific ones and they did not recur.
- Model: Qwen3.8-Flash-Next W4A16 (small, local). Harness-computed FACTS in tool results / turn messages can work;
  ADVICE text is ignored or hurts; optional structure is ignored; required params get filled but not used.
  State-lab (k/20 right next moves at frozen stuck states, control -> with fact): "moved objects since level start"
  0 -> 16; "touching same-colour objects listed separately" 5 -> 10; "your own confirmed rules quoted" 3 -> 9;
  "chaser stands where you were 2 moves ago" 1 -> 5; re-probe / untried-action / action-count / clock facts: no effect.

## Data (read-only)
Compact extracts: `lab2/extract/<run>/<game>.requests.jsonl` (one row per model request, 0-based `i`): i, step,
action (number of the NEXT game action), level (level at request time), within_turn, n_msgs, last_role, last_text
(newest message: tool result or turn message, <= 6000 chars), reasoning (model's thinking for its reply), content,
code (python it ran), usage (prompt/completion tokens), finish. `lab2/extract/<run>/<game>.actions.jsonl`: one row
per game action: n, a, level (level AFTER the action), step, cleared, over (GAME OVER), changed, batch.
Runs with extracts: SF (this run), U, Ucore, A, B, C, comb, keep, ms1, ms2, pf, uc1, uc2, uc3 (not every game).
Raw logs (big; parse with scripts, never cat): SF = `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs/v4-stall-fix-10x/v1`
(`<gid>_p0_requests.jsonl` = full messages of every request; `artifacts/<gid>_p0_events.jsonl` = every action's 64x64
board, plus analysis transcripts with HH:MM:SS timestamps; `score_timeline.jsonl` = levels/tokens vs `t_bench_s`).
Other runs' dirs: RUNS in `lab2/extract.py`. Earlier stall reads (other runs): `lab2/stalls/*.md`.
Paths are relative to S = /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad.
Harness code (to check what a line means): `S/wt/stall-fix/ARC3-Inference/inference/agent/` (`memory_layout.py`,
`tool_agent.py`, `death_facts.py`, `click_runs.py`, `board_facts.py`).

## What to return (your final message IS the report; <= 900 words, plain markdown)
1. The level: mechanics and the winning route (from the fastest run's actions), and the fast run's key insight;
   quote its reasoning at its turning point (run, request index).
2. The SF stall as a timeline (requests, tokens, minutes; phases): what the model believed and did, with short quotes
   and request indices.
3. MEMORY AND PROMPT AUDIT (the main part): at the stall's decisive moments, what did `<this_level>`,
   `<previous_levels>`, the rules lines, and the new SF lines say? For each: correct / misleading / stale / noise /
   missing. Did the model cite or use them (quote)? Did any SF change (including the `continue` budget change, which
   adds rows) plausibly slow this level down? Compare with what the fast run's memory said at its turning point:
   was the difference in the memory, or in the model's own exploration?
4. Classify the stall's tokens (rough shares): (a) misperception a harness fact could fix, (b) wrong hypothesis held
   against evidence, (c) execution/tooling failure, (d) loop, (e) starvation/time, (f) needs an insight no generic
   fact gives, (g) misled by memory/prompt content (name the line).
5. ONE generic prompt or memory change that would have shortened THIS stall: exact text/behaviour, the detectable
   signal that triggers it (not hindsight), the request index where it would have fired, and whether it also applies
   to other games. Prefer facts and corrections of existing memory over new advice. Say if your best answer is
   "remove/repair an existing line" rather than "add one".
Constraints: read-only (never edit repos, run dirs or worktrees; do not write files outside your own reasoning);
run python under `nice -n 15`, one heavy job at a time; never execute game engine code.
