# SPEC: stall-fix (2026-10-14)

## Why
Goal: 10 of 25 games fully won in the 121-min Kaggle run. The last two full runs (U = mem-unified-10x 56.09, Ucore =
mem-unified-core-10x 51.15, 8 wins each) lost wins to single-level stalls (70-100 wall-min, 100-145k tokens on one
level). Seven stall reads (`lab2/stalls/*.md`, read the ones named in your task) found that six of seven stalls were
FALSE NEGATIVES FROM INVALID OR MISREAD TESTS: the model ran a test that silently failed or was misread, concluded
"X does not work / the mechanic is Y", and held that belief for 50-90k tokens. In most cases the harness hid or
blurred the evidence. A plain restart of the level loses in simulation; stalls are not loops of identical
experiments. The fix family: make test validity visible with harness-computed FACTS (never advice text: this
model ignores advice and optional structure; facts in tool results / turn messages work, e.g. state-lab
"moved objects" 0 -> 16/20).

The final run is ONE combined 121-min run tonight: arm `stall-fix-10x` = `mem-unified-10x` + all flags below.
Each change sits behind its own default-off env flag so it can be bisected later.

## Flags (three builders, one worktree each, all off `arm/mem-unified` head addf9879)
- Builder TOOL (`arm/stall-fix-tool`, worktree `wt/sf-tool`): `ARC3_ACTION_ERROR_RAISE`, `ARC3_ACTION_ALIASES`,
  `ARC3_MEM_COLOURS_FIRST`. Evidence: `lab2/stalls/U_m0r0_L3.md`, `lab2/stalls/Ucore_vc33_L4.md`, `Ucore_tn36_L4.md`.
- Builder DEATH (`arm/stall-fix-death`, `wt/sf-death`): `ARC3_DEATH_VERDICT`, `ARC3_UNDONE_FACT`. Evidence:
  `lab2/stalls/U_tu93_L4.md`, `lab2/stalls/Ucore_r11l_L2.md`.
- Builder CLICKS (`arm/stall-fix-clicks`, `wt/sf-clicks`): `ARC3_CLICK_RUNS`. Evidence: `lab2/stalls/U_sc25_L5.md`.

## Shared rules
- Paths: S = /private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/72bf2785-3acc-4763-9843-90d6df62b6a1/scratchpad.
  Harness code: `<worktree>/ARC3-Inference/inference/` (agent loop `agent/tool_agent.py`, sandbox
  `agent/python_tool_sandbox.py`, host/guards `framework/solver.py`, memory layout `agent/memory_layout.py`,
  board facts `agent/board_facts.py`, action names `agent/action_names.py`). Option reference
  `ARC3-Inference/CONFIGURATION.md` (the mem-unified rows ~line 590-605 show the house style).
- Python: `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/daniel-franzen-sol/ARC3-Inference/.venv/bin/python`
  run with cwd = `<worktree>/ARC3-Inference` (the `inference` package then resolves to your worktree; check once with
  `python -c "import inference; print(inference.__file__)"`). Always `nice -n 15`. Targeted tests freely; the FULL
  suite only once at the end, under the shared lock:
  `until mkdir $S/suite.lock 2>/dev/null; do sleep 30; done; nice -n 15 <python> -m pytest -q -x tests; rmdir $S/suite.lock`
  (always rmdir the lock, even on failure).
- Flags off => byte-identical requests to `arm/mem-unified` (the golden tests `tests/test_mem_unified*.py`,
  `tests/mem_unified_golden.py`, `tests/fixtures/pf_keep_v2_golden.json` must still pass). Read flags at call time
  like the neighbouring code does. If a flag must reach the sandbox subprocess, pass it the way existing flags are
  (e.g. `initial.get("anim_truth")`).
- Real data, not made-up boards: raw logs of the two runs are in
  `/Users/advaith/Documents/studying-stuff/experiments/arc-agi-v4/kaggle-outputs/v4-mem-unified-10x/v1` (U) and
  `.../v4-mem-unified-core-10x/v1` (Ucore): `<gid>_p0_requests.jsonl` (every request's full messages, and responses
  with usage), `artifacts/<gid>_p0_events.jsonl` (every action with its 64x64 `board`, level, game_over,
  board_changed, action_display; analysis steps). Older runs (xz) are under
  `daniel-franzen-sol/runs/kaggle/*/v*/`. Treat them as read-only data. Small test fixtures you cut from them go
  under `ARC3-Inference/tests/fixtures/` (gzip, keep each < 100 KB).
- Verify content, not placement: besides unit tests, REPLAY your detector over the full U and Ucore event logs
  (all 25 games each) and report per-game firing counts, and the exact text it would have produced at the
  reader-identified moments (named in your task). A fact that fires everywhere is noise; one that fires at the
  wrong moment misleads. Tune on that evidence and say what you tuned.
- Commits: small, one logical change each, imperative messages; end every commit message with these two lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01AULSqB7ZpbNcZV9DbF3pfR`
  Commit only to your branch; do NOT push, do NOT open PRs, do NOT edit `kaggle/arms.json` or `kaggle/runs.json`
  (the lead merges the three branches and defines the arm). Add your flags' rows to `CONFIGURATION.md`.
- Prompt strings are behaviour: add only the fact lines the spec names (wording may be tightened, keep it factual,
  no advice/imperatives to the model). Do not change any existing prompt text when your flags are off.
- Never run game engine code. Never edit run directories or other worktrees.
- Return (final message, <= 700 words): commits (hash + subject), what each flag does exactly (with one real example
  line), replay firing stats, the texts at the named moments, test results (targeted + full suite counts), and any
  deviation from the spec with the reason.
