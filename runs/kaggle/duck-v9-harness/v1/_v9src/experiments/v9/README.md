# v9: the harness owns the facts

A ground-up ARC-AGI-3 harness. The harness owns perception, the transition ledger, memory,
confidence numbers and the decision to spend an action; the model proposes interpretations,
memory patches, hypotheses and actions as typed JSON that the harness validates. Nothing the
model says becomes true without a check against the ledger.

Plan and decisions: `~/.claude/plans/also-evolve-the-idea-hidden-oasis.md` (approved
2026-09-10). Base: a vendored subset of Tycho (Apache-2.0, `vendor/tycho/`, see NOTICE and
PROVENANCE.md). Engine: SGLang serving Qwen3.8-27B-FP8 with thinking off (vLLM is the fallback).

## Loop

Every engine step: frame -> `perception.perceive` -> `tracker.update` (stable ids) ->
`diffing.compute_delta` -> `predict.compare` against the saved prediction -> one immutable
`ledger.Record`. On the first violated claim the rest of the committed batch is dropped.

Every model turn (queue empty or halted): `nodes.py` graph `recall -> world -> goal -> mode ->
propose -> commit`. `modes.mode_of` picks ORIENT / EXPLORE / VERIFY / EXECUTE / RECOVER from
harness state only; `modes.ROLE_PLAN` says which roles run (orient, interpret, decide, critic,
hypothesis) and with what max_tokens. Every role output is validated (`schemas.py`) and applied
by `session.apply_role_output`; rules the model proposes are compiled and scored by `rules.py`
against the whole ledger before they count.

## Run

    cd experiments && python -m v9.run --games re86 --level 3 --passes 2 --cap-minutes 70 \
        --base-url http://127.0.0.1:30000/v1 --model Qwen/Qwen3.8-27B-FP8 --out ../runs/<name>
    python -m v9.run --games re86 --agent random --max-actions 300 --out ../runs/smoke   # no model
    python -m v9.run --games re86 --agent fake --max-actions 200 --out ../runs/fake        # scripted model

Per-level tests should be seeded with what the harness knew at the boundary:
`--memory-from runs/<earlier>/<game>_p0.memory.json` restores labels, goals, questions, slots,
the level log, the rules (re-scored here before use) and the control map at half confidence;
ledger facts are never seeded. Read a run with `python -m v9.tools.report <run_dir> --calls`;
probe a server with `python -m v9.tools.probe_server`.

Outputs: `benchmark.json` (read by scripts/summarize_run.py), `stdout.log` with the
`[finished]` and `LEVEL_START_RESULT` lines, `run_manifest.json`, and per pass
`<game>_p<k>.{steps,turns,calls}.jsonl`, `.prompts.txt`, `.ledger.jsonl`, `.memory.json`.
On the Spark: `scripts/v9_level.sh` (queue-compatible driver), `scripts/serve_sglang_27b.sh`.

Tests: `scripts/spark_test.sh experiments/v9 -q` (never on the Mac).

## Files

| file | job |
|---|---|
| `engine.py` | open the offline Arcade, make a game, jump to a level |
| `vendor/tycho/harness.py` | the engine loop (`run_env`), per-level accounting, budget |
| `perception.py`, `tracker.py`, `diffing.py` | the two trusted inputs: deterministic perception and delta |
| `ledger.py`, `predict.py` | immutable records; prediction contract and comparison |
| `rules.py`, `control.py`, `facts.py` | typed rule DSL with evidence-computed confidence; control map; passability, consumption, precondition miner |
| `memory.py` | hypotheses, questions, goals, model-authored slots, cross-game store |
| `modes.py`, `critic.py`, `graph.py`, `nodes.py` | mode controller, commit gate, critic mapping, the turn graph |
| `schemas.py`, `prompts.py`, `client.py` | the model boundary: schemas, validators, job-framed prompts, structured calls |
| `session.py`, `agent.py` | per-game state and the Tycho `Agent` adapter |
| `repl.py`, `repl_child.py` | read-only persistent REPL for the model (ORIENT/RECOVER only) |
| `fakemodel.py`, `results.py`, `run.py` | scripted model, output writers, CLI |
