# Provenance of vendored Tycho code

Source: https://github.com/NIMI-research/Tycho at commit `f68912a` (2026-07-29, "Initial
release"), Apache-2.0, copyright 2026 Jens Lehmann, Andrei Aioanei, Sahar Vahdati. A read-only
clone lives on the GB10 at `~/arc3/reference/Tycho`.

| file here | upstream path | status |
|---|---|---|
| `agent.py` | `tycho/harness/agent.py` | verbatim |
| `actions.py` | `tycho/harness/actions.py` | verbatim |
| `scoring.py` | `tycho/harness/scoring.py` | verbatim |
| `harness.py` | `tycho/harness/harness.py` | **modified**: `run_env` keeps the engine loop, per-level action accounting, budget, no-op/revisit diagnostics, `TraceStep`/`EnvRunRecord`/`_build_record`; removed the resume journal, actor snapshot/restore, workspace callbacks (`record_solved`/`record_terminal`/`record_game_over`), status store and builder counters; added `start_level` (per-level tests via `set_level`), `deadline`, `on_step`, and a `None` budget when baselines are hidden (competition mode). |
| `serving.py` | `tycho/serving/public_backends.py` (`_post_json`, `normalize_usage`) and `tycho/serving/llm_client.py` (`_with_retry`, `is_context_limit_error`, `_is_retryable`, `_CONTEXT_MARKERS`) | **modified**: extracted the transport helpers only; the Anthropic/Responses backends, plugin extension mechanism, recording and telemetry are not carried. `openai_chat_tools` is replaced by `openai_chat_json` in `v9/client.py` (structured `response_format`, explicit `enable_thinking`). |
| `planner.py` | `tycho/workspace/wmlib_template.py` (`UNKNOWN`, `clone`, `_hashable`, `_cell_stats`, `_coverage_status`, `_predicts_change`, `_planner_key`, `_planner_successor`, `_acts`, `_norm_action`, `_target`, `_is_target`, `_should_prune`, `plan_bfs_with_diagnostics`, `plan_bfs`, `plan_astar_with_diagnostics`, `plan_astar`, `validate_plan`) | **modified**: extracted from the 1773-line workspace library; the file/workspace readers (`frames`, `transitions`, `verify`, ...) are not carried; `outcome`/`is_level_complete`/`is_game_over` are kept as the model contract. |

Nothing under `upstream/` in this repository was touched; the clone on the Spark is read-only.
