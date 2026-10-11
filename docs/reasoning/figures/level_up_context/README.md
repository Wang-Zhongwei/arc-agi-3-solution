# Level-up context

Two figures, runs with the 118,272-token context budget only, one panel per game
(highest average score first, average over that budget's trials in the title):

- `all_games_level_up_context.png`: prompt tokens on the call that cleared each level.
- `all_games_level_up_history_steps.png`: earlier agent turns still in that call's history.

Filled dots are measured from request logs. Hollow dots are rebuilt from the
analysis transcripts by replaying each turn's messages and trimming them to the logged
`history_messages` count. Context is then estimated from text, thinking and image counts
with a linear model fitted on the request-logged runs: 4.3% median leave-one-run-out error,
history within ±1 step on 98% of calls. A run's rebuild is dropped if more than 2% of its
calls would exceed its own logged budget, which catches summary and image-window harnesses.
Grey dots have no recoverable value. Details are in
`../../data/level_up_context/reconstruction_validation.json`.

Per-turn clear rate after a game's first trim is flat across 50–120k of turn-start context
(7.3–8.1%, 11 request-logged runs, 5,795 turns). So extra retained history does not measurably
raise the chance that a turn clears a level.

Regenerate from `docs/reasoning/scripts/` with the ARC3-Inference venv:

```bash
KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" python recover_level_up_metrics.py  # stream runs missing locally
python level_up_context_all_runs.py                                                    # data + figures
```
