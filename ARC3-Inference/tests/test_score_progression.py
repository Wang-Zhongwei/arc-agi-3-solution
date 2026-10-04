from types import SimpleNamespace as NS

import pytest

from inference.tools.eval import GameTrialResult
from inference.tools.score_progression import (
    build_action_progression,
    build_progression_lanes,
    build_score_progression,
    game_action_progression,
    game_score_progression,
)


def trial(game_id, points, actions=None, run_name='pass-0'):
    return GameTrialResult(game_id, run_name, 0, 'test', score_progression=points, action_progression=actions)


def test_generated_tokens_and_final_no_move_cost():
    from taaf.game import GameRun

    run = GameRun('test', 2, [1, 1], actions_per_level=[1, 1], levels_completed=2)
    run.history = [NS(generated_tokens=10, uncached_input_tokens=900),
                   NS(generated_tokens=20, uncached_input_tokens=800)]
    run.intermediate_states = [NS(raw=NS(levels_completed=n)) for n in [0, 1, 2]]
    run.final_generated_tokens = 5
    run.final_score = run._compute_final_score()
    assert game_score_progression(run, run.final_score) == [
        (0, 0), (10, pytest.approx(100 / 3)), (30, 100), (35, 100)]


def test_missing_or_inconsistent_history_is_not_a_curve():
    from taaf.game import GameRun

    run = GameRun('test', 1, [1])
    run.history = [NS(generated_tokens=10)]
    assert game_score_progression(run, 0) is None
    run.history = []
    run.intermediate_states = [NS(raw=NS(levels_completed=0))]
    assert game_score_progression(run, 50) is None
    result = build_score_progression([trial('a', [(0, 0), (10, 20)]), trial('b', None)])
    assert result['status'] == 'unavailable'
    assert result['trials_with_history'] == 1
    assert result['points'] == []


def test_pooling_equal_games_with_unequal_pass_counts_and_stopped_trials():
    result = build_score_progression([
        trial('a', [(0, 0), (10, 30), (20, 30)]),
        trial('a', [(0, 0), (10, 60), (20, 60)]),
        trial('b', [(0, 0), (40, 90)]),
    ])
    assert result['status'] == 'complete'
    assert result['points'] == [[0, 0], [10, 22.5], [20, 22.5], [30, 67.5]]


def test_export_includes_curve(monkeypatch):
    from inference.tools import eval as evaluation

    monkeypatch.setattr(evaluation, '_completed_trial_dirs', lambda _: [])
    game = trial('a', [(0, 0), (10, 25)])
    summary = evaluation.EvaluationSummary(25, 0, [evaluation.RunEvaluation('pass-0', 25, [game])], [])
    payload = evaluation.build_evaluation_payload(summary, run_dirs=[])
    assert payload['score_progression']['points'][-1] == [10, 25]
    assert payload['action_progression']['status'] == 'unavailable'
    assert payload['progression_lanes']['tokens'] == {'a': {'pass-0': [[0, 0], [10, 25]]}}


def test_zero_action_trial_retains_final_generated_token_cost():
    from taaf.game import GameRun

    run = GameRun('test', 1, [1], final_generated_tokens=17)
    assert game_score_progression(run, 0) == [(0, 0), (17, 0)]


def test_median_averages_passes_before_taking_median_across_games():
    result = build_score_progression([
        trial('a', [(0, 0), (10, 0)]),
        trial('a', [(0, 0), (10, 100)]),
        trial('b', [(0, 0), (20, 10)]),
        trial('c', [(0, 0), (30, 90)]),
    ])
    assert result['median_points'][-1] == [20, 50]
    assert result['points'][-1] == [20, 50]
    assert result['median_points'][1][1] == 0
    assert result['points'][1][1] == pytest.approx(50 / 3)
    assert [p[0] for p in result['median_points']] == [p[0] for p in result['points']]


def test_even_game_median_includes_zero_scores():
    result = build_score_progression([
        trial('a', [(0, 0), (10, 0)]),
        trial('b', [(0, 0), (10, 10)]),
        trial('c', [(0, 0), (10, 30)]),
        trial('d', [(0, 0), (10, 100)]),
    ])
    assert result['median_points'][-1] == [10, 20]
    assert result['points'][-1] == [10, 35]


def test_action_axis_steps_one_unit_per_action():
    from taaf.game import GameRun

    run = GameRun('test', 2, [1, 2], actions_per_level=[1, 2], levels_completed=2)
    run.history = [NS(generated_tokens=10), NS(generated_tokens=20), NS(generated_tokens=5)]
    run.intermediate_states = [NS(raw=NS(levels_completed=n)) for n in [0, 1, 1, 2]]
    run.final_generated_tokens = 7
    run.final_score = run._compute_final_score()
    assert game_action_progression(run, run.final_score) == [
        (0, 0), (1, pytest.approx(100 / 3)), (3, 100)]
    assert game_score_progression(run, run.final_score)[-1] == (42, 100)


def test_zero_action_trial_is_flat_at_zero_actions():
    from taaf.game import GameRun

    run = GameRun('test', 1, [1], final_generated_tokens=17)
    assert game_action_progression(run, 0) == [(0, 0), (0, 0)]


def test_action_progression_pools_like_token_progression():
    result = build_action_progression([
        trial('a', None, [(0, 0), (4, 50)]),
        trial('b', None, [(0, 0), (2, 100), (8, 100)]),
    ])
    assert result['x_metric'] == 'mean_actions_per_trial'
    assert result['points'] == [[0, 0], [2, 50], [4, 75], [6, 75]]


def test_lanes_keep_each_trial_and_skip_missing_histories():
    lanes = build_progression_lanes([
        trial('a', [(0, 0), (10, 30)], [(0, 0), (2, 30)], run_name='v1'),
        trial('a', None, None, run_name='v2'),
        trial('b', [(0, 0), (5, 0)], [(0, 0), (1, 0)], run_name='v1'),
    ])
    assert lanes['tokens'] == {'a': {'v1': [[0, 0], [10, 30]]}, 'b': {'v1': [[0, 0], [5, 0]]}}
    assert lanes['actions']['a'] == {'v1': [[0, 0], [2, 30]]}
