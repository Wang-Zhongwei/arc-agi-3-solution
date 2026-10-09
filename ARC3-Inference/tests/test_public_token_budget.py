"""The validation token budget never constrains a competition rerun."""

from types import SimpleNamespace

import pytest
from taaf.deploy_kaggle import KaggleTarget

from inference.framework.solver import _HarnessGameSession


@pytest.mark.parametrize(
    "mode,limit,tokens,expected",
    [
        ("public", 10, 110, True),
        ("public", 10, 109, False),
        ("public", None, 110, False),
        ("public", 0, 110, False),
        ("private", 10, 110, False),
        ("emulated_private", 10, 110, False),
        ("local", 10, 110, False),
    ],
)
def test_only_public_kaggle_games_have_a_token_budget(mode, limit, tokens, expected):
    runtime = KaggleTarget() if mode != "local" else None
    if mode == "private":
        runtime.is_competition_rerun = True
    elif mode == "emulated_private":
        runtime.actual_run_as_submission = True
    session = object.__new__(_HarnessGameSession)
    session.solver = SimpleNamespace(
        runtime_environment=runtime, public_max_generated_tokens_per_game=limit
    )
    session.analyzer = SimpleNamespace(generated_tokens=tokens)
    session.game_token_baseline = 100
    assert session.token_limit_reached() is expected


@pytest.mark.parametrize(
    "mode,legacy_limit,public_limit,expected",
    [
        ("public", 10, None, True),
        ("public", 10, 20, False),
        ("public", 10, 0, False),
        ("private", 10, None, False),
        ("private", 10, 10, False),
        ("emulated_private", 10, 10, False),
        ("local", 10, None, True),
        ("local", 10, 20, True),
        ("local", None, 10, False),
    ],
)
def test_legacy_budget_remains_compatible(mode, legacy_limit, public_limit, expected):
    runtime = KaggleTarget() if mode != "local" else None
    if mode == "private":
        runtime.is_competition_rerun = True
    elif mode == "emulated_private":
        runtime.actual_run_as_submission = True
    session = object.__new__(_HarnessGameSession)
    session.solver = SimpleNamespace(
        runtime_environment=runtime,
        max_generated_tokens_per_game=legacy_limit,
        public_max_generated_tokens_per_game=public_limit,
    )
    session.analyzer = SimpleNamespace(generated_tokens=110)
    session.game_token_baseline = 100
    assert session.token_limit_reached() is expected


@pytest.mark.parametrize("private", [False, True])
def test_old_solver_without_public_budget_field(private):
    runtime = KaggleTarget()
    runtime.is_competition_rerun = private
    session = object.__new__(_HarnessGameSession)
    session.solver = SimpleNamespace(runtime_environment=runtime, max_generated_tokens_per_game=10)
    session.analyzer = SimpleNamespace(generated_tokens=110)
    session.game_token_baseline = 100
    assert session.token_limit_reached() is (not private)
