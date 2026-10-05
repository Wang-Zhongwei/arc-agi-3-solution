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
