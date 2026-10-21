"""Agents that plug into the vendored Tycho `run_env` loop.

`RandomAgent` exists to prove the engine plumbing with no model at all (step 1 of the plan).
`V9Agent` is the real harness: it pops committed actions from a batch queue and only runs the
turn graph (a model request or several) when the queue is empty or a prediction was violated.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from arcengine import GameAction

from .vendor.tycho.agent import ActionChoice, Agent


class RandomAgent(Agent):
    """Uniform random over the frame's game actions (never RESET). Plumbing proof only."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def reset(self, game_id: str, available_actions: list[GameAction]) -> None:
        self.game_id = game_id

    def choose_action(self, frames, latest_frame, available_actions) -> ActionChoice:
        options = [a for a in available_actions if a != GameAction.RESET] or list(available_actions)
        action = self.rng.choice(options)
        if action == GameAction.ACTION6:
            return ActionChoice(action=action, x=self.rng.randrange(64), y=self.rng.randrange(64),
                                reasoning={"src": "random"})
        return ActionChoice(action=action, reasoning={"src": "random"})


class V9Agent(Agent):
    """The v9 harness as a Tycho Agent. The turn graph lives in v9.nodes; this class only owns
    the batch queue and the per-step evidence hand-off.

    Contract with run_env: `choose_action` is called once per engine step. Between model turns
    the agent replays the committed batch one action at a time; after every step the harness
    (in `observe`, called from the next `choose_action`) compares the prediction for the previous
    step with the observed delta and, on the first violation, flushes the rest of the batch.
    """

    name = "v9"

    def __init__(self, session: Any) -> None:
        # `session` is a v9.session.GameSession (built in step 5); it holds the ledger, memory,
        # model client, and the graph. Kept as an opaque handle here so this module has no model
        # dependency and the random smoke path stays importable without pydantic/requests.
        self.session = session

    def reset(self, game_id: str, available_actions: list[GameAction]) -> None:
        self.session.reset(game_id, available_actions)

    def choose_action(self, frames, latest_frame, available_actions) -> ActionChoice:
        return self.session.choose_action(frames, latest_frame, available_actions)

    def is_done(self, frames, latest_frame) -> bool:
        return self.session.is_done()

    def note_final_observation(self, frame, available_actions) -> None:
        self.session.note_final_observation(frame, available_actions)

    def note_external_reset(self) -> None:
        self.session.note_external_reset()

    @property
    def done_reason(self) -> Optional[str]:
        return getattr(self.session, "done_reason", None)
