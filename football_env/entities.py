"""Lightweight state containers for players and the ball.

These are plain data holders updated in-place by physics.py each step.
Keeping them dumb (no behavior) makes the physics/observation code easy
to unit test in isolation.
"""

from dataclasses import dataclass, field

import numpy as np

from football_env import constants as C


@dataclass
class Player:
    agent_id: str          # e.g. "team_0_player_03"
    team_id: int            # 0 or 1
    role: str                # GK / DF / MF / FW
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=C.np_zeros2)
    home_position: np.ndarray = field(default_factory=C.np_zeros2)

    @property
    def max_speed(self) -> float:
        if self.role == C.GK:
            return C.PLAYER_MAX_SPEED * C.GK_MAX_SPEED_MULT
        return C.PLAYER_MAX_SPEED


@dataclass
class Ball:
    position: np.ndarray = field(default_factory=C.np_zeros2)
    velocity: np.ndarray = field(default_factory=C.np_zeros2)
    owner_id: str = None    # agent_id of the player currently in control, or None
    last_touch_team: int = None  # team_id of whoever touched it last (for goal credit)
