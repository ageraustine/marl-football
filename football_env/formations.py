"""Builds starting positions for a team of a given size.

Kept separate from constants.py so curriculum training (2v2 -> 11v11) can
call build_formation(n) and get sensible spread-out positions for any n,
falling back to interpolation when n doesn't exactly match a preset.
"""

import numpy as np

from football_env import constants as C


def build_formation(n_players: int, attacking_right: bool):
    """Return list of (role, x, y) in pitch coordinates for n_players.

    attacking_right=True means this team's goal is on the LEFT (Team 0
    convention) and it lines up facing +X. attacking_right=False mirrors
    everything in X so the team faces -X (Team 1 convention).
    """
    preset = C.FORMATIONS_BY_SIZE.get(n_players)
    if preset is None:
        # Fall back: reuse the 11-a-side shape and just take the first n
        # entries (GK first, then defenders/midfielders/forwards in order).
        preset = C.FORMATION_11[:n_players]

    positions = []
    for role, x_frac, y_frac in preset:
        x = -C.HALF_LENGTH + x_frac * C.HALF_LENGTH
        y = y_frac * C.HALF_WIDTH
        if not attacking_right:
            x = -x
        positions.append((role, np.array([x, y], dtype=np.float32)))
    return positions
