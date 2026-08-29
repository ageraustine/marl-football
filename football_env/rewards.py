"""Reward = sparse goal reward (the thing we actually care about) plus
small dense shaping terms (the thing that makes it learnable at all).

All shaping terms are deliberately tiny relative to the goal reward so
the policy doesn't learn to farm shaping reward instead of scoring
(e.g. just walking towards the ball forever without ever passing/shooting).
"""

import numpy as np

from football_env import constants as C

GOAL_REWARD = 10.0
CONCEDE_PENALTY = -10.0
BALL_PROXIMITY_WEIGHT = 0.01     # per meter closed, only when own team lacks possession
POSSESSION_ADVANCE_WEIGHT = 0.02  # per meter the ball moves towards opponent goal, in possession
OUT_OF_BOUNDS_PENALTY = -0.05
TIME_PENALTY = -0.0005            # tiny, discourages pure stalling


def compute_rewards(players, ball, prev_positions, prev_ball_pos, scoring_team, went_out, team_sizes):
    """prev_positions: dict agent_id -> np.ndarray (position last step)
    prev_ball_pos: np.ndarray, ball position last step
    scoring_team: 0, 1, or None
    went_out: bool, whether the ball left the pitch this step
    team_sizes: dict team_id -> number of players (for even shared-goal split)
    """
    rewards = {agent_id: TIME_PENALTY for agent_id in players}

    if scoring_team is not None:
        for agent_id, p in players.items():
            rewards[agent_id] += GOAL_REWARD if p.team_id == scoring_team else CONCEDE_PENALTY
        # Goal ends the point of dense shaping this step; still fall through
        # so out-of-bounds/possession terms don't double up incorrectly.

    ball_advance = ball.position[0] - prev_ball_pos[0]  # +X progress

    for agent_id, p in players.items():
        team_attacks_positive_x = (p.team_id == 0)
        advance = ball_advance if team_attacks_positive_x else -ball_advance

        if ball.owner_id is not None and players[ball.owner_id].team_id == p.team_id:
            rewards[agent_id] += POSSESSION_ADVANCE_WEIGHT * advance
        elif ball.owner_id is None or players[ball.owner_id].team_id != p.team_id:
            prev_dist = np.linalg.norm(prev_positions[agent_id] - prev_ball_pos)
            new_dist = np.linalg.norm(p.position - ball.position)
            closed = prev_dist - new_dist
            rewards[agent_id] += BALL_PROXIMITY_WEIGHT * max(closed, 0.0)

        if went_out and ball.last_touch_team == p.team_id:
            rewards[agent_id] += OUT_OF_BOUNDS_PENALTY

    return rewards
