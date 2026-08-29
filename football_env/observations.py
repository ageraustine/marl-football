"""Builds a fixed-size, egocentric observation vector per agent.

Fixed-size padding (MAX_TEAMMATES / MAX_OPPONENTS) is what lets a policy
trained on 3v3 be reused as the initialization for 5v5 and then 11v11
during curriculum learning -- the observation *shape* never changes,
only how many of the padded slots are "real" (mask=1) vs empty (mask=0).
"""

import numpy as np

from football_env import constants as C

MAX_TEAMMATES = 10   # 11-a-side minus self
MAX_OPPONENTS = 11

ROLE_ONE_HOT = {C.GK: 0, C.DF: 1, C.MF: 2, C.FW: 3}
N_ROLES = 4

# self(7: pos2+vel2+has_ball1+time1+team_id1) + ball(5) + goals(4) + role(4)
# + teammates(10*5) + opponents(11*5)
SELF_FEATURES = 7
OBS_SIZE = SELF_FEATURES + 5 + 4 + N_ROLES + MAX_TEAMMATES * 5 + MAX_OPPONENTS * 5


def _norm_pos(p):
    return np.array([p[0] / C.HALF_LENGTH, p[1] / C.HALF_WIDTH], dtype=np.float32)


def _norm_vel(v):
    return np.array(v, dtype=np.float32) / C.PLAYER_MAX_SPEED


def build_observation(agent_id, players, ball, step_count, max_steps) -> np.ndarray:
    me = players[agent_id]
    own_goal_x = C.TEAM_0_GOAL_X if me.team_id == 0 else C.TEAM_1_GOAL_X
    opp_goal_x = C.TEAM_1_GOAL_X if me.team_id == 0 else C.TEAM_0_GOAL_X

    chunks = []

    # --- self: position, velocity, has_ball, time_remaining, team_id ---
    self_chunk = np.concatenate([
        _norm_pos(me.position),
        _norm_vel(me.velocity),
        [1.0 if ball.owner_id == agent_id else 0.0],
        [1.0 - step_count / max_steps],
        [float(me.team_id)],
    ]).astype(np.float32)
    chunks.append(self_chunk)

    # --- ball: relative position, relative velocity, is_free ---
    rel_ball_pos = (ball.position - me.position)
    rel_ball_pos_norm = np.array([rel_ball_pos[0] / C.PITCH_LENGTH, rel_ball_pos[1] / C.PITCH_WIDTH], dtype=np.float32)
    ball_chunk = np.concatenate([
        rel_ball_pos_norm,
        _norm_vel(ball.velocity),
        [1.0 if ball.owner_id is None else 0.0],
    ]).astype(np.float32)
    chunks.append(ball_chunk)

    # --- goals: relative vector to own goal and opponent goal ---
    goals_chunk = np.array([
        (own_goal_x - me.position[0]) / C.PITCH_LENGTH, -me.position[1] / C.PITCH_WIDTH,
        (opp_goal_x - me.position[0]) / C.PITCH_LENGTH, -me.position[1] / C.PITCH_WIDTH,
    ], dtype=np.float32)
    chunks.append(goals_chunk)

    # --- role one-hot ---
    role_chunk = np.zeros(N_ROLES, dtype=np.float32)
    role_chunk[ROLE_ONE_HOT[me.role]] = 1.0
    chunks.append(role_chunk)

    # --- teammates / opponents, sorted by distance, padded ---
    teammates, opponents = [], []
    for other_id, p in players.items():
        if other_id == agent_id:
            continue
        (teammates if p.team_id == me.team_id else opponents).append(p)

    teammates.sort(key=lambda p: np.linalg.norm(p.position - me.position))
    opponents.sort(key=lambda p: np.linalg.norm(p.position - me.position))

    def entity_rows(entities, max_n):
        rows = np.zeros((max_n, 5), dtype=np.float32)
        for i, p in enumerate(entities[:max_n]):
            rel = p.position - me.position
            rows[i] = [
                rel[0] / C.PITCH_LENGTH, rel[1] / C.PITCH_WIDTH,
                p.velocity[0] / C.PLAYER_MAX_SPEED, p.velocity[1] / C.PLAYER_MAX_SPEED,
                1.0,  # mask: this slot is a real player
            ]
        return rows.flatten()

    chunks.append(entity_rows(teammates, MAX_TEAMMATES))
    chunks.append(entity_rows(opponents, MAX_OPPONENTS))

    return np.concatenate(chunks).astype(np.float32)
