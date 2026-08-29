"""All the numeric simulation logic lives here, independent of PettingZoo
and independent of the renderer. This is deliberate: it means

    python -m tests.test_physics

can validate the sim with nothing but numpy installed, and the same
functions run identically whether you're training headless at 1000 steps/sec
or watching a raylib window at 20 steps/sec.
"""

import numpy as np

from football_env import constants as C
from football_env.entities import Ball, Player


def step_player(player: Player, move_action: np.ndarray, dt: float) -> None:
    """move_action is a 2D vector in [-1, 1]^2: desired direction * intensity.

    We accelerate the player towards (move_action * max_speed) rather than
    setting velocity directly -- this is what makes motion look like a
    football player and not a top-down twin-stick shooter.
    """
    move_action = np.clip(move_action, -1.0, 1.0)
    target_velocity = move_action * player.max_speed
    delta = target_velocity - player.velocity
    delta_norm = np.linalg.norm(delta)

    if delta_norm < 1e-6:
        accel = np.zeros(2, dtype=np.float32)
    else:
        # Accelerate towards target, but decelerating (no input) uses a
        # separate, slightly gentler constant so stopping feels natural.
        rate = C.PLAYER_MAX_ACCEL if np.linalg.norm(move_action) > 0.05 else C.PLAYER_FRICTION
        step = min(rate * dt, delta_norm)
        accel = (delta / delta_norm) * step

    player.velocity = player.velocity + accel
    speed = np.linalg.norm(player.velocity)
    if speed > player.max_speed:
        player.velocity = player.velocity / speed * player.max_speed

    player.position = player.position + player.velocity * dt
    clamp_to_pitch_bounds(player.position, C.PLAYER_RADIUS)


def clamp_to_pitch_bounds(position: np.ndarray, radius: float) -> None:
    """In-place clamp so players can't run off the edge of the world.
    (The ball has its own, separate out-of-bounds handling since going
    out is a meaningful event for the ball, not just a wall.)
    """
    position[0] = np.clip(position[0], -C.HALF_LENGTH - 5, C.HALF_LENGTH + 5)
    position[1] = np.clip(position[1], -C.HALF_WIDTH - 5, C.HALF_WIDTH + 5)


def step_ball(ball: Ball, dt: float) -> None:
    if ball.owner_id is None:
        speed = np.linalg.norm(ball.velocity)
        if speed > 1e-6:
            decel = min(C.BALL_FRICTION * dt, speed)
            ball.velocity = ball.velocity - (ball.velocity / speed) * decel
        ball.position = ball.position + ball.velocity * dt


def snap_ball_to_owner(ball: Ball, players: dict) -> None:
    """A dribbling player carries the ball just ahead of their feet."""
    if ball.owner_id is None:
        return
    owner = players[ball.owner_id]
    if np.linalg.norm(owner.velocity) > 0.3:
        heading = owner.velocity / np.linalg.norm(owner.velocity)
    else:
        heading = np.array([1.0, 0.0], dtype=np.float32) if owner.team_id == 0 else np.array([-1.0, 0.0], dtype=np.float32)
    ball.position = owner.position + heading * (C.PLAYER_RADIUS + C.BALL_RADIUS + 0.15)
    ball.velocity = owner.velocity.copy()


def resolve_possession(ball: Ball, players: dict) -> None:
    """Free (unowned) balls are picked up by the nearest player within
    POSSESSION_RADIUS. A slow-moving ball near a player is easy to trap;
    a fast ball requires the player to be even closer (simple heuristic
    so you can't just stand still and vacuum up every pass).
    """
    if ball.owner_id is not None:
        return

    best_id, best_dist = None, None
    for agent_id, p in players.items():
        dist = np.linalg.norm(p.position - ball.position)
        speed_penalty = min(np.linalg.norm(ball.velocity) * 0.03, 0.5)
        if dist <= (C.POSSESSION_RADIUS - speed_penalty):
            if best_dist is None or dist < best_dist:
                best_id, best_dist = agent_id, dist

    if best_id is not None:
        ball.owner_id = best_id
        ball.last_touch_team = players[best_id].team_id
        ball.velocity = np.zeros(2, dtype=np.float32)


def apply_kick(ball: Ball, player: Player, kick_type: int, kick_dir: np.ndarray) -> None:
    """kick_type in {KICK_PASS, KICK_SHOOT, KICK_CLEAR}. Releases possession
    and sends the ball off with the corresponding power in kick_dir.
    """
    norm = np.linalg.norm(kick_dir)
    direction = kick_dir / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    power = C.KICK_POWERS.get(kick_type, C.KICK_POWER_PASS)

    ball.owner_id = None
    ball.velocity = direction * power
    ball.last_touch_team = player.team_id
    ball.position = player.position + direction * (C.PLAYER_RADIUS + C.BALL_RADIUS + 0.1)


def check_goal(ball: Ball) -> int:
    """Returns the team_id that SCORED (i.e. the ball crossed the opponent's
    goal line), or None. Team 0 defends x=-HALF_LENGTH, Team 1 defends
    x=+HALF_LENGTH, so a ball crossing the +X line is scored BY team 0.
    """
    if abs(ball.position[1]) > C.GOAL_HALF_WIDTH:
        return None
    if ball.position[0] >= C.HALF_LENGTH:
        return 0
    if ball.position[0] <= -C.HALF_LENGTH:
        return 1
    return None


def check_out_of_bounds(ball: Ball) -> bool:
    """Simplified out-of-bounds: touchlines only trigger a reset-in-place
    (no real throw-in physics/animation). Goal-line-but-missing-the-goal
    is treated as a goal kick in the same simplified way. This is a
    deliberate scope cut -- see README 'Simplifications' section.
    """
    return bool(abs(ball.position[1]) > C.HALF_WIDTH or abs(ball.position[0]) > C.HALF_LENGTH + C.GOAL_DEPTH)
