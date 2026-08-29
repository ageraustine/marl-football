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

    Stamina is updated first, so player.max_speed (which reads stamina)
    reflects this step's fatigue before it's used to clamp velocity --
    that keeps "velocity never exceeds current max_speed" true at every
    step boundary, not just on average.
    """
    update_stamina(player, dt)

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
    """True if the ball has left the pitch (touchline or goal line) without
    a goal being scored. Does not classify *which* restart applies --
    see classify_restart() for that. No stoppage time or set-piece
    defensive positioning is modeled; see README 'Limitations'.
    """
    return bool(abs(ball.position[1]) > C.HALF_WIDTH or abs(ball.position[0]) > C.HALF_LENGTH + C.GOAL_DEPTH)


# ---------------------------------------------------------------------------
# Stamina
# ---------------------------------------------------------------------------

def update_stamina(player: Player, dt: float) -> None:
    """Sprinting (above STAMINA_SPRINT_THRESHOLD of base top speed) drains
    stamina; anything slower recovers it. Player.max_speed reads this
    value back to scale top speed, so fatigue compounds over a match.
    """
    speed = np.linalg.norm(player.velocity)
    speed_frac = speed / player.base_max_speed if player.base_max_speed > 0 else 0.0

    if speed_frac > C.STAMINA_SPRINT_THRESHOLD:
        player.stamina = max(0.0, player.stamina - C.STAMINA_DRAIN_RATE * dt)
    else:
        player.stamina = min(C.STAMINA_MAX, player.stamina + C.STAMINA_RECOVERY_RATE * dt)


# ---------------------------------------------------------------------------
# Offside
# ---------------------------------------------------------------------------

def team_attack_sign(team_id: int) -> int:
    """+1 if the team attacks towards +X, -1 if towards -X. Fixed for the
    whole match -- see README 'Limitations' re: no half-time end swap.
    """
    return 1 if team_id == 0 else -1


def offside_flagged_teammates(passer: Player, players: dict) -> set:
    """Called at the moment `passer` plays a KICK_PASS. Returns the set of
    the passer's own teammates who are in an offside position: ahead of
    both the ball and the second-last opponent, and in the opponent's
    half. Mirrors the Law 11 definition, simplified to not distinguish
    the goalkeeper as a special case (in practice the keeper is almost
    always the deepest player anyway, so this rarely changes the result).
    """
    sign = team_attack_sign(passer.team_id)
    opponents = [p for p in players.values() if p.team_id != passer.team_id]
    if not opponents:
        return set()

    # "Advancement" = position along this team's attacking direction;
    # higher = closer to the opponent's goal.
    opponent_advancement = sorted((p.position[0] * sign for p in opponents), reverse=True)
    second_last_defender_line = (
        opponent_advancement[1] if len(opponent_advancement) >= 2 else opponent_advancement[0]
    )
    ball_line = passer.position[0] * sign  # ball is at the passer's feet at the moment of the kick
    offside_threshold = max(second_last_defender_line, ball_line)

    flagged = set()
    for agent_id, p in players.items():
        if agent_id == passer.agent_id or p.team_id != passer.team_id:
            continue
        advancement = p.position[0] * sign
        if advancement > offside_threshold and advancement > 0:
            flagged.add(agent_id)
    return flagged


# ---------------------------------------------------------------------------
# Restarts: throw-in / corner kick / goal kick
# ---------------------------------------------------------------------------

def team_defending_goal_line(goal_x: float) -> int:
    """Which team's own goal line is at this x coordinate."""
    return 1 if goal_x > 0 else 0


def classify_restart(ball: Ball):
    """Determine the restart type for a ball that has gone out of bounds
    (check_out_of_bounds() must already be True). Returns
    (restart_type, defending_team) where defending_team is only
    meaningful for corner/goal_kick (whose goal line the ball crossed).
    """
    if abs(ball.position[1]) > C.HALF_WIDTH and abs(ball.position[0]) <= C.HALF_LENGTH:
        return C.RESTART_THROW_IN, None

    if abs(ball.position[0]) > C.HALF_LENGTH:
        goal_x = C.HALF_LENGTH if ball.position[0] > 0 else -C.HALF_LENGTH
        defending_team = team_defending_goal_line(goal_x)
        last_touch = ball.last_touch_team if ball.last_touch_team is not None else 1 - defending_team
        if last_touch == defending_team:
            # Defending team put it behind their own line -> corner to the attackers.
            return C.RESTART_CORNER, defending_team
        else:
            # Attacking team put it out over the defenders' line -> goal kick.
            return C.RESTART_GOAL_KICK, defending_team

    return None, None


def restart_ball_state(ball: Ball, restart_type: str, defending_team):
    """Returns (new_ball_position, restart_owning_team) for the given
    restart type, positioned the way the real restart would be taken
    from (touchline spot / corner arc / edge of the goal area).
    """
    if restart_type == C.RESTART_THROW_IN:
        y = C.HALF_WIDTH if ball.position[1] > 0 else -C.HALF_WIDTH
        x = float(np.clip(ball.position[0], -C.HALF_LENGTH, C.HALF_LENGTH))
        last_touch = ball.last_touch_team if ball.last_touch_team is not None else 0
        restart_team = 1 - last_touch
        return np.array([x, y], dtype=np.float32), restart_team

    goal_x = C.HALF_LENGTH if defending_team == 1 else -C.HALF_LENGTH
    into_field = team_attack_sign(defending_team)  # direction from their own goal towards the field

    if restart_type == C.RESTART_CORNER:
        y = C.HALF_WIDTH if ball.position[1] > 0 else -C.HALF_WIDTH
        return np.array([goal_x, y], dtype=np.float32), 1 - defending_team

    if restart_type == C.RESTART_GOAL_KICK:
        x = goal_x + into_field * (C.PENALTY_AREA_LENGTH * 0.4)
        return np.array([x, 0.0], dtype=np.float32), defending_team

    raise ValueError(f"unknown restart_type: {restart_type}")