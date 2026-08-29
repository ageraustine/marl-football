"""
Central place for every tunable number in the simulation.
Keeping these in one file means curriculum stages (2v2 -> 11v11) and
renderer/sim can agree on geometry without importing each other.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Pitch geometry (meters). Standard FIFA pitch: 105m x 68m.
# Origin (0, 0) is the center circle. +X = towards Team 1's goal (right).
# ---------------------------------------------------------------------------
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2
HALF_WIDTH = PITCH_WIDTH / 2

GOAL_WIDTH = 7.32
GOAL_HALF_WIDTH = GOAL_WIDTH / 2
GOAL_DEPTH = 2.0

PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32

# Team 0 defends the LEFT goal (-HALF_LENGTH), attacks the RIGHT (+HALF_LENGTH)
# Team 1 defends the RIGHT goal, attacks the LEFT. This is fixed for the
# whole episode (no half-time swap, kept simple on purpose).
TEAM_0_GOAL_X = -HALF_LENGTH
TEAM_1_GOAL_X = HALF_LENGTH

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
SIM_HZ = 20                # physics steps per simulated second
DT = 1.0 / SIM_HZ
EPISODE_SECONDS = 180      # 3 simulated minutes per episode by default
MAX_EPISODE_STEPS = EPISODE_SECONDS * SIM_HZ

# ---------------------------------------------------------------------------
# Player physics
# ---------------------------------------------------------------------------
PLAYER_RADIUS = 0.35
PLAYER_MAX_SPEED = 8.0          # m/s, roughly elite sprint speed
PLAYER_MAX_ACCEL = 6.0          # m/s^2
PLAYER_FRICTION = 4.5           # deceleration when no input, m/s^2
GK_MAX_SPEED_MULT = 0.9         # keepers slightly slower, more agile is future work

# ---------------------------------------------------------------------------
# Ball physics
# ---------------------------------------------------------------------------
BALL_RADIUS = 0.11
BALL_FRICTION = 1.8             # rolling deceleration, m/s^2
BALL_MAX_SPEED = 30.0
POSSESSION_RADIUS = 0.9         # distance at which a player can control the ball
KICK_POWER_PASS = 12.0
KICK_POWER_SHOT = 22.0
KICK_POWER_CLEAR = 18.0

# ---------------------------------------------------------------------------
# Action space
# kick_type: 0 = no kick (dribble/hold), 1 = pass, 2 = shoot, 3 = clear
# ---------------------------------------------------------------------------
KICK_NONE, KICK_PASS, KICK_SHOOT, KICK_CLEAR = 0, 1, 2, 3
KICK_POWERS = {
    KICK_PASS: KICK_POWER_PASS,
    KICK_SHOOT: KICK_POWER_SHOT,
    KICK_CLEAR: KICK_POWER_CLEAR,
}

# ---------------------------------------------------------------------------
# Restart types (out-of-bounds classification)
# ---------------------------------------------------------------------------
RESTART_THROW_IN = "throw_in"
RESTART_CORNER = "corner"
RESTART_GOAL_KICK = "goal_kick"

# ---------------------------------------------------------------------------
# Stamina
# ---------------------------------------------------------------------------
STAMINA_MAX = 100.0
STAMINA_DRAIN_RATE = 6.0        # points/sec, at full sprint
STAMINA_RECOVERY_RATE = 3.0     # points/sec, when jogging or standing
STAMINA_SPRINT_THRESHOLD = 0.6  # fraction of base max speed above which stamina drains
STAMINA_MIN_SPEED_MULT = 0.6    # top-speed multiplier at zero stamina

# ---------------------------------------------------------------------------
# Roles + 4-3-3 kickoff formation, expressed as fractions of half-pitch so the
# same formation function works for any squad size subset (curriculum).
# x_frac: 0 = own goal line, 1 = halfway line. Mirrored for the away team.
# y_frac: -1 = left touchline, 1 = right touchline.
# ---------------------------------------------------------------------------
GK, DF, MF, FW = "GK", "DF", "MF", "FW"

FORMATION_11 = [
    (GK, 0.03, 0.0),
    (DF, 0.20, -0.7), (DF, 0.20, -0.25), (DF, 0.20, 0.25), (DF, 0.20, 0.7),
    (MF, 0.55, -0.6), (MF, 0.55, 0.0), (MF, 0.55, 0.6),
    (FW, 0.85, -0.4), (FW, 0.85, 0.0), (FW, 0.85, 0.4),
]

# Smaller squads for curriculum training reuse the same shape, truncated
# and re-spread. See football_env.formations.build_formation().
FORMATION_5 = [
    (GK, 0.05, 0.0),
    (DF, 0.30, -0.4), (DF, 0.30, 0.4),
    (FW, 0.75, -0.3), (FW, 0.75, 0.3),
]

FORMATION_3 = [
    (GK, 0.05, 0.0),
    (DF, 0.35, 0.0),
    (FW, 0.80, 0.0),
]

FORMATION_2 = [
    (DF, 0.35, 0.0),
    (FW, 0.80, 0.0),
]

FORMATIONS_BY_SIZE = {
    2: FORMATION_2,
    3: FORMATION_3,
    5: FORMATION_5,
    11: FORMATION_11,
}

# ---------------------------------------------------------------------------
# Colors (RGBA tuples, used by the raylib renderer)
# ---------------------------------------------------------------------------
COLOR_PITCH_DARK = (56, 142, 60, 255)
COLOR_PITCH_LIGHT = (67, 160, 71, 255)
COLOR_LINES = (240, 240, 240, 255)
COLOR_TEAM_0 = (220, 50, 50, 255)
COLOR_TEAM_1 = (50, 90, 220, 255)
COLOR_BALL = (250, 250, 250, 255)
COLOR_BALL_SHADOW = (20, 20, 20, 90)
COLOR_POSSESSION_RING = (255, 215, 0, 255)
COLOR_UI_BG = (15, 20, 15, 220)
COLOR_UI_TEXT = (240, 240, 240, 255)

PITCH_STRIPE_COUNT = 12


def np_zeros2():
    return np.zeros(2, dtype=np.float32)