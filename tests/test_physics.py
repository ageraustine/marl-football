"""Fast sanity checks for the physics module, runnable with just numpy:

    python -m tests.test_physics

These don't require pettingzoo/raylib/ray to be installed, so they're a
quick way to confirm the core simulation logic is sound before setting
up the full training stack.
"""

import numpy as np

from football_env import constants as C
from football_env import physics
from football_env.entities import Ball, Player


def make_player(x=0.0, y=0.0, team_id=0, role=C.FW, agent_id="p"):
    return Player(agent_id=agent_id, team_id=team_id, role=role,
                  position=np.array([x, y], dtype=np.float32))


def test_player_accelerates_towards_input():
    p = make_player()
    for _ in range(30):
        physics.step_player(p, np.array([1.0, 0.0], dtype=np.float32), C.DT)
    assert p.velocity[0] > 0, "player should have gained positive x velocity"
    assert p.velocity[0] <= p.max_speed + 1e-3, "should not exceed max speed"


def test_player_decelerates_with_no_input():
    p = make_player()
    p.velocity = np.array([5.0, 0.0], dtype=np.float32)
    for _ in range(50):
        physics.step_player(p, np.zeros(2, dtype=np.float32), C.DT)
    assert np.linalg.norm(p.velocity) < 0.5, "player should slow to a near-stop"


def test_possession_pickup_within_radius():
    ball = Ball(position=np.array([0.0, 0.0], dtype=np.float32))
    players = {"p1": make_player(x=0.2, y=0.0, agent_id="p1")}
    physics.resolve_possession(ball, players)
    assert ball.owner_id == "p1"


def test_possession_not_picked_up_out_of_radius():
    ball = Ball(position=np.array([0.0, 0.0], dtype=np.float32))
    players = {"p1": make_player(x=5.0, y=0.0, agent_id="p1")}
    physics.resolve_possession(ball, players)
    assert ball.owner_id is None


def test_kick_releases_and_moves_ball():
    ball = Ball(position=np.array([0.0, 0.0], dtype=np.float32), owner_id="p1")
    p = make_player(agent_id="p1")
    physics.apply_kick(ball, p, C.KICK_SHOOT, np.array([1.0, 0.0], dtype=np.float32))
    assert ball.owner_id is None
    assert ball.velocity[0] > 0


def test_goal_detection():
    ball = Ball(position=np.array([C.HALF_LENGTH + 0.1, 0.0], dtype=np.float32))
    assert physics.check_goal(ball) == 0
    ball.position = np.array([-C.HALF_LENGTH - 0.1, 0.0], dtype=np.float32)
    assert physics.check_goal(ball) == 1
    ball.position = np.array([C.HALF_LENGTH + 0.1, C.GOAL_HALF_WIDTH + 1.0], dtype=np.float32)
    assert physics.check_goal(ball) is None


def test_ball_out_of_bounds():
    ball = Ball(position=np.array([0.0, C.HALF_WIDTH + 1.0], dtype=np.float32))
    assert physics.check_out_of_bounds(ball) is True
    ball.position = np.array([0.0, 0.0], dtype=np.float32)
    assert physics.check_out_of_bounds(ball) is False


def run_all():
    tests = [obj for name, obj in globals().items() if name.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ok  {t.__name__}")
    print(f"\n{passed}/{len(tests)} physics tests passed")


if __name__ == "__main__":
    run_all()
