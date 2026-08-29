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


def test_stamina_drains_while_sprinting():
    p = make_player()
    p.velocity = np.array([p.base_max_speed, 0.0], dtype=np.float32)  # full sprint
    start = p.stamina
    for _ in range(50):  # 2.5s at 20Hz
        physics.update_stamina(p, C.DT)
    assert p.stamina < start, "stamina should drop while sprinting"


def test_stamina_recovers_while_resting():
    p = make_player()
    p.stamina = 40.0
    p.velocity = np.zeros(2, dtype=np.float32)  # standing still
    for _ in range(50):
        physics.update_stamina(p, C.DT)
    assert p.stamina > 40.0, "stamina should recover while not sprinting"


def test_fatigue_reduces_top_speed():
    p = make_player()
    p.stamina = C.STAMINA_MAX
    fresh_speed = p.max_speed
    p.stamina = 0.0
    tired_speed = p.max_speed
    assert tired_speed < fresh_speed
    assert tired_speed == pytest_approx(fresh_speed * C.STAMINA_MIN_SPEED_MULT)


def pytest_approx(x, tol=1e-4):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol
    return _Approx()


def test_offside_flagged_when_ahead_of_defenders_and_ball():
    passer = make_player(x=0.0, y=0.0, team_id=0, agent_id="passer")
    teammate_ahead = make_player(x=40.0, y=5.0, team_id=0, agent_id="mate_ahead")
    teammate_deep = make_player(x=-10.0, y=0.0, team_id=0, agent_id="mate_deep")
    defender_1 = make_player(x=30.0, y=0.0, team_id=1, role=C.DF, agent_id="def1")
    defender_2 = make_player(x=25.0, y=0.0, team_id=1, role=C.GK, agent_id="gk1")
    players = {p.agent_id: p for p in [passer, teammate_ahead, teammate_deep, defender_1, defender_2]}

    flagged = physics.offside_flagged_teammates(passer, players)
    assert "mate_ahead" in flagged
    assert "mate_deep" not in flagged


def test_no_offside_in_own_half():
    passer = make_player(x=-40.0, y=0.0, team_id=0, agent_id="passer")
    teammate = make_player(x=-30.0, y=0.0, team_id=0, agent_id="mate")  # ahead of passer, still own half
    defender = make_player(x=-35.0, y=0.0, team_id=1, agent_id="def1")
    players = {p.agent_id: p for p in [passer, teammate, defender]}

    flagged = physics.offside_flagged_teammates(passer, players)
    assert "mate" not in flagged, "own-half positions are never offside"


def test_classify_restart_throw_in():
    ball = Ball(position=np.array([10.0, C.HALF_WIDTH + 0.5], dtype=np.float32), last_touch_team=0)
    restart_type, defending_team = physics.classify_restart(ball)
    assert restart_type == C.RESTART_THROW_IN
    pos, restart_team = physics.restart_ball_state(ball, restart_type, defending_team)
    assert restart_team == 1  # team 0 touched it last, so team 1 restarts
    assert abs(pos[1] - C.HALF_WIDTH) < 1e-4


def test_classify_restart_corner_when_defender_touched_last():
    ball = Ball(position=np.array([C.HALF_LENGTH + 0.5, 2.0], dtype=np.float32), last_touch_team=1)
    restart_type, defending_team = physics.classify_restart(ball)
    assert restart_type == C.RESTART_CORNER
    assert defending_team == 1
    pos, restart_team = physics.restart_ball_state(ball, restart_type, defending_team)
    assert restart_team == 0  # attackers (team 0) get the corner
    assert abs(pos[0] - C.HALF_LENGTH) < 1e-4


def test_classify_restart_goal_kick_when_attacker_touched_last():
    ball = Ball(position=np.array([C.HALF_LENGTH + 0.5, 2.0], dtype=np.float32), last_touch_team=0)
    restart_type, defending_team = physics.classify_restart(ball)
    assert restart_type == C.RESTART_GOAL_KICK
    assert defending_team == 1
    pos, restart_team = physics.restart_ball_state(ball, restart_type, defending_team)
    assert restart_team == 1  # defending team (1) gets the goal kick
    assert pos[0] < C.HALF_LENGTH  # pulled back into the field, not on the line


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